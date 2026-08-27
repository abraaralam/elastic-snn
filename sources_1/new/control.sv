`timescale 1ns / 1ps

module control (
    input  logic        clk,
    input  logic        rst_n,

    // Router Interface (Incoming)
    input  logic        aer_valid_in,
    input  logic [24:0] aer_data_in,
    output logic        aer_ready_out,

    // Router Interface (Outgoing)
    output logic        aer_valid_out,
    output logic [24:0] aer_data_out,

    // Memory (RAM) Interface
    output logic [11:0] ram_addr,
    output logic        ram_rd_en,
    output logic        ram_wr_en,

    // Datapath Interface
    output logic        enable_execute,
    output logic        spike_sign,
    input  logic        spike_fired_in
);

    // FSM States
    typedef enum logic [1:0] {
        IDLE_FETCH   = 2'b00,
        MEMORY_WAIT  = 2'b01,
        EXECUTE_FIRE = 2'b10,
        WRITEBACK    = 2'b11
    } state_t;

    state_t current_state, next_state;

    // Internal Register for AER packet retention
    logic [24:0] latched_aer;

    // 1. State Register & Data Latch (Sequential)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE_FETCH;
            latched_aer   <= 25'd0;
        end else begin
            current_state <= next_state;
            // Save the packet so we don't lose the address if the router drops the bus
            if (current_state == IDLE_FETCH && aer_valid_in) begin
                latched_aer <= aer_data_in; 
            end
        end
    end

    // 2. Next State Logic (Combinational)
    always_comb begin
        next_state = current_state; 
        case (current_state)
            IDLE_FETCH:   if (aer_valid_in) next_state = MEMORY_WAIT;
            MEMORY_WAIT:  next_state = EXECUTE_FIRE;
            EXECUTE_FIRE: next_state = WRITEBACK;
            WRITEBACK:    next_state = IDLE_FETCH;
        endcase
    end

    // 3. Output Logic (Combinational)
    always_comb begin
        // Default assignments to prevent inferred latches
        aer_ready_out  = 1'b0;
        aer_valid_out  = 1'b0;
        aer_data_out   = 25'd0;
        ram_addr       = latched_aer[12:1]; // 12-bit Position field
        ram_rd_en      = 1'b0;
        ram_wr_en      = 1'b0;
        enable_execute = 1'b0;
        spike_sign     = latched_aer[0];    // 1-bit Sign field

        case (current_state)
            IDLE_FETCH: begin
                aer_ready_out = 1'b1; // Tell router we can accept a spike
                
                // Drive RAM address immediately from the live bus to save a cycle
                ram_addr = aer_data_in[12:1]; 
                if (aer_valid_in) ram_rd_en = 1'b1;
            end
            
            MEMORY_WAIT: begin
                // Just wait for the synchronous RAM to output data
                ram_rd_en = 1'b1; 
            end
            
            EXECUTE_FIRE: begin
                enable_execute = 1'b1; // Tell datapath to do the math
                
                if (spike_fired_in) begin
                    aer_valid_out = 1'b1;
                    // Build outgoing AER: Keep ID, output new position, set sign
                    aer_data_out  = {latched_aer[24:13], latched_aer[12:1], 1'b0}; 
                end
            end
            
            WRITEBACK: begin
                ram_wr_en = 1'b1; // Save new membrane/tracer values
            end
        endcase
    end
endmodule