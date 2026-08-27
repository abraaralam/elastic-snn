`timescale 1ns / 1ps

module adder_tree_16 (
    input  logic signed [15:0] weights [15:0], // 16 input weights
    output logic signed [15:0] total_sum
);
    // Wires separating each level of tree
    logic signed [15:0] level1 [7:0];
    logic signed [15:0] level2 [3:0];
    logic signed [15:0] level3 [1:0];

    always_comb begin
        // Level 1 - 16 to 8
        for (int i = 0; i < 8; i++) begin
            level1[i] = weights[2*i] + weights[2*i+1];
        end
        
        // Level 2 - 8 to 4
        for (int i = 0; i < 4; i++) begin
            level2[i] = level1[2*i] + level1[2*i+1];
        end
        
        // Level 3 - 4 to 2
        for (int i = 0; i < 2; i++) begin
            level3[i] = level2[2*i] + level2[2*i+1];
        end
        
        // Level 4 - 2 to 1
        total_sum = level3[0] + level3[1];
    end
endmodule