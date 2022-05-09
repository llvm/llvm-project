// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @a_is_2d(%a : vector<2x2xi32>, %b : vector<4x4xi8>) -> vector<2x2xi32> {
    // expected-error@+1 {{operand `a` should be 1-dimensional}}
    %0 = arm_neon.2d.sdot %a, %b, %b : vector<4x4xi8>, vector<4x4xi8> to vector<2x2xi32>
    return %0 : vector<2x2xi32>
}

// -----

func.func @b_is_3d(%a : vector<4xi32>, %b : vector<1x4x4xi8>) -> vector<4xi32> {
    // expected-error@+1 {{operand `b` should be 2-dimensional}}
    %0 = arm_neon.2d.sdot %a, %b, %b : vector<1x4x4xi8>, vector<1x4x4xi8> to vector<4xi32>
    return %0 : vector<4xi32>
}

// -----

func.func @b_has_2_columns(%a : vector<4xi32>, %b : vector<4x2xi8>) -> vector<4xi32> {
    // expected-error@+1 {{operand `b` should have 4 columns}}
    %0 = arm_neon.2d.sdot %a, %b, %b : vector<4x2xi8>, vector<4x2xi8> to vector<4xi32>
    return %0 : vector<4xi32>
}

// -----

func.func @b_has_2_rows_but_a_has_length_4(%a : vector<4xi32>, %b : vector<2x4xi8>) -> vector<4xi32> {
    // expected-error@+1 {{operand `b` should have as many rows as the size of operand `a`}}
    %0 = arm_neon.2d.sdot %a, %b, %b : vector<2x4xi8>, vector<2x4xi8> to vector<4xi32>
    return %0 : vector<4xi32>
}
