// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
func.func @unary_e8mf2(%rvl: ui32) {
  %const = arith.constant 0 : i32
  // expected-error@+1 {{must use 2-bit opcode}}
  vcix.unary.ro e8mf2 %const, %rvl { opcode = 1 : i1, rs2 = 30 : i5, rd = 31 : i5 } : (i32, ui32)
  return
}
// -----
func.func @binary_fv(%op1: f32, %op2 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> {
  // expected-error@+1 {{with a floating point scalar can only use 1-bit opcode}}
  %0 = vcix.binary %op1, %op2, %rvl { opcode = 2 : i2 } : (f32, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// -----
func.func @ternary_fvv(%op1: f32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> {
  // expected-error@+1 {{with a floating point scalar can only use 1-bit opcode}}
  %0 = vcix.ternary %op1, %op2, %op3, %rvl { opcode = 2 : i2 } : (f32, vector<[4] x f32>, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// -----
func.func @wide_ternary_fvw(%op1: f32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) -> vector<[4] x f64> {
  // expected-error@+1 {{with a floating point scalar can only use 1-bit opcode}}
  %0 = vcix.wide.ternary %op1, %op2, %op3, %rvl { opcode = 2 : i2 } : (f32, vector<[4] x f32>, vector<[4] x f64>, ui32) -> vector<[4] x f64>
  return %0 : vector<[4] x f64>
}

// -----
func.func @wide_ternary_vvw(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> {
  // expected-error@+1 {{result type is not widened type of op2}}
  %0 = vcix.wide.ternary %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (vector<[4] x f32>, vector<[4] x f32>, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// -----
func.func @binary_fv_wrong_vtype(%op1: f32, %op2 : vector<[32] x f32>, %rvl : ui32) -> vector<[32] x f32> {
  // expected-error@+1 {{used type does not represent RVV-compatible scalable vector type}}
  %0 = vcix.binary %op1, %op2, %rvl { opcode = 1 : i1 } : (f32, vector<[32] x f32>, ui32) -> vector<[32] x f32>
  return %0 : vector<[32] x f32>
}

// -----
func.func @binary_fv_vls_rvl(%op1: f32, %op2 : vector<4 x f32>, %rvl : ui32) -> vector<4 x f32> {
  // expected-error@+1 {{'rvl' must not be specified if operation is done on a fixed vector type}}
  %0 = vcix.binary %op1, %op2, %rvl { opcode = 1 : i1 } : (f32, vector<4 x f32>, ui32) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

// -----
func.func @binary_nonconst(%val: i5, %op: vector<[4] x f32>, %rvl: ui32) {
  // expected-error@+1 {{immediate operand must be a constant}}
  vcix.binary.ro %val, %op, %rvl { opcode = 1 : i2, rd = 30 : i5 } : (i5, vector<[4] x f32>, ui32)
  return
}
