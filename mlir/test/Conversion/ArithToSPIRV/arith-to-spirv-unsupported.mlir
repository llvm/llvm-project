// RUN: mlir-opt -split-input-file -convert-arith-to-spirv -verify-diagnostics %s

///===----------------------------------------------------------------------===//
// Binary ops
//===----------------------------------------------------------------------===//

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64, Shader], []>, #spirv.resource_limits<>>
} {

func.func @unsupported_5elem_vector(%arg0: vector<5xi32>) {
  // expected-error@+1 {{failed to legalize operation 'arith.subi'}}
  %1 = arith.subi %arg0, %arg0: vector<5xi32>
  return
}

} // end module

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64, Shader], []>, #spirv.resource_limits<>>
} {

func.func @unsupported_2x2elem_vector(%arg0: vector<2x2xi32>) {
  // expected-error@+1 {{failed to legalize operation 'arith.muli'}}
  %2 = arith.muli %arg0, %arg0: vector<2x2xi32>
  return
}

} // end module

// -----

func.func @int_vector4_invalid(%arg0: vector<2xi16>) {
  // expected-error @+2 {{failed to legalize operation 'arith.divui'}}
  // expected-error @+1 {{bitwidth emulation is not implemented yet on unsigned op}}
  %0 = arith.divui %arg0, %arg0: vector<2xi16>
  return
}

///===----------------------------------------------------------------------===//
// Constant ops
//===----------------------------------------------------------------------===//

// -----

func.func @unsupported_constant_i64_0() {
  // expected-error @+1 {{failed to legalize operation 'arith.constant'}}
  %0 = arith.constant 0 : i64
  return
}

// -----

func.func @unsupported_constant_i64_1() {
  // expected-error @+1 {{failed to legalize operation 'arith.constant'}}
  %0 = arith.constant 4294967296 : i64 // 2^32
  return
}

// -----

func.func @unsupported_constant_vector_2xi64_0() {
  // expected-error @+1 {{failed to legalize operation 'arith.constant'}}
  %1 = arith.constant dense<0> : vector<2xi64>
  return
}

// -----

func.func @unsupported_constant_f64_0() {
  // expected-error @+1 {{failed to legalize operation 'arith.constant'}}
  %1 = arith.constant 0.0 : f64
  return
}

// -----

func.func @unsupported_constant_vector_2xf64_0() {
  // expected-error @+1 {{failed to legalize operation 'arith.constant'}}
  %1 = arith.constant dense<0.0> : vector<2xf64>
  return
}

// -----

func.func @unsupported_constant_tensor_2xf64_0() {
  // expected-error @+1 {{failed to legalize operation 'arith.constant'}}
  %1 = arith.constant dense<0.0> : tensor<2xf64>
  return
}

///===----------------------------------------------------------------------===//
// Type emulation
//===----------------------------------------------------------------------===//

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// Check that we do not emualte i64 by truncating to i32.
func.func @unsupported_i64(%arg0: i64) {
  // expected-error@+1 {{failed to legalize operation 'arith.addi'}}
  %2 = arith.addi %arg0, %arg0: i64
  return
}

} // end module

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// Check that we do not emualte f64 by truncating to i32.
func.func @unsupported_f64(%arg0: f64) {
  // expected-error@+1 {{failed to legalize operation 'arith.addf'}}
  %2 = arith.addf %arg0, %arg0: f64
  return
}

} // end module
