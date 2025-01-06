// RUN: mlir-opt -split-input-file -convert-arith-to-spirv -verify-diagnostics %s

///===----------------------------------------------------------------------===//
// Cast ops
//===----------------------------------------------------------------------===//

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Float16, Kernel], []>, #spirv.resource_limits<>>
} {

func.func @experimental_constrained_fptrunc(%arg0 : f32) {
  // expected-error@+1 {{failed to legalize operation 'arith.truncf'}}
  %3 = arith.truncf %arg0 to_nearest_away : f32 to f16
  return
}

} // end module

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

// -----

func.func @int_vector_invalid_bitwidth(%arg0: vector<2xi12>) {
  // expected-error @+1 {{failed to legalize operation 'arith.addi'}}
  %0 = arith.addi %arg0, %arg0: vector<2xi12>
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

// -----

func.func @constant_dense_resource_non_existant() {
  // expected-error @+2 {{failed to legalize operation 'arith.constant'}}
  // expected-error @+1 {{could not find resource blob}}
  %0 = arith.constant dense_resource<non_existant> : tensor<5xf32>  
  return
}

// -----

module {
func.func @constant_dense_resource_invalid_buffer() {
  // expected-error @+2 {{failed to legalize operation 'arith.constant'}}
  // expected-error @+1 {{resource is not a valid buffer}}
  %0 = arith.constant dense_resource<dense_resource_test_2xi32> : vector<2xi32>  
  return
  }
}
// This is a buffer of wrong type and shape
{-#
  dialect_resources: {
    builtin: {
      dense_resource_test_2xi32: "0x0800000054A3B53ED6C0B33E55D1A2BDE5D2BB3E"
    }
  }
#-}

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

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// i64 is not a valid result type in this target env.
func.func @type_conversion_failure(%arg0: i32) {
  // expected-error@+1 {{failed to legalize operation 'arith.extsi'}}
  %2 = arith.extsi %arg0 : i32 to i64
  return
}

} // end module
