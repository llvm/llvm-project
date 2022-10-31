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

func.func @unsupported_constant_0() {
  // expected-error @+1 {{failed to legalize operation 'arith.constant'}}
  %0 = arith.constant 4294967296 : i64 // 2^32
  return
}

// -----

func.func @unsupported_constant_1() {
  // expected-error @+1 {{failed to legalize operation 'arith.constant'}}
  %1 = arith.constant -2147483649 : i64 // -2^31 - 1
  return
}

// -----

func.func @unsupported_constant_2() {
  // expected-error @+1 {{failed to legalize operation 'arith.constant'}}
  %2 = arith.constant -2147483649 : i64 // -2^31 - 1
  return
}
