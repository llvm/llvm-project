// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @shuffle_i8(%v: i8, %i: i32) -> i8 {
  // expected-error @below {{requires a 16-bit value or a multiple of 32 bits}}
  %0 = agx.shuffle xor %v, %i : i8
  return %0 : i8
}

// -----

func.func @reduce_int_mul(%v: i32) -> i32 {
  // expected-error @below {{there is no integer multiplication reduction}}
  %0 = agx.reduce mul %v : i32
  return %0 : i32
}

// -----

func.func @reduce_float_and(%v: f32) -> f32 {
  // expected-error @below {{reduction 'and' does not apply to floating point values}}
  %0 = agx.reduce and %v : f32
  return %0 : f32
}

// -----

func.func @reduce_int_minnumf(%v: i32) -> i32 {
  // expected-error @below {{reduction 'minnumf' requires floating point values}}
  %0 = agx.reduce minnumf %v : i32
  return %0 : i32
}
