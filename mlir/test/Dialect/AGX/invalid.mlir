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

// -----

func.func @mma_shape(%a: vector<4xf32>) -> vector<4xf32> {
  // expected-error @+1 {{operand #0 must be vector of 16-bit float or bfloat16 type or 32-bit float values of length 2}}
  %0 = agx.mma %a * %a + %a : vector<4xf32>, vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

func.func @mma_accumulator(%a: vector<2xf32>, %c: vector<2xf16>) -> vector<2xf16> {
  // expected-error @+1 {{accumulator 'f16' cannot accumulate 'f32' products}}
  %0 = agx.mma %a * %a + %c : vector<2xf32>, vector<2xf16>
  return %0 : vector<2xf16>
}

// -----

func.func @tile_size(%A: memref<?x?xf16>, %acc: vector<32xf32>, %i: index) -> vector<32xf32> {
  // expected-error @+1 {{requires m, n and k to be multiples of 8}}
  %0 = agx.mma_tile %A[%i, %i] * %A[%i, %i] + %acc tile(32, 32, 4)
      : memref<?x?xf16>, memref<?x?xf16>, vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @tile_acc(%A: memref<?x?xf16>, %acc: vector<16xf32>, %i: index) -> vector<16xf32> {
  // expected-error @+1 {{requires an accumulator of m * n / 32 = 32 elements per thread, got 16}}
  %0 = agx.mma_tile %A[%i, %i] * %A[%i, %i] + %acc tile(32, 32, 8)
      : memref<?x?xf16>, memref<?x?xf16>, vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

func.func @fragment_type(%A: memref<?x?xf16>, %i: index) -> vector<2xf32> {
  // expected-error @+1 {{fragment 'vector<2xf32>' does not match the elements of 'memref<?x?xf16>'}}
  %0 = agx.matrix_load %A[%i, %i] : memref<?x?xf16> -> vector<2xf32>
  return %0 : vector<2xf32>
}

// -----

func.func @lane_shift(%x: f32) -> f32 {
  // expected-error @+1 {{shift 4 does not stay inside a segment of 4 lanes}}
  %0 = agx.fma_lane_shift %x, %x, %x shift 4 width 4 : f32
  return %0 : f32
}

// -----

func.func @uniform_thread_dependent() -> index {
  %0 = agx.uniform_region -> (index) {
    // expected-error @+1 {{is thread-dependent and cannot be in an agx.uniform_region}}
    %t = gpu.thread_id x
    agx.yield %t : index
  }
  return %0 : index
}
