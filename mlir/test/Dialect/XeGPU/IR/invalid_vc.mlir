// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// -----
func.func @test_create_nd_tdesc_vc_1(%src: memref<24xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // expected-error@+1 {{Expecting the rank of shape, strides, offsets and memref type should match with each other}}
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] {mode = vc} : memref<24xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_create_nd_tdesc_vc_3(%input: memref<?xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // expected-error@+1 {{Expecting the rank of shape, strides, offsets and memref type should match with each other}}
  %1 = xegpu.create_nd_tdesc %input[%c0, %c1], [%c8, %c16], [%c16, %c1] {mode = vc} : memref<?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}


// -----
func.func @test_create_nd_tdesc_vc_4(%input: memref<?x?xf32>) {
  %c1 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  // expected-error@+1 {{Expecting the rank of shape, strides, offsets and memref type should match with each other}}
  %1 = xegpu.create_nd_tdesc %input[%c1], [%c8], [%c1] {mode = vc}
                              : memref<?x?xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_create_nd_tdesc_vc_5(%input: memref<24x32x64xf32>) {
  %c1 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  // expected-error@+1 {{operand #0 must be 1D/2D memref}}
  %1 = xegpu.create_nd_tdesc %input[%c1, %c1, %c8] {mode = vc}
                              : memref<24x32x64xf32> -> !xegpu.tensor_desc<8x16x8xf32>
  return
}

// -----
func.func @test_create_tdesc(%src: ui64, %offsets : vector<16x8xindex>) {
  // expected-error@+1 {{operand #1 must be vector of index values of ranks 1}}
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc}
                              : ui64, vector<16x8xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  return
}

// -----
func.func @test_load_gather(%src: ui64, %offsets : vector<16xindex>) {
  %0 = arith.constant dense<1>: vector<16x8xi1>
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {mode = vc, chunk_size_per_lane = 8}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 8}
                              : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf16, #xegpu.scattered>

  // expected-error@+1 {{Result shape doesn't match TensorDesc shape.}}
  %2 = xegpu.load %1, %0 {mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = uncached}
                          : !xegpu.tensor_desc<16x8xf16, #xegpu.scattered>, vector<16x8xi1> -> vector<8x8x4xf16>
  return
}
