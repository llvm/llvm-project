// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
func.func @test_create_nd_tdesc_vc_1(%src: memref<24xf32>) {
  // expected-error@+1 {{Expecting the TensorDesc rank is up to 2 and not greater than the ranks of shape, strides, offsets or the memref source}}
  %1 = xegpu.create_nd_tdesc %src[0] : memref<24xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// -----

func.func @test_create_nd_tdesc_vc_2(%src: memref<24x32xf32>) {
  // expected-error@+1 {{TensorDesc should have the same element type with the source if it is a memref}}
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf16>
  return
}

// -----
func.func @test_prefetch_nd_vc_1(%src: memref<24x32xf16>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  // expected-error@+1 {{invlid l1_hint: #xegpu.cache_hint<write_back>}}
  xegpu.prefetch_nd %1 <{l1_hint = #xegpu.cache_hint<write_back>}>: !xegpu.tensor_desc<8x16xf16>
  return
}

// -----
func.func @test_prefetch_nd_vc_2(%src: memref<24xf16>) {
  %1 = xegpu.create_tdesc %src[0, 1, 2, 3, 4, 5, 6, 7]
        : memref<24xf16> -> !xegpu.tensor_desc<8xf16, #xegpu.tdesc_attr<scattered=true>>
  // expected-error@+1 {{Expects a non-scattered TensorDesc}}
  xegpu.prefetch_nd %1 <{l1_hint = #xegpu.cache_hint<cached>}>
        : !xegpu.tensor_desc<8xf16, #xegpu.tdesc_attr<scattered=true>>
  return
}

// -----
func.func @test_load_nd_vc_1(%src: memref<8x16xf16>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  // expected-error@+1 {{invlid l1_hint: #xegpu.cache_hint<write_back>}}
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<write_back>}>
      : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  return
}

// -----
func.func @test_load_nd_vc_2(%src: memref<16xf16>) {
  %1 = xegpu.create_tdesc %src[0, 2, 4, 6, 8, 10, 12, 14] {chunk_size = 2}
        : memref<16xf16> -> !xegpu.tensor_desc<8x2xf16, #xegpu.tdesc_attr<scattered=true>>
  // expected-error@+1 {{Expects a non-scattered TensorDesc.}}
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>}>
      : !xegpu.tensor_desc<8x2xf16, #xegpu.tdesc_attr<scattered=true>> -> vector<8x2xf16>
  return
}

// -----
func.func @test_load_nd_vc_3(%src: memref<8x16xf16>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<16xf16>
  // expected-warning@+1 {{Invalid Packed Attr.}}
  %2 = xegpu.load_nd %1 <{packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
        : !xegpu.tensor_desc<16xf16> -> vector<16xf16>
  return
}

// -----
func.func @test_store_nd_vc_1(%dst: memref<24x32xf16>) {
  %1 = arith.constant dense<1.0>: vector<24x32xf16>
  %2 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16>
  // expected-error@+1 {{invlid l1_hint: #xegpu.cache_hint<streaming>}}
  xegpu.store_nd %1, %2 <{l1_hint = #xegpu.cache_hint<streaming>}>: vector<24x32xf16>, !xegpu.tensor_desc<24x32xf16>
  return
}

// -----
func.func @test_store_nd_vc_2(%dst: memref<16xf16>) {
  %1 = arith.constant dense<1.0>: vector<8x2xf16>
  %2 = xegpu.create_tdesc %dst[0, 2, 4, 6, 8, 10, 12, 14] {chunk_size = 2}
        : memref<16xf16> -> !xegpu.tensor_desc<8x2xf16, #xegpu.tdesc_attr<scattered=true>>
  // expected-error@+1 {{Expects a non-scattered TensorDesc}}
  xegpu.store_nd %1, %2 <{l1_hint = #xegpu.cache_hint<streaming>}>
        : vector<8x2xf16>, !xegpu.tensor_desc<8x2xf16, #xegpu.tdesc_attr<scattered=true>>
  return
}

// -----
func.func @test_update_nd_offset_1(%dst: memref<16xf16>) {
  %1 = xegpu.create_tdesc %dst[0, 2, 4, 6, 8, 10, 12, 14] {chunk_size = 2}
        : memref<16xf16> -> !xegpu.tensor_desc<8x2xf16, #xegpu.tdesc_attr<scattered=true>>
  // expected-error@+1 {{Expects a non-scattered TensorDesc}}
  xegpu.update_nd_offset %1, [0, 2] : !xegpu.tensor_desc<8x2xf16, #xegpu.tdesc_attr<scattered=true>>
  return
}

// -----
func.func @test_create_tdesc_vc_1(%src: ui64) {
  // expected-error@+1 {{Expects a scattered TensorDesc}}
  %1 = xegpu.create_tdesc %src[0, 2, 4, 6, 8, 10, 12, 14] {chunk_size = 2}
        : ui64 -> !xegpu.tensor_desc<8x2xf16>
  return
}

// -----
func.func @test_create_tdesc_vc_2(%src: ui64) {
  // expected-error@+1 {{Incorrect TensorDesc shape}}
  %1 = xegpu.create_tdesc %src[0, 2, 4, 6, 8, 10, 12, 14] {chunk_size = 2}
        : ui64 -> !xegpu.tensor_desc<8x4xf16, #xegpu.tdesc_attr<scattered = true>>
  return
}

// -----
func.func @test_prefetch_vc_1(%src: memref<24x32xf16>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16>
  // expected-error@+1 {{Expects a scattered TensorDesc}}
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<write_back>}>: !xegpu.tensor_desc<24x32xf16>
  return
}

// -----
func.func @test_prefetch_vc_2(%src: ui64) {
  %1 = xegpu.create_tdesc %src[0, 8, 16, 24] {chunk_size = 2} : ui64  -> !xegpu.tensor_desc<4x2xf32, #xegpu.tdesc_attr<scattered = true>>
  // expected-error@+1 {{invlid l1_hint: #xegpu.cache_hint<write_back>}}
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<write_back>}>: !xegpu.tensor_desc<4x2xf32, #xegpu.tdesc_attr<scattered = true>>
  return
}

// -----
func.func @test_load_gather_vc_1(%src: memref<24x32xf16>) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<4x2xf16>
  // expected-error@+1 {{Expects a scattered TensorDesc}}
  %2 = xegpu.load %1, %0 <{l1_hint = #xegpu.cache_hint<cached>}>
      : !xegpu.tensor_desc<4x2xf16>, vector<4xi1> -> vector<4x2xf16>
  return
}

// -----
func.func @test_load_gather_vc_2(%src: ui64) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %1 = xegpu.create_tdesc %src[0, 8, 16, 24] {chunk_size = 2} : ui64
        -> !xegpu.tensor_desc<4x2xf32, #xegpu.tdesc_attr<scattered = true>>
  // expected-error@+1 {{invlid l1_hint: #xegpu.cache_hint<write_back>}}
  %2 = xegpu.load %1, %0 <{l1_hint = #xegpu.cache_hint<write_back>}>
        : !xegpu.tensor_desc<4x2xf32, #xegpu.tdesc_attr<scattered = true>>, vector<4xi1>
          -> vector<4x2xf32>
  return
}

// -----
func.func @test_store_scatter_vc_1(%src: memref<24x32xf32>) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %1 = arith.constant dense<2.9>: vector<4x2xf32>
  %2 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<4x2xf32>
  // expected-error@+1 {{Expects a scattered TensorDesc}}
  xegpu.store %1, %2, %0 <{l1_hint = #xegpu.cache_hint<cached>}>
        : vector<4x2xf32>, !xegpu.tensor_desc<4x2xf32>, vector<4xi1>
  return
}

// -----
func.func @test_store_scatter_vc_2(%src: ui64) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %1 = arith.constant dense<2.9>: vector<4x2xf32>
  %2 = xegpu.create_tdesc %src[0, 8, 16, 24] {chunk_size = 2}
          : ui64 -> !xegpu.tensor_desc<4x2xf32, #xegpu.tdesc_attr<scattered = true>>
  // expected-error@+1 {{invlid l1_hint: #xegpu.cache_hint<streaming>}}
  xegpu.store %1, %2, %0 <{l1_hint = #xegpu.cache_hint<streaming>}> : vector<4x2xf32>,
          !xegpu.tensor_desc<4x2xf32, #xegpu.tdesc_attr<scattered = true>>, vector<4xi1>
  return
}

// -----
func.func @test_dpas_vc_1(%a : vector<8x8xf16>, %b: vector<8x16x2xf16>) {
  // expected-error@+1 {{K-dimension mismatch}}
  %1 = xegpu.dpas %a, %b : vector<8x8xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  return
}

// -----
func.func @test_dpas_vc_2(%a : vector<8x8x2xf16>, %b: vector<8x16x2xf16>) {
  // expected-error@+1 {{expecting lhs to be a 2D vector, and rhs to be either 2D or 3D (packed) vector}}
  %1 = xegpu.dpas %a, %b : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  return
}

// -----
func.func @test_atomic_rmw(%src: ui64, %value : vector<16x8xf32>, %mask : vector<16xi1>) {
  %1 = xegpu.create_tdesc %src[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {chunk_size = 8}: ui64 -> !xegpu.tensor_desc<16x8xf32, #xegpu.tdesc_attr<scattered = true>>
  // expected-error@+1 {{failed to verify that all of {tensorDesc, mask, value, result} have same shape}}
  xegpu.atomic_rmw addf %1, %mask, %value: !xegpu.tensor_desc<16x8xf32, #xegpu.tdesc_attr<scattered = true>>, vector<16xi1>, vector<16x8xf32> -> vector<16x8xf32>
  gpu.return
}