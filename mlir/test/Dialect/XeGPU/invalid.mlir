// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
func.func @test_create_nd_tdesc_vc_1(%src: memref<24xf32>) {
  // expected-error@+1 {{Expecting the rank of shape, strides, offsets, source memref type (if source is a memref) and TensorDesc should match with each other. They currenlty are 2D.}}
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
func.func @test_dpas_vc_1(%a : vector<8x4x2xf16>, %b: vector<8x16x2xf16>) {
  // expected-error@+1 {{K-dimension or vnni-factor mismatch}}
  %1 = xegpu.dpas %a, %b : vector<8x4x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  return
}

// -----
func.func @test_dpas_vc_2(%a : vector<8x16xf16>, %b: vector<8x16x2xf16>) {
  // expected-error@+1 {{lhs and rhs rank does not match for dpas op, or their rank is not 3}}
  %1 = xegpu.dpas %a, %b : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  return
}

// -----
func.func @test_dpas_vc_3(%a : vector<8x16xf16>, %b: vector<16x16xf16>) {
  // expected-error@+1 {{lhs and rhs rank does not match for dpas op, or their rank is not 3}}
  %1 = xegpu.dpas %a, %b : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return
}

// -----
func.func @test_dpas_vc_4(%a : vector<8x8x2xf16>, %b: vector<8x16x2xf16>, %c : vector<8x16xf16>) {
  // expected-error@+1 {{Accumulator and Result for dpas op should have the same type}}
  %1 = xegpu.dpas %a, %b, %c : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf16> -> vector<8x16xf32>
  return
}