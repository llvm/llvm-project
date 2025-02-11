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
func.func @test_create_nd_tdesc_vc_3(%src: memref<2x24x32xf32, 3>) {
  // expected-error@+1 {{SLM is not supported for 2D block tensor}}
  %1 = xegpu.create_nd_tdesc %src[0, 0, 0] : memref<2x24x32xf32, 3> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = slm>>
  return
}

// -----
func.func @test_create_nd_tdesc_vc_4(%src: memref<2x24x32xf32, 3>) {
  // expected-error@+1 {{Memory space mismatch}}
  %1 = xegpu.create_nd_tdesc %src[0, 0, 0] : memref<2x24x32xf32, 3> -> !xegpu.tensor_desc<16xf32>
  return
}

// -----
func.func @test_prefetch_nd_vc_1(%src: memref<24x32xf16>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<write_back>}}
  xegpu.prefetch_nd %1 <{l1_hint = #xegpu.cache_hint<write_back>}>: !xegpu.tensor_desc<8x16xf16>
  return
}

// -----
func.func @test_prefetch_nd_vc_2(%src: memref<24xf16>) {
  %0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
  %1 = xegpu.create_tdesc %src, %0 : memref<24xf16>, vector<8xindex>
                -> !xegpu.tensor_desc<8xf16, #xegpu.scatter_tdesc_attr<>>
  // expected-error@+1 {{Expects a non-scattered TensorDesc}}
  xegpu.prefetch_nd %1 <{l1_hint = #xegpu.cache_hint<cached>}>
        : !xegpu.tensor_desc<8xf16, #xegpu.scatter_tdesc_attr<>>
  return
}

// -----
func.func @test_load_nd_vc_1(%src: memref<8x16xf16>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<write_back>}}
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<write_back>}>
      : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  return
}

// -----
func.func @test_load_nd_vc_2(%src: memref<16xf16>) {
  %0 = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14]> : vector<8xindex>
  %1 = xegpu.create_tdesc %src, %0 : memref<16xf16>, vector<8xindex>
          -> !xegpu.tensor_desc<8x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{Expects a non-scattered TensorDesc.}}
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>}>
      : !xegpu.tensor_desc<8x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>> -> vector<8x2xf16>
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
func.func @test_load_nd_vc_4(%src: memref<24x32xf32>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
    !xegpu.tensor_desc<8x16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  // expected-error@+1 {{Result shape doesn't match TensorDesc shape.}}
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>,
      l2_hint = #xegpu.cache_hint<uncached>}>
    : !xegpu.tensor_desc<8x16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
    -> vector<8x2xf32>
  return
}

// -----
func.func @test_load_nd_vc_5(%src: memref<24x32xf32>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
    !xegpu.tensor_desc<16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  // expected-error@+1 {{Result shape doesn't match TensorDesc shape.}}
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>,
      l2_hint = #xegpu.cache_hint<uncached>}>
    : !xegpu.tensor_desc<16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
    -> vector<8xf32>
  return
}

// -----
func.func @test_load_nd_vc_6(%src: memref<24x32xf32>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
    !xegpu.tensor_desc<8x16xf32>
  // expected-error@+1 {{Result shape doesn't match TensorDesc shape.}}
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>,
      l2_hint = #xegpu.cache_hint<uncached>}>
    : !xegpu.tensor_desc<8x16xf32> -> vector<8x1xf32>
  return
}

// -----
func.func @test_store_nd_vc_1(%dst: memref<24x32xf16>) {
  %1 = arith.constant dense<1.0>: vector<24x32xf16>
  %2 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<streaming>}}
  xegpu.store_nd %1, %2 <{l1_hint = #xegpu.cache_hint<streaming>}>: vector<24x32xf16>, !xegpu.tensor_desc<24x32xf16>
  return
}

// -----
func.func @test_store_nd_vc_2(%dst: memref<16xf16>) {
  %0 = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14]> : vector<8xindex>
  %1 = arith.constant dense<1.0>: vector<8x2xf16>
  %2 = xegpu.create_tdesc %dst, %0 : memref<16xf16>, vector<8xindex>
            -> !xegpu.tensor_desc<8x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{Expects a non-scattered TensorDesc}}
  xegpu.store_nd %1, %2 <{l1_hint = #xegpu.cache_hint<streaming>}>
        : vector<8x2xf16>, !xegpu.tensor_desc<8x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  return
}

// -----
func.func @test_store_nd_vc_3(%dst: memref<24x32xf32>, %data: vector<8x2xf32>) {
  %1 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf32> ->
    !xegpu.tensor_desc<8x16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  // expected-error@+1 {{Result shape doesn't match TensorDesc shape.}}
  xegpu.store_nd %data, %1
    : vector<8x2xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  return
}

// -----
func.func @test_store_nd_vc_4(%dst: memref<24x32xf32>, %data: vector<2xf32>) {
  %1 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf32> ->
    !xegpu.tensor_desc<16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  // expected-error@+1 {{Result shape doesn't match TensorDesc shape.}}
  xegpu.store_nd %data, %1
    : vector<2xf32>, !xegpu.tensor_desc<16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  return
}

// -----
func.func @test_store_nd_vc_5(%dst: memref<24x32xf32>, %data: vector<8x1xf32>) {
  %1 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf32> ->
    !xegpu.tensor_desc<8x16xf32>
  // expected-error@+1 {{Result shape doesn't match TensorDesc shape.}}
  xegpu.store_nd %data, %1 : vector<8x1xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @test_update_nd_offset_1(%dst: memref<16xf16>) {
  %0 = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14]> : vector<8xindex>
  %1 = xegpu.create_tdesc %dst, %0 : memref<16xf16>, vector<8xindex>
            -> !xegpu.tensor_desc<8x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{Expects a non-scattered TensorDesc}}
  xegpu.update_nd_offset %1, [0, 2] : !xegpu.tensor_desc<8x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  return
}

// -----
func.func @test_create_tdesc_vc_1(%src: ui64) {
  %0 = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14]> : vector<8xindex>
  // expected-error@+1 {{Expects a scattered TensorDesc}}
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<8xindex> -> !xegpu.tensor_desc<8xf16>
  return
}

// -----
func.func @test_create_tdesc_vc_2(%src: ui64) {
  %0 = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14]> : vector<8xindex>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<8xindex>
  // expected-error@+1 {{expected chunk blocks for 2D tensor}}
          -> !xegpu.tensor_desc<8x4xf16, #xegpu.scatter_tdesc_attr<>>
  return
}

// -----
func.func @test_create_tdesc_vc_3(%src: memref<?xf32>) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  // expected-error@+1 {{Memory space mismatch}}
  %1 = xegpu.create_tdesc %src, %0 : memref<?xf32>, vector<4xindex>
          -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<memory_space = slm, chunk_size = 2>>
  return
}

// -----
func.func @test_create_tdesc_vc_4(%src: memref<?xf32>) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %0 : memref<?xf32>, vector<4xindex>
  // expected-error@+1 {{invalid chunk size}}
          -> !xegpu.tensor_desc<4x5xf32, #xegpu.scatter_tdesc_attr<chunk_size = 5>>
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
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex>
          -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<write_back>}}
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<write_back>}>: !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  return
}

// -----
func.func @test_create_tdesc_sg_map_1(%src: ui64) {
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  // expected-error@+1 {{outer layout distribution and data mapping must be 1 for 1D tensor}}
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.sg_map<wi_layout = [4, 1], wi_data = [1, 1]>>
  return
}

// -----
func.func @test_create_tdesc_sg_map_2(%src: ui64) {
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  // expected-error@+1 {{cannot map over non-contiguous scattered row elements}}
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>, #xegpu.sg_map<wi_layout = [1, 4], wi_data = [2, 1]>>
  return
}

// -----
func.func @test_create_tdesc_sg_map_3(%src: ui64) {
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  // expected-error@+1 {{work item data mapping must match the number of contiguous elements}}
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x3xf32, #xegpu.scatter_tdesc_attr<chunk_size = 3>, #xegpu.sg_map<wi_layout = [4, 1], wi_data = [1, 2]>>
  return
}

// -----
func.func @test_load_gather_sg_map_1(%src: ui64) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>, #xegpu.sg_map<wi_layout = [4, 1], wi_data = [1, 2]>>
  // expected-error@+1 {{Unexpected result shape(Expected shape: [2, 1], Given shape: [1, 2])}}
  %2 = xegpu.load %1, %0 <{l1_hint = #xegpu.cache_hint<cached>, transpose}> : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>, #xegpu.sg_map<wi_layout = [4, 1], wi_data = [1, 2]>>, vector<4xi1> -> vector<1x2xf32>
  return
}

// -----
func.func @test_load_gather_sg_map_2(%src: ui64) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>, #xegpu.sg_map<wi_layout = [4, 1], wi_data = [1, 2]>>
  // expected-error@+1 {{Unexpected result shape(Expected shape: [2, 1], Given shape: [2])}}
  %2 = xegpu.load %1, %0 <{l1_hint = #xegpu.cache_hint<cached>, transpose}> : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>, #xegpu.sg_map<wi_layout = [4, 1], wi_data = [1, 2]>>, vector<4xi1> -> vector<2xf32>
  return
}

// -----
func.func @test_load_gather_sg_map_3(%src: ui64) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>, #xegpu.sg_map<wi_layout = [4, 1], wi_data = [1, 2]>>
  // expected-error@+1 {{Chunk size, vector size and wi_data must match}}
  %2 = xegpu.load %1, %0 <{l1_hint = #xegpu.cache_hint<cached>, transpose}> : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>, #xegpu.sg_map<wi_layout = [4, 1], wi_data = [1, 2]>>, vector<4xi1> -> vector<1xf32>
  return
}


// -----
func.func @test_store_scatter_sg_map_1(%src: ui64) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %val = arith.constant dense<2.9>: vector<1x2xf32>
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>, #xegpu.sg_map<wi_layout = [4, 1], wi_data = [1, 2]>>
  // expected-error@+1 {{Unexpected value shape(Expected shape: [2, 1], Given shape: [1, 2])}}
  xegpu.store %val, %1, %0 <{l1_hint = #xegpu.cache_hint<cached>, transpose}> : vector<1x2xf32>, !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>, #xegpu.sg_map<wi_layout = [4, 1], wi_data = [1, 2]>>, vector<4xi1>
  return
}

// -----
func.func @test_store_scatter_sg_map_2(%src: ui64) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %val = arith.constant dense<2.9>: vector<2xf32>
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>, #xegpu.sg_map<wi_layout = [4, 1], wi_data = [1, 2]>>
  // expected-error@+1 {{Unexpected value shape(Expected shape: [2, 1], Given shape: [2])}}
  xegpu.store %val, %1, %0 <{l1_hint = #xegpu.cache_hint<cached>, transpose}> : vector<2xf32>, !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>, #xegpu.sg_map<wi_layout = [4, 1], wi_data = [1, 2]>>, vector<4xi1>
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
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<1>: vector<4xi1>
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex>
        -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<write_back>}}
  %2 = xegpu.load %1, %0 <{l1_hint = #xegpu.cache_hint<write_back>}>
        : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1>
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
  %cst = arith.constant dense<[0, 8, 16, 24]>: vector<4xindex>
  %0 = arith.constant dense<1>: vector<4xi1>
  %1 = arith.constant dense<2.9>: vector<4x2xf32>
  %2 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex>
              -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<streaming>}}
  xegpu.store %1, %2, %0 <{l1_hint = #xegpu.cache_hint<streaming>}> : vector<4x2xf32>,
          !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1>
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
func.func @test_atomic_rmw(%src: ui64, %value : vector<16x4xf32>, %mask : vector<16xi1>) {
  %0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8>>
  // expected-error@+1 {{failed to verify that all of {tensorDesc, value, result} have same shape}}
  xegpu.atomic_rmw addf %1, %mask, %value: !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<16xi1>, vector<16x4xf32> -> vector<16x8xf32>
  return
}

// -----
func.func @tensor_desc_invalid_rank(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{expected 1D or 2D tensor}}
      !xegpu.tensor_desc<16x2x2xf32>
  return
}

// -----
func.func @tensor_desc_invalid_rank_1(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{expected 1D or 2D tensor}}
      !xegpu.tensor_desc<f32>
  return
}

// -----
func.func @tensor_desc_1D_invalid_map_layout(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{outer layout distribution and data mapping must be 1 for 1D tensor}}
      !xegpu.tensor_desc<16xf32, #xegpu.sg_map<wi_layout = [2, 16], wi_data = [1, 1]>>
  return
}

// -----
func.func @tensor_desc_1D_invalid_map_data(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{outer layout distribution and data mapping must be 1 for 1D tensor}}
      !xegpu.tensor_desc<16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [2, 1]>>
  return
}

// -----
func.func @tensor_desc_invalid_map_layout(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{cannot distribute 8 over 16 work items with 1 elements each}}
      !xegpu.tensor_desc<4x8xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
  return
}

// -----
func.func @tensor_desc_invalid_map_layout_1(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{cannot distribute 4 over 8 work items with 1 elements each}}
      !xegpu.tensor_desc<4x8xf32, #xegpu.sg_map<wi_layout = [8, 2], wi_data = [1, 1]>>
  return
}

// -----
func.func @tensor_desc_invalid_map_data(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{cannot distribute 4 over 2 work items with 4 elements each}}
      !xegpu.tensor_desc<4x8xf32, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [4, 1]>>
  return
}

// -----
func.func @tensor_desc_invalid_map_data_1(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{cannot distribute 4 over 8 work items with 1 elements each}}
      !xegpu.tensor_desc<4x8xf32, #xegpu.sg_map<wi_layout = [8, 2], wi_data = [1, 2]>>
  return
}

// -----
func.func @tensor_desc_scatter_invalid_map_data(%src: ui64) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex> ->
      // expected-error@+1 {{cannot map over non-contiguous scattered row elements}}
      !xegpu.tensor_desc<4x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        #xegpu.sg_map<wi_layout = [1, 1], wi_data = [2, 1]>>
  return
}

// -----
func.func @tensor_desc_scatter_invalid_map_data_1(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      // expected-error@+1 {{work item data mapping must match the number of contiguous elements}}
      !xegpu.tensor_desc<16xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 1>,
        #xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 2]>>
  return
}

// -----
func.func @tensor_desc_scatter_invalid_chunk_size_1D(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      // expected-error@+1 {{expected non-contiguous elements for 1D tensor}}
      !xegpu.tensor_desc<16xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        #xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 2]>>
  return
}

// -----
func.func @tensor_desc_scatter_invalid_chunk_size_2D(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      // expected-error@+1 {{expected chunk blocks for 2D tensor}}
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 1>,
        #xegpu.sg_map<wi_layout = [8, 1], wi_data = [1, 2]>>
  return
}
