// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
func.func @create_nd_tdesc_1(%src: memref<24xf32>) {
  // expected-error@+1 {{Expecting the TensorDesc rank is not greater than the ranks of shape, strides, offsets or the memref source}}
  %1 = xegpu.create_nd_tdesc %src[0] : memref<24xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// -----

func.func @create_nd_tdesc_2(%src: memref<24x32xf32>) {
  // expected-error@+1 {{TensorDesc should have the same element type with the source if it is a memref}}
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf16>
  return
}

// -----
func.func @create_nd_tdesc_3(%src: memref<2x24x32xf32, 3>) {
  // expected-error@+1 {{SLM is only supported for 1D block tensor}}
  %1 = xegpu.create_nd_tdesc %src[0, 0, 0] : memref<2x24x32xf32, 3> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = slm>>
  return
}

// -----
func.func @create_nd_tdesc_4(%src: memref<2x24x32xf32, 3>) {
  // expected-error@+1 {{Memory space mismatch}}
  %1 = xegpu.create_nd_tdesc %src[0, 0, 0] : memref<2x24x32xf32, 3> -> !xegpu.tensor_desc<16xf32>
  return
}

// -----
func.func @create_nd_tdesc_5(%src: memref<128x128xf32>) {
  // expected-error@+1 {{cannot distribute [128, 128] using #xegpu.layout<sg_layout = [4, 2], sg_data = [24, 48]>}}
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [24, 48]>>
  return
}

// -----
func.func @create_nd_tdesc_6(%src: memref<128x128xf32>) {
  // expected-error@+1 {{cannot distribute [128, 128] using #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64], inst_data = [24, 48]>}}
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64], inst_data = [24, 48]>>
  return
}

// -----
func.func @create_nd_tdesc_7(%src: memref<128x128xf32>) {
  // expected-error@+1 {{cannot distribute [128, 128] using #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64], inst_data = [64, 32]>}}
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64], inst_data = [64, 32]>>
  return
}

// -----
func.func @create_nd_tdesc_8(%src: ui64) {
  // expected-error@+1 {{'xegpu.create_nd_tdesc' op Expecting strides and shape to be present for integer source}}
  %1 = xegpu.create_nd_tdesc %src : ui64-> !xegpu.tensor_desc<128x128xf32>
  return
}

// -----
func.func @create_nd_tdesc_9(%src: ui64) {
  // expected-error@+1 {{expected mixed offsets rank to match mixed sizes rank}}
  %1 = xegpu.create_nd_tdesc %src[0, 0] : ui64-> !xegpu.tensor_desc<128x128xf32>
  return
}


// -----
func.func @prefetch_nd_vc_1(%src: memref<24x32xf16>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<write_back>}}
  xegpu.prefetch_nd %1 <{l1_hint = #xegpu.cache_hint<write_back>}>: !xegpu.tensor_desc<8x16xf16>
  return
}

// -----
func.func @prefetch_nd_vc_2(%src: memref<24xf16>) {
  %0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
  %1 = xegpu.create_tdesc %src, %0 : memref<24xf16>, vector<8xindex>
                -> !xegpu.tensor_desc<8xf16, #xegpu.scatter_tdesc_attr<>>
  // expected-error@+1 {{Expects a non-scattered TensorDesc}}
  xegpu.prefetch_nd %1 <{l1_hint = #xegpu.cache_hint<cached>}>
        : !xegpu.tensor_desc<8xf16, #xegpu.scatter_tdesc_attr<>>
  return
}

// -----
func.func @load_nd_vc_1(%src: memref<8x16xf16>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<write_back>}}
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<write_back>}>
      : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  return
}

// -----
func.func @load_nd_vc_2(%src: memref<16xf16>) {
  %0 = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14]> : vector<8xindex>
  %1 = xegpu.create_tdesc %src, %0 : memref<16xf16>, vector<8xindex>
          -> !xegpu.tensor_desc<8x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{Expects a non-scattered TensorDesc.}}
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>}>
      : !xegpu.tensor_desc<8x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>> -> vector<8x2xf16>
  return
}

// -----
func.func @load_nd_vc_3(%src: memref<8x16xf16>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<16xf16>
  // expected-warning@+1 {{Invalid Packed Attr.}}
  %2 = xegpu.load_nd %1 <{packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
        : !xegpu.tensor_desc<16xf16> -> vector<16xf16>
  return
}

// -----
func.func @load_nd_vc_4(%src: memref<24x32xf32>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
    !xegpu.tensor_desc<8x16xf32>
  // expected-error@+1 {{Result shape [8, 1] is not consistent with tensor descriptor}}
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>,
      l2_hint = #xegpu.cache_hint<uncached>}>
    : !xegpu.tensor_desc<8x16xf32> -> vector<8x1xf32>
  return
}

// -----
func.func @subgroup_load_nd_9(%src: memref<4x8x16xf16>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0, 0] : memref<4x8x16xf16> -> !xegpu.tensor_desc<4x8x16xf16>
  // expected-error@+1 {{Expects a 1D or 2D TensorDesc}}
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x8x16xf16> -> vector<4x8x16xf16>
  return
}

// -----
func.func @subgroup_load_nd_offset_1(%src: memref<4x8x16xf16>, %x : index) {
  %1 = xegpu.create_nd_tdesc %src: memref<4x8x16xf16> -> !xegpu.tensor_desc<16xf16>
// expected-error@+1 {{Mismatched ranks between offsets and tensor descriptor}}
  %2 = xegpu.load_nd %1[0, 0] : !xegpu.tensor_desc<16xf16> -> vector<16xf16>
  return
}

// -----
func.func @subgroup_load_nd_offset_2(%src: memref<4x8x16xf16>, %x : index) {
  %3 = xegpu.create_nd_tdesc %src: memref<4x8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
    // expected-error@+1 {{Mismatched ranks between offsets and tensor descriptor}}
  xegpu.prefetch_nd %3[0] : !xegpu.tensor_desc<8x16xf16>
  return
}

// -----
func.func @subgroup_load_nd_offset_3(%src: memref<4x8x16xf16>, %x : index) {  
  %3 = xegpu.create_nd_tdesc %src: memref<4x8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %5 = xegpu.load_nd %3[0, 0] : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    // expected-error@+1 {{Mismatched ranks between offsets and tensor descriptor}}
  xegpu.store_nd %5, %3[%x] : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  return
}

// -----
func.func @load_nd_layout(%src: memref<24x32xf32>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<16xf32>
  // expected-error@+1 {{Result shape [3] is not a valid distribution for tensor descriptor}}
  %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>,
      l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16xf32> -> vector<3xf32>
  return
}

// -----
func.func @load_nd_simt(%src: memref<24x32xf32>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  // expected-error@+1 {{TensorDesc doesn't need LayoutAttr for SIMT code}}
  %2 = xegpu.load_nd %1 : !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8xf32>
  return
}

// -----
func.func @store_nd_vc_1(%dst: memref<24x32xf16>) {
  %1 = arith.constant dense<1.0>: vector<24x32xf16>
  %2 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<streaming>}}
  xegpu.store_nd %1, %2 <{l1_hint = #xegpu.cache_hint<streaming>}>: vector<24x32xf16>, !xegpu.tensor_desc<24x32xf16>
  return
}

// -----
func.func @store_nd_vc_2(%dst: memref<16xf16>) {
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
func.func @store_nd_vc_3(%dst: memref<24x32xf16>) {
  %1 = arith.constant dense<1.0>: vector<2x24x32xf16>
  %2 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16, #xegpu.block_tdesc_attr<array_length = 2>>
  // expected-error@+1 {{array length is not supported by store_nd}}
  xegpu.store_nd %1, %2: vector<2x24x32xf16>, !xegpu.tensor_desc<24x32xf16, #xegpu.block_tdesc_attr<array_length = 2>>
  return
}

// -----
func.func @store_nd_vc_4(%dst: memref<8x24x32xf16>) {
  %1 = arith.constant dense<1.0>: vector<8x24x32xf16>
  %2 = xegpu.create_nd_tdesc %dst[0, 0, 0] : memref<8x24x32xf16> -> !xegpu.tensor_desc<8x24x32xf16>
  // expected-error@+1 {{Expects a 1D or 2D TensorDesc}}
  xegpu.store_nd %1, %2 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<8x24x32xf16>, !xegpu.tensor_desc<8x24x32xf16>
  return
}

// -----
func.func @store_nd_simt(%dst: memref<24x32xf32>, %data: vector<3xf32>) {
  %1 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<16xf32>
  // expected-error@+1 {{Value shape [3] is not a valid distribution for tensor descriptor}}
  xegpu.store_nd %data, %1 : vector<3xf32>, !xegpu.tensor_desc<16xf32>
  return
}

// -----
func.func @store_nd_simt(%src: memref<24x32xf32>, %data: vector<8xf32>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  // expected-error@+1 {{TensorDesc doesn't need LayoutAttr for SIMT code}}
  xegpu.store_nd %data, %1 : vector<8xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  return
}

// -----
func.func @store_nd_vc_5(%dst: memref<24x32xf32>, %data: vector<8x1xf32>) {
  %1 = xegpu.create_nd_tdesc %dst[0, 0] : memref<24x32xf32> ->
    !xegpu.tensor_desc<8x16xf32>
  // expected-error@+1 {{Value shape [8, 1] is not consistent with tensor descriptor}}
  xegpu.store_nd %data, %1 : vector<8x1xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @update_nd_offset_1(%dst: memref<16xf16>) {
  %0 = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14]> : vector<8xindex>
  %1 = xegpu.create_tdesc %dst, %0 : memref<16xf16>, vector<8xindex>
            -> !xegpu.tensor_desc<8x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{Expects a non-scattered TensorDesc}}
  xegpu.update_nd_offset %1, [0, 2] : !xegpu.tensor_desc<8x2xf16, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  return
}

// -----
func.func @create_tdesc_vc_1(%src: ui64) {
  %0 = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14]> : vector<8xindex>
  // expected-error@+1 {{Expects a scattered TensorDesc}}
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<8xindex> -> !xegpu.tensor_desc<8xf16>
  return
}

// -----
func.func @create_tdesc_vc_2(%src: memref<?xf32>) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %0 : memref<?xf32>, vector<4xindex>
  // expected-error@+1 {{invalid chunk size}}
          -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<chunk_size = 0>>
  return
}

// -----
func.func @create_tdesc_vc_3(%src: memref<?xf32>) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  // expected-error@+1 {{Memory space mismatch}}
  %1 = xegpu.create_tdesc %src, %0 : memref<?xf32>, vector<4xindex>
          -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<memory_space = slm, chunk_size = 2>>
  return
}

// -----
func.func @create_tdesc_vc_4(%src: memref<?xf32>) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %0 : memref<?xf32>, vector<4xindex>
  // expected-error@+1 {{expected last dim of tensor to match chunk size}}
          -> !xegpu.tensor_desc<4x5xf32, #xegpu.scatter_tdesc_attr<chunk_size = 4>>
  return
}

// -----
func.func @create_tdesc_vc_5(%src: memref<?xf16>) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %0 : memref<?xf16>, vector<4xindex>
  // expected-error@+1 {{last dim of tensor to be a multiple of 2}}
          -> !xegpu.tensor_desc<4x3xf16, #xegpu.scatter_tdesc_attr<chunk_size = 3>>
  return
}


// -----
func.func @prefetch_vc_1(%src: memref<24x32xf16>) {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16>
  // expected-error@+1 {{Expects a scattered TensorDesc}}
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<write_back>}>: !xegpu.tensor_desc<24x32xf16>
  return
}

// -----
func.func @prefetch_vc_2(%src: ui64) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<4xindex>
          -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<write_back>}}
  xegpu.prefetch %1 <{l1_hint = #xegpu.cache_hint<write_back>}>: !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  return
}

// -----
func.func @create_tdesc_layout_1(%src: ui64) {
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  // expected-error@+1 {{expected layout rank to match tensor rank}}
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex> -> !xegpu.tensor_desc<4xf32, #xegpu.scatter_tdesc_attr<>, #xegpu.layout<lane_layout = [4, 1], lane_data = [1, 1]>>
  return
}

// -----
func.func @create_tdesc_layout_2(%src: ui64) {
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  // expected-error@+1 {{expected last dim of lane_data to be a multiple of: 2}}
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x4xf16, #xegpu.scatter_tdesc_attr<chunk_size = 4>, #xegpu.layout<lane_layout = [4, 1], lane_data = [1, 1]>>
  return
}

// -----
func.func @load_gather_simt_1(%src: ui64) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{Value shape [6] is neither a valid distribution for SIMT nor consistent with the tensor descriptor for SIMD}}
  %2 = xegpu.load %1, %0 <{l1_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1> -> vector<6xf32>
  return
}

// -----
func.func @store_scatter_simt_1(%src: ui64) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %val = arith.constant dense<2.9>: vector<6xf32>
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex> -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{Value shape [6] is neither a valid distribution for SIMT nor consistent with the tensor descriptor for SIMD}}
  xegpu.store %val, %1, %0 <{l1_hint = #xegpu.cache_hint<cached>}> : vector<6xf32>, !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<4xi1>
  return
}

// -----
func.func @load_gather_vc_1(%src: memref<24x32xf16>) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf16> -> !xegpu.tensor_desc<4x2xf16>
  // expected-error@+1 {{Expects a scattered TensorDesc}}
  %2 = xegpu.load %1, %0 <{l1_hint = #xegpu.cache_hint<cached>}>
      : !xegpu.tensor_desc<4x2xf16>, vector<4xi1> -> vector<4x2xf16>
  return
}

// -----
func.func @load_gather_vc_2(%src: ui64) {
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
func.func @load_gather_vc_3(%src: ui64) {
  %cst = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %0 = arith.constant dense<1>: vector<8xi1>
  %1 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex>
        -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{Mask should match TensorDesc except the chunk size dim}}
  %2 = xegpu.load %1, %0
        : !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<8xi1>
          -> vector<4x2xf32>
  return
}

// -----
func.func @prefetch_offset_wi_1(%src: memref<4x4xf32>) {
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  // expected-error@+1 {{Expecting the source is a 1D memref or pointer}}
  xegpu.prefetch %src[%offsets]: memref<4x4xf32>, vector<1xindex>
  return
}

// -----
func.func @load_gather_offset_sg(%src: memref<?xf16>) {
  %offsets = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<8xi1>
  // expected-error@+1 {{Mask should match value except the chunk size dim}}
  %2 = xegpu.load %src[%offsets], %mask
        : memref<?xf16>, vector<4xindex>, vector<8xi1>
          -> vector<4x2xf16>
  return
}

// -----
func.func @load_gather_offset_wi(%src: ui64) {
  %mask = arith.constant dense<1>: vector<1xi1>
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  // expected-error@+1 {{value elements must match chunk size}}
  %2 = xegpu.load %src[%offsets], %mask <{chunk_size = 2}> : ui64,  vector<1xindex>, vector<1xi1> -> vector<4xf32>
  return
}

// -----
func.func @store_scatter_offset_wi_1(%src: memref<?xf16>) {
  %val = arith.constant dense<2.9>: vector<4xf16>
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  %mask = arith.constant dense<1>: vector<1xi1>
  // expected-error@+1 {{value elements must match chunk size}}
  xegpu.store %val, %src[%offsets], %mask 
        : vector<4xf16>, memref<?xf16>, vector<1xindex>, vector<1xi1>
  return
}

// -----
func.func @store_scatter_offset_wi_2(%src: memref<4x4xf16>) {
  %val = arith.constant dense<2.9>: vector<4xf16>
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  %mask = arith.constant dense<1>: vector<1xi1>
  // expected-error@+1 {{Expecting the dest is a 1D memref or pointer}}
  xegpu.store %val, %src[%offsets], %mask 
        : vector<4xf16>, memref<4x4xf16>, vector<1xindex>, vector<1xi1>
  return
}

// -----
func.func @load_gather_offset_wi_2(%src: ui64) {
  %mask = arith.constant dense<1>: vector<1xi1>
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  // expected-error@+1 {{value elements must match chunk size}}
  %2 = xegpu.load %src[%offsets], %mask <{chunk_size = 2}> : ui64,  vector<1xindex>, vector<1xi1> -> vector<4xf16>
  return
}

// -----
func.func @load_gather_offset_wi_1(%src: memref<4x4xf32>) {
  %mask = arith.constant dense<1>: vector<1xi1>
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  // expected-error@+1 {{Expecting the source is a 1D memref or pointer}}
  %2 = xegpu.load %src[%offsets], %mask <{chunk_size = 2}> : memref<4x4xf32>,  vector<1xindex>, vector<1xi1> -> vector<2xf32>
  return
}

// -----
func.func @store_scatter_vc_1(%src: memref<24x32xf32>) {
  %0 = arith.constant dense<1>: vector<4xi1>
  %1 = arith.constant dense<2.9>: vector<4x2xf32>
  %2 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<4x2xf32>
  // expected-error@+1 {{Expects a scattered TensorDesc}}
  xegpu.store %1, %2, %0 <{l1_hint = #xegpu.cache_hint<cached>}>
        : vector<4x2xf32>, !xegpu.tensor_desc<4x2xf32>, vector<4xi1>
  return
}

// -----
func.func @store_scatter_vc_2(%src: ui64) {
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
func.func @store_scatter_vc_3(%src: ui64) {
  %cst = arith.constant dense<[0, 8, 16, 24]>: vector<4xindex>
  %0 = arith.constant dense<1>: vector<8xi1>
  %1 = arith.constant dense<2.9>: vector<4x2xf32>
  %2 = xegpu.create_tdesc %src, %cst : ui64, vector<4xindex>
              -> !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>
  // expected-error@+1 {{Mask should match TensorDesc except the chunk size dim}}
  xegpu.store %1, %2, %0 : vector<4x2xf32>,
          !xegpu.tensor_desc<4x2xf32, #xegpu.scatter_tdesc_attr<chunk_size = 2>>, vector<8xi1>
  return
}

// -----
func.func @dpas_vc_1(%a : vector<8x8xf16>, %b: vector<8x16x2xf16>) {
  // expected-error@+1 {{K-dimension mismatch}}
  %1 = xegpu.dpas %a, %b : vector<8x8xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  return
}

// -----
func.func @dpas_vc_2(%a : vector<8x8x2xf16>, %b: vector<8x16x2xf16>) {
  // expected-error@+1 {{expecting lhs and result to be a 2D vector, and rhs to be either 2D or 3D (packed) vector}}
  %1 = xegpu.dpas %a, %b : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  return
}

// -----
func.func @dpas_3(%a : vector<8x8xf16>, %b: vector<8x16x2xf16>) {
  // expected-error@+1 {{K-dimension mismatch}}
  %1 = xegpu.dpas %a, %b : vector<8x8xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  return
}

// -----
func.func @dpas_4(%a : vector<16x16xf16>, %b: vector<8x16x2xf16>) {
  // expected-error@+1 {{M-dimension mismatch}}
  %1 = xegpu.dpas %a, %b : vector<16x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  return
}

// -----
func.func @dpas_5(%a : vector<8x16xf16>, %b: vector<8x8x2xf16>) {
  // expected-error@+1 {{N-dimension mismatch}}
  %1 = xegpu.dpas %a, %b : vector<8x16xf16>, vector<8x8x2xf16> -> vector<8x16xf32>
  return
}

// -----
func.func @dpas_simt_1(%a : vector<8xf16>, %b: vector<15xf16>) {
  // expected-error@+1 {{Expecting B operand to be a multiple of 32 bits}}
  %1 = xegpu.dpas %a, %b : vector<8xf16>, vector<15xf16> -> vector<8xf32>
  return
}

// -----
func.func @atomic_rmw(%src: ui64, %value : vector<16x4xf32>, %mask : vector<16xi1>) {
  %0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
  %1 = xegpu.create_tdesc %src, %0 : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8>>
  // expected-error@+1 {{failed to verify that all of {tensorDesc, value, result} have same shape}}
  xegpu.atomic_rmw addf %1, %mask, %value: !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8>>, vector<16xi1>, vector<16x4xf32> -> vector<16x8xf32>
  return
}

// -----
func.func @tensor_desc_invalid_rank_1(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{expected non-zero rank tensor}}
      !xegpu.tensor_desc<f32>
  return
}

// -----
func.func @tensor_desc_1D_invalid_map_layout(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{expected layout rank to match tensor rank}}
      !xegpu.tensor_desc<16xf32,  #xegpu.layout<lane_layout = [2, 16], lane_data = [1, 1]>>
  return
}

// -----
func.func @tensor_desc_1D_invalid_map_data(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{expected layout rank to match tensor rank}}
      !xegpu.tensor_desc<16xf32,  #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
  return
}

// -----
func.func @tensor_desc_invalid_map_layout(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{cannot distribute [4, 8] using #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}}
      !xegpu.tensor_desc<4x8xf32,  #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  return
}

// -----
func.func @tensor_desc_invalid_map_layout_1(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{cannot distribute [4, 8] using #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 1]>}}
      !xegpu.tensor_desc<4x8xf32,  #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 1]>>
  return
}

// -----
func.func @tensor_desc_invalid_map_data(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{cannot distribute [4, 8] using #xegpu.layout<lane_layout = [2, 8], lane_data = [4, 1]>}}
      !xegpu.tensor_desc<4x8xf32,  #xegpu.layout<lane_layout = [2, 8], lane_data = [4, 1]>>
  return
}

// -----
func.func @tensor_desc_invalid_map_data_1(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
      // expected-error@+1 {{cannot distribute [4, 8] using #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 2]>}}
      !xegpu.tensor_desc<4x8xf32,  #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 2]>>
  return
}

// -----
func.func @tensor_desc_scatter_invalid_chunk_size_1D(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      // expected-error@+1 {{expected non-contiguous elements for 1D tensor}}
      !xegpu.tensor_desc<16xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
         #xegpu.layout<lane_layout = [1, 8], lane_data = [1, 2]>>
  return
}

// -----
func.func @tensor_desc_scatter_invalid_chunk_size_2D(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      // expected-error@+1 {{expected last dim of tensor to match chunk size}}
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 4>,
         #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 2]>>
  return
}

// -----
func.func @convert_layout_unmatch(%a: vector<32x64xf16>) {
  // expected-error@+1 {{expected input layout and target layout be WgLayout or SgLayout at the same time}}
  %2 = xegpu.convert_layout %a <{input_layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>,
                                target_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<32x64xf16>
  gpu.return
}

// -----
func.func @tensor_desc_invalid_layout_attr(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
         // expected-error@+1 {{expected at least one of sg_layout, inst_data or lane_layout}}
         #xegpu.layout<sg_data = [16, 2], lane_data = [1, 2]>>
  return
}

// -----
func.func @tensor_desc_rank_mismatch(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        // expected-error@+1 {{expected sg_layout and lane_layout to have the same rank}}
        #xegpu.layout<sg_layout = [1, 1, 1], sg_data = [16, 2, 1], lane_layout = [8, 1], lane_data = [1, 2]>>
  return
}

// -----
func.func @tensor_desc_rank_mismatch(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        // expected-error@+1 {{expected sg_layout and inst_data to have the same rank}}
        #xegpu.layout<sg_layout = [1, 1, 1], sg_data = [16, 2, 1], inst_data = [16, 2]>>
  return
}

// -----
func.func @tensor_desc_rank_mismatch(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        // expected-error@+1 {{expected inst_data and lane_layout to have the same rank}}
        #xegpu.layout<inst_data = [16, 2, 1], lane_layout = [8, 1], lane_data = [1, 2]>>
  return
}

// -----
func.func @tensor_desc_rank_mismatch(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        // expected-error@+1 {{expected lane_data and lane_layout to have the same rank}}
        #xegpu.layout<inst_data = [16, 2], lane_layout = [8, 1], lane_data = [1, 2, 1]>>
  return
}

// -----
func.func @tensor_desc_rank_mismatch(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        // expected-error@+1 {{expected sg_data and sg_layout to have the same rank}}
        #xegpu.layout<sg_layout = [1, 1], sg_data = [16, 2, 1], inst_data = [16, 2]>>
  return
}

// -----
func.func @tensor_desc_rank_mismatch(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      // expected-error@+1 {{expected layout rank to match tensor rank}}
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        #xegpu.layout<sg_layout = [1], sg_data = [32], inst_data = [16]>>
  return
}

// -----
func.func @tensor_desc_invalid_sg_data(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        // expected-error@+1 {{expected sg_layout being used with sg_data}}
        #xegpu.layout<sg_data = [16, 2], lane_layout = [8, 1], lane_data = [1, 2]>>
  return
}

// -----
func.func @tensor_desc_rank_mismatch(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        // expected-error@+1 {{expected lane_layout being used with lane_data}}
        #xegpu.layout<inst_data = [16, 2], lane_data = [1, 2]>>
  return
}

// -----
func.func @tensor_desc_rank_mismatch(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        // expected-error@+1 {{expected sg_layout/lane_layout being used with order}}
        #xegpu.layout<inst_data = [16, 2], order = [0, 1]>>
  return
}

// -----
func.func @tensor_desc_rank_mismatch(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        // expected-error@+1 {{expected order and sg_layout to have the same rank}}
        #xegpu.layout<sg_layout = [1, 1], sg_data = [16, 2], order = [0, 1, 2]>>
  return
}

// -----
func.func @tensor_desc_invalid_sg_data(%src: ui64, %offsets: vector<16xindex>) {
  %1 = xegpu.create_tdesc %src, %offsets : ui64, vector<16xindex> ->
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.scatter_tdesc_attr<chunk_size = 2>,
        // expected-error@+1 {{expected order and lane_layout to have the same rank}}
        #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 2], order = [0, 1, 2]>>
  return
}

// -----
#l = #xegpu.layout<sg_layout = [16, 1, 1], sg_data = [1, 8, 2]>
// expected-error@+1 {{repeated dim (2) in slice attribute}}
#s = #xegpu.slice<#l, dims = [2, 2]>
func.func @slice_attr_repeat_dim() {
  %offsets = arith.constant {layout_result_0 = #s} dense<0.8> : vector<16x8xindex>
  return
}

// -----
#l = #xegpu.layout<sg_layout = [16, 1, 1], sg_data = [1, 8, 2]>
// expected-error@+1 {{invalid dim (3) in slice attribute}}
#s = #xegpu.slice<#l, dims = [3]>
func.func @slice_attr_repeat_dim() {
  %offsets = arith.constant {layout_result_0 = #s} dense<0.8> : vector<16x8xindex>
  return
}

