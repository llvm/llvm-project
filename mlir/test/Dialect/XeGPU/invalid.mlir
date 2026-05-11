// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
func.func @create_nd_tdesc_1(%src: memref<24xf32>) {
  // expected-error@+1 {{Expecting the TensorDesc rank is not greater than the ranks of shape, strides or the memref source}}
  %1 = xegpu.create_nd_tdesc %src : memref<24xf32> -> !xegpu.tensor_desc<8x16xf32>
  return
}

// -----

func.func @create_nd_tdesc_2(%src: memref<24x32xf32>) {
  // expected-error@+1 {{TensorDesc should have the same element type with the source if it is a memref}}
  %1 = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf16>
  return
}

// -----
func.func @create_nd_tdesc_3(%src: memref<2x24x32xf32, 3>) {
  // expected-error@+1 {{SLM is only supported for 1D block tensor}}
  %1 = xegpu.create_nd_tdesc %src : memref<2x24x32xf32, 3> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = slm>>
  return
}

// -----
func.func @create_nd_tdesc_4(%src: memref<2x24x32xf32, 3>) {
  // expected-error@+1 {{Memory space mismatch}}
  %1 = xegpu.create_nd_tdesc %src : memref<2x24x32xf32, 3> -> !xegpu.tensor_desc<16xf32>
  return
}

// -----
func.func @create_nd_tdesc_5(%src: memref<128x128xf32>) {
  // expected-error@+1 {{cannot distribute [128, 128] using #xegpu.layout<sg_layout = [4, 2], sg_data = [24, 48]>}}
  %1 = xegpu.create_nd_tdesc %src : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [24, 48]>>
  return
}

// -----
func.func @create_nd_tdesc_6(%src: memref<128x128xf32>) {
  // expected-error@+1 {{cannot distribute [128, 128] using #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64], inst_data = [24, 48]>}}
  %1 = xegpu.create_nd_tdesc %src : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64], inst_data = [24, 48]>>
  return
}

// -----
func.func @create_nd_tdesc_7(%src: memref<128x128xf32>) {
  // expected-error@+1 {{cannot distribute [128, 128] using #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64], inst_data = [64, 32]>}}
  %1 = xegpu.create_nd_tdesc %src : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [32, 64], inst_data = [64, 32]>>
  return
}

// -----
func.func @create_nd_tdesc_8(%src: ui64) {
  // expected-error@+1 {{'xegpu.create_nd_tdesc' op expecting strides and shape to be present for integer source}}
  %1 = xegpu.create_nd_tdesc %src : ui64-> !xegpu.tensor_desc<128x128xf32>
  return
}

// -----
func.func @create_nd_tdesc_9(%src: ui64) {
  // expected-error@+1 {{expecting strides and shape to be present for integer source}}
  %1 = xegpu.create_nd_tdesc %src : ui64-> !xegpu.tensor_desc<128x128xf32>
  return
}

// -----
func.func @create_nd_tdesc_10(%src: memref<24xindex>) {
  // expected-error @+1 {{unsupported element type 'index': expected integer or float}}
  %1 = xegpu.create_nd_tdesc %src : memref<24xindex> -> !xegpu.tensor_desc<24xindex>
  return
}

// -----
func.func @prefetch_nd_vc_1(%src: memref<24x32xf16>) {
  %1 = xegpu.create_nd_tdesc %src : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<write_back>}}
  xegpu.prefetch_nd %1[0, 0] <{l1_hint = #xegpu.cache_hint<write_back>}>: !xegpu.tensor_desc<8x16xf16>
  return
}

// -----
func.func @load_nd_vc_1(%src: memref<8x16xf16>) {
  %1 = xegpu.create_nd_tdesc %src : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<write_back>}}
  %2 = xegpu.load_nd %1[0, 0] <{l1_hint = #xegpu.cache_hint<write_back>}>
      : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  return
}

// -----
func.func @load_nd_vc_3(%src: memref<8x16xf16>) {
  %1 = xegpu.create_nd_tdesc %src : memref<8x16xf16> -> !xegpu.tensor_desc<16xf16>
  // expected-warning@+1 {{Invalid Packed Attr.}}
  %2 = xegpu.load_nd %1[0] <{packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
        : !xegpu.tensor_desc<16xf16> -> vector<16xf16>
  return
}

// -----
func.func @load_nd_vc_4(%src: memref<24x32xf32>) {
  %1 = xegpu.create_nd_tdesc %src : memref<24x32xf32> ->
    !xegpu.tensor_desc<8x16xf32>
  // expected-error@+1 {{Result shape [8, 1] is not consistent with tensor descriptor}}
  %2 = xegpu.load_nd %1[0, 0] <{l1_hint = #xegpu.cache_hint<cached>,
      l2_hint = #xegpu.cache_hint<uncached>}>
    : !xegpu.tensor_desc<8x16xf32> -> vector<8x1xf32>
  return
}

// -----
func.func @subgroup_load_nd_9(%src: memref<4x8x16xf16>) {
  %1 = xegpu.create_nd_tdesc %src : memref<4x8x16xf16> -> !xegpu.tensor_desc<4x8x16xf16>
  // expected-error@+1 {{Expects a 1D or 2D TensorDesc}}
  %2 = xegpu.load_nd %1[0, 0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<4x8x16xf16> -> vector<4x8x16xf16>
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
  %1 = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<16xf32>
  // expected-error@+1 {{Result shape [3] is not a valid distribution for tensor descriptor}}
  %2 = xegpu.load_nd %1[0] <{l1_hint = #xegpu.cache_hint<cached>,
      l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16xf32> -> vector<3xf32>
  return
}

// -----
func.func @load_nd_simt(%src: memref<24x32xf32>) {
  %1 = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  // expected-error@+1 {{TensorDesc doesn't need LayoutAttr for SIMT code}}
  %2 = xegpu.load_nd %1[0, 0] : !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8xf32>
  return
}

// -----
func.func @store_nd_vc_1(%dst: memref<24x32xf16>) {
  %1 = arith.constant dense<1.0>: vector<24x32xf16>
  %2 = xegpu.create_nd_tdesc %dst : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<streaming>}}
  xegpu.store_nd %1, %2[0, 0] <{l1_hint = #xegpu.cache_hint<streaming>}>: vector<24x32xf16>, !xegpu.tensor_desc<24x32xf16>
  return
}

// -----
func.func @store_nd_vc_3(%dst: memref<24x32xf16>) {
  %1 = arith.constant dense<1.0>: vector<2x24x32xf16>
  %2 = xegpu.create_nd_tdesc %dst : memref<24x32xf16> -> !xegpu.tensor_desc<24x32xf16, #xegpu.block_tdesc_attr<array_length = 2>>
  // expected-error@+1 {{array length is not supported by store_nd}}
  xegpu.store_nd %1, %2[0, 0]: vector<2x24x32xf16>, !xegpu.tensor_desc<24x32xf16, #xegpu.block_tdesc_attr<array_length = 2>>
  return
}

// -----
func.func @store_nd_vc_4(%dst: memref<8x24x32xf16>) {
  %1 = arith.constant dense<1.0>: vector<8x24x32xf16>
  %2 = xegpu.create_nd_tdesc %dst : memref<8x24x32xf16> -> !xegpu.tensor_desc<8x24x32xf16>
  // expected-error@+1 {{Expects a 1D or 2D TensorDesc}}
  xegpu.store_nd %1, %2[0, 0, 0] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<8x24x32xf16>, !xegpu.tensor_desc<8x24x32xf16>
  return
}

// -----
func.func @store_nd_simt(%dst: memref<24x32xf32>, %data: vector<3xf32>) {
  %1 = xegpu.create_nd_tdesc %dst : memref<24x32xf32> -> !xegpu.tensor_desc<16xf32>
  // expected-error@+1 {{Value shape [3] is not a valid distribution for tensor descriptor}}
  xegpu.store_nd %data, %1[0] : vector<3xf32>, !xegpu.tensor_desc<16xf32>
  return
}

// -----
func.func @store_nd_simt(%src: memref<24x32xf32>, %data: vector<8xf32>) {
  %1 = xegpu.create_nd_tdesc %src : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  // expected-error@+1 {{TensorDesc doesn't need LayoutAttr for SIMT code}}
  xegpu.store_nd %data, %1[0, 0] : vector<8xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  return
}

// -----
func.func @store_nd_vc_5(%dst: memref<24x32xf32>, %data: vector<8x1xf32>) {
  %1 = xegpu.create_nd_tdesc %dst : memref<24x32xf32> ->
    !xegpu.tensor_desc<8x16xf32>
  // expected-error@+1 {{Value shape [8, 1] is not consistent with tensor descriptor}}
  xegpu.store_nd %data, %1[0, 0] : vector<8x1xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
func.func @prefetch_vc_2(%src: memref<?xf32>) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<write_back>}}
  xegpu.prefetch %src[%0] <{l1_hint = #xegpu.cache_hint<write_back>}> : memref<?xf32>, vector<4xindex>
  return
}

// -----
func.func @load_gather_vc_2(%src: memref<?xf32>) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = arith.constant dense<1>: vector<4xi1>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<write_back>}}
  %2 = xegpu.load %src[%0], %1 <{l1_hint = #xegpu.cache_hint<write_back>}>
      : memref<?xf32>, vector<4xindex>, vector<4xi1> -> vector<4xf32>
  return
}

// -----
func.func @load_gather_vc_3(%src: memref<?xf32>) {
  %offsets = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<8xi1>
  // expected-error@+1 {{Mask should match value except the chunk size dim}}
  %2 = xegpu.load %src[%offsets], %mask <{chunk_size = 2}>
      : memref<?xf32>, vector<4xindex>, vector<8xi1> -> vector<4x2xf32>
  return
}

// -----
func.func @load_gather_simt_1(%src: memref<?xf32>) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = arith.constant dense<1>: vector<4xi1>
  // expected-error@+1 {{value elements must match chunk size}}
  %2 = xegpu.load %src[%0], %1 <{chunk_size = 2}>
      : memref<?xf32>, vector<4xindex>, vector<4xi1> -> vector<6xf32>
  return
}

// -----
func.func @store_scatter_vc_2(%dst: memref<?xf32>) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = arith.constant dense<1>: vector<4xi1>
  %2 = arith.constant dense<2.9>: vector<4xf32>
  // expected-error@+1 {{invalid l1_hint: #xegpu.cache_hint<streaming>}}
  xegpu.store %2, %dst[%0], %1 <{l1_hint = #xegpu.cache_hint<streaming>}>
      : vector<4xf32>, memref<?xf32>, vector<4xindex>, vector<4xi1>
  return
}

// -----
func.func @store_scatter_vc_3(%dst: memref<?xf32>) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = arith.constant dense<1>: vector<8xi1>
  %2 = arith.constant dense<2.9>: vector<4x2xf32>
  // expected-error@+1 {{Mask should match value except the chunk size dim}}
  xegpu.store %2, %dst[%0], %1 <{chunk_size = 2}>
      : vector<4x2xf32>, memref<?xf32>, vector<4xindex>, vector<8xi1>
  return
}

// -----
func.func @store_scatter_simt_1(%dst: memref<?xf32>) {
  %0 = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %1 = arith.constant dense<1>: vector<4xi1>
  %2 = arith.constant dense<2.9>: vector<6xf32>
  // expected-error@+1 {{value elements must match chunk size}}
  xegpu.store %2, %dst[%0], %1 <{chunk_size = 2}>
      : vector<6xf32>, memref<?xf32>, vector<4xindex>, vector<4xi1>
  return
}

// -----
func.func @prefetch_offset_wi_1(%src: memref<4x4xf32>) {
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  // expected-error@+1 {{op operand #0 must be 1D memref}}
  xegpu.prefetch %src[%offsets]: memref<4x4xf32>, vector<1xindex>
  return
}

// -----
func.func @prefetch_offset_wi_4(%src: memref<16xf32>) {
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  // expected-error@+1 {{offset_align_byte only allowed with integer source.}}
  xegpu.prefetch %src[%offsets] <{offset_align_byte = 4}>: memref<16xf32>, vector<1xindex>
  return
}

// -----
func.func @prefetch_offset_wi_5(%src: i64) {
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  // expected-error@+1 {{offset_align_byte is required with integer source.}}
  xegpu.prefetch %src[%offsets] : i64, vector<1xindex>
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
  %2 = xegpu.load %src[%offsets], %mask <{chunk_size = 2}> : ui64,  vector<1xindex>, vector<1xi1> -> vector<3xf32>
  return
}

// -----
func.func @store_scatter_offset_wi_1(%src: memref<?xf16>) {
  %val = arith.constant dense<2.9>: vector<4xf16>
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  %mask = arith.constant dense<1>: vector<1xi1>
  // expected-error@+1 {{Mask should match value except the chunk size dim}}
  xegpu.store %val, %src[%offsets], %mask
        : vector<4xf16>, memref<?xf16>, vector<1xindex>, vector<1xi1>
  return
}

// -----
func.func @store_scatter_offset_wi_2(%src: memref<4x4xf16>) {
  %val = arith.constant dense<2.9>: vector<4xf16>
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  %mask = arith.constant dense<1>: vector<1xi1>
  // expected-error@+1 {{op operand #1 must be 1D memref}}
  xegpu.store %val, %src[%offsets], %mask
        : vector<4xf16>, memref<4x4xf16>, vector<1xindex>, vector<1xi1>
  return
}

// -----
func.func @store_scatter_offset_wi_4(%src: !xegpu.tensor_desc<1x1xf32>) {
  %val = arith.constant dense<2.9>: vector<1xf16>
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  %mask = arith.constant dense<1>: vector<1xi1>
  // expected-error@+1 {{op operand #1 must be 1D memref}}
  xegpu.store %val, %src[%offsets], %mask
        : vector<1xf16>, !xegpu.tensor_desc<1x1xf32>, vector<1xindex>, vector<1xi1>
  return
}

// -----
func.func @load_gather_offset_wi_4(%src: !xegpu.tensor_desc<1x2xf16>) {
  %mask = arith.constant dense<1>: vector<1xi1>
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  // expected-error@+1 {{op operand #0 must be 1D memref}}
  %2 = xegpu.load %src[%offsets], %mask <{chunk_size = 2}> : !xegpu.tensor_desc<1x2xf16>, vector<1xindex>, vector<1xi1> -> vector<2xf16>
  return
}

// -----
func.func @load_gather_offset_wi_2(%src: ui64) {
  %mask = arith.constant dense<1>: vector<1xi1>
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  // expected-error@+1 {{value elements must match chunk size}}
  %2 = xegpu.load %src[%offsets], %mask <{chunk_size = 2}> : ui64,  vector<1xindex>, vector<1xi1> -> vector<3xf16>
  return
}

// -----
func.func @load_gather_offset_wi_1(%src: memref<4x4xf32>) {
  %mask = arith.constant dense<1>: vector<1xi1>
  %offsets = arith.constant dense<[0]> : vector<1xindex>
  // expected-error@+1 {{op operand #0 must be 1D memref}}
  %2 = xegpu.load %src[%offsets], %mask <{chunk_size = 2}> : memref<4x4xf32>,  vector<1xindex>, vector<1xi1> -> vector<2xf32>
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
  // expected-error@+1 {{op A operand must be a 2D vector}}
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
func.func @tensor_desc_invalid_rank_1(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src : memref<24x32xf32> ->
      // expected-error@+1 {{expected non-zero rank tensor}}
      !xegpu.tensor_desc<f32>
  return
}

// -----
func.func @tensor_desc_1D_invalid_map_layout(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src : memref<24x32xf32> ->
      // expected-error@+1 {{expected layout rank to match tensor rank}}
      !xegpu.tensor_desc<16xf32,  #xegpu.layout<lane_layout = [2, 16], lane_data = [1, 1]>>
  return
}

// -----
func.func @tensor_desc_1D_invalid_map_data(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src : memref<24x32xf32> ->
      // expected-error@+1 {{expected layout rank to match tensor rank}}
      !xegpu.tensor_desc<16xf32,  #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
  return
}

// -----
func.func @tensor_desc_invalid_map_layout(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src : memref<24x32xf32> ->
      // expected-error@+1 {{cannot distribute [4, 8] using #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}}
      !xegpu.tensor_desc<4x8xf32,  #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  return
}

// -----
func.func @tensor_desc_invalid_map_layout_1(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src : memref<24x32xf32> ->
      // expected-error@+1 {{cannot distribute [4, 8] using #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 1]>}}
      !xegpu.tensor_desc<4x8xf32,  #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 1]>>
  return
}

// -----
func.func @tensor_desc_invalid_map_data_1(%src: memref<24x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src : memref<24x32xf32> ->
      // expected-error@+1 {{cannot distribute [4, 8] using #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 2]>}}
      !xegpu.tensor_desc<4x8xf32,  #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 2]>>
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
func.func @layout_rank_mismatch_sg_lane(%src: memref<?xf32>) {
  %offsets = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<4xi1>
  %2 = xegpu.load %src[%offsets], %mask
      // expected-error@below {{expected sg_layout and lane_layout to have the same rank}}
      {layout = #xegpu.layout<sg_layout = [1, 1, 1], sg_data = [16, 2, 1], lane_layout = [8, 1], lane_data = [1, 2]>}
      : memref<?xf32>, vector<4xindex>, vector<4xi1> -> vector<4xf32>
  return
}

// -----
func.func @layout_rank_mismatch_sg_inst(%src: memref<?xf32>) {
  %offsets = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<4xi1>
  %2 = xegpu.load %src[%offsets], %mask
      // expected-error@below {{expected sg_layout and inst_data to have the same rank}}
      {layout = #xegpu.layout<sg_layout = [1, 1, 1], sg_data = [16, 2, 1], inst_data = [16, 2]>}
      : memref<?xf32>, vector<4xindex>, vector<4xi1> -> vector<4xf32>
  return
}

// -----
func.func @layout_rank_mismatch_inst_lane(%src: memref<?xf32>) {
  %offsets = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<4xi1>
  %2 = xegpu.load %src[%offsets], %mask
      // expected-error@below {{expected inst_data and lane_layout to have the same rank}}
      {layout = #xegpu.layout<inst_data = [16, 2, 1], lane_layout = [8, 1], lane_data = [1, 2]>}
      : memref<?xf32>, vector<4xindex>, vector<4xi1> -> vector<4xf32>
  return
}

// -----
func.func @layout_rank_mismatch_lane_data(%src: memref<?xf32>) {
  %offsets = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<4xi1>
  %2 = xegpu.load %src[%offsets], %mask
      // expected-error@below {{expected lane_data and lane_layout to have the same rank}}
      {layout = #xegpu.layout<inst_data = [16, 2], lane_layout = [8, 1], lane_data = [1, 2, 1]>}
      : memref<?xf32>, vector<4xindex>, vector<4xi1> -> vector<4xf32>
  return
}

// -----
func.func @layout_rank_mismatch_sg_data(%src: memref<?xf32>) {
  %offsets = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<4xi1>
  %2 = xegpu.load %src[%offsets], %mask
      // expected-error@below {{expected sg_data and sg_layout to have the same rank}}
      {layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [16, 2, 1], inst_data = [16, 2]>}
      : memref<?xf32>, vector<4xindex>, vector<4xi1> -> vector<4xf32>
  return
}

// -----
func.func @layout_rank_mismatch_tensor(%src: memref<16x32xf32>) {
  %0 = xegpu.create_nd_tdesc %src : memref<16x32xf32> ->
      // expected-error@+1 {{expected layout rank to match tensor rank}}
      !xegpu.tensor_desc<16x2xf32,
        #xegpu.layout<sg_layout = [1], sg_data = [32], inst_data = [16]>>
  return
}

// -----
func.func @layout_sg_data_missing(%src: memref<?xf32>) {
  %offsets = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<4xi1>
  %2 = xegpu.load %src[%offsets], %mask
      // expected-error@below {{sg_layout and sg_data must be used together}}
      {layout = #xegpu.layout<sg_layout = [2, 1], lane_layout = [8, 1], lane_data = [1, 2]>}
      : memref<?xf32>, vector<4xindex>, vector<4xi1> -> vector<4xf32>
  return
}

// -----
func.func @layout_lane_data_missing(%src: memref<?xf32>) {
  %offsets = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<4xi1>
  %2 = xegpu.load %src[%offsets], %mask
      // expected-error@below {{lane_layout and lane_data must be used together}}
      {layout = #xegpu.layout<inst_data = [16, 2], lane_layout = [16, 1]>}
      : memref<?xf32>, vector<4xindex>, vector<4xi1> -> vector<4xf32>
  return
}

// -----
func.func @layout_order_without_layout(%src: memref<?xf32>) {
  %offsets = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<4xi1>
  %2 = xegpu.load %src[%offsets], %mask
      // expected-error@below {{expected sg_layout/lane_layout being used with order}}
      {layout = #xegpu.layout<inst_data = [16, 2], order = [0, 1]>}
      : memref<?xf32>, vector<4xindex>, vector<4xi1> -> vector<4xf32>
  return
}

// -----
func.func @layout_order_rank_mismatch_sg(%src: memref<?xf32>) {
  %offsets = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<4xi1>
  %2 = xegpu.load %src[%offsets], %mask
      // expected-error@below {{expected order and sg_layout to have the same rank}}
      {layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [16, 2], order = [0, 1, 2]>}
      : memref<?xf32>, vector<4xindex>, vector<4xi1> -> vector<4xf32>
  return
}

// -----
func.func @layout_order_rank_mismatch_lane(%src: memref<?xf32>) {
  %offsets = arith.constant dense<[0, 8, 16, 24]> : vector<4xindex>
  %mask = arith.constant dense<1>: vector<4xi1>
  %2 = xegpu.load %src[%offsets], %mask
      // expected-error@below {{expected order and lane_layout to have the same rank}}
      {layout = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 2], order = [0, 1, 2]>}
      : memref<?xf32>, vector<4xindex>, vector<4xi1> -> vector<4xf32>
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
func.func @create_mem_desc_non_slm() {
  %m = memref.alloca() {alignment = 1024} : memref<2048xi8, 1>
  // expected-error@+1 {{operand #0 must be reside in share memory and statically 1d shaped memref }}
  %mem_desc = xegpu.create_mem_desc %m : memref<2048xi8, 1> -> !xegpu.mem_desc<16x64xf16>
  return
}

// -----
func.func @create_mem_desc_mismatch_sizes() {
  %m = memref.alloca() {alignment = 1024} : memref<2048xi8, 3>
  // expected-error@+1 {{failed to verify that all of {source, mem_desc} have same size in bits}}
  %mem_desc = xegpu.create_mem_desc %m : memref<2048xi8, 3> -> !xegpu.mem_desc<16x32xf16>
  return
}

// -----
func.func @load_mem_desc_mismatch_element_type(%arg0: !xegpu.mem_desc<16x64xf16>) {
  // expected-error@+1 {{failed to verify that all of {mem_desc, res} have same element type}}
  %data = xegpu.load_matrix %arg0[8, 8]: !xegpu.mem_desc<16x64xf16> -> vector<8x16xf32>
  return
}

// -----
func.func @load_mem_desc_invalid_result_size(%arg0: !xegpu.mem_desc<16x64xf16>) {
  // expected-error@+1 {{data shape must not exceed mem_desc shape}}
  %data = xegpu.load_matrix %arg0[8, 8]: !xegpu.mem_desc<16x64xf16> -> vector<32x16xf16>
  return
}

// -----
func.func @load_mem_desc_invalid_rank(%arg0: !xegpu.mem_desc<64xf16>) {
  // expected-error@+1 {{mem_desc must be 2D or greater}}
  %data = xegpu.load_matrix %arg0[16]: !xegpu.mem_desc<64xf16> -> vector<16xf16>
  return
}

// -----
func.func @store_mem_desc_mismatch_element_type(%arg0: !xegpu.mem_desc<16x64xf16>, %arg1: vector<16x16xf32>) {
  // expected-error@+1 {{failed to verify that all of {mem_desc, data} have same element type}}
  xegpu.store_matrix %arg1, %arg0[8, 8] : vector<16x16xf32>, !xegpu.mem_desc<16x64xf16>
  return
}

// -----
func.func @store_mem_desc_invalid_data_size(%arg0: !xegpu.mem_desc<16x64xf16>, %arg1: vector<32x32xf16>) {
  // expected-error@+1 {{data shape must not exceed mem_desc shape}}
  xegpu.store_matrix %arg1, %arg0[8, 8] : vector<32x32xf16>, !xegpu.mem_desc<16x64xf16>
  return
}

// -----
func.func @store_mem_desc_invalid_rank(%arg0: !xegpu.mem_desc<64xf16>, %arg1: vector<32xf16>) {
  // expected-error@+1 {{mem_desc must be 2D or greater}}
  xegpu.store_matrix %arg1, %arg0[32] : vector<32xf16>, !xegpu.mem_desc<64xf16>
  return
}

// -----
func.func @simt_store_matrix_vector_nonlinear(%arg0: !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<stride = [32, 1]>>, %arg1: vector<2x16xf32>) {
  // expected-error@+1 {{With subgroup_block_io, accessed data must be contiguous and coalesced}}
  xegpu.store_matrix %arg1, %arg0[0, 0] {subgroup_block_io, layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} :
        vector<2x16xf32>, !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<stride = [32, 1]>>
  return
}

// -----
func.func @simt_store_matrix_vector_noncoalesced(%arg0: !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<stride = [1, 32], block = [1, 16]>>, %arg1: vector<16x2xf32>) {
  // expected-error@+1 {{With subgroup_block_io, the distributed dimensions must be contiguous}}
  xegpu.store_matrix %arg1, %arg0[0, 0] {subgroup_block_io, layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 2]>} :
        vector<16x2xf32>, !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<stride = [1, 32], block = [1, 16]>>
  return
}

// -----
func.func @simt_store_matrix_vector_noncoalesced(%arg0: !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<stride = [32, 1], block = [1, 17]>>, %arg1: vector<16x2xf32>) {
  // expected-error@+1 {{With subgroup_block_io, the block shape must match the lane layout}}
  xegpu.store_matrix %arg1, %arg0[0, 0] {subgroup_block_io, layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} :
        vector<16x2xf32>, !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<stride = [32, 1], block = [1, 17]>>
  return
}

// -----
func.func @truncf_invalid_result_size(%a: vector<8x16xf16>) {
  // expected-error@+1 {{op input type must be wider than result type}}
  %1 = xegpu.truncf %a : vector<8x16xf16> -> vector<8x16xf32>
  return
}

// -----
func.func @dpas_mx_acc_result_type_mismatch(%a : vector<8x16xf8E5M2>, %b: vector<16x16xf8E5M2>, %acc: vector<8x16xbf16>) {
  // expected-error@+1 {{Accumulator type must match result type.}}
  %1 = xegpu.dpas_mx %a, %b, %acc : vector<8x16xf8E5M2>, vector<16x16xf8E5M2>, vector<8x16xbf16> -> vector<8x16xf32>
  return
}

// -----
func.func @dpas_mx_a_not_2d(%a : vector<128xf8E5M2>, %b: vector<16x16xf8E5M2>) {
  // expected-error@+1 {{A operand must be a 2D vector.}}
  %1 = xegpu.dpas_mx %a, %b : vector<128xf8E5M2>, vector<16x16xf8E5M2> -> vector<8x16xf32>
  return
}

// -----
func.func @dpas_mx_b_not_2d(%a : vector<8x16xf8E5M2>, %b: vector<256xf8E5M2>) {
  // expected-error@+1 {{B operand must be a 2D or 3D vector.}}
  %1 = xegpu.dpas_mx %a, %b : vector<8x16xf8E5M2>, vector<256xf8E5M2> -> vector<8x16xf32>
  return
}

// -----
func.func @dpas_mx_result_not_2d(%a : vector<8x16xf8E5M2>, %b: vector<16x16xf8E5M2>) {
  // expected-error@+1 {{Result must be a 2D vector.}}
  %1 = xegpu.dpas_mx %a, %b : vector<8x16xf8E5M2>, vector<16x16xf8E5M2> -> vector<128xf32>
  return
}

// -----
func.func @dpas_mx_k_dimension_mismatch(%a : vector<8x16xf8E5M2>, %b: vector<8x16xf8E5M2>) {
  // expected-error@+1 {{K-dimension mismatch: A has K=16 but B has K=8.}}
  %1 = xegpu.dpas_mx %a, %b : vector<8x16xf8E5M2>, vector<8x16xf8E5M2> -> vector<8x16xf32>
  return
}

// -----
func.func @dpas_mx_m_dimension_mismatch(%a : vector<8x16xf8E5M2>, %b: vector<16x16xf8E5M2>) {
  // expected-error@+1 {{M-dimension mismatch: A has M=8 but result has M=16.}}
  %1 = xegpu.dpas_mx %a, %b : vector<8x16xf8E5M2>, vector<16x16xf8E5M2> -> vector<16x16xf32>
  return
}

// -----
func.func @dpas_mx_n_dimension_mismatch(%a : vector<8x16xf8E5M2>, %b: vector<16x16xf8E5M2>) {
  // expected-error@+1 {{N-dimension mismatch: B has N=16 but result has N=8.}}
  %1 = xegpu.dpas_mx %a, %b : vector<8x16xf8E5M2>, vector<16x16xf8E5M2> -> vector<8x8xf32>
  return
}


// -----
func.func @dpas_mx_scale_a_m_mismatch(%a : vector<8x16xf8E5M2>, %b: vector<16x16xf8E5M2>, %acc: vector<8x16xf32>, %scale_a: vector<4x2xf8E8M0FNU>) {
  // expected-error@+1 {{Scale A M dimension [4] must match A M dimension [8].}}
  %1 = xegpu.dpas_mx %a, %b, %acc scale_a = %scale_a : vector<8x16xf8E5M2>, vector<16x16xf8E5M2>, vector<8x16xf32>, vector<4x2xf8E8M0FNU> -> vector<8x16xf32>
  return
}

// -----
func.func @dpas_mx_scale_b_n_mismatch(%a : vector<8x16xf8E5M2>, %b: vector<16x16xf8E5M2>, %acc: vector<8x16xf32>, %scale_a: vector<8x2xf8E8M0FNU>, %scale_b: vector<2x8xf8E8M0FNU>) {
  // expected-error@+1 {{Scale B N dimension [8] must match B N dimension [16].}}
  %1 = xegpu.dpas_mx %a, %b, %acc scale_a = %scale_a scale_b = %scale_b : vector<8x16xf8E5M2>, vector<16x16xf8E5M2>, vector<8x16xf32>, vector<8x2xf8E8M0FNU>, vector<2x8xf8E8M0FNU> -> vector<8x16xf32>
  return
}

// -----
func.func @dpas_mx_scale_k_mismatch(%a : vector<8x16xf8E5M2>, %b: vector<16x16xf8E5M2>, %acc: vector<8x16xf32>, %scale_a_val: vector<8x2xf8E8M0FNU>, %scale_b_val: vector<4x16xf8E8M0FNU>) {
  // expected-error@+1 {{Scale K dimension mismatch: scale_a has K=2 but scale_b has K=4.}}
  %1 = xegpu.dpas_mx %a, %b, %acc scale_a = %scale_a_val scale_b = %scale_b_val : vector<8x16xf8E5M2>, vector<16x16xf8E5M2>, vector<8x16xf32>, vector<8x2xf8E8M0FNU>, vector<4x16xf8E8M0FNU> -> vector<8x16xf32>
  return
}

// -----
#layout_a = #xegpu.layout<sg_layout = [1, 1], sg_data = [8, 32]>
#layout_b = #xegpu.layout<sg_layout = [1, 1], sg_data = [32, 16]>
#layout_cd = #xegpu.layout<sg_layout = [1, 1], sg_data = [8, 16]>
func.func @dpas_mx_layout_not_distributable(%a : vector<8x16xf8E5M2>, %b: vector<16x16xf8E5M2>) {
  // expected-error@+1 {{A shape is not distributable with the layout}}
  %1 = xegpu.dpas_mx %a, %b {layout_a = #layout_a, layout_b = #layout_b, layout_cd = #layout_cd} : vector<8x16xf8E5M2>, vector<16x16xf8E5M2> -> vector<8x16xf32>
  return
}

// -----
#layout_a_scale_invalid = #xegpu.layout<sg_layout = [1, 1], sg_data = [5, 3]>
func.func @dpas_mx_scale_a_layout_not_distributable(%a : vector<8x16xf8E5M2>, %b: vector<16x16xf8E5M2>, %acc: vector<8x16xf32>, %scale_a_val: vector<8x2xf8E8M0FNU>) {
  // expected-error@+1 {{ScaleA shape is not distributable with the layout}}
  %1 = xegpu.dpas_mx %a, %b, %acc scale_a = %scale_a_val {layout_a_scale = #layout_a_scale_invalid} : vector<8x16xf8E5M2>, vector<16x16xf8E5M2>, vector<8x16xf32>, vector<8x2xf8E8M0FNU> -> vector<8x16xf32>
  return
}

// -----
#layout_b_scale_invalid = #xegpu.layout<sg_layout = [1, 1], sg_data = [3, 11]>
func.func @dpas_mx_scale_b_layout_not_distributable(%a : vector<8x16xf8E5M2>, %b: vector<16x16xf8E5M2>, %acc: vector<8x16xf32>, %scale_a_val: vector<8x2xf8E8M0FNU>, %scale_b_val: vector<2x16xf8E8M0FNU>) {
  // expected-error@+1 {{ScaleB shape is not distributable with the layout}}
  %1 = xegpu.dpas_mx %a, %b, %acc scale_a = %scale_a_val scale_b = %scale_b_val {layout_b_scale = #layout_b_scale_invalid} : vector<8x16xf8E5M2>, vector<16x16xf8E5M2>, vector<8x16xf32>, vector<8x2xf8E8M0FNU>, vector<2x16xf8E8M0FNU> -> vector<8x16xf32>
  return
}
