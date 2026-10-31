// RUN: mlir-opt -xevm-attach-target='chip=cri' -test-xegpu-propagate-layouts="layout-kind=inst" -split-input-file -verify-diagnostics %s

// A shape cast that splits one source dim into several result dims can only
// take a consumer layout whose lanes own data the collapsed source dim can
// describe, i.e. one contiguous run per lane.

// Here the lanes split dim 2 (2 of 4) as well as dim 3 (4 of 16). Stretching the
// inner dims gives lane_data = [1, 1, 2, 4], so lane 0 owns dim 2 in [0, 2) and
// dim 3 in [0, 4): the result offsets {0, 1, 2, 3} and {16, 17, 18, 19} - two
// runs, 16 apart - whereas the collapsed lane_data = 8 claims the contiguous
// source columns [0, 8). No lane_data fixes this, so the consumer layout is
// rejected and no layout reaches the source of the cast.
gpu.module @test {
  func.func @shape_cast_split_lanes_split_two_dims(%arg0: memref<16x128xbf16>) {
    // expected-warning@+1 {{op has users but no layout assigned for its result}}
    %0 = xegpu.create_nd_tdesc %arg0 : memref<16x128xbf16> -> !xegpu.tensor_desc<16x128xbf16>
    // expected-warning@+1 {{op has users but no layout assigned for its result}}
    %1 = xegpu.load_nd %0[0, 0] : !xegpu.tensor_desc<16x128xbf16> -> vector<16x128xbf16>
    // expected-warning@+1 {{Failed to infer source layout for shape_cast; the result layout required by its consumers cannot be collapsed onto the source shape.}}
    %2 = vector.shape_cast %1 : vector<16x128xbf16> to vector<16x2x4x16xbf16>
    %3 = xegpu.convert_layout %2
       <{target_layout = #xegpu.layout<inst_data = [1, 2, 2, 4], lane_layout = [1, 2, 2, 4], lane_data = [1, 1, 1, 1]>}>
       : vector<16x2x4x16xbf16>
    return
  }
}

// -----
// Same requirement one dim out: the consumer asks each lane for 2 elements of
// dim 1, which are 64 elements apart in the result, so the lane's data is not a
// single run of the collapsed source dim either.
gpu.module @test {
  func.func @shape_cast_split_outer_dim_lane_data(%arg0: memref<16x128xbf16>) {
    // expected-warning@+1 {{op has users but no layout assigned for its result}}
    %0 = xegpu.create_nd_tdesc %arg0 : memref<16x128xbf16> -> !xegpu.tensor_desc<16x128xbf16>
    // expected-warning@+1 {{op has users but no layout assigned for its result}}
    %1 = xegpu.load_nd %0[0, 0] : !xegpu.tensor_desc<16x128xbf16> -> vector<16x128xbf16>
    // expected-warning@+1 {{Failed to infer source layout for shape_cast; the result layout required by its consumers cannot be collapsed onto the source shape.}}
    %2 = vector.shape_cast %1 : vector<16x128xbf16> to vector<16x2x4x16xbf16>
    %3 = xegpu.convert_layout %2
       <{target_layout = #xegpu.layout<inst_data = [1, 2, 2, 8], lane_layout = [1, 1, 2, 8], lane_data = [1, 2, 1, 1]>}>
       : vector<16x2x4x16xbf16>
    return
  }
}

// -----
// The inst tile has to collapse too: inst_data = [1, 2, 2, 8] takes only 2 of
// the 4 elements of dim 2, so the tile is two 16-element chunks of the source
// row rather than the 32 contiguous elements the collapsed inst_data claims.
gpu.module @test {
  func.func @shape_cast_split_partial_inst_data(%arg0: memref<16x128xbf16>) {
    // expected-warning@+1 {{op has users but no layout assigned for its result}}
    %0 = xegpu.create_nd_tdesc %arg0 : memref<16x128xbf16> -> !xegpu.tensor_desc<16x128xbf16>
    // expected-warning@+1 {{op has users but no layout assigned for its result}}
    %1 = xegpu.load_nd %0[0, 0] : !xegpu.tensor_desc<16x128xbf16> -> vector<16x128xbf16>
    // expected-warning@+1 {{Failed to infer source layout for shape_cast; the result layout required by its consumers cannot be collapsed onto the source shape.}}
    %2 = vector.shape_cast %1 : vector<16x128xbf16> to vector<16x2x4x16xbf16>
    %3 = xegpu.convert_layout %2
       <{target_layout = #xegpu.layout<inst_data = [1, 2, 2, 8], lane_layout = [1, 1, 2, 8], lane_data = [1, 1, 1, 1]>}>
       : vector<16x2x4x16xbf16>
    return
  }
}

// -----
// An inst_data-only anchor is completed from the offsets' contiguity. Every
// offset here is the same, so a lane can take only one element, and 16 lanes
// cannot cover inst_data = [1, 32] with one access each.
gpu.module @test {
func.func @scatter_inst_data_needs_contiguous_offsets(%src: memref<512xf32>) {
  %mask = arith.constant dense<1> : vector<16x32xi1>
  %offset = arith.constant dense<12> : vector<16x32xindex>
  %v = arith.constant dense<1.0> : vector<16x32xf32>
  // expected-error@+1 {{Failed to identify lane layouts for the specified inst_data.}}
  xegpu.store %v, %src[%offset], %mask <{layout = #xegpu.layout<inst_data = [1, 32]>}>
      : vector<16x32xf32>, memref<512xf32>, vector<16x32xindex>, vector<16x32xi1>
  return
}
}

// -----
// Runs of 3 (the block width) allow one element per lane, so inst_data =
// [1, 32] has no completion that keeps every access inside one block.
gpu.module @test {
func.func @store_matrix_inst_data_needs_contiguous_runs(
    %dst: !xegpu.mem_desc<8x96xf32, #xegpu.mem_layout<block = [8, 3]>>) {
  %v = arith.constant dense<1.0> : vector<8x96xf32>
  // expected-error@+1 {{Failed to identify lane layouts for the specified inst_data.}}
  xegpu.store_matrix %v, %dst[0, 0] <{layout = #xegpu.layout<inst_data = [1, 32]>}>
      : vector<8x96xf32>, !xegpu.mem_desc<8x96xf32, #xegpu.mem_layout<block = [8, 3]>>
  return
}
}
