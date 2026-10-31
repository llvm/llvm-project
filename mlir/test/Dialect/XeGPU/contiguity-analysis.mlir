// RUN: mlir-opt -split-input-file \
// RUN:   -test-xegpu-coalesce-gather-scatter="analyze-only=true" %s | FileCheck %s

// Contiguity analysis: stamps the `contiguity` attribute on gather/scatter ops
// whose offsets are contiguous (runs of >= 2) along the innermost dimension.
// The stamped value is the inner-dim contiguity, rounded down to a divisor of
// the inner extent. The analysis is target-independent and mask-independent;
// the lane_data split and any access-pattern gating live in the consumer.

// -----
// 1-D vector.step -> stride-1, fully contiguous over the 32-element inner dim.
// CHECK-LABEL: func.func @load_step_offsets(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguity = 32 : i64}>
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
func.func @load_step_offsets(%ptr: i64) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// Dense stride-1 constant offsets -> contiguity 32.
// CHECK-LABEL: func.func @load_dense_ap_offsets(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguity = 32 : i64}>
func.func @load_dense_ap_offsets(%ptr: i64) -> vector<32xi32> {
  %offsets = arith.constant dense<[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
  ]> : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xi32>
  return %v : vector<32xi32>
}

// -----
// Stride-4 offsets: not contiguous, no attribute stamped.
// CHECK-LABEL: func.func @load_stride4_no_attr(
// CHECK: xegpu.load
// CHECK-NOT: contiguity
// CHECK-SAME: : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
func.func @load_stride4_no_attr(%ptr: i64) -> vector<32xf32> {
  %c4 = arith.constant 4 : index
  %step = vector.step : vector<32xindex>
  %splat = vector.broadcast %c4 : index to vector<32xindex>
  %offsets = arith.muli %step, %splat : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// All-equal offsets: inner dim is constant, not contiguous -> no attribute.
// CHECK-LABEL: func.func @load_broadcast_offsets_no_attr(
// CHECK: xegpu.load
// CHECK-NOT: contiguity
func.func @load_broadcast_offsets_no_attr(%ptr: i64) -> vector<32xf32> {
  %offsets = arith.constant dense<0> : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// The analysis is mask-independent: a partial mask still gets the attribute
// (the mask check is a consumer concern).
// CHECK-LABEL: func.func @load_partial_mask(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguity = 32 : i64}>
func.func @load_partial_mask(%ptr: i64, %mask: vector<32xi1>) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// Store with vector.step offsets -> contiguity 32 on the store.
// CHECK-LABEL: func.func @store_step_offsets(
// CHECK: xegpu.store
// CHECK-SAME: <{contiguity = 32 : i64}>
func.func @store_step_offsets(%ptr: i64, %v: vector<32xf32>) {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  xegpu.store %v, %ptr[%offsets], %mask
      : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
  return
}

// -----
// 2-D leading-unit dim: contiguity measured on the inner dim -> 32.
// CHECK-LABEL: func.func @load_2d_leading_unit(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguity = 32 : i64}>
func.func @load_2d_leading_unit(%ptr: i64) -> vector<1x32xf32> {
  %step = vector.step : vector<32xindex>
  %offsets = vector.shape_cast %step : vector<32xindex> to vector<1x32xindex>
  %mask = arith.constant dense<true> : vector<1x32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<1x32xindex>, vector<1x32xi1> -> vector<1x32xf32>
  return %v : vector<1x32xf32>
}

// -----
// True 2-D dense AP: each row stride-1 over 16 -> contiguity 16.
// CHECK-LABEL: func.func @load_2d_dense_ap(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguity = 16 : i64}>
func.func @load_2d_dense_ap(%ptr: i64) -> vector<2x16xf32> {
  %offsets = arith.constant dense<[
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
  ]> : vector<2x16xindex>
  %mask = arith.constant dense<true> : vector<2x16xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<2x16xindex>, vector<2x16xi1> -> vector<2x16xf32>
  return %v : vector<2x16xf32>
}

// -----
// 2-D dense values whose inner row is not a stride-1 AP: no attribute.
// CHECK-LABEL: func.func @load_2d_non_ap(
// CHECK: xegpu.load
// CHECK-NOT: contiguity
func.func @load_2d_non_ap(%ptr: i64) -> vector<2x16xf32> {
  %offsets = arith.constant dense<[
    [0, 1, 2, 3, 4, 5, 6, 7, 100, 9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
  ]> : vector<2x16xindex>
  %mask = arith.constant dense<true> : vector<2x16xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<2x16xindex>, vector<2x16xi1> -> vector<2x16xf32>
  return %v : vector<2x16xf32>
}

// -----
// `divui` by a constant equal to the inner stride recovers stride-1
// contiguity: (step * 2) / 2 = [0,1,..,31] -> contiguity 32. The dividend is
// derived from vector.step (not a constant) so the divui is genuinely
// exercised even if the solver later folds all-constant arith ops.
// CHECK-LABEL: func.func @load_divui_recovers(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguity = 32 : i64}>
func.func @load_divui_recovers(%ptr: i64) -> vector<32xf32> {
  %step = vector.step : vector<32xindex>
  %c2 = arith.constant dense<2> : vector<32xindex>
  %even = arith.muli %step, %c2 : vector<32xindex>
  %offsets = arith.divui %even, %c2 : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// `divui` by a constant that does not divide the inner stride: not recovered.
// CHECK-LABEL: func.func @load_divui_non_divisor(
// CHECK: xegpu.load
// CHECK-NOT: contiguity
func.func @load_divui_non_divisor(%ptr: i64) -> vector<16xf32> {
  %step = vector.step : vector<16xindex>
  %c2 = arith.constant dense<2> : vector<16xindex>
  %even = arith.muli %step, %c2 : vector<16xindex>
  %c3 = arith.constant dense<3> : vector<16xindex>
  %offsets = arith.divui %even, %c3 : vector<16xindex>
  %mask = arith.constant dense<true> : vector<16xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<16xindex>, vector<16xi1> -> vector<16xf32>
  return %v : vector<16xf32>
}

// -----
// `remui` collapsing each element to a constant residue: inner dim is uniform,
// not contiguous -> no attribute.
// CHECK-LABEL: func.func @load_remui_inner_uniform(
// CHECK: xegpu.load
// CHECK-NOT: contiguity
func.func @load_remui_inner_uniform(%ptr: i64) -> vector<16xf32> {
  %step = vector.step : vector<16xindex>
  %c2 = arith.constant dense<2> : vector<16xindex>
  %even = arith.muli %step, %c2 : vector<16xindex>
  %offsets = arith.remui %even, %c2 : vector<16xindex>
  %mask = arith.constant dense<true> : vector<16xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<16xindex>, vector<16xi1> -> vector<16xf32>
  return %v : vector<16xf32>
}

// -----
// `shli` then `shrui` cancel: (step << 1) >> 1 -> stride 1 -> contiguity 32.
// CHECK-LABEL: func.func @load_shli_then_shrui(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguity = 32 : i64}>
func.func @load_shli_then_shrui(%ptr: i64) -> vector<32xf32> {
  %step = vector.step : vector<32xindex>
  %k = arith.constant dense<1> : vector<32xindex>
  %doubled = arith.shli %step, %k : vector<32xindex>
  %offsets = arith.shrui %doubled, %k : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// `shli` alone scales the stride to 2: not contiguous -> no attribute.
// CHECK-LABEL: func.func @load_shli_unchanged(
// CHECK: xegpu.load
// CHECK-NOT: contiguity
func.func @load_shli_unchanged(%ptr: i64) -> vector<32xf32> {
  %step = vector.step : vector<32xindex>
  %k = arith.constant dense<1> : vector<32xindex>
  %offsets = arith.shli %step, %k : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// `select` between two AP arms with matching inner-dim properties preserves
// contiguity 32.
// CHECK-LABEL: func.func @load_select_two_aps(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguity = 32 : i64}>
func.func @load_select_two_aps(%ptr: i64, %cond: i1) -> vector<32xf32> {
  %step = vector.step : vector<32xindex>
  %a = arith.constant dense<0>  : vector<32xindex>
  %b = arith.constant dense<64> : vector<32xindex>
  %baseA = arith.addi %step, %a : vector<32xindex>
  %baseB = arith.addi %step, %b : vector<32xindex>
  %offsets = arith.select %cond, %baseA, %baseB : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// Scalar arithmetic keeps its divisibility: `%sg * 2` is even, so after the
// broadcast `(base + 2 * step) / 2` is an exact division with stride 1.
// CHECK-LABEL: func.func @load_div_even_scalar_base(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguity = 32 : i64}>
func.func @load_div_even_scalar_base(%ptr: i64, %sg: index) -> vector<32xf32> {
  %c2 = arith.constant 2 : index
  %base = arith.muli %sg, %c2 : index
  %bbase = vector.broadcast %base : index to vector<32xindex>
  %step = vector.step : vector<32xindex>
  %two = vector.broadcast %c2 : index to vector<32xindex>
  %step2 = arith.muli %step, %two : vector<32xindex>
  %sum = arith.addi %bbase, %step2 : vector<32xindex>
  %offsets = arith.divui %sum, %two : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// A pre-existing `contiguity` takes precedence: the analysis leaves it alone.
// CHECK-LABEL: func.func @user_attr_preserved(
// CHECK: xegpu.load
// CHECK-SAME: <{contiguity = 2 : i64}>
func.func @user_attr_preserved(%ptr: i64) -> vector<32xf32> {
  %offsets = vector.step : vector<32xindex>
  %mask = arith.constant dense<true> : vector<32xi1>
  %v = xegpu.load %ptr[%offsets], %mask <{contiguity = 2 : i64}>
      : i64, vector<32xindex>, vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----
// load_matrix / store_matrix take one scalar offset per dim, so their
// contiguity comes from the mem_desc memory layout. Row-major without a block:
// each row of the tile is one run, so the contiguity is the inner extent.
// CHECK-LABEL: func.func @load_matrix_row_major(
// CHECK: xegpu.load_matrix
// CHECK-SAME: <{contiguity = 32 : i64}>
func.func @load_matrix_row_major(%m: !xegpu.mem_desc<32x64xf32>) -> vector<8x32xf32> {
  %v = xegpu.load_matrix %m[0, 32] : !xegpu.mem_desc<32x64xf32> -> vector<8x32xf32>
  return %v : vector<8x32xf32>
}

// -----
// Without a block the offsets do not matter: a dynamic origin is still one run
// per row.
// CHECK-LABEL: func.func @store_matrix_row_major_dynamic_offsets(
// CHECK: xegpu.store_matrix
// CHECK-SAME: <{contiguity = 16 : i64}>
func.func @store_matrix_row_major_dynamic_offsets(%m: !xegpu.mem_desc<32x64xf32>,
    %v: vector<8x16xf32>, %y: index, %x: index) {
  xegpu.store_matrix %v, %m[%y, %x] : vector<8x16xf32>, !xegpu.mem_desc<32x64xf32>, index, index
  return
}

// -----
// 1-D mem_desc.
// CHECK-LABEL: func.func @load_matrix_1d(
// CHECK: xegpu.load_matrix
// CHECK-SAME: <{contiguity = 16 : i64}>
func.func @load_matrix_1d(%m: !xegpu.mem_desc<64xf16>, %x: index) -> vector<16xf16> {
  %v = xegpu.load_matrix %m[%x] : !xegpu.mem_desc<64xf16>, index -> vector<16xf16>
  return %v : vector<16xf16>
}

// -----
// Column-major: neighbours along the innermost data dim are 32 apart in SLM, so
// no attribute.
// CHECK-LABEL: func.func @load_matrix_col_major_no_attr(
// CHECK: xegpu.load_matrix
// CHECK-NOT: contiguity
// CHECK-SAME: -> vector<8x32xf32>
func.func @load_matrix_col_major_no_attr(
    %m: !xegpu.mem_desc<32x64xf32, #xegpu.mem_layout<stride = [1, 32]>>) -> vector<8x32xf32> {
  %v = xegpu.load_matrix %m[0, 0]
      : !xegpu.mem_desc<32x64xf32, #xegpu.mem_layout<stride = [1, 32]>> -> vector<8x32xf32>
  return %v : vector<8x32xf32>
}

// -----
// Blocked: a run is contiguous only inside one block, so a 32-wide tile over
// 16-wide blocks gets 16.
// CHECK-LABEL: func.func @load_matrix_blocked(
// CHECK: xegpu.load_matrix
// CHECK-SAME: <{contiguity = 16 : i64}>
func.func @load_matrix_blocked(
    %m: !xegpu.mem_desc<32x64xf32, #xegpu.mem_layout<block = [8, 16]>>) -> vector<8x32xf32> {
  %v = xegpu.load_matrix %m[0, 0]
      : !xegpu.mem_desc<32x64xf32, #xegpu.mem_layout<block = [8, 16]>> -> vector<8x32xf32>
  return %v : vector<8x32xf32>
}

// -----
// Blocked, but the tile [8, 24) along the inner dim fits in one 32-wide block:
// one run.
// CHECK-LABEL: func.func @load_matrix_blocked_fits_one_block(
// CHECK: xegpu.load_matrix
// CHECK-SAME: <{contiguity = 16 : i64}>
func.func @load_matrix_blocked_fits_one_block(
    %m: !xegpu.mem_desc<32x64xf32, #xegpu.mem_layout<block = [8, 32]>>) -> vector<8x16xf32> {
  %v = xegpu.load_matrix %m[0, 8]
      : !xegpu.mem_desc<32x64xf32, #xegpu.mem_layout<block = [8, 32]>> -> vector<8x16xf32>
  return %v : vector<8x16xf32>
}

// -----
// Blocked, with an origin of 4 inside a 16-wide block: runs of 4 never cross a
// block edge, longer runs would.
// CHECK-LABEL: func.func @load_matrix_blocked_misaligned(
// CHECK: xegpu.load_matrix
// CHECK-SAME: <{contiguity = 4 : i64}>
func.func @load_matrix_blocked_misaligned(
    %m: !xegpu.mem_desc<32x64xf32, #xegpu.mem_layout<block = [8, 16]>>) -> vector<8x32xf32> {
  %v = xegpu.load_matrix %m[0, 4]
      : !xegpu.mem_desc<32x64xf32, #xegpu.mem_layout<block = [8, 16]>> -> vector<8x32xf32>
  return %v : vector<8x32xf32>
}

// -----
// Blocked, with a dynamic origin the analysis proves to be a multiple of 16
// (`%sg * 32 + 16`): runs of 16 stay in one block.
// CHECK-LABEL: func.func @store_matrix_blocked_aligned_dynamic(
// CHECK: xegpu.store_matrix
// CHECK-SAME: <{contiguity = 16 : i64}>
func.func @store_matrix_blocked_aligned_dynamic(
    %m: !xegpu.mem_desc<32x256xf32, #xegpu.mem_layout<block = [8, 16]>>,
    %v: vector<8x32xf32>, %sg: index) {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %base = arith.muli %sg, %c32 : index
  %x = arith.addi %base, %c16 : index
  xegpu.store_matrix %v, %m[0, %x]
      : vector<8x32xf32>, !xegpu.mem_desc<32x256xf32, #xegpu.mem_layout<block = [8, 16]>>, index
  return
}

// -----
// Blocked, with a dynamic origin of unknown alignment: a run could cross a block
// edge anywhere, so no attribute.
// CHECK-LABEL: func.func @store_matrix_blocked_unknown_offset_no_attr(
// CHECK: xegpu.store_matrix
// CHECK-NOT: contiguity
// CHECK-SAME: vector<8x32xf32>
func.func @store_matrix_blocked_unknown_offset_no_attr(
    %m: !xegpu.mem_desc<32x256xf32, #xegpu.mem_layout<block = [8, 16]>>,
    %v: vector<8x32xf32>, %x: index) {
  xegpu.store_matrix %v, %m[0, %x]
      : vector<8x32xf32>, !xegpu.mem_desc<32x256xf32, #xegpu.mem_layout<block = [8, 16]>>, index
  return
}

// -----
// A scalar payload has no inner run.
// CHECK-LABEL: func.func @load_matrix_scalar_no_attr(
// CHECK: xegpu.load_matrix
// CHECK-NOT: contiguity
// CHECK-SAME: -> f32
func.func @load_matrix_scalar_no_attr(%m: !xegpu.mem_desc<32x64xf32>) -> f32 {
  %v = xegpu.load_matrix %m[0, 0] : !xegpu.mem_desc<32x64xf32> -> f32
  return %v : f32
}

// -----
// A truncating cast does not keep the value: 258 cast to i8 is 2. With 6-wide
// blocks, an origin of 2 only allows runs of 2.
// CHECK-LABEL: func.func @load_matrix_truncating_index_cast(
// CHECK: xegpu.load_matrix
// CHECK-SAME: <{contiguity = 2 : i64}>
func.func @load_matrix_truncating_index_cast(
    %m: !xegpu.mem_desc<8x192xf32, #xegpu.mem_layout<block = [8, 6]>>) -> vector<8x96xf32> {
  %c258 = arith.constant 258 : index
  %narrow = arith.index_cast %c258 : index to i8
  %x = arith.index_cast %narrow : i8 to index
  %v = xegpu.load_matrix %m[0, %x]
      : !xegpu.mem_desc<8x192xf32, #xegpu.mem_layout<block = [8, 6]>>, index -> vector<8x96xf32>
  return %v : vector<8x96xf32>
}

// -----
// An unsigned cast does not keep a negative value: -6 as i8 becomes 250. Only
// the low bits survive, so the origin is known to be even, and runs of 2 are
// all that 6-wide blocks allow.
// CHECK-LABEL: func.func @load_matrix_unsigned_index_cast(
// CHECK: xegpu.load_matrix
// CHECK-SAME: <{contiguity = 2 : i64}>
func.func @load_matrix_unsigned_index_cast(
    %m: !xegpu.mem_desc<8x384xf32, #xegpu.mem_layout<block = [8, 6]>>) -> vector<8x96xf32> {
  %cm6 = arith.constant -6 : i8
  %x = arith.index_castui %cm6 : i8 to index
  %v = xegpu.load_matrix %m[0, %x]
      : !xegpu.mem_desc<8x384xf32, #xegpu.mem_layout<block = [8, 6]>>, index -> vector<8x96xf32>
  return %v : vector<8x96xf32>
}

// -----
// A sign-extending cast keeps every value, so `%sg * 16` keeps its
// divisibility through the cast to index.
// CHECK-LABEL: func.func @store_matrix_sign_extending_index_cast(
// CHECK: xegpu.store_matrix
// CHECK-SAME: <{contiguity = 16 : i64}>
func.func @store_matrix_sign_extending_index_cast(
    %m: !xegpu.mem_desc<8x256xf32, #xegpu.mem_layout<block = [8, 16]>>,
    %v: vector<8x32xf32>, %sg: i32) {
  %c16 = arith.constant 16 : i32
  %base = arith.muli %sg, %c16 : i32
  %x = arith.index_cast %base : i32 to index
  xegpu.store_matrix %v, %m[0, %x]
      : vector<8x32xf32>, !xegpu.mem_desc<8x256xf32, #xegpu.mem_layout<block = [8, 16]>>, index
  return
}

// -----
// A pre-existing `contiguity` on a matrix op takes precedence.
// CHECK-LABEL: func.func @load_matrix_user_attr_preserved(
// CHECK: xegpu.load_matrix
// CHECK-SAME: <{contiguity = 2 : i64}>
func.func @load_matrix_user_attr_preserved(%m: !xegpu.mem_desc<32x64xf32>) -> vector<8x32xf32> {
  %v = xegpu.load_matrix %m[0, 0] <{contiguity = 2 : i64}>
      : !xegpu.mem_desc<32x64xf32> -> vector<8x32xf32>
  return %v : vector<8x32xf32>
}
