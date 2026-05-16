// RUN: mlir-opt -test-xegpu-recover-temporary-layouts -split-input-file %s | FileCheck %s

// -----
// Test scf.for: Recovery should propagate layout from the store_nd consumer
// of the loop result back to the scf.for result, scf.yield operands, and
// the arith.constant init value. Tensor desc types start without layouts
// and only anchor ops (load_nd, store_nd, dpas) carry layout attributes.

gpu.module @test_for {
// CHECK-LABEL: gpu.func @for_basic
gpu.func @for_basic(%arg0: memref<8x128xf16>, %arg1: memref<128x16xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: -> !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %0 = xegpu.create_nd_tdesc %arg0 : memref<8x128xf16>
      -> !xegpu.tensor_desc<8x16xf16>
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
  %1 = xegpu.create_nd_tdesc %arg1 : memref<128x16xf16>
      -> !xegpu.tensor_desc<16x16xf16>
  // Recovery propagates layout from dpas (via store_nd) back to arith.constant.
  // CHECK: arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
  // CHECK-SAME: dense<0.000000e+00> : vector<8x16xf32>
  %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  // CHECK: scf.for
  %2 = scf.for %arg3 = %c0 to %c128 step %c16
      iter_args(%arg6 = %cst) -> (vector<8x16xf32>) {
    %4 = xegpu.load_nd %0[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
        : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    %5 = xegpu.load_nd %1[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
        : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    %6 = xegpu.dpas %4, %5, %arg6
        {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
         layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
         layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
        : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    // Recovery propagates layout to scf.yield vector operand.
    // CHECK: scf.yield {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    scf.yield %6
        : vector<8x16xf32>
  // Recovery sets layout_result_0 on the scf.for for the vector result.
  // CHECK: layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
  }
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %3 = xegpu.create_nd_tdesc %arg2 : memref<8x16xf32>
      -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %2, %3[%c0, %c0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.return
}
}

// -----
// Test scf.while: Recovery should propagate layout from the store_nd consumer
// of the while result back through scf.condition (which branches to parent).
// The scf.yield in the "do" region branches to the "before" region (not
// parent), so propagateRegionResultsToYieldOperands skips it.

gpu.module @test_while {
// CHECK-LABEL: gpu.func @while_basic
gpu.func @while_basic(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
  %0 = xegpu.create_nd_tdesc %arg0 : memref<1024xf32>
      -> !xegpu.tensor_desc<256xf32>
  %1 = xegpu.load_nd %0[0] {layout = #xegpu.layout<sg_layout = [16], sg_data = [16]>}
      : !xegpu.tensor_desc<256xf32> -> vector<256xf32>
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
  %2 = xegpu.create_nd_tdesc %arg1 : memref<1024xf32>
      -> !xegpu.tensor_desc<256xf32>

  // CHECK: scf.while
  %3:2 = scf.while (%arg2 = %1, %arg3 = %c0_i32)
      : (vector<256xf32>, i32) -> (vector<256xf32>, i32) {
    %4 = arith.cmpi slt, %arg3, %c10_i32 : i32
    // Recovery propagates layout to scf.condition vector operand.
    // CHECK: scf.condition
    // CHECK-SAME: {layout_operand_1 = #xegpu.layout<sg_layout = [16], sg_data = [16]>}
    scf.condition(%4) %arg2, %arg3 : vector<256xf32>, i32
  } do {
  ^bb0(%arg2: vector<256xf32>, %arg3: i32):
    xegpu.store_nd %arg2, %2[0] {layout = #xegpu.layout<sg_layout = [16], sg_data = [16]>}
        : vector<256xf32>, !xegpu.tensor_desc<256xf32>
    %4 = arith.addi %arg3, %c1_i32 : i32
    %6 = xegpu.load_nd %0[256] {layout = #xegpu.layout<sg_layout = [16], sg_data = [16]>}
        : !xegpu.tensor_desc<256xf32> -> vector<256xf32>
    // Recovery propagates layout to scf.yield in the "do" region via
    // sibling region propagation (from "before" region arg back to "do" yield).
    // CHECK: scf.yield {layout_operand_0 = #xegpu.layout<sg_layout = [16], sg_data = [16]>}
    scf.yield %6, %4 : vector<256xf32>, i32
  // Recovery sets layout_result_0 on the scf.while for the vector result.
  // CHECK: } attributes {layout_operand_0 = #xegpu.layout<sg_layout = [16], sg_data = [16]>,
  // CHECK-SAME: layout_result_0 = #xegpu.layout<sg_layout = [16], sg_data = [16]>}
  }
  xegpu.store_nd %3#0, %2[0] {layout = #xegpu.layout<sg_layout = [16], sg_data = [16]>}
      : vector<256xf32>, !xegpu.tensor_desc<256xf32>
  gpu.return
}
}

// -----
// Test scf.if: Recovery should propagate layout from the dpas consumer of the
// if result back to the scf.if result and both yield operands.

gpu.module @test_if {
// CHECK-LABEL: gpu.func @if_basic
gpu.func @if_basic(
    %arg0: !xegpu.tensor_desc<8x16xf16>,
    %arg1: !xegpu.tensor_desc<16x16xf16>,
    %arg2: i1,
    %arg3: !xegpu.tensor_desc<8x16xf32>) {
  %0 = xegpu.load_nd %arg0[0, 0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  // CHECK: scf.if
  %1 = scf.if %arg2 -> (vector<16x16xf16>) {
    %3 = xegpu.load_nd %arg1[0, 0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
        : !xegpu.tensor_desc<16x16xf16>
        -> vector<16x16xf16>
    // Recovery propagates layout to scf.yield operand in "then" region.
    // CHECK: scf.yield {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
    scf.yield %3 : vector<16x16xf16>
  } else {
    %3 = xegpu.load_nd %arg1[0, 0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
        : !xegpu.tensor_desc<16x16xf16>
        -> vector<16x16xf16>
    // Recovery propagates layout to scf.yield operand in "else" region.
    // CHECK: scf.yield {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
    scf.yield %3 : vector<16x16xf16>
  // Recovery sets layout_result_0 on the scf.if for the vector result.
  // CHECK: } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
  }
  %2 = xegpu.dpas %0, %1
      {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
       layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %2, %arg3[0, 0] {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.return
}
}
