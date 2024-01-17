// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s
// CHECK-LABEL: func @test_update_nd_offset_vc_0({{.*}}) {
func.func @test_update_nd_offset_vc_0(%src: memref<24x32xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]}}, %{{c[0-9]}}]
  // CHECK-SAME: {mode = #xegpu<mode_kind vc>} : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] {mode = vc}
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  // CHECK: xegpu.load_nd %{{[0-9]}}
  // CHECK-SAME: {l1_hint = #xegpu<cache_kind cached>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  %2 = xegpu.load_nd %1 {mode = vc, l1_hint = cached, l2_hint = uncached}
              : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

  // CHECK: xegpu.update_nd_offset %{{[0-9]}}, [%{{c[0-9]}}, %{{c[0-9]}}] {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  %3 = xegpu.update_nd_offset %1, [%c0, %c1] {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>

  return
}
