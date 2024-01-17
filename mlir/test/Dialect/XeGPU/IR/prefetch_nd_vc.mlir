// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s
// CHECK-LABEL: func @test_prefetch_nd_tdesc_vc_0({{.*}}) {
func.func @test_prefetch_nd_tdesc_vc_0(%src: memref<24x32xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]}}, %{{c[0-9]}}] {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] {mode = vc} : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  // CHECK: xegpu.prefetch_nd %{{[0-9]}} {mode = #xegpu<mode_kind vc>} : !xegpu.tensor_desc<8x16xf32>
  xegpu.prefetch_nd %1 {mode = vc} : !xegpu.tensor_desc<8x16xf32>

  return
}

// CHECK-LABEL: func @test_prefetch_nd_tdesc_vc_1({{.*}}) {
func.func @test_prefetch_nd_tdesc_vc_1(%src: memref<24x32xf16>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{arg[0-9]}}, %{{arg[0-9]}}] 
  // CHECK-SAME: {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y] {mode = vc} : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>

  // CHECK: xegpu.prefetch_nd %{{[0-9]}} 
  // CHECK-SAME: {l1_hint = #xegpu<cache_kind cached>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>
  xegpu.prefetch_nd %1 {mode = vc, l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<8x16xf16>
  return
}


// CHECK-LABEL: func @test_prefetch_nd_tdesc_vc_i8({{.*}}) {
func.func @test_prefetch_nd_tdesc_vc_i8(%src: memref<24x32xi8>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]}}, %{{c[0-9]}}] {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xi8> -> !xegpu.tensor_desc<8x16xi8>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] {mode = vc} : memref<24x32xi8> -> !xegpu.tensor_desc<8x16xi8>

  // CHECK: xegpu.prefetch_nd %{{[0-9]}} {mode = #xegpu<mode_kind vc>} : !xegpu.tensor_desc<8x16xi8>
  xegpu.prefetch_nd %1 {mode = vc} : !xegpu.tensor_desc<8x16xi8>

  return
}

// CHECK-LABEL: func @test_prefetch_nd_tdesc_vc_bf16({{.*}}) {
func.func @test_prefetch_nd_tdesc_vc_bf16(%src: memref<24x32xbf16>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{arg[0-9]}}, %{{arg[0-9]}}]
  // CHECK-SAME: {mode = #xegpu<mode_kind vc>} : memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y] {mode = vc}
      : memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16>
  // CHECK: xegpu.prefetch_nd %{{[0-9]}}
  // CHECK-SAME: {l1_hint = #xegpu<cache_kind uncached>, l2_hint = #xegpu<cache_kind cached>, mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xbf16>
  xegpu.prefetch_nd %1 {mode = vc, l1_hint = uncached, l2_hint = cached}: !xegpu.tensor_desc<8x16xbf16>
  return
}
