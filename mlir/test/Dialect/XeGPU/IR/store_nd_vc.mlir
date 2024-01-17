// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @test_store_nd_vc_bf16({{.*}}) {
func.func @test_store_nd_vc_bf16(%src: memref<24x32xbf16>, %dst: memref<24x32xbf16>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]}}, %{{c[0-9]}}] {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] {mode = vc} : memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16>

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]}}, %{{c[0-9]}}] {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16>
  %2 = xegpu.create_nd_tdesc %dst[%c0, %c1] {mode = vc} : memref<24x32xbf16> -> !xegpu.tensor_desc<8x16xbf16>

  // CHECK: xegpu.load_nd %{{[0-9]}}
  // CHECK-SAME: {l1_hint = #xegpu<cache_kind cached>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xbf16> -> vector<8x16xbf16>
  %3 = xegpu.load_nd %1 {mode = vc, l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<8x16xbf16> -> vector<8x16xbf16>

  // CHECK: xegpu.store_nd %{{[0-9]}}, %{{[0-9]}}
  // CHECK-SAME: {l1_hint = #xegpu<cache_kind write_back>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>} 
  // CHECK-SAME: vector<8x16xbf16>, !xegpu.tensor_desc<8x16xbf16>
  xegpu.store_nd %3, %2 {mode = vc, l1_hint = write_back, l2_hint = uncached}: vector<8x16xbf16>, !xegpu.tensor_desc<8x16xbf16>
  return
}

// CHECK-LABEL: func @test_store_nd_vc_f64({{.*}}) {
func.func @test_store_nd_vc_f64(%src: memref<24x32xf64>, %dst: memref<24x32xf64>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]}}, %{{c[0-9]}}] {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xf64> -> !xegpu.tensor_desc<8x16xf64>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] {mode = vc} : memref<24x32xf64> -> !xegpu.tensor_desc<8x16xf64>

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]}}, %{{c[0-9]}}] {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xf64> -> !xegpu.tensor_desc<8x16xf64>
  %2 = xegpu.create_nd_tdesc %dst[%c0, %c1] {mode = vc}
      : memref<24x32xf64> -> !xegpu.tensor_desc<8x16xf64>

  // CHECK: xegpu.load_nd %{{[0-9]}}
  // CHECK-SAME: {l1_hint = #xegpu<cache_kind cached>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>} 
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf64> -> vector<8x16xf64>
  %3 = xegpu.load_nd %1 {mode = vc, l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<8x16xf64> -> vector<8x16xf64>

  // CHECK: xegpu.store_nd %{{[0-9]}}, %{{[0-9]}}
  // CHECK-SAME: {l1_hint = #xegpu<cache_kind write_back>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>} 
  // CHECK-SAME: vector<8x16xf64>, !xegpu.tensor_desc<8x16xf64>
  xegpu.store_nd %3, %2 {mode = vc, l1_hint = write_back, l2_hint = uncached}: vector<8x16xf64>, !xegpu.tensor_desc<8x16xf64>
  return
}

// CHECK-LABEL: func @test_store_nd_vc_i8({{.*}}) {
func.func @test_store_nd_vc_i8(%src: memref<24x32xi8>, %dst: memref<24x32xi8>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]}}, %{{c[0-9]}}] {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xi8> -> !xegpu.tensor_desc<8x16xi8>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] {mode = vc}
      : memref<24x32xi8> -> !xegpu.tensor_desc<8x16xi8>

  // CHECK: xegpu.create_nd_tdesc %{{arg[0-9]}}[%{{c[0-9]}}, %{{c[0-9]}}] {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xi8> -> !xegpu.tensor_desc<8x16xi8>
  %2 = xegpu.create_nd_tdesc %dst[%c0, %c1] {mode = vc}
      : memref<24x32xi8> -> !xegpu.tensor_desc<8x16xi8>

  // CHECK: xegpu.load_nd %{{[0-9]}}
  // CHECK-SAME: {l1_hint = #xegpu<cache_kind cached>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>} 
  // CHECK-SAME: !xegpu.tensor_desc<8x16xi8> -> vector<8x16xi8>
  %3 = xegpu.load_nd %1 {mode = vc, l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<8x16xi8> -> vector<8x16xi8>

  // CHECK: xegpu.store_nd %{{[0-9]}}, %{{[0-9]}}
  // CHECK-SAME: {l1_hint = #xegpu<cache_kind write_back>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>} 
  // CHECK-SAME: vector<8x16xi8>, !xegpu.tensor_desc<8x16xi8>
  xegpu.store_nd %3, %2 {mode = vc, l1_hint = write_back, l2_hint = uncached}: vector<8x16xi8>, !xegpu.tensor_desc<8x16xi8>
  return
}
