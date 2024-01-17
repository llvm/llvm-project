// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @test_create_nd_tdesc_vc({{.*}}) {
func.func @test_create_nd_tdesc_vc(%src: memref<24x32xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc {{.*}} {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] {mode = vc}
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  // CHECK: xegpu.create_nd_tdesc {{.*}} {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %2 = xegpu.create_nd_tdesc %src[2, 4] {mode = vc}
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  return
}

// CHECK-LABEL: func @test_create_tdesc_vc({{.*}}) {
func.func @test_create_tdesc_vc(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc {{.*}} {chunk_size_per_lane = 2 : i64, mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.tdesc_attr<memory_scope = slm, #xegpu.scattered>>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 2}
                          : ui64, vector<16 x index> -> !xegpu.tensor_desc<16x2xf32, #xegpu.tdesc_attr<memory_scope = slm, #xegpu.scattered>>
  return
}

// CHECK-LABEL: func @test_load_nd_vc({{.*}}) {
func.func @test_load_nd_vc(%src: memref<24x32xf16>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc {{.*}} {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y] {mode = vc}
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>

  // CHECK: xegpu.load_nd {{.*}} {l1_hint = #xegpu<cache_kind cached>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>, vnni_axis = 0 : i64}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  %2 = xegpu.load_nd %1 {mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = uncached} : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  return
}

// CHECK-LABEL: func @test_store_nd_vc({{.*}}) {
func.func @test_store_nd_vc(%src: memref<24x32xf16>, %dst: memref<24x32xf16>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc {{.*}} {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] {mode = vc}
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>

  // CHECK: xegpu.create_nd_tdesc {{.*}} {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %2 = xegpu.create_nd_tdesc %dst[%c0, %c1] {mode = vc}
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>

  // CHECK: xegpu.load_nd {{.*}} {l1_hint = #xegpu<cache_kind cached>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1 {mode=vc, l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>

  // CHECK: xegpu.store_nd {{%[0-9], %[0-9]}} {l1_hint = #xegpu<cache_kind write_back>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>} 
  // CHECK-SAME: vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  xegpu.store_nd %3, %2 {mode = vc, l1_hint = write_back, l2_hint = uncached}: vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  return
}

// CHECK-LABEL: func @test_dpas_vc({{.*}}) {
func.func @test_dpas_vc(%a : vector<8x8x2xf16>, %b: vector<8x16x2xf16>) {
  // CHECK: xegpu.dpas {{.*}} {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  %1 = xegpu.dpas %a, %b {mode = vc}: vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  return
}

// CHECK-LABEL: func @test_update_nd_offset_vc({{.*}}) {
func.func @test_update_nd_offset_vc(%src: memref<24x32xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc {{.*}} {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1] {mode = vc}
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  // CHECK: xegpu.load_nd {{%[0-9]}} {l1_hint = #xegpu<cache_kind cached>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  %2 = xegpu.load_nd %1 {mode = vc, l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

  // CHECK: xegpu.update_nd_offset {{%[0-9]}}, [{{%c[0-9], %c[0-9]}}] {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  %3 = xegpu.update_nd_offset %1, [%c0, %c1] {mode = vc}: !xegpu.tensor_desc<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>

  return
}

// CHECK-LABEL: func @test_prefetch_nd_vc({{.*}}) {
func.func @test_prefetch_nd_vc(%src: memref<24x32xf16>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc {{.*}} {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y] {mode = vc} : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK: xegpu.prefetch_nd {{%[0-9]}} {l1_hint = #xegpu<cache_kind cached>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16>
  xegpu.prefetch_nd %1 {mode = vc, l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<8x16xf16>
  return
}
