// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s


// CHECK-LABEL: func @test_load_gather_vc({{.*}}) {
func.func @test_load_gather_vc(%src: ui64, %offsets : vector<16xindex>) {
  %0 = arith.constant dense<1>: vector<16xi1>
  // CHECK: xegpu.create_tdesc %{{arg[0-9]}}, %{{arg[0-9]}} {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc}: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>

  // CHECK: xegpu.load %{{[0-9]}}, %{{.*}} {l1_hint = #xegpu<cache_kind cached>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>} 
  // CHECK-SAME: !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  %2 = xegpu.load %1, %0 {mode = vc, l1_hint = cached, l2_hint = uncached}
                : !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  return
}

// CHECK-LABEL: func @test_load_gather_vc_2({{.*}}) {
func.func @test_load_gather_vc_2(%src: ui64, %offsets : vector<16xindex>) {
  %0 = arith.constant dense<1>: vector<16x8xi1>
  // CHECK: xegpu.create_tdesc %{{arg[0-9]}}, %{{arg[0-9]}} {chunk_size_per_lane = 8 : i64, mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 8}
                : ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>

  // CHECK: xegpu.load %{{[0-9]}}, %{{.*}} {l1_hint = #xegpu<cache_kind cached>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>, transpose = array<i64: 1, 0>} 
  // CHECK-SAME: !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>, vector<16x8xi1> -> vector<8x16xf32>
  %2 = xegpu.load %1, %0 {mode = vc, transpose = [1, 0], l1_hint = cached, l2_hint = uncached}
               : !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>, vector<16x8xi1> -> vector<8x16xf32>
  return
}

// CHECK-LABEL: func @test_load_gather_vc_3({{.*}}) {
func.func @test_load_gather_vc_3(%src: ui64, %offsets : vector<16xindex>) {
  %0 = arith.constant dense<1>: vector<16xi1>
  // CHECK: xegpu.create_tdesc %{{arg[0-9]}}, %{{arg[0-9]}} {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered> 
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 1}
                : ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>

  // CHECK: xegpu.load %{{[0-9]}}, %{{.*}} {l1_hint = #xegpu<cache_kind cached>, l2_hint = #xegpu<cache_kind uncached>, mode = #xegpu<mode_kind vc>} 
  // CHECK-SAME: !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  %2 = xegpu.load %1, %0 {mode = vc, l1_hint = cached, l2_hint = uncached}
                : !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  return
}
