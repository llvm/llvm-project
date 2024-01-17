// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s


// CHECK-LABEL: func @test_create_tdesc_vc({{.*}}) {
func.func @test_create_tdesc_vc(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %{{arg[0-9]}}, %{{arg[0-9]}} {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc}: ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_2({{.*}}) {
func.func @test_create_tdesc_vc_2(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %{{arg[0-9]}}, %{{arg[0-9]}} {mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.tdesc_attr<memory_scope = slm, #xegpu.scattered>>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc} : ui64, vector<16 x index>
                            -> !xegpu.tensor_desc<16xf32, #xegpu.tdesc_attr<memory_scope = slm, #xegpu.scattered>>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_3({{.*}}) {
func.func @test_create_tdesc_vc_3(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %{{arg[0-9]}}, %{{arg[0-9]}} {chunk_size_per_lane = 8 : i64, mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 8}
                                          : ui64, vector<16 x index> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>
  return
}

// CHECK-LABEL: func @test_create_tdesc_vc_4({{.*}}) {
func.func @test_create_tdesc_vc_4(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %{{arg[0-9]}}, %{{arg[0-9]}} {chunk_size_per_lane = 2 : i64, mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.tdesc_attr<memory_scope = slm, #xegpu.scattered>>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 2}
                        : ui64, vector<16 x index> -> !xegpu.tensor_desc<16x2xf32, #xegpu.tdesc_attr<memory_scope = slm, #xegpu.scattered>>
  return
}


// CHECK-LABEL: func @test_create_tdesc_vc_5({{.*}}) {
func.func @test_create_tdesc_vc_5(%src: memref<?xf32>, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc %{{arg[0-9]}}, %{{arg[0-9]}} {chunk_size_per_lane = 2 : i64, mode = #xegpu<mode_kind vc>}
  // CHECK-SAME: memref<?xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.tdesc_attr<memory_scope = slm, #xegpu.scattered>>
  %1 = xegpu.create_tdesc %src, %offsets {mode = vc, chunk_size_per_lane = 2}
              : memref<?xf32>, vector<16 x index> -> !xegpu.tensor_desc<16x2xf32, #xegpu.tdesc_attr<memory_scope = slm, #xegpu.scattered>>
  return
}
