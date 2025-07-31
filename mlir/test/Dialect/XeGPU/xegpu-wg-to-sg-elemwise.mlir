// RUN: mlir-opt --xegpu-wg-to-sg-distribute -split-input-file %s | FileCheck %s

gpu.module @test_elementwise_ops {

  // CHECK-LABEL: unary_ops_sg_layout_only
  gpu.func @unary_ops_sg_layout_only(%a: memref<24x32xf32>) {
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8]>>
    %load_a = xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8]>>
      -> vector<24x32xf32>
    // CHECK: math.exp {{.*}} : vector<12x8xf32>
    %exp = math.exp %load_a
      {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8]>}
      : vector<24x32xf32>
    // CHECK: arith.negf {{.*}} : vector<12x8xf32>
    %negf = arith.negf %load_a
      {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8]>}
      : vector<24x32xf32>
    gpu.return
  }

  // CHECK-LABEL: unary_ops
  gpu.func @unary_ops(%a: memref<24x32xf32>) {
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %load_a = xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    // CHECK: math.exp {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
    %exp = math.exp %load_a
      {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
      : vector<24x32xf32>
    // CHECK: arith.negf {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
    %negf = arith.negf %load_a
      {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
      : vector<24x32xf32>
    gpu.return
  }

  // CHECK-LABEL: binary_ops
  gpu.func @binary_ops(%a: memref<24x32xf32>, %b: memref<24x32xf32>) {
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %load_a = xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    %load_b = xegpu.load_nd %tdesc_b
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    // CHECK: arith.addf {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}
    // CHECK-SAME: : vector<12x8xf32>
    %addf = arith.addf %load_a, %load_b
      {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
      : vector<24x32xf32>
    // CHECK: math.powf {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}
    // CHECK-SAME: : vector<12x8xf32>
    %powf = math.powf %load_a, %load_b
      {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
      : vector<24x32xf32>
    gpu.return
  }

  // CHECK-LABEL: ternary_ops
  gpu.func @ternary_ops(%a: memref<24x32xf32>, %b: memref<24x32xf32>, %c: memref<24x32xi1>) {
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %tdesc_c = xegpu.create_nd_tdesc %c[0, 0] : memref<24x32xi1>
      -> !xegpu.tensor_desc<24x32xi1, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %load_a = xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    %load_b = xegpu.load_nd %tdesc_b
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    %load_c = xegpu.load_nd %tdesc_c
      : !xegpu.tensor_desc<24x32xi1, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xi1>
    // CHECK: arith.select {{.*}}, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}
    // CHECK-SAME: : vector<12x8xi1>, vector<12x8xf32>
    %select = arith.select %load_c, %load_a, %load_b
      {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
      : vector<24x32xi1>, vector<24x32xf32>
    // CHECK: math.fma  {{.*}}, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}
    // CHECK-SAME: : vector<12x8xf32>
    %fma = math.fma %load_a, %load_b, %load_a
      {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
      : vector<24x32xf32>
    gpu.return
  }

  // CHECK-LABEL: type_conversion_ops
  gpu.func @type_conversion_ops(%a: memref<24x32xf32>, %b: memref<24x32xi32>) {
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<24x32xi32>
      -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %load_a = xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    %load_b = xegpu.load_nd %tdesc_b
      : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xi32>
    // CHECK: arith.truncf {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}
    // CHECK-SAME: : vector<12x8xf32> to vector<12x8xf16>
    %truncf = arith.truncf %load_a
      {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
      : vector<24x32xf32> to vector<24x32xf16>
    // CHECK: arith.bitcast {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}
    // CHECK-SAME: : vector<12x8xi32> to vector<12x8xf32>
    %bitcast = arith.bitcast %load_b
      {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
      : vector<24x32xi32> to vector<24x32xf32>
    gpu.return
  }

  // CHECK-LABEL: comparison_ops
  gpu.func @comparison_ops(%a: memref<24x32xf32>, %b: memref<24x32xf32>, %c: memref<24x32xi32>, %d: memref<24x32xi32>) {
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %tdesc_c = xegpu.create_nd_tdesc %c[0, 0] : memref<24x32xi32>
      -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %tdesc_d = xegpu.create_nd_tdesc %d[0, 0] : memref<24x32xi32>
      -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %load_a = xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    %load_b = xegpu.load_nd %tdesc_b
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    %load_c = xegpu.load_nd %tdesc_c
      : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xi32>
    %load_d = xegpu.load_nd %tdesc_d
      : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xi32>
    // CHECK: arith.cmpf ult, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}
    // CHECK-SAME: : vector<12x8xf32>
    %cmpf = arith.cmpf ult, %load_a, %load_b
      {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
      : vector<24x32xf32>
    // CHECK: arith.cmpi eq, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}
    // CHECK-SAME: : vector<12x8xi32>
    %cmpi = arith.cmpi eq, %load_c, %load_d
      {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
      : vector<24x32xi32>
    gpu.return
  }

  // 1 to N decomposition of elementwise operations
  // CHECK-LABEL: elementwise_ops_rr_assignment
  gpu.func @elementwise_ops_rr_assignment(%a: memref<24x32xf32>, %b: memref<24x32xf32>) {
     %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    %load_a = xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    %load_b = xegpu.load_nd %tdesc_b
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    // CHECK-COUNT-12: arith.negf {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>}
    // CHECK-SAME-COUNT-12: : vector<2x2xf32>
    // CHECK-NOT: arith.negf
    %negf = arith.negf %load_a
      {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
      : vector<24x32xf32>
    // CHECK-COUNT-12: math.powf {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>}
    // CHECK-SAME-COUNT-12: : vector<2x2xf32>
    // CHECK-NOT: math.powf
    %powf = math.powf %load_a, %load_b
      {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
      : vector<24x32xf32>
    gpu.return
  }
}
