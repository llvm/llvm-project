// RUN: mlir-opt --xegpu-wg-to-sg-distribute -split-input-file %s | FileCheck %s

gpu.module @elementwise_ops {
   // CHECK-LABEL: elemwise_ops
   gpu.func @elemwise_ops(%a: memref<24x32xf32>, %b: memref<24x32xf32>, %c: memref<24x32xi32>, %d: memref<24x32xi32>) {
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

        // Floating point ops
        // CHECK: arith.addf {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.subf {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.exp {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.sqrt {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.absf {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.cos {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.cosh {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.acos {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.acosh {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.sin {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.sinh {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.asin {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.asinh {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.tan {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.tanh {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.atan {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.atan2 {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.atanh {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.erf {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.log {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.log2 {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.floor {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.ceil {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.powf {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.rsqrt {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.negf {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.mulf {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.divf {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.maximumf {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.minimumf {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.addi {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.subi {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.muli {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.shli {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.shrsi {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.shrui {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.divsi {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.divui {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.remsi {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.remui {{.*}}, {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        %addf = arith.addf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %subf = arith.subf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %exp = math.exp %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %sqrt = math.sqrt %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %absf = math.absf %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cos = math.cos %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cosh = math.cosh %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %acos = math.acos %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %acosh = math.acosh %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %sin = math.sin %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %sinh = math.sinh %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %asin = math.asin %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %asinh = math.asinh %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %tan = math.tan %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %tanh = math.tanh %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %atan = math.atan %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %atan2 = math.atan2 %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %atanh = math.atanh %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %erf = math.erf %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %log = math.log %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %log2 = math.log2 %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %floor = math.floor %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %ceil = math.ceil %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %powf = math.powf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %rsqrt = math.rsqrt %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %negf = arith.negf %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %mulf = arith.mulf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %divf = arith.divf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %maximumf = arith.maximumf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %minimumf = arith.minimumf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>

        // Integer ops
        %addi = arith.addi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %subi = arith.subi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %muli = arith.muli %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %shli = arith.shli %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %shrsi = arith.shrsi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %shrui = arith.shrui %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %divsi = arith.divsi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %divui = arith.divui %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %remsi = arith.remsi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %remui = arith.remui %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>

        gpu.return
    }

    // 1 to N decomposition of elementwise operations
    // CHECK-LABEL: elemwise_ops_rr_assignment
    gpu.func @elemwise_ops_rr_assignment(%a: memref<24x32xf32>, %b: memref<24x32xf32>, %c: memref<24x32xi32>, %d: memref<24x32xi32>) {
        %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
          -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
        %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<24x32xf32>
          -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
        %tdesc_c = xegpu.create_nd_tdesc %c[0, 0] : memref<24x32xi32>
          -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
        %tdesc_d = xegpu.create_nd_tdesc %d[0, 0] : memref<24x32xi32>
          -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>

        %load_a = xegpu.load_nd %tdesc_a
          : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
          -> vector<24x32xf32>
        %load_b = xegpu.load_nd %tdesc_b
          : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
          -> vector<24x32xf32>
        %load_c = xegpu.load_nd %tdesc_c
          : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
          -> vector<24x32xi32>
        %load_d = xegpu.load_nd %tdesc_d
          : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
          -> vector<24x32xi32>

        // Floating point ops
        // CHECK-COUNT-12: arith.addf {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.addf
        // CHECK-COUNT-12: arith.subf {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.subf
        // CHECK-COUNT-12: math.exp {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.exp
        // CHECK-COUNT-12: math.sqrt {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.sqrt
        // CHECK-COUNT-12: math.absf {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.absf
        // CHECK-COUNT-12: math.cos {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.cos
        // CHECK-COUNT-12: math.cosh {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.cosh
        // CHECK-COUNT-12: math.acos {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.acos
        // CHECK-COUNT-12: math.acosh {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.acosh
        // CHECK-COUNT-12: math.sin {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.sin
        // CHECK-COUNT-12: math.sinh {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.sinh
        // CHECK-COUNT-12: math.asin {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.asin
        // CHECK-COUNT-12: math.asinh {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.asinh
        // CHECK-COUNT-12: math.tan {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.tan
        // CHECK-COUNT-12: math.tanh {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.tanh
        // CHECK-COUNT-12: math.atan {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.atan
        // CHECK-COUNT-12: math.atan2 {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.atan2
        // CHECK-COUNT-12: math.atanh {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.atanh
        // CHECK-COUNT-12: math.erf {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.erf
        // CHECK-COUNT-12: math.log {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.log
        // CHECK-COUNT-12: math.log2 {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.log2
        // CHECK-COUNT-12: math.floor {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.floor
        // CHECK-COUNT-12: math.ceil {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.ceil
        // CHECK-COUNT-12: math.powf {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.powf
        // CHECK-COUNT-12: math.rsqrt {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: math.rsqrt
        // CHECK-COUNT-12: arith.negf {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.negf
        // CHECK-COUNT-12: arith.mulf {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.mulf
        // CHECK-COUNT-12: arith.divf {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.divf
        // CHECK-COUNT-12: arith.maximumf {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.maximumf
        // CHECK-COUNT-12: arith.minimumf {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.minimumf
        // CHECK-COUNT-12: arith.addi {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.addi
        // CHECK-COUNT-12: arith.subi {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.subi
        // CHECK-COUNT-12: arith.muli {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.muli
        // CHECK-COUNT-12: arith.shli {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.shli
        // CHECK-COUNT-12: arith.shrsi {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.shrsi
        // CHECK-COUNT-12: arith.shrui {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.shrui
        // CHECK-COUNT-12: arith.divsi {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.divsi
        // CHECK-COUNT-12: arith.divui {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.divui
        // CHECK-COUNT-12: arith.remsi {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.remsi
        // CHECK-COUNT-12: arith.remui {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.remui
        %addf = arith.addf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %subf = arith.subf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %exp = math.exp %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %sqrt = math.sqrt %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %absf = math.absf %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cos = math.cos %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cosh = math.cosh %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %acos = math.acos %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %acosh = math.acosh %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %sin = math.sin %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %sinh = math.sinh %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %asin = math.asin %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %asinh = math.asinh %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %tan = math.tan %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %tanh = math.tanh %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %atan = math.atan %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %atan2 = math.atan2 %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %atanh = math.atanh %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %erf = math.erf %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %log = math.log %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %log2 = math.log2 %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %floor = math.floor %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %ceil = math.ceil %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %powf = math.powf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %rsqrt = math.rsqrt %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %negf = arith.negf %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %mulf = arith.mulf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %divf = arith.divf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %maximumf = arith.maximumf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %minimumf = arith.minimumf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>

        // Integer ops
        %addi = arith.addi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %subi = arith.subi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %muli = arith.muli %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %shli = arith.shli %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %shrsi = arith.shrsi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %shrui = arith.shrui %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %divsi = arith.divsi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %divui = arith.divui %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %remsi = arith.remsi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %remui = arith.remui %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>

        gpu.return
    }

    // CHECK-LABEL: type_conversion_ops
    gpu.func @type_conversion_ops(
        %a: memref<24x32xf32>, %b: memref<24x32xi32>, %c: memref<24x32xi32>, %d: memref<24x32xi32>) {
        %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
          -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
        %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<24x32xi32>
          -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
        %tdesc_c = xegpu.create_nd_tdesc %c[0, 0] : memref<24x32xi32>
          -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
        %tdesc_d = xegpu.create_nd_tdesc %d[0, 0] : memref<24x32xi32>
          -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>

        %load_a = xegpu.load_nd %tdesc_a
          : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
          -> vector<24x32xf32>
        %load_b = xegpu.load_nd %tdesc_b
          : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
          -> vector<24x32xi32>
        %load_c = xegpu.load_nd %tdesc_c
          : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
          -> vector<24x32xi32>
        %load_d = xegpu.load_nd %tdesc_d
          : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
          -> vector<24x32xi32>

        // CHECK: arith.truncf {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32> to vector<12x8xf16>
        // CHECK: arith.trunci {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32> to vector<12x8xi16>
        // CHECK: arith.extf {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf16> to vector<12x8xf32>
        // CHECK: arith.extsi {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi16> to vector<12x8xi32>
        // CHECK: arith.extui {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi16> to vector<12x8xi32>
        // CHECK: arith.sitofp {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32> to vector<12x8xf32>
        // CHECK: arith.uitofp {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32> to vector<12x8xf32>
        // CHECK: arith.fptosi {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32> to vector<12x8xi32>
        // CHECK: arith.fptoui {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32> to vector<12x8xi32>
        // CHECK: arith.index_castui {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32> to vector<12x8xindex>
        // CHECK: arith.index_cast {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xindex> to vector<12x8xi32>
        // CHECK: arith.bitcast {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32> to vector<12x8xf32>
        // TruncFOp: f32 -> f16
        %truncf = arith.truncf %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32> to vector<24x32xf16>
        // TruncIOp: i32 -> i16
        %trunci = arith.trunci %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32> to vector<24x32xi16>
        // ExtFOp: f16 -> f32
        %truncf16 = arith.truncf %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32> to vector<24x32xf16>
        %extf = arith.extf %truncf16
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf16> to vector<24x32xf32>
        // ExtSIOp: i16 -> i32
        %extsi = arith.extsi %trunci
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi16> to vector<24x32xi32>
        // ExtUIOp: i16 -> i32 (unsigned)
        %extui = arith.extui %trunci
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi16> to vector<24x32xi32>
        // SIToFPOp: i32 -> f32
        %sitofp = arith.sitofp %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32> to vector<24x32xf32>
        // UIToFPOp: i32 -> f32 (unsigned)
        %uitofp = arith.uitofp %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32> to vector<24x32xf32>
        // FPToSIOp: f32 -> i32
        %fptosi = arith.fptosi %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32> to vector<24x32xi32>
        // FPToUIOp: f32 -> i32 (unsigned)
        %fptoui = arith.fptoui %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32> to vector<24x32xi32>
        // IndexCastUIOp: i32 -> index
        %indexcastui = arith.index_castui %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32> to vector<24x32xindex>
        // IndexCastOp: index -> i32
        %indexcast = arith.index_cast %indexcastui
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xindex> to vector<24x32xi32>
        // BitcastOp: i32 -> f32
        %bitcast = arith.bitcast %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32> to vector<24x32xf32>
        gpu.return
    }


    // CHECK-LABEL: gpu.func @type_conversion_ops_rr_assignment
    gpu.func @type_conversion_ops_rr_assignment(
        %a: memref<24x32xf32>, %b: memref<24x32xi32>, %c: memref<24x32xi32>, %d: memref<24x32xi32>) {
        %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
          -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
        %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<24x32xi32>
          -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
        %tdesc_c = xegpu.create_nd_tdesc %c[0, 0] : memref<24x32xi32>
          -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
        %tdesc_d = xegpu.create_nd_tdesc %d[0, 0] : memref<24x32xi32>
          -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>

        %load_a = xegpu.load_nd %tdesc_a
          : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
          -> vector<24x32xf32>
        %load_b = xegpu.load_nd %tdesc_b
          : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
          -> vector<24x32xi32>
        %load_c = xegpu.load_nd %tdesc_c
          : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
          -> vector<24x32xi32>
        %load_d = xegpu.load_nd %tdesc_d
          : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
          -> vector<24x32xi32>

        // CHECK-COUNT-12: arith.truncf {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32> to vector<2x2xf16>
        // CHECK-NOT: arith.truncf
        // CHECK-COUNT-12: arith.trunci {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32> to vector<2x2xi16>
        // CHECK-NOT: arith.trunci
        // CHECK-COUNT-12: arith.extf {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf16> to vector<2x2xf32>
        // CHECK-NOT: arith.extf
        // CHECK-COUNT-12: arith.extsi {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi16> to vector<2x2xi32>
        // CHECK-NOT: arith.extsi
        // CHECK-COUNT-12: arith.extui {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi16> to vector<2x2xi32>
        // CHECK-NOT: arith.extui
        // CHECK-COUNT-12: arith.sitofp {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32> to vector<2x2xf32>
        // CHECK-NOT: arith.sitofp
        // CHECK-COUNT-12: arith.uitofp {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32> to vector<2x2xf32>
        // CHECK-NOT: arith.uitofp
        // CHECK-COUNT-12: arith.fptosi {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32> to vector<2x2xi32>
        // CHECK-NOT: arith.fptosi
        // CHECK-COUNT-12: arith.fptoui {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32> to vector<2x2xi32>
        // CHECK-NOT: arith.fptoui
        // CHECK-COUNT-12: arith.index_castui {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32> to vector<2x2xindex>
        // CHECK-NOT: arith.index_castui
        // CHECK-COUNT-12: arith.index_cast {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xindex> to vector<2x2xi32>
        // CHECK-NOT: arith.index_cast
        // CHECK-COUNT-12: arith.bitcast {{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32> to vector<2x2xf32>
        // CHECK-NOT: arith.bitcast
        // TruncFOp: f32 -> f16
        %truncf = arith.truncf %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32> to vector<24x32xf16>
        // TruncIOp: i32 -> i16
        %trunci = arith.trunci %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32> to vector<24x32xi16>
        // ExtFOp: f16 -> f32
        %truncf16 = arith.truncf %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32> to vector<24x32xf16>
        %extf = arith.extf %truncf16
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf16> to vector<24x32xf32>
        // ExtSIOp: i16 -> i32
        %extsi = arith.extsi %trunci
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi16> to vector<24x32xi32>
        // ExtUIOp: i16 -> i32 (unsigned)
        %extui = arith.extui %trunci
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi16> to vector<24x32xi32>
        // SIToFPOp: i32 -> f32
        %sitofp = arith.sitofp %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32> to vector<24x32xf32>
        // UIToFPOp: i32 -> f32 (unsigned)
        %uitofp = arith.uitofp %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32> to vector<24x32xf32>
        // FPToSIOp: f32 -> i32
        %fptosi = arith.fptosi %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32> to vector<24x32xi32>
        // FPToUIOp: f32 -> i32 (unsigned)
        %fptoui = arith.fptoui %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32> to vector<24x32xi32>
        // IndexCastUIOp: i32 -> index
        %indexcastui = arith.index_castui %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32> to vector<24x32xindex>
        // IndexCastOp: index -> i32
        %indexcast = arith.index_cast %indexcastui
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xindex> to vector<24x32xi32>
        // BitcastOp: i32 -> f32
        %bitcast = arith.bitcast %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32> to vector<24x32xf32>
        gpu.return
    }

    // CHECK-LABEL: gpu.func @comparison_ops
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

        // CHECK: arith.cmpi eq, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.cmpi ne, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.cmpi slt, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.cmpi sle, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.cmpi sgt, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.cmpi sge, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.cmpi ult, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.cmpi ule, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.cmpi ugt, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.cmpi uge, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.cmpf oeq, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf ogt, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf oge, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf olt, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf ole, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf one, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf ord, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf ueq, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf ugt, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf uge, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf ult, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf ule, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf une, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf uno, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // Integer comparisons
        %cmpi_eq = arith.cmpi eq, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_ne = arith.cmpi ne, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_slt = arith.cmpi slt, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_sle = arith.cmpi sle, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_sgt = arith.cmpi sgt, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_sge = arith.cmpi sge, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_ult = arith.cmpi ult, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_ule = arith.cmpi ule, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_ugt = arith.cmpi ugt, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_uge = arith.cmpi uge, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>

        // Floating point comparisons
        %cmpf_oeq = arith.cmpf oeq, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ogt = arith.cmpf ogt, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_oge = arith.cmpf oge, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_olt = arith.cmpf olt, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ole = arith.cmpf ole, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_one = arith.cmpf one, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ord = arith.cmpf ord, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ueq = arith.cmpf ueq, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ugt = arith.cmpf ugt, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_uge = arith.cmpf uge, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ult = arith.cmpf ult, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ule = arith.cmpf ule, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_une = arith.cmpf une, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_uno = arith.cmpf uno, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        gpu.return
    }

    // CHECK-LABEL: gpu.func @comparison_ops_rr_assignment
    gpu.func @comparison_ops_rr_assignment(%a: memref<24x32xf32>, %b: memref<24x32xf32>, %c: memref<24x32xi32>, %d: memref<24x32xi32>) {
        %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
          -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
        %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<24x32xf32>
          -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
        %tdesc_c = xegpu.create_nd_tdesc %c[0, 0] : memref<24x32xi32>
          -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
        %tdesc_d = xegpu.create_nd_tdesc %d[0, 0] : memref<24x32xi32>
          -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>

        %load_a = xegpu.load_nd %tdesc_a
          : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
          -> vector<24x32xf32>
        %load_b = xegpu.load_nd %tdesc_b
          : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
          -> vector<24x32xf32>
        %load_c = xegpu.load_nd %tdesc_c
          : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
          -> vector<24x32xi32>
        %load_d = xegpu.load_nd %tdesc_d
          : !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
          -> vector<24x32xi32>

        // CHECK-COUNT-12: arith.cmpi eq, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.cmpi eq
        // CHECK-COUNT-12: arith.cmpi ne, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.cmpi ne
        // CHECK-COUNT-12: arith.cmpi slt, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.cmpi slt
        // CHECK-COUNT-12: arith.cmpi sle, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.cmpi sle
        // CHECK-COUNT-12: arith.cmpi sgt, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.cmpi sgt
        // CHECK-COUNT-12: arith.cmpi sge, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.cmpi sge
        // CHECK-COUNT-12: arith.cmpi ult, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.cmpi ult
        // CHECK-COUNT-12: arith.cmpi ule, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.cmpi ule
        // CHECK-COUNT-12: arith.cmpi ugt, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.cmpi ugt
        // CHECK-COUNT-12: arith.cmpi uge, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xi32>
        // CHECK-NOT: arith.cmpi uge
        // Floating point comparisons
        // CHECK-COUNT-12: arith.cmpf oeq, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf oeq
        // CHECK-COUNT-12: arith.cmpf ogt, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf ogt
        // CHECK-COUNT-12: arith.cmpf oge, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf oge
        // CHECK-COUNT-12: arith.cmpf olt, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf olt
        // CHECK-COUNT-12: arith.cmpf ole, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf ole
        // CHECK-COUNT-12: arith.cmpf one, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf one
        // CHECK-COUNT-12: arith.cmpf ord, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf ord
        // CHECK-COUNT-12: arith.cmpf ueq, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf ueq
        // CHECK-COUNT-12: arith.cmpf ugt, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf ugt
        // CHECK-COUNT-12: arith.cmpf uge, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf uge
        // CHECK-COUNT-12: arith.cmpf ult, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf ult
        // CHECK-COUNT-12: arith.cmpf ule, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf ule
        // CHECK-COUNT-12: arith.cmpf une, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf une
        // CHECK-COUNT-12: arith.cmpf uno, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} : vector<2x2xf32>
        // CHECK-NOT: arith.cmpf uno

        // Integer comparisons
        %cmpi_eq = arith.cmpi eq, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_ne = arith.cmpi ne, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_slt = arith.cmpi slt, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_sle = arith.cmpi sle, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_sgt = arith.cmpi sgt, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_sge = arith.cmpi sge, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_ult = arith.cmpi ult, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_ule = arith.cmpi ule, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_ugt = arith.cmpi ugt, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cmpi_uge = arith.cmpi uge, %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xi32>

        // Floating point comparisons
        %cmpf_oeq = arith.cmpf oeq, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ogt = arith.cmpf ogt, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_oge = arith.cmpf oge, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_olt = arith.cmpf olt, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ole = arith.cmpf ole, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_one = arith.cmpf one, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ord = arith.cmpf ord, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ueq = arith.cmpf ueq, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ugt = arith.cmpf ugt, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_uge = arith.cmpf uge, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ult = arith.cmpf ult, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_ule = arith.cmpf ule, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_une = arith.cmpf une, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf_uno = arith.cmpf uno, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
          : vector<24x32xf32>

        gpu.return
    }

   // CHECK-LABEL: gpu.func @elementwise_ops
   gpu.func @elementwise_ops(
        %a: memref<24x32xf32>, %b: memref<24x32xf32>, %c: memref<24x32xi32>, %d: memref<24x32xi32>, %e: memref<24x32xi1>) {
        %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
          -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
        %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<24x32xf32>
          -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
        %tdesc_c = xegpu.create_nd_tdesc %c[0, 0] : memref<24x32xi32>
          -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
        %tdesc_d = xegpu.create_nd_tdesc %d[0, 0] : memref<24x32xi32>
          -> !xegpu.tensor_desc<24x32xi32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
        %tdesc_e = xegpu.create_nd_tdesc %e[0, 0] : memref<24x32xi1>
          -> !xegpu.tensor_desc<24x32xi1, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>

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
        %load_e = xegpu.load_nd %tdesc_e
          : !xegpu.tensor_desc<24x32xi1, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
          -> vector<24x32xi1>

        // CHECK: arith.andi {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.ori {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.xori {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.ceildivsi {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.ceildivui {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.floordivsi {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.maxnumf {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.maxsi {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.maxui {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.minnumf {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.minsi {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.minui {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: arith.remf {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.cmpf ult, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: arith.select {{.*}}, {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi1>, vector<12x8xf32>
        // CHECK: math.absi {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: math.cbrt {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.copysign {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.ctpop {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: math.erfc {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.exp2 {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.expm1 {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.fpowi {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>, vector<12x8xi32>
        // CHECK: math.ipowi {{.*}}, {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xi32>
        // CHECK: math.log10 {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.log1p {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.round {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.roundeven {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // CHECK: math.trunc {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>} : vector<12x8xf32>
        // arith ops
        %andi = arith.andi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %ori = arith.ori %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %xori = arith.xori %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %ceildivsi = arith.ceildivsi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %ceildivui = arith.ceildivui %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %floordivsi = arith.floordivsi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %maxnumf = arith.maxnumf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %maxsi = arith.maxsi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %maxui = arith.maxui %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %minnumf = arith.minnumf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %minsi = arith.minsi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %minui = arith.minui %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %remf = arith.remf %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %cmpf = arith.cmpf ult, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %select = arith.select %cmpf, %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi1>, vector<24x32xf32>
        
        // math ops
        %absi = math.absi %load_c
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %cbrt = math.cbrt %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %copysign = math.copysign %load_a, %load_b
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %ctpop = math.ctpop %load_c
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %erfc = math.erfc %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %exp2 = math.exp2 %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %expm1 = math.expm1 %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %fpowi = math.fpowi %load_a, %load_c
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>, vector<24x32xi32>
        %ipowi = math.ipowi %load_c, %load_d
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xi32>
        %log10 = math.log10 %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %log1p = math.log1p %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %round = math.round %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %roundeven = math.roundeven %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        %trunc = math.trunc %load_a
          {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>}
          : vector<24x32xf32>
        gpu.return
    }
}