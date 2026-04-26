// RUN: mlir-opt --xegpu-wg-to-sg-distribute -split-input-file %s | FileCheck %s

gpu.module @test_distribution {
  // CHECK-LABEL: create_nd_tdesc
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @create_nd_tdesc(%src: memref<256x128xf32>) {
      // CHECK-COUNT-4: xegpu.create_nd_tdesc %[[ARG_0]] : memref<256x128xf32>
      // CHECK-SAME: -> !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      // CHECK-NOT: xegpu.create_nd_tdesc
      %tdesc = xegpu.create_nd_tdesc %src: memref<256x128xf32>
        -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
      gpu.return
  }

  // CHECK-LABEL: load_nd
  gpu.func @load_nd(%src: memref<256x128xf32>) {
    // CHECK-COUNT-4: xegpu.load_nd {{%.*}}[{{%.*}}, {{%.*}}] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf32>
    // CHECK-NOT: xegpu.load_nd
    %tdesc = xegpu.create_nd_tdesc %src: memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load = xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf32>
    gpu.return
  }

  // CHECK-LABEL: store_nd_with_offset
  gpu.func @store_nd_with_offset(%src: memref<256x128xf32>) {
    // CHECK-COUNT-4: xegpu.store_nd %{{.*}}, {{%.*}}[{{%.*}}, {{%.*}}] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK-NOT: xegpu.store_nd
    %tdesc = xegpu.create_nd_tdesc %src: memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf32>
    xegpu.store_nd %load, %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }

  // CHECK-LABEL: prefetch_nd
  gpu.func @prefetch_nd(%src: memref<256x128xf32>) {
    // CHECK-COUNT-4: xegpu.prefetch_nd {{%.*}}[{{%.*}}, {{%.*}}] : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK-NOT: xegpu.prefetch_nd
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.prefetch_nd %tdesc[0, 0]
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }

  // CHECK-LABEL: dpas
  // CHECK-SAME: (%[[ARG_0:.*]]: memref<256x128xf16>, %[[ARG_1:.*]]: memref<128x256xf16>)
  gpu.func @dpas(%a: memref<256x128xf16>, %b: memref<128x256xf16>) {
    // CHECK-COUNT-4: xegpu.create_nd_tdesc %[[ARG_0]] : memref<256x128xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK-COUNT-4: xegpu.load_nd {{%.*}}[{{%.*}}, {{%.*}}] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf16>
    // CHECK-COUNT-4: xegpu.create_nd_tdesc %[[ARG_1]] : memref<128x256xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
    // CHECK-COUNT-4: xegpu.load_nd {{%.*}}[{{%.*}}, {{%.*}}]  <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
    // CHECK-COUNT-16: xegpu.dpas %{{.*}}, %{{.*}} {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16x16xf16>, vector<16x16xf16> -> vector<16x16xf32>
    // CHECK-NOT: xegpu.dpas
    %tdesc_a = xegpu.create_nd_tdesc %a : memref<256x128xf16>
      -> !xegpu.tensor_desc<256x128xf16, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load_a =  xegpu.load_nd %tdesc_a[0, 0] {layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<256x128xf16, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf16>
    %tdesc_b = xegpu.create_nd_tdesc %b : memref<128x256xf16>
      -> !xegpu.tensor_desc<128x256xf16, #xegpu.layout<sg_layout = [4, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [2, 1]>>
    %load_b =  xegpu.load_nd %tdesc_b[0, 0] {layout = #xegpu.layout<sg_layout = [4, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [2, 1]>}
      : !xegpu.tensor_desc<128x256xf16, #xegpu.layout<sg_layout = [4, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [2, 1]>>
      -> vector<128x256xf16>
    %dpas = xegpu.dpas %load_a, %load_b
       {layout_a = #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>,
        layout_b = #xegpu.layout<sg_layout = [4, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [2, 1]>,
        layout_cd =  #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<256x128xf16>, vector<128x256xf16> -> vector<256x256xf32>
    gpu.return
  }

  // CHECK-LABEL: vector_reduce_dim_1
  gpu.func @vector_reduce_dim_1(%src: memref<256x64xf32>) {
    // CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<16xf32>
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>, dims = [1]>} dense<1.0> : vector<256xf32>
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x64xf32>
      -> !xegpu.tensor_desc<256x64xf32, #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>>
    %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>}
      : !xegpu.tensor_desc<256x64xf32, #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>>
      -> vector<256x64xf32>
    // CHECK-COUNT-2: vector.multi_reduction <add>, {{.*}}, %[[C0:.*]] [1] : vector<16x64xf32> to vector<16xf32>
    // CHECK-NOT: vector.multi_reduction
    // CHECK-COUNT-2: arith.addf {{.*}}, {{.*}} : vector<16xf32>
    // CHECK-NOT: arith.addf
    %reduce = vector.multi_reduction <add>, %load, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>, dims = [1]>} [1]
      : vector<256x64xf32> to vector<256xf32>
    gpu.return
  }

  // CHECK-LABEL: non_splat_constant
  gpu.func @non_splat_constant() {
    // CHECK-DAG: %[[CST:.*]] = arith.constant dense<{{.*}}0{{.*}}, {{.*}}16{{.*}}> : vector<2x1xindex>
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[T1:.*]] = arith.remui %[[SGID]], %[[C8:.*]] : index
    // CHECK-DAG: %[[T2:.*]] = arith.muli %[[T1]], %[[C2:.*]] : index
    // CHECK-DAG: %[[T3:.*]] = arith.remui %[[T2]], %[[C32:.*]] : index
    // CHECK-DAG: %[[T4:.*]] = arith.addi %[[T2]], %[[C16:.*]] : index
    // CHECK-DAG: %[[T5:.*]] = arith.remui %[[T4]], %[[C32_6:.*]] : index
    // CHECK-DAG: %[[T6:.*]] = arith.muli %[[T3]], %[[C16_10:.*]] : index
    // CHECK-DAG: %[[T7:.*]] = arith.addi %[[C0_11:.*]], %[[T6]] : index
    // CHECK-DAG: %[[T8:.*]] = arith.muli %[[C0_4:.*]], %[[C0_9:.*]] : index
    // CHECK-DAG: %[[T9:.*]] = arith.addi %[[T7]], %[[T8]] : index
    // CHECK-DAG: %[[T10:.*]] = vector.broadcast %[[T9]] : index to vector<2x1xindex>
    // CHECK-DAG: %[[T11:.*]] = arith.addi %[[CST]], %[[T10]] : vector<2x1xindex>
    // CHECK-DAG: %[[T12:.*]] = arith.muli %[[T5]], %[[C16_10:.*]] : index
    // CHECK-DAG: %[[T13:.*]] = arith.addi %[[C0_12:.*]], %[[T12]] : index
    // CHECK-DAG: %[[T14:.*]] = arith.muli %[[C0_8:.*]], %[[C0_9:.*]] : index
    // CHECK-DAG: %[[T15:.*]] = arith.addi %[[T13]], %[[T14]] : index
    // CHECK-DAG: %[[T16:.*]] = vector.broadcast %[[T15]] : index to vector<2x1xindex>
    // CHECK-DAG: %[[T17:.*]] = arith.addi %[[CST]], %[[T16]] : vector<2x1xindex>
    %cst_2 = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [8, 1], sg_data = [2, 1]>} dense<[[0], [16], [32], [48], [64], [80], [96], [112], [128], [144], [160], [176], [192], [208], [224], [240], [256], [272], [288], [304], [320], [336], [352], [368], [384], [400], [416], [432], [448], [464], [480], [496]]> : vector<32x1xindex>
    gpu.return
  }

  // CHECK-LABEL: vector_transpose
  gpu.func @vector_transpose(%src: memref<256x128xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32>
        -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 16], lane_layout = [16, 1], lane_data = [1, 1], order =[0, 1]>>
    %load = xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 16], lane_layout = [16, 1], lane_data = [1, 1], order =[0, 1]>}
        : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 16], lane_layout = [16, 1], lane_data = [1, 1], order =[0, 1]>>
        -> vector<256x128xf32>
    // CHECK-COUNT-2: vector.transpose {{.*}}, [1, 0] : vector<32x16xf32> to vector<16x32xf32>
    // CHECK-NOT: vector.transpose
    %trans = vector.transpose %load, [1, 0] {layout_result_0 = #xegpu.layout<sg_layout = [4, 8], sg_data = [16, 32], lane_layout = [1, 16], lane_data = [1, 1]>}
    : vector<256x128xf32> to vector<128x256xf32>
      gpu.return
  }

  // CHECK-LABEL: vector_mask_2D
  gpu.func @vector_mask_2D() {
    // CHECK-COUNT-4: vector.create_mask {{.*}}, {{.*}} : vector<16x16xi1>
    // CHECK-NOT: vector.create_mask
    %constant_mask = vector.constant_mask [16, 16] {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16]>} : vector<256x128xi1>
    gpu.return
  }

  gpu.func @vector_create_mask_2D() {
    // CHECK-COUNT-4: vector.create_mask {{.*}}, {{.*}} : vector<16x16xi1>
    // CHECK-NOT: vector.create_mask
    %cst16 = arith.constant 16 : index
    %constant_mask = vector.create_mask %cst16, %cst16 {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16]>} : vector<256x128xi1>
    gpu.return
  }

  // CHECK-LABEL: distribute_shapecast_expandunitdims_broadcast
  // CHECK: %[[CAST:.*]] = vector.shape_cast %[[REDUCE:.*]] : vector<8xf32> to vector<8x1xf32>
  // CHECK: %[[BCAST:.*]] = vector.broadcast %[[CAST]] : vector<8x1xf32> to vector<8x128xf32>
  gpu.func @distribute_shapecast_expandunitdims_broadcast(%arg0: memref<4096x128xf32>, %arg1: memref<4096x128xf32>) {
    %cst_0 = arith.constant {layout_result_0=#xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>, dims = [1]>} dense<0xFF800000> : vector<256xf32>
    %block_id_x = gpu.block_id x
    %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x128xf32> -> !xegpu.tensor_desc<256x128xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>>
    %1 = xegpu.load_nd %0[%block_id_x, 0] {layout = #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>}  : !xegpu.tensor_desc<256x128xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>> -> vector<256x128xf32>
    %2 = vector.multi_reduction <maximumf>, %1, %cst_0 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [8, 1], inst_data = [8, 1]>, dims = [1]>} [1] : vector<256x128xf32> to vector<256xf32>
    %3 = vector.shape_cast %2 {layout_result_0 =  #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 1], inst_data = [8, 1]>, layout_operand_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [8, 1], inst_data = [8, 1]>, dims = [1]>} : vector<256xf32> to vector<256x1xf32>
    %4 = vector.broadcast %3 {layout_result_0 =  #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>} : vector<256x1xf32>to vector<256x128xf32>
    %9 = xegpu.create_nd_tdesc %arg0 : memref<4096x128xf32> -> !xegpu.tensor_desc<256x128xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>>
    xegpu.store_nd %4, %9[%block_id_x, 0] : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>>
    gpu.return
  }

  // CHECK-LABEL: gpu.func @reduction_cross_sg_rr
  gpu.func @reduction_cross_sg_rr(%arg0: memref<2048xf32, 1>) kernel {
    // CHECK: %[[CST_OFFSETS0:.*]] = arith.constant dense<0> : vector<4x16xindex>
    // CHECK: %[[CST_OFFSETS1:.*]] = arith.constant dense<0> : vector<4x16xindex>
    // CHECK: %[[CST_ACC0:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK: %[[CST_ACC1:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK: %[[CST_MASK0:.*]] = arith.constant dense<true> : vector<4x16xi1>
    // CHECK: %[[CST_MASK1:.*]] = arith.constant dense<true> : vector<4x16xi1>
    //
    // CHECK: %[[LOAD0:.*]] = xegpu.load %arg0[%[[CST_OFFSETS0]]], %[[CST_MASK0]]
    // CHECK-SAME: -> vector<4x16xf32>
    // CHECK: %[[LOAD1:.*]] = xegpu.load %arg0[%[[CST_OFFSETS1]]], %[[CST_MASK1]]
    // CHECK-SAME: -> vector<4x16xf32>
    //
    // Local reductions
    // CHECK: %[[NEUTRAL0:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK: %[[LOCAL_RED0:.*]] = vector.multi_reduction <add>, %[[LOAD0]], %[[NEUTRAL0]] [1] : vector<4x16xf32> to vector<4xf32>
    // CHECK: %[[NEUTRAL1:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK: %[[LOCAL_RED1:.*]] = vector.multi_reduction <add>, %[[LOAD1]], %[[NEUTRAL1]] [1] : vector<4x16xf32> to vector<4xf32>
    //
    // Shape cast for SLM store
    // CHECK: %[[SC0:.*]] = vector.shape_cast %[[LOCAL_RED0]] : vector<4xf32> to vector<4x1xf32>
    // CHECK: %[[SC1:.*]] = vector.shape_cast %[[LOCAL_RED1]] : vector<4xf32> to vector<4x1xf32>
    //
    // SLM allocation and mem_desc
    // CHECK: %[[SLM:.*]] = memref.alloca() : memref<512xi8, 3>
    // CHECK: %[[MEMDESC:.*]] = xegpu.create_mem_desc %[[SLM]] : memref<512xi8, 3> -> !xegpu.mem_desc<8x16xf32>
    //
    // Store to SLM
    // CHECK: xegpu.store_matrix %[[SC0]], %[[MEMDESC]]{{.*}} : vector<4x1xf32>, !xegpu.mem_desc<8x16xf32>
    // CHECK: xegpu.store_matrix %[[SC1]], %[[MEMDESC]]{{.*}} : vector<4x1xf32>, !xegpu.mem_desc<8x16xf32>
    // CHECK: gpu.barrier
    //
    // Load from SLM
    // CHECK: %[[SLM_LOAD0:.*]] = xegpu.load_matrix %[[MEMDESC]]{{.*}} -> vector<4x16xf32>
    // CHECK: %[[SLM_LOAD1:.*]] = xegpu.load_matrix %[[MEMDESC]]{{.*}} -> vector<4x16xf32>
    //
    // Final reduction
    // CHECK: %[[FINAL_NEUTRAL:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK: %[[FINAL_RED0:.*]] = vector.multi_reduction <add>, %[[SLM_LOAD0]], %[[FINAL_NEUTRAL]] [1] : vector<4x16xf32> to vector<4xf32>
    // CHECK: %[[RES0:.*]] = arith.addf %[[FINAL_RED0]], %[[CST_ACC0]] : vector<4xf32>
    // CHECK: %[[FINAL_RED1:.*]] = vector.multi_reduction <add>, %[[SLM_LOAD1]], %[[FINAL_NEUTRAL]] [1] : vector<4x16xf32> to vector<4xf32>
    // CHECK: %[[RES1:.*]] = arith.addf %[[FINAL_RED1]], %[[CST_ACC1]] : vector<4xf32>

    %offset = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [1, 16], sg_data = [4, 16]>} dense<0> : vector<8x256xindex>
    %acc = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [1, 16], sg_data = [4, 16]>, dims = [1]>} dense<0.000000e+00> : vector<8xf32>
    %mask = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [1, 16], sg_data = [4, 16]>} dense<true> : vector<8x256xi1>
    %val = xegpu.load %arg0[%offset], %mask <{layout = #xegpu.layout<sg_layout = [1, 16], sg_data = [4, 16]>}> : memref<2048xf32, 1>, vector<8x256xindex>, vector<8x256xi1> -> vector<8x256xf32>
    %reduce = vector.multi_reduction <add>, %val, %acc {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [1, 16], sg_data = [4, 16]>, dims = [1]>} [1] : vector<8x256xf32> to vector<8xf32>
    gpu.return
  }

  // CHECK-LABEL: splat_constant
  gpu.func @splat_constant() {
    // CHECK-COUNT-2: %[[CST:.*]] = arith.constant dense<0> : vector<4xindex>
    %cst_2 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [16, 1], sg_data = [16, 4], order = [0, 1]>, dims = [0]>}  dense<0> : vector<8xindex>
    gpu.return
  }

  // CHECK-LABEL: gpu.func @step_broadcast
  gpu.func @step_broadcast() {
    // CHECK: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
    // CHECK: %[[REM:.*]] = arith.remui %[[SGID]], %[[C16]] : index
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
    // CHECK: %[[STEP:.*]] = vector.step : vector<4xindex>
    // CHECK: %[[BCST0:.*]] = vector.broadcast %[[C0:.*]] : index to vector<4xindex>
    // CHECK: %[[ADD0:.*]] = arith.addi %[[STEP]], %[[BCST0]] : vector<4xindex>
    // CHECK: %[[BCST4:.*]] = vector.broadcast %[[C4:.*]] : index to vector<4xindex>
    // CHECK: %[[ADD4:.*]] = arith.addi %[[STEP]], %[[BCST4]] : vector<4xindex>
    // CHECK: %[[RES0:.*]] = vector.broadcast %[[ADD0]] : vector<4xindex> to vector<16x4xindex>
    // CHECK: %[[RES1:.*]] = vector.broadcast %[[ADD4]] : vector<4xindex> to vector<16x4xindex>
    %2 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [16, 1], sg_data = [16, 4]>, dims = [0]>} : vector<8xindex>
    %bcast = vector.broadcast %2 {layout_result_0 = #xegpu.layout<sg_layout = [16, 1], sg_data = [16, 4]>} : vector<8xindex> to vector<256x8xindex>
    gpu.return
  }

  // CHECK-LABEL: create_nd_tdesc_with_shared_data
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @create_nd_tdesc_with_shared_data(%src: memref<256x128xf32>) {
    // CHECK: xegpu.create_nd_tdesc %[[ARG_0]] : memref<256x128xf32> -> !xegpu.tensor_desc<16x64xf32>
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32>
      -> !xegpu.tensor_desc<128x64xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 64]>>
    gpu.return
  }

  // CHECK-LABEL: broadcast
  // CHECK-SAME: %[[ARG_0:.*]]: memref<128x1xf32>
  gpu.func @broadcast(%src: memref<128x1xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<128x1xf32>
      -> !xegpu.tensor_desc<128x1xf32, #xegpu.layout<sg_layout = [4, 1], sg_data = [16, 1], lane_layout = [8, 1], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0] {layout =  #xegpu.layout<sg_layout = [4, 1], sg_data = [16, 1], lane_layout = [8, 1], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<128x1xf32, #xegpu.layout<sg_layout = [4, 1], sg_data = [16, 1], lane_layout = [8, 1], lane_data = [1, 1]>>
      -> vector<128x1xf32>
    // CHECK-COUNT-4: vector.broadcast {{.*}} : vector<16x1xf32> to vector<16x32xf32>
    // CHECK-NOT: vector.broadcast
    %broadcast = vector.broadcast %load
      {layout_result_0 = #xegpu.layout<sg_layout = [4, 1], sg_data = [16, 32], lane_layout = [8, 1], lane_data = [1, 1]>}
      : vector<128x1xf32> to vector<128x64xf32>
    gpu.return
  }

  gpu.func @scf_for(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %0 = xegpu.create_nd_tdesc %arg0 : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    %1 = xegpu.create_nd_tdesc %arg1 : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    // CHECK-LABEL: scf.for
    scf.for %arg2 = %c0 to %c1024 step %c256 {
      %3 = xegpu.load_nd %0[%arg2] {layout = #xegpu.layout<sg_layout = [8], sg_data = [16]>} : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>> -> vector<256xf32>
      xegpu.store_nd %3, %1[%arg2]  : vector<256xf32>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    }
    gpu.return
  }

  gpu.func @scf_while_and_condition(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256 = arith.constant 256 : index
    %0 = xegpu.create_nd_tdesc %arg0 : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    %1 = xegpu.load_nd %0[0] {layout = #xegpu.layout<sg_layout = [8], sg_data = [16]>} : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>> -> vector<256xf32>
    %2 = xegpu.create_nd_tdesc %arg1 : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    // CHECK: scf.while ({{.*}}) : (vector<16xf32>, vector<16xf32>, i32) -> (vector<16xf32>, vector<16xf32>, i32)
    %3:2 = scf.while (%arg2 = %1, %arg3 = %c0_i32) : (vector<256xf32>, i32) -> (vector<256xf32>, i32) {
      %4 = arith.cmpi slt, %arg3, %c10_i32 : i32
      // CHECK: scf.condition{{.*}} : vector<16xf32>, vector<16xf32>, i32
      scf.condition(%4) %arg2, %arg3 : vector<256xf32>, i32
    } do {
    // CHECK: ([[arg2:%.+]]: vector<16xf32>, [[arg3:%.+]]: vector<16xf32>, [[arg4:%.+]]: i32)
    ^bb0(%arg2: vector<256xf32>, %arg3: i32):
      xegpu.store_nd %arg2, %2[0]  : vector<256xf32>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
      %4 = arith.addi %arg3, %c1_i32 : i32
      %6 = xegpu.load_nd %0[%c256] {layout =  #xegpu.layout<sg_layout = [8], sg_data = [16]>} : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>> -> vector<256xf32>
      scf.yield %6, %4 : vector<256xf32>, i32
    }
    gpu.return
  }

  gpu.func @scf_if(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
    %c10 = arith.constant 10 : index
    %0 = gpu.subgroup_id : index
    %1 = xegpu.create_nd_tdesc %arg0 : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    %2 = xegpu.create_nd_tdesc %arg1 : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    %3 = arith.cmpi eq, %0, %c10 : index
    // CHECK-LABEL: scf.if
    // CHECK-SAME: (vector<16xf32>, vector<16xf32>)
    %4 = scf.if %3 -> (vector<256xf32>) {
      %5 = xegpu.load_nd %1[0] {layout = #xegpu.layout<sg_layout = [8], sg_data = [16]>} : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>> -> vector<256xf32>
      // CHECK-LABEL: scf.yield
      // CHECK-SAME: vector<16xf32>, vector<16xf32>
      scf.yield %5 : vector<256xf32>
    } else {
      %5 = xegpu.load_nd %2[0] {layout = #xegpu.layout<sg_layout = [8], sg_data = [16]>} : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>> -> vector<256xf32>
      // CHECK-LABEL: scf.yield
      // CHECK-SAME: vector<16xf32>, vector<16xf32>
      scf.yield %5 : vector<256xf32>
    } {layout_result_0 = #xegpu.layout<sg_layout = [8], sg_data = [16]>}
    xegpu.store_nd %4, %1[0]  : vector<256xf32>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    gpu.return
  }

  gpu.func @scf_if_tensor_desc(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
    %c10 = arith.constant 10 : index
    %id = gpu.subgroup_id : index

    %t = xegpu.create_nd_tdesc %arg0 : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    %d = xegpu.load_nd %t[0] {layout = #xegpu.layout<sg_layout = [8], sg_data = [16]>}: !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>> -> vector<256xf32>

    %0 = arith.cmpi eq, %id, %c10 : index
    // CHECK-LABEL: scf.if
    // CHECK-SAME: (!xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>)
    %1 = scf.if %0 -> (!xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>) {
      %2 = xegpu.create_nd_tdesc %arg0 : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
      // CHECK-LABEL: scf.yield
      // CHECK-SAME: !xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>
      scf.yield %2 : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    } else {
      %3 = xegpu.create_nd_tdesc %arg1 : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
      // CHECK-LABEL: scf.yield
      // CHECK-SAME: !xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>
      scf.yield %3 : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    }
    xegpu.store_nd %d, %1[0] : vector<256xf32>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    gpu.return
  }

  gpu.func @convert_layout_optimal(%arg0: memref<32x64xf32>) {
    %0 = xegpu.create_nd_tdesc %arg0 : memref<32x64xf32> -> !xegpu.tensor_desc<32x64xf32, #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [16, 16]>>
    // CHECK-COUNT-2: xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<inst_data = [16, 16]>> -> vector<16x16xf32>
    // CHECK-COUNT-2: xegpu.convert_layout {{.*}} <{input_layout = #xegpu.layout<inst_data = [16, 16]>, target_layout = #xegpu.layout<inst_data = [8, 16]>}> : vector<16x16xf32>
    %1 = xegpu.load_nd %0[0, 0] {layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [16, 16]>} : !xegpu.tensor_desc<32x64xf32, #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [16, 16]>> -> vector<32x64xf32>
    %2 = xegpu.convert_layout %1 <{input_layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [16, 16]>,
                                   target_layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [8, 16]>}> : vector<32x64xf32>
    gpu.return
  }

}
