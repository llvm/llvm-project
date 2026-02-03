// RUN: mlir-opt --xegpu-wg-to-sg-distribute -split-input-file %s | FileCheck %s

gpu.module @test_distribution {
  // CHECK-LABEL: create_nd_tdesc_no_offset
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @create_nd_tdesc_no_offset(%src: memref<256x128xf32>) {
      // CHECK-COUNT-4: xegpu.create_nd_tdesc %[[ARG_0]] : memref<256x128xf32>
      // CHECK-SAME: -> !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      // CHECK-NOT: xegpu.create_nd_tdesc
      %tdesc = xegpu.create_nd_tdesc %src: memref<256x128xf32>
        -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
      gpu.return
  }

  // CHECK-LABEL: load_nd_tdesc_with_offset
  gpu.func @load_nd_tdesc_with_offset(%src: memref<256x128xf32>) {
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
    // CHECK-COUNT-4: xegpu.store_nd %{{.*}}, {{%.*}}[{{%.*}}, {{%.*}}] : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK-NOT: xegpu.store_nd
    %tdesc = xegpu.create_nd_tdesc %src: memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf32>
    xegpu.store_nd %load, %tdesc[0, 0]
      : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }

  // CHECK-LABEL: prefetch_nd_tdesc_with_offset
  gpu.func @prefetch_nd_tdesc_with_offset(%src: memref<256x128xf32>) {
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
    // CHECK-COUNT-2: vector.transpose {{.*}}, [1, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1], order = [1, 0]>} : vector<32x16xf32> to vector<16x32xf32>
    // CHECK-NOT: vector.transpose
    %trans = vector.transpose %load, [1, 0] {layout_result_0 = #xegpu.layout<sg_layout = [4, 8], sg_data = [16, 32], lane_layout = [1, 16], lane_data = [1, 1], order =[1, 0]>} : vector<256x128xf32> to vector<128x256xf32>
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
  // CHECK: %[[CAST:.*]] = vector.shape_cast %[[REDUCE:.*]] {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : vector<8xf32> to vector<8x1xf32>
  // CHECK: %[[BCAST:.*]] = vector.broadcast %[[CAST]] {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : vector<8x1xf32> to vector<8x128xf32>
  gpu.func @distribute_shapecast_expandunitdims_broadcast(%arg0: memref<4096x128xf32>, %arg1: memref<4096x128xf32>) {
    %cst_0 = arith.constant {layout_result_0=#xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>, dims = [1]>} dense<0xFF800000> : vector<256xf32>
    %block_id_x = gpu.block_id  x
    %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x128xf32> -> !xegpu.tensor_desc<256x128xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>>
    %1 = xegpu.load_nd %0[%block_id_x, 0]  : !xegpu.tensor_desc<256x128xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>> -> vector<256x128xf32>
    %2 = vector.multi_reduction <maximumf>, %1, %cst_0 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>, dims = [1]>} [1] : vector<256x128xf32> to vector<256xf32>
    %3 = vector.shape_cast %2 {layout_result_0 =  #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>, layout_operand_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>, dims = [1]>} : vector<256xf32> to vector<256x1xf32>
    %4 = vector.broadcast %3 {layout_result_0 =  #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>} : vector<256x1xf32>to vector<256x128xf32>
    %9 = xegpu.create_nd_tdesc %arg0 : memref<4096x128xf32> -> !xegpu.tensor_desc<256x128xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>>
    xegpu.store_nd %4, %9[%block_id_x, 0] : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 128], inst_data = [8, 16]>>
    gpu.return
  }

}
