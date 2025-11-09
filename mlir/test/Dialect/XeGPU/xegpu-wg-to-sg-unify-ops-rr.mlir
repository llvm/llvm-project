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
    // CHECK-COUNT-4: xegpu.load_nd {{%.*}}[{{%.*}}, {{%.*}}] : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf32>
    // CHECK-NOT: xegpu.load_nd
    %tdesc = xegpu.create_nd_tdesc %src: memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load = xegpu.load_nd %tdesc[0, 0]
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
    %load =  xegpu.load_nd %tdesc[0, 0]
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
    // CHECK-COUNT-4: xegpu.load_nd {{%.*}}[{{%.*}}, {{%.*}}] : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf16>
    // CHECK-COUNT-4: xegpu.create_nd_tdesc %[[ARG_1]] : memref<128x256xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
    // CHECK-COUNT-4: xegpu.load_nd {{%.*}}[{{%.*}}, {{%.*}}] : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
    // CHECK-COUNT-16: xegpu.dpas %{{.*}}, %{{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16x16xf16>, vector<16x16xf16> -> vector<16x16xf32>
    // CHECK-NOT: xegpu.dpas
    %tdesc_a = xegpu.create_nd_tdesc %a : memref<256x128xf16>
      -> !xegpu.tensor_desc<256x128xf16, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load_a =  xegpu.load_nd %tdesc_a[0, 0]
      : !xegpu.tensor_desc<256x128xf16, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf16>
    %tdesc_b = xegpu.create_nd_tdesc %b : memref<128x256xf16>
      -> !xegpu.tensor_desc<128x256xf16, #xegpu.layout<sg_layout = [4, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [2, 1]>>
    %load_b =  xegpu.load_nd %tdesc_b[0, 0]
      : !xegpu.tensor_desc<128x256xf16, #xegpu.layout<sg_layout = [4, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [2, 1]>>
      -> vector<128x256xf16>
    %dpas = xegpu.dpas %load_a, %load_b
      {layout_result_0 =  #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<256x128xf16>, vector<128x256xf16> -> vector<256x256xf32>
    gpu.return
  }

  // CHECK-LABEL: vector_reduce_dim_1
  gpu.func @vector_reduce_dim_1(%src: memref<256x64xf32>) {
    // CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<16xf32>
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>, dims = [1]>} dense<1.0> : vector<256xf32>
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x64xf32>
      -> !xegpu.tensor_desc<256x64xf32, #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>>
    %load =  xegpu.load_nd %tdesc[0, 0]
      : !xegpu.tensor_desc<256x64xf32, #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>>
      -> vector<256x64xf32>
    // CHECK-COUNT-2: vector.multi_reduction <add>, {{.*}}, %[[CST]] [1] : vector<16x64xf32> to vector<16xf32>
    // CHECK-NOT: vector.multi_reduction
    %reduce = vector.multi_reduction <add>, %load, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>, dims = [1]>} [1]
      : vector<256x64xf32> to vector<256xf32>
    gpu.return
  }

  gpu.func @non_splat_constant() {
    // CHECK-DAG: %[[BASECST:.*]] = arith.constant dense<{{.*}}> : vector<2x1xindex>
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[REMU1:.*]] = index.remu %[[SGID]], %[[C1:.*]]
    // CHECK-DAG: %[[DIVU:.*]] = index.divu %[[SGID]], %[[C1:.*]]
    // CHECK-DAG: %[[REMU2:.*]] = index.remu %[[DIVU]], %[[C8:.*]]
    // CHECK-DAG: %[[MUL:.*]] = index.mul %[[REMU2]], %[[C2:.*]]
    // CHECK-DAG: %[[REMU3:.*]] = index.remu %[[MUL]], %[[C32:.*]]
    // CHECK-DAG: %[[REMU4:.*]] = index.remu %[[REMU1]], %[[C1:.*]]
    // CHECK-DAG: %[[ADD16:.*]] = arith.addi %[[MUL]], %[[C16:.*]] : index
    // CHECK-DAG: %[[REMU5:.*]] = index.remu %[[ADD16]], %[[C32:.*]]
    // CHECK-DAG: %[[REMU6:.*]] = index.remu %[[REMU1]], %[[C1:.*]]
    // CHECK-DAG: %[[STRIDE1:.*]] = arith.muli %[[REMU3]], %[[C16:.*]] : index
    // CHECK-DAG: %[[ADDSTRIDES:.*]] = arith.addi %[[C0:.*]], %[[STRIDE1]] : index
    // CHECK-DAG: %[[STRIDE2:.*]] = arith.muli %[[REMU4]], %[[C0:.*]] : index
    // CHECK-DAG: %[[ADDSTRIDES1:.*]] = arith.addi %[[ADDSTRIDES]], %[[STRIDE2]] : index
    // CHECK-DAG: %[[BCAST1:.*]] = vector.broadcast %[[ADDSTRIDES1]] : index to vector<2x1xindex>
    // CHECK-DAG: %[[RESULT1:.*]] = arith.addi %[[BASECST]], %[[BCAST1]] : vector<2x1xindex>
    // CHECK-DAG: %[[STRIDE3:.*]] = arith.muli %[[REMU5]], %[[C16:.*]] : index
    // CHECK-DAG: %[[ADDSTRIDES2:.*]] = arith.addi %[[C0:.*]], %[[STRIDE3]] : index
    // CHECK-DAG: %[[STRIDE4:.*]] = arith.muli %[[REMU6]], %[[C0:.*]] : index
    // CHECK-DAG: %[[ADDSTRIDES3:.*]] = arith.addi %[[ADDSTRIDES2]], %[[STRIDE4]] : index
    // CHECK-DAG: %[[BCAST2:.*]] = vector.broadcast %[[ADDSTRIDES3]] : index to vector<2x1xindex>
    // CHECK-DAG: %[[RESULT2:.*]] = arith.addi %[[BASECST]], %[[BCAST2]] : vector<2x1xindex>
    %cst_2 = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [8, 1], sg_data = [2, 1]>} dense<[[0], [16], [32], [48], [64], [80], [96], [112], [128], [144], [160], [176], [192], [208], [224], [240], [256], [272], [288], [304], [320], [336], [352], [368], [384], [400], [416], [432], [448], [464], [480], [496]]> : vector<32x1xindex>
    gpu.return
  }

  // CHECK-LABEL: vector_transpose
  gpu.func @vector_transpose(%src: memref<256x128xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32>
        -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 16], lane_layout = [16, 1], lane_data = [1, 1], order =[0, 1]>>
    %load = xegpu.load_nd %tdesc[0, 0]
        : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 16], lane_layout = [16, 1], lane_data = [1, 1], order =[0, 1]>>
        -> vector<256x128xf32>
    // CHECK-COUNT-2: vector.transpose {{.*}}, [1, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1], order = [1, 0]>} : vector<32x16xf32> to vector<16x32xf32>
    // CHECK-NOT: vector.transpose
    %trans = vector.transpose %load, [1, 0] {layout_result_0 = #xegpu.layout<sg_layout = [4, 8], sg_data = [16, 32], lane_layout = [1, 16], lane_data = [1, 1], order =[1, 0]>} : vector<256x128xf32> to vector<128x256xf32>
      gpu.return
  }
}

