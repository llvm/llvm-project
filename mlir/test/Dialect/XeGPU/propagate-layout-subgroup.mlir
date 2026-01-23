// RUN: mlir-opt -xevm-attach-target='chip=pvc' -xegpu-propagate-layout="layout-kind=subgroup" -split-input-file %s | FileCheck %s

gpu.module @test {
  // CHECK-LABEL: store_nd
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  func.func @store_nd(%src: memref<256x128xf32>) {
    // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG_0]] : memref<256x128xf32>
    // CHECK-SAME: -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>>
    // CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]] <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>}>
    // CHECK-SAME: : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>>
    // CHECK-SAME: -> vector<256x128xf32>
    // CHECK: xegpu.store_nd %[[LOAD]], %[[TDESC]] <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>}>
    // CHECK-SAME: : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>>
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32> -> !xegpu.tensor_desc<256x128xf32>
    %load = xegpu.load_nd %tdesc : !xegpu.tensor_desc<256x128xf32> -> vector<256x128xf32>
    xegpu.store_nd %load, %tdesc <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>}>
      : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32>
    return
  }
}

// -----

gpu.module @test {
  // CHECK-LABEL: vector_transpose
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  // CHECK-SAME: %[[ARG_1:.*]]: memref<128x256xf32>
  func.func @vector_transpose(%src: memref<256x128xf32>, %src1: memref<128x256xf32>) {
    // CHECK: %[[TDESC_LD:.*]] = xegpu.create_nd_tdesc %[[ARG_0]] : memref<256x128xf32> ->
    // CHECK-SAME: !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [4, 8], sg_data = [64, 32], order = [0, 1]>>
    // CHECK: %[[TDESC_ST:.*]] = xegpu.create_nd_tdesc %[[ARG_1]] : memref<128x256xf32> ->
    // CHECK-SAME: !xegpu.tensor_desc<128x256xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], order = [1, 0]>>

    // CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC_LD]][0, 0] <{layout = #xegpu.layout<sg_layout = [4, 8], sg_data = [64, 32], order = [0, 1]>}>
    // CHECK-SAME: !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [4, 8], sg_data = [64, 32], order = [0, 1]>> -> vector<256x128xf32>

    // CHECK: %[[TRANSPOSED:.*]] = vector.transpose %2, [1, 0]
    // CHECK-SAME {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], order = [1, 0]>} : vector<256x128xf32> to vector<128x256xf32>

    // CHECK: xegpu.store_nd %[[TRANSPOSED]], %[[TDESC_ST]][0, 0]
    // CHECK-SAME: <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], order = [1, 0]>}> : vector<128x256xf32>,
    // CHECK-SAME: !xegpu.tensor_desc<128x256xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], order = [1, 0]>>
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32> -> !xegpu.tensor_desc<256x128xf32>
    %tdesc1 = xegpu.create_nd_tdesc %src1 : memref<128x256xf32> -> !xegpu.tensor_desc<128x256xf32>
    %load = xegpu.load_nd %tdesc[0, 0] : !xegpu.tensor_desc<256x128xf32> -> vector<256x128xf32>
    %trans = vector.transpose %load, [1, 0] : vector<256x128xf32> to vector<128x256xf32>
    xegpu.store_nd %trans, %tdesc1[0, 0] <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], order = [1, 0]>}>
        : vector<128x256xf32>, !xegpu.tensor_desc<128x256xf32>
    return
  }
}

// -----
gpu.module @test {
  // CHECK-LABEL: vector_transpose
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  // CHECK-SAME: %[[ARG_1:.*]]: memref<128x256xf32>
  gpu.func @vector_transpose(%src: memref<256x128xf32>, %src1: memref<128x256xf32>) kernel attributes
      {known_block_size = array<i32: 1, 32, 16>} {
    // CHECK: %[[TDESC_LD:.*]] = xegpu.create_nd_tdesc %[[ARG_0]] : memref<256x128xf32> ->
    // CHECK-SAME: !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>>
    // CHECK: %[[TDESC_ST:.*]] = xegpu.create_nd_tdesc %[[ARG_1]] : memref<128x256xf32> ->
    // CHECK-SAME: !xegpu.tensor_desc<128x256xf32, #xegpu.layout<sg_layout = [4, 8], sg_data = [32, 32]>>

    // CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC_LD]][0, 0] <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>}> :
    // CHECK-SAME: !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>> -> vector<256x128xf32>

    // CHECK: %[[TRANSPOSED:.*]] = vector.transpose %2, [1, 0]
    // CHECK-SAME {layout_result_0 = #xegpu.layout<sg_layout = [4, 8], sg_data = [32, 32]>} : vector<256x128xf32> to vector<128x256xf32>

    // CHECK: xegpu.store_nd %[[TRANSPOSED]], %[[TDESC_ST]][0, 0]
    // CHECK-SAME: <{layout = #xegpu.layout<sg_layout = [4, 8], sg_data = [32, 32]>}> : vector<128x256xf32>,
    // CHECK-SAME: !xegpu.tensor_desc<128x256xf32, #xegpu.layout<sg_layout = [4, 8], sg_data = [32, 32]>>
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32> -> !xegpu.tensor_desc<256x128xf32>
    %tdesc1 = xegpu.create_nd_tdesc %src1 : memref<128x256xf32> -> !xegpu.tensor_desc<128x256xf32>
    %load = xegpu.load_nd %tdesc[0, 0] : !xegpu.tensor_desc<256x128xf32> -> vector<256x128xf32>
    %trans = vector.transpose %load, [1, 0] : vector<256x128xf32> to vector<128x256xf32>
    xegpu.store_nd %trans, %tdesc1[0, 0] : vector<128x256xf32>, !xegpu.tensor_desc<128x256xf32>
    gpu.return
  }
}

// -----
gpu.module @test {
  // CHECK-LABEL: dpas
  // CHECK-SAME: %[[A_MEMREF:.*]]: memref<128x128xf16>, %[[B_MEMREF:.*]]: memref<128x128xf16>
  // CHECK-SAME: %[[CD_MEMREF:.*]]: memref<128x128xf32>
  gpu.func @dpas(%a: memref<128x128xf16>, %b: memref<128x128xf16>, %d: memref<128x128xf32>) kernel attributes
      {known_block_size = array<i32: 1, 64, 16>} {
  // CHECK: %[[TDESC_A:.*]] = xegpu.create_nd_tdesc %[[A_MEMREF]] : memref<128x128xf16> ->
  // CHECK-SAME: !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>>

  // CHECK: %[[A_LOADED:.*]] = xegpu.load_nd %[[TDESC_A]]
  // CHECK-SAME: <{layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>}>
  // CHECK-SAME: : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>> -> vector<128x128xf16>

  // CHECK: %[[TDESC_B:.*]] = xegpu.create_nd_tdesc %[[B_MEMREF]] : memref<128x128xf16> ->
  // CHECK-SAME: !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>>

  // CHECK: %[[B_LOADED:.*]] = xegpu.load_nd %[[TDESC_B]] <{layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>}>
  // CHECK-SAME: : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>> -> vector<128x128xf16>

  // CHECK: %[[DPAS_RES:.*]] = xegpu.dpas %[[A_LOADED]], %[[B_LOADED]]
  // CHECK-SAME: {layout_a = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>,
  // CHECK-SAME: layout_b = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>,
  // CHECK-SAME: layout_cd = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>} :
  // CHECK-SAME: vector<128x128xf16>, vector<128x128xf16> -> vector<128x128xf32>

  // CHECK: %[[TDESC_ST:.*]] = xegpu.create_nd_tdesc %[[CD_MEMREF]] : memref<128x128xf32> ->
  // CHECK-SAME: !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>>

  // CHECK: xegpu.store_nd %[[DPAS_RES]], %[[TDESC_ST]][0, 0]
  // CHECK-SAME: <{layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>}> :
  // CHECK-SAME: vector<128x128xf32>, !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>>

    %tdesc_a = xegpu.create_nd_tdesc %a : memref<128x128xf16> -> !xegpu.tensor_desc<128x128xf16>
    %load_a =  xegpu.load_nd %tdesc_a : !xegpu.tensor_desc<128x128xf16> -> vector<128x128xf16>
    %tdesc_b = xegpu.create_nd_tdesc %b : memref<128x128xf16> -> !xegpu.tensor_desc<128x128xf16>
    %load_b =  xegpu.load_nd %tdesc_b : !xegpu.tensor_desc<128x128xf16> -> vector<128x128xf16>
    %dpas = xegpu.dpas %load_a, %load_b : vector<128x128xf16>, vector<128x128xf16> -> vector<128x128xf32>
    %tdesc_cd = xegpu.create_nd_tdesc %d : memref<128x128xf32> -> !xegpu.tensor_desc<128x128xf32>
    xegpu.store_nd %dpas, %tdesc_cd[0, 0] : vector<128x128xf32>, !xegpu.tensor_desc<128x128xf32>
    gpu.return
  }
}
