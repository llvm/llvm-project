// RUN: mlir-opt --xegpu-wg-to-sg-distribute -split-input-file %s | FileCheck %s

// CHECK-DAG: #map = affine_map<()[s0] -> (s0 floordiv 32)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 mod 32)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (0)>
// CHECK-DAG: #map3 = affine_map<()[s0] -> (s0 floordiv 4)>
// CHECK-DAG: #map4 = affine_map<()[s0] -> (s0 mod 4)>
// CHECK-DAG: #map5 = affine_map<()[s0] -> ((s0 mod 32) floordiv 16)>
// CHECK-DAG: #map6 = affine_map<()[s0] -> (s0 mod 16)>
// CHECK-DAG: #map7 = affine_map<()[s0] -> ((s0 mod 16) floordiv 4)>
gpu.module @test_distribution {
  // CHECK-LABEL: create_nd_tdesc_no_offset
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @create_nd_tdesc_no_offset(%src: memref<256x128xf32>) {
    // CHECK: xegpu.create_nd_tdesc %[[ARG_0]] : memref<256x128xf32>
    // CHECK-SAME: -> !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32>
        -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
      gpu.return
  }

  // CHECK-LABEL: create_nd_tdesc_with_ptr
  // CHECK-SAME: %[[ARG_0:.*]]: ui64
  gpu.func @create_nd_tdesc_with_ptr(%src: ui64, %w : index, %h : index, %x : index, %y : index) {
    // CHECK: xegpu.create_nd_tdesc %[[ARG_0]], shape : [{{.*}}, {{.*}}], strides : [{{.*}}, {{.*}}] : ui64
    // CHECK-SAME: -> !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %c1 = arith.constant 1 : index
    %tdesc = xegpu.create_nd_tdesc %src, shape:[%h, %w], strides: [%w, %c1] : ui64
        -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
      gpu.return
  }

  // CHECK-LABEL: load_nd_tdesc_with_offset
  gpu.func @load_nd_tdesc_with_offset(%src: memref<256x128xf32>) {
    //CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<256x128xf32> -> !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    //CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    //CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
    //CHECK-DAG: %[[SGIDX:.*]] = arith.remui %[[SGID]], %[[C4]]
    //CHECK-DAG: %[[SGIDY_TMP:.*]] = arith.divui %[[SGID]], %[[C4]]
    //CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
    //CHECK-DAG: %[[SGIDY:.*]] = arith.remui %[[SGIDY_TMP]], %[[C8]]
    //CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
    //CHECK-DAG: %[[L_OFF_Y:.*]] = arith.muli %[[SGIDY]], %[[C32]] : index
    //CHECK-DAG: %[[L_OFF_X:.*]] = arith.muli %[[SGIDX]], %[[C32_1:.*]] : index
    //CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
    //CHECK-DAG: %[[OFF_Y:.*]] = arith.remui %[[L_OFF_Y]], %[[C256]] : index
    //CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
    //CHECK-DAG: %[[OFF_X:.*]] = arith.remui %[[L_OFF_X]], %[[C128]] : index
    //CHECK-DAG: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]][{{%.*}}, {{%.*}}] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<32x32xf32>
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf32>
    gpu.return
  }

  // CHECK-LABEL: store_nd_with_offsets
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @store_nd_with_offsets(%src: memref<256x128xf32>) {
    //CHECK: xegpu.store_nd %{{.*}}, {{%.*}}[{{%.*}}, {{%.*}}]  : vector<32x32xf32>, !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %tdesc = xegpu.create_nd_tdesc %src: memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf32>
    xegpu.store_nd %load, %tdesc[0, 0]
      : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
}

  // CHECK-LABEL: prefetch_nd_tdesc_with_offset
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @prefetch_nd_tdesc_with_offset(%src: memref<256x128xf32>) {
    //CHECK: xegpu.prefetch_nd %{{.*}}[{{%.*}}, {{%.*}}] : !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %cst0 = arith.constant 0 : index
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.prefetch_nd %tdesc[%cst0, %cst0]
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }

  // CHECK-LABEL: dpas
  gpu.func @dpas(%a: memref<128x128xf16>, %b: memref<128x128xf16>) {
    // CHECK: %[[DPAS:.*]] = xegpu.dpas %{{.*}}, %{{.*}} {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16x128xf16>, vector<128x16xf16> -> vector<16x16xf32>
    %tdesc_a = xegpu.create_nd_tdesc %a : memref<128x128xf16>
      -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load_a =  xegpu.load_nd %tdesc_a[0, 0] {layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128], lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<128x128xf16>
    %tdesc_b = xegpu.create_nd_tdesc %b : memref<128x128xf16>
      -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16], lane_layout = [1, 16], lane_data = [2, 1]>>
    %load_b =  xegpu.load_nd %tdesc_b[0, 0] {layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16], lane_layout = [1, 16], lane_data = [2, 1]>}
      : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16], lane_layout = [1, 16], lane_data = [2, 1]>>
      -> vector<128x128xf16>
    %dpas = xegpu.dpas %load_a, %load_b
       {layout_a = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128], lane_layout = [1, 16], lane_data = [1, 1]>,
        layout_b = #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16], lane_layout = [1, 16], lane_data = [2, 1]>,
        layout_cd =  #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>}    
      : vector<128x128xf16>, vector<128x128xf16> -> vector<128x128xf32>
    gpu.return
  }

  // CHECK-LABEL: dpas_no_sg_data
  gpu.func @dpas_no_sg_data(%a: memref<128x128xf16>, %b: memref<128x128xf16>) {
    // CHECK: %[[DPAS:.*]] = xegpu.dpas %{{.*}}, %{{.*}} {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1], order = [1, 0]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1], order = [1, 0]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1], order = [1, 0]>} : vector<16x16xf16>, vector<16x16xf16> -> vector<16x16xf32>
    %tdesc_a = xegpu.create_nd_tdesc %a : memref<128x128xf16>
      -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [1, 1],
      order = [1, 0]>>
    %load_a =  xegpu.load_nd %tdesc_a[0, 0] {layout = #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [1, 1],
      order = [1, 0]>}
      : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [1, 1],
      order = [1, 0]>>
      -> vector<128x128xf16>
    %tdesc_b = xegpu.create_nd_tdesc %b : memref<128x128xf16>
      -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [2, 1],
      order = [1, 0]>>
    %load_b =  xegpu.load_nd %tdesc_b[0, 0] {layout = #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [2, 1], order = [1, 0]> }
      : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [2, 1],
      order = [1, 0]>>
      -> vector<128x128xf16>
    %dpas = xegpu.dpas %load_a, %load_b
      {layout_a = #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [1, 1],
      order = [1, 0]>,
       layout_b = #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [2, 1],
      order = [1, 0]>,
       layout_cd =  #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [1, 1], order = [1, 0]>}
      : vector<128x128xf16>, vector<128x128xf16> -> vector<128x128xf32>
    gpu.return
  }

  // CHECK-LABEL: dpas_with_no_create_nd_desc
  gpu.func @dpas_with_no_create_nd_desc(%a: vector<256x128xf32>, %b: vector<128x256xf32>) {
    // CHECK-NOT: vector<32x32xf32>
    %dpas = xegpu.dpas %a, %b
      {layout_a =  #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128], lane_layout = [1, 16], lane_data = [1, 1]>, 
       layout_b =  #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16], lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_cd =  #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<256x128xf32>, vector<128x256xf32> -> vector<256x256xf32>
    gpu.return
  }

  // CHECK-LABEL: broadcast_dim1
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x1xf32>
  gpu.func @broadcast_dim1(%src: memref<256x1xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x1xf32>
      -> !xegpu.tensor_desc<256x1xf32, #xegpu.layout<sg_layout = [8, 1], sg_data = [32, 1], lane_layout = [8, 1], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [32, 1], lane_layout = [8, 1], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<256x1xf32, #xegpu.layout<sg_layout = [8, 1], sg_data = [32, 1], lane_layout = [8, 1], lane_data = [1, 1]>>
      -> vector<256x1xf32>
    // CHECK: vector.broadcast {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>}
    // CHECK-SAME: : vector<32x1xf32> to vector<32x32xf32>
    %broadcast = vector.broadcast %load
      {layout_result_0 = #xegpu.layout<sg_layout = [8, 1], sg_data = [32, 32], lane_layout = [8, 1], lane_data = [1, 1]>}
      : vector<256x1xf32> to vector<256x32xf32>
    gpu.return
  }

  // CHECK-LABEL: broadcast_dim0
  // CHECK-SAME: %[[ARG_0:.*]]: memref<1x128xf32>
  gpu.func @broadcast_dim0(%src: memref<1x128xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<1x128xf32>
      -> !xegpu.tensor_desc<1x128xf32, #xegpu.layout<sg_layout = [1, 4], sg_data = [1, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [1, 4], sg_data = [1, 32], lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<1x128xf32, #xegpu.layout<sg_layout = [1, 4], sg_data = [1, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<1x128xf32>
    // CHECK: vector.broadcast {{.*}} {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    // CHECK-SAME: : vector<1x32xf32> to vector<32x32xf32>
    %broadcast = vector.broadcast %load
      {layout_result_0 = #xegpu.layout<sg_layout = [1, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<1x128xf32> to vector<32x128xf32>
    gpu.return
  }

  // CHECK-LABEL: gemm_with_load_store_offset
  // CHECK-SAME: %[[ARG_0:.*]]: memref<1024x1024xf16>, %[[ARG_1:.*]]: memref<1024x1024xf16>, %[[ARG_2:.*]]: memref<1024x1024xf32>
  gpu.func @gemm_with_load_store_offset(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) {
    //CHECK: [[c0:%.+]] = arith.constant 0 : index
    //CHECK: [[c128:%.+]] = arith.constant 128 : index
    //CHECK: [[c1024:%.+]] = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %0 = arith.muli %block_id_x, %c128 : index
    %1 = arith.muli %block_id_y, %c128 : index
    %2 = xegpu.create_nd_tdesc %arg2 : memref<1024x1024xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>>
    // CHECK: [[DESC_A:%.+]] = xegpu.create_nd_tdesc %[[ARG_0]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x128xf16>
    // CHECK: [[DESC_B:%.+]] = xegpu.create_nd_tdesc %[[ARG_1]] : memref<1024x1024xf16> -> !xegpu.tensor_desc<128x16xf16>
    %3 = xegpu.create_nd_tdesc %arg0 : memref<1024x1024xf16> -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>>
    %4 = xegpu.create_nd_tdesc %arg1 : memref<1024x1024xf16> -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>>
    // load_nd with offset
    %5 = xegpu.load_nd %2[%0, %1] {layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>}: !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>> -> vector<128x128xf32>
    %6 = xegpu.load_nd %3[%0, %c0] {layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>}: !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>> -> vector<128x128xf16>
    %7 = xegpu.load_nd %4[%c0, %1] {layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>}: !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>> -> vector<128x128xf16>
    // scf.for loop
    //      CHECK: [[scf:%.+]]:3 = scf.for [[arg3:%.+]] = [[c0]] to [[c1024]] step [[c128]]
    // CHECK-SAME: iter_args([[arg4:%.+]] = {{.*}}, [[arg5:%.+]] = {{.*}}, [[arg6:%.+]] = {{.*}}) ->
    // CHECK-SAME: (vector<16x128xf16>, vector<128x16xf16>, vector<16x16xf32>)
    //      CHECK: [[c:%.+]] = xegpu.dpas [[arg4]], [[arg5]], [[arg6]] : vector<16x128xf16>, vector<128x16xf16>, vector<16x16xf32> -> vector<16x16xf32>
    //      CHECK: [[a:%.+]] = xegpu.load_nd [[DESC_A]][{{%.*}}, {{%.*}}]  : !xegpu.tensor_desc<16x128xf16> -> vector<16x128xf16>
    //      CHECK: [[b:%.+]] = xegpu.load_nd [[DESC_B]][{{%.*}}, {{%.*}}]  : !xegpu.tensor_desc<128x16xf16> -> vector<128x16xf16>
    //      CHECK: scf.yield [[a]], [[b]], [[c]] : vector<16x128xf16>, vector<128x16xf16>, vector<16x16xf32>
    %8:3 = scf.for %arg3 = %c0 to %c1024 step %c128 iter_args(%arg4 = %6, %arg5 = %7, %arg6 = %5)
        -> (vector<128x128xf16>, vector<128x128xf16>, vector<128x128xf32>) {
      // load_nd with offset inside loop
      %9 = xegpu.dpas %arg4, %arg5, %arg6 
          {layout_a = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>,
           layout_b = #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>,
           layout_cd = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>}
          : vector<128x128xf16>, vector<128x128xf16>, vector<128x128xf32> -> vector<128x128xf32>
      %10 = xegpu.load_nd %3[%arg3, %c0] {layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>}: !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>> -> vector<128x128xf16>
      %11 = xegpu.load_nd %4[%c0, %arg3] {layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>}: !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>> -> vector<128x128xf16>
      scf.yield %10, %11, %9 : vector<128x128xf16>, vector<128x128xf16>, vector<128x128xf32>
    }
    // store_nd with offset
    xegpu.store_nd %8#2, %2[%0, %1] : vector<128x128xf32>, !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>>
    gpu.return
  }

  // CHECK-LABEL: @subgroup_id_range
  gpu.func @subgroup_id_range(%src: memref<256x128xf32>, %src1: memref<128x256xf32>, %src2: memref<128x64xf32>) {
    %sg_id = gpu.subgroup_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c31 = arith.constant 31 : index
    %c3 = arith.constant 3 : index
    %cond1 = arith.cmpi sge, %sg_id, %c0 : index
    %cond2 = arith.cmpi slt, %sg_id, %c1 : index
    %cond = arith.andi %cond1, %cond2 : i1
    scf.if %cond {
        // CHECK-NOT: index.sub
        %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32>
          -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [8, 4], lane_data = [1, 1]>>
        %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [8, 4], lane_data = [1, 1]>}
          : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [8, 4], lane_data = [1, 1]>>
          -> vector<256x128xf32>
    } {sg_id_range = #xegpu.range<[0, 32]>}
    %cond3 = arith.cmpi sge, %sg_id, %c2 : index
    %cond4 = arith.cmpi slt, %sg_id, %c31 : index
    %cond5 = arith.andi %cond3, %cond4 : i1
    scf.if %cond5 {
      // CHECK: %[[SGID:.*]] = gpu.subgroup_id : index
      // CHECK: %[[C2:.*]] = arith.constant 2 : index
      // CHECK: %[[SUB:.*]] = index.sub %{{.*}}, %[[C2]]
      %tdesc = xegpu.create_nd_tdesc %src2 : memref<128x64xf32>
        -> !xegpu.tensor_desc<128x64xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], lane_layout = [8, 4], lane_data = [1, 1]>>
      %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], lane_layout = [8, 4], lane_data = [1, 1]>}
        : !xegpu.tensor_desc<128x64xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], lane_layout = [8, 4], lane_data = [1, 1]>>
        -> vector<128x64xf32>
      %exp = math.exp %load {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], lane_layout = [8, 4], lane_data = [1, 1]>} : vector<128x64xf32>
    }{sg_id_range = #xegpu.range<[2, 18]>}
    gpu.return
  }

  // CHECK-LABEL: @subgroup_id_range_nested_if
  gpu.func @subgroup_id_range_nested_if(%src: memref<256x128xf32>, %src1: memref<128x64xf32>) {
    %sg_id = gpu.subgroup_id : index
    %c1 = arith.constant 1 : i1
    %c3 = arith.constant 3 : index
    %c32 = arith.constant 32 : index
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [8, 4], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [8, 4], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [8, 4], lane_data = [1, 1]>>
      -> vector<256x128xf32>
    %cond1 = arith.cmpi sge, %sg_id, %c3 : index
    %cond2 = arith.cmpi slt, %sg_id, %c32 : index
    %cond = arith.andi %cond1, %cond2 : i1
    scf.if %c1 {
      scf.if %cond {
        // CHECK: %[[SGID:.*]] = gpu.subgroup_id : index
        // CHECK: %[[C3:.*]] = arith.constant 3 : index
        // CHECK: %[[SUB:.*]] = index.sub %{{.*}}, %[[C3]]
        %td = xegpu.create_nd_tdesc %src1 : memref<128x64xf32>
          -> !xegpu.tensor_desc<128x64xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], lane_layout = [8, 4], lane_data = [1, 1]>>
        %ld =  xegpu.load_nd %td[0, 0] {layout = #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], lane_layout = [8, 4], lane_data = [1, 1]>}
          : !xegpu.tensor_desc<128x64xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], lane_layout = [8, 4], lane_data = [1, 1]>>
          -> vector<128x64xf32>
        %exp = math.exp %ld {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], lane_layout = [8, 4], lane_data = [1, 1]>} : vector<128x64xf32>
    }
  } {sg_id_range = #xegpu.range<[3, 19]>}
  gpu.return
  }

  // CHECK-LABEL: @load_gather
  // CHECK-SAME: %[[ARG0:.*]]: memref<?xf16>
  gpu.func @load_gather(%src : memref<?xf16>) {
    // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<32x4xindex>
    // CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<32x4xi1>
    // CHECK: %[[LOAD:.*]] = xegpu.load %[[ARG0]][%[[CST]]], %[[MASK]] <{chunk_size = 1 : i64, l1_hint = #xegpu.cache_hint<cached>}>
    // CHECK-SAME: : memref<?xf16>, vector<32x4xindex>, vector<32x4xi1> -> vector<32x4xf16>
    %offset =  arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 4]>} dense<0> : vector<256x16xindex>
    %mask = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 4]>} dense<1> : vector<256x16xi1>
    %load = xegpu.load %src[%offset], %mask {chunk_size = 1, layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 4]>, l1_hint = #xegpu.cache_hint<cached>}
      : memref<?xf16>, vector<256x16xindex>, vector<256x16xi1> -> vector<256x16xf16>
    gpu.return
  }

  // CHECK-LABEL: @store_scatter
  // CHECK-SAME: %[[ARG0:.*]]: memref<256xf16>
  gpu.func @store_scatter(%dest : memref<256xf16>) {
    // CHECK: %[[VAL:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [8]>} dense<2.550000e+01> : vector<8xf16>
    // CHECK: %[[CST:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [8]>} dense<0> : vector<8xindex>
    // CHECK: %[[MASK:.*]] = arith.constant {layout_result_0 = #xegpu.layout<inst_data = [8]>} dense<true> : vector<8xi1>
    // CHECK: xegpu.store %[[VAL]], %[[ARG0]][%[[CST]]], %[[MASK]] <{chunk_size = 1 : i64, l1_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<inst_data = [8]>}>
     // CHECK-SAME: : vector<8xf16>, memref<256xf16>, vector<8xindex>, vector<8xi1>
    %val = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [32], sg_data = [8], inst_data = [8]>} dense<25.5> : vector<256xf16>
    %offset = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [32], sg_data = [8], inst_data = [8]>} dense<0> : vector<256xindex>
    %mask = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [32], sg_data = [8], inst_data = [8]>} dense<1> : vector<256xi1>
    xegpu.store %val, %dest[%offset], %mask {chunk_size = 1, layout = #xegpu.layout<sg_layout = [32], sg_data = [8], inst_data = [8]>,
                                             layout_operand_2 = #xegpu.layout<sg_layout = [32], sg_data = [8], inst_data = [8]>,
                                             layout_operand_3 = #xegpu.layout<sg_layout = [32], sg_data = [8], inst_data = [8]>,
                                             l1_hint = #xegpu.cache_hint<cached>}
      : vector<256xf16>, memref<256xf16>, vector<256xindex>, vector<256xi1>
    gpu.return
  }

  // CHECK-LABEL: @load_with_non_unit_chunk_size
  // CHECK-SAME: %[[ARG0:.*]]: memref<?xf16>
  gpu.func @load_with_non_unit_chunk_size(%src : memref<?xf16>) {
    // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<8xindex>
    // CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<8xi1>
    // CHECK: %[[LOAD:.*]] = xegpu.load %[[ARG0]][%[[CST]]], %[[MASK]] <{chunk_size = 4 : i64, l1_hint = #xegpu.cache_hint<cached>}>
    // CHECK-SAME: : memref<?xf16>, vector<8xindex>, vector<8xi1> -> vector<8x4xf16>
    %offset =  arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [32], sg_data = [8]>} dense<0> : vector<256xindex>
    %mask = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [32], sg_data = [8]>} dense<1> : vector<256xi1>
    %load = xegpu.load %src[%offset], %mask {chunk_size = 4, layout = #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 4]>, l1_hint = #xegpu.cache_hint<cached>}
      : memref<?xf16>, vector<256xindex>, vector<256xi1> -> vector<256x4xf16>
    gpu.return
  }

  // CHECK-LABEL: distribute_load_matrix
  // CHECK-SAME: [[arg0:%.+]]: memref<32768xi8, 3>
  gpu.func @distribute_load_matrix(%arg0: memref<32768xi8, 3>) {
    //CHECK: [[mdesc:%.+]] = xegpu.create_mem_desc [[arg0]] : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
    //CHECK: [[sgid:%.+]] = gpu.subgroup_id : index
    //CHECK: [[c4:%.+]] = arith.constant 4 : index
    //CHECK: [[sgidx:%.+]] = arith.remui [[sgid]], [[c4]] : index
    //CHECK: [[sgidy_tmp:%.+]] = arith.divui [[sgid]], [[c4]] : index
    //CHECK: [[c2:%.+]] = arith.constant 2 : index
    //CHECK: [[sgidy:%.+]] = arith.remui [[sgidy_tmp]], [[c2]] : index
    //CHECK: [[c32:%.+]] = arith.constant 32 : index
    //CHECK: [[l_off_y:%.+]] = arith.muli [[sgidy]], [[c32]] : index
    //CHECK: [[c32_0:%.+]] = arith.constant 32 : index
    //CHECK: [[l_off_x:%.+]] = arith.muli [[sgidx]], [[c32_0]] : index
    //CHECK: [[c64:%.+]] = arith.constant 64 : index
    //CHECK: [[off_y:%.+]] = arith.remui [[l_off_y]], [[c64]] : index
    //CHECK: [[c128:%.+]] = arith.constant 128 : index
    //CHECK: [[off_x:%.+]] = arith.remui [[l_off_x]], [[c128]] : index
    //CHECK: xegpu.load_matrix [[mdesc]][[[off_y]], [[off_x]]] <{layout = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}>: !xegpu.mem_desc<64x128xf32>, index, index -> vector<32x32xf32>
    %0 = xegpu.create_mem_desc %arg0 : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
    %1 = xegpu.load_matrix %0[0, 0] <{layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32], lane_layout = [2, 8], lane_data = [1, 1]>}>: !xegpu.mem_desc<64x128xf32> -> vector<64x128xf32>
    gpu.return
  }

  //CHECK-LABEL: distribute_store_matrix
  //CHECK-SAME: [[arg0:%.+]]: memref<32768xi8, 3>
  gpu.func @distribute_store_matrix(%arg0 : memref<32768xi8, 3>) {
    //CHECK: [[cst:%.+]] = arith.constant dense<1.000000e+00> : vector<32x32xf32>
    //CHECK: [[mdesc:%.+]] = xegpu.create_mem_desc [[arg0]] : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
    //CHECK: [[sgid:%.+]] = gpu.subgroup_id : index
    //CHECK: [[c4:%.+]] = arith.constant 4 : index
    //CHECK: [[sgidx:%.+]] = arith.remui [[sgid]], [[c4]] : index
    //CHECK: [[sgidy_tmp:%.+]] = arith.divui [[sgid]], [[c4]] : index
    //CHECK: [[c2:%.+]] = arith.constant 2 : index
    //CHECK: [[sgidy:%.+]] = arith.remui [[sgidy_tmp]], [[c2]] : index
    //CHECK: [[c32:%.+]] = arith.constant 32 : index
    //CHECK: [[l_off_y:%.+]] = arith.muli [[sgidy]], [[c32]] : index
    //CHECK: [[c32_0:%.+]] = arith.constant 32 : index
    //CHECK: [[l_off_x:%.+]] = arith.muli [[sgidx]], [[c32_0]] : index
    //CHECK: [[c64:%.+]] = arith.constant 64 : index
    //CHECK: [[off_y:%.+]] = arith.remui [[l_off_y]], [[c64]] : index
    //CHECK: [[c128:%.+]] = arith.constant 128 : index
    //CHECK: [[off_x:%.+]] = arith.remui [[l_off_x]], [[c128]] : index
    //CHECK: xegpu.store_matrix [[cst]], [[mdesc]][[[off_y]], [[off_x]]] : vector<32x32xf32>, !xegpu.mem_desc<64x128xf32>, index, index
    %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>} dense<1.0> : vector<64x128xf32>
    %mdesc = xegpu.create_mem_desc %arg0 : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
    xegpu.store_matrix %cst, %mdesc[0, 0] {layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>} : vector<64x128xf32>, !xegpu.mem_desc<64x128xf32>
    gpu.return
  }

  // CHECK-LABEL: @vector_reduce_dim_0
  gpu.func @vector_reduce_dim_0(%src: memref<4x128xf32>) {
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [1, 32], sg_data = [4, 4]>, dims = [0]>} dense<1.0> : vector<128xf32>
    %tdesc = xegpu.create_nd_tdesc %src : memref<4x128xf32>
      -> !xegpu.tensor_desc<4x128xf32, #xegpu.layout<sg_layout = [1, 32], sg_data = [4, 4]>>
    %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [1, 32], sg_data = [4, 4]>}
      : !xegpu.tensor_desc<4x128xf32, #xegpu.layout<sg_layout = [1, 32], sg_data = [4, 4]>>
      -> vector<4x128xf32>
    // CHECK: vector.multi_reduction <add>, {{.*}}, {{.*}} [0] : vector<4x4xf32> to vector<4xf32>
    %reduce = vector.multi_reduction <add>, %load, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [1, 32], sg_data = [4, 4]>, dims = [0]>} [0]
      : vector<4x128xf32> to vector<128xf32>
    gpu.return
  }

  // CHECK-LABEL: @vector_reduce_dim_1
  gpu.func @vector_reduce_dim_1(%src: memref<256x64xf32>) {
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [16, 1], sg_data = [16, 64]>, dims = [1]>} dense<1.0> : vector<256xf32>
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x64xf32>
      -> !xegpu.tensor_desc<256x64xf32, #xegpu.layout<sg_layout = [16, 1], sg_data = [16, 64]>>
    %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [16, 1], sg_data = [16, 64]>}
      : !xegpu.tensor_desc<256x64xf32, #xegpu.layout<sg_layout = [16, 1], sg_data = [16, 64]>>
      -> vector<256x64xf32>
    // CHECK: vector.multi_reduction <add>, {{.*}}, {{.*}} [1] : vector<16x64xf32> to vector<16xf32>
    %reduce = vector.multi_reduction <add>, %load, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [16, 1], sg_data = [16, 64]>, dims = [1]>} [1]
      : vector<256x64xf32> to vector<256xf32>
    gpu.return
  }

  // CHECK-LABEL: @vector_reduce_4D
   gpu.func @vector_reduce_4D(%src: ui64) {
      %cst_acc = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 2, 6, 1], sg_data = [1, 1, 1, 32]>, dims = [3]>} dense<0.0> : vector<4x2x6xf16>
      %offset = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [4, 2, 6, 1], sg_data = [1, 1, 1, 32]>} dense<0>  : vector<4x2x6x32xindex>
      %mask = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [4, 2, 6, 1], sg_data = [1, 1, 1, 32]>} dense<true> : vector<4x2x6x32xi1>
      %load = xegpu.load %src[%offset], %mask  {layout = #xegpu.layout<sg_layout = [4, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : ui64, vector<4x2x6x32xindex>, vector<4x2x6x32xi1> -> vector<4x2x6x32xf16>
      // CHECK: vector.multi_reduction <add>, {{.*}}, {{.*}} [3] : vector<1x1x1x32xf16> to vector<1x1x1xf16>
      %reduce = vector.multi_reduction <add>, %load, %cst_acc {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 2, 6, 1], sg_data = [1, 1, 1, 32]>, dims = [3]>} [3]
      : vector<4x2x6x32xf16> to vector<4x2x6xf16>
      gpu.return
    }

  // CHECK-LABEL: vector_step_op
  gpu.func @vector_step_op_slice_attr() {
    //CHECK: [[sgId:%.+]] = gpu.subgroup_id : index
    //CHECK: [[c8:%.+]] = arith.constant 8 : index
    //CHECK: [[sgidx:%.+]] = arith.remui [[sgId]], [[c8]] : index
    //CHECK: [[sgidy_tmp:%.+]] = arith.divui [[sgId]], [[c8]] : index
    //CHECK: [[c4:%.+]] = arith.constant 4 : index
    //CHECK: [[sgidy:%.+]] = arith.remui [[sgidy_tmp]], [[c4]] : index
    //CHECK: [[c32:%.+]] = arith.constant 32 : index
    //CHECK: [[LY:%.+]] = arith.muli [[sgidy]], [[c32]] : index
    //CHECK: [[c128:%.+]] = arith.constant 128 : index
    //CHECK: [[MODY:%.+]] = arith.remui [[LY]], [[c128]] : index
    //CHECK: [[BASE:%.+]] = vector.step : vector<32xindex>
    //CHECK: [[CAST:%.+]] = vector.broadcast [[MODY]] : index to vector<32xindex>
    //CHECK: [[ADD:%.+]] = arith.addi [[BASE]], [[CAST]] : vector<32xindex>
    %step = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8], sg_data = [32, 32]>, dims = [1]>}: vector<128xindex>
    gpu.return
  }

  gpu.func @vector_step_op_layout_attr() {
    //CHECK: [[sgId:%.+]] = gpu.subgroup_id : index
    //CHECK: [[c16:%.+]] = arith.constant 16 : index
    //CHECK: [[sgidx:%.+]] = arith.remui [[sgId]], [[c16]] : index
    //CHECK: [[c8:%.+]] = arith.constant 8 : index
    //CHECK: [[LOCALY:%.+]] = arith.muli [[sgidx]], [[c8]] : index
    //CHECK: [[c128:%.+]] = arith.constant 128 : index
    //CHECK: [[MODY:%.+]] = arith.remui [[LOCALY]], [[c128]] : index
    //CHECK: [[BASE:%.+]] = vector.step : vector<8xindex>
    //CHECK: [[CAST:%.+]] = vector.broadcast [[MODY]] : index to vector<8xindex>
    //CHECK: [[ADD:%.+]] = arith.addi [[BASE]], [[CAST]] : vector<8xindex>
    %step = vector.step {layout_result_0 = #xegpu.layout<sg_layout = [16], sg_data = [8]>}: vector<128xindex>
    gpu.return
  }

  // CHECK-LABEL: constant_with_slice_attr
  gpu.func @constant_with_slice_attr() {
    //CHECK: [[cst:%.+]] = arith.constant dense<10> : vector<1xindex>
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 2, 6, 1], sg_data = [1, 1, 1, 1]>, dims = [1, 2, 3]>} dense<10> : vector<4xindex>
    gpu.return
  }

  // CHECK-LABEL: vector_shape_cast
  gpu.func @vector_shape_cast() {
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 4], sg_data = [1, 1, 1, 32]>, dims = [0, 1, 2]>} dense<10> : vector<128xindex>
    %step = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 4], sg_data = [1, 1, 1, 32]>, dims = [0, 1, 2]>} : vector<128xindex>
    %muli = arith.muli %cst, %step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 4], sg_data = [1, 1, 1, 32]>, dims = [0, 1, 2]>} : vector<128xindex>
    //CHECK: vector.shape_cast {{.*}} : vector<32xindex> to vector<1x1x1x32xindex>
    %shape_cast = vector.shape_cast %muli {layout_result_0 = #xegpu.layout<sg_layout = [8, 1, 1, 4], sg_data = [1, 1, 1, 32]>, layout_operand_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 4], sg_data = [1, 1, 1, 32]>, dims = [0, 1, 2]>} : vector<128xindex> to vector<1x1x1x128xindex>
    gpu.return
  }

  // CHECK-LABEL: vector_broadcast
  gpu.func @vector_broadcast(%arg0: index, %arg1: index) {
    %muli = arith.muli %arg0, %arg1 : index
    // CHECK: vector.broadcast {{.*}} : index to vector<1x1x1x32xindex>
    %broadcast = vector.broadcast %muli {layout_result_0 = #xegpu.layout<sg_layout = [4, 2, 6, 1], sg_data = [1, 1, 1, 32]>} : index to vector<4x2x6x32xindex>
    gpu.return
  }

  // CHECK-LABEL: vector_transpose
  gpu.func @vector_transpose(%src: memref<256x32xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x32xf32>
        -> !xegpu.tensor_desc<256x32xf32, #xegpu.layout<sg_layout = [4, 8], sg_data = [64, 32], lane_layout = [16, 1], lane_data = [1, 1], order =[0, 1]>>
    %load = xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [4, 8], sg_data = [64, 32], lane_layout = [16, 1], lane_data = [1, 1], order =[0, 1]>}
        : !xegpu.tensor_desc<256x32xf32, #xegpu.layout<sg_layout = [4, 8], sg_data = [64, 32], lane_layout = [16, 1], lane_data = [1, 1], order =[0, 1]>>
        -> vector<256x32xf32>
    //CHECK: vector.transpose {{.*}}, [1, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1], order = [1, 0]>} : vector<64x32xf32> to vector<32x64xf32>
    %trans = vector.transpose %load, [1, 0] {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], lane_layout = [1, 16], lane_data = [1, 1], order =[1, 0]>} : vector<256x32xf32> to vector<32x256xf32>
      gpu.return
  }

  // CHECK-LABEL: non_splat_constant_2D
  gpu.func @non_splat_constant_2D() {
    // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : vector<1x1xindex>
    // CHECK-DAG: %[[T0:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[T1:.*]] = arith.remui %[[T0]], %[[C32:.*]] : index
    // CHECK-DAG: %[[T2:.*]] = arith.remui %[[T1]], %[[C32_4:.*]] : index
    // CHECK-DAG: %[[T3:.*]] = arith.muli %[[T2]], %[[C16:.*]] : index
    // CHECK-DAG: %[[T4:.*]] = arith.addi %[[C0_8:.*]], %[[T3]] : index
    // CHECK-DAG: %[[T5:.*]] = arith.muli %[[C0_6:.*]], %[[C0_7:.*]] : index
    // CHECK-DAG: %[[T6:.*]] = arith.addi %[[T4]], %[[T5]] : index
    // CHECK-DAG: %[[T7:.*]] = vector.broadcast %[[T6]] : index to vector<1x1xindex>
    // CHECK-DAG: %[[T8:.*]] = arith.addi %[[CST]], %[[T7]] : vector<1x1xindex>
    %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>} dense<[[0], [16], [32], [48], [64], [80], [96], [112], [128], [144], [160], [176], [192], [208], [224], [240], [256], [272], [288], [304], [320], [336], [352], [368], [384], [400], [416], [432], [448], [464], [480], [496]]> : vector<32x1xindex>
    gpu.return
  }

  // CHECK-LABEL: non_splat_constant_2D_non_unit_dim
  gpu.func @non_splat_constant_2D_non_unit_dim() {
    // CHECK-DAG: %[[BASECST:.*]] = arith.constant dense<{{\[}}{{\[}}0, 16{{\]}}, {{\[}}8, 24{{\]}}{{\]}}> : vector<2x2xindex>
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[SGIDX:.*]] = arith.remui %[[SGID]], %{{.*}}
    // CHECK-DAG: %[[SGIDY_TMP:.*]] = arith.divui %[[SGID]], %{{.*}}
    // CHECK-DAG: %[[SGIDY:.*]] = arith.remui %[[SGIDY_TMP]], %{{.*}}
    // CHECK-DAG: %[[MULY:.*]] = arith.muli %[[SGIDY]], %[[C2:.*]] : index
    // CHECK-DAG: %[[MULX:.*]] = arith.muli %[[SGIDX]], %{{.*}} : index
    // CHECK-DAG: %[[REMU_Y:.*]] = arith.remui %[[MULY]], %[[C8:.*]] : index
    // CHECK-DAG: %[[REMU_X:.*]] = arith.remui %[[MULX]], %{{.*}} : index
    // CHECK-DAG: %[[MUL5:.*]] = arith.muli %[[REMU_Y]], %{{.*}} : index
    // CHECK-DAG: %[[ADD:.*]] = arith.addi %[[C0:.*]], %[[MUL5]] : index
    // CHECK-DAG: %[[MUL6:.*]] = arith.muli %[[REMU_X]], %[[C16:.*]] : index
    // CHECK-DAG: %[[ADDSTRIDES:.*]] = arith.addi %[[ADD]], %[[MUL6]] : index
    // CHECK-DAG: %[[BCAST:.*]] = vector.broadcast %[[ADDSTRIDES]] : index to vector<2x2xindex>
    // CHECK-DAG: %[[ADDCST:.*]] = arith.addi %[[BASECST]], %[[BCAST]] : vector<2x2xindex>
    %cst_8x8 = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2]>} dense<[
         [0, 16, 32, 48, 64, 80, 96, 112],
         [8, 24, 40, 56, 72, 88, 104, 120],
         [16, 32, 48, 64, 80, 96, 112, 128],
         [24, 40, 56, 72, 88, 104, 120, 136],
         [32, 48, 64, 80, 96, 112, 128, 144],
         [40, 56, 72, 88, 104, 120, 136, 152],
         [48, 64, 80, 96, 112, 128, 144, 160],
         [56, 72, 88, 104, 120, 136, 152, 168]
      ]> : vector<8x8xindex>
      gpu.return
  }

  // CHECK-LABEL: non_splat_constant
  gpu.func @non_splat_constant() {
    // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : vector<1xindex>
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[REMU:.*]] = arith.remui %[[SGID]], %{{.*}}
    // CHECK-DAG: %[[REMU2:.*]] = arith.remui %[[REMU]], %{{.*}}
    // CHECK-DAG: %[[MUL:.*]] = arith.muli %[[REMU2]], %[[C16:.*]] : index
    // CHECK-DAG: %[[ADDSTRIDES:.*]] = arith.addi %[[C0:.*]], %[[MUL]] : index
    // CHECK-DAG: %[[BCAST:.*]] = vector.broadcast %[[ADDSTRIDES]] : index to vector<1xindex>
    // CHECK-DAG: %[[ADD:.*]] = arith.addi %[[CST]], %[[BCAST]] : vector<1xindex>
    %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [32], sg_data = [1]>} dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496]> : vector<32xindex>
    // CHECK: arith.constant dense<{{\[}}{{\[}}0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15{{\]}}{{\]}}> : vector<1x16xindex>
    %cst_1 = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [32, 1], sg_data = [1, 16]>} dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : vector<1x16xindex>
    gpu.return
  }

  // CHECK-LABEL: scalar_broadcast
  gpu.func @scalar_broadcast(%arg0: index) {
    // CHECK: vector.broadcast {{.*}} : index to vector<1x1x1xindex>
    %broadcast = vector.broadcast %arg0 {layout_result_0 = #xegpu.layout<sg_layout = [4, 8, 1], sg_data = [1, 1, 1]>} : index to vector<4x1x1xindex>
    gpu.return
  }

  // CHECK-LABEL: vector_mask_1D
  gpu.func @vector_mask_1D() {
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[REMU:.*]] = arith.remui %[[SGID]], %[[C2:.*]]
    // CHECK-DAG: %[[MUL:.*]] = arith.muli %[[REMU]], %[[C16:.*]] : index
    // CHECK-DAG: %[[REMU2:.*]] = arith.remui %[[MUL]], %[[C32:.*]] : index
    // CHECK-DAG: %[[SUB:.*]] = arith.subi %[[C8:.*]], %[[REMU2]] : index
    // CHECK-DAG: %[[MAX:.*]] = arith.maxsi %[[SUB]], %[[C0:.*]] : index
    // CHECK-DAG: %[[MIN:.*]] = arith.minsi %[[MAX]], %[[C16:.*]] : index
    // CHECK-DAG: %[[MASK:.*]] = vector.create_mask %[[MIN]] : vector<16xi1>
    %constant_mask = vector.constant_mask [8] {layout_result_0 = #xegpu.layout<sg_layout = [2], sg_data = [16]>} : vector<32xi1>
    gpu.return
  }

  // CHECK-LABEL: vector_mask_2D
  gpu.func @vector_mask_2D() {
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[SGIDX:.*]] = arith.remui %[[SGID]], %[[C4:.*]]
    // CHECK-DAG: %[[SGIDY_TMP:.*]] = arith.divui %[[SGID]], %[[C4:.*]]
    // CHECK-DAG: %[[SGIDY:.*]] = arith.remui %[[SGIDY_TMP]], %[[C8:.*]]
    // CHECK-DAG: %[[ROW:.*]] = arith.muli %[[SGIDY]], %[[C32:.*]] : index
    // CHECK-DAG: %[[COL:.*]] = arith.muli %[[SGIDX]], %[[C32:.*]] : index
    // CHECK-DAG: %[[MODROW:.*]] = arith.remui %[[ROW]], %[[C256:.*]] : index
    // CHECK-DAG: %[[MODCOL:.*]] = arith.remui %[[COL]], %[[C128:.*]] : index
    // CHECK-DAG: %[[SUBROW:.*]] = arith.subi %[[C16:.*]], %[[MODROW]] : index
    // CHECK-DAG: %[[MAXROW:.*]] = arith.maxsi %[[SUBROW]], %[[C4:.*]] : index
    // CHECK-DAG: %[[MINROW:.*]] = arith.minsi %[[MAXROW]], %[[C32:.*]] : index
    // CHECK-DAG: %[[SUBCOL:.*]] = arith.subi %[[C16:.*]], %[[MODCOL]] : index
    // CHECK-DAG: %[[MAXCOL:.*]] = arith.maxsi %[[SUBCOL]], %[[C7:.*]] : index
    // CHECK-DAG: %[[MINCOL:.*]] = arith.minsi %[[MAXCOL]], %[[C32:.*]] : index
    // CHECK-DAG: %[[MASK:.*]] = vector.create_mask %[[MINROW]], %[[MINCOL]] : vector<32x32xi1>
    %constant_mask = vector.constant_mask [16, 16] {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x128xi1>
    gpu.return
  }

  // CHECK-LABEL: vector_create_mask_1D
  gpu.func @vector_create_mask_1D() {
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[REMU:.*]] = arith.remui %[[SGID]], %[[C2:.*]]
    // CHECK-DAG: %[[MUL:.*]] = arith.muli %[[REMU]], %[[C16:.*]]
    // CHECK-DAG: %[[REMU2:.*]] = arith.remui %[[MUL]], %[[C32:.*]]
    // CHECK-DAG: %[[SUB:.*]] = arith.subi %[[C8:.*]], %[[REMU2]] : index
    // CHECK-DAG: %[[MAX:.*]] = arith.maxsi %[[SUB]], %[[C0:.*]] : index
    // CHECK-DAG: %[[MIN:.*]] = arith.minsi %[[MAX]], %[[C16:.*]] : index
    // CHECK-DAG: %[[MASK:.*]] = vector.create_mask %[[MIN]] : vector<16xi1>
    %cst8 = arith.constant 8 : index
    %constant_mask = vector.create_mask %cst8 {layout_result_0 = #xegpu.layout<sg_layout = [2], sg_data = [16]>} : vector<32xi1>
    gpu.return
  }

  // CHECK-LABEL: vector_create_mask_2D
  gpu.func @vector_create_mask_2D() {
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[SGIDX:.*]] = arith.remui %[[SGID]], %[[C4:.*]]
    // CHECK-DAG: %[[SGIDY_TMP:.*]] = arith.divui %[[SGID]], %[[C4:.*]]
    // CHECK-DAG: %[[SGIDY:.*]] = arith.remui %[[SGIDY_TMP]], %[[C8:.*]]
    // CHECK-DAG: %[[ROW:.*]] = arith.muli %[[SGIDY]], %[[C32:.*]]
    // CHECK-DAG: %[[COL:.*]] = arith.muli %[[SGIDX]], %[[C32:.*]]
    // CHECK-DAG: %[[MODROW:.*]] = arith.remui %[[ROW]], %[[C256:.*]]
    // CHECK-DAG: %[[MODCOL:.*]] = arith.remui %[[COL]], %[[C128:.*]]
    // CHECK-DAG: %[[SUBROW:.*]] = arith.subi %[[C16:.*]], %[[MODROW]] : index
    // CHECK-DAG: %[[MAXROW:.*]] = arith.maxsi %[[SUBROW]], %[[C0:.*]] : index
    // CHECK-DAG: %[[MINROW:.*]] = arith.minsi %[[MAXROW]], %[[C32:.*]] : index
    // CHECK-DAG: %[[SUBCOL:.*]] = arith.subi %[[C16:.*]], %[[MODCOL]] : index
    // CHECK-DAG: %[[MAXCOL:.*]] = arith.maxsi %[[SUBCOL]], %[[C0:.*]] : index
    // CHECK-DAG: %[[MINCOL:.*]] = arith.minsi %[[MAXCOL]], %[[C32:.*]] : index
    // CHECK-DAG: %[[MASK:.*]] = vector.create_mask %[[MINROW]], %[[MINCOL]] : vector<32x32xi1>
    %cst16 = arith.constant 16 : index
    %constant_mask = vector.create_mask %cst16, %cst16 {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x128xi1>
    gpu.return
  }

  // CHECK-LABEL: distribute_load_slice_attr
  gpu.func @distribute_load_slice_attr() {
    %2 = memref.alloca() {alignment = 1024} : memref<4096xf32>
    %offset =  arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [8], sg_data = [32], inst_data = [16]> } dense<0> : vector<256xindex>
    %mask = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [8], sg_data = [32], inst_data = [16]> } dense<1> : vector<256xi1>

    // CHECK: %[[LOAD:.*]] = xegpu.load {{.*}} <{chunk_size = 1 : i64, layout = #xegpu.slice<#xegpu.layout<inst_data = [8, 16]>, dims = [0]>}>
    // CHECK-SAME: memref<4096xf32>, vector<32xindex>, vector<32xi1> -> vector<32xf32>
    %3 = xegpu.load %2[%offset], %mask {chunk_size = 1, layout = #xegpu.slice<#xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [8, 16]>, dims = [0]> } : memref<4096xf32>, vector<256xindex>, vector<256xi1> -> vector<256xf32>

    // CHECK: %[[BROADCAST:.*]] = vector.broadcast %[[LOAD]] {layout_result_0 = #xegpu.layout<inst_data = [8, 16]>} : vector<32xf32> to vector<32x32xf32>
    %4 = vector.broadcast %3 {layout_result_0 =
        #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [8, 16]>} : vector<256xf32> to vector<256x256xf32>
    gpu.return
  }

  // CHECK-LABEL: gpu.func @vector_reduce_cross_sg_dim_1
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>)
  gpu.func @vector_reduce_cross_sg_dim_1(%src: memref<?xf32>) {
    // CHECK-DAG: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<1x32xf32>
    // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0> : vector<1x1x32xindex>
    // CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<true> : vector<1x1x32xi1>
    // CHECK-DAG: %[[LOAD:.*]] = xegpu.load %{{.*}}[%[[CST_0]]], %[[CST_1]] <{chunk_size = 1 : i64}> : memref<?xf32>, vector<1x1x32xindex>, vector<1x1x32xi1> -> vector<1x1x32xf32>
    // CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<0.000000e+00> : vector<1x32xf32>
    // CHECK-DAG: %[[LOCAL_REDUCE:.*]] = vector.multi_reduction <add>, %[[LOAD]], %[[CST_2]] [1] : vector<1x1x32xf32> to vector<1x32xf32>
    // CHECK-DAG: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[LOCAL_REDUCE]] : vector<1x32xf32> to vector<1x32xf32>
    // CHECK-DAG: %[[ALLOCA:.*]] = memref.alloca() : memref<4096xi8, 3>
    // CHECK-DAG: %[[MEM_DESC:.*]] = xegpu.create_mem_desc %[[ALLOCA]] : memref<4096xi8, 3> -> !xegpu.mem_desc<32x32xf32>
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[AFFINE1:.*]] = affine.apply #map()[%[[SGID]]]
    // CHECK-DAG: %[[AFFINE2:.*]] = affine.apply #map1()[%[[SGID]]]
    // CHECK-DAG: %[[AFFINE3:.*]] = affine.apply #map2()[%[[SGID]]]
    // CHECK-DAG: %[[MUL1:.*]] = arith.muli {{.*}}, %[[C1:.*]] : index
    // CHECK-DAG: %[[ROW_OFFSET:.*]] = arith.addi %[[C0:.*]], %[[MUL1]] : index
    // CHECK-DAG: %[[MUL2:.*]] = arith.muli %[[AFFINE1]], %[[C1:.*]] : index
    // CHECK-DAG: %[[ADD1:.*]] = arith.addi %[[C0:.*]], %[[MUL2]] : index
    // CHECK-DAG: %[[MUL3:.*]] = arith.muli %[[AFFINE3]], %[[C1:.*]] : index
    // CHECK-DAG: %[[ADD2:.*]] = arith.addi %[[ADD1]], %[[MUL3]] : index
    // CHECK-DAG: %[[COL_OFFSET:.*]] = arith.muli %[[ADD2]], %[[C32:.*]] : index
    // CHECK-DAG: xegpu.store_matrix %[[SHAPE_CAST]], %[[MEM_DESC]][%[[ROW_OFFSET]], %[[COL_OFFSET]]] <{layout = #xegpu.slice<#xegpu.layout<>, dims = [1]>}>: vector<1x32xf32>, !xegpu.mem_desc<32x32xf32>, index, index
    // CHECK-DAG: gpu.barrier
    // CHECK-DAG: %[[LOAD_SLM:.*]] = xegpu.load_matrix %[[MEM_DESC]][%[[C0:.*]], %[[COL_OFFSET]]] : !xegpu.mem_desc<32x32xf32>, index, index -> vector<32x32xf32>
    // CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<0.000000e+00> : vector<32xf32>
    // CHECK-DAG: %[[FINAL_REDUCE:.*]] = vector.multi_reduction <add>, %[[LOAD_SLM]], %[[CST_3]] [0] : vector<32x32xf32> to vector<32xf32>
    // CHECK-DAG: %[[SHAPE_CAST_FINAL:.*]] = vector.shape_cast %[[CST]] : vector<1x32xf32> to vector<32xf32>
    // CHECK-DAG: %{{.*}} = arith.addf %[[FINAL_REDUCE]], %[[SHAPE_CAST_FINAL]] : vector<32xf32>
    // CHECK-DAG: gpu.return
    %cst_3 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [1, 32, 1], sg_data = [1, 1, 32]>, dims = [1]>} dense<1.0> : vector<1x32xf32>
    %offset = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [1, 32, 1], sg_data = [1, 1, 32]>} dense<0> : vector<1x32x32xindex>
    %mask = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [1, 32, 1], sg_data = [1, 1, 32]>} dense<true> : vector<1x32x32xi1>
    %14 = xegpu.load %src[%offset], %mask {chunk_size = 1, layout = #xegpu.layout<sg_layout = [1, 32, 1], sg_data = [1, 1, 32]>} : memref<?xf32>, vector<1x32x32xindex>, vector<1x32x32xi1> -> vector<1x32x32xf32>
    %15 = vector.multi_reduction <add>, %14, %cst_3 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [1, 32, 1], sg_data = [1, 1, 32]>, dims = [1]>} [1] : vector<1x32x32xf32> to vector<1x32xf32>
    gpu.return
  }

  // CHECK-LABEL: gpu.func @vector_reduce_cross_sg_dim_0
  // CHECK-SAME: (%[[ARG0:.*]]: memref<256x128xf32>)
  gpu.func @vector_reduce_cross_sg_dim_0(%src: memref<256x128xf32>) {
    // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<32xf32>
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[REM1:.*]] = arith.remui %[[SGID]], %[[C4:.*]] : index
    // CHECK-DAG: %[[DIV1:.*]] = arith.divui %[[SGID]], %[[C4:.*]] : index
    // CHECK-DAG: %[[REM2:.*]] = arith.remui %[[DIV1]], %[[C8:.*]] : index
    // CHECK-DAG: %[[MUL1:.*]] = arith.muli %[[REM2]], %[[C32:.*]] : index
    // CHECK-DAG: %[[MUL2:.*]] = arith.muli %[[REM1]], %[[C32:.*]] : index
    // CHECK-DAG: %[[REM3:.*]] = arith.remui %[[MUL1]], %[[C256:.*]] : index
    // CHECK-DAG: %[[REM4:.*]] = arith.remui %[[MUL2]], %[[C128:.*]] : index
    // CHECK-DAG: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[REM3]], %[[REM4]]] : memref<256x128xf32> -> !xegpu.tensor_desc<32x32xf32>
    // CHECK-DAG: %[[LOAD_ND:.*]] = xegpu.load_nd %[[TDESC]] : !xegpu.tensor_desc<32x32xf32> -> vector<32x32xf32>
    // CHECK-DAG: %[[CST_LOCAL:.*]] = arith.constant dense<0.000000e+00> : vector<32xf32>
    // CHECK-DAG: %[[LOCAL_REDUCE:.*]] = vector.multi_reduction <add>, %[[LOAD_ND]], %[[CST_LOCAL]] [0] : vector<32x32xf32> to vector<32xf32>
    // CHECK-DAG: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[LOCAL_REDUCE]] : vector<32xf32> to vector<1x32xf32>
    // CHECK-DAG: %[[ALLOCA:.*]] = memref.alloca() : memref<4096xi8, 3>
    // CHECK-DAG: %[[MEM_DESC:.*]] = xegpu.create_mem_desc %[[ALLOCA]] : memref<4096xi8, 3> -> !xegpu.mem_desc<8x128xf32>
    // CHECK-DAG: %[[SGID2:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[AFFINE1:.*]] = affine.apply #map3()[%[[SGID2]]]
    // CHECK-DAG: %[[AFFINE2:.*]] = affine.apply #map4()[%[[SGID2]]]
    // CHECK-DAG: %[[MUL3:.*]] = arith.muli {{.*}}, %[[C1:.*]] : index
    // CHECK-DAG: %[[ROW_OFFSET:.*]] = arith.addi %[[C0:.*]], %[[MUL3]] : index
    // CHECK-DAG: %[[MUL4:.*]] = arith.muli {{.*}}, %[[C1:.*]] : index
    // CHECK-DAG: %[[ADD1:.*]] = arith.addi %[[C0:.*]], %[[MUL4]] : index
    // CHECK-DAG: %[[COL_OFFSET:.*]] = arith.muli %[[ADD1]], %[[C32:.*]] : index
    // CHECK-DAG: xegpu.store_matrix %[[SHAPE_CAST]], %[[MEM_DESC]][%[[ROW_OFFSET]], %[[COL_OFFSET]]] <{layout = #xegpu.slice<#xegpu.layout<>, dims = [0]>}>: vector<1x32xf32>, !xegpu.mem_desc<8x128xf32>, index, index
    // CHECK-DAG: gpu.barrier
    // CHECK-DAG: %[[LOAD_SLM:.*]] = xegpu.load_matrix %[[MEM_DESC]][%[[C0:.*]], %[[COL_OFFSET]]] : !xegpu.mem_desc<8x128xf32>, index, index -> vector<8x32xf32>
    // CHECK-DAG: %[[CST_CROSS_SG_1:.*]] = arith.constant dense<0.000000e+00> : vector<32xf32>
    // CHECK-DAG: %[[FINAL_REDUCE:.*]] = vector.multi_reduction <add>, %[[LOAD_SLM]], %[[CST_CROSS_SG_1]] [0] : vector<8x32xf32> to vector<32xf32>
    // CHECK-DAG: arith.addf %[[FINAL_REDUCE]], %[[CST:.*]] : vector<32xf32>
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>, dims = [0]>} dense<0.0> : vector<128xf32>
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>>
    %load =  xegpu.load_nd %tdesc
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>>
      -> vector<256x128xf32>
    %reduce = vector.multi_reduction <add>, %load, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>, dims = [0]>} [0]
      : vector<256x128xf32> to vector<128xf32>
    gpu.return
  }

  // CHECK-LABEL: gpu.func @vector_reduce_multi_dim
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>)
  gpu.func @vector_reduce_multi_dim(%src: memref<?xf32>) {
    // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1x1xf32>
    // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0> : vector<1x1x32x32xindex>
    // CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<true> : vector<1x1x32x32xi1>
    // CHECK-DAG: %[[LOAD:.*]] = xegpu.load %{{.*}}[%[[CST_0]]], %[[CST_1]] <{chunk_size = 1 : i64}> : memref<?xf32>, vector<1x1x32x32xindex>, vector<1x1x32x32xi1> -> vector<1x1x32x32xf32>
    // CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<0.000000e+00> : vector<1x1xf32>
    // CHECK-DAG: %[[LOCAL_REDUCE:.*]] = vector.multi_reduction <add>, %[[LOAD]], %[[CST_2]] [2, 3] : vector<1x1x32x32xf32> to vector<1x1xf32>
    // CHECK-DAG: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[LOCAL_REDUCE]] : vector<1x1xf32> to vector<1x1xf32>
    // CHECK-DAG: %[[ALLOCA:.*]] = memref.alloca() : memref<256xi8, 3>
    // CHECK-DAG: %[[MEM_DESC:.*]] = xegpu.create_mem_desc %[[ALLOCA]] : memref<256xi8, 3> -> !xegpu.mem_desc<16x4xf32>
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
    // CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
    // CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
    // CHECK-DAG: %[[AFFINE1:.*]] = affine.apply #map()[%[[SGID]]]
    // CHECK-DAG: %[[AFFINE2:.*]] = affine.apply #map1()[%[[SGID]]]
    // CHECK-DAG: %[[AFFINE3:.*]] = affine.apply #map5()[%[[SGID]]]
    // CHECK-DAG: %[[AFFINE4:.*]] = affine.apply #map6()[%[[SGID]]]
    // CHECK-DAG: %[[AFFINE5:.*]] = affine.apply #map7()[%[[SGID]]]
    // CHECK-DAG: %[[AFFINE6:.*]] = affine.apply #map4()[%[[SGID]]]
    // CHECK-DAG: %[[MUL1:.*]] = arith.muli {{.*}}, %[[C1:.*]] : index
    // CHECK-DAG: %[[ADD1:.*]] = arith.addi %[[C0:.*]], %[[MUL1]] : index
    // CHECK-DAG: %[[MUL2:.*]] = arith.muli {{.*}}, %[[C4:.*]] : index
    // CHECK-DAG: %[[ROW_OFFSET:.*]] = arith.addi %[[ADD1]], %[[MUL2]] : index
    // CHECK-DAG: %[[MUL3:.*]] = arith.muli {{.*}}, %[[C1:.*]] : index
    // CHECK-DAG: %[[ADD2:.*]] = arith.addi %[[C0:.*]], %[[MUL3]] : index
    // CHECK-DAG: %[[MUL4:.*]] = arith.muli {{.*}}, %[[C2:.*]] : index
    // CHECK-DAG: %[[ADD3:.*]] = arith.addi %[[ADD2]], %[[MUL4]] : index
    // CHECK-DAG: %[[COL_OFFSET:.*]] = arith.muli %[[ADD3]], %[[C1:.*]] : index
    // CHECK-DAG: xegpu.store_matrix %[[SHAPE_CAST]], %[[MEM_DESC]][%[[ROW_OFFSET]], %[[COL_OFFSET]]] <{layout = #xegpu.slice<#xegpu.layout<>, dims = [2, 3]>}>: vector<1x1xf32>, !xegpu.mem_desc<16x4xf32>, index, index
    // CHECK-DAG: gpu.barrier
    // CHECK-DAG: %[[LOAD_SLM:.*]] = xegpu.load_matrix %[[MEM_DESC]][%[[C0:.*]], %[[COL_OFFSET]]] : !xegpu.mem_desc<16x4xf32>, index, index -> vector<16x1xf32>
    // CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
    // CHECK-DAG: %[[FINAL_REDUCE:.*]] = vector.multi_reduction <add>, %[[LOAD_SLM]], %[[CST_3]] [0] : vector<16x1xf32> to vector<1xf32>
    // CHECK-DAG: %[[SHAPE_CAST_FINAL:.*]] = vector.shape_cast %[[CST]] : vector<1x1xf32> to vector<1xf32>
    // CHECK-DAG: %[[FINAL_ADD:.*]] = arith.addf %[[FINAL_REDUCE]], %[[SHAPE_CAST_FINAL]] : vector<1xf32>
    // CHECK-DAG: gpu.return
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 4, 4], sg_data = [1, 1, 32, 32]>, dims = [2, 3]>} dense<0.0> : vector<2x2xf32>
    %offset = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 4, 4], sg_data = [1, 1, 32, 32]>} dense<0> : vector<2x2x128x128xindex>
    %mask = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 4, 4], sg_data = [1, 1, 32, 32]>} dense<true> : vector<2x2x128x128xi1>
    %load = xegpu.load %src[%offset], %mask {chunk_size = 1, layout = #xegpu.layout<sg_layout = [2, 2, 4, 4], sg_data = [1, 1, 32, 32]>} : memref<?xf32>, vector<2x2x128x128xindex>, vector<2x2x128x128xi1> -> vector<2x2x128x128xf32>
    %reduce = vector.multi_reduction <add>, %load, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 4, 4], sg_data = [1, 1, 32, 32]>, dims = [2, 3]>} [2, 3] : vector<2x2x128x128xf32> to vector<2x2xf32>
    gpu.return
  }

  // CHECK-LABEL: gpu.func @vector_reduce_multi_dim_nou_unit_local_reduction
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>)
  gpu.func @vector_reduce_multi_dim_nou_unit_local_reduction(%src: memref<?xf32>) {
    // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16x16xf32>
    // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0> : vector<16x16x32x32xindex>
    // CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<true> : vector<16x16x32x32xi1>
    // CHECK-DAG: %[[LOAD:.*]] = xegpu.load %[[ARG0]][%[[CST_0]]], %[[CST_1]] <{chunk_size = 1 : i64}> : memref<?xf32>, vector<16x16x32x32xindex>, vector<16x16x32x32xi1> -> vector<16x16x32x32xf32>
    // CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<0.000000e+00> : vector<16x16xf32>
    // CHECK-DAG: %[[LOCAL_REDUCE:.*]] = vector.multi_reduction <add>, %[[LOAD]], %[[CST_2]] [2, 3] : vector<16x16x32x32xf32> to vector<16x16xf32>
    // CHECK-DAG: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[LOCAL_REDUCE]] : vector<16x16xf32> to vector<1x256xf32>
    // CHECK-DAG: %[[ALLOCA:.*]] = memref.alloca() : memref<65536xi8, 3>
    // CHECK-DAG: %[[MEM_DESC:.*]] = xegpu.create_mem_desc %[[ALLOCA]] : memref<65536xi8, 3> -> !xegpu.mem_desc<16x1024xf32>
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[AFFINE1:.*]] = affine.apply #map()[%[[SGID]]]
    // CHECK-DAG: %[[AFFINE2:.*]] = affine.apply #map1()[%[[SGID]]]
    // CHECK-DAG: %[[AFFINE3:.*]] = affine.apply #map5()[%[[SGID]]]
    // CHECK-DAG: %[[AFFINE4:.*]] = affine.apply #map6()[%[[SGID]]]
    // CHECK-DAG: %[[AFFINE5:.*]] = affine.apply #map7()[%[[SGID]]]
    // CHECK-DAG: %[[AFFINE6:.*]] = affine.apply #map4()[%[[SGID]]]
    // CHECK-DAG: %[[MUL1:.*]] = arith.muli {{.*}}, %[[C1:.*]] : index
    // CHECK-DAG: %[[ADD1:.*]] = arith.addi %[[C0:.*]], %[[MUL1]] : index
    // CHECK-DAG: %[[MUL2:.*]] = arith.muli {{.*}}, %[[C4:.*]] : index
    // CHECK-DAG: %[[ROW_OFFSET:.*]] = arith.addi %[[ADD1]], %[[MUL2]] : index
    // CHECK-DAG: %[[MUL3:.*]] = arith.muli {{.*}}, %[[C1:.*]] : index
    // CHECK-DAG: %[[ADD2:.*]] = arith.addi %[[C0:.*]], %[[MUL3]] : index
    // CHECK-DAG: %[[MUL4:.*]] = arith.muli {{.*}}, %[[C2:.*]] : index
    // CHECK-DAG: %[[ADD3:.*]] = arith.addi %[[ADD2]], %[[MUL4]] : index
    // CHECK-DAG: %[[COL_OFFSET:.*]] = arith.muli %[[ADD3]], %[[C256:.*]] : index
    // CHECK-DAG: xegpu.store_matrix %[[SHAPE_CAST]], %[[MEM_DESC]][%[[ROW_OFFSET]], %[[COL_OFFSET]]] <{layout = #xegpu.slice<#xegpu.layout<>, dims = [2, 3]>}>: vector<1x256xf32>, !xegpu.mem_desc<16x1024xf32>, index, index
    // CHECK-DAG: gpu.barrier
    // CHECK-DAG: %[[LOAD_SLM:.*]] = xegpu.load_matrix %[[MEM_DESC]][%[[C0:.*]], %[[COL_OFFSET]]] : !xegpu.mem_desc<16x1024xf32>, index, index -> vector<16x256xf32>
    // CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<0.000000e+00> : vector<256xf32>
    // CHECK-DAG: %[[FINAL_REDUCE:.*]] = vector.multi_reduction <add>, %[[LOAD_SLM]], %[[CST_3]] [0] : vector<16x256xf32> to vector<256xf32>
    // CHECK-DAG: %[[SHAPE_CAST_FINAL:.*]] = vector.shape_cast %[[CST]] : vector<16x16xf32> to vector<256xf32>
    // CHECK-DAG: %[[FINAL_ADD:.*]] = arith.addf %[[FINAL_REDUCE]], %[[SHAPE_CAST_FINAL]] : vector<256xf32>
    // CHECK-DAG: gpu.return
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 4, 4], sg_data = [16, 16, 32, 32]>, dims = [2, 3]>} dense<0.0> : vector<32x32xf32>
    %offset = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 4, 4], sg_data = [16, 16, 32, 32]>} dense<0> : vector<32x32x128x128xindex>
    %mask = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 2, 4, 4], sg_data = [16, 16, 32, 32]>} dense<true> : vector<32x32x128x128xi1>
    %load = xegpu.load %src[%offset], %mask {chunk_size = 1, layout = #xegpu.layout<sg_layout = [2, 2, 4, 4], sg_data = [16, 16, 32, 32]>} : memref<?xf32>, vector<32x32x128x128xindex>, vector<32x32x128x128xi1> -> vector<32x32x128x128xf32>
    %reduce = vector.multi_reduction <add>, %load, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 2, 4, 4], sg_data = [16, 16, 32, 32]>, dims = [2, 3]>} [2, 3] : vector<32x32x128x128xf32> to vector<32x32xf32>
    gpu.return
  }

  // CHECK-LABEL: load_nd_tdesc_with_anchor_layout
  gpu.func @load_nd_tdesc_with_anchor_layout(%src: memref<256x128xf32>) {
    //CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<256x128xf32> -> !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK: xegpu.load_nd %[[TDESC]][{{%.*}}, {{%.*}}] <{layout = #xegpu.layout<inst_data = [32, 16], lane_layout = [1, 16], lane_data = [1, 1]>}>
    // CHECK-SAME: : !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<32x32xf32>
    %load =  xegpu.load_nd %tdesc[0, 0] <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [32, 16],lane_layout = [1, 16], lane_data = [1, 1]>}>
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf32>
    gpu.return
  }

  // CHECK-LABEL: distribute_nested_slice
  // CHECK: %[[V0:.*]] = vector.shape_cast %{{.*}} : vector<32x32xf32> to vector<32x1x32x1xf32>
  // CHECK: %[[V1:.*]] = vector.broadcast %[[V0]] : vector<32x1x32x1xf32> to vector<32x16x32x16xf32>
  // CHECK: %[[V2:.*]] = vector.shape_cast %[[V1]] : vector<32x16x32x16xf32> to vector<32x16x32x16x1xf32>
  // CHECK: %[[V3:.*]] = vector.broadcast %[[V2]] : vector<32x16x32x16x1xf32> to vector<32x16x32x16x16xf32>
  // CHECK: %[[V4:.*]] = vector.shape_cast %[[V3]] : vector<32x16x32x16x16xf32> to vector<32x16x1x32x16x16xf32>
  // CHECK: %[[V5:.*]] = vector.broadcast %[[V4]] : vector<32x16x1x32x16x16xf32> to vector<32x16x16x32x16x16xf32>
  gpu.func @distribute_nested_slice(%src: memref<256x256xf32>) {

    %tdesc = xegpu.create_nd_tdesc %src : memref<256x256xf32>
      -> !xegpu.tensor_desc<256x256xf32, #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32]>>

    %load =  xegpu.load_nd %tdesc[0, 0] {layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32]>}
      : !xegpu.tensor_desc<256x256xf32, #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32]>>
      -> vector<256x256xf32>

    %load2 = xegpu.convert_layout %load <{input_layout = #xegpu.layout<sg_layout = [8, 8],  sg_data = [32, 32]>, target_layout = #xegpu.slice<#xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>, dims=[2]>, dims=[4]>, dims=[1, 3]>}> : vector<256x256xf32>

    %scast = vector.shape_cast %load2 {layout_result_0 = #xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>, dims=[2]>, dims=[4]>, layout_operand_0 = #xegpu.slice<#xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>, dims=[2]>, dims=[4]>, dims=[1, 3]>} : vector<256x256xf32> to vector<256x1x256x1xf32>

    %bcast = vector.broadcast %scast {layout_result_0 = #xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>, dims=[2]>, dims=[4]>, layout_operand_0 = #xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>, dims=[2]>, dims=[4]>} : vector<256x1x256x1xf32> to vector<256x16x256x16xf32>

    %scast1 = vector.shape_cast %bcast {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>, dims=[2]>, layout_operand_0 = #xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>, dims=[2]>, dims=[4]>} : vector<256x16x256x16xf32> to vector<256x16x256x16x1xf32>

    %bcast1 = vector.broadcast %scast1 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>, dims=[2]>, layout_operand_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>, dims=[2]>}  : vector<256x16x256x16x1xf32> to vector<256x16x256x16x16xf32>

    %scast2 = vector.shape_cast %bcast1 {layout_result_0 =
        #xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>, layout_operand_0 = #xegpu.slice<#xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>, dims=[2]>} : vector<256x16x256x16x16xf32> to vector<256x16x1x256x16x16xf32>

    %bcast2 = vector.broadcast %scast2 {layout_result_0 =
        #xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>, layout_operand_0 =
        #xegpu.layout<sg_layout = [8, 1, 1, 8, 1, 1], sg_data = [32, 16, 16, 32, 16, 16]>} : vector<256x16x1x256x16x16xf32> to vector<256x16x16x256x16x16xf32>
    gpu.return
  }

}
