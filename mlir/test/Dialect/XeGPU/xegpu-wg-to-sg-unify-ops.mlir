// RUN: mlir-opt --xegpu-wg-to-sg-distribute -split-input-file %s | FileCheck %s

//CHECK: #map = affine_map<()[s0] -> (s0 floordiv 4)>
//CHECK: #map1 = affine_map<()[s0] -> (s0 mod 4)>
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
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @load_nd_tdesc_with_offset(%src: memref<256x128xf32>) {
    //CHECK: [[SGID:%.+]] = gpu.subgroup_id : index
    //CHECK: [[SGIDY:%.+]] = affine.apply #map()[[[SGID]]]
    //CHECK: [[SGIDX:%.+]] = affine.apply #map1()[[[SGID]]]
    //CHECK: %[[LOAD:.*]] = xegpu.load_nd {{%.*}}[{{%.*}}, {{%.*}}] : !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<32x32xf32>
    %tdesc = xegpu.create_nd_tdesc %src: memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0]
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf32>
    gpu.return
  }

  // CHECK-LABEL: store_nd_with_offsets
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @store_nd_with_offsets(%src: memref<256x128xf32>) {
    //CHECK: [[SGID:%.+]] = gpu.subgroup_id : index
    //CHECK: [[SGIDY:%.+]] = affine.apply #map()[[[SGID]]]
    //CHECK: [[SGIDX:%.+]] = affine.apply #map1()[[[SGID]]]
    //CHECK: xegpu.store_nd %{{.*}}, {{%.*}}[{{%.*}}, {{%.*}}]  : vector<32x32xf32>, !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %tdesc = xegpu.create_nd_tdesc %src: memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0]
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf32>
    xegpu.store_nd %load, %tdesc[0, 0]
      : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
}

  // CHECK-LABEL: prefetch_nd_tdesc_with_offset
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @prefetch_nd_tdesc_with_offset(%src: memref<256x128xf32>) {
    //CHECK: [[SGID:%.+]] = gpu.subgroup_id : index
    //CHECK: [[SGIDY:%.+]] = affine.apply #map()[[[SGID]]]
    //CHECK: [[SGIDX:%.+]] = affine.apply #map1()[[[SGID]]]
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
    // CHECK: %[[DPAS:.*]] = xegpu.dpas %{{.*}}, %{{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16x128xf16>, vector<128x16xf16> -> vector<16x16xf32>
    %tdesc_a = xegpu.create_nd_tdesc %a : memref<128x128xf16>
      -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load_a =  xegpu.load_nd %tdesc_a[0, 0]
      : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<128x128xf16>
    %tdesc_b = xegpu.create_nd_tdesc %b : memref<128x128xf16>
      -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16], lane_layout = [1, 16], lane_data = [2, 1]>>
    %load_b =  xegpu.load_nd %tdesc_b[0, 0]
      : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16], lane_layout = [1, 16], lane_data = [2, 1]>>
      -> vector<128x128xf16>
    %dpas = xegpu.dpas %load_a, %load_b
      {layout_result_0 =  #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<128x128xf16>, vector<128x128xf16> -> vector<128x128xf32>
    gpu.return
  }

  // CHECK-LABEL: dpas_no_sg_data
  gpu.func @dpas_no_sg_data(%a: memref<128x128xf16>, %b: memref<128x128xf16>) {
    // CHECK: %[[DPAS:.*]] = xegpu.dpas %{{.*}}, %{{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1], order = [1, 0]>} : vector<16x16xf16>, vector<16x16xf16> -> vector<16x16xf32>
    %tdesc_a = xegpu.create_nd_tdesc %a : memref<128x128xf16>
      -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [1, 1],
      order = [1, 0]>>
    %load_a =  xegpu.load_nd %tdesc_a[0, 0]
      : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [1, 1],
      order = [1, 0]>>
      -> vector<128x128xf16>
    %tdesc_b = xegpu.create_nd_tdesc %b : memref<128x128xf16>
      -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [2, 1],
      order = [1, 0]>>
    %load_b =  xegpu.load_nd %tdesc_b[0, 0]
      : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [2, 1],
      order = [1, 0]>>
      -> vector<128x128xf16>
    %dpas = xegpu.dpas %load_a, %load_b
      {layout_result_0 =  #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [1, 1], order = [1, 0]>}
      : vector<128x128xf16>, vector<128x128xf16> -> vector<128x128xf32>
    gpu.return
  }

  // CHECK-LABEL: dpas_with_no_create_nd_desc
  gpu.func @dpas_with_no_create_nd_desc(%a: vector<256x128xf32>, %b: vector<128x256xf32>) {
    // CHECK-NOT: vector<32x32xf32>
    %dpas = xegpu.dpas %a, %b
      {layout =  #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<256x128xf32>, vector<128x256xf32> -> vector<256x256xf32>
    gpu.return
  }

  // CHECK-LABEL: broadcast_dim1
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x1xf32>
  gpu.func @broadcast_dim1(%src: memref<256x1xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x1xf32>
      -> !xegpu.tensor_desc<256x1xf32, #xegpu.layout<sg_layout = [8, 1], sg_data = [32, 1], lane_layout = [8, 1], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc[0, 0]
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
    %load =  xegpu.load_nd %tdesc[0, 0]
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
    %5 = xegpu.load_nd %2[%0, %1] : !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>> -> vector<128x128xf32>
    %6 = xegpu.load_nd %3[%0, %c0] : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>> -> vector<128x128xf16>
    %7 = xegpu.load_nd %4[%c0, %1] : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>> -> vector<128x128xf16>
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
      %9 = xegpu.dpas %arg4, %arg5, %arg6 {layout_result_0 = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>}
                          : vector<128x128xf16>, vector<128x128xf16>, vector<128x128xf32> -> vector<128x128xf32>
      %10 = xegpu.load_nd %3[%arg3, %c0] : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>> -> vector<128x128xf16>
      %11 = xegpu.load_nd %4[%c0, %arg3] : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>> -> vector<128x128xf16>
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
        %load =  xegpu.load_nd %tdesc[0, 0]
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
      %load =  xegpu.load_nd %tdesc[0, 0]
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
    %load =  xegpu.load_nd %tdesc[0, 0]
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
        %ld =  xegpu.load_nd %td[0, 0]
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
    %load = xegpu.load %src[%offset], %mask {chunk_size = 1, layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 4]>, l1_hint = #xegpu.cache_hint<cached>}
      : memref<?xf16>, vector<256x16xindex>, vector<256x16xi1> -> vector<256x16xf16>
    gpu.return
  }

  // CHECK-LABEL: @store_scatter
  // CHECK-SAME: %[[ARG0:.*]]: memref<256xf16>
  gpu.func @store_scatter(%dest : memref<256xf16>) {
    // CHECK: %[[VAL:.*]] = arith.constant dense<2.550000e+01> : vector<8xf16>
    // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<8xindex>
    // CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<8xi1>
    // CHECK: xegpu.store %[[VAL]], %[[ARG0]][%[[CST]]], %[[MASK]] <{chunk_size = 1 : i64, l1_hint = #xegpu.cache_hint<cached>}>
    // CHECK-SAME: : vector<8xf16>, memref<256xf16>, vector<8xindex>, vector<8xi1>
    %val = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [32], sg_data = [8]>} dense<25.5> : vector<256xf16>
    %offset = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [32], sg_data = [8]>} dense<0> : vector<256xindex>
    %mask = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [32], sg_data = [8]>} dense<1> : vector<256xi1>
    xegpu.store %val, %dest[%offset], %mask {chunk_size = 1, layout = #xegpu.layout<sg_layout = [32], sg_data = [8]>, l1_hint = #xegpu.cache_hint<cached>}
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
    %load = xegpu.load %src[%offset], %mask {chunk_size = 4, layout_result_0 = #xegpu.layout<sg_layout = [32, 1], sg_data = [8, 4]>, l1_hint = #xegpu.cache_hint<cached>}
      : memref<?xf16>, vector<256xindex>, vector<256xi1> -> vector<256x4xf16>
    gpu.return
  }

  // CHECK-LABEL: distribute_load_matrix
  // CHECK-SAME: [[arg0:%.+]]: memref<32768xi8, 3>
  gpu.func @distribute_load_matrix(%arg0: memref<32768xi8, 3>) {
    //CHECK: [[mdesc:%.+]] = xegpu.create_mem_desc [[arg0]] : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
    //CHECK: [[sgid:%.+]] = gpu.subgroup_id : index
    //CHECK: [[c2:%.+]] = arith.constant 2 : index
    //CHECK: [[c4:%.+]] = arith.constant 4 : index
    //CHECK: [[c4_0:%.+]] = arith.constant 4 : index
    //CHECK: [[id_y:%.+]] = affine.apply #map()[[[sgid]]]
    //CHECK: [[id_x:%.+]] = affine.apply #map1()[[[sgid]]]
    //CHECK: [[c32:%.+]] = arith.constant 32 : index
    //CHECK: [[l_off_y:%.+]] = index.mul [[id_y]], [[c32]]
    //CHECK: [[c32_1:%.+]] = arith.constant 32 : index
    //CHECK: [[l_off_x:%.+]] = index.mul [[id_x]], [[c32_1]]
    //CHECK: [[c0:%.+]] = arith.constant 0 : index
    //CHECK: [[c0_1:%.+]] = arith.constant 0 : index
    //CHECK: [[l_off_y_0:%.+]] = arith.addi [[l_off_y]], [[c0]] : index
    //CHECK: [[l_off_x_0:%.+]] = arith.addi [[l_off_x]], [[c0_1]] : index
    //CHECK: [[c64:%.+]] = arith.constant 64 : index
    //CHECK: [[off_y:%.+]] = index.remu [[l_off_y_0]], [[c64]]
    //CHECK: [[c128:%.+]] = arith.constant 128 : index
    //CHECK: [[off_x:%.+]] = index.remu [[l_off_x_0]], [[c128]]
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
    //CHECK: [[c2:%.+]] = arith.constant 2 : index
    //CHECK: [[c4:%.+]] = arith.constant 4 : index
    //CHECK: [[c4_0:%.+]] = arith.constant 4 : index
    //CHECK: [[id_y:%.+]] = affine.apply #map()[[[sgid]]]
    //CHECK: [[id_x:%.+]] = affine.apply #map1()[[[sgid]]]
    //CHECK: [[c32:%.+]] = arith.constant 32 : index
    //CHECK: [[l_off_y_0:%.+]] = index.mul [[id_y]], [[c32]]
    //CHECK: [[c32_1:%.+]] = arith.constant 32 : index
    //CHECK: [[l_off_x_0:%.+]] = index.mul [[id_x]], [[c32_1]]
    //CHECK: [[c0:%.+]] = arith.constant 0 : index
    //CHECK: [[c0_2:%.+]] = arith.constant 0 : index
    //CHECK: [[l_off_y:%.+]] = arith.addi [[l_off_y_0]], [[c0]] : index
    //CHECK: [[l_off_x:%.+]] = arith.addi [[l_off_x_0]], [[c0_2]] : index
    //CHECK: [[c64:%.+]] = arith.constant 64 : index
    //CHECK: [[off_y:%.+]] = index.remu [[l_off_y]], [[c64]]
    //CHECK: [[c128:%.+]] = arith.constant 128 : index
    //CHECK: [[off_x:%.+]] = index.remu [[l_off_x]], [[c128]]
    //CHECK: xegpu.store_matrix [[cst]], [[mdesc]][[[off_y]], [[off_x]]] : vector<32x32xf32>, !xegpu.mem_desc<64x128xf32>, index, index
    %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>} dense<1.0> : vector<64x128xf32>
    %mdesc = xegpu.create_mem_desc %arg0 : memref<32768xi8, 3> -> !xegpu.mem_desc<64x128xf32>
    xegpu.store_matrix %cst, %mdesc[0, 0] {layout = #xegpu.layout<sg_layout = [2, 4], sg_data = [32, 32]>} : vector<64x128xf32>, !xegpu.mem_desc<64x128xf32>
    gpu.return
  }
}
