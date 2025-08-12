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
    //CHECK: [[C8:%.+]] = arith.constant 8 : index
    //CHECK: [[C4:%.+]] = arith.constant 4 : index
    //CHECK: [[C4_1:%.+]] = arith.constant 4 : index
    //CHECK: [[SGIDY:%.+]] = affine.apply #map()[[[SGID]]]
    //CHECK: [[SGIDX:%.+]] = affine.apply #map1()[[[SGID]]]
    //CHECK: [[C32:%.+]] = arith.constant 32 : index
    //CHECK: [[LY:%.+]] = index.mul [[SGIDY]], [[C32]]
    //CHECK: [[C32_1:%.+]] = arith.constant 32 : index
    //CHECK: [[LX:%.+]] = index.mul [[SGIDX]], [[C32_1]]
    //CHECK: [[C0:%.+]] = arith.constant 0 : index
    //CHECK: [[C0_1:%.+]] = arith.constant 0 : index
    //CHECK: [[UY:%.+]] = arith.addi [[LY]], [[C0]] : index
    //CHECK: [[UX:%.+]] = arith.addi [[LX]], [[C0_1]] : index
    //CHECK: [[C256:%.+]] = arith.constant 256 : index
    //CHECK: [[Y:%.+]] = index.remu [[UY]], [[C256]]
    //CHECK: [[C128:%.+]] = arith.constant 128 : index
    //CHECK: [[X:%.+]] = index.remu [[UX]], [[C128]]
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
    //CHECK: [[C8:%.+]] = arith.constant 8 : index
    //CHECK: [[C4:%.+]] = arith.constant 4 : index
    //CHECK: [[C4_1:%.+]] = arith.constant 4 : index
    //CHECK: [[SGIDY:%.+]] = affine.apply #map()[[[SGID]]]
    //CHECK: [[SGIDX:%.+]] = affine.apply #map1()[[[SGID]]]
    //CHECK: [[C32:%.+]] = arith.constant 32 : index
    //CHECK: [[LY:%.+]] = index.mul [[SGIDY]], [[C32]]
    //CHECK: [[C32_1:%.+]] = arith.constant 32 : index
    //CHECK: [[LX:%.+]] = index.mul [[SGIDX]], [[C32_1]]
    //CHECK: [[C0:%.+]] = arith.constant 0 : index
    //CHECK: [[C0_1:%.+]] = arith.constant 0 : index
    //CHECK: [[UY:%.+]] = arith.addi [[LY]], [[C0]] : index
    //CHECK: [[UX:%.+]] = arith.addi [[LX]], [[C0_1]] : index
    //CHECK: [[C256:%.+]] = arith.constant 256 : index
    //CHECK: [[Y:%.+]] = index.remu [[UY]], [[C256]]
    //CHECK: [[C128:%.+]] = arith.constant 128 : index
    //CHECK: [[X:%.+]] = index.remu [[UX]], [[C128]]
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
    //CHECK: [[C8:%.+]] = arith.constant 8 : index
    //CHECK: [[C4:%.+]] = arith.constant 4 : index
    //CHECK: [[C4_1:%.+]] = arith.constant 4 : index
    //CHECK: [[SGIDY:%.+]] = affine.apply #map()[[[SGID]]]
    //CHECK: [[SGIDX:%.+]] = affine.apply #map1()[[[SGID]]]
    //CHECK: [[C32:%.+]] = arith.constant 32 : index
    //CHECK: [[LY:%.+]] = index.mul [[SGIDY]], [[C32]]
    //CHECK: [[C32_1:%.+]] = arith.constant 32 : index
    //CHECK: [[LX:%.+]] = index.mul [[SGIDX]], [[C32_1]]
    //CHECK: [[C0:%.+]] = arith.constant 0 : index
    //CHECK: [[C0_1:%.+]] = arith.constant 0 : index
    //CHECK: [[UY:%.+]] = arith.addi [[LY]], [[C0]] : index
    //CHECK: [[UX:%.+]] = arith.addi [[LX]], [[C0_1]] : index
    //CHECK: [[C256:%.+]] = arith.constant 256 : index
    //CHECK: [[Y:%.+]] = index.remu [[UY]], [[C256]]
    //CHECK: [[C128:%.+]] = arith.constant 128 : index
    //CHECK: [[X:%.+]] = index.remu [[UX]], [[C128]]
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

  gpu.func @scf_for(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) {
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
    %2 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>>
    %3 = xegpu.load_nd %2 : !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>> -> vector<128x128xf32>
    %4 = xegpu.create_nd_tdesc %arg0[%0, %c0] : memref<1024x1024xf16> -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>>
    %5 = xegpu.create_nd_tdesc %arg1[%c0, %1] : memref<1024x1024xf16> -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>>

    //      CHECK: [[scf:%.+]]:3 = scf.for [[arg3:%.+]] = [[c0]] to [[c1024]] step [[c128]]
    // CHECK-SAME: iter_args([[arg4:%.+]] = {{.*}}, [[arg5:%.+]] = {{.*}}, [[arg6:%.+]] = {{.*}}) ->
    // CHECK-SAME: (!xegpu.tensor_desc<16x128xf16>, !xegpu.tensor_desc<128x16xf16>, vector<16x16xf32>)
    //      CHECK: [[a:%.+]] = xegpu.load_nd [[arg4]] : !xegpu.tensor_desc<16x128xf16> -> vector<16x128xf16>
    //      CHECK: [[b:%.+]] = xegpu.load_nd [[arg5]] : !xegpu.tensor_desc<128x16xf16> -> vector<128x16xf16>
    //      CHECK: [[c:%.+]] = xegpu.dpas [[a]], [[b]], [[arg6]] : vector<16x128xf16>, vector<128x16xf16>, vector<16x16xf32> -> vector<16x16xf32>
    //      CHECK: [[at:%.+]] = xegpu.update_nd_offset [[arg4]], [[[c0]], [[c128]]] : !xegpu.tensor_desc<16x128xf16>
    //      CHECK: [[bt:%.+]] = xegpu.update_nd_offset [[arg5]], [[[c128]], [[c0]]] : !xegpu.tensor_desc<128x16xf16>
    //      CHECK: scf.yield [[at]], [[bt]], [[c]] : !xegpu.tensor_desc<16x128xf16>, !xegpu.tensor_desc<128x16xf16>, vector<16x16xf32>
    %6:3 = scf.for %arg3 = %c0 to %c1024 step %c128 iter_args(%arg4 = %4, %arg5 = %5, %arg6 = %3)
        -> (!xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>>,
            !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>>, vector<128x128xf32>) {
      %8 = xegpu.load_nd %arg4  : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>> -> vector<128x128xf16>
      %9 = xegpu.load_nd %arg5  : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>> -> vector<128x128xf16>
      %10 = xegpu.dpas %8, %9, %arg6 {layout_result_0 = #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>}
                          : vector<128x128xf16>, vector<128x128xf16>, vector<128x128xf32> -> vector<128x128xf32>
      %11 = xegpu.update_nd_offset %arg4, [%c0, %c128] : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>>
      %12 = xegpu.update_nd_offset %arg5, [%c128, %c0] : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>>
      scf.yield %11, %12, %10 : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128]>>,
                                !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16]>>, vector<128x128xf32>
    }
    %7 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32>
            -> !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>>
    xegpu.store_nd %6#2, %7  : vector<128x128xf32>, !xegpu.tensor_desc<128x128xf32, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16]>>
    gpu.return
  }
}
