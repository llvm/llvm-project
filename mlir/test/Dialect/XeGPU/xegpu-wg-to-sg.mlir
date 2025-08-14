// RUN: mlir-opt --xegpu-wg-to-sg-distribute -split-input-file %s | FileCheck %s

//CHECK: #map = affine_map<()[s0] -> (s0 floordiv 4)>
//CHECK: #map1 = affine_map<()[s0] -> (s0 mod 4)>
gpu.module @test_1_1_assignment {
  // CHECK-LABEL: create_nd_tdesc
  // CHECK-SAME: [[ARG_0:%.*]]: memref<256x128xf32>
  gpu.func @create_nd_tdesc(%src: memref<256x128xf32>) {
    //CHECK: [[SGID:%.+]] = gpu.subgroup_id : index
    //CHECK: [[SGIDY:%.+]] = affine.apply #map()[[[SGID]]]
    //CHECK: [[SGIDX:%.+]] = affine.apply #map1()[[[SGID]]]
    //CHECK: [[C32:%.+]] = arith.constant 32 : index
    //CHECK: [[LY:%.+]] = index.mul [[SGIDY]], [[C32]]
    //CHECK: [[LX:%.+]] = index.mul [[SGIDX]], [[C32]]
    //CHECK: [[C0:%.+]] = arith.constant 0 : index
    //CHECK: [[C0_1:%.+]] = arith.constant 0 : index
    //CHECK: [[UY:%.+]] = arith.addi [[LY]], [[C0]] : index
    //CHECK: [[UX:%.+]] = arith.addi [[LX]], [[C0_1]] : index
    //CHECK: [[C256:%.+]] = arith.constant 256 : index
    //CHECK: [[Y:%.+]] = index.remu [[UY]], [[C256]]
    //CHECK: [[C128:%.+]] = arith.constant 128 : index
    //CHECK: [[X:%.+]] = index.remu [[UX]], [[C128]]
    //CHECK: [[TDESC:%.+]] = xegpu.create_nd_tdesc [[ARG_0]][[[Y]], [[X]]] : memref<256x128xf32> -> !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }

  // CHECK-LABEL: load_nd_tdesc
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @load_nd_tdesc(%src: memref<256x128xf32>) {
    // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][{{%.*}}, {{%.*}}] : memref<256x128xf32>
    // CHECK-SAME: -> !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]]
    // CHECK-SAME: : !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK-SAME: -> vector<32x32xf32>
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf32>
    gpu.return
  }

  // CHECK-LABEL: store_nd
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @store_nd(%src: memref<256x128xf32>) {
    // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][{{%.*}}, {{%.*}}] : memref<256x128xf32>
    // CHECK-SAME: -> !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]]
    // CHECK-SAME: : !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK-SAME: -> vector<32x32xf32>
    // CHECK: xegpu.store_nd %[[LOAD]], %[[TDESC]]
    // CHECK-SAME: : vector<32x32xf32>, !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load = xegpu.load_nd %tdesc
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf32>
    xegpu.store_nd %load, %tdesc
      : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
}

// CHECK-LABEL: update_nd
// CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
gpu.func @update_nd(%src: memref<256x128xf32>){
  // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][{{%.*}}, {{%.*}}] : memref<256x128xf32>
  // CHECK-SAME: -> !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  // CHECK: %[[UPDATE:.*]] = xegpu.update_nd_offset %[[TDESC]], [0, 16]
  // CHECK-SAME: : !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
    -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
  %update = xegpu.update_nd_offset %tdesc, [0, 16]
    : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}

// CHECK-LABEL: dpas
gpu.func @dpas(%a: memref<128x128xf16>, %b: memref<128x128xf16>) {
    // CHECK: %[[DPAS:.*]] = xegpu.dpas %{{.*}}, %{{.*}} {layout_result_0 =  #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16x128xf16>, vector<128x16xf16> -> vector<16x16xf32>
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<128x128xf16>
      -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load_a =  xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 128], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<128x128xf16>
    %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<128x128xf16>
      -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], sg_data = [128, 16], lane_layout = [1, 16], lane_data = [2, 1]>>
    %load_b =  xegpu.load_nd %tdesc_b
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
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<128x128xf16>
      -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [1, 1],
      order = [1, 0]>>
    %load_a =  xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [1, 1],
      order = [1, 0]>>
      -> vector<128x128xf16>
    %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<128x128xf16>
      -> !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [2, 1],
      order = [1, 0]>>
    %load_b =  xegpu.load_nd %tdesc_b
      : !xegpu.tensor_desc<128x128xf16, #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [2, 1],
      order = [1, 0]>>
      -> vector<128x128xf16>
    %dpas = xegpu.dpas %load_a, %load_b
      {layout_result_0 =  #xegpu.layout<sg_layout = [8, 8], lane_layout = [1, 16], lane_data = [1, 1], order = [1, 0]>}
      : vector<128x128xf16>, vector<128x128xf16> -> vector<128x128xf32>
    gpu.return
  }

  // CHECK-LABEL: prefetch_nd_tdesc
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @prefetch_nd_tdesc(%src: memref<256x128xf32>) {
    // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][{{%.*}}, {{%.*}}] : memref<256x128xf32>
    // CHECK-SAME: -> !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK: xegpu.prefetch_nd %[[TDESC]]
    // CHECK-SAME: : !xegpu.tensor_desc<32x32xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.prefetch_nd %tdesc
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }

  // CHECK-LABEL: dpas_with_no_create_nd_desc
  gpu.func @dpas_with_no_create_nd_desc(%a: vector<256x128xf32>, %b: vector<128x256xf32>) {
    // CHECK-NOT: vector<32x32xf32>
    %dpas = xegpu.dpas %a, %b
      {layout =  #xegpu.layout<sg_layout = [2, 2], sg_data = [12, 12], lane_layout = [2, 2], lane_data = [1, 1]>}
      : vector<256x128xf32>, vector<128x256xf32> -> vector<256x256xf32>
    gpu.return
  }

  // CHECK-LABEL: broadcast_dim1
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x1xf32>
  gpu.func @broadcast_dim1(%src: memref<256x1xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x1xf32>
      -> !xegpu.tensor_desc<256x1xf32, #xegpu.layout<sg_layout = [8, 1], sg_data = [32, 1], lane_layout = [8, 1], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc
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
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<1x128xf32>
      -> !xegpu.tensor_desc<1x128xf32, #xegpu.layout<sg_layout = [1, 4], sg_data = [1, 32], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc
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

  gpu.func @scf_while_and_condition(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = xegpu.create_nd_tdesc %arg0[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
    %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>> -> vector<256xf32>
    %2 = xegpu.create_nd_tdesc %arg1[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>

    // CHECK: scf.while {{.*}} : (vector<16xf32>, i32) -> (vector<16xf32>, i32)
    %3:2 = scf.while (%arg2 = %1, %arg3 = %c0_i32) : (vector<256xf32>, i32) -> (vector<256xf32>, i32) {
      %4 = arith.cmpi slt, %arg3, %c10_i32 : i32
      // CHECK: scf.condition{{.*}} : vector<16xf32>, i32
      scf.condition(%4) %arg2, %arg3 : vector<256xf32>, i32
    } do {
    // CHECK: ([[arg2:%.+]]: vector<16xf32>, [[arg3:%.+]]: i32)
    ^bb0(%arg2: vector<256xf32>, %arg3: i32):
      xegpu.store_nd %arg2, %2  : vector<256xf32>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
      %4 = arith.addi %arg3, %c1_i32 : i32
      %5 = xegpu.update_nd_offset %0, [256] : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
      %6 = xegpu.load_nd %5  : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>> -> vector<256xf32>
      scf.yield %6, %4 : vector<256xf32>, i32
    }
    gpu.return
  }

  gpu.func @scf_if(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
    %c10 = arith.constant 10 : index
    %id = gpu.subgroup_id : index

    %0 = xegpu.create_nd_tdesc %arg0[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
    %1 = xegpu.create_nd_tdesc %arg1[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>

    %4 = arith.cmpi eq, %id, %c10 : index
    // CHECK-LABEL: scf.if
    //  CHECK-SAME: (vector<16xf32>)
    %5 = scf.if %4 -> (vector<256xf32>) {
      // CHECK-LABEL: xegpu.load_nd
      //  CHECK-SAME: !xegpu.tensor_desc<16xf32> -> vector<16xf32>
      %2 = xegpu.load_nd %0 : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>> -> vector<256xf32>
      // CHECK-LABEL: scf.yield
      //  CHECK-SAME: vector<16xf32>
      scf.yield %2 : vector<256xf32>
    } else {
      // CHECK-LABEL: xegpu.load_nd
      //  CHECK-SAME: !xegpu.tensor_desc<16xf32> -> vector<16xf32>
      %3 = xegpu.load_nd %1 : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>> -> vector<256xf32>
      // CHECK-LABEL: scf.yield
      //  CHECK-SAME: vector<16xf32>
      scf.yield %3 : vector<256xf32>
    } {layout_result_0 = #xegpu.layout<sg_layout = [16], sg_data = [16]>}
    xegpu.store_nd %5, %0 : vector<256xf32>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
    gpu.return
  }

  gpu.func @scf_if_tensor_desc(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
    %c10 = arith.constant 10 : index
    %id = gpu.subgroup_id : index

    %t = xegpu.create_nd_tdesc %arg0[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
    %d = xegpu.load_nd %t : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>> -> vector<256xf32>

    %0 = arith.cmpi eq, %id, %c10 : index
    // CHECK-LABEL: scf.if
    //  CHECK-SAME: (!xegpu.tensor_desc<16xf32>)
    %1 = scf.if %0 -> (!xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>) {
      // CHECK-LABEL: xegpu.create_nd_tdesc
      //  CHECK-SAME: memref<1024xf32> -> !xegpu.tensor_desc<16xf32>
      %2 = xegpu.create_nd_tdesc %arg0[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
      // CHECK-LABEL: scf.yield
      //  CHECK-SAME: !xegpu.tensor_desc<16xf32>
      scf.yield %2 : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
    } else {
      // CHECK-LABEL: xegpu.create_nd_tdesc
      //  CHECK-SAME: memref<1024xf32> -> !xegpu.tensor_desc<16xf32>
      %3 = xegpu.create_nd_tdesc %arg1[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
      // CHECK-LABEL: scf.yield
      //  CHECK-SAME: !xegpu.tensor_desc<16xf32>
      scf.yield %3 : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
    }
    xegpu.store_nd %d, %1 : vector<256xf32>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [16], sg_data = [16]>>
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
        %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
          -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [8, 4], lane_data = [1, 1]>>
        %load =  xegpu.load_nd %tdesc
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
      %tdesc = xegpu.create_nd_tdesc %src2[0, 0] : memref<128x64xf32>
        -> !xegpu.tensor_desc<128x64xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], lane_layout = [8, 4], lane_data = [1, 1]>>
      %load =  xegpu.load_nd %tdesc
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
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], lane_layout = [8, 4], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc
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
        %td = xegpu.create_nd_tdesc %src1[0, 0] : memref<128x64xf32>
          -> !xegpu.tensor_desc<128x64xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], lane_layout = [8, 4], lane_data = [1, 1]>>
        %ld =  xegpu.load_nd %td
          : !xegpu.tensor_desc<128x64xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], lane_layout = [8, 4], lane_data = [1, 1]>>
          -> vector<128x64xf32>
        %exp = math.exp %ld {layout_result_0 = #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], lane_layout = [8, 4], lane_data = [1, 1]>} : vector<128x64xf32>
    }
  } {sg_id_range = #xegpu.range<[3, 19]>}
  gpu.return
  }

  // CHECK-LABEL: distribute_constant
  gpu.func @distribute_constant() {
    // CHECK: arith.constant dense<1.000000e+00> : vector<32x32xf32>
    %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>} dense<1.0> : vector<256x128xf32>
    gpu.return
  }
}
