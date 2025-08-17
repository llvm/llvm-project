// RUN: mlir-opt --xegpu-wg-to-sg-distribute -split-input-file %s | FileCheck %s

#map = affine_map<()[s0] -> (s0 floordiv 4)>
#map1 = affine_map<()[s0] -> (s0 mod 4)>

gpu.module @test_round_robin_assignment {
  // CHECK-LABEL: create_nd_tdesc
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @create_nd_tdesc(%src: memref<256x128xf32>) {
      // CHECK-COUNT-4: xegpu.create_nd_tdesc %[[ARG_0]][%{{.*}}, %{{.*}}] : memref<256x128xf32>
      // CHECK-SAME: -> !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      // CHECK-NOT: xegpu.create_nd_tdesc
      %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
        -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
      gpu.return
    }

  // CHECK-LABEL: create_nd_tdesc_with_shared_data
  // CHECK-SAME: [[ARG_0:%.*]]: memref<256x128xf32>
  gpu.func @create_nd_tdesc_with_shared_data(%src: memref<256x128xf32>) {
    //CHECK: [[sgId:%.+]] = gpu.subgroup_id : index
    //CHECK: [[IdY:%.+]] = affine.apply #map()[[[sgId]]]
    //CHECK: [[IdX:%.+]] = affine.apply #map1()[[[sgId]]]
    //CHECK: [[C16:%.+]] = arith.constant 16 : index
    //CHECK: [[LY:%.+]] = index.mul [[IdY]], [[C16]]
    //CHECK: [[C64:%.+]] = arith.constant 64 : index
    //CHECK: [[LX:%.+]] = index.mul [[IdX]], [[C64]]
    //CHECK: [[C0:%.+]] = arith.constant 0 : index
    //CHECK: [[C0_1:%.+]] = arith.constant 0 : index
    //CHECK: [[ADDY:%.+]] = arith.addi [[LY]], [[C0]] : index
    //CHECK: [[ADDX:%.+]] = arith.addi [[LX]], [[C0_1]] : index
    //CHECK: [[C128:%.+]] = arith.constant 128 : index
    //CHECK: [[offY:%.+]] = index.remu [[ADDY]], [[C128]]
    //CHECK: [[C64_2:%.+]] = arith.constant 64 : index
    //CHECK: [[offX:%.+]] = index.remu [[ADDX]], [[C64_2]]
    //CHECK: xegpu.create_nd_tdesc [[ARG_0]][[[offY]], [[offX]]] : memref<256x128xf32> -> !xegpu.tensor_desc<16x64xf32>
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
      -> !xegpu.tensor_desc<128x64xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 64]>>
    gpu.return
  }

  // CHECK-LABEL: load_nd_tdesc
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @load_nd_tdesc(%src: memref<256x128xf32>) {
      %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
        -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
      // CHECK-COUNT-4: xegpu.load_nd %{{.*}}
      // CHECK-SAME-COUNT-4: : !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      // CHECK-SAME-COUNT-4: -> vector<16x16xf32>
      // CHECK-NOT: xegpu.load_nd
      %load =  xegpu.load_nd %tdesc
        : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
        -> vector<256x128xf32>
      gpu.return
    }

  // CHECK-LABEL: store_nd
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @store_nd(%src: memref<256x128xf32>) {
      %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
        -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
      // CHECK-COUNT-4: xegpu.store_nd %{{.*}}, %{{.*}}
      // CHECK-SAME-COUNT-4: : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      // CHECK-NOT : xegpu.store_nd
      %load = xegpu.load_nd %tdesc
        : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
        -> vector<256x128xf32>
      xegpu.store_nd %load, %tdesc
        : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
      gpu.return
  }

  // CHECK-LABEL: update_nd
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @update_nd(%src: memref<256x128xf32>){
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
      ->  !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK-COUNT-4: xegpu.update_nd_offset %{{.*}}, [0, 16]
    // CHECK-SAME-COUNT-4: : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>>
    // CHECK-NOT: xegpu.update_nd_offset
    %update = xegpu.update_nd_offset %tdesc, [0, 16]
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }

  // CHECK-LABEL: dpas
  // CHECK-SAME: (%[[ARG_0:.*]]: memref<256x128xf16>, %[[ARG_1:.*]]: memref<128x256xf16>)
  gpu.func @dpas(%a: memref<256x128xf16>, %b: memref<128x256xf16>) {
    // CHECK-COUNT-4: xegpu.create_nd_tdesc %[[ARG_0]][%{{.*}}, %{{.*}}] : memref<256x128xf16>
    // CHECK-SAME-COUNT-4: -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK-NOT: xegpu.create_nd_tdesc
    // CHECK-COUNT-4: xegpu.create_nd_tdesc %[[ARG_1]][%{{.*}}, %{{.*}}] : memref<128x256xf16>
    // CHECK-SAME-COUNT-4: -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [4, 8], lane_data = [1, 1]>>
    // CHECK-NOT: xegpu.create_nd_tdesc
    // CHECK-COUNT-16: xegpu.dpas %{{.*}}, %{{.*}}
    // CHECK-SAME-COUNT-16: {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    // CHECK-SAME-COUNT-16: : vector<16x16xf16>, vector<16x16xf16> -> vector<16x16xf32>
    // CHECK-NOT: xegpu.dpas
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<256x128xf16>
      -> !xegpu.tensor_desc<256x128xf16, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    %load_a =  xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<256x128xf16, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
      -> vector<256x128xf16>
    %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<128x256xf16>
      -> !xegpu.tensor_desc<128x256xf16, #xegpu.layout<sg_layout = [4, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [2, 1]>>
    %load_b =  xegpu.load_nd %tdesc_b
      : !xegpu.tensor_desc<128x256xf16, #xegpu.layout<sg_layout = [4, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [2, 1]>>
      -> vector<128x256xf16>
    %dpas = xegpu.dpas %load_a, %load_b
      {layout_result_0 =  #xegpu.layout<sg_layout = [8, 8], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<256x128xf16>, vector<128x256xf16> -> vector<256x256xf32>
    gpu.return
  }

  // CHECK-LABEL: prefetch_nd_tdesc
  // CHECK-SAME: %[[ARG_0:.*]]: memref<256x128xf32>
  gpu.func @prefetch_nd_tdesc(%src: memref<256x128xf32>) {
    // CHECK-COUNT-4: xegpu.prefetch_nd %{{.*}}
    // CHECK-SAME-COUNT-4: !xegpu.tensor_desc<256x128xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK-NOT: xegpu.prefetch_nd
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<256x128xf32>
      -> !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.prefetch_nd %tdesc
      : !xegpu.tensor_desc<256x128xf32, #xegpu.layout<sg_layout = [8, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }

  // CHECK-LABEL: broadcast
  // CHECK-SAME: %[[ARG_0:.*]]: memref<128x1xf32>
  gpu.func @broadcast(%src: memref<128x1xf32>) {
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<128x1xf32>
      -> !xegpu.tensor_desc<128x1xf32, #xegpu.layout<sg_layout = [4, 1], sg_data = [16, 1], lane_layout = [8, 1], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc
      : !xegpu.tensor_desc<128x1xf32, #xegpu.layout<sg_layout = [4, 1], sg_data = [16, 1], lane_layout = [8, 1], lane_data = [1, 1]>>
      -> vector<128x1xf32>
    // CHECK-COUNT-2: vector.broadcast {{.*}}
    // CHECK-SAME-COUNT-2: {layout_result_0 = #xegpu.layout<lane_layout = [8, 1], lane_data = [1, 1]>}
    // CHECK-SAME-COUNT-2: : vector<16x1xf32> to vector<16x32xf32>
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
    %0 = xegpu.create_nd_tdesc %arg0[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    %1 = xegpu.create_nd_tdesc %arg1[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    // CHECK-LABEL: scf.for
    // CHECK-SAME: (!xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>)
    %2:2 = scf.for %arg2 = %c0 to %c1024 step %c256 iter_args(%arg3 = %0, %arg4 = %1)
        -> (!xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>) {
      %3 = xegpu.load_nd %0  : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>> -> vector<256xf32>
      xegpu.store_nd %3, %arg3  : vector<256xf32>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
      %4 = xegpu.update_nd_offset %arg3, [256] : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
      %5 = xegpu.update_nd_offset %arg4, [256] : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
      // CHECK-LABEL: scf.yield
      // CHECK-SAME: !xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>
      scf.yield %4, %5 : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    }
    gpu.return
  }

  gpu.func @scf_while_and_condition(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = xegpu.create_nd_tdesc %arg0[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>> -> vector<256xf32>
    %2 = xegpu.create_nd_tdesc %arg1[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    //CHECK: scf.while ({{.*}}) : (vector<16xf32>, vector<16xf32>, i32) -> (vector<16xf32>, vector<16xf32>, i32)
    %3:2 = scf.while (%arg2 = %1, %arg3 = %c0_i32) : (vector<256xf32>, i32) -> (vector<256xf32>, i32) {
      %4 = arith.cmpi slt, %arg3, %c10_i32 : i32
      //CHECK: scf.condition{{.*}} : vector<16xf32>, vector<16xf32>, i32
      scf.condition(%4) %arg2, %arg3 : vector<256xf32>, i32
    } do {
    // CHECK: ([[arg2:%.+]]: vector<16xf32>, [[arg3:%.+]]: vector<16xf32>, [[arg4:%.+]]: i32)
    ^bb0(%arg2: vector<256xf32>, %arg3: i32):
      xegpu.store_nd %arg2, %2  : vector<256xf32>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
      %4 = arith.addi %arg3, %c1_i32 : i32
      %5 = xegpu.update_nd_offset %0, [256] : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
      %6 = xegpu.load_nd %5  : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>> -> vector<256xf32>
      scf.yield %6, %4 : vector<256xf32>, i32
    }
    gpu.return
  }

  gpu.func @scf_if(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
    %c10 = arith.constant 10 : index
    %0 = gpu.subgroup_id : index
    %1 = xegpu.create_nd_tdesc %arg0[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    %2 = xegpu.create_nd_tdesc %arg1[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    %3 = arith.cmpi eq, %0, %c10 : index
    // CHECK-LABEL: scf.if
    //  CHECK-SAME: (vector<16xf32>, vector<16xf32>)
    %4 = scf.if %3 -> (vector<256xf32>) {
      %5 = xegpu.load_nd %1  : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>> -> vector<256xf32>
      // CHECK-LABEL: scf.yield
      //  CHECK-SAME: vector<16xf32>, vector<16xf32>
      scf.yield %5 : vector<256xf32>
    } else {
      %5 = xegpu.load_nd %2  : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>> -> vector<256xf32>
      // CHECK-LABEL: scf.yield
      //  CHECK-SAME: vector<16xf32>, vector<16xf32>
      scf.yield %5 : vector<256xf32>
    } {layout_result_0 = #xegpu.layout<sg_layout = [8], sg_data = [16]>}
    xegpu.store_nd %4, %1  : vector<256xf32>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    gpu.return
  }

  gpu.func @scf_if_tensor_desc(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
    %c10 = arith.constant 10 : index
    %id = gpu.subgroup_id : index

    %t = xegpu.create_nd_tdesc %arg0[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    %d = xegpu.load_nd %t : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>> -> vector<256xf32>

    %0 = arith.cmpi eq, %id, %c10 : index
    // CHECK-LABEL: scf.if
    //  CHECK-SAME: (!xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>)
    %1 = scf.if %0 -> (!xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>) {
      %2 = xegpu.create_nd_tdesc %arg0[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
      // CHECK-LABEL: scf.yield
      //  CHECK-SAME: !xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>
      scf.yield %2 : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    } else {
      %3 = xegpu.create_nd_tdesc %arg1[0] : memref<1024xf32> -> !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
      // CHECK-LABEL: scf.yield
      //  CHECK-SAME: !xegpu.tensor_desc<16xf32>, !xegpu.tensor_desc<16xf32>
      scf.yield %3 : !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    }
    xegpu.store_nd %d, %1 : vector<256xf32>, !xegpu.tensor_desc<256xf32, #xegpu.layout<sg_layout = [8], sg_data = [16]>>
    gpu.return
  }

  gpu.func @convert_layout_optimal(%arg0: memref<32x64xf32>) {
    %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<32x64xf32> -> !xegpu.tensor_desc<32x64xf32, #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [16, 16]>>
    //CHECK-2: xegpu.load_nd {{.*}}  : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<inst_data = [16, 16]>> -> vector<16x16xf32>
    //CHECK-2: xegpu.convert_layout {{.*}} <{input_layout = #xegpu.layout<inst_data = [16, 16]>, target_layout = #xegpu.layout<inst_data = [8, 16]>}> : vector<16x16xf32>
    %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<32x64xf32, #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [16, 16]>> -> vector<32x64xf32>
    %2 = xegpu.convert_layout %1 <{input_layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [16, 16]>,
                                   target_layout = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [8, 16]>}> : vector<32x64xf32>
    gpu.return
  }
}
