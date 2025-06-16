// RUN: mlir-opt --xegpu-wg-to-sg-distribute -split-input-file %s | FileCheck %s

//CHECK: #map = affine_map<()[s0] -> (s0 floordiv 4)>
//CHECK: #map1 = affine_map<()[s0] -> (s0 mod 4)>
gpu.module @test_1_1_assignment {
  // CHECK-LABEL: test_create_nd_tdesc
  // CHECK-SAME: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_create_nd_tdesc(%src: memref<24x32xf32>) {
  // CHECK: %[[SGID:.*]] = gpu.subgroup_id
  // CHECK: %[[C12:.*]] = arith.constant 12 : index
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  // CHECK: %[[C8:.*]] = arith.constant 8 : index
  // CHECK: %[[DIV:.*]] = affine.apply #map()[%[[SGID]]]
  // CHECK: %[[REM:.*]] = affine.apply #map1()[%[[SGID]]]
  // CHECK: %[[MUL1:.*]] = index.mul %[[DIV]], %[[C12]]
  // CHECK: %[[MUL2:.*]] = index.mul %[[REM]], %[[C8]]
  // CHECK: %[[C24:.*]] = arith.constant 24 : index
  // CHECK: %[[MOD:.*]] = index.remu %[[MUL1]], %[[C24]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[ADD1:.*]] = index.add %[[MOD]], %[[C0]]
  // CHECK: %[[C32:.*]] = arith.constant 32 : index
  // CHECK: %[[MOD1:.*]] = index.remu %[[MUL2]], %[[C32]]
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[ADD2:.*]] = index.add %[[MOD1]], %[[C0_1]]
  // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][%[[ADD1]], %[[ADD2]]] : memref<24x32xf32>
  // CHECK-SAME: -> !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
  // CHECK: gpu.return
  %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32>
    -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
  gpu.return
  }

  // CHECK-LABEL: test_load_nd_tdesc
  // CHECK-SAME: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_load_nd_tdesc(%src: memref<24x32xf32>) {
    // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][{{%.*}}, {{%.*}}] : memref<24x32xf32>
    // CHECK-SAME: -> !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
    // CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]]
    // CHECK-SAME: : !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
    // CHECK-SAME: -> vector<12x8xf32>
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %load =  xegpu.load_nd %tdesc
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    gpu.return
  }

  // CHECK-LABEL: test_store_nd
  // CHECK-SAME: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_store_nd(%src: memref<24x32xf32>) {
    // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][{{%.*}}, {{%.*}}] : memref<24x32xf32>
    // CHECK-SAME: -> !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
    // CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]]
    // CHECK-SAME: : !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
    // CHECK-SAME: -> vector<12x8xf32>
    // CHECK: xegpu.store_nd %[[LOAD]], %[[TDESC]]
    // CHECK-SAME: : vector<12x8xf32>, !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %load = xegpu.load_nd %tdesc
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    xegpu.store_nd %load, %tdesc
      : vector<24x32xf32>, !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    gpu.return
}

// CHECK-LABEL: test_update_nd
// CHECK-SAME: %[[ARG_0:.*]]: memref<24x32xf32>
gpu.func @test_update_nd(%src: memref<24x32xf32>){
  // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][{{%.*}}, {{%.*}}] : memref<24x32xf32>
  // CHECK-SAME: -> !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
  // CHECK: %[[UPDATE:.*]] = xegpu.update_nd_offset %[[TDESC]], [0, 16]
  // CHECK-SAME: : !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
  %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32>
    -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
  %update = xegpu.update_nd_offset %tdesc, [0, 16]
    : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
  gpu.return
}

// CHECK-LABEL: test_dpas
// CHECK-SAME: %[[ARG_0:.*]]: memref<24x32xf32>
// CHECK-SAME: %[[ARG_1:.*]]: memref<32x24xf32>
gpu.func @test_dpas(%a: memref<24x32xf32>, %b: memref<32x24xf32>) {
    // CHECK: %[[TDESC_A:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][{{%.*}}, {{%.*}}] : memref<24x32xf32>
    // CHECk-SAME: -> !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
    // CHECK: %[[LOAD_A:.*]] = xegpu.load_nd %[[TDESC_A]]
    // CHECK-SAME: : !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
    // CHECK-SAME: -> vector<12x8xf32>
    // CHECK: %[[TDESC_B:.*]] = xegpu.create_nd_tdesc %[[ARG_1]][{{%.*}}, {{%.*}}] : memref<32x24xf32>
    // CHECK-SAME: -> !xegpu.tensor_desc<8x12xf32, #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 1]>>
    // CHECK: %[[LOAD_B:.*]] = xegpu.load_nd %[[TDESC_B]]
    // CHECK-SAME: : !xegpu.tensor_desc<8x12xf32, #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 1]>>
    // CHECK-SAME: -> vector<8x12xf32>
    // CHECK: %[[DPAS:.*]] = xegpu.dpas %[[LOAD_A]], %[[LOAD_B]]
    // CHECK-SAME: {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>}
    // CHECK-SAME: : vector<12x8xf32>, vector<8x12xf32> -> vector<12x12xf32>
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    %load_a =  xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<32x24xf32>
      -> !xegpu.tensor_desc<32x24xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [8, 12], lane_layout = [8, 2], lane_data = [1, 1]>>
    %load_b =  xegpu.load_nd %tdesc_b
      : !xegpu.tensor_desc<32x24xf32, #xegpu.layout<sg_layout = [4, 2], sg_data = [8, 12], lane_layout = [8, 2], lane_data = [1, 1]>>
      -> vector<32x24xf32>
    %dpas = xegpu.dpas %load_a, %load_b
      {layout_result_0 =  #xegpu.layout<sg_layout = [2, 2], sg_data = [12, 12], lane_layout = [2, 2], lane_data = [1, 1]>}
      : vector<24x32xf32>, vector<32x24xf32> -> vector<24x24xf32>
    gpu.return
  }


// CHECK-LABEL: test_dpas_no_sg_data
// CHECK-SAME: %[[ARG_0:.*]]: memref<24x32xf32>
// CHECK-SAME: %[[ARG_1:.*]]: memref<32x24xf32>
gpu.func @test_dpas_no_sg_data(%a: memref<24x32xf32>, %b: memref<32x24xf32>) {
    // CHECK: %[[TDESC_A:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][{{%.*}}, {{%.*}}] : memref<24x32xf32>
    // CHECk-SAME: -> !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
    // CHECK: %[[LOAD_A:.*]] = xegpu.load_nd %[[TDESC_A]]
    // CHECK-SAME: : !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
    // CHECK-SAME: -> vector<12x8xf32>
    // CHECK: %[[TDESC_B:.*]] = xegpu.create_nd_tdesc %[[ARG_1]][{{%.*}}, {{%.*}}] : memref<32x24xf32>
    // CHECK-SAME: -> !xegpu.tensor_desc<8x12xf32, #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 1]>>
    // CHECK: %[[LOAD_B:.*]] = xegpu.load_nd %[[TDESC_B]]
    // CHECK-SAME: : !xegpu.tensor_desc<8x12xf32, #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 1]>>
    // CHECK-SAME: -> vector<8x12xf32>
    // CHECK: %[[DPAS:.*]] = xegpu.dpas %[[LOAD_A]], %[[LOAD_B]]
    // CHECK-SAME: {layout_result_0 =  #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>}
    // CHECK-SAME: : vector<12x8xf32>, vector<8x12xf32> -> vector<12x12xf32>
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], lane_layout = [2, 8], lane_data = [1, 1]>>
    %load_a =  xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], lane_layout = [2, 8], lane_data = [1, 1]>>
      -> vector<24x32xf32>
    %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<32x24xf32>
      -> !xegpu.tensor_desc<32x24xf32, #xegpu.layout<sg_layout = [4, 2], lane_layout = [8, 2], lane_data = [1, 1]>>
    %load_b =  xegpu.load_nd %tdesc_b
      : !xegpu.tensor_desc<32x24xf32, #xegpu.layout<sg_layout = [4, 2], lane_layout = [8, 2], lane_data = [1, 1]>>
      -> vector<32x24xf32>
    %dpas = xegpu.dpas %load_a, %load_b
      {layout_result_0 =  #xegpu.layout<sg_layout = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
      : vector<24x32xf32>, vector<32x24xf32> -> vector<24x24xf32>
    gpu.return
  }

  // CHECK-LABEL: test_prefetch_nd_tdesc
  // CHECK-SAME: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_prefetch_nd_tdesc(%src: memref<24x32xf32>) {
    // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][{{%.*}}, {{%.*}}] : memref<24x32xf32>
    // CHECK-SAME: -> !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
    // CHECK: xegpu.prefetch_nd %[[TDESC]]
    // CHECK-SAME: : !xegpu.tensor_desc<12x8xf32, #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>>
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    xegpu.prefetch_nd %tdesc
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [12, 8], lane_layout = [2, 8], lane_data = [1, 1]>>
    gpu.return
  }

  // CHECK-LABEL: test_dpas_with_no_create_nd_desc
  gpu.func @test_dpas_with_no_create_nd_desc(%a: vector<24x32xf32>, %b: vector<32x24xf32>) {
    // CHECK-NOT: vector<12x12xf32>
    %dpas = xegpu.dpas %a, %b
      {layout =  #xegpu.layout<sg_layout = [2, 2], sg_data = [12, 12], lane_layout = [2, 2], lane_data = [1, 1]>}
      : vector<24x32xf32>, vector<32x24xf32> -> vector<24x24xf32>
    gpu.return
  }

  gpu.func @test_scf_for(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf32>) {
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

  gpu.func @test_scf_while_and_condition(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
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

  gpu.func @test_scf_if(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
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

  gpu.func @test_scf_if_tensor_desc(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
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


}
