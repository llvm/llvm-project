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
      {layout =  #xegpu.layout<sg_layout = [2, 2], sg_data = [12, 12], lane_layout = [2, 2], lane_data = [1, 1]>}
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
      {layout =  #xegpu.layout<sg_layout = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
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
}
