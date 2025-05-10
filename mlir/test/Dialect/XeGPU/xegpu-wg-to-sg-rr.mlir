// RUN: mlir-opt --xegpu-wg-to-sg -split-input-file %s | FileCheck %s

gpu.module @test_round_robin_assignment {
  // CHECK: test_create_nd_tdesc
  // CHECK: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_create_nd_tdesc(%src: memref<24x32xf32>) {
      // CHECK-COUNT-12: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][%{{.*}}, %{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>>
      %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
      gpu.return
    }

  // CHECK: test_load_nd_tdesc
  // CHECK: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_load_nd_tdesc(%src: memref<24x32xf32>) {
      %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
      // CHECK-COUNT-12: %[[LOAD:.*]] = xegpu.load_nd %{{.*}} : !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>> -> vector<2x2xf32>
      %load =  xegpu.load_nd %tdesc: !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>> -> vector<24x32xf32>
      gpu.return
    }

  // CHECK: test_store_nd
  // CHECK: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_store_nd(%src: memref<24x32xf32>) {
      %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
      // CHECK-COUNT-12: xegpu.store_nd %{{.*}}, %{{.*}} : vector<2x2xf32>, !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>>
      %load = xegpu.load_nd %tdesc: !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>> -> vector<24x32xf32>
      xegpu.store_nd %load, %tdesc: vector<24x32xf32>, !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
      gpu.return
  }

  // CHECK: test_update_nd
  // CHECK: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_update_nd(%src: memref<24x32xf32>){
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->  !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    // CHECK-COUNT-12: %[[UPDATE:.*]] = xegpu.update_nd_offset %{{.*}}, [0, 16] : !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>>
    %update = xegpu.update_nd_offset %tdesc, [0, 16] :  !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    gpu.return
  }

  // CHECK: test_dpas
  // CHECK: %[[ARG_0:.*]]: memref<24x32xf32>
  // CHECK: %[[ARG_1:.*]]: memref<32x24xf32>
  // CHECK: %[[ARG_2:.*]]: memref<24x24xf32>
  gpu.func @test_dpas(%a: memref<24x32xf32>, %b: memref<32x24xf32>, %c: memref<24x24xf32>) {
    // CHECK-COUNT-12: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG_0]][%{{.*}},
    // %{{.*}}] : memref<24x32xf32> -> !xegpu.tensor_desc<2x2xf32,
    // #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>> CHECK-COUNT-12:
    // %[[TDESC1:.*]] = xegpu.create_nd_tdesc %[[ARG_1]][%{{.*}}, %{{.*}}] :
    // memref<32x24xf32> -> !xegpu.tensor_desc<2x2xf32,
    // #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>> CHECK-COUNT-9:
    // %[[TDESC2:.*]] = xegpu.create_nd_tdesc %{{.*}}[%{{.*}}, %{{.*}}] :
    // memref<24x24xf32> -> !xegpu.tensor_desc<2x2xf32,
    // #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>> CHECK-COUNT-144:
    // %[[DPAS:.*]] = xegpu.dpas %{{.*}}, %{{.*}} {layout =
    // #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>} :
    // vector<2x2xf32>, vector<2x2xf32> -> vector<2x2xf32>
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<24x32xf32> -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    %load_a =  xegpu.load_nd %tdesc_a: !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>> -> vector<24x32xf32>
    %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<32x24xf32> -> !xegpu.tensor_desc<32x24xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    %load_b =  xegpu.load_nd %tdesc_b:  !xegpu.tensor_desc<32x24xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>> -> vector<32x24xf32>
    %tdesc_c = xegpu.create_nd_tdesc %c[0, 0] : memref<24x24xf32> -> !xegpu.tensor_desc<24x24xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    %dpas = xegpu.dpas %load_a, %load_b {layout =  #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>} : vector<24x32xf32>, vector<32x24xf32> -> vector<24x24xf32>
    gpu.return
  }
}
