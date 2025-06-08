// RUN: mlir-opt --xegpu-wg-to-sg-distribute -split-input-file %s | FileCheck %s

gpu.module @test_round_robin_assignment {
  // CHECK-LABEL: test_create_nd_tdesc
  // CHECK-SAME: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_create_nd_tdesc(%src: memref<24x32xf32>) {
      // CHECK-COUNT-12: xegpu.create_nd_tdesc %[[ARG_0]][%{{.*}}, %{{.*}}] : memref<24x32xf32>
      // CHECK-SAME: -> !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>>
      // CHECK-NOT: xegpu.create_nd_tdesc
      %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32>
        -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
      gpu.return
    }

  // CHECK-LABEL: test_load_nd_tdesc
  // CHECK-SAME: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_load_nd_tdesc(%src: memref<24x32xf32>) {
      %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32>
        -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
      // CHECK-COUNT-12: xegpu.load_nd %{{.*}}
      // CHECK-SAME-COUNT-12: : !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>>
      // CHECK-SAME-COUNT-12: -> vector<2x2xf32>
      // CHECK-NOT: xegpu.load_nd
      %load =  xegpu.load_nd %tdesc
        : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
        -> vector<24x32xf32>
      gpu.return
    }

  // CHECK-LABEL: test_store_nd
  // CHECK-SAME: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_store_nd(%src: memref<24x32xf32>) {
      %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32>
        -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
      // CHECK-COUNT-12: xegpu.store_nd %{{.*}}, %{{.*}}
      // CHECK-SAME-COUNT-12: : vector<2x2xf32>, !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>>
      // CHECK-NOT : xegpu.store_nd
      %load = xegpu.load_nd %tdesc
        : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
        -> vector<24x32xf32>
      xegpu.store_nd %load, %tdesc
        : vector<24x32xf32>, !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
      gpu.return
  }

  // CHECK-LABEL: test_update_nd
  // CHECK-SAME: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_update_nd(%src: memref<24x32xf32>){
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32>
      ->  !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    // CHECK-COUNT-12: xegpu.update_nd_offset %{{.*}}, [0, 16]
    // CHECK-SAME-COUNT-12: : !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>>
    // CHECK-NOT: xegpu.update_nd_offset
    %update = xegpu.update_nd_offset %tdesc, [0, 16]
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    gpu.return
  }

  // CHECK-LABEL: test_dpas
  // CHECK-SAME: (%[[ARG_0:.*]]: memref<8x8xf32>, %[[ARG_1:.*]]: memref<8x8xf32>, %[[ARG_2:.*]]: memref<8x8xf32>)
  gpu.func @test_dpas(%a: memref<8x8xf32>, %b: memref<8x8xf32>, %c: memref<8x8xf32>) {
    // CHECK-COUNT-4: xegpu.create_nd_tdesc %[[ARG_0]][%{{.*}}, %{{.*}}] : memref<8x8xf32>
    // CHECK-SAME-COUNT-4: -> !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>>
    // CHECK-NOT: xegpu.create_nd_tdesc
    // CHECK-COUNT-4: xegpu.create_nd_tdesc %[[ARG_1]][%{{.*}}, %{{.*}}] : memref<8x8xf32>
    // CHECK-SAME-COUNT-4: -> !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>>
    // CHECK-NOT: xegpu.create_nd_tdesc
    // CHECK-COUNT-4:  xegpu.create_nd_tdesc %{{.*}}[%{{.*}}, %{{.*}}] : memref<8x8xf32>
    // CHECK-SAME-COUNT-4: -> !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>>
    // CHECK-NOT: xegpu.create_nd_tdesc
    // CHECK-COUNT-16: xegpu.dpas %{{.*}}, %{{.*}}
    // CHECK-SAME-COUNT-16: {layout = #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>}
    // CHECK-SAME-COUNT-16: : vector<2x2xf32>, vector<2x2xf32> -> vector<2x2xf32>
    // CHECK-NOT: xegpu.dpas
    %tdesc_a = xegpu.create_nd_tdesc %a[0, 0] : memref<8x8xf32>
      -> !xegpu.tensor_desc<8x8xf32, #xegpu.layout<sg_layout = [2, 2], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    %load_a =  xegpu.load_nd %tdesc_a
      : !xegpu.tensor_desc<8x8xf32, #xegpu.layout<sg_layout = [2, 2], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
      -> vector<8x8xf32>
    %tdesc_b = xegpu.create_nd_tdesc %b[0, 0] : memref<8x8xf32>
      -> !xegpu.tensor_desc<8x8xf32, #xegpu.layout<sg_layout = [2, 2], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    %load_b =  xegpu.load_nd %tdesc_b
      : !xegpu.tensor_desc<8x8xf32, #xegpu.layout<sg_layout = [2, 2], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
      -> vector<8x8xf32>
    %tdesc_c = xegpu.create_nd_tdesc %c[0, 0] : memref<8x8xf32>
      -> !xegpu.tensor_desc<8x8xf32, #xegpu.layout<sg_layout = [2, 2], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    %dpas = xegpu.dpas %load_a, %load_b
      {layout =  #xegpu.layout<sg_layout = [2, 2], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>}
      : vector<8x8xf32>, vector<8x8xf32> -> vector<8x8xf32>
    gpu.return
  }

  // CHECK-LABEL: test_prefetch_nd_tdesc
  // CHECK-SAME: %[[ARG_0:.*]]: memref<24x32xf32>
  gpu.func @test_prefetch_nd_tdesc(%src: memref<24x32xf32>) {
    // CHECK-COUNT-12: xegpu.prefetch_nd %{{.*}}
    // CHECK-SAME-COUNT-12 : !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>>
    // CHECK-NOT: xegpu.prefetch_nd
    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32>
      -> !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    xegpu.prefetch_nd %tdesc
      : !xegpu.tensor_desc<24x32xf32, #xegpu.layout<sg_layout = [4, 4], sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
    gpu.return
  }
}
