// RUN: mlir-opt -int-range-optimizations %s | FileCheck %s
 
gpu.module @test_module {
  gpu.func @test_cluster_ranges() kernel attributes {known_cluster_size = array<i32: 8, 4, 1>} {
    %c0 = gpu.cluster_block_id x
    // CHECK: test.reflect_bounds {smax = 7 : index, smin = 0 : index, umax = 7 : index, umin = 0 : index}
    %c0_0 = test.reflect_bounds %c0 : index
    %c1 = gpu.cluster_block_id y
    // CHECK: test.reflect_bounds {smax = 3 : index, smin = 0 : index, umax = 3 : index, umin = 0 : index}
    %c1_0 = test.reflect_bounds %c1 : index
    %c2 = gpu.cluster_block_id z
    // CHECK: test.reflect_bounds {smax = 0 : index, smin = 0 : index, umax = 0 : index, umin = 0 : index}
    %c2_0 = test.reflect_bounds %c2 : index

    %d0 = gpu.cluster_dim_blocks x
    // CHECK: test.reflect_bounds {smax = 8 : index, smin = 8 : index, umax = 8 : index, umin = 8 : index}
    %d0_0 = test.reflect_bounds %d0 : index
    %d1 = gpu.cluster_dim_blocks y
    // CHECK: test.reflect_bounds {smax = 4 : index, smin = 4 : index, umax = 4 : index, umin = 4 : index}
    %d1_0 = test.reflect_bounds %d1 : index
    %d2 = gpu.cluster_dim_blocks z
    // CHECK: test.reflect_bounds {smax = 1 : index, smin = 1 : index, umax = 1 : index, umin = 1 : index}
    %d2_0 = test.reflect_bounds %d2 : index

    gpu.return
  }
}
