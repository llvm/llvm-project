// RUN: mlir-opt -test-int-range-inference -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @launch_func
func.func @launch_func(%arg0 : index) {
  %0 = test.with_bounds {
    umin = 3 : index, umax = 5 : index,
    smin = 3 : index, smax = 5 : index
  }
  %1 = test.with_bounds {
    umin = 7 : index, umax = 11 : index,
    smin = 7 : index, smax = 11 : index
  }
  gpu.launch blocks(%block_id_x, %block_id_y, %block_id_z) in (%grid_dim_x = %0, %grid_dim_y = %1, %grid_dim_z = %arg0)
      threads(%thread_id_x, %thread_id_y, %thread_id_z) in (%block_dim_x = %arg0, %block_dim_y = %0, %block_dim_z = %1) {

    // CHECK: test.reflect_bounds {smax = 5 : index, smin = 3 : index, umax = 5 : index, umin = 3 : index}
    // CHECK: test.reflect_bounds {smax = 11 : index, smin = 7 : index, umax = 11 : index, umin = 7 : index}
    // CHECK: test.reflect_bounds {smax = 4294967295 : index, smin = 1 : index, umax = 4294967295 : index, umin = 1 : index}
    %grid_dim_x0 = test.reflect_bounds %grid_dim_x
    %grid_dim_y0 = test.reflect_bounds %grid_dim_y
    %grid_dim_z0 = test.reflect_bounds %grid_dim_z

    // CHECK: test.reflect_bounds {smax = 4 : index, smin = 0 : index, umax = 4 : index, umin = 0 : index}
    // CHECK: test.reflect_bounds {smax = 10 : index, smin = 0 : index, umax = 10 : index, umin = 0 : index}
    // CHECK: test.reflect_bounds {smax = 4294967294 : index, smin = 0 : index, umax = 4294967294 : index, umin = 0 : index}
    %block_id_x0 = test.reflect_bounds %block_id_x
    %block_id_y0 = test.reflect_bounds %block_id_y
    %block_id_z0 = test.reflect_bounds %block_id_z

    // CHECK: test.reflect_bounds {smax = 4294967295 : index, smin = 1 : index, umax = 4294967295 : index, umin = 1 : index}
    // CHECK: test.reflect_bounds {smax = 5 : index, smin = 3 : index, umax = 5 : index, umin = 3 : index}
    // CHECK: test.reflect_bounds {smax = 11 : index, smin = 7 : index, umax = 11 : index, umin = 7 : index}
    %block_dim_x0 = test.reflect_bounds %block_dim_x
    %block_dim_y0 = test.reflect_bounds %block_dim_y
    %block_dim_z0 = test.reflect_bounds %block_dim_z

    // CHECK: test.reflect_bounds {smax = 4294967294 : index, smin = 0 : index, umax = 4294967294 : index, umin = 0 : index}
    // CHECK: test.reflect_bounds {smax = 4 : index, smin = 0 : index, umax = 4 : index, umin = 0 : index}
    // CHECK: test.reflect_bounds {smax = 10 : index, smin = 0 : index, umax = 10 : index, umin = 0 : index}
    %thread_id_x0 = test.reflect_bounds %thread_id_x
    %thread_id_y0 = test.reflect_bounds %thread_id_y
    %thread_id_z0 = test.reflect_bounds %thread_id_z

    // The launch bounds are not constant, and so this can't infer anything
    // CHECK: test.reflect_bounds {smax = 4294967294 : index, smin = 0 : index, umax = 4294967294 : index, umin = 0 : index}
    %thread_id_op = gpu.thread_id y
    %thread_id_op0 = test.reflect_bounds %thread_id_op
    gpu.terminator
  }

  func.return
}

// -----

// CHECK-LABEL: func @kernel
module attributes {gpu.container_module} {
  gpu.module @gpu_module {
    llvm.func @kernel() attributes {gpu.kernel} {

      %grid_dim_x = gpu.grid_dim x
      %grid_dim_y = gpu.grid_dim y
      %grid_dim_z = gpu.grid_dim z

      // CHECK: test.reflect_bounds {smax = 4294967295 : index, smin = 1 : index, umax = 4294967295 : index, umin = 1 : index}
      // CHECK: test.reflect_bounds {smax = 4294967295 : index, smin = 1 : index, umax = 4294967295 : index, umin = 1 : index}
      // CHECK: test.reflect_bounds {smax = 4294967295 : index, smin = 1 : index, umax = 4294967295 : index, umin = 1 : index}
      %grid_dim_x0 = test.reflect_bounds %grid_dim_x
      %grid_dim_y0 = test.reflect_bounds %grid_dim_y
      %grid_dim_z0 = test.reflect_bounds %grid_dim_z

      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %block_id_z = gpu.block_id z

      // CHECK: test.reflect_bounds {smax = 4294967294 : index, smin = 0 : index, umax = 4294967294 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 4294967294 : index, smin = 0 : index, umax = 4294967294 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 4294967294 : index, smin = 0 : index, umax = 4294967294 : index, umin = 0 : index}
      %block_id_x0 = test.reflect_bounds %block_id_x
      %block_id_y0 = test.reflect_bounds %block_id_y
      %block_id_z0 = test.reflect_bounds %block_id_z

      %block_dim_x = gpu.block_dim x
      %block_dim_y = gpu.block_dim y
      %block_dim_z = gpu.block_dim z

      // CHECK: test.reflect_bounds {smax = 4294967295 : index, smin = 1 : index, umax = 4294967295 : index, umin = 1 : index}
      // CHECK: test.reflect_bounds {smax = 4294967295 : index, smin = 1 : index, umax = 4294967295 : index, umin = 1 : index}
      // CHECK: test.reflect_bounds {smax = 4294967295 : index, smin = 1 : index, umax = 4294967295 : index, umin = 1 : index}
      %block_dim_x0 = test.reflect_bounds %block_dim_x
      %block_dim_y0 = test.reflect_bounds %block_dim_y
      %block_dim_z0 = test.reflect_bounds %block_dim_z

      %thread_id_x = gpu.thread_id x
      %thread_id_y = gpu.thread_id y
      %thread_id_z = gpu.thread_id z

      // CHECK: test.reflect_bounds {smax = 4294967294 : index, smin = 0 : index, umax = 4294967294 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 4294967294 : index, smin = 0 : index, umax = 4294967294 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 4294967294 : index, smin = 0 : index, umax = 4294967294 : index, umin = 0 : index}
      %thread_id_x0 = test.reflect_bounds %thread_id_x
      %thread_id_y0 = test.reflect_bounds %thread_id_y
      %thread_id_z0 = test.reflect_bounds %thread_id_z

      %global_id_x = gpu.global_id x
      %global_id_y = gpu.global_id y
      %global_id_z = gpu.global_id z

      // CHECK: test.reflect_bounds {smax = 9223372036854775807 : index, smin = -9223372036854775808 : index, umax = -8589934592 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 9223372036854775807 : index, smin = -9223372036854775808 : index, umax = -8589934592 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 9223372036854775807 : index, smin = -9223372036854775808 : index, umax = -8589934592 : index, umin = 0 : index}
      %global_id_x0 = test.reflect_bounds %global_id_x
      %global_id_y0 = test.reflect_bounds %global_id_y
      %global_id_z0 = test.reflect_bounds %global_id_z

      %subgroup_size = gpu.subgroup_size : index
      %lane_id = gpu.lane_id
      %num_subgroups = gpu.num_subgroups : index
      %subgroup_id = gpu.subgroup_id : index

      // CHECK: test.reflect_bounds {smax = 128 : index, smin = 1 : index, umax = 128 : index, umin = 1 : index}
      // CHECK: test.reflect_bounds {smax = 127 : index, smin = 0 : index, umax = 127 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 4294967295 : index, smin = 1 : index, umax = 4294967295 : index, umin = 1 : index}
      // CHECK: test.reflect_bounds {smax = 4294967294 : index, smin = 0 : index, umax = 4294967294 : index, umin = 0 : index}
      %subgroup_size0 = test.reflect_bounds %subgroup_size
      %lane_id0 = test.reflect_bounds %lane_id
      %num_subgroups0 = test.reflect_bounds %num_subgroups
      %subgroup_id0 = test.reflect_bounds %subgroup_id

      llvm.return
    }
  }
}

// -----

// CHECK-LABEL: func @annotated_kernel
module attributes {gpu.container_module} {
  gpu.module @gpu_module {
    gpu.func @annotated_kernel() kernel
      attributes {gpu.known_block_size = array<i32: 8, 12, 16>,
          gpu.known_grid_size = array<i32: 20, 24, 28>} {

      %grid_dim_x = gpu.grid_dim x
      %grid_dim_y = gpu.grid_dim y
      %grid_dim_z = gpu.grid_dim z

      // CHECK: test.reflect_bounds {smax = 20 : index, smin = 20 : index, umax = 20 : index, umin = 20 : index}
      // CHECK: test.reflect_bounds {smax = 24 : index, smin = 24 : index, umax = 24 : index, umin = 24 : index}
      // CHECK: test.reflect_bounds {smax = 28 : index, smin = 28 : index, umax = 28 : index, umin = 28 : index}
      %grid_dim_x0 = test.reflect_bounds %grid_dim_x
      %grid_dim_y0 = test.reflect_bounds %grid_dim_y
      %grid_dim_z0 = test.reflect_bounds %grid_dim_z

      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %block_id_z = gpu.block_id z

      // CHECK: test.reflect_bounds {smax = 19 : index, smin = 0 : index, umax = 19 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 23 : index, smin = 0 : index, umax = 23 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 27 : index, smin = 0 : index, umax = 27 : index, umin = 0 : index}
      %block_id_x0 = test.reflect_bounds %block_id_x
      %block_id_y0 = test.reflect_bounds %block_id_y
      %block_id_z0 = test.reflect_bounds %block_id_z

      %block_dim_x = gpu.block_dim x
      %block_dim_y = gpu.block_dim y
      %block_dim_z = gpu.block_dim z

      // CHECK: test.reflect_bounds {smax = 8 : index, smin = 8 : index, umax = 8 : index, umin = 8 : index}
      // CHECK: test.reflect_bounds {smax = 12 : index, smin = 12 : index, umax = 12 : index, umin = 12 : index}
      // CHECK: test.reflect_bounds {smax = 16 : index, smin = 16 : index, umax = 16 : index, umin = 16 : index}
      %block_dim_x0 = test.reflect_bounds %block_dim_x
      %block_dim_y0 = test.reflect_bounds %block_dim_y
      %block_dim_z0 = test.reflect_bounds %block_dim_z

      %thread_id_x = gpu.thread_id x
      %thread_id_y = gpu.thread_id y
      %thread_id_z = gpu.thread_id z

      // CHECK: test.reflect_bounds {smax = 7 : index, smin = 0 : index, umax = 7 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 11 : index, smin = 0 : index, umax = 11 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 15 : index, smin = 0 : index, umax = 15 : index, umin = 0 : index}
      %thread_id_x0 = test.reflect_bounds %thread_id_x
      %thread_id_y0 = test.reflect_bounds %thread_id_y
      %thread_id_z0 = test.reflect_bounds %thread_id_z

      %global_id_x = gpu.global_id x
      %global_id_y = gpu.global_id y
      %global_id_z = gpu.global_id z

      // CHECK: test.reflect_bounds {smax = 159 : index, smin = 0 : index, umax = 159 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 287 : index, smin = 0 : index, umax = 287 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 447 : index, smin = 0 : index, umax = 447 : index, umin = 0 : index}
      %global_id_x0 = test.reflect_bounds %global_id_x
      %global_id_y0 = test.reflect_bounds %global_id_y
      %global_id_z0 = test.reflect_bounds %global_id_z

      %subgroup_size = gpu.subgroup_size : index
      %lane_id = gpu.lane_id
      %num_subgroups = gpu.num_subgroups : index
      %subgroup_id = gpu.subgroup_id : index

      // CHECK: test.reflect_bounds {smax = 128 : index, smin = 1 : index, umax = 128 : index, umin = 1 : index}
      // CHECK: test.reflect_bounds {smax = 127 : index, smin = 0 : index, umax = 127 : index, umin = 0 : index}
      // CHECK: test.reflect_bounds {smax = 4294967295 : index, smin = 1 : index, umax = 4294967295 : index, umin = 1 : index}
      // CHECK: test.reflect_bounds {smax = 4294967294 : index, smin = 0 : index, umax = 4294967294 : index, umin = 0 : index}
      %subgroup_size0 = test.reflect_bounds %subgroup_size
      %lane_id0 = test.reflect_bounds %lane_id
      %num_subgroups0 = test.reflect_bounds %num_subgroups
      %subgroup_id0 = test.reflect_bounds %subgroup_id

      gpu.return
    }
  }
}

