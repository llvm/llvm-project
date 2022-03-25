// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK-LABEL:func @no_args(%{{.*}}: index)
  func @no_args(%sz : index) {
    // CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}})
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %sz, %grid_y = %sz, %grid_z = %sz)
               threads(%tx, %ty, %tz) in (%block_x = %sz, %block_y = %sz, %block_z = %sz) {
      // CHECK: gpu.terminator
      gpu.terminator
    }
    return
  }

  // CHECK-LABEL:func @args(%{{.*}}: index, %{{.*}}: index, %{{.*}}: f32, %{{.*}}: memref<?xf32, 1>) {
  func @args(%blk : index, %thrd : index, %float : f32, %data : memref<?xf32,1>) {
    // CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}})
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %blk, %grid_y = %blk, %grid_z = %blk)
               threads(%tx, %ty, %tz) in (%block_x = %thrd, %block_y = %thrd, %block_z = %thrd) {
      "use"(%float) : (f32) -> ()
      "use"(%data) : (memref<?xf32,1>) -> ()
      // CHECK: gpu.terminator
      gpu.terminator
    }
    return
  }

  gpu.module @kernels {
    gpu.func @kernel_1(%arg0 : f32, %arg1 : memref<?xf32, 1>) kernel {
      %tIdX = gpu.thread_id x
      %tIdY = gpu.thread_id y
      %tIdZ = gpu.thread_id z

      %bDimX = gpu.block_dim x
      %bDimY = gpu.block_dim y
      %bDimZ = gpu.block_dim z

      %bIdX = gpu.block_id x
      %bIdY = gpu.block_id y
      %bIdZ = gpu.block_id z

      %gDimX = gpu.grid_dim x
      %gDimY = gpu.grid_dim y
      %gDimZ = gpu.grid_dim z

      %gIdX = gpu.global_id x
      %gIdY = gpu.global_id y
      %gIdZ = gpu.global_id z

      %sgId = gpu.subgroup_id : index
      %numSg = gpu.num_subgroups : index
      %SgSi = gpu.subgroup_size : index

      %one = arith.constant 1.0 : f32
      %sum = gpu.all_reduce add %one {} : (f32) -> (f32)

      %width = arith.constant 7 : i32
      %offset = arith.constant 3 : i32
      // CHECK: gpu.shuffle xor %{{.*}}, %{{.*}}, %{{.*}} : f32
      %shfl, %pred = gpu.shuffle xor %arg0, %offset, %width : f32
      // CHECK: gpu.shuffle up %{{.*}}, %{{.*}}, %{{.*}} : f32
      %shfl1, %pred1 = gpu.shuffle up %arg0, %offset, %width : f32
      // CHECK: gpu.shuffle down %{{.*}}, %{{.*}}, %{{.*}} : f32
      %shfl2, %pred2 = gpu.shuffle down %arg0, %offset, %width : f32
      // CHECK: gpu.shuffle idx %{{.*}}, %{{.*}}, %{{.*}} : f32
      %shfl3, %pred3 = gpu.shuffle idx %arg0, %offset, %width : f32

      "gpu.barrier"() : () -> ()

      "some_op"(%bIdX, %tIdX) : (index, index) -> ()
      %42 = memref.load %arg1[%bIdX] : memref<?xf32, 1>
      gpu.return
    }

    gpu.func @kernel_2() kernel {
      gpu.return
    }
  }

  func @foo() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<?xf32, 1>)
    // CHECK: %{{.*}} = arith.constant 8
    %cst = arith.constant 8 : index
    %c0 = arith.constant 0 : i32
    %t0 = gpu.wait async

    // CHECK: gpu.launch_func @kernels::@kernel_1 blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}}) args(%{{.*}} : f32, %{{.*}} : memref<?xf32, 1>)
    gpu.launch_func @kernels::@kernel_1 blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst) args(%0 : f32, %1 : memref<?xf32, 1>)

    gpu.launch_func @kernels::@kernel_1 blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst) dynamic_shared_memory_size %c0 args(%0 : f32, %1 : memref<?xf32, 1>)

    // CHECK: gpu.launch_func @kernels::@kernel_2 blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})
    gpu.launch_func @kernels::@kernel_2 blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)

    // CHECK: %{{.*}} = gpu.launch_func async [%{{.*}}] @kernels::@kernel_2 blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})
    %t1 = gpu.launch_func async [%t0] @kernels::@kernel_2  blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)

    return
  }

  gpu.module @gpu_funcs {
    // CHECK-LABEL: gpu.func @kernel_1({{.*}}: f32)
    // CHECK:       workgroup
    // CHECK:       private
    // CHECK:       attributes
    gpu.func @kernel_1(%arg0: f32)
        workgroup(%arg1: memref<42xf32, 3>)
        private(%arg2: memref<2xf32, 5>, %arg3: memref<1xf32, 5>)
        kernel
        attributes {foo="bar"} {
      "use"(%arg1) : (memref<42xf32, 3>) -> ()
      "use"(%arg2) : (memref<2xf32, 5>) -> ()
      "use"(%arg3) : (memref<1xf32, 5>) -> ()
      gpu.return
    }

    // CHECK-LABEL gpu.func @printf_test
    // CHECK: (%[[ARG0:.*]]: i32)
    // CHECK: gpu.printf "Value: %d" %[[ARG0]] : i32
    gpu.func @printf_test(%arg0 : i32) {
      gpu.printf "Value: %d" %arg0 : i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @no_attribution
    // CHECK: {
    gpu.func @no_attribution(%arg0: f32) {
      gpu.return
    }

    // CHECK-LABEL: @no_attribution_attrs
    // CHECK:       attributes
    // CHECK:       {
    gpu.func @no_attribution_attrs(%arg0: f32) attributes {foo="bar"} {
      gpu.return
    }

    // CHECK-LABEL: @workgroup_only
    // CHECK:       workgroup({{.*}}: {{.*}})
    // CHECK:       {
    gpu.func @workgroup_only() workgroup(%arg0: memref<42xf32, 3>) {
      gpu.return
    }
    // CHECK-LABEL: @private_only
    // CHECK:       private({{.*}}: {{.*}})
    // CHECK:       {
    gpu.func @private_only() private(%arg0: memref<2xf32, 5>) {
      gpu.return
    }

    // CHECK-LABEL: @empty_attribution
    // CHECK:       {
    gpu.func @empty_attribution(%arg0: f32) workgroup() private() {
      gpu.return
    }
  }

  gpu.module @explicit_attributions {
    // CHECK-LABEL: gpu.func @kernel_1({{.*}}: f32, {{.*}}: memref<?xf32>) workgroup({{.*}}: memref<5xf32, 3>) private({{.*}}: memref<5xf32, 5>)
    "gpu.func"() ({
    ^bb0(%arg0: f32, %arg1: memref<?xf32>, %arg2: memref<5xf32, 3>, %arg3: memref<5xf32, 5>):
      "gpu.return"() : () -> ()
    } ) {function_type = (f32, memref<?xf32>) -> (), gpu.kernel, sym_name = "kernel_1", workgroup_attributions = 1: i64} : () -> ()
  }

  func @alloc() {
    // CHECK-LABEL: func @alloc()

    // CHECK: %[[m0:.*]] = gpu.alloc () : memref<13xf32, 1>
    %m0 = gpu.alloc () : memref<13xf32, 1>
    // CHECK: gpu.dealloc %[[m0]] : memref<13xf32, 1>
    gpu.dealloc %m0 : memref<13xf32, 1>

    %t0 = gpu.wait async
    // CHECK: %[[m1:.*]], %[[t1:.*]] = gpu.alloc async [{{.*}}] () : memref<13xf32, 1>
    %m1, %t1 = gpu.alloc async [%t0] () : memref<13xf32, 1>
    // CHECK: gpu.dealloc async [%[[t1]]] %[[m1]] : memref<13xf32, 1>
    %t2 = gpu.dealloc async [%t1] %m1 : memref<13xf32, 1>

    return
  }

  func @async_token(%arg0 : !gpu.async.token) -> !gpu.async.token {
    // CHECK-LABEL: func @async_token({{.*}}: !gpu.async.token)
    // CHECK: return {{.*}} : !gpu.async.token
    return %arg0 : !gpu.async.token
  }

  func @async_wait() {
    // CHECK-LABEL: func @async_wait
    // CHECK: %[[t0:.*]] = gpu.wait async
    %0 = gpu.wait async
    // CHECK: %[[t1:.*]] = gpu.wait async [%[[t0]]]
    %1 = gpu.wait async [%0]
    // CHECK: %{{.*}} = gpu.wait async [%[[t0]], %[[t1]]]
    %2 = gpu.wait async [%0, %1]
    // CHECK: gpu.wait [%[[t0]], %[[t1]]]
    // CHECK-NOT: async
    gpu.wait [%0, %1]
    // CHECK: gpu.wait
    // CHECK-NOT: async
    gpu.wait // Valid, but a no-op.
    return
  }

  func @memcpy(%dst : memref<3x7xf32>, %src : memref<3x7xf32, 1>) {
    // CHECK-LABEL: func @memcpy
    // CHECK: gpu.memcpy {{.*}}, {{.*}} : memref<3x7xf32>, memref<3x7xf32, 1>
    gpu.memcpy %dst, %src : memref<3x7xf32>, memref<3x7xf32, 1>
    // CHECK: %[[t0:.*]] = gpu.wait async
    %0 = gpu.wait async
    // CHECK: {{.*}} = gpu.memcpy async [%[[t0]]] {{.*}}, {{.*}} : memref<3x7xf32>, memref<3x7xf32, 1>
    %1 = gpu.memcpy async [%0] %dst, %src : memref<3x7xf32>, memref<3x7xf32, 1>
    return
  }

  func @memset(%dst : memref<3x7xf32>, %value : f32) {
    // CHECK-LABEL: func @memset
    // CHECK: gpu.memset {{.*}}, {{.*}} : memref<3x7xf32>, f32
    gpu.memset %dst, %value : memref<3x7xf32>, f32
    // CHECK: %[[t0:.*]] = gpu.wait async
    %0 = gpu.wait async
    // CHECK: {{.*}} = gpu.memset async [%[[t0]]] {{.*}}, {{.*}} : memref<3x7xf32>, f32
    %1 = gpu.memset async [%0] %dst, %value : memref<3x7xf32>, f32
    return
  }

  func @mmamatrix_valid_element_type(%src : memref<32x32xf16, affine_map<(d0, d1) -> (d0 * 64 + d1)>>){
    // CHECK-LABEL: func @mmamatrix_valid_element_type
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    // CHECK: %[[wg:.*]] = memref.alloca()
    %i = arith.constant 16 : index
    // CHECK: %[[i:.*]] = arith.constant 16 : index
     %cst = arith.constant 1.000000e+00 : f32
    // CHECK: %[[cst:.*]] = arith.constant 1.000000e+00 : f32
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %i] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    // CHECK: gpu.subgroup_mma_load_matrix %[[wg]][%[[i]], %[[i]]] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    %s = gpu.subgroup_mma_load_matrix %src[%i, %i] {leadDimension = 64 : index} : memref<32x32xf16, affine_map<(d0, d1) -> (d0 * 64 + d1)>> -> !gpu.mma_matrix<16x16xf16, "AOp">
    // CHECK: gpu.subgroup_mma_load_matrix %{{.*}}[%[[i]], %[[i]]] {leadDimension = 64 : index} : memref<32x32xf16, #{{.*}}> -> !gpu.mma_matrix<16x16xf16, "AOp">
    %1 = gpu.subgroup_mma_constant_matrix %cst : !gpu.mma_matrix<16x16xf32, "COp">
    // CHECK: gpu.subgroup_mma_elementwise addf %{{.*}}, %{{.*}} : (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">
    %2 = gpu.subgroup_mma_elementwise addf %1, %1 : (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">
    // CHECK: gpu.subgroup_mma_elementwise maxf %{{.*}}, %{{.*}} : (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">
    %3 = gpu.subgroup_mma_elementwise maxf %2, %1 : (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">
    return
  }

  func @async_cp(%dst : memref<2x7x5xf32, 3>, %src : memref<4x5xf32>){
    // CHECK-LABEL: func @async_cp
    %c0 = arith.constant 0 : index
    // CHECK: gpu.device_async_copy %{{.*}}[{{.*}}, {{.*}}], %{{.*}}[{{.*}}, {{.*}}, {{.*}}], 4 : memref<4x5xf32> to memref<2x7x5xf32, 3>
    %0 = gpu.device_async_copy %src[%c0, %c0], %dst[%c0, %c0, %c0], 4 : memref<4x5xf32> to memref<2x7x5xf32, 3>
    // CHECK: %{{.*}} = gpu.device_async_create_group
    %token = gpu.device_async_create_group %0
    // CHECK: gpu.device_async_wait %{{.*}} {numGroups = 1 : i32}
    gpu.device_async_wait %token {numGroups = 1 : i32}
    return
  }

  // CHECK-LABEL: func @set_default_device
  func @set_default_device(%arg0: i32) {
    // CHECK: gpu.set_default_device
    gpu.set_default_device %arg0
    return
  }
}
