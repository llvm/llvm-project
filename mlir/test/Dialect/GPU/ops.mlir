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
      %tIdX = "gpu.thread_id"() {dimension = "x"} : () -> (index)
      %tIdY = "gpu.thread_id"() {dimension = "y"} : () -> (index)
      %tIdZ = "gpu.thread_id"() {dimension = "z"} : () -> (index)

      %bDimX = "gpu.block_dim"() {dimension = "x"} : () -> (index)
      %bDimY = "gpu.block_dim"() {dimension = "y"} : () -> (index)
      %bDimZ = "gpu.block_dim"() {dimension = "z"} : () -> (index)

      %bIdX = "gpu.block_id"() {dimension = "x"} : () -> (index)
      %bIdY = "gpu.block_id"() {dimension = "y"} : () -> (index)
      %bIdZ = "gpu.block_id"() {dimension = "z"} : () -> (index)

      %gDimX = "gpu.grid_dim"() {dimension = "x"} : () -> (index)
      %gDimY = "gpu.grid_dim"() {dimension = "y"} : () -> (index)
      %gDimZ = "gpu.grid_dim"() {dimension = "z"} : () -> (index)

      %sgId = gpu.subgroup_id : index
      %numSg = gpu.num_subgroups : index
      %SgSi = gpu.subgroup_size : index

      %one = constant 1.0 : f32
      %sum = "gpu.all_reduce"(%one) ({}) {op = "add"} : (f32) -> (f32)

      %width = constant 7 : i32
      %offset = constant 3 : i32
      // CHECK: gpu.shuffle %{{.*}}, %{{.*}}, %{{.*}} xor : f32
      %shfl, %pred = gpu.shuffle %arg0, %offset, %width xor : f32

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
    // CHECK: %{{.*}} = constant 8
    %cst = constant 8 : index
    %c0 = constant 0 : i32
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
    "gpu.func"() ( {
    ^bb0(%arg0: f32, %arg1: memref<?xf32>, %arg2: memref<5xf32, 3>, %arg3: memref<5xf32, 5>):
      "gpu.return"() : () -> ()
    } ) {gpu.kernel, sym_name = "kernel_1", type = (f32, memref<?xf32>) -> (), workgroup_attributions = 1: i64} : () -> ()
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

  func @mmamatrix_valid_element_type(){
    // CHECK-LABEL: func @mmamatrix_valid_element_type
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    // CHECK: %[[wg:.*]] = memref.alloca()
    %i = constant 16 : index
    // CHECK: %[[i:.*]] = constant 16 : index
     %cst = constant 1.000000e+00 : f32
    // CHECK: %[[cst:.*]] = constant 1.000000e+00 : f32
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %i] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    // CHECK: gpu.subgroup_mma_load_matrix %[[wg]][%[[i]], %[[i]]] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    %1 = gpu.subgroup_mma_constant_matrix %cst : !gpu.mma_matrix<16x16xf32, "COp">
    // CHECK: gpu.subgroup_mma_constant_matrix %[[cst]] : !gpu.mma_matrix<16x16xf32, "COp">
    return
  }
}
