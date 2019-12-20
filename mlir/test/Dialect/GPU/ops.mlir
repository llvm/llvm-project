// RUN: mlir-opt %s | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK-LABEL:func @no_args(%{{.*}}: index)
  func @no_args(%sz : index) {
    // CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}})
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %sz, %grid_y = %sz, %grid_z = %sz)
               threads(%tx, %ty, %tz) in (%block_x = %sz, %block_y = %sz, %block_z = %sz) {
      // CHECK: gpu.return
      gpu.return
    }
    return
  }

  // CHECK-LABEL:func @args(%{{.*}}: index, %{{.*}}: index, %{{.*}}: f32, %{{.*}}: memref<?xf32, 1>) {
  func @args(%blk : index, %thrd : index, %float : f32, %data : memref<?xf32,1>) {
    // CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : f32, memref<?xf32, 1>
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %blk, %grid_y = %blk, %grid_z = %blk)
               threads(%tx, %ty, %tz) in (%block_x = %thrd, %block_y = %thrd, %block_z = %thrd)
               args(%kernel_arg0 = %float, %kernel_arg1 = %data) : f32, memref<?xf32, 1> {
      // CHECK: gpu.return
      gpu.return
    }
    return
  }

  // It is possible to use values passed into the region as arguments.
  // CHECK-LABEL: func @passing_values
  func @passing_values(%blk : index, %thrd : index, %float : f32, %data : memref<?xf32,1>) {
    // CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : f32, memref<?xf32, 1>
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %blk, %grid_y = %blk, %grid_z = %blk)
               threads(%tx, %ty, %tz) in (%block_x = %thrd, %block_y = %thrd, %block_z = %thrd)
               args(%kernel_arg0 = %float, %kernel_arg1 = %data) : f32, memref<?xf32, 1> {
      // CHECK: "use"(%{{.*}})
      "use"(%kernel_arg0): (f32) -> ()
      // CHECK: gpu.return
      gpu.return
    }
    return
  }

  // It is possible to use values defined in nested regions as long as they don't
  // cross kernel launch region boundaries.
  // CHECK-LABEL: func @nested_isolation
  func @nested_isolation(%sz : index) {
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %sz, %grid_y = %sz, %grid_z = %sz)
               threads(%tx, %ty, %tz) in (%block_x = %sz, %block_y = %sz, %block_z = %sz) {
      "region"() ({
        // CHECK: %{{.*}} = "produce"()
        %val = "produce"() : () -> (index)
        "region"() ({
          // CHECK: "use"(%{{.*}})
          "use"(%val) : (index) -> ()
        }) : () -> ()
      }) : () -> ()
      // CHECK: gpu.return
      gpu.return
    }
    return
  }

  module @kernels attributes {gpu.kernel_module} {
    gpu.func @kernel_1(%arg0 : f32, %arg1 : memref<?xf32, 1>) attributes {gpu.kernel} {
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

      %one = constant 1.0 : f32
      %sum = "gpu.all_reduce"(%one) ({}) {op = "add"} : (f32) -> (f32)

      %width = constant 7 : i32
      %offset = constant 3 : i32
      // CHECK: gpu.shuffle %{{.*}}, %{{.*}}, %{{.*}} xor : f32
      %shfl, %pred = gpu.shuffle %arg0, %offset, %width xor : f32

      "gpu.barrier"() : () -> ()

      "some_op"(%bIdX, %tIdX) : (index, index) -> ()
      %42 = load %arg1[%bIdX] : memref<?xf32, 1>
      gpu.return
    }

    gpu.func @kernel_2(%arg0: f32, %arg1: memref<?xf32, 1>) attributes {gpu.kernel} {
      gpu.return
    }
  }

  func @foo() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<?xf32, 1>)
    // CHECK: %{{.*}} = constant 8
    %cst = constant 8 : index

    // CHECK: "gpu.launch_func"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {kernel = "kernel_1", kernel_module = @kernels} : (index, index, index, index, index, index, f32, memref<?xf32, 1>) -> ()
    "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1)
    { kernel = "kernel_1", kernel_module = @kernels }
        : (index, index, index, index, index, index, f32, memref<?xf32, 1>) -> ()

    // CHECK: "gpu.launch_func"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {kernel = "kernel_2", kernel_module = @kernels} : (index, index, index, index, index, index, f32, memref<?xf32, 1>) -> ()
    "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1)
    { kernel = "kernel_2", kernel_module = @kernels }
        : (index, index, index, index, index, index, f32, memref<?xf32, 1>) -> ()

    return
  }

  module @gpu_funcs attributes {gpu.kernel_module} {
    // CHECK-LABEL: gpu.func @kernel_1({{.*}}: f32) -> f32
    // CHECK:       workgroup
    // CHECK:       private
    // CHECK:       attributes
    gpu.func @kernel_1(%arg0: f32) -> f32
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

}
