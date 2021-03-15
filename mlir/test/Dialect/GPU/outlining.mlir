// RUN: mlir-opt -allow-unregistered-dialect -gpu-kernel-outlining -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: module attributes {gpu.container_module}

// CHECK-LABEL: func @launch()
func @launch() {
  // CHECK: %[[ARG0:.*]] = "op"() : () -> f32
  %0 = "op"() : () -> (f32)
  // CHECK: %[[ARG1:.*]] = "op"() : () -> memref<?xf32, 1>
  %1 = "op"() : () -> (memref<?xf32, 1>)
  // CHECK: %[[GDIMX:.*]] = constant 8
  %gDimX = constant 8 : index
  // CHECK: %[[GDIMY:.*]] = constant 12
  %gDimY = constant 12 : index
  // CHECK: %[[GDIMZ:.*]] = constant 16
  %gDimZ = constant 16 : index
  // CHECK: %[[BDIMX:.*]] = constant 20
  %bDimX = constant 20 : index
  // CHECK: %[[BDIMY:.*]] = constant 24
  %bDimY = constant 24 : index
  // CHECK: %[[BDIMZ:.*]] = constant 28
  %bDimZ = constant 28 : index

  // CHECK: gpu.launch_func @launch_kernel::@launch_kernel blocks in (%[[GDIMX]], %[[GDIMY]], %[[GDIMZ]]) threads in (%[[BDIMX]], %[[BDIMY]], %[[BDIMZ]]) args(%[[ARG0]] : f32, %[[ARG1]] : memref<?xf32, 1>)
  // CHECK-NOT: gpu.launch blocks
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gDimX, %grid_y = %gDimY,
                                       %grid_z = %gDimZ)
             threads(%tx, %ty, %tz) in (%block_x = %bDimX, %block_y = %bDimY,
                                        %block_z = %bDimZ) {
    "use"(%0): (f32) -> ()
    "some_op"(%bx, %block_x) : (index, index) -> ()
    %42 = memref.load %1[%tx] : memref<?xf32, 1>
    gpu.terminator
  }
  return
}


// CHECK-LABEL: module @launch_kernel
// CHECK-NEXT: gpu.func @launch_kernel
// CHECK-SAME: (%[[KERNEL_ARG0:.*]]: f32, %[[KERNEL_ARG1:.*]]: memref<?xf32, 1>)
// CHECK-NEXT: %[[BID:.*]] = "gpu.block_id"() {dimension = "x"} : () -> index
// CHECK-NEXT: = "gpu.block_id"() {dimension = "y"} : () -> index
// CHECK-NEXT: = "gpu.block_id"() {dimension = "z"} : () -> index
// CHECK-NEXT: %[[TID:.*]] = "gpu.thread_id"() {dimension = "x"} : () -> index
// CHECK-NEXT: = "gpu.thread_id"() {dimension = "y"} : () -> index
// CHECK-NEXT: = "gpu.thread_id"() {dimension = "z"} : () -> index
// CHECK-NEXT: = "gpu.grid_dim"() {dimension = "x"} : () -> index
// CHECK-NEXT: = "gpu.grid_dim"() {dimension = "y"} : () -> index
// CHECK-NEXT: = "gpu.grid_dim"() {dimension = "z"} : () -> index
// CHECK-NEXT: %[[BDIM:.*]] = "gpu.block_dim"() {dimension = "x"} : () -> index
// CHECK-NEXT: = "gpu.block_dim"() {dimension = "y"} : () -> index
// CHECK-NEXT: = "gpu.block_dim"() {dimension = "z"} : () -> index
// CHECK-NEXT: br ^[[BLOCK:.*]]
// CHECK-NEXT: ^[[BLOCK]]:
// CHECK-NEXT: "use"(%[[KERNEL_ARG0]]) : (f32) -> ()
// CHECK-NEXT: "some_op"(%[[BID]], %[[BDIM]]) : (index, index) -> ()
// CHECK-NEXT: = memref.load %[[KERNEL_ARG1]][%[[TID]]] : memref<?xf32, 1>

// -----

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: @multiple_launches
func @multiple_launches() {
  // CHECK: %[[CST:.*]] = constant 8 : index
  %cst = constant 8 : index
  // CHECK: gpu.launch_func @multiple_launches_kernel::@multiple_launches_kernel blocks in (%[[CST]], %[[CST]], %[[CST]]) threads in (%[[CST]], %[[CST]], %[[CST]])
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst,
                                       %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst,
                                        %block_z = %cst) {
    gpu.terminator
  }
  // CHECK: gpu.launch_func @multiple_launches_kernel_0::@multiple_launches_kernel blocks in (%[[CST]], %[[CST]], %[[CST]]) threads in (%[[CST]], %[[CST]], %[[CST]])
  gpu.launch blocks(%bx2, %by2, %bz2) in (%grid_x2 = %cst, %grid_y2 = %cst,
                                          %grid_z2 = %cst)
             threads(%tx2, %ty2, %tz2) in (%block_x2 = %cst, %block_y2 = %cst,
                                           %block_z2 = %cst) {
    gpu.terminator
  }
  return
}

// CHECK: module @multiple_launches_kernel
// CHECK: func @multiple_launches_kernel
// CHECK: module @multiple_launches_kernel_0
// CHECK: func @multiple_launches_kernel

// -----

// CHECK-LABEL: @extra_constants_not_inlined
func @extra_constants_not_inlined(%arg0: memref<?xf32>) {
  // CHECK: %[[CST:.*]] = constant 8 : index
  %cst = constant 8 : index
  %cst2 = constant 2 : index
  %c0 = constant 0 : index
  %cst3 = "secret_constant"() : () -> index
  // CHECK: gpu.launch_func @extra_constants_not_inlined_kernel::@extra_constants_not_inlined_kernel blocks in (%[[CST]], %[[CST]], %[[CST]]) threads in (%[[CST]], %[[CST]], %[[CST]]) args({{.*}} : memref<?xf32>, {{.*}} : index)
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst,
                                       %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst,
                                        %block_z = %cst) {
    "use"(%cst2, %arg0, %cst3) : (index, memref<?xf32>, index) -> ()
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @extra_constants_not_inlined_kernel(%{{.*}}: memref<?xf32>, %{{.*}}: index)
// CHECK: constant 2

// -----

// CHECK-LABEL: @extra_constants
// CHECK-SAME: %[[ARG0:.*]]: memref<?xf32>
func @extra_constants(%arg0: memref<?xf32>) {
  // CHECK: %[[CST:.*]] = constant 8 : index
  %cst = constant 8 : index
  %cst2 = constant 2 : index
  %c0 = constant 0 : index
  %cst3 = memref.dim %arg0, %c0 : memref<?xf32>
  // CHECK: gpu.launch_func @extra_constants_kernel::@extra_constants_kernel blocks in (%[[CST]], %[[CST]], %[[CST]]) threads in (%[[CST]], %[[CST]], %[[CST]]) args(%[[ARG0]] : memref<?xf32>)
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst,
                                       %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst,
                                        %block_z = %cst) {
    "use"(%cst2, %arg0, %cst3) : (index, memref<?xf32>, index) -> ()
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @extra_constants_kernel(
// CHECK-SAME: %[[KARG0:.*]]: memref<?xf32>
// CHECK: constant 2
// CHECK: constant 0
// CHECK: memref.dim %[[KARG0]]

// -----

// CHECK-LABEL: @extra_constants_noarg
// CHECK-SAME: %[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>
func @extra_constants_noarg(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
  // CHECK: %[[CST:.*]] = constant 8 : index
  %cst = constant 8 : index
  %cst2 = constant 2 : index
  %c0 = constant 0 : index
  // CHECK: memref.dim %[[ARG1]]
  %cst3 = memref.dim %arg1, %c0 : memref<?xf32>
  // CHECK: gpu.launch_func @extra_constants_noarg_kernel::@extra_constants_noarg_kernel blocks in (%[[CST]], %[[CST]], %[[CST]]) threads in (%[[CST]], %[[CST]], %[[CST]]) args(%[[ARG0]] : memref<?xf32>, {{.*}} : index)
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst,
                                       %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst,
                                        %block_z = %cst) {
    "use"(%cst2, %arg0, %cst3) : (index, memref<?xf32>, index) -> ()
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @extra_constants_noarg_kernel(
// CHECK-SAME: %[[KARG0:.*]]: memref<?xf32>, %[[KARG1:.*]]: index
// CHECK: %[[KCST:.*]] = constant 2
// CHECK: "use"(%[[KCST]], %[[KARG0]], %[[KARG1]])

// -----

// CHECK-LABEL: @multiple_uses
func @multiple_uses(%arg0 : memref<?xf32>) {
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  // CHECK: gpu.func {{.*}} {
  // CHECK:   %[[C2:.*]] = constant 2 : index
  // CHECK:   "use1"(%[[C2]], %[[C2]])
  // CHECK:   "use2"(%[[C2]])
  // CHECK:   gpu.return
  // CHECK: }
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1,
                                       %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1,
                                        %block_z = %c1) {
    "use1"(%c2, %c2) : (index, index) -> ()
    "use2"(%c2) : (index) -> ()
    gpu.terminator
  }
  return
}

// -----

// CHECK-LABEL: @multiple_uses2
func @multiple_uses2(%arg0 : memref<*xf32>) {
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %d = memref.dim %arg0, %c2 : memref<*xf32>
  // CHECK: gpu.func {{.*}} {
  // CHECK:   %[[C2:.*]] = constant 2 : index
  // CHECK:   %[[D:.*]] = memref.dim %[[ARG:.*]], %[[C2]]
  // CHECK:   "use1"(%[[D]])
  // CHECK:   "use2"(%[[C2]], %[[C2]])
  // CHECK:   "use3"(%[[ARG]])
  // CHECK:   gpu.return
  // CHECK: }
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1,
                                       %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1,
                                        %block_z = %c1) {
    "use1"(%d) : (index) -> ()
    "use2"(%c2, %c2) : (index, index) -> ()
    "use3"(%arg0) : (memref<*xf32>) -> ()
    gpu.terminator
  }
  return
}

// -----

llvm.mlir.global internal @global(42 : i64) : i64

//CHECK-LABEL: @function_call
func @function_call(%arg0 : memref<?xf32>) {
  %cst = constant 8 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst,
                                       %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst,
                                        %block_z = %cst) {
    call @device_function() : () -> ()
    call @device_function() : () -> ()
    %0 = llvm.mlir.addressof @global : !llvm.ptr<i64>
    gpu.terminator
  }
  return
}

func @device_function() {
  call @recursive_device_function() : () -> ()
  return
}

func @recursive_device_function() {
  call @recursive_device_function() : () -> ()
  return
}

// CHECK: gpu.module @function_call_kernel {
// CHECK:   gpu.func @function_call_kernel()
// CHECK:     call @device_function() : () -> ()
// CHECK:     call @device_function() : () -> ()
// CHECK:     llvm.mlir.addressof @global : !llvm.ptr<i64>
// CHECK:     gpu.return
//
// CHECK:   llvm.mlir.global internal @global(42 : i64) : i64
//
// CHECK:   func @device_function()
// CHECK:   func @recursive_device_function()
// CHECK-NOT:   func @device_function
