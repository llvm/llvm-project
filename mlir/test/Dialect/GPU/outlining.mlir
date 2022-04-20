// RUN: mlir-opt -allow-unregistered-dialect -gpu-launch-sink-index-computations -gpu-kernel-outlining -split-input-file -verify-diagnostics %s | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect -gpu-launch-sink-index-computations -gpu-kernel-outlining=data-layout-str='#dlti.dl_spec<#dlti.dl_entry<index,32:i32>>' -split-input-file %s | FileCheck --check-prefix CHECK-DL %s

// CHECK: module attributes {gpu.container_module}

// CHECK-LABEL: func @launch()
func.func @launch() {
  // CHECK: %[[ARG0:.*]] = "op"() : () -> f32
  %0 = "op"() : () -> (f32)
  // CHECK: %[[ARG1:.*]] = "op"() : () -> memref<?xf32, 1>
  %1 = "op"() : () -> (memref<?xf32, 1>)
  // CHECK: %[[GDIMX:.*]] = arith.constant 8
  %gDimX = arith.constant 8 : index
  // CHECK: %[[GDIMY:.*]] = arith.constant 12
  %gDimY = arith.constant 12 : index
  // CHECK: %[[GDIMZ:.*]] = arith.constant 16
  %gDimZ = arith.constant 16 : index
  // CHECK: %[[BDIMX:.*]] = arith.constant 20
  %bDimX = arith.constant 20 : index
  // CHECK: %[[BDIMY:.*]] = arith.constant 24
  %bDimY = arith.constant 24 : index
  // CHECK: %[[BDIMZ:.*]] = arith.constant 28
  %bDimZ = arith.constant 28 : index

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

// CHECK-DL-LABEL: gpu.module @launch_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>}

// CHECK-LABEL: gpu.module @launch_kernel
// CHECK-NEXT: gpu.func @launch_kernel
// CHECK-SAME: (%[[KERNEL_ARG0:.*]]: f32, %[[KERNEL_ARG1:.*]]: memref<?xf32, 1>)
// CHECK-NEXT: %[[BID:.*]] = gpu.block_id x
// CHECK-NEXT: = gpu.block_id y
// CHECK-NEXT: = gpu.block_id z
// CHECK-NEXT: %[[TID:.*]] = gpu.thread_id x
// CHECK-NEXT: = gpu.thread_id y
// CHECK-NEXT: = gpu.thread_id z
// CHECK-NEXT: = gpu.grid_dim x
// CHECK-NEXT: = gpu.grid_dim y
// CHECK-NEXT: = gpu.grid_dim z
// CHECK-NEXT: %[[BDIM:.*]] = gpu.block_dim x
// CHECK-NEXT: = gpu.block_dim y
// CHECK-NEXT: = gpu.block_dim z
// CHECK-NEXT: cf.br ^[[BLOCK:.*]]
// CHECK-NEXT: ^[[BLOCK]]:
// CHECK-NEXT: "use"(%[[KERNEL_ARG0]]) : (f32) -> ()
// CHECK-NEXT: "some_op"(%[[BID]], %[[BDIM]]) : (index, index) -> ()
// CHECK-NEXT: = memref.load %[[KERNEL_ARG1]][%[[TID]]] : memref<?xf32, 1>

// -----

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: @multiple_launches
func.func @multiple_launches() {
  // CHECK: %[[CST:.*]] = arith.constant 8 : index
  %cst = arith.constant 8 : index
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

  // With async and async deps.
  // CHECK: %[[TOKEN:.*]] = gpu.wait async
  // CHECK: gpu.launch_func async [%[[TOKEN]]] @multiple_launches_kernel_1::@multiple_launches_kernel blocks in (%[[CST]], %[[CST]], %[[CST]]) threads in (%[[CST]], %[[CST]], %[[CST]])
  %t = gpu.wait async
  %u = gpu.launch async [%t] blocks(%bx2, %by2, %bz2) in (%grid_x2 = %cst, %grid_y2 = %cst,
                                          %grid_z2 = %cst)
             threads(%tx2, %ty2, %tz2) in (%block_x2 = %cst, %block_y2 = %cst,
                                           %block_z2 = %cst) {
    gpu.terminator
  }

  // CHECK: gpu.launch_func async @multiple_launches_kernel_2::@multiple_launches_kernel blocks in (%[[CST]], %[[CST]], %[[CST]]) threads in (%[[CST]], %[[CST]], %[[CST]])
  %v = gpu.launch async blocks(%bx2, %by2, %bz2) in (%grid_x2 = %cst, %grid_y2 = %cst,
                                     %grid_z2 = %cst)
             threads(%tx2, %ty2, %tz2) in (%block_x2 = %cst, %block_y2 = %cst,
                                           %block_z2 = %cst) {
    gpu.terminator
  }

  return
}

// CHECK-DL-LABEL: gpu.module @multiple_launches_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>}
// CHECK-DL-LABEL: gpu.module @multiple_launches_kernel_0 attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>}

// CHECK: gpu.module @multiple_launches_kernel
// CHECK: func @multiple_launches_kernel
// CHECK: module @multiple_launches_kernel_0
// CHECK: func @multiple_launches_kernel

// -----

// CHECK-LABEL: @extra_constants_not_inlined
func.func @extra_constants_not_inlined(%arg0: memref<?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 8 : index
  %cst = arith.constant 8 : index
  %cst2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
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

// CHECK-DL-LABEL: gpu.module @extra_constants_not_inlined_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>}

// CHECK-LABEL: func @extra_constants_not_inlined_kernel(%{{.*}}: memref<?xf32>, %{{.*}}: index)
// CHECK: arith.constant 2

// -----

// CHECK-LABEL: @extra_constants
// CHECK-SAME: %[[ARG0:.*]]: memref<?xf32>
func.func @extra_constants(%arg0: memref<?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 8 : index
  %cst = arith.constant 8 : index
  %cst2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
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

// CHECK-DL-LABEL: gpu.module @extra_constants_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>}

// CHECK-LABEL: func @extra_constants_kernel(
// CHECK-SAME: %[[KARG0:.*]]: memref<?xf32>
// CHECK: arith.constant 2
// CHECK: arith.constant 0
// CHECK: memref.dim %[[KARG0]]

// -----

// CHECK-LABEL: @extra_constants_noarg
// CHECK-SAME: %[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>
func.func @extra_constants_noarg(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 8 : index
  %cst = arith.constant 8 : index
  %cst2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
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

// CHECK-DL-LABEL: gpu.module @extra_constants_noarg_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>}

// CHECK-LABEL: func @extra_constants_noarg_kernel(
// CHECK-SAME: %[[KARG0:.*]]: memref<?xf32>, %[[KARG1:.*]]: index
// CHECK: %[[KCST:.*]] = arith.constant 2
// CHECK: "use"(%[[KCST]], %[[KARG0]], %[[KARG1]])

// -----

// CHECK-LABEL: @multiple_uses
func.func @multiple_uses(%arg0 : memref<?xf32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: gpu.func {{.*}} {
  // CHECK:   %[[C2:.*]] = arith.constant 2 : index
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

// CHECK-DL-LABEL: gpu.module @multiple_uses_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>}

// -----

// CHECK-LABEL: @multiple_uses2
func.func @multiple_uses2(%arg0 : memref<*xf32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d = memref.dim %arg0, %c2 : memref<*xf32>
  // CHECK: gpu.func {{.*}} {
  // CHECK:   %[[C2:.*]] = arith.constant 2 : index
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

// CHECK-DL-LABEL: gpu.module @multiple_uses2_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>}

// -----

llvm.mlir.global internal @global(42 : i64) : i64

//CHECK-LABEL: @function_call
func.func @function_call(%arg0 : memref<?xf32>) {
  %cst = arith.constant 8 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst,
                                       %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst,
                                        %block_z = %cst) {
    func.call @device_function() : () -> ()
    func.call @device_function() : () -> ()
    %0 = llvm.mlir.addressof @global : !llvm.ptr<i64>
    gpu.terminator
  }
  return
}

func.func @device_function() {
  call @recursive_device_function() : () -> ()
  return
}

func.func @recursive_device_function() {
  call @recursive_device_function() : () -> ()
  return
}

// CHECK-DL-LABEL: gpu.module @function_call_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>}

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
