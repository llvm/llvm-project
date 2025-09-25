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

// CHECK-DL-LABEL: gpu.module @launch_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}
// CHECK-LABEL: gpu.module @launch_kernel
// CHECK-NEXT: gpu.func @launch_kernel
// CHECK-SAME: (%[[KERNEL_ARG0:.*]]: f32, %[[KERNEL_ARG1:.*]]: memref<?xf32, 1>)
// CHECK-SAME: known_block_size = array<i32: 20, 24, 28>
// CHECK-SAME: known_grid_size = array<i32: 8, 12, 16>
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
// CHECK-NEXT: "use"(%[[KERNEL_ARG0]]) : (f32) -> ()
// CHECK-NEXT: "some_op"(%[[BID]], %[[BDIM]]) : (index, index) -> ()
// CHECK-NEXT: = memref.load %[[KERNEL_ARG1]][%[[TID]]] : memref<?xf32, 1>

// -----

// Verify that we can outline a CFG
// CHECK-LABEL:  gpu.func @launchCFG_kernel(
// CHECK: cf.br
// CHECK: gpu.return
func.func @launchCFG() {
  %0 = "op"() : () -> (f32)
  %1 = "op"() : () -> (memref<?xf32, 1>)
  %gDimX = arith.constant 8 : index
  %gDimY = arith.constant 12 : index
  %gDimZ = arith.constant 16 : index
  %bDimX = arith.constant 20 : index
  %bDimY = arith.constant 24 : index
  %bDimZ = arith.constant 28 : index

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gDimX, %grid_y = %gDimY,
                                       %grid_z = %gDimZ)
             threads(%tx, %ty, %tz) in (%block_x = %bDimX, %block_y = %bDimY,
                                        %block_z = %bDimZ) {
    "use"(%0): (f32) -> ()
    cf.br ^bb1
  ^bb1:
    "some_op"(%bx, %block_x) : (index, index) -> ()
    %42 = memref.load %1[%tx] : memref<?xf32, 1>
    gpu.terminator
  }
  return
}


// -----

// This test checks gpu-out-lining can handle gpu.launch kernel from an llvm.func
// CHECK-LABEL: @launch_from_llvm_func
llvm.func @launch_from_llvm_func() {
  // CHECK: %[[ARG0:.*]] = "op"() : () -> f32
  %0 = "op"() : () -> (f32)
  // CHECK: %[[ARG1:.*]] = "op"() : () -> memref<?xf32, 1>
  %1 = "op"() : () -> (memref<?xf32, 1>)

  // CHECK: %[[DIM:.*]] = arith.constant 1
  %dim = arith.constant 1 : index

  // CHECK: gpu.launch_func @launch_from_llvm_func_kernel::@launch_from_llvm_func_kernel
  // CHECK-SAME: (%[[DIM]], %[[DIM]], %[[DIM]])
  // CHECK-SAME: (%[[DIM]], %[[DIM]], %[[DIM]]) args(%[[ARG0]] : f32, %[[ARG1]] : memref<?xf32, 1>)
  // CHECK-NEXT: llvm.return

  // CHECK: gpu.func {{.*}} kernel attributes
  // CHECK-SAME: known_block_size = array<i32: 1, 1, 1>
  // CHECK-SAME: known_grid_size = array<i32: 1, 1, 1>
  // CHECK: gpu.return
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %dim, %grid_y = %dim,
                                       %grid_z = %dim)
             threads(%tx, %ty, %tz) in (%block_x = %dim, %block_y = %dim,
                                        %block_z = %dim) {
    "use"(%0): (f32) -> ()
    "some_op"(%bx, %block_x) : (index, index) -> ()
    %2 = memref.load %1[%tx] : memref<?xf32, 1>
    gpu.terminator
  }
  llvm.return
}

// CHECK-DL-LABEL: gpu.module @launch_from_llvm_func_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}

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

// CHECK-DL-LABEL: gpu.module @multiple_launches_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}
// CHECK-DL-LABEL: gpu.module @multiple_launches_kernel_0 attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}

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

// CHECK-DL-LABEL: gpu.module @extra_constants_not_inlined_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}

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

// CHECK-DL-LABEL: gpu.module @extra_constants_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}

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

// CHECK-DL-LABEL: gpu.module @extra_constants_noarg_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}

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

// CHECK-DL-LABEL: gpu.module @multiple_uses_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}

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

// CHECK-DL-LABEL: gpu.module @multiple_uses2_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}

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
    %0 = llvm.mlir.addressof @global : !llvm.ptr
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

// CHECK-DL-LABEL: gpu.module @function_call_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}

// CHECK: gpu.module @function_call_kernel {
// CHECK:   gpu.func @function_call_kernel()
// CHECK:     call @device_function() : () -> ()
// CHECK:     call @device_function() : () -> ()
// CHECK:     llvm.mlir.addressof @global : !llvm.ptr
// CHECK:     gpu.return
//
// CHECK:   llvm.mlir.global internal @global(42 : i64) {addr_space = 0 : i32} : i64
//
// CHECK:   func @device_function()
// CHECK:   func @recursive_device_function()
// CHECK-NOT:   func @device_function

// -----

// CHECK-LABEL: @non_constant_launches
func.func @non_constant_launches(%arg0 : index) {
  // CHECK-NOT: known_block_size
  // CHECK-NOT: known_grid_size
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %arg0, %grid_y = %arg0,
                                       %grid_z = %arg0)
             threads(%tx, %ty, %tz) in (%block_x = %arg0, %block_y = %arg0,
                                        %block_z = %arg0) {
    gpu.terminator
  }
  return
}

// CHECK-DL-LABEL: gpu.module @non_constant_launches_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}

// CHECK: module attributes {gpu.container_module}

// -----

// This test checks memory attributions for gpu.launch, using both workgroup and private attributions.
// CHECK-LABEL: func @launch_memory_attributions_0()
func.func @launch_memory_attributions_0() {
  %1 = "op"() : () -> (memref<?xf32, 1>)
  %128 = arith.constant 128 : index

  // CHECK: gpu.launch_func @launch_memory_attributions_0_kernel::@launch_memory_attributions_0_kernel
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %128, %grid_y = %128,
                                       %grid_z = %128)
             threads(%tx, %ty, %tz) in (%block_x = %128, %block_y = %128,
                                        %block_z = %128)
             workgroup(%shared: memref<42xf32, 3>)
             private(%priv0: memref<2xf32, 5>, %priv1: memref<1xf32, 5>) {
    "some_op"(%bx, %block_x) : (index, index) -> ()
    %42 = memref.load %1[%tx] : memref<?xf32, 1>
    %43 = memref.load %shared[%tx] : memref<42xf32, 3>
    %44 = memref.load %priv1[%tx] : memref<1xf32, 5>
    gpu.terminator
  }
  return
}

// CHECK-DL-LABEL: gpu.module @launch_memory_attributions_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}

// CHECK-LABEL: gpu.module @launch_memory_attributions_0_kernel
// CHECK-NEXT: gpu.func @launch_memory_attributions_0_kernel
// CHECK-SAME: workgroup(%[[KERNEL_ARG1:.*]] : memref<42xf32, 3>)
// CHECK-SAME: private(%[[KERNEL_ARG2:.*]] : memref<2xf32, 5>, %[[KERNEL_ARG3:.*]] : memref<1xf32, 5>)
// CHECK: %[[TID:.*]] = gpu.thread_id x
// CHECK: = memref.load %[[KERNEL_ARG1]][%[[TID]]] : memref<42xf32, 3>
// CHECK-NEXT: = memref.load %[[KERNEL_ARG3]][%[[TID]]] : memref<1xf32, 5>

// -----

// This test checks correctness of private attributions in the absence of workgroup attributions.
// CHECK-LABEL: @launch_memory_attributions_1
func.func @launch_memory_attributions_1(%arg0 : memref<*xf32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d = memref.dim %arg0, %c2 : memref<*xf32>
  // CHECK: gpu.func {{.*}}  private(%[[KERNEL_ARG:.*]] : memref<3xf32, 5>) {{.*}} {
  // CHECK:   %[[C2:.*]] = arith.constant 2 : index
  // CHECK: = memref.load %[[KERNEL_ARG]][%[[C2]]] : memref<3xf32, 5>
  // CHECK:   gpu.return
  // CHECK: }
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1,
                                       %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1,
                                        %block_z = %c1)
             private(%priv0: memref<3xf32, 5>) {
    %42 = memref.load %priv0[%c2] : memref<3xf32, 5>
    gpu.terminator
  }
  return
}

// CHECK-DL-LABEL: gpu.module @launch_memory_attributions_1_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>}

// -----
// CHECK: module attributes {gpu.container_module}

// CHECK-LABEL: func @launch_cluster()
func.func @launch_cluster() {
  // CHECK: %[[ARG0:.*]] = "op"() : () -> f32
  %0 = "op"() : () -> (f32)
  // CHECK: %[[ARG1:.*]] = "op"() : () -> memref<?xf32, 1>
  %1 = "op"() : () -> (memref<?xf32, 1>)
  // CHECK: %[[CDIMX:.*]] = arith.constant 1
  %cDimX = arith.constant 1 : index
  // CHECK: %[[CDIMY:.*]] = arith.constant 2
  %cDimY = arith.constant 2 : index
  // CHECK: %[[CDIMZ:.*]] = arith.constant 1
  %cDimZ = arith.constant 1 : index
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

  // CHECK: gpu.launch_func @launch_cluster_kernel::@launch_cluster_kernel clusters in (%[[CDIMX]], %[[CDIMY]], %[[CDIMZ]]) blocks in (%[[GDIMX]], %[[GDIMY]], %[[GDIMZ]]) threads in (%[[BDIMX]], %[[BDIMY]], %[[BDIMZ]]) args(%[[ARG0]] : f32, %[[ARG1]] : memref<?xf32, 1>)
  // CHECK-NOT: gpu.launch blocks
  gpu.launch clusters(%cx, %cy, %cz) in (%cluster_x = %cDimX, %cluster_y = %cDimY,
                                       %cluster_z = %cDimZ)
             blocks(%bx, %by, %bz) in (%grid_x = %gDimX, %grid_y = %gDimY,
                                       %grid_z = %gDimZ)
             threads(%tx, %ty, %tz) in (%block_x = %bDimX, %block_y = %bDimY,
                                        %block_z = %bDimZ) {
    "use"(%0): (f32) -> ()
    "some_op"(%cx, %bx, %block_x) : (index, index, index) -> ()
    %42 = memref.load %1[%tx] : memref<?xf32, 1>
    gpu.terminator
  }
  return
}

// CHECK-LABEL: gpu.module @launch_cluster_kernel
// CHECK-NEXT: gpu.func @launch_cluster_kernel
// CHECK-SAME: (%[[KERNEL_ARG0:.*]]: f32, %[[KERNEL_ARG1:.*]]: memref<?xf32, 1>)
// CHECK-SAME: known_block_size = array<i32: 20, 24, 28>
// CHECK-SAME: known_grid_size = array<i32: 8, 12, 16>
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
// CHECK-NEXT: %[[CID:.*]] = gpu.cluster_id x
// CHECK-NEXT: = gpu.cluster_id y
// CHECK-NEXT: = gpu.cluster_id z
// CHECK-NEXT: %[[CDIM:.*]] = gpu.cluster_dim x
// CHECK-NEXT: = gpu.cluster_dim y
// CHECK-NEXT: = gpu.cluster_dim z
// CHECK-NEXT: "use"(%[[KERNEL_ARG0]]) : (f32) -> ()
// CHECK-NEXT: "some_op"(%[[CID]], %[[BID]], %[[BDIM]]) : (index, index, index) -> ()
// CHECK-NEXT: = memref.load %[[KERNEL_ARG1]][%[[TID]]] : memref<?xf32, 1>

// -----
// This test tests the two optional attributes `module` and `function` for gpu.launch
// CHECK-LABEL: func.func @testKernelAttributes()
// CHECK: gpu.launch_func  @test_module::@test_kernel_func blocks in (%[[GRID_X:.*]], %[[GRID_Y:.*]], %[[GRID_Z:.*]]) threads in (%[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]])
// CHECK: gpu.module @test_module
// CHECK: gpu.func @test_kernel_func()
func.func @testKernelAttributes() {
  %gDimX = arith.constant 8 : index
  %gDimY = arith.constant 12 : index
  %gDimZ = arith.constant 16 : index
  %bDimX = arith.constant 32 : index
  %bDimY = arith.constant 16 : index
  %bDimZ = arith.constant 8 : index

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gDimX, %grid_y = %gDimY, %grid_z = %gDimZ)
             threads(%tx, %ty, %tz) in (%block_x = %bDimX, %block_y = %bDimY, %block_z = %bDimZ)
             module(@test_module) function(@test_kernel_func) {
    "some_op"(%bx, %tx) : (index, index) -> ()
    gpu.terminator
  }
  return
}

// -----
// This test tests the two optional attributes `module` and `function` for gpu.launch, when kernelModule already exists.

// CHECK-LABEL: gpu.module @existing_module
// CHECK: gpu.func @test_kernel_func()
// CHECK: gpu.func @test_kernel_func_0()
// CHECK-NOT: gpu.module @testExistingModule_kernel
// CHECK-NOT: gpu.func @testExistingModule_kernel()
// CHECK: func.func @testExistingModule()
// CHECK: gpu.launch_func  @existing_module::@test_kernel_func_0 blocks in (%[[GRID_X:.*]], %[[GRID_Y:.*]], %[[GRID_Z:.*]]) threads in (%[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]])

gpu.module @existing_module {
  gpu.func @test_kernel_func() {
    gpu.return
  }
}

func.func @testExistingModule() {
  %gDimX = arith.constant 8 : index
  %gDimY = arith.constant 12 : index
  %gDimZ = arith.constant 16 : index
  %bDimX = arith.constant 32 : index
  %bDimY = arith.constant 16 : index
  %bDimZ = arith.constant 8 : index

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gDimX, %grid_y = %gDimY, %grid_z = %gDimZ)
             threads(%tx, %ty, %tz) in (%block_x = %bDimX, %block_y = %bDimY, %block_z = %bDimZ)
             module(@existing_module) function(@test_kernel_func) {
    "some_op"(%bx, %tx) : (index, index) -> ()
    gpu.terminator
  }
  return
}

// -----
// This test tests the optional attribute `module` for gpu.launch.
// CHECK-LABEL: func.func @testKernelModuleOnly()
// CHECK: gpu.launch_func  @test_module::@testKernelModuleOnly_kernel blocks in (%[[GRID_X:.*]], %[[GRID_Y:.*]], %[[GRID_Z:.*]]) threads in (%[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]])
// CHECK: gpu.module @test_module
// CHECK: gpu.func @testKernelModuleOnly_kernel()
func.func @testKernelModuleOnly() {
  %gDimX = arith.constant 8 : index
  %gDimY = arith.constant 12 : index
  %gDimZ = arith.constant 16 : index
  %bDimX = arith.constant 32 : index
  %bDimY = arith.constant 16 : index
  %bDimZ = arith.constant 8 : index

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gDimX, %grid_y = %gDimY, %grid_z = %gDimZ)
             threads(%tx, %ty, %tz) in (%block_x = %bDimX, %block_y = %bDimY, %block_z = %bDimZ)
             module(@test_module) {
    "some_op"(%bx, %tx) : (index, index) -> ()
    gpu.terminator
  }
  return
}

// -----
// This test tests the optional attribute `function` for gpu.launch.
// CHECK-LABEL: func.func @testKernelFuncOnly()
// CHECK: gpu.launch_func  @test_kernel_func::@test_kernel_func blocks in (%[[GRID_X:.*]], %[[GRID_Y:.*]], %[[GRID_Z:.*]]) threads in (%[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]])

// CHECK: gpu.module @test_kernel_func
// CHECK: gpu.func @test_kernel_func()
func.func @testKernelFuncOnly() {
  %gDimX = arith.constant 8 : index
  %gDimY = arith.constant 12 : index
  %gDimZ = arith.constant 16 : index
  %bDimX = arith.constant 32 : index
  %bDimY = arith.constant 16 : index
  %bDimZ = arith.constant 8 : index

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gDimX, %grid_y = %gDimY, %grid_z = %gDimZ)
             threads(%tx, %ty, %tz) in (%block_x = %bDimX, %block_y = %bDimY, %block_z = %bDimZ)
             function(@test_kernel_func) {
    "some_op"(%bx, %tx) : (index, index) -> ()
    gpu.terminator
  }
  return
}

// -----
// This test tests gpu.launch when optional attributes `module` and `function` are not specified.
// CHECK-LABEL: func.func @testNoAttributes()
// CHECK: gpu.launch_func  @testNoAttributes_kernel::@testNoAttributes_kernel blocks in (%[[GRID_X:.*]], %[[GRID_Y:.*]], %[[GRID_Z:.*]]) threads in (%[[BLOCK_X:.*]], %[[BLOCK_Y:.*]], %[[BLOCK_Z:.*]])

// CHECK: gpu.module @testNoAttributes_kernel
// CHECK: gpu.func @testNoAttributes_kernel()
func.func @testNoAttributes() {
  %gDimX = arith.constant 8 : index
  %gDimY = arith.constant 12 : index
  %gDimZ = arith.constant 16 : index
  %bDimX = arith.constant 32 : index
  %bDimY = arith.constant 16 : index
  %bDimZ = arith.constant 8 : index

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gDimX, %grid_y = %gDimY, %grid_z = %gDimZ)
             threads(%tx, %ty, %tz) in (%block_x = %bDimX, %block_y = %bDimY, %block_z = %bDimZ) {
    "some_op"(%bx, %tx) : (index, index) -> ()
    gpu.terminator
  }
  return
}

// -----

// This test tests nested `gpu.launch`.

// CHECK-LABEL: func.func @nested_launch(
//  CHECK-SAME:                          %[[ARG0:.*]]: index) {
//       CHECK:   gpu.launch_func  @nested_launch_kernel_0::@nested_launch_kernel blocks in (%[[ARG0]], %[[ARG0]], %[[ARG0]]) threads in (%[[ARG0]], %[[ARG0]], %[[ARG0]])  args(%[[ARG0]] : index)
//       CHECK: gpu.module @nested_launch_kernel
//       CHECK:   gpu.func @nested_launch_kernel() kernel
//       CHECK:     "some_op"
//       CHECK: gpu.module @nested_launch_kernel_0
//       CHECK:   gpu.func @nested_launch_kernel(%[[VAL_0:.*]]: index) kernel
//       CHECK:     gpu.launch_func  @nested_launch_kernel::@nested_launch_kernel blocks in (%[[VAL_0]], %[[VAL_0]], %[[VAL_0]]) threads in (%[[VAL_0]], %[[VAL_0]], %[[VAL_0]])
func.func @nested_launch(%sz : index) {
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %sz, %grid_y = %sz, %grid_z = %sz)
             threads(%tx, %ty, %tz) in (%block_x = %sz, %block_y = %sz, %block_z = %sz) {
    gpu.launch blocks(%bx1, %by1, %bz1) in (%grid_x1 = %sz, %grid_y1 = %sz, %grid_z1 = %sz)
               threads(%tx1, %ty1, %tz1) in (%block_x1 = %sz, %block_y1 = %sz, %block_z1 = %sz) {
      "some_op"(%bx1, %tx1) : (index, index) -> ()
      gpu.terminator
    }
    gpu.terminator
  }
  return
}
