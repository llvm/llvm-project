// RUN: mlir-opt -split-input-file %s | FileCheck %s --check-prefixes=CHECK
// Verify the printed output can be parsed.
// RUN: mlir-opt -split-input-file %s | mlir-opt -split-input-file | FileCheck %s --check-prefixes=CHECK
// Verify the generic form can be parsed.
// RUN: mlir-opt -split-input-file -mlir-print-op-generic %s | mlir-opt -split-input-file | FileCheck %s --check-prefixes=CHECK

// -----

// CHECK-LABEL: func @par_dim_sequential
func.func @par_dim_sequential() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%iv) = (%c0) to (%c4) step (%c1) {
    scf.reduce
  } {acc.par_dims = #acc<par_dims[sequential]>}
  return
}
// CHECK: scf.parallel
// CHECK: } {acc.par_dims = #acc<par_dims[sequential]>}

// -----

// CHECK-LABEL: func @par_dim_single_thread_x
func.func @par_dim_single_thread_x() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%iv) = (%c0) to (%c128) step (%c1) {
    scf.reduce
  } {acc.par_dims = #acc<par_dims[thread_x]>}
  return
}
// CHECK: scf.parallel
// CHECK: } {acc.par_dims = #acc<par_dims[thread_x]>}

// -----

// CHECK-LABEL: func @par_dims_block_thread
func.func @par_dims_block_thread() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c8, %c128) step (%c1, %c1) {
    scf.reduce
  } {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}
// CHECK: scf.parallel
// CHECK: } {acc.par_dims = #acc<par_dims[block_x, thread_x]>}

// -----

// All GPU parallel dimensions (par_dim values) in par_dims list
func.func @par_dims_all_dims() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%iv) = (%c0) to (%c1) step (%c1) {
    scf.reduce
  } {acc.par_dims = #acc<par_dims[block_x, block_y, block_z, thread_x, thread_y, thread_z]>}
  return
}
// CHECK: acc.par_dims = #acc<par_dims[block_x, block_y, block_z, thread_x, thread_y, thread_z]>

// -----

// 2D grid: block_y and thread_y
func.func @par_dims_2d_grid() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c4, %c32) step (%c1, %c1) {
    scf.reduce
  } {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
  return
}
// CHECK: acc.par_dims = #acc<par_dims[block_y, thread_y]>

// -----

// CHECK-LABEL: func @compute_region_single_dim
func.func @compute_region_single_dim(%data: memref<1024xf32>,
                                     %result: memref<f32>) {
  %c128 = arith.constant 128 : index
  %copyin = acc.copyin varPtr(%data : memref<1024xf32>) -> memref<1024xf32>
  %copy = acc.copyin varPtr(%result : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copy>}
  acc.kernel_environment dataOperands(%copyin, %copy : memref<1024xf32>, memref<f32>) {
    %w0 = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%arg0 = %w0)
        ins(%arg1 = %copyin, %arg2 = %copy) : (memref<1024xf32>, memref<f32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128_inner = arith.constant 128 : index
      %cst = arith.constant 0.000000e+00 : f32
      memref.store %cst, %arg2[] : memref<f32>
      scf.parallel (%iv) = (%c0) to (%c128_inner) step (%c1) {
        %val = memref.load %arg1[%iv] : memref<1024xf32>
        %cur = memref.load %arg2[] : memref<f32>
        %sum = arith.addf %cur, %val : f32
        memref.store %sum, %arg2[] : memref<f32>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  acc.copyout accPtr(%copy : memref<f32>) to varPtr(%result : memref<f32>) {dataClause = #acc<data_clause acc_copy>}
  acc.delete accPtr(%copyin : memref<1024xf32>)
  return
}
// CHECK: %[[W:.*]] = acc.par_width %{{.*}} {par_dim = #acc.par_dim<thread_x>}
// CHECK: acc.compute_region launch(%{{.*}} = %[[W]]) ins({{.*}}) : (memref<1024xf32>, memref<f32>) {
// CHECK:   acc.yield
// CHECK: } {origin = "acc.parallel"}

// -----

// CHECK-LABEL: func @compute_region_two_dims
func.func @compute_region_two_dims(%data: memref<8xi32>,
                                   %reduction_var: memref<i32>) {
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %copyin_data = acc.copyin varPtr(%data : memref<8xi32>) -> memref<8xi32>
  %copyin_red = acc.copyin varPtr(%reduction_var : memref<i32>) -> memref<i32> {dataClause = #acc<data_clause acc_reduction>}
  acc.kernel_environment dataOperands(%copyin_data, %copyin_red : memref<8xi32>, memref<i32>) {
    %w0 = acc.par_width %c8 {par_dim = #acc.par_dim<block_x>}
    %w1 = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%arg0 = %w0, %arg1 = %w1)
        ins(%arg2 = %copyin_data, %arg3 = %copyin_red) : (memref<8xi32>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8_inner = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %init = acc.reduction_init %arg3 <add> : memref<i32> {
        %alloca = memref.alloca() : memref<i32>
        memref.store %c0_i32, %alloca[] : memref<i32>
        acc.yield %alloca : memref<i32>
      }
      scf.parallel (%iv) = (%c0) to (%c8_inner) step (%c1) {
        %v = memref.load %arg2[%iv] : memref<8xi32>
        %cur = memref.load %init[] : memref<i32>
        %sum = arith.addi %cur, %v : i32
        memref.store %sum, %init[] : memref<i32>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
      acc.reduction_combine %init into %arg3 <add> : memref<i32>
      acc.yield
    } {origin = "acc.parallel"}
  }
  acc.copyout accPtr(%copyin_red : memref<i32>) to varPtr(%reduction_var : memref<i32>) {dataClause = #acc<data_clause acc_reduction>}
  acc.delete accPtr(%copyin_data : memref<8xi32>)
  return
}
// CHECK: %[[W0:.*]] = acc.par_width %{{.*}} {par_dim = #acc.par_dim<block_x>}
// CHECK: %[[W1:.*]] = acc.par_width %{{.*}} {par_dim = #acc.par_dim<thread_x>}
// CHECK: acc.compute_region launch(%{{.*}} = %[[W0]], %{{.*}} = %[[W1]]) ins({{.*}}) : (memref<8xi32>, memref<i32>) {
// CHECK:   acc.yield
// CHECK: } {origin = "acc.parallel"}

// -----

// CHECK-LABEL: func @compute_region_unknown_width
func.func @compute_region_unknown_width(%data: memref<100xf32>) {
  %copyin = acc.copyin varPtr(%data : memref<100xf32>) -> memref<100xf32>
  acc.kernel_environment dataOperands(%copyin : memref<100xf32>) {
    %w0 = acc.par_width {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%arg0 = %w0)
        ins(%arg1 = %copyin) : (memref<100xf32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c100 = arith.constant 100 : index
      scf.parallel (%iv) = (%c0) to (%c100) step (%c1) {
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_x]>}
      acc.yield
    } {origin = "acc.kernels"}
  }
  acc.delete accPtr(%copyin : memref<100xf32>)
  return
}
// CHECK: %[[W:.*]] = acc.par_width {par_dim = #acc.par_dim<thread_x>}
// CHECK: acc.compute_region launch(%{{.*}} = %[[W]]) ins({{.*}}) : (memref<100xf32>) {
// CHECK:   acc.yield
// CHECK: } {origin = "acc.kernels"}

// -----

// CHECK-LABEL: func @compute_region_no_launch
func.func @compute_region_no_launch(%a: memref<i32>, %b: memref<i32>) {
  %copy_a = acc.copyin varPtr(%a : memref<i32>) -> memref<i32> {dataClause = #acc<data_clause acc_copy>}
  %copy_b = acc.copyin varPtr(%b : memref<i32>) -> memref<i32> {dataClause = #acc<data_clause acc_copy>}
  acc.kernel_environment dataOperands(%copy_a, %copy_b : memref<i32>, memref<i32>) {
    acc.compute_region
        ins(%arg0 = %copy_a, %arg1 = %copy_b) : (memref<i32>, memref<i32>) {
      %c1 = arith.constant 1 : i32
      memref.store %c1, %arg0[] : memref<i32>
      memref.store %c1, %arg1[] : memref<i32>
      acc.yield
    } {origin = "acc.serial"}
  }
  acc.copyout accPtr(%copy_a : memref<i32>) to varPtr(%a : memref<i32>) {dataClause = #acc<data_clause acc_copy>}
  acc.copyout accPtr(%copy_b : memref<i32>) to varPtr(%b : memref<i32>) {dataClause = #acc<data_clause acc_copy>}
  return
}
// CHECK: acc.compute_region ins({{.*}}) : (memref<i32>, memref<i32>) {
// CHECK:   acc.yield
// CHECK: } {origin = "acc.serial"}

// -----

// CHECK-LABEL: func @compute_region_launch_only
func.func @compute_region_launch_only() {
  %c32 = arith.constant 32 : index
  %w0 = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%arg0 = %w0) {
    acc.yield
  } {origin = "acc.parallel"}
  return
}
// CHECK: %[[W:.*]] = acc.par_width %{{.*}} {par_dim = #acc.par_dim<thread_x>}
// CHECK: acc.compute_region launch(%{{.*}} = %[[W]]) {
// CHECK:   acc.yield
// CHECK: } {origin = "acc.parallel"}

// -----

// CHECK-LABEL: func @compute_region_all_fields
// CHECK-SAME: (%{{.*}}: memref<1024xf32>, %[[STREAM:.*]]: !gpu.async.token)
func.func @compute_region_all_fields(%data: memref<1024xf32>,
                                     %stream: !gpu.async.token) {
  %c128 = arith.constant 128 : index
  %c8 = arith.constant 8 : index
  %copyin = acc.copyin varPtr(%data : memref<1024xf32>) -> memref<1024xf32>
  acc.kernel_environment dataOperands(%copyin : memref<1024xf32>) {
    %w0 = acc.par_width %c8 {par_dim = #acc.par_dim<block_x>}
    %w1 = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region stream(%stream : !gpu.async.token)
        launch(%arg0 = %w0, %arg1 = %w1)
        ins(%arg2 = %copyin) : (memref<1024xf32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      scf.parallel (%iv) = (%c0) to (%c1024) step (%c1) {
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
      acc.yield
    } {kernel_func_name = @compute_kernel, kernel_module_name = @device_module, origin = "acc.parallel"}
  }
  acc.delete accPtr(%copyin : memref<1024xf32>)
  return
}
// CHECK: %[[W0:.*]] = acc.par_width %{{.*}} {par_dim = #acc.par_dim<block_x>}
// CHECK: %[[W1:.*]] = acc.par_width %{{.*}} {par_dim = #acc.par_dim<thread_x>}
// CHECK: acc.compute_region stream(%[[STREAM]] : !gpu.async.token) launch(%{{.*}} = %[[W0]], %{{.*}} = %[[W1]]) ins({{.*}}) : (memref<1024xf32>) {
// CHECK:   acc.yield
// CHECK: } {kernel_func_name = @compute_kernel, kernel_module_name = @device_module, origin = "acc.parallel"}

// -----

// CHECK-LABEL: func @compute_region_with_results
func.func @compute_region_with_results() -> i32 {
  %w0 = acc.par_width {par_dim = #acc.par_dim<thread_x>}
  %0 = acc.compute_region launch(%arg0 = %w0) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    acc.yield %c0_i32 : i32
  } {origin = "acc.parallel"}
  return %0 : i32
}
// CHECK: %[[W:.*]] = acc.par_width {par_dim = #acc.par_dim<thread_x>}
// CHECK: {{.*}} = acc.compute_region launch(%{{.*}} = %[[W]]) -> i32 {
// CHECK:   acc.yield
// CHECK: } {origin = "acc.parallel"}
