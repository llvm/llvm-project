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

// CHECK-LABEL: func @reduction_init_with_bounds
func.func @reduction_init_with_bounds(%arg0: memref<?xi32>, %lb: index, %ext: index) {
  %c1 = arith.constant 1 : index
  %bnd = acc.bounds lowerbound(%lb : index) extent(%ext : index) stride(%c1 : index)
  %0 = acc.reduction_init %arg0 bounds(%bnd) <add> : memref<?xi32> {
    acc.yield %arg0 : memref<?xi32>
  }
  return
}
// CHECK: %[[BND:.*]] = acc.bounds
// CHECK: acc.reduction_init %{{.*}} bounds(%[[BND]]) <add> : memref<?xi32>

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

// CHECK-LABEL: func @compute_region_empty
func.func @compute_region_empty() {
  acc.compute_region {
  } {origin = "acc.parallel"}
  return
}
// CHECK: acc.compute_region {
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

// CHECK-LABEL: func @parallel_reduction_pattern
func.func @parallel_reduction_pattern(%data: memref<8xi32>, %shared: memref<i32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %private = memref.alloca() {acc.par_dims = #acc<par_dims[thread_x]>} : memref<i32>
  memref.store %c0_i32, %private[] : memref<i32>
  %partial = scf.parallel (%iv) = (%c0) to (%c8) step (%c1) init (%c0_i32) -> i32 {
    %v = memref.load %data[%iv] : memref<8xi32>
    scf.reduce(%v : i32) {
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      scf.reduce.return %sum : i32
    }
  } {acc.par_dims = #acc<par_dims[thread_x]>}
  acc.reduction_accumulate %partial to %private <add>
      : i32 -> memref<i32> {par_dims = #acc<par_dims[thread_x]>}
  acc.reduction_combine %private into %shared <add> : memref<i32>
      {acc.par_dims = #acc<par_dims[thread_x]>}
  return
}
// CHECK: memref.alloca() {acc.par_dims = #acc<par_dims[thread_x]>}
// CHECK: scf.parallel
// CHECK: scf.reduce
// CHECK: acc.reduction_accumulate %{{.*}} to %{{.*}} <add> : i32 -> memref<i32> {par_dims = #acc<par_dims[thread_x]>}
// CHECK: acc.reduction_combine %{{.*}} into %{{.*}} <add> : memref<i32> {acc.par_dims = #acc<par_dims[thread_x]>}

// -----

// CHECK-LABEL: func @reduction_accumulate_thread_x
func.func @reduction_accumulate_thread_x(%partial: f32, %private: memref<f32>) {
  acc.reduction_accumulate %partial to %private <add>
      : f32 -> memref<f32> {par_dims = #acc<par_dims[thread_x]>}
  return
}
// CHECK: acc.reduction_accumulate %{{.*}} to %{{.*}} <add> : f32 -> memref<f32> {par_dims = #acc<par_dims[thread_x]>}

// -----

// CHECK-LABEL: func @reduction_accumulate_block_thread
func.func @reduction_accumulate_block_thread(%partial: i32, %private: memref<i32>) {
  acc.reduction_accumulate %partial to %private <add>
      : i32 -> memref<i32> {par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}
// CHECK: acc.reduction_accumulate %{{.*}} to %{{.*}} <add> : i32 -> memref<i32> {par_dims = #acc<par_dims[block_x, thread_x]>}

// -----

// CHECK-LABEL: func @reduction_accumulate_array
func.func @reduction_accumulate_array(%private: memref<4xi32>, %bounds: !acc.data_bounds_ty) {
  acc.reduction_accumulate_array %private bounds(%bounds) <add> : memref<4xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}
// CHECK: acc.reduction_accumulate_array  %{{.*}} bounds(%{{.*}}) <add> : memref<4xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}

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

// -----

// CHECK-LABEL: func @predicate_region_gang_vector_atomics
func.func @predicate_region_gang_vector_atomics(%c1: memref<i32>, %c2: memref<i32>) {
  %c3 = arith.constant 3 : index
  %c16 = arith.constant 16 : index
  %copy_c1 = acc.copyin varPtr(%c1 : memref<i32>) -> memref<i32> {dataClause = #acc<data_clause acc_copy>}
  %copy_c2 = acc.copyin varPtr(%c2 : memref<i32>) -> memref<i32> {dataClause = #acc<data_clause acc_copy>}
  acc.kernel_environment dataOperands(%copy_c1, %copy_c2 : memref<i32>, memref<i32>) {
    %w_gang = acc.par_width %c3 {par_dim = #acc.par_dim<block_x>}
    %w_vector = acc.par_width %c16 {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%arg0 = %w_gang, %arg1 = %w_vector)
        ins(%arg2 = %copy_c1, %arg3 = %copy_c2) : (memref<i32>, memref<i32>) {
      %c1_idx = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      scf.parallel (%i) = (%c1_idx) to (%c10) step (%c1_idx) {
        acc.predicate_region {
          acc.atomic.update %arg2 : memref<i32> {
          ^bb0(%old: i32):
            %one = arith.constant 1 : i32
            %sum = arith.addi %old, %one : i32
            acc.yield %sum : i32
          }
        }
        scf.parallel (%j) = (%c1_idx) to (%c10) step (%c1_idx) {
          acc.atomic.update %arg3 : memref<i32> {
          ^bb0(%old: i32):
            %one = arith.constant 1 : i32
            %sum = arith.addi %old, %one : i32
            acc.yield %sum : i32
          }
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  acc.copyout accPtr(%copy_c1 : memref<i32>) to varPtr(%c1 : memref<i32>) {dataClause = #acc<data_clause acc_copy>}
  acc.copyout accPtr(%copy_c2 : memref<i32>) to varPtr(%c2 : memref<i32>) {dataClause = #acc<data_clause acc_copy>}
  return
}
// CHECK: acc.predicate_region {
// CHECK:   acc.atomic.update
// CHECK: }

// -----

// CHECK-LABEL: func @predicate_region_gang_redundant_setup
func.func @predicate_region_gang_redundant_setup(%idx: memref<i32>, %table: memref<10xi32>) {
  %c3 = arith.constant 3 : index
  %c16 = arith.constant 16 : index
  %copy_idx = acc.copyin varPtr(%idx : memref<i32>) -> memref<i32> {dataClause = #acc<data_clause acc_copy>}
  %copy_table = acc.copyin varPtr(%table : memref<10xi32>) -> memref<10xi32>
  acc.kernel_environment dataOperands(%copy_idx, %copy_table : memref<i32>, memref<10xi32>) {
    %w_gang = acc.par_width %c3 {par_dim = #acc.par_dim<block_x>}
    %w_vector = acc.par_width %c16 {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%arg0 = %w_gang, %arg1 = %w_vector)
        ins(%arg2 = %copy_idx, %arg3 = %copy_table) : (memref<i32>, memref<10xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      %v = memref.load %arg3[%c0] : memref<10xi32>
      acc.predicate_region {
        memref.store %v, %arg2[] : memref<i32>
      }
      scf.parallel (%i) = (%c1) to (%c10) step (%c1) {
        %i_val = memref.load %arg2[] : memref<i32>
        %twice = arith.addi %i_val, %i_val : i32
        memref.store %twice, %arg2[] : memref<i32>
        scf.parallel (%j) = (%c1) to (%c10) step (%c1) {
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } {origin = "acc.kernels"}
  }
  acc.copyout accPtr(%copy_idx : memref<i32>) to varPtr(%idx : memref<i32>) {dataClause = #acc<data_clause acc_copy>}
  acc.delete accPtr(%copy_table : memref<10xi32>)
  return
}
// CHECK: acc.predicate_region {
// CHECK:   memref.store
// CHECK: }

// -----

// CHECK-LABEL: func @gpu_shared_memory_static
func.func @gpu_shared_memory_static() {
  %c1024 = arith.constant 1024 : index
  %sm = acc.gpu_shared_memory(%c1024)
      {num_copies = 1 : i64, static_upper_bound_bytes = 4096 : i64}
      : (index) -> memref<?xf32, #gpu.address_space<workgroup>>
  return
}
// CHECK: acc.gpu_shared_memory(%{{.*}}) {num_copies = 1 : i64, static_upper_bound_bytes = 4096 : i64}
// CHECK-SAME: : (index) -> memref<?xf32, #gpu.address_space<workgroup>>

// -----

// CHECK-LABEL: func @gpu_shared_memory_runtime_sized
func.func @gpu_shared_memory_runtime_sized() {
  %c128 = arith.constant 128 : index
  %sm = acc.gpu_shared_memory(%c128)
      {num_copies = 1 : i64,
       static_upper_bound_bytes = 1560 : i64,
       dynamic_shared_memory_scaling_bytes = 12 : i64,
       dynamic_shared_memory_fixed_bytes = 24 : i64}
      : (index) -> memref<?xf32, #gpu.address_space<workgroup>>
  return
}
// CHECK: acc.gpu_shared_memory(%{{.*}}) {dynamic_shared_memory_fixed_bytes = 24 : i64, dynamic_shared_memory_scaling_bytes = 12 : i64, num_copies = 1 : i64, static_upper_bound_bytes = 1560 : i64}

// -----

// CHECK-LABEL: func @gpu_shared_memory_worker_copies
func.func @gpu_shared_memory_worker_copies() {
  %sm = acc.gpu_shared_memory()
      {num_copies = 4 : i64, static_upper_bound_bytes = 256 : i64}
      : () -> memref<8xf32, #gpu.address_space<workgroup>>
  return
}
// CHECK: acc.gpu_shared_memory {num_copies = 4 : i64, static_upper_bound_bytes = 256 : i64}
