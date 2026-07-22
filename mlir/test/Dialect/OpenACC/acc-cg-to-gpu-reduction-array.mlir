// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A per-thread array accumulator (memref.alloca) for a block+thread reduction is
// reduced element-by-element across the parallel dimensions with gpu.all_reduce -
// the array analog of the scalar acc.reduction_accumulate.

// CHECK-LABEL: func.func @array_reduction
// CHECK: gpu.launch
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<2xi32>
// CHECK-NOT: acc.reduction_accumulate_array
// CHECK-NOT: acc.bounds
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[ELT:.*]] = memref.load %[[ALLOCA]][%[[IV]]] : memref<2xi32>
// CHECK:   %[[RED:.*]] = gpu.all_reduce add %[[ELT]]
// Per-thread alloca: the all_reduce result is stored unpredicated.
// CHECK:   memref.store %[[RED]], %[[ALLOCA]][%[[IV]]] : memref<2xi32>
// CHECK: }

func.func @array_reduction(%arg0: memref<2xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<2xi32>) -> memref<2xi32> {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "r"}
  acc.kernel_environment dataOperands(%0 : memref<2xi32>) {
    %c1_pw = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %bx = acc.par_width %c1_pw {par_dim = #acc.par_dim<block_x>}
    %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg2 = %0) : (memref<2xi32>) {
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %2 = acc.reduction_init %arg2 <add> : memref<2xi32> {
        %alloca = memref.alloca() : memref<2xi32>
        scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
          memref.store %c0_i32, %alloca[%i] : memref<2xi32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        acc.yield %alloca : memref<2xi32>
      }
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        scf.parallel (%tx_iv) = (%c0) to (%ktx) step (%c1) {
          %3 = memref.load %2[%c0] : memref<2xi32>
          %4 = arith.addi %3, %c1_i32 : i32
          memref.store %4, %2[%c0] : memref<2xi32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      %b = acc.bounds extent(%c2 : index)
      acc.reduction_accumulate_array %2 bounds(%b) <add> : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
      acc.reduction_combine_region %2 into %arg2 : memref<2xi32> {
        scf.for %i = %c0 to %c2 step %c1 {
          %3 = memref.load %2[%i] : memref<2xi32>
          %4 = memref.load %arg2[%i] : memref<2xi32>
          %5 = arith.addi %3, %4 : i32
          memref.store %5, %arg2[%i] : memref<2xi32>
        }
      }
      acc.yield
    } {origin = "acc.parallel"}
  }
  acc.copyout accPtr(%0 : memref<2xi32>) to varPtr(%arg0 : memref<2xi32>) {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "r"}
  return
}

// CHECK-LABEL: func.func @array_reduction_small_shared
// CHECK: memref.alloc() : memref<2xi32>
// CHECK-NOT: gpu.all_reduce
// CHECK-NOT: acc.reduction_accumulate_array
func.func @array_reduction_small_shared() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c2 = arith.constant 2 : index
    %shared = memref.alloc() : memref<2xi32>
    %bounds = acc.bounds extent(%c2 : index)
    acc.reduction_accumulate_array %shared bounds(%bounds) <add>
        : memref<2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    memref.dealloc %shared : memref<2xi32>
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @array_reduction_strided_extent
// CHECK: gpu.launch
// CHECK: %[[LB:.*]] = arith.constant 1 : index
// CHECK: %[[STEP:.*]] = arith.constant 2 : index
// CHECK: %[[EXTENT:.*]] = arith.constant 3 : index
// CHECK: %[[SPAN:.*]] = arith.muli %[[EXTENT]], %[[STEP]] : index
// CHECK: %[[UB:.*]] = arith.addi %[[LB]], %[[SPAN]] : index
// CHECK: scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]]
func.func @array_reduction_strided_extent() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %c1_b = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %local = memref.alloca() : memref<8xi32>
    %bounds = acc.bounds lowerbound(%c1_b : index) extent(%c3 : index)
        stride(%c2 : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<8xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// A dynamically-shaped accumulator (a strided view whose type conveys no size)
// is classified per-thread from par_dims: a thread dimension means per-thread
// storage, so lowering emits the per-element gpu.all_reduce.
//
// CHECK-LABEL: func.func @array_reduction_dynamic_par_dims
// CHECK: scf.for
// CHECK: gpu.all_reduce add
func.func @array_reduction_dynamic_par_dims(%buf: memref<?xi32>, %n: index) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg0 = %buf, %ext = %n) : (memref<?xi32>, index) {
    %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%ext], strides: [1]
        : memref<?xi32> to memref<?xi32, strided<[1]>>
    %bounds = acc.bounds extent(%ext : index)
    acc.reduction_accumulate_array %view bounds(%bounds) <add>
        : memref<?xi32, strided<[1]>>
        {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @rank_two_array_reduction
// CHECK: %[[R2_ALLOCA:.*]] = memref.alloca() : memref<2x3xi32>
// CHECK: scf.for %[[R2_I:.*]] =
// CHECK:   scf.for %[[R2_J:.*]] =
// CHECK:     memref.store %{{.*}}, %[[R2_ALLOCA]][%[[R2_I]], %[[R2_J]]] : memref<2x3xi32>
// CHECK: scf.for %[[R2_LINEAR:.*]] =
// CHECK: %[[R2_J_IDX:.*]] = arith.remui %[[R2_LINEAR]], %{{.*}} : index
// CHECK: %[[R2_ROW:.*]] = arith.divui %[[R2_LINEAR]], %{{.*}} : index
// CHECK: %[[R2_I_IDX:.*]] = arith.remui %[[R2_ROW]], %{{.*}} : index
// CHECK: memref.load %[[R2_ALLOCA]][%[[R2_I_IDX]], %[[R2_J_IDX]]] : memref<2x3xi32>
// CHECK: %[[R2_RESULT:.*]] = gpu.all_reduce add
// CHECK: memref.store %[[R2_RESULT]], %[[R2_ALLOCA]][%[[R2_I_IDX]], %[[R2_J_IDX]]]

func.func @rank_two_array_reduction() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize [#acc<par_dims[block_x, thread_x]>] : () -> !acc.private_type<memref<2x3xi32>>
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg0 = %private) : (!acc.private_type<memref<2x3xi32>>) {
    %c6 = arith.constant 6 : index
    %local = acc.private_local %arg0 {acc.par_dims = #acc<par_dims[block_x, thread_x]>} : (!acc.private_type<memref<2x3xi32>>) -> memref<2x3xi32>
    %bounds = acc.bounds extent(%c6 : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add> : memref<2x3xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @rank_three_array_reduction
// CHECK: %[[R3_ALLOCA:.*]] = memref.alloca() : memref<2x2x2xi32>
// CHECK: scf.for %[[R3_I:.*]] =
// CHECK:   scf.for %[[R3_J:.*]] =
// CHECK:     scf.for %[[R3_K:.*]] =
// CHECK:       memref.store %{{.*}}, %[[R3_ALLOCA]][%[[R3_I]], %[[R3_J]], %[[R3_K]]] : memref<2x2x2xi32>
// CHECK: scf.for %[[R3_LINEAR:.*]] =
// CHECK: %[[R3_K_IDX:.*]] = arith.remui %[[R3_LINEAR]], %{{.*}} : index
// CHECK: %[[R3_PLANE:.*]] = arith.divui %[[R3_LINEAR]], %{{.*}} : index
// CHECK: %[[R3_J_IDX:.*]] = arith.remui %[[R3_PLANE]], %{{.*}} : index
// CHECK: %[[R3_ROW:.*]] = arith.divui %[[R3_PLANE]], %{{.*}} : index
// CHECK: %[[R3_I_IDX:.*]] = arith.remui %[[R3_ROW]], %{{.*}} : index
// CHECK: memref.load %[[R3_ALLOCA]][%[[R3_I_IDX]], %[[R3_J_IDX]], %[[R3_K_IDX]]] : memref<2x2x2xi32>
// CHECK: %[[R3_RESULT:.*]] = gpu.all_reduce add
// CHECK: memref.store %[[R3_RESULT]], %[[R3_ALLOCA]][%[[R3_I_IDX]], %[[R3_J_IDX]], %[[R3_K_IDX]]]

func.func @rank_three_array_reduction() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  %private = acc.privatize [#acc<par_dims[block_x, thread_x]>] : () -> !acc.private_type<memref<2x2x2xi32>>
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg0 = %private) : (!acc.private_type<memref<2x2x2xi32>>) {
    %c8 = arith.constant 8 : index
    %local = acc.private_local %arg0 {acc.par_dims = #acc<par_dims[block_x, thread_x]>} : (!acc.private_type<memref<2x2x2xi32>>) -> memref<2x2x2xi32>
    %bounds = acc.bounds extent(%c8 : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add> : memref<2x2x2xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// Unsupported dynamic multi-rank accumulators must not generate invalid
// indexing operations.
// CHECK-LABEL: func.func @dynamic_rank_two_array_reduction
// CHECK: gpu.launch
// CHECK-NOT: memref.load
// CHECK: gpu.terminator
func.func @dynamic_rank_two_array_reduction(%local: memref<?x?xi32>, %extent: index) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg0 = %local, %ext = %extent) : (memref<?x?xi32>, index) {
    %bounds = acc.bounds extent(%ext : index)
    acc.reduction_accumulate_array %arg0 bounds(%bounds) <add> : memref<?x?xi32> {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: func.func @rank_two_partial_bounds_strided_layout
// CHECK: %[[LB:.*]] = arith.constant 5 : index
// CHECK: %[[STEP:.*]] = arith.constant 2 : index
// CHECK: %[[EXTENT:.*]] = arith.constant 3 : index
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<3x4xi32, strided<[8, 2]>>
// CHECK: %[[SPAN:.*]] = arith.muli %[[EXTENT]], %[[STEP]] : index
// CHECK: %[[UB:.*]] = arith.addi %[[LB]], %[[SPAN]] : index
// CHECK: scf.for %[[LINEAR:.*]] = %[[LB]] to %[[UB]] step %[[STEP]]
// CHECK: %[[COL:.*]] = arith.remui %[[LINEAR]], %{{.*}} : index
// CHECK: %[[ROW_LINEAR:.*]] = arith.divui %[[LINEAR]], %{{.*}} : index
// CHECK: %[[ROW:.*]] = arith.remui %[[ROW_LINEAR]], %{{.*}} : index
// CHECK: memref.load %[[ALLOCA]][%[[ROW]], %[[COL]]] : memref<3x4xi32, strided<[8, 2]>>
func.func @rank_two_partial_bounds_strided_layout() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) {
    %lb = arith.constant 5 : index
    %step = arith.constant 2 : index
    %extent = arith.constant 3 : index
    %local = memref.alloca() : memref<3x4xi32, strided<[8, 2]>>
    %bounds = acc.bounds lowerbound(%lb : index) extent(%extent : index)
        stride(%step : index)
    acc.reduction_accumulate_array %local bounds(%bounds) <add>
        : memref<3x4xi32, strided<[8, 2]>>
        {par_dims = #acc<par_dims[block_x, thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}
