// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.func(scf-parallel-loop-fusion)' -split-input-file | FileCheck %s

func @fuse_empty_loops() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.yield
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.yield
  }
  return
}
// CHECK-LABEL: func @fuse_empty_loops
// CHECK:        [[C2:%.*]] = arith.constant 2 : index
// CHECK:        [[C0:%.*]] = arith.constant 0 : index
// CHECK:        [[C1:%.*]] = arith.constant 1 : index
// CHECK:        scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:       to ([[C2]], [[C2]]) step ([[C1]], [[C1]]) {
// CHECK:          scf.yield
// CHECK:        }
// CHECK-NOT:    scf.parallel

// -----

func @fuse_two(%A: memref<2x2xf32>, %B: memref<2x2xf32>,
                    %C: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sum = memref.alloc()  : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    %C_elem = memref.load %C[%i, %j] : memref<2x2xf32>
    %sum_elem = arith.addf %B_elem, %C_elem : f32
    memref.store %sum_elem, %sum[%i, %j] : memref<2x2xf32>
    scf.yield
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %sum_elem = memref.load %sum[%i, %j] : memref<2x2xf32>
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    %product_elem = arith.mulf %sum_elem, %A_elem : f32
    memref.store %product_elem, %result[%i, %j] : memref<2x2xf32>
    scf.yield
  }
  memref.dealloc %sum : memref<2x2xf32>
  return
}
// CHECK-LABEL: func @fuse_two
// CHECK-SAME:   ([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}, [[C:%.*]]: {{.*}},
// CHECK-SAME:    [[RESULT:%.*]]: {{.*}}) {
// CHECK:      [[C2:%.*]] = arith.constant 2 : index
// CHECK:      [[C0:%.*]] = arith.constant 0 : index
// CHECK:      [[C1:%.*]] = arith.constant 1 : index
// CHECK:      [[SUM:%.*]] = memref.alloc()
// CHECK:      scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:     to ([[C2]], [[C2]]) step ([[C1]], [[C1]]) {
// CHECK:        [[B_ELEM:%.*]] = memref.load [[B]]{{\[}}[[I]], [[J]]]
// CHECK:        [[C_ELEM:%.*]] = memref.load [[C]]{{\[}}[[I]], [[J]]]
// CHECK:        [[SUM_ELEM:%.*]] = arith.addf [[B_ELEM]], [[C_ELEM]]
// CHECK:        memref.store [[SUM_ELEM]], [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK:        [[SUM_ELEM_:%.*]] = memref.load [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK:        [[A_ELEM:%.*]] = memref.load [[A]]{{\[}}[[I]], [[J]]]
// CHECK:        [[PRODUCT_ELEM:%.*]] = arith.mulf [[SUM_ELEM_]], [[A_ELEM]]
// CHECK:        memref.store [[PRODUCT_ELEM]], [[RESULT]]{{\[}}[[I]], [[J]]]
// CHECK:        scf.yield
// CHECK:      }
// CHECK:      memref.dealloc [[SUM]]

// -----

func @fuse_three(%lhs: memref<100x10xf32>, %rhs: memref<100xf32>,
                      %result: memref<100x10xf32>) {
  %c100 = arith.constant 100 : index
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %broadcast_rhs = memref.alloc() : memref<100x10xf32>
  %diff = memref.alloc() : memref<100x10xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c100, %c10) step (%c1, %c1) {
    %rhs_elem = memref.load %rhs[%i] : memref<100xf32>
    memref.store %rhs_elem, %broadcast_rhs[%i, %j] : memref<100x10xf32>
    scf.yield
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c100, %c10) step (%c1, %c1) {
    %lhs_elem = memref.load %lhs[%i, %j] : memref<100x10xf32>
    %broadcast_rhs_elem = memref.load %broadcast_rhs[%i, %j] : memref<100x10xf32>
    %diff_elem = arith.subf %lhs_elem, %broadcast_rhs_elem : f32
    memref.store %diff_elem, %diff[%i, %j] : memref<100x10xf32>
    scf.yield
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c100, %c10) step (%c1, %c1) {
    %diff_elem = memref.load %diff[%i, %j] : memref<100x10xf32>
    %exp_elem = math.exp %diff_elem : f32
    memref.store %exp_elem, %result[%i, %j] : memref<100x10xf32>
    scf.yield
  }
  memref.dealloc %broadcast_rhs : memref<100x10xf32>
  memref.dealloc %diff : memref<100x10xf32>
  return
}
// CHECK-LABEL: func @fuse_three
// CHECK-SAME: ([[LHS:%.*]]: memref<100x10xf32>, [[RHS:%.*]]: memref<100xf32>,
// CHECK-SAME:  [[RESULT:%.*]]: memref<100x10xf32>) {
// CHECK:      [[C100:%.*]] = arith.constant 100 : index
// CHECK:      [[C10:%.*]] = arith.constant 10 : index
// CHECK:      [[C0:%.*]] = arith.constant 0 : index
// CHECK:      [[C1:%.*]] = arith.constant 1 : index
// CHECK:      [[BROADCAST_RHS:%.*]] = memref.alloc()
// CHECK:      [[DIFF:%.*]] = memref.alloc()
// CHECK:      scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:     to ([[C100]], [[C10]]) step ([[C1]], [[C1]]) {
// CHECK:        [[RHS_ELEM:%.*]] = memref.load [[RHS]]{{\[}}[[I]]]
// CHECK:        memref.store [[RHS_ELEM]], [[BROADCAST_RHS]]{{\[}}[[I]], [[J]]]
// CHECK:        [[LHS_ELEM:%.*]] = memref.load [[LHS]]{{\[}}[[I]], [[J]]]
// CHECK:        [[BROADCAST_RHS_ELEM:%.*]] = memref.load [[BROADCAST_RHS]]
// CHECK:        [[DIFF_ELEM:%.*]] = arith.subf [[LHS_ELEM]], [[BROADCAST_RHS_ELEM]]
// CHECK:        memref.store [[DIFF_ELEM]], [[DIFF]]{{\[}}[[I]], [[J]]]
// CHECK:        [[DIFF_ELEM_:%.*]] = memref.load [[DIFF]]{{\[}}[[I]], [[J]]]
// CHECK:        [[EXP_ELEM:%.*]] = math.exp [[DIFF_ELEM_]]
// CHECK:        memref.store [[EXP_ELEM]], [[RESULT]]{{\[}}[[I]], [[J]]]
// CHECK:        scf.yield
// CHECK:      }
// CHECK:      memref.dealloc [[BROADCAST_RHS]]
// CHECK:      memref.dealloc [[DIFF]]

// -----

func @do_not_fuse_nested_ploop1() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.parallel (%k, %l) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.yield
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_nested_ploop1
// CHECK:        scf.parallel
// CHECK:          scf.parallel
// CHECK:        scf.parallel

// -----

func @do_not_fuse_nested_ploop2() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.yield
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.parallel (%k, %l) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      scf.yield
    }
    scf.yield
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_nested_ploop2
// CHECK:        scf.parallel
// CHECK:        scf.parallel
// CHECK:          scf.parallel

// -----

func @do_not_fuse_loops_unmatching_num_loops() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.yield
  }
  scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
    scf.yield
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_loops_unmatching_num_loops
// CHECK:        scf.parallel
// CHECK:        scf.parallel

// -----

func @do_not_fuse_loops_with_side_effecting_ops_in_between() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.yield
  }
  %buffer  = memref.alloc() : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.yield
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_loops_with_side_effecting_ops_in_between
// CHECK:        scf.parallel
// CHECK:        scf.parallel

// -----

func @do_not_fuse_loops_unmatching_iteration_space() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c4, %c4) step (%c2, %c2) {
    scf.yield
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.yield
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_loops_unmatching_iteration_space
// CHECK:        scf.parallel
// CHECK:        scf.parallel

// -----

func @do_not_fuse_unmatching_write_read_patterns(
    %A: memref<2x2xf32>, %B: memref<2x2xf32>,
    %C: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %common_buf = memref.alloc() : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    %C_elem = memref.load %C[%i, %j] : memref<2x2xf32>
    %sum_elem = arith.addf %B_elem, %C_elem : f32
    memref.store %sum_elem, %common_buf[%i, %j] : memref<2x2xf32>
    scf.yield
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %k = arith.addi %i, %c1 : index
    %sum_elem = memref.load %common_buf[%k, %j] : memref<2x2xf32>
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    %product_elem = arith.mulf %sum_elem, %A_elem : f32
    memref.store %product_elem, %result[%i, %j] : memref<2x2xf32>
    scf.yield
  }
  memref.dealloc %common_buf : memref<2x2xf32>
  return
}
// CHECK-LABEL: func @do_not_fuse_unmatching_write_read_patterns
// CHECK:        scf.parallel
// CHECK:        scf.parallel

// -----

func @do_not_fuse_unmatching_read_write_patterns(
    %A: memref<2x2xf32>, %B: memref<2x2xf32>, %common_buf: memref<2x2xf32>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sum = memref.alloc() : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    %C_elem = memref.load %common_buf[%i, %j] : memref<2x2xf32>
    %sum_elem = arith.addf %B_elem, %C_elem : f32
    memref.store %sum_elem, %sum[%i, %j] : memref<2x2xf32>
    scf.yield
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %k = arith.addi %i, %c1 : index
    %sum_elem = memref.load %sum[%k, %j] : memref<2x2xf32>
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    %product_elem = arith.mulf %sum_elem, %A_elem : f32
    memref.store %product_elem, %common_buf[%j, %i] : memref<2x2xf32>
    scf.yield
  }
  memref.dealloc %sum : memref<2x2xf32>
  return
}
// CHECK-LABEL: func @do_not_fuse_unmatching_read_write_patterns
// CHECK:        scf.parallel
// CHECK:        scf.parallel

// -----

func @do_not_fuse_loops_with_memref_defined_in_loop_bodies() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %buffer  = memref.alloc() : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.yield
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %A = memref.subview %buffer[%c0, %c0][%c2, %c2][%c1, %c1]
      : memref<2x2xf32> to memref<?x?xf32, offset: ?, strides:[?, ?]>
    %A_elem = memref.load %A[%i, %j] : memref<?x?xf32, offset: ?, strides:[?, ?]>
    scf.yield
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_loops_with_memref_defined_in_loop_bodies
// CHECK:        scf.parallel
// CHECK:        scf.parallel

// -----

func @nested_fuse(%A: memref<2x2xf32>, %B: memref<2x2xf32>,
                    %C: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sum = memref.alloc()  : memref<2x2xf32>
  scf.parallel (%k) = (%c0) to (%c2) step (%c1) {
    scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
      %C_elem = memref.load %C[%i, %j] : memref<2x2xf32>
      %sum_elem = arith.addf %B_elem, %C_elem : f32
      memref.store %sum_elem, %sum[%i, %j] : memref<2x2xf32>
      scf.yield
    }
    scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      %sum_elem = memref.load %sum[%i, %j] : memref<2x2xf32>
      %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
      %product_elem = arith.mulf %sum_elem, %A_elem : f32
      memref.store %product_elem, %result[%i, %j] : memref<2x2xf32>
      scf.yield
    }
  }
  memref.dealloc %sum : memref<2x2xf32>
  return
}
// CHECK-LABEL: func @nested_fuse
// CHECK-SAME:   ([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}, [[C:%.*]]: {{.*}},
// CHECK-SAME:    [[RESULT:%.*]]: {{.*}}) {
// CHECK:      [[C2:%.*]] = arith.constant 2 : index
// CHECK:      [[C0:%.*]] = arith.constant 0 : index
// CHECK:      [[C1:%.*]] = arith.constant 1 : index
// CHECK:      [[SUM:%.*]] = memref.alloc()
// CHECK:      scf.parallel
// CHECK:        scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:       to ([[C2]], [[C2]]) step ([[C1]], [[C1]]) {
// CHECK:          [[B_ELEM:%.*]] = memref.load [[B]]{{\[}}[[I]], [[J]]]
// CHECK:          [[C_ELEM:%.*]] = memref.load [[C]]{{\[}}[[I]], [[J]]]
// CHECK:          [[SUM_ELEM:%.*]] = arith.addf [[B_ELEM]], [[C_ELEM]]
// CHECK:          memref.store [[SUM_ELEM]], [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK:          [[SUM_ELEM_:%.*]] = memref.load [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK:          [[A_ELEM:%.*]] = memref.load [[A]]{{\[}}[[I]], [[J]]]
// CHECK:          [[PRODUCT_ELEM:%.*]] = arith.mulf [[SUM_ELEM_]], [[A_ELEM]]
// CHECK:          memref.store [[PRODUCT_ELEM]], [[RESULT]]{{\[}}[[I]], [[J]]]
// CHECK:          scf.yield
// CHECK:        }
// CHECK:      }
// CHECK:      memref.dealloc [[SUM]]
