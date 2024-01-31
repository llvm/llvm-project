// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(scf-parallel-loop-fusion))' -split-input-file | FileCheck %s

func.func @fuse_empty_loops() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @fuse_empty_loops
// CHECK-DAG:    [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG:    [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:    [[C1:%.*]] = arith.constant 1 : index
// CHECK:        scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:       to ([[C2]], [[C2]]) step ([[C1]], [[C1]]) {
// CHECK:          scf.reduce
// CHECK:        }
// CHECK-NOT:    scf.parallel

// -----

func.func @fuse_two(%A: memref<2x2xf32>, %B: memref<2x2xf32>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1fp = arith.constant 1.0 : f32
  %sum = memref.alloc()  : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    %sum_elem = arith.addf %B_elem, %c1fp : f32
    memref.store %sum_elem, %sum[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %sum_elem = memref.load %sum[%i, %j] : memref<2x2xf32>
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    %product_elem = arith.mulf %sum_elem, %A_elem : f32
    memref.store %product_elem, %B[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  memref.dealloc %sum : memref<2x2xf32>
  return
}
// CHECK-LABEL: func @fuse_two
// CHECK-SAME:   ([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}) {
// CHECK-DAG:  [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG:  [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:  [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:  [[C1FP:%.*]] = arith.constant 1.
// CHECK:      [[SUM:%.*]] = memref.alloc()
// CHECK:      scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:     to ([[C2]], [[C2]]) step ([[C1]], [[C1]]) {
// CHECK:        [[B_ELEM:%.*]] = memref.load [[B]]{{\[}}[[I]], [[J]]]
// CHECK:        [[SUM_ELEM:%.*]] = arith.addf [[B_ELEM]], [[C1FP]]
// CHECK:        memref.store [[SUM_ELEM]], [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK-NOT:  scf.parallel
// CHECK:        [[SUM_ELEM_:%.*]] = memref.load [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK:        [[A_ELEM:%.*]] = memref.load [[A]]{{\[}}[[I]], [[J]]]
// CHECK:        [[PRODUCT_ELEM:%.*]] = arith.mulf [[SUM_ELEM_]], [[A_ELEM]]
// CHECK:        memref.store [[PRODUCT_ELEM]], [[B]]{{\[}}[[I]], [[J]]]
// CHECK:        scf.reduce
// CHECK:      }
// CHECK:      memref.dealloc [[SUM]]

// -----

func.func @fuse_three(%A: memref<2x2xf32>, %B: memref<2x2xf32>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1fp = arith.constant 1.0 : f32
  %c2fp = arith.constant 2.0 : f32
  %sum = memref.alloc()  : memref<2x2xf32>
  %prod = memref.alloc()  : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    %sum_elem = arith.addf %B_elem, %c1fp : f32
    memref.store %sum_elem, %sum[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %sum_elem = memref.load %sum[%i, %j] : memref<2x2xf32>
    %product_elem = arith.mulf %sum_elem, %c2fp : f32
    memref.store %product_elem, %prod[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) { 
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    %res_elem = arith.addf %A_elem, %c2fp : f32
    memref.store %res_elem, %B[%i, %j] : memref<2x2xf32>
  }
  memref.dealloc %sum : memref<2x2xf32>
  memref.dealloc %prod : memref<2x2xf32>
  return
}
// CHECK-LABEL: func @fuse_three
// CHECK-SAME:   ([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}) {
// CHECK-DAG:  [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG:  [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:  [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:  [[C1FP:%.*]] = arith.constant 1.
// CHECK-DAG:  [[C2FP:%.*]] = arith.constant 2.
// CHECK:      [[SUM:%.*]] = memref.alloc()
// CHECK:      [[PROD:%.*]] = memref.alloc()
// CHECK:      scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:     to ([[C2]], [[C2]]) step ([[C1]], [[C1]]) {
// CHECK:        [[B_ELEM:%.*]] = memref.load [[B]]{{\[}}[[I]], [[J]]]
// CHECK:        [[SUM_ELEM:%.*]] = arith.addf [[B_ELEM]], [[C1FP]]
// CHECK:        memref.store [[SUM_ELEM]], [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK-NOT:  scf.parallel
// CHECK:        [[SUM_ELEM_:%.*]] = memref.load [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK:        [[PRODUCT_ELEM:%.*]] = arith.mulf [[SUM_ELEM_]], [[C2FP]]
// CHECK:        memref.store [[PRODUCT_ELEM]], [[PROD]]{{\[}}[[I]], [[J]]]
// CHECK-NOT:  scf.parallel
// CHECK:        [[A_ELEM:%.*]] = memref.load [[A]]{{\[}}[[I]], [[J]]]
// CHECK:        [[RES_ELEM:%.*]] = arith.addf [[A_ELEM]], [[C2FP]]
// CHECK:        memref.store [[RES_ELEM]], [[B]]{{\[}}[[I]], [[J]]]
// CHECK:        scf.reduce
// CHECK:      }
// CHECK:      memref.dealloc [[SUM]]
// CHECK:      memref.dealloc [[PROD]]

// -----

func.func @do_not_fuse_nested_ploop1() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.parallel (%k, %l) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      scf.reduce
    }
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_nested_ploop1
// CHECK:        scf.parallel
// CHECK:          scf.parallel
// CHECK:        scf.parallel

// -----

func.func @do_not_fuse_nested_ploop2() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.parallel (%k, %l) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      scf.reduce
    }
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_nested_ploop2
// CHECK:        scf.parallel
// CHECK:        scf.parallel
// CHECK:          scf.parallel

// -----

func.func @do_not_fuse_loops_unmatching_num_loops() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_loops_unmatching_num_loops
// CHECK:        scf.parallel
// CHECK:        scf.parallel

// -----

func.func @do_not_fuse_loops_with_side_effecting_ops_in_between() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.reduce
  }
  %buffer  = memref.alloc() : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_loops_with_side_effecting_ops_in_between
// CHECK:        scf.parallel
// CHECK:        scf.parallel

// -----

func.func @do_not_fuse_loops_unmatching_iteration_space() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c4, %c4) step (%c2, %c2) {
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_loops_unmatching_iteration_space
// CHECK:        scf.parallel
// CHECK:        scf.parallel

// -----

func.func @do_not_fuse_unmatching_write_read_patterns(
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
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %k = arith.addi %i, %c1 : index
    %sum_elem = memref.load %common_buf[%k, %j] : memref<2x2xf32>
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    %product_elem = arith.mulf %sum_elem, %A_elem : f32
    memref.store %product_elem, %result[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  memref.dealloc %common_buf : memref<2x2xf32>
  return
}
// CHECK-LABEL: func @do_not_fuse_unmatching_write_read_patterns
// CHECK:        scf.parallel
// CHECK:        scf.parallel

// -----

func.func @do_not_fuse_unmatching_read_write_patterns(
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
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %k = arith.addi %i, %c1 : index
    %sum_elem = memref.load %sum[%k, %j] : memref<2x2xf32>
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    %product_elem = arith.mulf %sum_elem, %A_elem : f32
    memref.store %product_elem, %common_buf[%j, %i] : memref<2x2xf32>
    scf.reduce
  }
  memref.dealloc %sum : memref<2x2xf32>
  return
}
// CHECK-LABEL: func @do_not_fuse_unmatching_read_write_patterns
// CHECK:        scf.parallel
// CHECK:        scf.parallel

// -----

func.func @do_not_fuse_loops_with_memref_defined_in_loop_bodies() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %buffer  = memref.alloc() : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %A = memref.subview %buffer[%c0, %c0][%c2, %c2][%c1, %c1]
      : memref<2x2xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %A_elem = memref.load %A[%i, %j] : memref<?x?xf32, strided<[?, ?], offset: ?>>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_loops_with_memref_defined_in_loop_bodies
// CHECK:        scf.parallel
// CHECK:        scf.parallel

// -----

func.func @nested_fuse(%A: memref<2x2xf32>, %B: memref<2x2xf32>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1fp = arith.constant 1.0 : f32
  %sum = memref.alloc()  : memref<2x2xf32>
  scf.parallel (%k) = (%c0) to (%c2) step (%c1) {
    scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
      %sum_elem = arith.addf %B_elem, %c1fp : f32
      memref.store %sum_elem, %sum[%i, %j] : memref<2x2xf32>
      scf.reduce
    }
    scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      %sum_elem = memref.load %sum[%i, %j] : memref<2x2xf32>
      %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
      %product_elem = arith.mulf %sum_elem, %A_elem : f32
      memref.store %product_elem, %B[%i, %j] : memref<2x2xf32>
      scf.reduce
    }
  }
  memref.dealloc %sum : memref<2x2xf32>
  return
}
// CHECK-LABEL: func @nested_fuse
// CHECK-SAME:   ([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}) {
// CHECK-DAG:  [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG:  [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:  [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:  [[C1FP:%.*]] = arith.constant 1.
// CHECK:      [[SUM:%.*]] = memref.alloc()
// CHECK:      scf.parallel
// CHECK:        scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:       to ([[C2]], [[C2]]) step ([[C1]], [[C1]]) {
// CHECK:          [[B_ELEM:%.*]] = memref.load [[B]]{{\[}}[[I]], [[J]]]
// CHECK:          [[SUM_ELEM:%.*]] = arith.addf [[B_ELEM]], [[C1FP]]
// CHECK:          memref.store [[SUM_ELEM]], [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK-NOT:   scf.parallel
// CHECK:          [[SUM_ELEM_:%.*]] = memref.load [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK:          [[A_ELEM:%.*]] = memref.load [[A]]{{\[}}[[I]], [[J]]]
// CHECK:          [[PRODUCT_ELEM:%.*]] = arith.mulf [[SUM_ELEM_]], [[A_ELEM]]
// CHECK:          memref.store [[PRODUCT_ELEM]], [[B]]{{\[}}[[I]], [[J]]]
// CHECK:          scf.reduce
// CHECK:        }
// CHECK:      }
// CHECK:      memref.dealloc [[SUM]]

// -----

func.func @do_not_fuse_alias(%A: memref<2x2xf32>, %B: memref<2x2xf32>,
                             %C: memref<2x2xf32>, %result: memref<2x2xf32>,
                             %sum: memref<2x2xf32>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    %C_elem = memref.load %C[%i, %j] : memref<2x2xf32>
    %sum_elem = arith.addf %B_elem, %C_elem : f32
    memref.store %sum_elem, %sum[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %sum_elem = memref.load %sum[%i, %j] : memref<2x2xf32>
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    %product_elem = arith.mulf %sum_elem, %A_elem : f32
    memref.store %product_elem, %result[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  return
}
// %sum and %result may alias with other args, do not fuse loops
// CHECK-LABEL: func @do_not_fuse_alias
// CHECK:      scf.parallel
// CHECK:      scf.parallel

// -----

func.func @fuse_when_1st_has_multiple_stores(
  %A: memref<2x2xf32>, %B: memref<2x2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0fp = arith.constant 0.0 : f32
  %sum = memref.alloc()  : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    memref.store %c0fp, %sum[%i, %j] : memref<2x2xf32>
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    %sum_elem = arith.addf %B_elem, %B_elem : f32
    memref.store %sum_elem, %sum[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %sum_elem = memref.load %sum[%i, %j] : memref<2x2xf32>
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    %product_elem = arith.mulf %sum_elem, %A_elem : f32
    memref.store %product_elem, %B[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  memref.dealloc %sum : memref<2x2xf32>
  return
}
// CHECK-LABEL: func @fuse_when_1st_has_multiple_stores
// CHECK-SAME:   ([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}) {
// CHECK-DAG:  [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:  [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:  [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG:  [[C0F32:%.*]] = arith.constant 0.
// CHECK:      [[SUM:%.*]] = memref.alloc()
// CHECK:      scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:     to ([[C2]], [[C2]]) step ([[C1]], [[C1]]) {
// CHECK:        [[B_ELEM:%.*]] = memref.load [[B]]{{\[}}[[I]], [[J]]]
// CHECK:        [[SUM_ELEM:%.*]] = arith.addf [[B_ELEM]], [[B_ELEM]]
// CHECK:        memref.store [[SUM_ELEM]], [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK-NOT:  scf.parallel
// CHECK:        [[SUM_ELEM:%.*]] = memref.load [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK:        [[A_ELEM:%.*]] = memref.load [[A]]{{\[}}[[I]], [[J]]]
// CHECK:        [[PRODUCT_ELEM:%.*]] = arith.mulf
// CHECK:        memref.store [[PRODUCT_ELEM]], [[B]]{{\[}}[[I]], [[J]]]
// CHECK:        scf.reduce
// CHECK:      }
// CHECK:      memref.dealloc [[SUM]]

// -----

func.func @do_not_fuse_multiple_stores_on_diff_indices(
  %A: memref<2x2xf32>, %B: memref<2x2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0fp = arith.constant 0.0 : f32
  %sum = memref.alloc()  : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    memref.store %c0fp, %sum[%i, %j] : memref<2x2xf32>
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    %sum_elem = arith.addf %B_elem, %B_elem : f32
    memref.store %sum_elem, %sum[%c0, %j] : memref<2x2xf32>
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %sum_elem = memref.load %sum[%i, %j] : memref<2x2xf32>
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    %product_elem = arith.mulf %sum_elem, %A_elem : f32
    memref.store %product_elem, %B[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  memref.dealloc %sum : memref<2x2xf32>
  return
}
// CHECK-LABEL: func @do_not_fuse_multiple_stores_on_diff_indices
// CHECK-SAME:   ([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}) {
// CHECK-DAG:  [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:  [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:  [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG:  [[C0F32:%.*]] = arith.constant 0.
// CHECK:      [[SUM:%.*]] = memref.alloc()
// CHECK:      scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:     to ([[C2]], [[C2]]) step ([[C1]], [[C1]]) {
// CHECK:        [[B_ELEM:%.*]] = memref.load [[B]]{{\[}}[[I]], [[J]]]
// CHECK:        [[SUM_ELEM:%.*]] = arith.addf [[B_ELEM]], [[B_ELEM]]
// CHECK:        memref.store [[SUM_ELEM]], [[SUM]]{{\[}}[[C0]], [[J]]]
// CHECK:        scf.reduce
// CHECK:     scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK:        [[SUM_ELEM:%.*]] = memref.load [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK:        [[A_ELEM:%.*]] = memref.load [[A]]{{\[}}[[I]], [[J]]]
// CHECK:        [[PRODUCT_ELEM:%.*]] = arith.mulf
// CHECK:        memref.store [[PRODUCT_ELEM]], [[B]]{{\[}}[[I]], [[J]]]
// CHECK:        scf.reduce
// CHECK:      }
// CHECK:      memref.dealloc [[SUM]]
