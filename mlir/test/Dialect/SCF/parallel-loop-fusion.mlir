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

func.func @fuse_ops_between(%A: f32, %B: f32) -> f32 {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.reduce
  }
  %res = arith.addf %A, %B : f32
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.reduce
  }
  return %res : f32
}
// CHECK-LABEL: func @fuse_ops_between
// CHECK-DAG:    [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:    [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:    [[C2:%.*]] = arith.constant 2 : index
// CHECK:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
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

func.func @do_not_fuse_loops_with_nonfull_alias_defined_in_loop_bodies() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1fp = arith.constant 1.0 : f32
  %buffer  = memref.alloc() : memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c1) step (%c1, %c1) {
    memref.store %c1fp, %buffer[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c1) step (%c1, %c1) {
    %A = memref.subview %buffer[%i, %c0][2, 1][1, 1] : memref<2x2xf32> to memref<2x1xf32, strided<[2, 1], offset: ?>>
    %A_elem = memref.load %A[%i, %j] : memref<2x1xf32, strided<[2, 1], offset: ?>>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_loops_with_nonfull_alias_defined_in_loop_bodies
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

// -----

func.func @fuse_same_indices_by_affine_apply(
  %A: memref<2x2xf32>, %B: memref<2x2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %sum = memref.alloc()  : memref<2x3xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    %1 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%i, %j)
    memref.store %B_elem, %sum[%i, %1] : memref<2x3xf32>
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %1 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%i, %j)
    %sum_elem = memref.load %sum[%i, %1] : memref<2x3xf32>
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    %product = arith.mulf %sum_elem, %A_elem : f32
    memref.store %product, %B[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  memref.dealloc %sum : memref<2x3xf32>
  return
}
// CHECK:      #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: fuse_same_indices_by_affine_apply
// CHECK-SAME:  (%[[ARG0:.*]]: memref<2x2xf32>, %[[ARG1:.*]]: memref<2x2xf32>) {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[ALLOC:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK-NEXT:  scf.parallel (%[[ARG2:.*]], %[[ARG3:.*]]) = (%[[C0]], %[[C0]]) to (%[[C2]], %[[C2]]) step (%[[C1]], %[[C1]]) {
// CHECK-NEXT:    %[[S0:.*]] = memref.load %[[ARG1]][%[[ARG2]], %[[ARG3]]] : memref<2x2xf32>
// CHECK-NEXT:    %[[S1:.*]] = affine.apply #[[$MAP]](%[[ARG2]], %[[ARG3]])
// CHECK-NEXT:    memref.store %[[S0]], %[[ALLOC]][%[[ARG2]], %[[S1]]] : memref<2x3xf32>
// CHECK-NEXT:    %[[S2:.*]] = affine.apply #[[$MAP]](%[[ARG2]], %[[ARG3]])
// CHECK-NEXT:    %[[S3:.*]] = memref.load %[[ALLOC]][%[[ARG2]], %[[S2]]] : memref<2x3xf32>
// CHECK-NEXT:    %[[S4:.*]] = memref.load %[[ARG0]][%[[ARG2]], %[[ARG3]]] : memref<2x2xf32>
// CHECK-NEXT:    %[[S5:.*]] = arith.mulf %[[S3]], %[[S4]] : f32
// CHECK-NEXT:    memref.store %[[S5]], %[[ARG1]][%[[ARG2]], %[[ARG3]]] : memref<2x2xf32>
// CHECK-NEXT:    scf.reduce
// CHECK-NEXT:  }
// CHECK-NEXT:  memref.dealloc %[[ALLOC]] : memref<2x3xf32>
// CHECK-NEXT:  return

// -----

func.func @do_not_fuse_affine_apply_to_non_ind_var(
  %A: memref<2x2xf32>, %B: memref<2x2xf32>, %OffsetA: index, %OffsetB: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %sum = memref.alloc()  : memref<2x3xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    %1 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%i, %OffsetA)
    memref.store %B_elem, %sum[%i, %1] : memref<2x3xf32>
    scf.reduce
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %1 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%i, %OffsetB)
    %sum_elem = memref.load %sum[%i, %1] : memref<2x3xf32>
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    %product = arith.mulf %sum_elem, %A_elem : f32
    memref.store %product, %B[%i, %j] : memref<2x2xf32>
    scf.reduce
  }
  memref.dealloc %sum : memref<2x3xf32>
  return
}
// CHECK:       #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: do_not_fuse_affine_apply_to_non_ind_var
// CHECK-SAME:  (%[[ARG0:.*]]: memref<2x2xf32>, %[[ARG1:.*]]: memref<2x2xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK-NEXT:    scf.parallel (%[[ARG4:.*]], %[[ARG5:.*]]) = (%[[C0]], %[[C0]]) to (%[[C2]], %[[C2]]) step (%[[C1]], %[[C1]]) {
// CHECK-NEXT:      %[[S0:.*]] = memref.load %[[ARG1]][%[[ARG4]], %[[ARG5]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[S1:.*]] = affine.apply #[[$MAP]](%[[ARG4]], %[[ARG2]])
// CHECK-NEXT:      memref.store %[[S0]], %[[ALLOC]][%[[ARG4]], %[[S1]]] : memref<2x3xf32>
// CHECK-NEXT:      scf.reduce
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.parallel (%[[ARG4:.*]], %[[ARG5:.*]]) = (%[[C0]], %[[C0]]) to (%[[C2]], %[[C2]]) step (%[[C1]], %[[C1]]) {
// CHECK-NEXT:      %[[S0:.*]] = affine.apply #[[$MAP]](%[[ARG4]], %[[ARG3]])
// CHECK-NEXT:      %[[S1:.*]] = memref.load %[[ALLOC]][%[[ARG4]], %[[S0]]] : memref<2x3xf32>
// CHECK-NEXT:      %[[S2:.*]] = memref.load %[[ARG0]][%[[ARG4]], %[[ARG5]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[S3:.*]] = arith.mulf %[[S1]], %[[S2]] : f32
// CHECK-NEXT:      memref.store %[[S3]], %[[ARG1]][%[[ARG4]], %[[ARG5]]] : memref<2x2xf32>
// CHECK-NEXT:      scf.reduce
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %[[ALLOC]] : memref<2x3xf32>
// CHECK-NEXT:    return

// -----

func.func @fuse_trivial_rank_reducing_subview() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1fp = arith.constant 1.0 : f32
  %buf = memref.alloc() : memref<1x2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    memref.store %c1fp, %buf[%c0, %i, %j] : memref<1x2x2xf32>
    scf.reduce
  }
  %sub = memref.subview %buf[0, 0, 0][1, 2, 2][1, 1, 1]
      : memref<1x2x2xf32> to memref<2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %v = memref.load %sub[%i, %j] : memref<2x2xf32>
    memref.store %v, %buf[%c0, %i, %j] : memref<1x2x2xf32>
    scf.reduce
  }
  memref.dealloc %buf : memref<1x2x2xf32>
  return
}
// CHECK-LABEL: func @fuse_trivial_rank_reducing_subview
// CHECK:       %[[BUF:.*]] = memref.alloc() : memref<1x2x2xf32>
// CHECK:       %[[SUB:.*]] = memref.subview %[[BUF]]
// CHECK:       scf.parallel
// CHECK:         memref.store {{.*}}, %[[BUF]]
// CHECK:         %[[L:.*]] = memref.load %[[SUB]]
// CHECK:         memref.store %[[L]], %[[BUF]]
// CHECK-NOT:   scf.parallel
// CHECK:       memref.dealloc %[[BUF]] : memref<1x2x2xf32>

// -----

func.func @do_not_fuse_nontrivial_subview_offset() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1fp = arith.constant 1.0 : f32
  %buf = memref.alloc() : memref<2x2x2xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    memref.store %c1fp, %buf[%c0, %i, %j] : memref<2x2x2xf32>
    scf.reduce
  }
  %sub = memref.subview %buf[1, 0, 0][1, 2, 2][1, 1, 1]
      : memref<2x2x2xf32> to memref<2x2xf32, strided<[2, 1], offset: 4>>
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %v = memref.load %sub[%i, %j]
        : memref<2x2xf32, strided<[2, 1], offset: 4>>
    memref.store %v, %buf[%c0, %i, %j] : memref<2x2x2xf32>
    scf.reduce
  }
  memref.dealloc %buf : memref<2x2x2xf32>
  return
}
// CHECK-LABEL: func @do_not_fuse_nontrivial_subview_offset
// CHECK:       scf.parallel
// CHECK:       scf.parallel

// -----

func.func @fuse_vector_load_store(%A: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %vec0 = arith.constant dense<0.0> : vector<4xf32>
  scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
    vector.store %vec0, %A[%i, %c0] : memref<4x4xf32>, vector<4xf32>
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
    %v = vector.load %A[%i, %c0] : memref<4x4xf32>, vector<4xf32>
    vector.store %v, %A[%i, %c0] : memref<4x4xf32>, vector<4xf32>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @fuse_vector_load_store
// CHECK:       scf.parallel (%[[I:.*]]) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// CHECK:         vector.store
// CHECK:         %[[V:.*]] = vector.load
// CHECK:         vector.store %[[V]]
// CHECK-NOT:   scf.parallel

// -----

func.func @do_not_fuse_vector_different_indices(%A: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %vec0 = arith.constant dense<0.0> : vector<4xf32>
  scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
    vector.store %vec0, %A[%i, %c0] : memref<4x4xf32>, vector<4xf32>
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
    %j = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
    %v = vector.load %A[%j, %c0] : memref<4x4xf32>, vector<4xf32>
    vector.store %v, %A[%i, %c0] : memref<4x4xf32>, vector<4xf32>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_vector_different_indices
// CHECK:       scf.parallel
// CHECK:       scf.parallel

// -----

func.func @fuse_vector_transfer_same_indices(%A: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
    %v = vector.transfer_read %A[%i, %c0], %zero {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : memref<4x4xf32>, vector<4xf32>
    vector.transfer_write %v, %A[%i, %c0] {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : vector<4xf32>, memref<4x4xf32>
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
    %v = vector.transfer_read %A[%i, %c0], %zero {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : memref<4x4xf32>, vector<4xf32>
    vector.transfer_write %v, %A[%i, %c0] {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : vector<4xf32>, memref<4x4xf32>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @fuse_vector_transfer_same_indices
// CHECK:       scf.parallel
// CHECK:         vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:         vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:         vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:         vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK-NOT:   scf.parallel

// -----

func.func @do_not_fuse_vector_transfer_different_indices(%A: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
    %v = vector.transfer_read %A[%i, %c0], %zero {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : memref<4x4xf32>, vector<4xf32>
    vector.transfer_write %v, %A[%i, %c0] {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : vector<4xf32>, memref<4x4xf32>
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
    %j = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
    %v = vector.transfer_read %A[%j, %c0], %zero {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : memref<4x4xf32>, vector<4xf32>
    vector.transfer_write %v, %A[%i, %c0] {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : vector<4xf32>, memref<4x4xf32>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_vector_transfer_different_indices
// CHECK:       scf.parallel
// CHECK:       scf.parallel

// -----

func.func @fuse_vector_transfer_with_subview(%A: memref<1x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %vec = arith.constant dense<1.0> : vector<4xf32>
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %sub = memref.subview %A[0, 0][1, 4][1, 1] : memref<1x4xf32> to memref<4xf32>
    vector.transfer_write %vec, %sub[%c0] {permutation_map = affine_map<(d0) -> (d0)>, in_bounds = [true]} : vector<4xf32>, memref<4xf32>
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %sum = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> f32 {
      %v = memref.load %A[%c0, %k] : memref<1x4xf32>
      %n = arith.addf %v, %acc : f32
      scf.yield %n : f32
    }
    memref.store %sum, %A[%c0, %c0] : memref<1x4xf32>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @fuse_vector_transfer_with_subview
// CHECK:       scf.parallel
// CHECK:         vector.transfer_write
// CHECK:         scf.for
// CHECK-NOT:   scf.parallel

// -----

func.func @do_not_fuse_vector_transfer_nontrivial_subview(%A: memref<2x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %v = vector.transfer_read %A[%c0, %i], %zero {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : memref<2x4xf32>, vector<1xf32>
    vector.transfer_write %v, %A[%c0, %i] {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : vector<1xf32>, memref<2x4xf32>
    scf.reduce
  }
    %sub = memref.subview %A[1, 0][1, 4][1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: 4>>
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %v = vector.transfer_read %sub[%i], %zero {in_bounds = [true]} : memref<4xf32, strided<[1], offset: 4>>, vector<1xf32>
    vector.transfer_write %v, %sub[%i] {in_bounds = [true]} : vector<1xf32>, memref<4xf32, strided<[1], offset: 4>>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_vector_transfer_nontrivial_subview
// CHECK:       scf.parallel
// CHECK:       scf.parallel

// -----

func.func @do_not_fuse_vector_transfer_different_masks(%A: memref<1x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %mask_true = vector.create_mask %c1 : vector<1xi1>
  %mask_false = vector.create_mask %c0 : vector<1xi1>
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %v = vector.transfer_read %A[%c0, %i], %zero, %mask_true {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : memref<1x4xf32>, vector<1xf32>
    vector.transfer_write %v, %A[%c0, %i], %mask_true {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : vector<1xf32>, memref<1x4xf32>
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %v = vector.transfer_read %A[%c0, %i], %zero, %mask_false {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : memref<1x4xf32>, vector<1xf32>
    vector.transfer_write %v, %A[%c0, %i], %mask_false {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : vector<1xf32>, memref<1x4xf32>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_vector_transfer_different_masks
// CHECK:       scf.parallel
// CHECK:       scf.parallel

// -----

func.func @fuse_vector_transfer_subview_rank_reducing(%A: memref<1x4xf32>, %B: memref<1x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %vec = arith.constant dense<1.0> : vector<4xf32>
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %sub = memref.subview %A[%i, %c0][1, 4][1, 1] : memref<1x4xf32> to memref<4xf32, strided<[1], offset: ?>>
    vector.transfer_write %vec, %sub[%c0] {permutation_map = affine_map<(d0) -> (d0)>, in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: ?>>
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %sum = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> f32 {
      %v = memref.load %A[%i, %k] : memref<1x4xf32>
      %n = arith.addf %v, %acc : f32
      scf.yield %n : f32
    }
    memref.store %sum, %B[%i, %c0] : memref<1x4xf32>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @fuse_vector_transfer_subview_rank_reducing
// CHECK:       scf.parallel
// CHECK:         vector.transfer_write
// CHECK:         scf.for
// CHECK-NOT:   scf.parallel

// -----

func.func @do_not_fuse_vector_transfer_subview_offset(%A: memref<1x4xf32>, %B: memref<1x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %vec = arith.constant dense<1.0> : vector<4xf32>
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %sub = memref.subview %A[%i, %c0][1, 4][1, 1] : memref<1x4xf32> to memref<4xf32, strided<[1], offset: ?>>
    vector.transfer_write %vec, %sub[%c0] {permutation_map = affine_map<(d0) -> (d0)>, in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: ?>>
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %sum = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> f32 {
      %v = memref.load %A[%i, %k] : memref<1x4xf32>
      %n = arith.addf %v, %acc : f32
      scf.yield %n : f32
    }
    // Read from an offset alias to prevent fusion.
    %off = memref.subview %A[%i, %c1][1, 3][1, 1] : memref<1x4xf32> to memref<3xf32, strided<[1], offset: ?>>
    %v0 = memref.load %off[%c0] : memref<3xf32, strided<[1], offset: ?>>
    %res = arith.addf %sum, %v0 : f32
    memref.store %res, %B[%i, %c0] : memref<1x4xf32>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @do_not_fuse_vector_transfer_subview_offset
// CHECK:       scf.parallel
// CHECK:       scf.parallel

// -----

func.func @fuse_vector_transfer_no_subview(%A: memref<1x4xf32>, %B: memref<1x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %vec = arith.constant dense<2.0> : vector<4xf32>
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    vector.transfer_write %vec, %A[%c0, %i] {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]} : vector<4xf32>, memref<1x4xf32>
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %sum = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> f32 {
      %v = memref.load %A[%c0, %k] : memref<1x4xf32>
      %n = arith.addf %v, %acc : f32
      scf.yield %n : f32
    }
    memref.store %sum, %B[%c0, %c0] : memref<1x4xf32>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @fuse_vector_transfer_no_subview
// CHECK:       vector.transfer_write
// CHECK:       scf.for
// CHECK-NOT:   scf.parallel

// -----

func.func @fuse_vector_transfer_scalar_load_rank2(%A: memref<2x4xf32>, %B: memref<2x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %vec = arith.constant dense<1.0> : vector<2x4xf32>
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    vector.transfer_write %vec, %A[%c0, %c0] {permutation_map = affine_map<(d0, d1) -> (d0, d1)>, in_bounds = [true, true]} : vector<2x4xf32>, memref<2x4xf32>
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %v0 = memref.load %A[%c0, %c1] : memref<2x4xf32>
    %v1 = memref.load %A[%c1, %c2] : memref<2x4xf32>
    %sum = arith.addf %v0, %v1 : f32
    memref.store %sum, %B[%c0, %c0] : memref<2x4xf32>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @fuse_vector_transfer_scalar_load_rank2
// CHECK:       scf.parallel
// CHECK:         vector.transfer_write
// CHECK:         memref.load
// CHECK:         memref.load
// CHECK-NOT:   scf.parallel

// -----

func.func @fuse_vector_transfer_scalar_load_loop_rank2(%A: memref<2x4xf32>, %B: memref<2x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %vec = arith.constant dense<2.0> : vector<2x4xf32>
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    vector.transfer_write %vec, %A[%c0, %c0] {permutation_map = affine_map<(d0, d1) -> (d0, d1)>, in_bounds = [true, true]} : vector<2x4xf32>, memref<2x4xf32>
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %sum = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> f32 {
      %v = memref.load %A[%c1, %k] : memref<2x4xf32>
      %n = arith.addf %v, %acc : f32
      scf.yield %n : f32
    }
    memref.store %sum, %B[%c0, %c0] : memref<2x4xf32>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @fuse_vector_transfer_scalar_load_loop_rank2
// CHECK:       scf.parallel
// CHECK:         vector.transfer_write
// CHECK:         scf.for
// CHECK-NOT:   scf.parallel

// -----

func.func @fuse_vector_store_scalar_load_rank2(%A: memref<2x4xf32>, %B: memref<2x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %vec = arith.constant dense<3.0> : vector<2x4xf32>
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    vector.store %vec, %A[%c0, %c0] : memref<2x4xf32>, vector<2x4xf32>
    scf.reduce
  }
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %v0 = memref.load %A[%c1, %c2] : memref<2x4xf32>
    %v1 = memref.load %A[%c0, %c3] : memref<2x4xf32>
    %sum = arith.addf %v0, %v1 : f32
    memref.store %sum, %B[%c0, %c0] : memref<2x4xf32>
    scf.reduce
  }
  return
}
// CHECK-LABEL: func @fuse_vector_store_scalar_load_rank2
// CHECK:       scf.parallel
// CHECK:         vector.store
// CHECK:         memref.load
// CHECK:         memref.load
// CHECK-NOT:   scf.parallel

// -----

func.func @fuse_reductions_two(%A: memref<2x2xf32>, %B: memref<2x2xf32>) -> (f32, f32) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init1 = arith.constant 1.0 : f32
  %init2 = arith.constant 2.0 : f32
  %res1 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init(%init1) -> f32 {
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    scf.reduce(%A_elem : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.addf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  %res2 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init(%init2) -> f32 {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    scf.reduce(%B_elem : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.mulf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  return %res1, %res2 : f32, f32
}

// CHECK-LABEL: func @fuse_reductions_two
//  CHECK-SAME:  (%[[A:.*]]: memref<2x2xf32>, %[[B:.*]]: memref<2x2xf32>) -> (f32, f32)
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[INIT1:.*]] = arith.constant 1.000000e+00 : f32
//   CHECK-DAG:   %[[INIT2:.*]] = arith.constant 2.000000e+00 : f32
//       CHECK:   %[[RES:.*]]:2 = scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
//  CHECK-SAME:   to (%[[C2]], %[[C2]]) step (%[[C1]], %[[C1]])
//  CHECK-SAME:   init (%[[INIT1]], %[[INIT2]]) -> (f32, f32)
//       CHECK:   %[[VAL_A:.*]] = memref.load %[[A]][%[[I]], %[[J]]]
//       CHECK:   %[[VAL_B:.*]] = memref.load %[[B]][%[[I]], %[[J]]]
//       CHECK:   scf.reduce(%[[VAL_A]], %[[VAL_B]] : f32, f32) {
//       CHECK:   ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
//       CHECK:     %[[R:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
//       CHECK:     scf.reduce.return %[[R]] : f32
//       CHECK:   }
//       CHECK:   ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
//       CHECK:     %[[R:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f32
//       CHECK:     scf.reduce.return %[[R]] : f32
//       CHECK:   }
//       CHECK:   return %[[RES]]#0, %[[RES]]#1 : f32, f32

// -----

func.func @fuse_reductions_three(%A: memref<2x2xf32>, %B: memref<2x2xf32>, %C: memref<2x2xf32>) -> (f32, f32, f32) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init1 = arith.constant 1.0 : f32
  %init2 = arith.constant 2.0 : f32
  %init3 = arith.constant 3.0 : f32
  %res1 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init(%init1) -> f32 {
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    scf.reduce(%A_elem : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.addf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  %res2 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init(%init2) -> f32 {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    scf.reduce(%B_elem : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.mulf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  %res3 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init(%init3) -> f32 {
    %A_elem = memref.load %C[%i, %j] : memref<2x2xf32>
    scf.reduce(%A_elem : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.addf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  return %res1, %res2, %res3 : f32, f32, f32
}

// CHECK-LABEL: func @fuse_reductions_three
//  CHECK-SAME:  (%[[A:.*]]: memref<2x2xf32>, %[[B:.*]]: memref<2x2xf32>, %[[C:.*]]: memref<2x2xf32>) -> (f32, f32, f32)
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[INIT1:.*]] = arith.constant 1.000000e+00 : f32
//   CHECK-DAG:   %[[INIT2:.*]] = arith.constant 2.000000e+00 : f32
//   CHECK-DAG:   %[[INIT3:.*]] = arith.constant 3.000000e+00 : f32
//       CHECK:   %[[RES:.*]]:3 = scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
//  CHECK-SAME:   to (%[[C2]], %[[C2]]) step (%[[C1]], %[[C1]])
//  CHECK-SAME:   init (%[[INIT1]], %[[INIT2]], %[[INIT3]]) -> (f32, f32, f32)
//       CHECK:   %[[VAL_A:.*]] = memref.load %[[A]][%[[I]], %[[J]]]
//       CHECK:   %[[VAL_B:.*]] = memref.load %[[B]][%[[I]], %[[J]]]
//       CHECK:   %[[VAL_C:.*]] = memref.load %[[C]][%[[I]], %[[J]]]
//       CHECK:   scf.reduce(%[[VAL_A]], %[[VAL_B]], %[[VAL_C]] : f32, f32, f32) {
//       CHECK:   ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
//       CHECK:     %[[R:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
//       CHECK:     scf.reduce.return %[[R]] : f32
//       CHECK:   }
//       CHECK:   ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
//       CHECK:     %[[R:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f32
//       CHECK:     scf.reduce.return %[[R]] : f32
//       CHECK:   }
//       CHECK:   ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
//       CHECK:     %[[R:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
//       CHECK:     scf.reduce.return %[[R]] : f32
//       CHECK:   }
//       CHECK:   return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : f32, f32, f32

// -----

func.func @reductions_use_res(%A: memref<2x2xf32>, %B: memref<2x2xf32>) -> (f32, f32) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init1 = arith.constant 1.0 : f32
  %res1 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init(%init1) -> f32 {
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    scf.reduce(%A_elem : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.addf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  %res2 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init(%res1) -> f32 {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    scf.reduce(%B_elem : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.mulf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  return %res1, %res2 : f32, f32
}

// %res1 is used as second scf.parallel arg, cannot fuse
// CHECK-LABEL: func @reductions_use_res
// CHECK:      scf.parallel
// CHECK:      scf.parallel

// -----

func.func @reductions_use_res_inside(%A: memref<2x2xf32>, %B: memref<2x2xf32>) -> (f32, f32) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init1 = arith.constant 1.0 : f32
  %init2 = arith.constant 2.0 : f32
  %res1 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init(%init1) -> f32 {
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    scf.reduce(%A_elem : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.addf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  %res2 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init(%init2) -> f32 {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    %sum = arith.addf %B_elem, %res1 : f32
    scf.reduce(%sum : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.mulf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  return %res1, %res2 : f32, f32
}

// %res1 is used inside second scf.parallel, cannot fuse
// CHECK-LABEL: func @reductions_use_res_inside
// CHECK:      scf.parallel
// CHECK:      scf.parallel

// -----

func.func @reductions_use_res_between(%A: memref<2x2xf32>, %B: memref<2x2xf32>) -> (f32, f32, f32) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init1 = arith.constant 1.0 : f32
  %init2 = arith.constant 2.0 : f32
  %res1 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init(%init1) -> f32 {
    %A_elem = memref.load %A[%i, %j] : memref<2x2xf32>
    scf.reduce(%A_elem : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.addf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  %res3 = arith.addf %res1, %init2 : f32
  %res2 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init(%init2) -> f32 {
    %B_elem = memref.load %B[%i, %j] : memref<2x2xf32>
    scf.reduce(%B_elem : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.mulf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  return %res1, %res2, %res3 : f32, f32, f32
}

// instruction in between the loops uses the first loop result
// CHECK-LABEL: func @reductions_use_res_between
// CHECK:      scf.parallel
// CHECK:      scf.parallel
