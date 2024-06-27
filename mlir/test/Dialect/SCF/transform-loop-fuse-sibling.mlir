// RUN: mlir-opt %s -transform-interpreter --cse --canonicalize -split-input-file -verify-diagnostics | FileCheck %s
// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics | FileCheck %s --check-prefix CHECK-NOCLEANUP

// CHECK: func.func @fuse_1st_for_into_2nd([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @fuse_1st_for_into_2nd(%A: tensor<128xf32>, %B: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C16:%.*]] = arith.constant 16 : index
  // CHECK-DAG: [[C128:%.*]] = arith.constant 128 : index
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: [[R0:%.*]]:2 = scf.for [[IV:%.*]] = [[C0]] to [[C128]] step [[C16]] iter_args([[IA:%.*]] = [[A]], [[IB:%.*]] = [[B]]) {{.*}}
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %A) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[ASLICE:%.*]] = vector.transfer_read [[A]][[[IV]]], [[ZERO]]
  // CHECK-DAG:   [[SLICE0:%.*]] = vector.transfer_read [[IA]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT1:%.*]] = arith.addf [[SLICE0]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT0:%.*]] = vector.transfer_write [[OUT1]], [[IA]][[[IV]]]
    %2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %B) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[SLICE1:%.*]] = vector.transfer_read [[IB]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT2:%.*]] = arith.addf [[SLICE1]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT1:%.*]] = vector.transfer_write [[OUT2]], [[IB]][[[IV]]]
    %dup2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
  // CHECK: scf.yield [[WRT0]], [[WRT1]] : {{.*}}
    scf.yield %dup6 : tensor<128xf32>
  }
  return %1, %dup1 : tensor<128xf32>, tensor<128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#0 into %for#1 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @fuse_two_parallel
// CHECK-SAME:   ([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}) {
func.func @fuse_two_parallel(%A: memref<2x2xf32>, %B: memref<2x2xf32>) {
// CHECK-DAG:  [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG:  [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:  [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:  [[C1FP:%.*]] = arith.constant 1.
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1fp = arith.constant 1.0 : f32
// CHECK:      [[SUM:%.*]] = memref.alloc()
  %sum = memref.alloc()  : memref<2x2xf32>
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
// CHECK:      memref.dealloc [[SUM]]
  memref.dealloc %sum : memref<2x2xf32>
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.parallel"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %parallel:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %parallel#0 into %parallel#1 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @fuse_two_parallel_reverse
// CHECK-SAME:   ([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}) {
func.func @fuse_two_parallel_reverse(%A: memref<2x2xf32>, %B: memref<2x2xf32>) {
// CHECK-DAG:  [[C2:%.*]] = arith.constant 2 : index
// CHECK-DAG:  [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:  [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:  [[C1FP:%.*]] = arith.constant 1.
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1fp = arith.constant 1.0 : f32
// CHECK:      [[SUM:%.*]] = memref.alloc()
  %sum = memref.alloc()  : memref<2x2xf32>
// CHECK:      scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:     to ([[C2]], [[C2]]) step ([[C1]], [[C1]]) {
// CHECK:        [[SUM_ELEM_:%.*]] = memref.load [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK:        [[A_ELEM:%.*]] = memref.load [[A]]{{\[}}[[I]], [[J]]]
// CHECK:        [[PRODUCT_ELEM:%.*]] = arith.mulf [[SUM_ELEM_]], [[A_ELEM]]
// CHECK:        memref.store [[PRODUCT_ELEM]], [[B]]{{\[}}[[I]], [[J]]]
// CHECK-NOT:  scf.parallel
// CHECK:        [[B_ELEM:%.*]] = memref.load [[B]]{{\[}}[[I]], [[J]]]
// CHECK:        [[SUM_ELEM:%.*]] = arith.addf [[B_ELEM]], [[C1FP]]
// CHECK:        memref.store [[SUM_ELEM]], [[SUM]]{{\[}}[[I]], [[J]]]
// CHECK:        scf.reduce
// CHECK:      }
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
// CHECK:      memref.dealloc [[SUM]]
  memref.dealloc %sum : memref<2x2xf32>
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.parallel"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %parallel:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %parallel#1 into %parallel#0 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @fuse_reductions_two
//  CHECK-SAME:  (%[[A:.*]]: memref<2x2xf32>, %[[B:.*]]: memref<2x2xf32>) -> (f32, f32)
func.func @fuse_reductions_two(%A: memref<2x2xf32>, %B: memref<2x2xf32>) -> (f32, f32) {
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
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.parallel"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %parallel:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %parallel#0 into %parallel#1 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: func.func @fuse_2nd_for_into_1st([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @fuse_2nd_for_into_1st(%A: tensor<128xf32>, %B: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C16:%.*]] = arith.constant 16 : index
  // CHECK-DAG: [[C128:%.*]] = arith.constant 128 : index
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: [[R0:%.*]]:2 = scf.for [[IV:%.*]] = [[C0]] to [[C128]] step [[C16]] iter_args([[IB:%.*]] = [[B]], [[IA:%.*]] = [[A]]) {{.*}}
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %A) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[ASLICE:%.*]] = vector.transfer_read [[A]][[[IV]]], [[ZERO]]
  // CHECK-DAG:   [[SLICE0:%.*]] = vector.transfer_read [[IB]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT1:%.*]] = arith.addf [[SLICE0]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT0:%.*]] = vector.transfer_write [[OUT1]], [[IB]][[[IV]]]
    %2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %B) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[SLICE1:%.*]] = vector.transfer_read [[IA]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT2:%.*]] = arith.addf [[SLICE1]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT1:%.*]] = vector.transfer_write [[OUT2]], [[IA]][[[IV]]]
    %dup2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
  // NB: the dominance check used to fail on the following line,
  // however the defining op for the value of %arg3 occurs above the source loop and hence is safe
  // and %arg4 is a block argument of the scope of the loops and hence is safe
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
  // CHECK: scf.yield [[WRT0]], [[WRT1]] : {{.*}}
    scf.yield %dup6 : tensor<128xf32>
  }
  return %1, %dup1 : tensor<128xf32>, tensor<128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#1 into %for#0 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: func.func @matmul_fuse_1st_forall_into_2nd([[A1:%.*]]: {{.*}}, [[A2:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @matmul_fuse_1st_forall_into_2nd(%A1 : tensor<128x128xf32>, %A2 : tensor<128x128xf32>, %B : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK: scf.forall ([[I:%.*]]) in (4) shared_outs([[S1:%.*]] = [[IN1:%.*]], [[S2:%.*]] = [[IN2:%.*]]) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  // CHECK:   [[T:%.*]] = affine.apply
  // CHECK:   tensor.extract_slice [[A2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   tensor.extract_slice [[S1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   [[OUT1:%.*]] = linalg.matmul
  // CHECK:   tensor.extract_slice [[A1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   tensor.extract_slice [[S2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   [[OUT2:%.*]] = linalg.matmul
  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice [[OUT1]] into [[S1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:     tensor.parallel_insert_slice [[OUT2]] into [[S2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   }
  // CHECK: }
  %out1 = linalg.matmul ins(%A1, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out2 = linalg.matmul ins(%A2, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_forall %mm1 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_forall %mm2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_loop = transform.loop.fuse_sibling %loop2 into %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: func.func @matmul_fuse_2nd_forall_into_1st([[A1:%.*]]: {{.*}}, [[A2:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @matmul_fuse_2nd_forall_into_1st(%A1 : tensor<128x128xf32>, %A2 : tensor<128x128xf32>, %B : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK: scf.forall ([[I:%.*]]) in (4) shared_outs([[S1:%.*]] = [[IN1:%.*]], [[S2:%.*]] = [[IN2:%.*]]) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  // CHECK:   [[T:%.*]] = affine.apply
  // CHECK:   tensor.extract_slice [[A1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   tensor.extract_slice [[S1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   [[OUT1:%.*]] = linalg.matmul
  // CHECK:   tensor.extract_slice [[A2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   tensor.extract_slice [[S2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   [[OUT2:%.*]] = linalg.matmul
  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice [[OUT1]] into [[S1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:     tensor.parallel_insert_slice [[OUT2]] into [[S2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   }
  // CHECK: }
  %out1 = linalg.matmul ins(%A1, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out2 = linalg.matmul ins(%A2, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_forall %mm1 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_forall %mm2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_loop = transform.loop.fuse_sibling %loop1 into %loop2 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-NOCLEANUP: func.func @fuse_no_iter_args([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @fuse_no_iter_args(%A: tensor<128xf32>, %B: tensor<128xf32>) {
  // CHECK-NOCLEANUP: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-NOCLEANUP: [[C16:%.*]] = arith.constant 16 : index
  // CHECK-NOCLEANUP: [[C128:%.*]] = arith.constant 128 : index
  // CHECK-NOCLEANUP: [[ZERO:%.*]] = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK-NOCLEANUP: scf.for [[IV:%.*]] = [[C0]] to [[C128]] step [[C16]] {{.*}}
  scf.for %arg0 = %c0 to %c128 step %c16 {
  // CHECK-NOCLEANUP:   [[ASLICE:%.*]] = vector.transfer_read [[A]][[[IV]]], [[ZERO]]
    %2 = vector.transfer_read %A[%arg0], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    scf.yield
  }
  scf.for %arg0 = %c0 to %c128 step %c16 {
  // CHECK-NOCLEANUP:   [[BSLICE:%.*]] = vector.transfer_read [[B]][[[IV]]], [[ZERO]]
    %dup2 = vector.transfer_read %B[%arg0], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    scf.yield
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#0 into %for#1 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

func.func @source_for_uses_result_of_target_for_err(%A: tensor<128xf32>, %B: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  // expected-error @below {{user of results of target should be properly dominated by source}}
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %A) -> (tensor<128xf32>) {
    %2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %1) -> (tensor<128xf32>) {
    %dup2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %dup6 : tensor<128xf32>
  }
  return %1, %dup1 : tensor<128xf32>, tensor<128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#0 into %for#1 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

func.func @source_forall_uses_result_of_target_forall_err(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  // expected-error @below {{user of results of target should be properly dominated by source}}
  %out1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out2 = linalg.matmul ins(%A, %out1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_forall %mm1 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_forall %mm2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_loop = transform.loop.fuse_sibling %loop1 into %loop2 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @target_for_region_uses_result_of_source_for_err(%A: tensor<128xf32>, %B: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %A) -> (tensor<128xf32>) {
    %2 = vector.transfer_read %A[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  // expected-error @below {{values used inside regions of target should be properly dominated by source}}
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %B) -> (tensor<128xf32>) {
    // expected-note @below {{see operation}}
    %dup2 = vector.transfer_read %1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %dup6 : tensor<128xf32>
  }
  return %1, %dup1 : tensor<128xf32>, tensor<128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#1 into %for#0 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

func.func @target_forall_depends_on_value_not_dominated_by_source_forall_err(%A1 : tensor<128x128xf32>, %A2 : tensor<128x128xf32>, %B : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %buf1_alloc = tensor.empty() : tensor<128x128xf32>
  %buf1 = linalg.fill ins(%zero : f32) outs(%buf1_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out1 = linalg.matmul ins(%A1, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%buf1 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out_alloc2 = tensor.empty() : tensor<128x128xf32>
  %buf2 = linalg.fill ins(%zero : f32) outs(%buf1_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>
  // expected-error @below {{operands of target should be properly dominated by source}}
  %out2 = linalg.matmul ins(%A2, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%buf2 : tensor<128x128xf32>) -> tensor<128x128xf32>

  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_forall %mm1 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_forall %mm2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_loop = transform.loop.fuse_sibling %loop2 into %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @non_matching_iteration_spaces_err(%A: memref<2x2xf32>, %B: memref<2x2xf32>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1fp = arith.constant 1.0 : f32
  %sum = memref.alloc()  : memref<2x2xf32>
  // expected-error @below {{target and source iteration spaces must be equal}}
  scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
    %B_elem = memref.load %B[%i, %c0] : memref<2x2xf32>
    %sum_elem = arith.addf %B_elem, %c1fp : f32
    memref.store %sum_elem, %sum[%i, %c0] : memref<2x2xf32>
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
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.parallel"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %parallel:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %parallel#0 into %parallel#1 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

func.func @non_matching_loop_types_err(%A: memref<2xf32>, %B: memref<2xf32>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1fp = arith.constant 1.0 : f32
  %sum = memref.alloc()  : memref<2xf32>
  // expected-error @below {{target and source must be same loop type}}
  scf.for %i = %c0 to %c2 step %c1 {
    %B_elem = memref.load %B[%i] : memref<2xf32>
    %sum_elem = arith.addf %B_elem, %c1fp : f32
    memref.store %sum_elem, %sum[%i] : memref<2xf32>
  }
  scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
    %sum_elem = memref.load %sum[%i] : memref<2xf32>
    %A_elem = memref.load %A[%i] : memref<2xf32>
    %product_elem = arith.mulf %sum_elem, %A_elem : f32
    memref.store %product_elem, %B[%i] : memref<2xf32>
    scf.reduce
  }
  memref.dealloc %sum : memref<2xf32>
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.parallel"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused = transform.loop.fuse_sibling %0 into %1 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: func.func @foreach_loop_pair_fuse([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @foreach_loop_pair_fuse(%arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C16:%.*]] = arith.constant 16 : index
  // CHECK-DAG: [[C128:%.*]] = arith.constant 128 : index
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: [[RST:%.*]]:2 = scf.for [[IV:%.*]] = [[C0]] to [[C128]] step [[C16]] iter_args([[IB0:%.*]] = [[B]], [[IB1:%.*]] = [[B]]) {{.*}}
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[ASLICE:%.*]] = vector.transfer_read [[A]][[[IV]]], [[ZERO]]
  // CHECK-DAG:   [[SLICE0:%.*]] = vector.transfer_read [[IB0]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT1:%.*]] = arith.addf [[SLICE0]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT0:%.*]] = vector.transfer_write [[OUT1]], [[IB0]][[[IV]]]
    %2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  } {target_loops}
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[SLICE1:%.*]] = vector.transfer_read [[IB1]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT2:%.*]] = arith.addf [[SLICE1]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT1:%.*]] = vector.transfer_write [[OUT2]], [[IB1]][[[IV]]]
    %dup2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
  // CHECK: scf.yield [[WRT0]], [[WRT1]] : {{.*}}
    scf.yield %dup6 : tensor<128xf32>
  } {source_loops}
  %2 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %arg2) -> (tensor<128xf32>)  {
  // CHECK-DAG:   [[ASLICE:%.*]] = vector.transfer_read [[A]][[[IV]]], [[ZERO]]
  // CHECK-DAG:   [[SLICE0:%.*]] = vector.transfer_read [[IB0]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT1:%.*]] = arith.addf [[SLICE0]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT0:%.*]] = vector.transfer_write [[OUT1]], [[IB0]][[[IV]]]
    %2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<32xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<32xf32>
    %5 = arith.addf %3, %2 : vector<32xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<32xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  } {target_loops}
  %dup2 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[SLICE1:%.*]] = vector.transfer_read [[IB1]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT2:%.*]] = arith.addf [[SLICE1]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT1:%.*]] = vector.transfer_write [[OUT2]], [[IB1]][[[IV]]]
    %dup2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<32xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<32xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<32xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<32xf32>, tensor<128xf32>
  // CHECK: scf.yield [[WRT0]], [[WRT1]] : {{.*}}
    scf.yield %dup6 : tensor<128xf32>
  } {source_loops}
  return %1, %dup1, %2, %dup2 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %target_loops = transform.structured.match ops{["scf.for"]} attributes {target_loops} in %arg0 : (!transform.any_op) -> !transform.any_op
    %source_loops = transform.structured.match ops{["scf.for"]} attributes {source_loops} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.foreach %target_loops, %source_loops : !transform.any_op, !transform.any_op {
    ^bb0(%target_loop: !transform.any_op, %source_loop: !transform.any_op):
      %fused = transform.loop.fuse_sibling %target_loop into %source_loop : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    }
    transform.yield
  }
}
