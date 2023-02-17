// RUN: mlir-opt %s -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs" -canonicalize -split-input-file | FileCheck %s

//      CHECK: func @buffer_forwarding_conflict(
// CHECK-SAME:   %[[FUNC_ARG:[0-9a-zA-Z]*]]: memref<?xf32>
// CHECK-SAME:   %[[sz:[0-9a-zA-Z]*]]: index
func.func @buffer_forwarding_conflict(
  %t: tensor<?xf32> {bufferization.buffer_layout = affine_map<(d0) -> (d0)>, bufferization.writable = true},
  %sz: index)
    -> (tensor<?xf32>, tensor<?xf32>)
{
  %f0 = arith.constant 0.0: f32

  //     CHECK: %[[EXTRACT_SLICE_ALLOC:.*]] = memref.alloc(%[[sz]])
  //     CHECK: linalg.fill ins({{.*}} : f32) outs(%[[EXTRACT_SLICE_ALLOC]] : memref<?xf32>)
  // Alloc is needed for the **first** insert_slice (due to backward traversal during analysis).
  //     CHECK: %[[DIM:.*]] = memref.dim %[[FUNC_ARG]]
  // This allocs the whole dim to allow for a full clone of t.
  //     CHECK: %[[ALLOC:.*]] = memref.alloc(%[[DIM]])
  // tensor.empty itself does not alloc but forwards to the **second**
  // insert_slice. The pass replaces the tensor.empty with an out-of-place
  // extract_slice.
  %a = tensor.empty(%sz) : tensor<?xf32>
  %f = linalg.fill ins(%f0 : f32) outs(%a : tensor<?xf32>) -> tensor<?xf32>

  //     CHECK: memref.copy %[[FUNC_ARG]], %[[ALLOC]] : memref<?xf32> to memref<?xf32>
  //     CHECK: %[[SV0_ALLOC:.*]] = memref.subview %[[ALLOC]][0] [%[[sz]]] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
  //     CHECK: memref.copy %[[EXTRACT_SLICE_ALLOC]], %[[SV0_ALLOC]] : memref<?xf32> to memref<?xf32, strided<[1]>>
  %r0 = tensor.insert_slice %f into %t[0][%sz][1]: tensor<?xf32> into tensor<?xf32>

  //     CHECK: %[[T_SUBVIEW:.*]] =  memref.subview %[[FUNC_ARG]][42] [%[[sz]]] [1]
  //     CHECK: memref.copy %[[EXTRACT_SLICE_ALLOC]], %[[T_SUBVIEW]]
  %r1 = tensor.insert_slice %f into %t[42][%sz][1]: tensor<?xf32> into tensor<?xf32>

  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

//      CHECK: func @buffer_forwarding_no_conflict(
// CHECK-SAME:   %[[FUNC_ARG:[0-9a-zA-Z]*]]: memref<?xf32>
// CHECK-SAME:   %[[sz:[0-9a-zA-Z]*]]: index
func.func @buffer_forwarding_no_conflict(
  %t: tensor<?xf32> {bufferization.buffer_layout = affine_map<(d0) -> (d0)>, bufferization.writable = true},
  %sz: index)
    -> (tensor<?xf32>)
{
  %f0 = arith.constant 0.0: f32

  // tensor.empty itself does not alloc but forwards to the insert_slice.
  // EmptyTensorOpElimination replaces the tensor.empty with an inplace
  // extract_slice.
  // CHECK: %[[T_SUBVIEW:.*]] =  memref.subview %[[FUNC_ARG]][42] [%[[sz]]] [1]
  %a = tensor.empty(%sz) : tensor<?xf32>

  // CHECK: linalg.fill ins({{.*}} : f32) outs(%[[T_SUBVIEW]] : memref<?xf32
  %f = linalg.fill ins(%f0 : f32) outs(%a : tensor<?xf32>) -> tensor<?xf32>

  // Self-copy canonicalizes away later.
  %r1 = tensor.insert_slice %f into %t[42][%sz][1]: tensor<?xf32> into tensor<?xf32>

  return %r1: tensor<?xf32>
}

// -----

//      CHECK: func @insertion_point_inside_loop(
// CHECK-SAME:     %[[t:.*]]: memref<?xf32, strided{{.*}}>, %[[sz:.*]]: index)
func.func @insertion_point_inside_loop(%t : tensor<?xf32>, %sz : index) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index

  // CHECK-NOT: memref.alloc
  %blank = tensor.empty() : tensor<5xf32>

  // CHECK: scf.for %[[iv:.*]] = %{{.*}} to %[[sz]] step %{{.*}} {
  %r = scf.for %iv = %c0 to %sz step %c5 iter_args(%bb = %t) -> (tensor<?xf32>) {
    // CHECK: %[[subview:.*]] = memref.subview %[[t]][%[[iv]]] [5] [1]
    %iv_i32 = arith.index_cast %iv : index to i32
    %f = arith.sitofp %iv_i32 : i32 to f32

    // CHECK: linalg.fill ins(%{{.*}}{{.*}}outs(%[[subview]]
    %filled = linalg.fill ins(%f : f32) outs(%blank : tensor<5xf32>) -> tensor<5xf32>

    // CHECK-NOT: memref.copy
    %inserted = tensor.insert_slice %filled into %bb[%iv][5][1] : tensor<5xf32> into tensor<?xf32>
    scf.yield %inserted : tensor<?xf32>
  }

  return %r : tensor<?xf32>
}

// -----

//      CHECK: func @insertion_point_outside_loop(
// CHECK-SAME:     %[[t:.*]]: memref<?xf32, strided{{.*}}>, %[[sz:.*]]: index, %[[idx:.*]]: index)
func.func @insertion_point_outside_loop(%t : tensor<?xf32>, %sz : index,
                                        %idx : index) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index

  // CHECK-NOT: memref.alloc
  // CHECK: %[[subview:.*]] = memref.subview %[[t]][%[[idx]]] [5] [1]
  %blank = tensor.empty() : tensor<5xf32>

  // CHECK: scf.for %[[iv:.*]] = %{{.*}} to %[[sz]] step %{{.*}} {
  %r = scf.for %iv = %c0 to %sz step %c5 iter_args(%bb = %t) -> (tensor<?xf32>) {
    %iv_i32 = arith.index_cast %iv : index to i32
    %f = arith.sitofp %iv_i32 : i32 to f32

    // CHECK: linalg.fill ins(%{{.*}}{{.*}}outs(%[[subview]]
    %filled = linalg.fill ins(%f : f32) outs(%blank : tensor<5xf32>) -> tensor<5xf32>

    // CHECK-NOT: memref.copy
    %inserted = tensor.insert_slice %filled into %bb[%idx][5][1] : tensor<5xf32> into tensor<?xf32>
    scf.yield %inserted : tensor<?xf32>
  }

  return %r : tensor<?xf32>
}

// -----

// EmptyTensorElimination does currently not apply to chains where the type is
// changing. This test just ensures that we do not crash or generate IR that
// does not verify.

// CHECK-LABEL: func @shape_mismatch
func.func @shape_mismatch(%t: tensor<5x6x128xf32>) -> tensor<5x6x128xf32> {
  %cst = arith.constant 8.0 : f32
  %0 = tensor.empty() : tensor<128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128xf32>) -> tensor<128xf32>
  %2 = tensor.expand_shape %1 [[0, 1, 2]]
      : tensor<128xf32> into tensor<1x1x128xf32>
  %3 = tensor.insert_slice %2 into %t[2, 3, 0][1, 1, 128][1, 1, 1]
      : tensor<1x1x128xf32> into tensor<5x6x128xf32>
  return %3 : tensor<5x6x128xf32>
}

// -----

//      CHECK: func @parallel_insert_slice(
// CHECK-SAME:   %[[FUNC_ARG:[0-9a-zA-Z]*]]: memref<?xf32>
// CHECK-SAME:   %[[sz:[0-9a-zA-Z]*]]: index
func.func @parallel_insert_slice(
  %t: tensor<?xf32> {bufferization.buffer_layout = affine_map<(d0) -> (d0)>, bufferization.writable = true},
  %sz: index)
    -> (tensor<?xf32>)
{
  %f0 = arith.constant 0.0: f32
  %c512 = arith.constant 512 : index

  %r1 = scf.forall (%iv) in (%c512) shared_outs(%o = %t) -> (tensor<?xf32>) {
    // tensor.empty itself does not alloc but forwards to the insert_slice.
    // EmptyTensorOpElimination replaces the tensor.empty with an inplace
    // extract_slice.
    // CHECK: %[[T_SUBVIEW:.*]] =  memref.subview %[[FUNC_ARG]][42] [%[[sz]]] [1]
    %a = tensor.empty(%sz) : tensor<?xf32>

    // CHECK: linalg.fill ins({{.*}} : f32) outs(%[[T_SUBVIEW]] : memref<?xf32
    %f = linalg.fill ins(%f0 : f32) outs(%a : tensor<?xf32>) -> tensor<?xf32>

    // Self-copy canonicalizes away later.
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %f into %o[42][%sz][1]: tensor<?xf32> into tensor<?xf32>
    }
  }

  return %r1: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @eleminate_multiple_ops(
//  CHECK-SAME:   %[[FUNC_ARG:[0-9a-zA-Z]*]]: memref<?xf32>
//  CHECK-SAME:   %[[sz:[0-9a-zA-Z]*]]: index
func.func @eleminate_multiple_ops(%t: tensor<?xf32> {bufferization.buffer_layout = affine_map<(d0) -> (d0)>}, %sz: index, %c: i1)
    -> (tensor<?xf32>)
{
  %cst1 = arith.constant 0.0: f32
  %cst2 = arith.constant 1.0: f32

  // CHECK: %[[r:.*]] = scf.if %{{.*}} -> (memref
  %if = scf.if %c -> tensor<?xf32> {
    // CHECK: %[[T_SUBVIEW_1:.*]] =  memref.subview %[[FUNC_ARG]][42] [%[[sz]]] [1]
    %a1 = tensor.empty(%sz) : tensor<?xf32>
    // CHECK: linalg.fill ins({{.*}} : f32) outs(%[[T_SUBVIEW_1]] : memref<?xf32
    %f1 = linalg.fill ins(%cst1 : f32) outs(%a1 : tensor<?xf32>) -> tensor<?xf32>
    // CHECK: scf.yield %[[T_SUBVIEW_1]]
    scf.yield %f1 : tensor<?xf32>
  } else {
      // CHECK: %[[T_SUBVIEW_2:.*]] =  memref.subview %[[FUNC_ARG]][42] [%[[sz]]] [1]
    %a2 = tensor.empty(%sz) : tensor<?xf32>
    // CHECK: linalg.fill ins({{.*}} : f32) outs(%[[T_SUBVIEW_2]] : memref<?xf32
    %f2 = linalg.fill ins(%cst2 : f32) outs(%a2 : tensor<?xf32>) -> tensor<?xf32>
    // CHECK: scf.yield %[[T_SUBVIEW_2]]
    scf.yield %f2 : tensor<?xf32>
  }

  // Self-copy could canonicalize away later.
  // CHECK: %[[T_SUBVIEW_3:.*]] =  memref.subview %[[FUNC_ARG]][42] [%[[sz]]] [1]
  // CHECK: memref.copy %[[r]], %[[T_SUBVIEW_3]]
  %r1 = tensor.insert_slice %if into %t[42][%sz][1]: tensor<?xf32> into tensor<?xf32>
  return %r1: tensor<?xf32>
}

// -----

// This is a regression test. Make sure that the tensor.extract_slice is not
// eliminated.

// CHECK-LABEL: func.func @regression_do_not_eliminate_non_empty(
//       CHECK:   memref.subview
//       CHECK:   memref.subview
//       CHECK:   memref.copy
func.func @regression_do_not_eliminate_non_empty(
    %t: tensor<10xf32>, %t2: tensor<10xf32>) -> tensor<10xf32> {
  %1 = tensor.extract_slice %t[0] [5] [1] : tensor<10xf32> to tensor<5xf32>
  %2 = tensor.insert_slice %1 into %t2[1] [5] [1]
      : tensor<5xf32> into tensor<10xf32>
  return %2 : tensor<10xf32>
}