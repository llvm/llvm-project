// RUN: mlir-opt %s -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries" -cse -canonicalize -split-input-file | FileCheck %s
// RUN: mlir-opt %s -eliminate-empty-tensors | FileCheck %s --check-prefix=CHECK-ELIM

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
  %blank = tensor.empty() : tensor<5xf32>

  // CHECK: scf.for %[[iv:.*]] = %{{.*}} to %[[sz]] step %{{.*}} {
  %r = scf.for %iv = %c0 to %sz step %c5 iter_args(%bb = %t) -> (tensor<?xf32>) {
    %iv_i32 = arith.index_cast %iv : index to i32
    %f = arith.sitofp %iv_i32 : i32 to f32

    // CHECK: %[[subview:.*]] = memref.subview %[[t]][%[[idx]]] [5] [1]
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
// changing. (Casts are supported.) This test just ensures that we do not crash
// or generate IR that does not verify.

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

// CHECK-LABEL: func @cast(
//  CHECK-SAME:     %[[t:.*]]: memref<256xf32,
//       CHECK:   %[[sv:.*]] = memref.subview %[[t]]
//       CHECK:   linalg.fill {{.*}} outs(%[[sv]]
//       CHECK:   return %[[t]]
func.func @cast(%t: tensor<256xf32>) -> tensor<256xf32> {
  %cst = arith.constant 8.0 : f32
  %c128 = arith.constant 128 : index
  %0 = tensor.empty(%c128) : tensor<?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  %2 = tensor.cast %1 : tensor<?xf32> to tensor<128xf32>
  %3 = tensor.insert_slice %2 into %t[2][128][1]
      : tensor<128xf32> into tensor<256xf32>
  return %3 : tensor<256xf32>
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

// -----

// This is a regression test. Make sure that there is no crash.

// CHECK-LABEL: func.func @regression_insert_of_bbarg(
func.func @regression_insert_of_bbarg(%t0: tensor<5xf32>, %t1: tensor<10xf32>) -> tensor<10xf32> {
  %0 = tensor.insert_slice %t0 into %t1 [2] [5] [1] : tensor<5xf32> into tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

// This is a regression test. Make sure that there is no crash.

// CHECK-LABEL: func.func @regression_eliminate_equivalent_only(
func.func @regression_eliminate_equivalent_only(%sz: index, %p: index, %t0: tensor<?x16xi8>) -> tensor<?x16xi8> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %27 = tensor.empty(%sz) : tensor<?x8xi32>
  %extracted_slice = tensor.extract_slice %27[0, 0] [%p, 8] [1, 1] : tensor<?x8xi32> to tensor<?x8xi32>
  %28 = scf.for %arg4 = %c0 to %c16 step %c8 iter_args(%arg5 = %t0) -> (tensor<?x16xi8>) {
    %inserted_slice = tensor.insert_slice %extracted_slice into %27[0, 0] [%sz, 8] [1, 1] : tensor<?x8xi32> into tensor<?x8xi32>
    %extracted_slice_2 = tensor.extract_slice %arg5[%p, %p] [%sz, 8] [1, 1] : tensor<?x16xi8> to tensor<?x8xi8>
    %32 = linalg.generic
        {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%inserted_slice : tensor<?x8xi32>) outs(%extracted_slice_2 : tensor<?x8xi8>) {
    ^bb0(%in: i32, %out: i8):
      %tr = arith.trunci %in : i32 to i8
      linalg.yield %tr : i8
    } -> tensor<?x8xi8>
    %inserted_slice_3 = tensor.insert_slice %32 into %arg5[%p, %arg4] [%sz, 8] [1, 1] : tensor<?x8xi8> into tensor<?x16xi8>
    scf.yield %inserted_slice_3 : tensor<?x16xi8>
  }
  func.return %28 : tensor<?x16xi8>
}

// -----

// CHECK-LABEL: func.func @regression_multiple_insertion_points(
//   CHECK-NOT:   memref.alloc
func.func @regression_multiple_insertion_points(%t1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %empty = tensor.empty() : tensor<2x5xf32>
  %f0 = arith.constant 5.5 : f32
  %0 = "test.foo"() : () -> (index)
  %1 = "test.bar"() : () -> (index)
  %filled = linalg.fill ins(%f0 : f32) outs(%empty : tensor<2x5xf32>) -> tensor<2x5xf32>
  %2 = tensor.insert_slice %filled into %t1 [%0, %1] [2, 5] [1, 1] : tensor<2x5xf32> into tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @materialize_in_destination(
//  CHECK-SAME:     %[[m:.*]]: memref<5xf32, strided<[?], offset: ?>>,
//       CHECK:   linalg.fill {{.*}} outs(%[[m]]
//       CHECK:   return %[[m]]
func.func @materialize_in_destination(%t: tensor<5xf32>, %f: f32) -> tensor<5xf32> {
  %0 = tensor.empty() : tensor<5xf32>
  %filled = linalg.fill ins(%f : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
  %1 = bufferization.materialize_in_destination %filled in %t : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  return %1 : tensor<5xf32>
}

// -----

// CHECK-LABEL: func @materialize_in_destination_buffer(
//  CHECK-SAME:     %[[m:.*]]: memref<5xf32>,
//  CHECK-NEXT:   linalg.fill {{.*}} outs(%[[m]]
//  CHECK-NEXT:   return
func.func @materialize_in_destination_buffer(%m: memref<5xf32>, %f: f32) {
  %0 = tensor.empty() : tensor<5xf32>
  %filled = linalg.fill ins(%f : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
  bufferization.materialize_in_destination %filled in restrict writable %m : (tensor<5xf32>, memref<5xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @linalg_copy(
//  CHECK-SAME:     %[[m:.*]]: memref<5xf32, strided<[?], offset: ?>>,
//       CHECK:   linalg.fill {{.*}} outs(%[[m]]
//       CHECK:   return %[[m]]
func.func @linalg_copy(%t: tensor<5xf32>, %f: f32) -> tensor<5xf32> {
  %0 = tensor.empty() : tensor<5xf32>
  %filled = linalg.fill ins(%f : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
  %1 = linalg.copy ins(%filled : tensor<5xf32>) outs(%t : tensor<5xf32>) -> tensor<5xf32>
  return %1 : tensor<5xf32>
}

// -----

// CHECK-LABEL: func @linalg_copy_empty(
// CHECK: %[[ret:.*]] = memref.alloc()
// CHECK-NEXT: return %[[ret]]
func.func @linalg_copy_empty() -> tensor<26xi32> {
  %0 = tensor.empty() : tensor<26xi32>
  %1 = linalg.copy ins(%0 : tensor<26xi32>) outs(%0 : tensor<26xi32>) -> tensor<26xi32>
  return %1 : tensor<26xi32>
}

// -----

// CHECK-ELIM-LABEL: func @multiple_materialize_in_destination_buffer(
//  CHECK-ELIM-SAME:     %[[m:.*]]: memref<5xf32>
//       CHECK-ELIM:   tensor.empty
//       CHECK-ELIM:   bufferization.to_tensor %[[m]] restrict writable
//       CHECK-ELIM:   bufferization.materialize_in_destination {{.*}} in writable %[[m]]
func.func @multiple_materialize_in_destination_buffer(%m: memref<5xf32>, %f: f32, %f2: f32, %c: i1) {
  %0 = tensor.empty() : tensor<5xf32>
  %filled = linalg.fill ins(%f : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>

  %1 = tensor.empty() : tensor<5xf32>
  %filled2 = linalg.fill ins(%f2 : f32) outs(%1 : tensor<5xf32>) -> tensor<5xf32>

  %selected = scf.if %c -> tensor<5xf32> {
    scf.yield %filled : tensor<5xf32>
  } else {
    scf.yield %filled2 : tensor<5xf32>
  }
  bufferization.materialize_in_destination %selected in restrict writable %m : (tensor<5xf32>, memref<5xf32>) -> ()
  return
}
