// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" -drop-equivalent-buffer-results -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-heuristic=fuzzer analysis-fuzzer-seed=23 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-heuristic=fuzzer analysis-fuzzer-seed=59 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-heuristic=fuzzer analysis-fuzzer-seed=91 bufferize-function-boundaries" -split-input-file -o /dev/null

// Test bufferization using memref types that have no layout map.
// RUN: mlir-opt %s -one-shot-bufferize="unknown-type-conversion=identity-layout-map bufferize-function-boundaries" -split-input-file -o /dev/null

// CHECK-LABEL: func private @insert_slice_fun
//  CHECK-SAME:   %[[A0:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>,
//  CHECK-SAME:   %[[A1:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>,
//  CHECK-SAME:   %[[t0:[a-zA-Z0-9]*]]: memref<4xf32, strided<[?], offset: ?>>,
//  CHECK-SAME:   %[[t1:[a-zA-Z0-9]*]]: memref<4xf32, strided<[?], offset: ?>>
func.func private @insert_slice_fun(
    %A0 : tensor<?xf32> {bufferization.writable = false},
    %A1 : tensor<?xf32> {bufferization.writable = true},
    %t0 : tensor<4xf32> {bufferization.writable = false},
    %t1 : tensor<4xf32> {bufferization.writable = true})
  ->  (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
{
  // Alloc and copy the whole result tensor. Copy the tensor.extract_slice.
  //      CHECK: %[[REALLOC3:.*]] = memref.alloc
  //      CHECK: memref.copy %[[A0]], %[[REALLOC3]]
  //      CHECK: %[[SV_A0:.*]] = memref.subview %[[REALLOC3]]
  //      CHECK: memref.copy %[[t0]], %[[SV_A0]]
  %r0 = tensor.insert_slice %t0 into %A0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // Alloc and copy the whole result tensor. Copy the tensor.extract_slice.
  //      CHECK: %[[REALLOC2:.*]] = memref.alloc
  //      CHECK: memref.copy %[[A0]]
  //      CHECK: %[[SV_A0_2:.*]] = memref.subview %[[REALLOC2]]
  //      CHECK: memref.copy %[[t1]], %[[SV_A0_2]]
  %r1 = tensor.insert_slice %t1 into %A0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //  Still alloc the large tensor because %A1 is read after. Copy the tensor.extract_slice.
  //      CHECK: %[[REALLOC1:.*]] = memref.alloc
  //      CHECK: memref.copy %[[A1]]
  //      CHECK: %[[SV_A1:.*]] = memref.subview %[[REALLOC1]]
  //      CHECK: memref.copy %[[t0]], %[[SV_A1]]
  %r2 = tensor.insert_slice %t0 into %A1[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //  Do not realloc the large tensor. Copy the tensor.extract_slice.
  //  CHECK-NOT: alloc
  //      CHECK: %[[SV_A1_2:.*]] = memref.subview %[[A1]]
  //      CHECK: memref.copy %[[t1]], %[[SV_A1_2]]
  %r3 = tensor.insert_slice %t1 into %A1[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //      CHECK: return %[[REALLOC3]], %[[REALLOC2]], %[[REALLOC1]] :
  // CHECK-SAME:   memref<?xf32>, memref<?xf32>, memref<?xf32>
  return %r0, %r1, %r2, %r3: tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_fun
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>
//  CHECK-SAME:   %[[t:[a-zA-Z0-9]*]]: memref<4xf32, strided<[?], offset: ?>>
func.func @insert_slice_fun(
    %A : tensor<?xf32> {bufferization.writable = true},
    %t : tensor<4xf32> {bufferization.writable = false})
  -> tensor<?xf32>
{
  %f0 = arith.constant 0.0 : f32

  //  CHECK-NOT: alloc
  //      CHECK: %[[SV_A:.*]] = memref.subview %[[A]]
  //      CHECK: memref.copy %[[t]], %[[SV_A]]
  %r0 = tensor.insert_slice %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  /// Overwrite A inplace.
  //      CHECK: linalg.fill ins({{.*}}{{.*}}outs(%[[A]]
  %r1 = linalg.fill ins(%f0 : f32) outs(%r0 : tensor<?xf32>) -> tensor<?xf32>

  //     CHECK: return
  // CHECK-NOT: tensor
  return %r1: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_fun
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>
//  CHECK-SAME:   %[[t:[a-zA-Z0-9]*]]: memref<4xf32, strided<[?], offset: ?>>
func.func @insert_slice_fun(
    %A : tensor<?xf32> {bufferization.writable = true},
    %t : tensor<4xf32> {bufferization.writable = false})
  -> tensor<?xf32>
{
  %f0 = arith.constant 0.0 : f32

  //      CHECK: linalg.fill ins({{.*}}{{.*}}outs(%[[A]]
  %r0 = linalg.fill ins(%f0 : f32) outs(%A : tensor<?xf32>) -> tensor<?xf32>

  //  CHECK-NOT: alloc
  //      CHECK: %[[SV_A:.*]] = memref.subview %[[A]]
  /// Overwrite A inplace by copying into the subview.
  //      CHECK: memref.copy %[[t]], %[[SV_A]]
  %r1 = tensor.insert_slice %t into %r0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //     CHECK: return
  // CHECK-NOT: tensor
  return %r1: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_fun_not_inplace
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>
//  CHECK-SAME:   %[[t:[a-zA-Z0-9]*]]: memref<4xf32, strided<[?], offset: ?>>
func.func @insert_slice_fun_not_inplace(
    %A : tensor<?xf32> {bufferization.writable = false},
    %t : tensor<4xf32> {bufferization.writable = false})
  -> tensor<?xf32>
{
  //      CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}) {alignment = 64 : i64} : memref<?xf32>
  //      CHECK: memref.copy %[[A]], %[[ALLOC]] : memref<?xf32{{.*}} to memref<?xf32>
  //      CHECK: %[[SV:.*]] = memref.subview %[[ALLOC]][0] [4] [1] : memref<?xf32> to memref<4xf32, strided<[1]>>
  //      CHECK: memref.copy %[[t]], %[[SV]] : memref<4xf32, strided{{.*}}> to memref<4xf32, strided<[1]>>
  %r0 = tensor.insert_slice %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //     CHECK: return %{{.*}} : memref<?xf32>
  return %r0: tensor<?xf32>
}

// -----

// This test case could bufferize in-place with a better analysis. However, it
// is simpler to let the canonicalizer fold away the tensor.insert_slice.

// CHECK-LABEL: func @tensor_cast_not_in_place(
//  CHECK-SAME:     %[[A:.*]]: memref<?xf32{{.*}}>, %[[B:.*]]: memref<?xf32{{.*}}>
//       CHECK:   %[[casted:.*]] = memref.cast %[[A]] : memref<?xf32, strided<[?], offset: ?>> to memref<4xf32, strided<[?], offset: ?>>
//       CHECK:   %[[alloc:.*]] = memref.alloc
//       CHECK:   memref.copy %[[casted]], %[[alloc]]
//       CHECK:   %[[subview:.*]] = memref.subview %[[A]][{{.*}}] [4] [1] : {{.*}} to memref<4xf32
//       CHECK:   memref.copy %[[alloc]], %[[subview]]
func.func @tensor_cast_not_in_place(
    %A : tensor<?xf32> {bufferization.writable = true},
    %B : tensor<?xf32> {bufferization.writable = false}, %idx: index)
  -> (tensor<?xf32>)
{
  %r0 = tensor.cast %A : tensor<?xf32> to tensor<4xf32>
  %r1 = tensor.insert_slice %r0 into %A[%idx][4][1] : tensor<4xf32> into tensor<?xf32>
  return %r1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @insert_op
//  CHECK-SAME:     %[[t1:.*]]: memref<?xf32, {{.*}}>, %[[s:.*]]: f32, %[[i:.*]]: index
func.func @insert_op(%t1 : tensor<?xf32> {bufferization.writable = true},
                     %s : f32, %i : index) -> tensor<?xf32> {
  // CHECK: memref.store %[[s]], %[[t1]][%[[i]]]
  %0 = tensor.insert %s into %t1[%i] : tensor<?xf32>
  // CHECK: return
  return %0 : tensor<?xf32>
}

// -----

// A regression test to make sure that we handle rank-reducing extract_slice
// correctly.

// CHECK-LABEL: func @rank_reducing
func.func @rank_reducing(
    %i: index, %j: index,
    %arg0: tensor<8x18x32xf32>)
      -> tensor<?x1x6x8xf32> {
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = bufferization.alloc_tensor() : tensor<4x1x6x8xf32>
  %1 = tensor.cast %0 : tensor<4x1x6x8xf32> to tensor<?x1x6x8xf32>
  %2 = bufferization.alloc_tensor() : tensor<1x6x8xf32>
  %5 = scf.for %arg7 = %c0 to %c32 step %c8 iter_args(%arg8 = %1) -> (tensor<?x1x6x8xf32>) {
    %7 = affine.apply affine_map<(d0) -> (d0 ceildiv 8)>(%arg7)
    %8 = tensor.extract_slice %arg0[%i, %j, %arg7] [1, 6, 8] [1, 1, 1] : tensor<8x18x32xf32> to tensor<1x6x8xf32>
    %9 = scf.for %arg9 = %c0 to %c6 step %c1 iter_args(%arg10 = %2) -> (tensor<1x6x8xf32>) {
      %11 = tensor.extract_slice %8[0, %arg9, 0] [1, 1, 8] [1, 1, 1] : tensor<1x6x8xf32> to tensor<1x1x8xf32>
      %12 = tensor.insert_slice %11 into %arg10[0, %arg9, 0] [1, 1, 8] [1, 1, 1] : tensor<1x1x8xf32> into tensor<1x6x8xf32>
      scf.yield %12 : tensor<1x6x8xf32>
    }
    %10 = tensor.insert_slice %9 into %arg8[%7, 0, 0, 0] [1, 1, 6, 8] [1, 1, 1, 1] : tensor<1x6x8xf32> into tensor<?x1x6x8xf32>
    scf.yield %10 : tensor<?x1x6x8xf32>
  }
  return %5: tensor<?x1x6x8xf32>
}

// -----

// CHECK-LABEL: func.func @rank_reducing_parallel_insert_slice
func.func @rank_reducing_parallel_insert_slice(%in: tensor<100xf32>, %out: tensor<200x100xf32>) {
  %c1 = arith.constant 1 : index
  %num_threads = arith.constant 100 : index

  // CHECK: scf.forall {{.*}} {
  %result = scf.forall (%thread_idx) in (%num_threads) shared_outs (%o = %out) -> tensor<200x100xf32> {
      %1 = tensor.extract_slice %in[%thread_idx][1][1] : tensor<100xf32> to tensor<1xf32>
      scf.forall.in_parallel {
        // CHECK: memref.subview %{{.*}}[%{{.*}}] [1] [1] : memref<100xf32, strided<[?], offset: ?>> to memref<1xf32, strided<[?], offset: ?>>
        // CHECK: memref.subview %{{.*}}[1, %{{.*}}] [1, 1] [1, 1] : memref<200x100xf32, strided<[?, ?], offset: ?>> to memref<1xf32, strided<[?], offset: ?>>
        tensor.parallel_insert_slice %1 into %o[1, %thread_idx][1, 1][1, 1] :
          tensor<1xf32> into tensor<200x100xf32>
      }
  }
  // CHECK: }
  return
}

// -----

// CHECK-LABEL: func.func @parallel_insert_full_slice_in_place
// CHECK-NOT:     memref.alloc()
func.func @parallel_insert_full_slice_in_place(%2: tensor<2xf32>) -> tensor<2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %3 = scf.forall (%arg0) in (1) shared_outs(%arg2 = %2) -> (tensor<2xf32>) {
    %fill = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<2xf32>) -> tensor<2xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %fill into %arg2[0] [2] [1] : tensor<2xf32> into tensor<2xf32>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return %3 : tensor<2xf32>
}

// -----

// This test case could bufferize in-place with a better analysis. However, it
// is simpler to let the canonicalizer fold away the tensor.insert_slice.

// CHECK-LABEL: func @insert_equivalent_tensor
func.func @insert_equivalent_tensor(%t: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: memref.alloc
  %cst = arith.constant 4.200000e+01 : f32
  // CHECK: linalg.fill
  %0 = linalg.fill ins(%cst : f32) outs(%t : tensor<10xf32>) -> tensor<10xf32>
  // CHECK: memref.copy
  %1 = tensor.insert_slice %0 into %t[0][10][1] : tensor<10xf32> into tensor<10xf32>
  return %1 : tensor<10xf32>
}

// -----

// CHECK-LABEL: func @pad_memory_space(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32, strided<[?], offset: ?>>
func.func @pad_memory_space(%t: tensor<?xf32>, %h1: index, %f: f32, %pos: index) -> f32
{
  // CHECK: %[[alloc_tensor:.*]] = memref.alloc{{.*}} : memref<?xf32, 3>
  // CHECK: memref.copy %[[t]], %[[alloc_tensor]]
  %0 = bufferization.alloc_tensor() copy(%t)
      {memory_space = 3 : i64} : tensor<?xf32>
  // CHECK: %[[padded_alloc:.*]] = memref.alloc() {{.*}} : memref<15xf32, 3>
  // CHECK: linalg.map
  // CHECK:     outs(%[[padded_alloc]] : memref<15xf32, 3>)
  // CHECK:   linalg.yield %{{.*}}
  // CHECK: }
  // CHECK: %[[subview:.*]] = memref.subview {{.*}} : memref<15xf32, 3> to memref<?xf32, strided<[1], offset: 2>, 3>
  // CHECK: memref.copy %[[alloc_tensor]], %[[subview]]
  %1 = tensor.pad %0 low[2] high[%h1] {
  ^bb0(%arg0: index):
    tensor.yield %f : f32
  } : tensor<?xf32> to tensor<15xf32>
  // CHECK: memref.load {{.*}} : memref<15xf32, 3>
  %2 = tensor.extract %1[%pos] : tensor<15xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: func @insert_slice_regression(
//  CHECK-SAME:   %[[t:.*]]: memref<10xf32,{{.*}}>, %[[b:.*]]: memref<5xf32
func.func @insert_slice_regression(%t: tensor<10xf32>, %b: tensor<5xf32>) -> tensor<10xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: %[[alloc:.*]] = memref.alloc() {{.*}} : memref<10xf32>
  // CHECK: linalg.fill {{.*}} outs(%[[alloc]] : memref<10xf32>)
  %1 = linalg.fill ins(%cst : f32) outs(%t : tensor<10xf32>) -> tensor<10xf32>

  // Read %1 so that it does not DCE away.
  %vec = vector.transfer_read %1[%c0], %cst : tensor<10xf32>, vector<10xf32>
  vector.print %vec : vector<10xf32>

  // Write back a different value (not %1).
  // CHECK: %[[subview:.*]] = memref.subview %[[t]][0] [5] [1]
  // CHECK: memref.copy %[[b]], %[[subview]]
  %2 = tensor.insert_slice %b into %t[0][5][1] : tensor<5xf32> into tensor<10xf32>
  return %2 : tensor<10xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_full_overwrite(
//  CHECK-SAME:   %[[t:.*]]: memref<10xf32,{{.*}}>, %[[b:.*]]: memref<10xf32,{{.*}}>
func.func @insert_slice_full_overwrite(%t: tensor<10xf32>, %b: tensor<10xf32>) -> tensor<10xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: linalg.fill {{.*}} outs(%[[t]] : memref<10xf32,{{.*}}>)
  %1 = linalg.fill ins(%cst : f32) outs(%t : tensor<10xf32>) -> tensor<10xf32>

  // Read %1 so that it does not DCE away.
  %vec = vector.transfer_read %1[%c0], %cst : tensor<10xf32>, vector<10xf32>
  vector.print %vec : vector<10xf32>

  // Write back a different value (not %1).
  // CHECK: %[[subview:.*]] = memref.subview %[[t]][0] [10] [1]
  // CHECK: memref.copy %[[b]], %[[subview]]
  %2 = tensor.insert_slice %b into %t[0][10][1] : tensor<10xf32> into tensor<10xf32>
  return %2 : tensor<10xf32>
}

// -----

// CHECK-LABEL: func @dim_not_reading(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32
func.func @dim_not_reading(%t: tensor<?xf32>, %f: f32, %pos: index)
    -> (tensor<?xf32>, index)
{
  %c0 = arith.constant 0 : index
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy
  //     CHECK: memref.store %{{.*}}, %[[t]]
  %0 = tensor.insert %f into %t[%pos] : tensor<?xf32>
  //     CHECK: memref.dim %[[t]]
  %1 = tensor.dim %t, %c0 : tensor<?xf32>
  return %0, %1 : tensor<?xf32>, index
}

// -----

//       CHECK: #[[$map:.*]] = affine_map<(d0) -> (d0 + 5)>
// CHECK-LABEL: func.func private @cast_retains_buffer_layout(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32, #[[$map]]>, %[[sz:.*]]: index) -> memref<?xf32, strided<[1], offset: 7>> {
//       CHECK:   %[[casted:.*]] = memref.cast %[[t]] : memref<?xf32, #[[$map]]> to memref<10xf32, #[[$map]]>
//       CHECK:   %[[slice:.*]] = memref.subview %[[casted]][2] [%[[sz]]] [1] : memref<10xf32, #[[$map]]> to memref<?xf32, strided<[1], offset: 7>>
//       CHECK:   return %[[slice]]
func.func private @cast_retains_buffer_layout(
    %t: tensor<?xf32>
        {bufferization.buffer_layout = affine_map<(d0) -> (d0 + 5)>},
    %sz: index)
  -> (tensor<10xf32>, tensor<?xf32>)
{
  %casted = tensor.cast %t : tensor<?xf32> to tensor<10xf32>
  %slice = tensor.extract_slice %casted[2][%sz][1] : tensor<10xf32> to tensor<?xf32>

  // Note: The %casted return type is folded away because both buffers are
  // equivalent. Therefore, we currently loose some static type information
  // in the caller.
  return %casted, %slice : tensor<10xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func private @cast_retains_buffer_layout_strided(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32, strided<[1], offset: 5>>, %[[sz:.*]]: index) -> memref<?xf32, strided<[1], offset: 7>> {
//       CHECK:   %[[casted:.*]] = memref.cast %[[t]] : memref<?xf32, strided<[1], offset: 5>> to memref<10xf32, strided<[1], offset: 5>>
//       CHECK:   %[[slice:.*]] = memref.subview %[[casted]][2] [%[[sz]]] [1] : memref<10xf32, strided<[1], offset: 5>> to memref<?xf32, strided<[1], offset: 7>>
//       CHECK:   return %[[slice]]
func.func private @cast_retains_buffer_layout_strided(
    %t: tensor<?xf32>
        {bufferization.buffer_layout = strided<[1], offset: 5>},
    %sz: index)
  -> (tensor<10xf32>, tensor<?xf32>)
{
  %casted = tensor.cast %t : tensor<?xf32> to tensor<10xf32>
  %slice = tensor.extract_slice %casted[2][%sz][1] : tensor<10xf32> to tensor<?xf32>

  // Note: The %casted return type is folded away because both buffers are
  // equivalent. Therefore, we currently loose some static type information
  // in the caller.
  return %casted, %slice : tensor<10xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func.func @parallel_insert_slice_source_out_of_place
func.func @parallel_insert_slice_source_out_of_place(%in: tensor<1xf32>, %out: tensor<100xf32>, %f: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %num_threads = arith.constant 50 : index

  // CHECK: scf.forall {{.*}} {
  %result = scf.forall (%thread_idx) in (%num_threads) shared_outs (%o = %out) -> tensor<100xf32> {
      // The tensor.insert must bufferize out-of-place.
      // CHECK: memref.alloc
      // CHECK: memref.store
      %insert = tensor.insert %f into %in[%c0] : tensor<1xf32>
      %r = tensor.extract %in[%c0] : tensor<1xf32>
      vector.print %r : f32

      // CHECK: memref.copy
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %insert into %o[%thread_idx][1][1] :
          tensor<1xf32> into tensor<100xf32>
      }
  }
  // CHECK: }
  return
}

// -----

// CHECK-LABEL: func @tensor.reshape(
func.func @tensor.reshape() -> tensor<2x2x5xf32> {
  // CHECK-DAG: %[[M1:.*]] = memref.cast %{{.*}} : memref<2x10xf32> to memref<?x10xf32>
  %t1_static = arith.constant dense<0.> : tensor<2x10xf32>
  %t1 = tensor.cast %t1_static : tensor<2x10xf32> to tensor<?x10xf32>

  // CHECK: %[[SHAPE:.*]] = memref.get_global @{{.*}} : memref<3xi64>
  %shape = arith.constant dense<[2, 2, 5]> : tensor<3xi64>

  // CHECK: %[[RESHAPED:.*]] = memref.reshape %[[M1]](%[[SHAPE]]) : (memref<?x10xf32>, memref<3xi64>) -> memref<2x2x5xf32>
  %reshaped = tensor.reshape %t1(%shape) : (tensor<?x10xf32>, tensor<3xi64>) -> tensor<2x2x5xf32>

  // CHECK: return %[[RESHAPED]]
  return %reshaped : tensor<2x2x5xf32>
}

// -----

// CHECK-LABEL: func @tensor_reshape_aliasing
//  CHECK-SAME:  (%[[ARG0:.+]]: index, %[[ARG1:.+]]: index)
func.func @tensor_reshape_aliasing(%arg0: index, %arg1: index) -> tensor<?x?xf32> {
  %t1_static = arith.constant dense<0.> : tensor<10xf32>
  // CHECK-DAG: %[[T1:.+]] = memref.cast
  %t1 = tensor.cast %t1_static : tensor<10xf32> to tensor<?xf32>

  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index

  // CHECK-DAG: %[[SHAPE:.+]] = memref.alloc() {{.*}} : memref<2xindex>
  %shape = bufferization.alloc_tensor() : tensor<2xindex>
  // CHECK: memref.store %[[ARG0]], %[[SHAPE]][%[[C0]]]
  %shape.0 = tensor.insert %arg0 into %shape[%c0] : tensor<2xindex>
  // CHECK: memref.store %[[ARG1]], %[[SHAPE]][%[[C1]]]
  %shape.1 = tensor.insert %arg1 into %shape.0[%c1] : tensor<2xindex>

  // CHECK: %[[RESHAPED:.+]] = memref.reshape %[[T1]](%[[SHAPE]])
  %reshaped = tensor.reshape %t1(%shape.1) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK: return %[[RESHAPED]]
  return %reshaped : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @reshape_with_non_identity_layout(
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]*]]: memref<2x2xf32, strided<[?, ?], offset: ?>, 3>,
// CHECK-SAME:    %[[LAYOUT:[a-zA-Z0-9]*]]: memref<2xi32, strided<[?], offset: ?>>,
func.func @reshape_with_non_identity_layout(%arg0: memref<2x2xf32, strided<[?, ?], offset: ?>, 3>, %arg1: tensor<2xi32>, %idx: index) -> f32 {
  %t = bufferization.to_tensor %arg0 restrict : memref<2x2xf32, strided<[?, ?], offset: ?>, 3> to tensor<2x2xf32>

  // CHECK: %[[SUBVIEW:.+]] = memref.subview %[[INPUT]][1, 0] [1, 2] [1, 1] : memref<2x2xf32, strided<[?, ?], offset: ?>, 3> to memref<2xf32, strided<[?], offset: ?>, 3>
  %extracted_slice = tensor.extract_slice %t[1, 0] [1, 2] [1, 1] : tensor<2x2xf32> to tensor<2xf32>

  // To satisify the constraints of memref.reshape, the subview must be
  // reallocated a buffer with an identity layout.
  // CHECK: %[[ALLOC:.+]] = memref.alloc() {{.*}} : memref<2xf32, 3>
  // CHECK: memref.copy %[[SUBVIEW]], %[[ALLOC]]
  // CHECK: %[[RESHAPED:.+]] = memref.reshape %[[ALLOC]](%[[LAYOUT]]) : (memref<2xf32, 3>, memref<2xi32, strided<[?], offset: ?>>) -> memref<1x2xf32, 3>
  %reshape = tensor.reshape %extracted_slice(%arg1) : (tensor<2xf32>, tensor<2xi32>) -> tensor<1x2xf32>

  %r = tensor.extract %reshape[%idx, %idx] : tensor<1x2xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: func @collapse_shape_regression(
//  CHECK-SAME:     %[[t:.*]]: memref<10x20xf32,
func.func @collapse_shape_regression(
    %t: tensor<10x20xf32>, %f: f32, %idx: index) {
  // CHECK: %[[subview:.*]] = memref.subview %[[t]]
  %0 = tensor.extract_slice %t [2, 3] [5, 6] [1, 1]
      : tensor<10x20xf32> to tensor<5x6xf32>

  // Insert a copy because the original %0 is read later.
  // CHECK: %[[alloc1:.*]] = memref.alloc() {{.*}} : memref<5x6xf32>
  // CHECK: memref.copy %[[subview]], %[[alloc1]]
  // CHECK: memref.store {{.*}}, %[[alloc1]]
  tensor.insert %f into %0[%idx, %idx] : tensor<5x6xf32>

  // Insert a copy because the shape cannot be collapsed.
  // CHECK: %[[alloc2:.*]] = memref.alloc() {{.*}} : memref<5x6xf32>
  // CHECK: memref.copy %[[subview]], %[[alloc2]]
  // CHECK: memref.collapse_shape %[[alloc2]]
  tensor.collapse_shape %0[[0, 1]] : tensor<5x6xf32> into tensor<30xf32>
  return
}

// -----

// CHECK-LABEL: func private @mult_return_callee(
//  CHECK-SAME:   %[[T:.*]]: memref<?xf32, strided<[?], offset: ?>>, %[[COND:.*]]: i1,
//  CHECK-SAME:   %[[A:.*]]: index, %[[B:.*]]: index) -> index {
//       CHECK:   cf.cond_br %[[COND]], ^bb1, ^bb2
//       CHECK: ^bb1:
//       CHECK:   return %[[A]] : index
//       CHECK: ^bb2:
//       CHECK:   return %[[B]] : index
func.func private @mult_return_callee(%t: tensor<?xf32>,  %cond:i1, %a: index, %b: index) -> (tensor<10xf32>, index) {
  %casted = tensor.cast %t : tensor<?xf32> to tensor<10xf32>
  cf.cond_br %cond,^a, ^b
^a:
  return %casted, %a : tensor<10xf32>, index
^b:
  return %casted, %b : tensor<10xf32>, index
}

// CHECK-LABEL: func @mult_return(
//  CHECK-SAME:   %[[T:.*]]: memref<?xf32, strided<[?], offset: ?>>, %[[COND:.*]]: i1,
//  CHECK-SAME:   %[[A:.*]]: index, %[[B:.*]]: index) -> (memref<?xf32, strided<[?], offset: ?>>, index) {
func.func @mult_return(%t: tensor<?xf32>,  %cond:i1, %a: index, %b: index) -> (tensor<10xf32>, index) {
  // CHECK: %[[RET:.*]] = call @mult_return_callee(%[[T]], %[[COND]], %[[A]], %[[B]]) : (memref<?xf32, strided<[?], offset: ?>>, i1, index, index) -> index
  // CHECK: return %[[T]], %[[RET]] : memref<?xf32, strided<[?], offset: ?>>, index
  %t_res, %v = func.call @mult_return_callee(%t, %cond, %a, %b) : (tensor<?xf32>, i1, index, index) -> (tensor<10xf32>, index) 
  return %t_res, %v : tensor<10xf32>, index
}
