// RUN: mlir-opt %s -linalg-comprehensive-func-bufferize -split-input-file | FileCheck %s
// RUN: mlir-opt %s -linalg-comprehensive-func-bufferize=test-analysis-only -split-input-file | FileCheck %s --check-prefix=ANALYSIS

// CHECK-DAG: #[[$map_2d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @fill_inplace(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: tensor<?xf32> {linalg.inplaceable = true})
func @fill_inplace(%A : tensor<?xf32> {linalg.inplaceable = true}) -> tensor<?xf32> {
  //     CHECK: %[[I:.*]] = memref.buffer_cast %[[A]] : memref<?xf32, #[[$map_2d_dyn]]>

  //     CHECK: %[[F0:.*]] = constant 0.000000e+00 : f32
  %f0 = constant 0.0 : f32

  /// Inplaceable, no alloc
  // CHECK-NOT: alloc
  //     CHECK: linalg.fill(%[[I]], %[[F0]]) : memref<?xf32, #[[$map_2d_dyn]]>, f32
  %r = linalg.fill(%A, %f0) : tensor<?xf32>, f32 -> tensor<?xf32>

  //     CHECK:  %[[R:.*]] = memref.tensor_load %[[I]] : memref<?xf32, #[[$map_2d_dyn]]>
  //     CHECK:  return %[[R]] : tensor<?xf32>
  return %r: tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_2d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

/// No linalg.inplaceable flag, must allocate.
// CHECK-LABEL: func @not_inplace(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: tensor<?xf32>)
func @not_inplace(%A : tensor<?xf32>) -> tensor<?xf32> {
  //     CHECK: %[[I:.*]] = memref.buffer_cast %[[A]] : memref<?xf32, #[[$map_2d_dyn]]>

  //     CHECK: %[[D0:.*]] = memref.dim %[[I]], {{.*}} : memref<?xf32, #[[$map_2d_dyn]]>
  //     CHECK: %[[ALLOC:.*]] = memref.alloc(%[[D0]]) : memref<?xf32>
  //     CHECK: %[[I2:.*]] = memref.cast %[[ALLOC]] : memref<?xf32> to memref<?xf32, #map>

  //     CHECK: %[[F0:.*]] = constant 0.000000e+00 : f32
  %f0 = constant 0.0 : f32

  //     CHECK: linalg.fill(%[[I2]], %[[F0]]) : memref<?xf32, #[[$map_2d_dyn]]>, f32
  %r = linalg.fill(%A, %f0) : tensor<?xf32>, f32 -> tensor<?xf32>

  //     CHECK:  dealloc %[[ALLOC]] : memref<?xf32>
  //     CHECK:  %[[R:.*]] = memref.tensor_load %[[I2]] : memref<?xf32, #[[$map_2d_dyn]]>
  //     CHECK:  return %[[R]] : tensor<?xf32>
  return %r: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @not_inplace
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: tensor<?x?xf32>
func @not_inplace(%A : tensor<?x?xf32> {linalg.inplaceable = true}) -> tensor<?x?xf32> {
  %f0 = constant 0.0 : f32

  //       CHECK: %[[BUFFER_CAST:.*]] = memref.buffer_cast %[[A]] : memref<?x?xf32

  /// Cross-op multiple uses of %A, the first op which has interfering reads must alloc.
  //       CHECK: %[[ALLOC:.*]] = memref.alloc
  //       CHECK: %[[CAST:.*]] = memref.cast %[[ALLOC]]
  //       CHECK: linalg.fill(%[[CAST]]
  %f = linalg.fill(%A, %f0) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>

  /// The second op has no interfering reads and can reuse.
  //   CHECK-NOT: alloc
  //       CHECK: linalg.matmul{{.*}}outs(%[[BUFFER_CAST]]
  %r = linalg.matmul  ins(%f, %f: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%A: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  return %r: tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @not_inplace
func @not_inplace(%A : tensor<?x?xf32> {linalg.inplaceable = true}) -> tensor<?x?xf32> {
  /// Within op multiple uses of %A, must alloc.
  // CHECK: alloc
  %r = linalg.matmul  ins(%A, %A: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%A: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  return %r: tensor<?x?xf32>
}
// -----

// CHECK-LABEL: func @vec_inplace
func @vec_inplace(%A : tensor<?xf32> {linalg.inplaceable = true}, %vec : vector<4xf32>)
    -> tensor<?xf32>
{
  %c0 = constant 0 : index
  // CHECK-NOT: alloc
  %r = vector.transfer_write %vec, %A[%c0] : vector<4xf32>, tensor<?xf32>
  return %r: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @vec_not_inplace
func @vec_not_inplace(%A : tensor<?xf32> {linalg.inplaceable = true}, %vec : vector<4xf32>)
    -> (tensor<?xf32>, tensor<?xf32>)
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  //       CHECK: %[[BUFFER_CAST:.*]] = memref.buffer_cast {{.*}} : memref<?xf32, #[[$map_2d_dyn]]>

  /// Cross-op multiple uses of %A, the first vector.transfer which has interfering reads must alloc.
  //      CHECK: %[[ALLOC:.*]] = memref.alloc
  // CHECK-NEXT: vector.transfer_write {{.*}}, %[[ALLOC]]
  %r0 = vector.transfer_write %vec, %A[%c0] : vector<4xf32>, tensor<?xf32>

  /// The second vector.transfer has no interfering reads and can reuse the buffer.
  //  CHECK-NOT: alloc
  // CHECK-NEXT: vector.transfer_write {{.*}}, %[[BUFFER_CAST]]
  %r1 = vector.transfer_write %vec, %A[%c1] : vector<4xf32>, tensor<?xf32>
  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @subtensor_insert_fun
func @subtensor_insert_fun(%A : tensor<?xf32> {linalg.inplaceable = true}, %t : tensor<4xf32>)
  ->  tensor<?xf32>
{
  //      CHECK: %[[BUFFER_CAST_A:.*]] = memref.buffer_cast {{.*}} : memref<?xf32
  //      CHECK: %[[BUFFER_CAST_B:.*]] = memref.buffer_cast {{.*}} : memref<4xf32

  //  CHECK-NOT: alloc
  //      CHECK: %[[SV:.*]] = memref.subview %[[BUFFER_CAST_A]]
  //      CHECK: linalg.copy(%[[BUFFER_CAST_B]], %[[SV]])
  %r0 = subtensor_insert %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>
  return %r0: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @subtensor_insert_fun
func @subtensor_insert_fun(%A : tensor<?xf32> {linalg.inplaceable = true}, %t : tensor<4xf32>)
  -> tensor<?xf32>
{
  %f0 = constant 0.0 : f32

  //      CHECK: %[[BUFFER_CAST_A:.*]] = memref.buffer_cast {{.*}} : memref<?xf32
  //      CHECK: %[[BUFFER_CAST_B:.*]] = memref.buffer_cast {{.*}} : memref<4xf32

  //  CHECK-NOT: alloc
  //      CHECK: %[[SV:.*]] = memref.subview %[[BUFFER_CAST_A]]
  //      CHECK: linalg.copy(%[[BUFFER_CAST_B]], %[[SV]])
  %r0 = subtensor_insert %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  /// Overwrite BUFFER_CAST_A inplace.
  //      CHECK: linalg.fill(%[[BUFFER_CAST_A]]
  %r1 = linalg.fill(%r0, %f0) : tensor<?xf32>, f32 -> tensor<?xf32>
  return %r1: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @subtensor_insert_fun
func @subtensor_insert_fun(%A : tensor<?xf32> {linalg.inplaceable = true}, %t : tensor<4xf32>)
  -> tensor<?xf32>
{
  %f0 = constant 0.0 : f32

  //      CHECK: %[[BUFFER_CAST_A:.*]] = memref.buffer_cast {{.*}} : memref<?xf32
  //      CHECK: %[[BUFFER_CAST_B:.*]] = memref.buffer_cast {{.*}} : memref<4xf32

  //      CHECK: linalg.fill(%[[BUFFER_CAST_A]]
  %r0 = linalg.fill(%A, %f0) : tensor<?xf32>, f32 -> tensor<?xf32>

  //  CHECK-NOT: alloc
  //      CHECK: %[[SV:.*]] = memref.subview %[[BUFFER_CAST_A]]
  /// Overwrite BUFFER_CAST_A inplace by copying into the subview.
  //      CHECK: linalg.copy(%[[BUFFER_CAST_B]], %[[SV]])
  %r1 = subtensor_insert %t into %r0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  return %r1: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @subtensor_insert_fun_not_inplace
func @subtensor_insert_fun_not_inplace(%A : tensor<?xf32>, %t : tensor<4xf32>)
  -> tensor<?xf32>
{
  //      CHECK: %[[BUFFER_CAST_A:.*]] = memref.buffer_cast {{.*}} : memref<?xf32
  //      CHECK: %[[BUFFER_CAST_B:.*]] = memref.buffer_cast {{.*}} : memref<4xf32

  //      CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}) : memref<?xf32>
  //      CHECK: linalg.copy(%[[BUFFER_CAST_A]], %[[ALLOC]]) : memref<?xf32{{.*}}, memref<?xf32>
  //      CHECK: %[[SV:.*]] = memref.subview %[[ALLOC]][0] [4] [1] : memref<?xf32> to memref<4xf32>
  //      CHECK: linalg.copy(%[[BUFFER_CAST_B]], %[[SV]]) : memref<4xf32, #map>, memref<4xf32>
  //      CHECK: memref.dealloc %[[ALLOC]] : memref<?xf32>
  %r0 = subtensor_insert %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>
  return %r0: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @subtensor_insert_fun_not_inplace
func @subtensor_insert_fun_not_inplace(%A : tensor<?xf32> {linalg.inplaceable = true}, %t : tensor<4xf32>)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  %f0 = constant 0.0 : f32

  //      CHECK: %[[BUFFER_CAST_A:.*]] = memref.buffer_cast {{.*}} : memref<?xf32
  //      CHECK: %[[BUFFER_CAST_B:.*]] = memref.buffer_cast {{.*}} : memref<4xf32

  //      CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}) : memref<?xf32>
  //      CHECK: linalg.copy(%[[BUFFER_CAST_A]], %[[ALLOC]]) : memref<?xf32{{.*}}, memref<?xf32>
  //      CHECK: %[[SV:.*]] = memref.subview %[[ALLOC]][0] [4] [1] : memref<?xf32> to memref<4xf32>
  //      CHECK: linalg.copy(%[[BUFFER_CAST_B]], %[[SV]]) : memref<4xf32, #map>, memref<4xf32>
  %r0 = subtensor_insert %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // TODO: WAW optimization where result is overwritten without being read.
  //      CHECK: linalg.fill(%[[BUFFER_CAST_A]]
  //      CHECK: memref.dealloc %[[ALLOC]] : memref<?xf32>
  %r1 = linalg.fill(%A, %f0) : tensor<?xf32>, f32 -> tensor<?xf32>
  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @subtensor_fun
func @subtensor_fun(%A : tensor<?xf32> {linalg.inplaceable = true})
  ->  tensor<4xf32>
{
  //      CHECK: %[[BUFFER_CAST_A:.*]] = memref.buffer_cast {{.*}} : memref<?xf32

  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xf32>
  // CHECK: %[[SV:.*]] = memref.subview %[[BUFFER_CAST_A]][0] [4] [1]
  // CHECK: linalg.copy(%[[SV]], %[[ALLOC]])
  %r0 = subtensor %A[0][4][1] : tensor<?xf32> to tensor<4xf32>
  return %r0: tensor<4xf32>
}

// -----

// ANALYSIS-LABEL: func @subtensor_readonly_use
func @subtensor_readonly_use(
    %A : tensor<?x?xf32> {linalg.inplaceable = true},
    %B : tensor<4x4xf32>, %C : tensor<4x4xf32>) ->  tensor<4x4xf32>
{
  // subtensor is only used as a read.
  //     ANALYSIS: subtensor {{.*}} {__inplace_results_attr__ = ["true"]}
  %sA = subtensor %A[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>
  // matmul output operand is not inplaceable at the function boundary.
  //     ANALYSIS: linalg.matmul {{.*}}
  // ANALYSIS-NOT: {__inplace_results_attr__ = ["true"]}
  %D = linalg.matmul  ins(%sA, %B: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%B: tensor<4x4xf32>)
    -> tensor<4x4xf32>
  return %D: tensor<4x4xf32>
}

// -----

// ANALYSIS-LABEL: func @subtensor_nonmatching_subtensor_insert_inplace
func @subtensor_nonmatching_subtensor_insert_inplace(
    %A : tensor<?xf32> {linalg.inplaceable = true}, %idx: index)
  ->  tensor<?xf32>
{
  // subtensor has no matching subtensor_insert and is not just used by known
  // readonly ops.
  //     ANALYSIS: subtensor {{.*}}
  // ANALYSIS-NOT: {__inplace_results_attr__ = ["true"]}
  %r0 = subtensor %A[0][4][1] : tensor<?xf32> to tensor<4xf32>
  // subtensor_insert can bufferize inplace fine.
  //     ANALYSIS: subtensor_insert {{.*}} {__inplace_results_attr__ = ["true"]}
  %r1 = subtensor_insert %r0 into %A[%idx][4][1] : tensor<4xf32> into tensor<?xf32>
  return %r1: tensor<?xf32>
}

// -----

// ANALYSIS-LABEL: func @subtensor_nonmatching_subtensor_insert_non_inplace
func @subtensor_nonmatching_subtensor_insert_non_inplace(
    %A : tensor<?xf32> {linalg.inplaceable = false}, %idx: index)
  ->  tensor<?xf32>
{
  // subtensor has no matching subtensor_insert and is not just used by known
  // readonly ops.
  //     ANALYSIS: subtensor {{.*}} {__inplace_results_attr__ = ["true"]}
  %r0 = subtensor %A[0][4][1] : tensor<?xf32> to tensor<4xf32>
  // subtensor_insert cannot bufferize inplace.
  //     ANALYSIS: subtensor_insert {{.*}}
  // ANALYSIS-NOT: {__inplace_results_attr__ = ["true"]}
  %r1 = subtensor_insert %r0 into %A[%idx][4][1] : tensor<4xf32> into tensor<?xf32>
  return %r1: tensor<?xf32>
}

// -----

// ANALYSIS-LABEL: func @subtensor_matching_subtensor_insert
func @subtensor_matching_subtensor_insert(%A : tensor<?xf32> {linalg.inplaceable = true})
  ->  tensor<?xf32>
{
  // subtensor has a matching subtensor_insert that bufferizes inplace.
  // TODO: Atm subtensor is not inplaceable but can be.
  // In the grander scheme, this will canonicalize away beforehand.
  //     ANALYSIS: subtensor {{.*}}
  // ANALYSIS-NOT: {__inplace_results_attr__ = ["true"]}
  %r0 = subtensor %A[0][4][1] : tensor<?xf32> to tensor<4xf32>
  // subtensor_insert can bufferize inplace fine.
  //     ANALYSIS: subtensor_insert {{.*}} {__inplace_results_attr__ = ["true"]}
  %r1 = subtensor_insert %r0 into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>
  return %r1: tensor<?xf32>
}

// -----

// ANALYSIS-LABEL: func @subtensor_matching_and_nonmatching_1
func @subtensor_matching_and_nonmatching_1(%A : tensor<?xf32> {linalg.inplaceable = true}, %idx: index)
  ->  (tensor<?xf32>, tensor<?xf32>)
{
  // %r1 is not inplaceable and %r2 is a matching subtensor_insert so %r0 could
  // be inplaceable.
  // In the grander scheme, %r2 will canonicalize away beforehand but %r0 will still
  // not be inplaceable as the production of %r1 may involve a self-copy.
  //     ANALYSIS: subtensor {{.*}}
  // ANALYSIS-NOT: {__inplace_results_attr__ = ["true"]}
  %r0 = subtensor %A[0][4][1] : tensor<?xf32> to tensor<4xf32>
  //     ANALYSIS: subtensor_insert {{.*}}
  // ANALYSIS-NOT: {__inplace_results_attr__ = ["true"]}
  %r1 = subtensor_insert %r0 into %A[%idx][4][1] : tensor<4xf32> into tensor<?xf32>
  //     ANALYSIS: subtensor_insert {{.*}} {__inplace_results_attr__ = ["true"]}
  %r2 = subtensor_insert %r0 into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>
  return %r1, %r2: tensor<?xf32>, tensor<?xf32>
}

// -----

// ANALYSIS-LABEL: func @subtensor_matching_and_nonmatching_2
func @subtensor_matching_and_nonmatching_2(%A : tensor<?xf32> {linalg.inplaceable = true}, %idx: index)
  ->  (tensor<?xf32>, tensor<?xf32>)
{
  // %r1 is not inplaceable and %r2 is a matching subtensor_insert so %r0 should
  // be inplaceable.
  // In the grander scheme, %r2 will canonicalize away beforehand and %r0 will become
  // inplaceable by reducing to the `subtensor_nonmatching_subtensor_insert_non_inplace`
  // case,
  //     ANALYSIS: subtensor {{.*}}
  // ANALYSIS-NOT: {__inplace_results_attr__ = ["true"]}
  %r0 = subtensor %A[0][4][1] : tensor<?xf32> to tensor<4xf32>
  //     ANALYSIS: subtensor_insert {{.*}}
  // ANALYSIS-NOT: {__inplace_results_attr__ = ["true"]}
  %r2 = subtensor_insert %r0 into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>
  //     ANALYSIS: subtensor_insert {{.*}} {__inplace_results_attr__ = ["true"]}
  %r1 = subtensor_insert %r0 into %A[%idx][4][1] : tensor<4xf32> into tensor<?xf32>

  return %r1, %r2: tensor<?xf32>, tensor<?xf32>
}

// -----

// TODO: unknown ops, linalg chain success, linalg chain failure.

