// RUN: mlir-opt %s -linalg-comprehensive-func-bufferize -split-input-file | FileCheck %s

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
