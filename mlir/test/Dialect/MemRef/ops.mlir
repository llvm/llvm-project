// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[$strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECK-DAG: #[[$strided2DOFF0:.*]] = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
// CHECK-DAG: #[[$strided3DOFF0:.*]] = affine_map<(d0, d1, d2)[s0, s1] -> (d0 * s0 + d1 * s1 + d2)>

// CHECK-LABEL: test_buffer_cast
func @test_buffer_cast(%arg0: tensor<?xi64>, %arg1: tensor<*xi64>) -> (memref<?xi64, affine_map<(d0) -> (d0 + 7)>>, memref<*xi64, 1>) {
  %0 = memref.buffer_cast %arg0 : memref<?xi64, affine_map<(d0) -> (d0 + 7)>>
  %1 = memref.buffer_cast %arg1 : memref<*xi64, 1>
  return %0, %1 : memref<?xi64, affine_map<(d0) -> (d0 + 7)>>, memref<*xi64, 1>
}

// CHECK-LABEL: func @memref_reinterpret_cast
func @memref_reinterpret_cast(%in: memref<?xf32>)
    -> memref<10x?xf32, offset: ?, strides: [?, 1]> {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %out = memref.reinterpret_cast %in to
           offset: [%c0], sizes: [10, %c10], strides: [%c10, 1]
           : memref<?xf32> to memref<10x?xf32, offset: ?, strides: [?, 1]>
  return %out : memref<10x?xf32, offset: ?, strides: [?, 1]>
}

// CHECK-LABEL: func @memref_reshape(
func @memref_reshape(%unranked: memref<*xf32>, %shape1: memref<1xi32>,
         %shape2: memref<2xi32>, %shape3: memref<?xi32>) -> memref<*xf32> {
  %dyn_vec = memref.reshape %unranked(%shape1)
               : (memref<*xf32>, memref<1xi32>) -> memref<?xf32>
  %dyn_mat = memref.reshape %dyn_vec(%shape2)
               : (memref<?xf32>, memref<2xi32>) -> memref<?x?xf32>
  %new_unranked = memref.reshape %dyn_mat(%shape3)
               : (memref<?x?xf32>, memref<?xi32>) -> memref<*xf32>
  return %new_unranked : memref<*xf32>
}

// CHECK-LABEL: memref.global @memref0 : memref<2xf32>
memref.global @memref0 : memref<2xf32>

// CHECK-LABEL: memref.global constant @memref1 : memref<2xf32> = dense<[0.000000e+00, 1.000000e+00]>
memref.global constant @memref1 : memref<2xf32> = dense<[0.0, 1.0]>

// CHECK-LABEL: memref.global @memref2 : memref<2xf32> = uninitialized
memref.global @memref2 : memref<2xf32>  = uninitialized

// CHECK-LABEL: memref.global "private" @memref3 : memref<2xf32> = uninitialized
memref.global "private" @memref3 : memref<2xf32>  = uninitialized

// CHECK-LABEL: memref.global "private" constant @memref4 : memref<2xf32> = uninitialized
memref.global "private" constant @memref4 : memref<2xf32>  = uninitialized

// CHECK-LABEL: func @write_global_memref
func @write_global_memref() {
  %0 = memref.get_global @memref0 : memref<2xf32>
  %1 = constant dense<[1.0, 2.0]> : tensor<2xf32>
  memref.tensor_store %1, %0 : memref<2xf32>
  return
}

// CHECK-LABEL: func @read_global_memref
func @read_global_memref() {
  %0 = memref.get_global @memref0 : memref<2xf32>
  %1 = memref.tensor_load %0 : memref<2xf32>
  return
}

// CHECK-LABEL: func @memref_clone
func @memref_clone() {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.cast %0 : memref<2xf32> to memref<*xf32>
  %2 = memref.clone %1 : memref<*xf32> to memref<*xf32>
  return
}

// CHECK-LABEL: func @memref_copy
func @memref_copy() {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.cast %0 : memref<2xf32> to memref<*xf32>
  %2 = memref.alloc() : memref<2xf32>
  %3 = memref.cast %0 : memref<2xf32> to memref<*xf32>
  memref.copy %1, %3 : memref<*xf32> to memref<*xf32>
  return
}

// CHECK-LABEL: func @memref_dealloc
func @memref_dealloc() {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.cast %0 : memref<2xf32> to memref<*xf32>
  memref.dealloc %1 : memref<*xf32>
  return
}


// CHECK-LABEL: func @memref_alloca_scope
func @memref_alloca_scope() {
  memref.alloca_scope {
    memref.alloca_scope.return
  }
  return
}

func @expand_collapse_shape_static(%arg0: memref<3x4x5xf32>,
                                   %arg1: tensor<3x4x5xf32>,
                                   %arg2: tensor<3x?x5xf32>) {
  // Reshapes that collapse and expand back a contiguous buffer.
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] :
    memref<3x4x5xf32> into memref<12x5xf32>
  %r0 = memref.expand_shape %0 [[0, 1], [2]] :
    memref<12x5xf32> into memref<3x4x5xf32>
  %1 = memref.collapse_shape %arg0 [[0], [1, 2]] :
    memref<3x4x5xf32> into memref<3x20xf32>
  %r1 = memref.expand_shape %1 [[0], [1, 2]] :
    memref<3x20xf32> into memref<3x4x5xf32>
  %2 = memref.collapse_shape %arg0 [[0, 1, 2]] :
    memref<3x4x5xf32> into memref<60xf32>
  %r2 = memref.expand_shape %2 [[0, 1, 2]] :
    memref<60xf32> into memref<3x4x5xf32>
  // Reshapes that expand and collapse back a contiguous buffer with some 1's.
  %3 = memref.expand_shape %arg0 [[0, 1], [2], [3, 4]] :
    memref<3x4x5xf32> into memref<1x3x4x1x5xf32>
  %r3 = memref.collapse_shape %3 [[0, 1], [2], [3, 4]] :
    memref<1x3x4x1x5xf32> into memref<3x4x5xf32>
  // Reshapes on tensors.
  %t0 = linalg.tensor_expand_shape %arg1 [[0, 1], [2], [3, 4]] :
    tensor<3x4x5xf32> into tensor<1x3x4x1x5xf32>
  %rt0 = linalg.tensor_collapse_shape %t0 [[0, 1], [2], [3, 4]] :
    tensor<1x3x4x1x5xf32> into tensor<3x4x5xf32>
  %t1 = linalg.tensor_expand_shape %arg2 [[0, 1], [2], [3, 4]] :
    tensor<3x?x5xf32> into tensor<1x3x?x1x5xf32>
  %rt1 = linalg.tensor_collapse_shape %t1 [[0], [1, 2], [3, 4]] :
    tensor<1x3x?x1x5xf32> into tensor<1x?x5xf32>
  return
}
// CHECK-LABEL: func @expand_collapse_shape_static
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<12x5xf32>
//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<12x5xf32> into memref<3x4x5xf32>
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0], [1, 2]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<3x20xf32>
//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0], [1, 2]]
//  CHECK-SAME:     memref<3x20xf32> into memref<3x4x5xf32>
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1, 2]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<60xf32>
//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1, 2]]
//  CHECK-SAME:     memref<60xf32> into memref<3x4x5xf32>
//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2], [3, 4]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<1x3x4x1x5xf32>
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3, 4]]
//  CHECK-SAME:     memref<1x3x4x1x5xf32> into memref<3x4x5xf32>
//
//       CHECK:   linalg.tensor_expand_shape {{.*}}: tensor<3x4x5xf32> into tensor<1x3x4x1x5xf32>
//       CHECK:   linalg.tensor_collapse_shape {{.*}}: tensor<1x3x4x1x5xf32> into tensor<3x4x5xf32>
//       CHECK:   linalg.tensor_expand_shape {{.*}}: tensor<3x?x5xf32> into tensor<1x3x?x1x5xf32>
//       CHECK:   linalg.tensor_collapse_shape {{.*}}: tensor<1x3x?x1x5xf32> into tensor<1x?x5xf32>


func @expand_collapse_shape_dynamic(%arg0: memref<?x?x?xf32>,
         %arg1: memref<?x?x?xf32, offset : 0, strides : [?, ?, 1]>,
         %arg2: memref<?x?x?xf32, offset : ?, strides : [?, ?, 1]>) {
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] :
    memref<?x?x?xf32> into memref<?x?xf32>
  %r0 = memref.expand_shape %0 [[0, 1], [2]] :
    memref<?x?xf32> into memref<?x4x?xf32>
  %1 = memref.collapse_shape %arg1 [[0, 1], [2]] :
    memref<?x?x?xf32, offset : 0, strides : [?, ?, 1]> into
    memref<?x?xf32, offset : 0, strides : [?, 1]>
  %r1 = memref.expand_shape %1 [[0, 1], [2]] :
    memref<?x?xf32, offset : 0, strides : [?, 1]> into
    memref<?x4x?xf32, offset : 0, strides : [?, ?, 1]>
  %2 = memref.collapse_shape %arg2 [[0, 1], [2]] :
    memref<?x?x?xf32, offset : ?, strides : [?, ?, 1]> into
    memref<?x?xf32, offset : ?, strides : [?, 1]>
  %r2 = memref.expand_shape %2 [[0, 1], [2]] :
    memref<?x?xf32, offset : ?, strides : [?, 1]> into
    memref<?x4x?xf32, offset : ?, strides : [?, ?, 1]>
  return
}
// CHECK-LABEL: func @expand_collapse_shape_dynamic
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?x?xf32> into memref<?x?xf32>
//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?xf32> into memref<?x4x?xf32>
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?x?xf32, #[[$strided3DOFF0]]> into memref<?x?xf32, #[[$strided2DOFF0]]>
//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?xf32, #[[$strided2DOFF0]]> into memref<?x4x?xf32, #[[$strided3DOFF0]]>
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?x?xf32, #[[$strided3D]]> into memref<?x?xf32, #[[$strided2D]]>
//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?xf32, #[[$strided2D]]> into memref<?x4x?xf32, #[[$strided3D]]>

func @expand_collapse_shape_zero_dim(%arg0 : memref<1x1xf32>, %arg1 : memref<f32>)
    -> (memref<f32>, memref<1x1xf32>) {
  %0 = memref.collapse_shape %arg0 [] : memref<1x1xf32> into memref<f32>
  %1 = memref.expand_shape %0 [] : memref<f32> into memref<1x1xf32>
  return %0, %1 : memref<f32>, memref<1x1xf32>
}
// CHECK-LABEL: func @expand_collapse_shape_zero_dim
//       CHECK:   memref.collapse_shape %{{.*}} [] : memref<1x1xf32> into memref<f32>
//       CHECK:   memref.expand_shape %{{.*}} [] : memref<f32> into memref<1x1xf32>

func @collapse_shape_to_dynamic
  (%arg0: memref<?x?x?x4x?xf32>) -> memref<?x?x?xf32> {
  %0 = memref.collapse_shape %arg0 [[0], [1], [2, 3, 4]] :
    memref<?x?x?x4x?xf32> into memref<?x?x?xf32>
  return %0 : memref<?x?x?xf32>
}
//      CHECK: func @collapse_shape_to_dynamic
//      CHECK:   memref.collapse_shape
// CHECK-SAME:    [0], [1], [2, 3, 4]
