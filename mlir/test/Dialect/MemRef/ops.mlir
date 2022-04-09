// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[$strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECK-DAG: #[[$strided2DOFF0:.*]] = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
// CHECK-DAG: #[[$strided3DOFF0:.*]] = affine_map<(d0, d1, d2)[s0, s1] -> (d0 * s0 + d1 * s1 + d2)>
// CHECK-DAG: #[[$strided2D42:.*]] = affine_map<(d0, d1) -> (d0 * 42 + d1)>

// CHECK-LABEL: func @memref_reinterpret_cast
func @memref_reinterpret_cast(%in: memref<?xf32>)
    -> memref<10x?xf32, offset: ?, strides: [?, 1]> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %out = memref.reinterpret_cast %in to
           offset: [%c0], sizes: [10, %c10], strides: [%c10, 1]
           : memref<?xf32> to memref<10x?xf32, offset: ?, strides: [?, 1]>
  return %out : memref<10x?xf32, offset: ?, strides: [?, 1]>
}

// CHECK-LABEL: func @memref_reinterpret_cast_static_to_dynamic_sizes
func @memref_reinterpret_cast_static_to_dynamic_sizes(%in: memref<?xf32>)
    -> memref<10x?xf32, offset: ?, strides: [?, 1]> {
  %out = memref.reinterpret_cast %in to
           offset: [1], sizes: [10, 10], strides: [1, 1]
           : memref<?xf32> to memref<10x?xf32, offset: ?, strides: [?, 1]>
  return %out : memref<10x?xf32, offset: ?, strides: [?, 1]>
}

// CHECK-LABEL: func @memref_reinterpret_cast_dynamic_offset
func @memref_reinterpret_cast_dynamic_offset(%in: memref<?xf32>, %offset: index)
    -> memref<10x?xf32, offset: ?, strides: [?, 1]> {
  %out = memref.reinterpret_cast %in to
           offset: [%offset], sizes: [10, 10], strides: [1, 1]
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
  %1 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  memref.tensor_store %1, %0 : memref<2xf32>
  return
}

// CHECK-LABEL: func @read_global_memref
func @read_global_memref() {
  %0 = memref.get_global @memref0 : memref<2xf32>
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

// CHECK-LABEL: func @expand_collapse_shape_static
func @expand_collapse_shape_static(
    %arg0: memref<3x4x5xf32>,
    %arg1: tensor<3x4x5xf32>,
    %arg2: tensor<3x?x5xf32>,
    %arg3: memref<30x20xf32, offset : 100, strides : [4000, 2]>,
    %arg4: memref<1x5xf32, affine_map<(d0, d1)[s0] -> (d0 * 5 + s0 + d1)>>,
    %arg5: memref<f32>,
    %arg6: memref<3x4x5xf32, offset: 0, strides: [240, 60, 10]>,
    %arg7: memref<1x2049xi64, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>>) {
  // Reshapes that collapse and expand back a contiguous buffer.
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<12x5xf32>
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] :
    memref<3x4x5xf32> into memref<12x5xf32>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<12x5xf32> into memref<3x4x5xf32>
  %r0 = memref.expand_shape %0 [[0, 1], [2]] :
    memref<12x5xf32> into memref<3x4x5xf32>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0], [1, 2]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<3x20xf32>
  %1 = memref.collapse_shape %arg0 [[0], [1, 2]] :
    memref<3x4x5xf32> into memref<3x20xf32>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0], [1, 2]]
//  CHECK-SAME:     memref<3x20xf32> into memref<3x4x5xf32>
  %r1 = memref.expand_shape %1 [[0], [1, 2]] :
    memref<3x20xf32> into memref<3x4x5xf32>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1, 2]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<60xf32>
  %2 = memref.collapse_shape %arg0 [[0, 1, 2]] :
    memref<3x4x5xf32> into memref<60xf32>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1, 2]]
//  CHECK-SAME:     memref<60xf32> into memref<3x4x5xf32>
  %r2 = memref.expand_shape %2 [[0, 1, 2]] :
      memref<60xf32> into memref<3x4x5xf32>

//       CHECK:   memref.expand_shape {{.*}} []
//  CHECK-SAME:     memref<f32> into memref<1x1xf32>
  %r5 = memref.expand_shape %arg5 [] :
      memref<f32> into memref<1x1xf32>

// Reshapes with a custom layout map.
//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0], [1, 2]]
  %l0 = memref.expand_shape %arg3 [[0], [1, 2]] :
      memref<30x20xf32, offset : 100, strides : [4000, 2]>
      into memref<30x4x5xf32, offset : 100, strides : [4000, 10, 2]>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
  %l1 = memref.expand_shape %arg3 [[0, 1], [2]] :
      memref<30x20xf32, offset : 100, strides : [4000, 2]>
      into memref<2x15x20xf32, offset : 100, strides : [60000, 4000, 2]>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0], [1, 2]]
  %r4 = memref.expand_shape %arg4 [[0], [1, 2]] :
      memref<1x5xf32, affine_map<(d0, d1)[s0] -> (d0 * 5 + s0 + d1)>> into
      memref<1x1x5xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 5 + s0 + d2 + d1 * 5)>>

  // Note: Only the collapsed two shapes are contiguous in the follow test case.
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
  %r6 = memref.collapse_shape %arg6 [[0, 1], [2]] :
      memref<3x4x5xf32, offset: 0, strides: [240, 60, 10]> into
      memref<12x5xf32, offset: 0, strides: [60, 10]>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1]]
  %r7 = memref.collapse_shape %arg7 [[0, 1]] :
      memref<1x2049xi64, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>> into
      memref<2049xi64, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>

  // Reshapes that expand and collapse back a contiguous buffer with some 1's.
//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2], [3, 4]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<1x3x4x1x5xf32>
  %3 = memref.expand_shape %arg0 [[0, 1], [2], [3, 4]] :
    memref<3x4x5xf32> into memref<1x3x4x1x5xf32>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3, 4]]
//  CHECK-SAME:     memref<1x3x4x1x5xf32> into memref<3x4x5xf32>
  %r3 = memref.collapse_shape %3 [[0, 1], [2], [3, 4]] :
    memref<1x3x4x1x5xf32> into memref<3x4x5xf32>

  // Reshapes on tensors.
//       CHECK:   tensor.expand_shape {{.*}}: tensor<3x4x5xf32> into tensor<1x3x4x1x5xf32>
  %t0 = tensor.expand_shape %arg1 [[0, 1], [2], [3, 4]] :
    tensor<3x4x5xf32> into tensor<1x3x4x1x5xf32>

//       CHECK:   tensor.collapse_shape {{.*}}: tensor<1x3x4x1x5xf32> into tensor<3x4x5xf32>
  %rt0 = tensor.collapse_shape %t0 [[0, 1], [2], [3, 4]] :
    tensor<1x3x4x1x5xf32> into tensor<3x4x5xf32>

//       CHECK:   tensor.expand_shape {{.*}}: tensor<3x?x5xf32> into tensor<1x3x?x1x5xf32>
  %t1 = tensor.expand_shape %arg2 [[0, 1], [2], [3, 4]] :
    tensor<3x?x5xf32> into tensor<1x3x?x1x5xf32>

//       CHECK:   tensor.collapse_shape {{.*}}: tensor<1x3x?x1x5xf32> into tensor<1x?x5xf32>
  %rt1 = tensor.collapse_shape %t1 [[0], [1, 2], [3, 4]] :
    tensor<1x3x?x1x5xf32> into tensor<1x?x5xf32>
  return
}

// CHECK-LABEL: func @expand_collapse_shape_dynamic
func @expand_collapse_shape_dynamic(%arg0: memref<?x?x?xf32>,
         %arg1: memref<?x?x?xf32, offset : 0, strides : [?, ?, 1]>,
         %arg2: memref<?x?x?xf32, offset : ?, strides : [?, ?, 1]>,
         %arg3: memref<?x42xf32, offset : 0, strides : [42, 1]>) {
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?x?xf32> into memref<?x?xf32>
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] :
    memref<?x?x?xf32> into memref<?x?xf32>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?xf32> into memref<?x4x?xf32>
  %r0 = memref.expand_shape %0 [[0, 1], [2]] :
    memref<?x?xf32> into memref<?x4x?xf32>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?x?xf32, #[[$strided3DOFF0]]> into memref<?x?xf32, #[[$strided2DOFF0]]>
  %1 = memref.collapse_shape %arg1 [[0, 1], [2]] :
    memref<?x?x?xf32, offset : 0, strides : [?, ?, 1]> into
    memref<?x?xf32, offset : 0, strides : [?, 1]>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?xf32, #[[$strided2DOFF0]]> into memref<?x4x?xf32, #[[$strided3DOFF0]]>
  %r1 = memref.expand_shape %1 [[0, 1], [2]] :
    memref<?x?xf32, offset : 0, strides : [?, 1]> into
    memref<?x4x?xf32, offset : 0, strides : [?, ?, 1]>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?x?xf32, #[[$strided3D]]> into memref<?x?xf32, #[[$strided2D]]>
  %2 = memref.collapse_shape %arg2 [[0, 1], [2]] :
    memref<?x?x?xf32, offset : ?, strides : [?, ?, 1]> into
    memref<?x?xf32, offset : ?, strides : [?, 1]>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?xf32, #[[$strided2D]]> into memref<?x4x?xf32, #[[$strided3D]]>
  %r2 = memref.expand_shape %2 [[0, 1], [2]] :
    memref<?x?xf32, offset : ?, strides : [?, 1]> into
    memref<?x4x?xf32, offset : ?, strides : [?, ?, 1]>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:     memref<?x42xf32, #[[$strided2D42]]> into memref<?xf32>
  %3 = memref.collapse_shape %arg3 [[0, 1]] :
    memref<?x42xf32, offset : 0, strides : [42, 1]> into
    memref<?xf32, offset : 0, strides : [1]>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:     memref<?xf32> into memref<?x42xf32>
  %r3 = memref.expand_shape %3 [[0, 1]] :
    memref<?xf32, offset : 0, strides : [1]> into memref<?x42xf32>
  return
}

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

// -----

// CHECK-LABEL: func @expand_collapse_shape_transposed_layout
func @expand_collapse_shape_transposed_layout(
    %m0: memref<?x?xf32, offset : 0, strides : [1, 10]>,
    %m1: memref<4x5x6xf32, offset : 0, strides : [1, ?, 1000]>) {

  %r0 = memref.expand_shape %m0 [[0], [1, 2]] :
    memref<?x?xf32, offset : 0, strides : [1, 10]> into
    memref<?x?x5xf32, offset : 0, strides : [1, 50, 10]>
  %rr0 = memref.collapse_shape %r0 [[0], [1, 2]] :
    memref<?x?x5xf32, offset : 0, strides : [1, 50, 10]> into
    memref<?x?xf32, offset : 0, strides : [1, 10]>

  %r1 = memref.expand_shape %m1 [[0, 1], [2], [3, 4]] :
    memref<4x5x6xf32, offset : 0, strides : [1, ?, 1000]> into 
    memref<2x2x5x2x3xf32, offset : 0, strides : [2, 1, ?, 3000, 1000]>
  %rr1 = memref.collapse_shape %r1 [[0, 1], [2], [3, 4]] :
    memref<2x2x5x2x3xf32, offset : 0, strides : [2, 1, ?, 3000, 1000]> into
    memref<4x5x6xf32, offset : 0, strides : [1, ?, 1000]>
  return
}

// -----

func @rank(%t : memref<4x4x?xf32>) {
  // CHECK: %{{.*}} = memref.rank %{{.*}} : memref<4x4x?xf32>
  %0 = "memref.rank"(%t) : (memref<4x4x?xf32>) -> index

  // CHECK: %{{.*}} = memref.rank %{{.*}} : memref<4x4x?xf32>
  %1 = memref.rank %t : memref<4x4x?xf32>
  return
}

// ------

// CHECK-LABEL: func @atomic_rmw
// CHECK-SAME: ([[BUF:%.*]]: memref<10xf32>, [[VAL:%.*]]: f32, [[I:%.*]]: index)
func @atomic_rmw(%I: memref<10xf32>, %val: f32, %i : index) {
  %x = memref.atomic_rmw addf %val, %I[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: memref.atomic_rmw addf [[VAL]], [[BUF]]{{\[}}[[I]]]
  return
}

// CHECK-LABEL: func @generic_atomic_rmw
// CHECK-SAME: ([[BUF:%.*]]: memref<1x2xf32>, [[I:%.*]]: index, [[J:%.*]]: index)
func @generic_atomic_rmw(%I: memref<1x2xf32>, %i : index, %j : index) {
  %x = memref.generic_atomic_rmw %I[%i, %j] : memref<1x2xf32> {
  // CHECK-NEXT: memref.generic_atomic_rmw [[BUF]]{{\[}}[[I]], [[J]]] : memref
    ^bb0(%old_value : f32):
      %c1 = arith.constant 1.0 : f32
      %out = arith.addf %c1, %old_value : f32
      memref.atomic_yield %out : f32
  // CHECK: index_attr = 8 : index
  } { index_attr = 8 : index }
  return
}
