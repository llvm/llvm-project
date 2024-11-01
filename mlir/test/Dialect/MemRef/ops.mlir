// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: func @memref_reinterpret_cast
func.func @memref_reinterpret_cast(%in: memref<?xf32>)
    -> memref<10x?xf32, strided<[?, 1], offset: ?>> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %out = memref.reinterpret_cast %in to
           offset: [%c0], sizes: [10, %c10], strides: [%c10, 1]
           : memref<?xf32> to memref<10x?xf32, strided<[?, 1], offset: ?>>
  return %out : memref<10x?xf32, strided<[?, 1], offset: ?>>
}

// CHECK-LABEL: func @memref_reinterpret_cast_static_to_dynamic_sizes
func.func @memref_reinterpret_cast_static_to_dynamic_sizes(%in: memref<?xf32>)
    -> memref<10x?xf32, strided<[?, 1], offset: ?>> {
  %out = memref.reinterpret_cast %in to
           offset: [1], sizes: [10, 10], strides: [1, 1]
           : memref<?xf32> to memref<10x?xf32, strided<[?, 1], offset: ?>>
  return %out : memref<10x?xf32, strided<[?, 1], offset: ?>>
}

// CHECK-LABEL: func @memref_reinterpret_cast_dynamic_offset
func.func @memref_reinterpret_cast_dynamic_offset(%in: memref<?xf32>, %offset: index)
    -> memref<10x?xf32, strided<[?, 1], offset: ?>> {
  %out = memref.reinterpret_cast %in to
           offset: [%offset], sizes: [10, 10], strides: [1, 1]
           : memref<?xf32> to memref<10x?xf32, strided<[?, 1], offset: ?>>
  return %out : memref<10x?xf32, strided<[?, 1], offset: ?>>
}

// CHECK-LABEL: func @memref_reshape(
func.func @memref_reshape(%unranked: memref<*xf32>, %shape1: memref<1xi32>,
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
func.func @write_global_memref() {
  %0 = memref.get_global @memref0 : memref<2xf32>
  %1 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  memref.tensor_store %1, %0 : memref<2xf32>
  return
}

// CHECK-LABEL: func @read_global_memref
func.func @read_global_memref() {
  %0 = memref.get_global @memref0 : memref<2xf32>
  return
}

// CHECK-LABEL: func @memref_copy
func.func @memref_copy() {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.cast %0 : memref<2xf32> to memref<*xf32>
  %2 = memref.alloc() : memref<2xf32>
  %3 = memref.cast %0 : memref<2xf32> to memref<*xf32>
  memref.copy %1, %3 : memref<*xf32> to memref<*xf32>
  return
}

// CHECK-LABEL: func @memref_dealloc
func.func @memref_dealloc() {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.cast %0 : memref<2xf32> to memref<*xf32>
  memref.dealloc %1 : memref<*xf32>
  return
}


// CHECK-LABEL: func @memref_alloca_scope
func.func @memref_alloca_scope() {
  memref.alloca_scope {
    memref.alloca_scope.return
  }
  return
}

// CHECK-LABEL: func @expand_collapse_shape_static
func.func @expand_collapse_shape_static(
    %arg0: memref<3x4x5xf32>,
    %arg1: tensor<3x4x5xf32>,
    %arg2: tensor<3x?x5xf32>,
    %arg3: memref<30x20xf32, strided<[4000, 2], offset: 100>>,
    %arg4: memref<1x5xf32, strided<[5, 1], offset: ?>>,
    %arg5: memref<f32>,
    %arg6: memref<3x4x5xf32, strided<[240, 60, 10], offset: 0>>,
    %arg7: memref<1x2049xi64, strided<[?, ?], offset: ?>>) {
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
      memref<30x20xf32, strided<[4000, 2], offset: 100>>
      into memref<30x4x5xf32, strided<[4000, 10, 2], offset: 100>>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
  %l1 = memref.expand_shape %arg3 [[0, 1], [2]] :
      memref<30x20xf32, strided<[4000, 2], offset: 100>>
      into memref<2x15x20xf32, strided<[60000, 4000, 2], offset: 100>>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0], [1, 2]]
  %r4 = memref.expand_shape %arg4 [[0], [1, 2]] :
      memref<1x5xf32, strided<[5, 1], offset: ?>> into
      memref<1x1x5xf32, strided<[5, 5, 1], offset: ?>>

  // Note: Only the collapsed two shapes are contiguous in the follow test case.
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
  %r6 = memref.collapse_shape %arg6 [[0, 1], [2]] :
      memref<3x4x5xf32, strided<[240, 60, 10], offset: 0>> into
      memref<12x5xf32, strided<[60, 10], offset: 0>>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1]]
  %r7 = memref.collapse_shape %arg7 [[0, 1]] :
      memref<1x2049xi64, strided<[?, ?], offset: ?>> into
      memref<2049xi64, strided<[?], offset: ?>>

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
func.func @expand_collapse_shape_dynamic(%arg0: memref<?x?x?xf32>,
         %arg1: memref<?x?x?xf32, strided<[?, ?, 1], offset: 0>>,
         %arg2: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>,
         %arg3: memref<?x42xf32, strided<[42, 1], offset: 0>>) {
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?x?xf32> into memref<?x?xf32>
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] :
    memref<?x?x?xf32> into memref<?x?xf32>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?xf32> into memref<?x4x?xf32>
  %r0 = memref.expand_shape %0 [[0, 1], [2]] :
    memref<?x?xf32> into memref<?x4x?xf32>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?x?xf32, strided<[?, ?, 1]>> into memref<?x?xf32, strided<[?, 1]>>
  %1 = memref.collapse_shape %arg1 [[0, 1], [2]] :
    memref<?x?x?xf32, strided<[?, ?, 1], offset: 0>> into
    memref<?x?xf32, strided<[?, 1], offset: 0>>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?xf32, strided<[?, 1]>> into memref<?x4x?xf32, strided<[?, ?, 1]>>
  %r1 = memref.expand_shape %1 [[0, 1], [2]] :
    memref<?x?xf32, strided<[?, 1], offset: 0>> into
    memref<?x4x?xf32, strided<[?, ?, 1], offset: 0>>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>> into memref<?x?xf32, strided<[?, 1], offset: ?>>
  %2 = memref.collapse_shape %arg2 [[0, 1], [2]] :
    memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>> into
    memref<?x?xf32, strided<[?, 1], offset: ?>>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?xf32, strided<[?, 1], offset: ?>> into memref<?x4x?xf32, strided<[?, ?, 1], offset: ?>>
  %r2 = memref.expand_shape %2 [[0, 1], [2]] :
    memref<?x?xf32, strided<[?, 1], offset: ?>> into
    memref<?x4x?xf32, strided<[?, ?, 1], offset: ?>>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:     memref<?x42xf32, strided<[42, 1]>> into memref<?xf32, strided<[1]>>
  %3 = memref.collapse_shape %arg3 [[0, 1]] :
    memref<?x42xf32, strided<[42, 1], offset: 0>> into
    memref<?xf32, strided<[1]>>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:     memref<?xf32, strided<[1]>> into memref<?x42xf32>
  %r3 = memref.expand_shape %3 [[0, 1]] :
    memref<?xf32, strided<[1]>> into memref<?x42xf32>
  return
}

func.func @expand_collapse_shape_zero_dim(%arg0 : memref<1x1xf32>, %arg1 : memref<f32>)
    -> (memref<f32>, memref<1x1xf32>) {
  %0 = memref.collapse_shape %arg0 [] : memref<1x1xf32> into memref<f32>
  %1 = memref.expand_shape %0 [] : memref<f32> into memref<1x1xf32>
  return %0, %1 : memref<f32>, memref<1x1xf32>
}
// CHECK-LABEL: func @expand_collapse_shape_zero_dim
//       CHECK:   memref.collapse_shape %{{.*}} [] : memref<1x1xf32> into memref<f32>
//       CHECK:   memref.expand_shape %{{.*}} [] : memref<f32> into memref<1x1xf32>

func.func @collapse_shape_to_dynamic
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
func.func @expand_collapse_shape_transposed_layout(
    %m0: memref<?x?xf32, strided<[1, 10], offset: 0>>,
    %m1: memref<4x5x6xf32, strided<[1, ?, 1000], offset: 0>>) {

  %r0 = memref.expand_shape %m0 [[0], [1, 2]] :
    memref<?x?xf32, strided<[1, 10], offset: 0>> into
    memref<?x?x5xf32, strided<[1, 50, 10], offset: 0>>
  %rr0 = memref.collapse_shape %r0 [[0], [1, 2]] :
    memref<?x?x5xf32, strided<[1, 50, 10], offset: 0>> into
    memref<?x?xf32, strided<[1, 10], offset: 0>>

  %r1 = memref.expand_shape %m1 [[0, 1], [2], [3, 4]] :
    memref<4x5x6xf32, strided<[1, ?, 1000], offset: 0>> into
    memref<2x2x5x2x3xf32, strided<[2, 1, ?, 3000, 1000], offset: 0>>
  %rr1 = memref.collapse_shape %r1 [[0, 1], [2], [3, 4]] :
    memref<2x2x5x2x3xf32, strided<[2, 1, ?, 3000, 1000], offset: 0>> into
    memref<4x5x6xf32, strided<[1, ?, 1000], offset: 0>>
  return
}

// -----

func.func @rank(%t : memref<4x4x?xf32>) {
  // CHECK: %{{.*}} = memref.rank %{{.*}} : memref<4x4x?xf32>
  %0 = "memref.rank"(%t) : (memref<4x4x?xf32>) -> index

  // CHECK: %{{.*}} = memref.rank %{{.*}} : memref<4x4x?xf32>
  %1 = memref.rank %t : memref<4x4x?xf32>
  return
}

// ------

// CHECK-LABEL: func @atomic_rmw
// CHECK-SAME: ([[BUF:%.*]]: memref<10xf32>, [[VAL:%.*]]: f32, [[I:%.*]]: index)
func.func @atomic_rmw(%I: memref<10xf32>, %val: f32, %i : index) {
  %x = memref.atomic_rmw addf %val, %I[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: memref.atomic_rmw addf [[VAL]], [[BUF]]{{\[}}[[I]]]
  return
}

// CHECK-LABEL: func @generic_atomic_rmw
// CHECK-SAME: ([[BUF:%.*]]: memref<1x2xf32>, [[I:%.*]]: index, [[J:%.*]]: index)
func.func @generic_atomic_rmw(%I: memref<1x2xf32>, %i : index, %j : index) {
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

// -----

func.func @extract_strided_metadata(%memref : memref<10x?xf32>)
  -> memref<?x?xf32, strided<[?, ?], offset: ?>> {

  %base, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %memref
    : memref<10x?xf32> -> memref<f32>, index, index, index, index, index

  %m2 = memref.reinterpret_cast %base to
      offset: [%offset],
      sizes: [%sizes#0, %sizes#1],
      strides: [%strides#0, %strides#1]
    : memref<f32> to memref<?x?xf32, strided<[?, ?], offset: ?>>

  return %m2: memref<?x?xf32, strided<[?, ?], offset: ?>>
}

// -----

// CHECK-LABEL: func @memref_realloc_ss
func.func @memref_realloc_ss(%src : memref<2xf32>) -> memref<4xf32>{
  %0 = memref.realloc %src : memref<2xf32> to memref<4xf32>
  return %0 : memref<4xf32>
}

// CHECK-LABEL: func @memref_realloc_sd
func.func @memref_realloc_sd(%src : memref<2xf32>, %d : index) -> memref<?xf32>{
  %0 = memref.realloc %src(%d) : memref<2xf32> to memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK-LABEL: func @memref_realloc_ds
func.func @memref_realloc_ds(%src : memref<?xf32>) -> memref<4xf32>{
  %0 = memref.realloc %src: memref<?xf32> to memref<4xf32>
  return %0 : memref<4xf32>
}

// CHECK-LABEL: func @memref_realloc_dd
func.func @memref_realloc_dd(%src : memref<?xf32>, %d: index)
  -> memref<?xf32>{
  %0 = memref.realloc %src(%d) : memref<?xf32> to memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK-LABEL: func @memref_extract_aligned_pointer
func.func @memref_extract_aligned_pointer(%src : memref<?xf32>) -> index {
  %0 = memref.extract_aligned_pointer_as_index %src : memref<?xf32> -> index
  return %0 : index
}
