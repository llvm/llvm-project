// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @subview_of_size_memcast
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: memref<4x6x16x32xi8>
//       CHECK:   %[[S:.+]] = memref.subview %[[ARG0]][0, 1, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<4x6x16x32xi8> to memref<16x32xi8, strided{{.*}}>
//       CHECK:   %[[M:.+]] = memref.cast %[[S]] : memref<16x32xi8, {{.*}}> to memref<16x32xi8, strided{{.*}}>
//       CHECK:   return %[[M]] : memref<16x32xi8, strided{{.*}}>
func.func @subview_of_size_memcast(%arg : memref<4x6x16x32xi8>) ->
  memref<16x32xi8, strided<[32, 1], offset: ?>>{
  %0 = memref.cast %arg : memref<4x6x16x32xi8> to memref<?x?x16x32xi8>
  %1 = memref.subview %0[0, 1, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] :
    memref<?x?x16x32xi8> to
    memref<16x32xi8, strided<[32, 1], offset: ?>>
  return %1 : memref<16x32xi8, strided<[32, 1], offset: ?>>
}

// -----

//       CHECK: func @subview_of_strides_memcast
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: memref<1x1x?xf32, strided{{.*}}>
//       CHECK:   %[[S:.+]] = memref.subview %[[ARG0]][0, 0, 0] [1, 1, 4]
//  CHECK-SAME:                    to memref<1x4xf32, strided<[7, 1], offset: ?>>
//       CHECK:   %[[M:.+]] = memref.cast %[[S]]
//  CHECK-SAME:                    to memref<1x4xf32, strided<[?, ?], offset: ?>>
//       CHECK:   return %[[M]]
func.func @subview_of_strides_memcast(%arg : memref<1x1x?xf32, strided<[35, 7, 1], offset: ?>>) -> memref<1x4xf32, strided<[?, ?], offset: ?>> {
  %0 = memref.cast %arg : memref<1x1x?xf32, strided<[35, 7, 1], offset: ?>> to memref<1x1x?xf32, strided<[?, ?, ?], offset: ?>>
  %1 = memref.subview %0[0, 0, 0] [1, 1, 4] [1, 1, 1] : memref<1x1x?xf32, strided<[?, ?, ?], offset: ?>> to memref<1x4xf32, strided<[?, ?], offset: ?>>
  return %1 : memref<1x4xf32, strided<[?, ?], offset: ?>>
}

// -----

// CHECK-LABEL: func @subview_of_static_full_size
// CHECK-SAME: %[[ARG0:.+]]: memref<4x6x16x32xi8>
// CHECK-NOT: memref.subview
// CHECK: return %[[ARG0]] : memref<4x6x16x32xi8>
func.func @subview_of_static_full_size(%arg0 : memref<4x6x16x32xi8>) -> memref<4x6x16x32xi8> {
  %0 = memref.subview %arg0[0, 0, 0, 0] [4, 6, 16, 32] [1, 1, 1, 1] : memref<4x6x16x32xi8> to memref<4x6x16x32xi8>
  return %0 : memref<4x6x16x32xi8>
}

// -----

func.func @subview_canonicalize(%arg0 : memref<?x?x?xf32>, %arg1 : index,
    %arg2 : index) -> memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = memref.subview %arg0[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : memref<?x?x?xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  return %0 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
}
// CHECK-LABEL: func @subview_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: memref<?x?x?xf32>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][0, %{{[a-zA-Z0-9_]+}}, 1]
//  CHECK-SAME:      [4, 1, %{{[a-zA-Z0-9_]+}}] [1, 1, 1]
//  CHECK-SAME:      : memref<?x?x?xf32> to memref<4x1x?xf32
//       CHECK:   %[[RESULT:.+]] = memref.cast %[[SUBVIEW]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @rank_reducing_subview_canonicalize(%arg0 : memref<?x?x?xf32>, %arg1 : index,
  %arg2 : index) -> memref<?x?xf32, strided<[?, 1], offset: ?>>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = memref.subview %arg0[%c0, %arg1, %c1] [%c4, 1, %arg2] [%c1, %c1, %c1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
  return %0 : memref<?x?xf32, strided<[?, 1], offset: ?>>
}
// CHECK-LABEL: func @rank_reducing_subview_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: memref<?x?x?xf32>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][0, %{{[a-zA-Z0-9_]+}}, 1]
//  CHECK-SAME:      [4, 1, %{{[a-zA-Z0-9_]+}}] [1, 1, 1]
//  CHECK-SAME:      : memref<?x?x?xf32> to memref<4x?xf32
//       CHECK:   %[[RESULT:.+]] = memref.cast %[[SUBVIEW]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @multiple_reducing_dims(%arg0 : memref<1x384x384xf32>,
    %arg1 : index, %arg2 : index, %arg3 : index) -> memref<?xf32, strided<[1], offset: ?>>
{
  %c1 = arith.constant 1 : index
  %0 = memref.subview %arg0[0, %arg1, %arg2] [1, %c1, %arg3] [1, 1, 1] : memref<1x384x384xf32> to memref<?x?xf32, strided<[384, 1], offset: ?>>
  %1 = memref.subview %0[0, 0] [1, %arg3] [1, 1] : memref<?x?xf32, strided<[384, 1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  return %1 : memref<?xf32, strided<[1], offset: ?>>
}
//       CHECK: func @multiple_reducing_dims
//       CHECK:   %[[REDUCED1:.+]] = memref.subview %{{.+}}[0, %{{.+}}, %{{.+}}] [1, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:       : memref<1x384x384xf32> to memref<1x?xf32, strided<[384, 1], offset: ?>>
//       CHECK:   %[[REDUCED2:.+]] = memref.subview %[[REDUCED1]][0, 0] [1, %{{.+}}] [1, 1]
//  CHECK-SAME:       : memref<1x?xf32, strided<[384, 1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>

// -----

func.func @multiple_reducing_dims_dynamic(%arg0 : memref<?x?x?xf32>,
    %arg1 : index, %arg2 : index, %arg3 : index) -> memref<?xf32, strided<[1], offset: ?>>
{
  %c1 = arith.constant 1 : index
  %0 = memref.subview %arg0[0, %arg1, %arg2] [1, %c1, %arg3] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
  %1 = memref.subview %0[0, 0] [1, %arg3] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  return %1 : memref<?xf32, strided<[1], offset: ?>>
}
//       CHECK: func @multiple_reducing_dims_dynamic
//       CHECK:   %[[REDUCED1:.+]] = memref.subview %{{.+}}[0, %{{.+}}, %{{.+}}] [1, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:       : memref<?x?x?xf32> to memref<1x?xf32, strided<[?, 1], offset: ?>>
//       CHECK:   %[[REDUCED2:.+]] = memref.subview %[[REDUCED1]][0, 0] [1, %{{.+}}] [1, 1]
//  CHECK-SAME:       : memref<1x?xf32, strided<[?, 1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>

// -----

func.func @multiple_reducing_dims_all_dynamic(%arg0 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>,
    %arg1 : index, %arg2 : index, %arg3 : index) -> memref<?xf32, strided<[?], offset: ?>>
{
  %c1 = arith.constant 1 : index
  %0 = memref.subview %arg0[0, %arg1, %arg2] [1, %c1, %arg3] [1, 1, 1]
      : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  %1 = memref.subview %0[0, 0] [1, %arg3] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
  return %1 : memref<?xf32, strided<[?], offset: ?>>
}
//       CHECK: func @multiple_reducing_dims_all_dynamic
//       CHECK:   %[[REDUCED1:.+]] = memref.subview %{{.+}}[0, %{{.+}}, %{{.+}}] [1, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:       : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<1x?xf32, strided<[?, ?], offset: ?>>
//       CHECK:   %[[REDUCED2:.+]] = memref.subview %[[REDUCED1]][0, 0] [1, %{{.+}}] [1, 1]
//  CHECK-SAME:       : memref<1x?xf32, strided<[?, ?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>

// -----

func.func @subview_negative_stride1(%arg0 : memref<?xf32>) -> memref<?xf32, strided<[?], offset: ?>>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant -1 : index
  %1 = memref.dim %arg0, %c0 : memref<?xf32>
  %2 = arith.addi %1, %c1 : index
  %3 = memref.subview %arg0[%2] [%1] [%c1] : memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
  return %3 : memref<?xf32, strided<[?], offset: ?>>
}
//       CHECK: func @subview_negative_stride1
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>)
//       CHECK:   %[[C1:.*]] = arith.constant 0
//       CHECK:   %[[C2:.*]] = arith.constant -1
//       CHECK:   %[[DIM1:.*]] = memref.dim %[[ARG0]], %[[C1]] : memref<?xf32>
//       CHECK:   %[[DIM2:.*]] = arith.addi %[[DIM1]], %[[C2]] : index
//       CHECK:   %[[RES1:.*]] = memref.subview %[[ARG0]][%[[DIM2]]] [%[[DIM1]]] [-1] : memref<?xf32> to memref<?xf32, strided<[-1], offset: ?>>
//       CHECK:   %[[RES2:.*]] = memref.cast %[[RES1]] : memref<?xf32, strided<[-1], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
//       CHECK:   return %[[RES2]] : memref<?xf32, strided<[?], offset: ?>>

// -----

func.func @subview_negative_stride2(%arg0 : memref<7xf32>) -> memref<?xf32, strided<[?], offset: ?>>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant -1 : index
  %1 = memref.dim %arg0, %c0 : memref<7xf32>
  %2 = arith.addi %1, %c1 : index
  %3 = memref.subview %arg0[%2] [%1] [%c1] : memref<7xf32> to memref<?xf32, strided<[?], offset: ?>>
  return %3 : memref<?xf32, strided<[?], offset: ?>>
}
//       CHECK: func @subview_negative_stride2
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<7xf32>)
//       CHECK:   %[[RES1:.*]] = memref.subview %[[ARG0]][6] [7] [-1] : memref<7xf32> to memref<7xf32, strided<[-1], offset: 6>>
//       CHECK:   %[[RES2:.*]] = memref.cast %[[RES1]] : memref<7xf32, strided<[-1], offset: 6>> to memref<?xf32, strided<[?], offset: ?>>
//       CHECK:   return %[[RES2]] : memref<?xf32, strided<[?], offset: ?>>

// -----

// CHECK-LABEL: func @dim_of_sized_view
//  CHECK-SAME:   %{{[a-z0-9A-Z_]+}}: memref<?xi8>
//  CHECK-SAME:   %[[SIZE:.[a-z0-9A-Z_]+]]: index
//       CHECK:   return %[[SIZE]] : index
func.func @dim_of_sized_view(%arg : memref<?xi8>, %size: index) -> index {
  %c0 = arith.constant 0 : index
  %0 = memref.reinterpret_cast %arg to offset: [0], sizes: [%size], strides: [1] : memref<?xi8> to memref<?xi8>
  %1 = memref.dim %0, %c0 : memref<?xi8>
  return %1 : index
}

// -----

// CHECK-LABEL: func @no_fold_subview_negative_size
//  CHECK:        %[[SUBVIEW:.+]] = memref.subview
//  CHECK:        return %[[SUBVIEW]]
func.func @no_fold_subview_negative_size(%input: memref<4x1024xf32>) -> memref<?x256xf32, strided<[1024, 1], offset: 2304>> {
  %cst = arith.constant -13 : index
  %0 = memref.subview %input[2, 256] [%cst, 256] [1, 1] : memref<4x1024xf32> to memref<?x256xf32, strided<[1024, 1], offset: 2304>>
  return %0 : memref<?x256xf32, strided<[1024, 1], offset: 2304>>
}

// -----

// CHECK-LABEL: func @no_fold_subview_zero_stride
//  CHECK:        %[[SUBVIEW:.+]] = memref.subview
//  CHECK:        return %[[SUBVIEW]]
func.func @no_fold_subview_zero_stride(%arg0 : memref<10xf32>) -> memref<1xf32, strided<[?], offset: 1>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %1 = memref.subview %arg0[1] [1] [%c0] : memref<10xf32> to memref<1xf32, strided<[?], offset: 1>>
  return %1 : memref<1xf32, strided<[?], offset: 1>>
}

// -----

// CHECK-LABEL: func @no_fold_of_store
//  CHECK:   %[[cst:.+]] = memref.cast %arg
//  CHECK:   memref.store %[[cst]]
func.func @no_fold_of_store(%arg : memref<32xi8>, %holder: memref<memref<?xi8>>) {
  %0 = memref.cast %arg : memref<32xi8> to memref<?xi8>
  memref.store %0, %holder[] : memref<memref<?xi8>>
  return
}

// -----

// Test case: Folding of memref.dim(memref.alloca(%size), %idx) -> %size
// CHECK-LABEL: func @dim_of_alloca(
//  CHECK-SAME:     %[[SIZE:[0-9a-z]+]]: index
//  CHECK-NEXT:   return %[[SIZE]] : index
func.func @dim_of_alloca(%size: index) -> index {
  %0 = memref.alloca(%size) : memref<?xindex>
  %c0 = arith.constant 0 : index
  %1 = memref.dim %0, %c0 : memref<?xindex>
  return %1 : index
}

// -----

// Test case: Folding of memref.dim(memref.alloca(rank(%v)), %idx) -> rank(%v)
// CHECK-LABEL: func @dim_of_alloca_with_dynamic_size(
//  CHECK-SAME:     %[[MEM:[0-9a-z]+]]: memref<*xf32>
//  CHECK-NEXT:   %[[RANK:.*]] = memref.rank %[[MEM]] : memref<*xf32>
//  CHECK-NEXT:   return %[[RANK]] : index
func.func @dim_of_alloca_with_dynamic_size(%arg0: memref<*xf32>) -> index {
  %0 = memref.rank %arg0 : memref<*xf32>
  %1 = memref.alloca(%0) : memref<?xindex>
  %c0 = arith.constant 0 : index
  %2 = memref.dim %1, %c0 : memref<?xindex>
  return %2 : index
}

// -----

// Test case: Folding of memref.dim(memref.reshape %v %shp, %idx) -> memref.load %shp[%idx]
// CHECK-LABEL: func @dim_of_memref_reshape(
//  CHECK-SAME:     %[[MEM:[0-9a-z]+]]: memref<*xf32>,
//  CHECK-SAME:     %[[SHP:[0-9a-z]+]]: memref<?xindex>
//  CHECK-NEXT:   %[[IDX:.*]] = arith.constant 3
//  CHECK-NEXT:   %[[DIM:.*]] = memref.load %[[SHP]][%[[IDX]]]
//  CHECK-NEXT:   memref.store
//   CHECK-NOT:   memref.dim
//       CHECK:   return %[[DIM]] : index
func.func @dim_of_memref_reshape(%arg0: memref<*xf32>, %arg1: memref<?xindex>)
    -> index {
  %c3 = arith.constant 3 : index
  %0 = memref.reshape %arg0(%arg1)
      : (memref<*xf32>, memref<?xindex>) -> memref<*xf32>
  // Update the shape to test that he load ends up in the right place.
  memref.store %c3, %arg1[%c3] : memref<?xindex>
  %1 = memref.dim %0, %c3 : memref<*xf32>
  return %1 : index
}

// -----

// Test case: Folding of memref.dim(memref.reshape %v %shp, %idx) -> memref.load %shp[%idx]
// CHECK-LABEL: func @dim_of_memref_reshape_i32(
//  CHECK-SAME:     %[[MEM:[0-9a-z]+]]: memref<*xf32>,
//  CHECK-SAME:     %[[SHP:[0-9a-z]+]]: memref<?xi32>
//  CHECK-NEXT:   %[[IDX:.*]] = arith.constant 3
//  CHECK-NEXT:   %[[DIM:.*]] = memref.load %[[SHP]][%[[IDX]]]
//  CHECK-NEXT:   %[[CAST:.*]] = arith.index_cast %[[DIM]]
//   CHECK-NOT:   memref.dim
//       CHECK:   return %[[CAST]] : index
func.func @dim_of_memref_reshape_i32(%arg0: memref<*xf32>, %arg1: memref<?xi32>)
    -> index {
  %c3 = arith.constant 3 : index
  %0 = memref.reshape %arg0(%arg1)
      : (memref<*xf32>, memref<?xi32>) -> memref<*xf32>
  %1 = memref.dim %0, %c3 : memref<*xf32>
  return %1 : index
}

// -----

// CHECK-LABEL: func @alloc_const_fold
func.func @alloc_const_fold() -> memref<?xf32> {
  // CHECK-NEXT: memref.alloc() : memref<4xf32>
  %c4 = arith.constant 4 : index
  %a = memref.alloc(%c4) : memref<?xf32>

  // CHECK-NEXT: memref.cast %{{.*}} : memref<4xf32> to memref<?xf32>
  // CHECK-NEXT: return %{{.*}} : memref<?xf32>
  return %a : memref<?xf32>
}

// -----

// CHECK-LABEL: func @alloc_alignment_const_fold
func.func @alloc_alignment_const_fold() -> memref<?xf32> {
  // CHECK-NEXT: memref.alloc() {alignment = 4096 : i64} : memref<4xf32>
  %c4 = arith.constant 4 : index
  %a = memref.alloc(%c4) {alignment = 4096 : i64} : memref<?xf32>

  // CHECK-NEXT: memref.cast %{{.*}} : memref<4xf32> to memref<?xf32>
  // CHECK-NEXT: return %{{.*}} : memref<?xf32>
  return %a : memref<?xf32>
}

// -----

// CHECK-LABEL: func @alloc_const_fold_with_symbols1(
//  CHECK: %[[c1:.+]] = arith.constant 1 : index
//  CHECK: %[[mem1:.+]] = memref.alloc({{.*}})[%[[c1]], %[[c1]]] : memref<?xi32, strided{{.*}}>
//  CHECK: return %[[mem1]] : memref<?xi32, strided{{.*}}>
func.func @alloc_const_fold_with_symbols1(%arg0 : index) -> memref<?xi32, strided<[?], offset: ?>> {
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%arg0)[%c1, %c1] : memref<?xi32, strided<[?], offset: ?>>
  return %0 : memref<?xi32, strided<[?], offset: ?>>
}

// -----

// CHECK-LABEL: func @alloc_const_fold_with_symbols2(
//  CHECK: %[[c1:.+]] = arith.constant 1 : index
//  CHECK: %[[mem1:.+]] = memref.alloc()[%[[c1]], %[[c1]]] : memref<1xi32, strided{{.*}}>
//  CHECK: %[[mem2:.+]] = memref.cast %[[mem1]] : memref<1xi32, strided{{.*}}> to memref<?xi32, strided{{.*}}>
//  CHECK: return %[[mem2]] : memref<?xi32, strided{{.*}}>
func.func @alloc_const_fold_with_symbols2() -> memref<?xi32, strided<[?], offset: ?>> {
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%c1)[%c1, %c1] : memref<?xi32, strided<[?], offset: ?>>
  return %0 : memref<?xi32, strided<[?], offset: ?>>
}

// -----
// CHECK-LABEL: func @allocator
// CHECK:   %[[alloc:.+]] = memref.alloc
// CHECK:   memref.store %[[alloc:.+]], %arg0
func.func @allocator(%arg0 : memref<memref<?xi32>>, %arg1 : index)  {
  %0 = memref.alloc(%arg1) : memref<?xi32>
  memref.store %0, %arg0[] : memref<memref<?xi32>>
  return
}

// -----

func.func @compose_collapse_of_collapse_zero_dim(%arg0 : memref<1x1x1xf32>)
    -> memref<f32> {
  %0 = memref.collapse_shape %arg0 [[0, 1, 2]]
      : memref<1x1x1xf32> into memref<1xf32>
  %1 = memref.collapse_shape %0 [] : memref<1xf32> into memref<f32>
  return %1 : memref<f32>
}
// CHECK-LABEL: func @compose_collapse_of_collapse_zero_dim
//       CHECK:   memref.collapse_shape %{{.*}} []
//  CHECK-SAME:     memref<1x1x1xf32> into memref<f32>

// -----

func.func @compose_collapse_of_collapse(%arg0 : memref<?x?x?x?x?xf32>)
    -> memref<?x?xf32> {
  %0 = memref.collapse_shape %arg0 [[0, 1], [2], [3, 4]]
      : memref<?x?x?x?x?xf32> into memref<?x?x?xf32>
  %1 = memref.collapse_shape %0 [[0, 1], [2]]
      : memref<?x?x?xf32> into memref<?x?xf32>
  return %1 : memref<?x?xf32>
}
// CHECK-LABEL: func @compose_collapse_of_collapse
//       CHECK:   memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2], [3, 4]]
//   CHECK-NOT:   memref.collapse_shape

// -----

func.func @do_not_compose_collapse_of_expand_non_identity_layout(
    %arg0: memref<?x?xf32, strided<[?, 1], offset: 0>>)
    -> memref<?xf32, strided<[?], offset: 0>> {
  %1 = memref.expand_shape %arg0 [[0, 1], [2]] :
    memref<?x?xf32, strided<[?, 1], offset: 0>> into
    memref<?x4x?xf32, strided<[?, ?, 1], offset: 0>>
  %2 = memref.collapse_shape %1 [[0, 1, 2]] :
    memref<?x4x?xf32, strided<[?, ?, 1], offset: 0>> into
    memref<?xf32, strided<[?], offset: 0>>
  return %2 : memref<?xf32, strided<[?], offset: 0>>
}
// CHECK-LABEL: func @do_not_compose_collapse_of_expand_non_identity_layout
// CHECK: expand
// CHECK: collapse

// -----

func.func @compose_expand_of_expand(%arg0 : memref<?x?xf32>)
    -> memref<?x6x4x5x?xf32> {
  %0 = memref.expand_shape %arg0 [[0, 1], [2]]
      : memref<?x?xf32> into memref<?x4x?xf32>
  %1 = memref.expand_shape %0 [[0, 1], [2], [3, 4]]
      : memref<?x4x?xf32> into memref<?x6x4x5x?xf32>
  return %1 : memref<?x6x4x5x?xf32>
}
// CHECK-LABEL: func @compose_expand_of_expand
//       CHECK:   memref.expand_shape %{{.*}} {{\[}}[0, 1, 2], [3, 4]]
//   CHECK-NOT:   memref.expand_shape

// -----

func.func @compose_expand_of_expand_of_zero_dim(%arg0 : memref<f32>)
    -> memref<1x1x1xf32> {
  %0 = memref.expand_shape %arg0 [] : memref<f32> into memref<1xf32>
  %1 = memref.expand_shape %0 [[0, 1, 2]]
      : memref<1xf32> into memref<1x1x1xf32>
  return %1 : memref<1x1x1xf32>
}
// CHECK-LABEL: func @compose_expand_of_expand_of_zero_dim
//       CHECK:   memref.expand_shape %{{.*}} []
//  CHECK-SAME:     memref<f32> into memref<1x1x1xf32>

// -----

func.func @fold_collapse_of_expand(%arg0 : memref<12x4xf32>) -> memref<12x4xf32> {
  %0 = memref.expand_shape %arg0 [[0, 1], [2]]
      : memref<12x4xf32> into memref<3x4x4xf32>
  %1 = memref.collapse_shape %0 [[0, 1], [2]]
      : memref<3x4x4xf32> into memref<12x4xf32>
  return %1 : memref<12x4xf32>
}
// CHECK-LABEL: func @fold_collapse_of_expand
//   CHECK-NOT:   linalg.{{.*}}_shape

// -----

func.func @fold_collapse_collapse_of_expand(%arg0 : memref<?x?xf32>)
    -> memref<?x?xf32> {
  %0 = memref.expand_shape %arg0 [[0, 1], [2]]
      : memref<?x?xf32> into memref<?x4x?xf32>
  %1 = memref.collapse_shape %0 [[0, 1], [2]]
      : memref<?x4x?xf32> into memref<?x?xf32>
  return %1 : memref<?x?xf32>
}
// CHECK-LABEL: @fold_collapse_collapse_of_expand
//   CHECK-NOT:   linalg.{{.*}}_shape

// -----

func.func @fold_memref_expand_cast(%arg0 : memref<?x?xf32>) -> memref<2x4x4xf32> {
  %0 = memref.cast %arg0 : memref<?x?xf32> to memref<8x4xf32>
  %1 = memref.expand_shape %0 [[0, 1], [2]]
      : memref<8x4xf32> into memref<2x4x4xf32>
  return %1 : memref<2x4x4xf32>
}

// CHECK-LABEL: @fold_memref_expand_cast
// CHECK: memref.expand_shape

// -----

// CHECK-LABEL:   func @collapse_after_memref_cast_type_change(
// CHECK-SAME:      %[[INPUT:.*]]: memref<?x512x1x1xf32>) -> memref<?x?xf32> {
// CHECK:           %[[COLLAPSED:.*]] = memref.collapse_shape %[[INPUT]]
// CHECK-SAME:         {{\[\[}}0], [1, 2, 3]] : memref<?x512x1x1xf32> into memref<?x512xf32>
// CHECK:           %[[DYNAMIC:.*]] = memref.cast %[[COLLAPSED]] :
// CHECK-SAME:         memref<?x512xf32> to memref<?x?xf32>
// CHECK:           return %[[DYNAMIC]] : memref<?x?xf32>
// CHECK:         }
func.func @collapse_after_memref_cast_type_change(%arg0 : memref<?x512x1x1xf32>) -> memref<?x?xf32> {
  %dynamic = memref.cast %arg0: memref<?x512x1x1xf32> to memref<?x?x?x?xf32>
  %collapsed = memref.collapse_shape %dynamic [[0], [1, 2, 3]] : memref<?x?x?x?xf32> into memref<?x?xf32>
  return %collapsed : memref<?x?xf32>
}

// -----

// CHECK-LABEL:   func @collapse_after_memref_cast(
// CHECK-SAME:      %[[INPUT:.*]]: memref<?x512x1x?xf32>) -> memref<?x?xf32> {
// CHECK:           %[[COLLAPSED:.*]] = memref.collapse_shape %[[INPUT]]
// CHECK-SAME:        {{\[\[}}0], [1, 2, 3]] : memref<?x512x1x?xf32> into memref<?x?xf32>
// CHECK:           return %[[COLLAPSED]] : memref<?x?xf32>
func.func @collapse_after_memref_cast(%arg0 : memref<?x512x1x?xf32>) -> memref<?x?xf32> {
  %dynamic = memref.cast %arg0: memref<?x512x1x?xf32> to memref<?x?x?x?xf32>
  %collapsed = memref.collapse_shape %dynamic [[0], [1, 2, 3]] : memref<?x?x?x?xf32> into memref<?x?xf32>
  return %collapsed : memref<?x?xf32>
}

// -----

// CHECK-LABEL:   func @collapse_after_memref_cast_type_change_dynamic(
// CHECK-SAME:      %[[INPUT:.*]]: memref<1x1x1x?xi64>) -> memref<?x?xi64> {
// CHECK:           %[[COLLAPSED:.*]] = memref.collapse_shape %[[INPUT]]
// CHECK-SAME:        {{\[\[}}0, 1, 2], [3]] : memref<1x1x1x?xi64> into memref<1x?xi64>
// CHECK:           %[[DYNAMIC:.*]] = memref.cast %[[COLLAPSED]] :
// CHECK-SAME:         memref<1x?xi64> to memref<?x?xi64>
// CHECK:           return %[[DYNAMIC]] : memref<?x?xi64>
func.func @collapse_after_memref_cast_type_change_dynamic(%arg0: memref<1x1x1x?xi64>) -> memref<?x?xi64> {
  %casted = memref.cast %arg0 : memref<1x1x1x?xi64> to memref<1x1x?x?xi64>
  %collapsed = memref.collapse_shape %casted [[0, 1, 2], [3]] : memref<1x1x?x?xi64> into memref<?x?xi64>
  return %collapsed : memref<?x?xi64>
}

// -----

func.func @reduced_memref(%arg0: memref<2x5x7x1xf32>, %arg1 :index)
    -> memref<1x4x1xf32, strided<[35, 7, 1], offset: ?>> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %0 = memref.subview %arg0[%arg1, %arg1, %arg1, 0] [%c1, %c4, %c1, 1] [1, 1, 1, 1]
      : memref<2x5x7x1xf32> to memref<?x?x?xf32, strided<[35, 7, 1], offset: ?>>
  %1 = memref.cast %0
      : memref<?x?x?xf32, strided<[35, 7, 1], offset: ?>> to
        memref<1x4x1xf32, strided<[35, 7, 1], offset: ?>>
  return %1 : memref<1x4x1xf32, strided<[35, 7, 1], offset: ?>>
}

// CHECK-LABEL: func @reduced_memref
//       CHECK:   %[[RESULT:.+]] = memref.subview
//  CHECK-SAME:       memref<2x5x7x1xf32> to memref<1x4x1xf32, strided{{.+}}>
//       CHECK:   return %[[RESULT]]

// -----

// CHECK-LABEL: func @fold_rank_memref
func.func @fold_rank_memref(%arg0 : memref<?x?xf32>) -> (index) {
  // Fold a rank into a constant
  // CHECK-NEXT: [[C2:%.+]] = arith.constant 2 : index
  %rank_0 = memref.rank %arg0 : memref<?x?xf32>

  // CHECK-NEXT: return [[C2]]
  return %rank_0 : index
}

// -----

func.func @fold_no_op_subview(%arg0 : memref<20x42xf32>) -> memref<20x42xf32, strided<[42, 1]>> {
  %0 = memref.subview %arg0[0, 0] [20, 42] [1, 1] : memref<20x42xf32> to memref<20x42xf32, strided<[42, 1]>>
  return %0 : memref<20x42xf32, strided<[42, 1]>>
}
// CHECK-LABEL: func @fold_no_op_subview(
//       CHECK:   %[[ARG0:.+]]: memref<20x42xf32>)
//       CHECK:   %[[CAST:.+]] = memref.cast %[[ARG0]]
//       CHECK:   return %[[CAST]]

// -----

func.func @no_fold_subview_with_non_zero_offset(%arg0 : memref<20x42xf32>) -> memref<20x42xf32, strided<[42, 1], offset: 1>> {
  %0 = memref.subview %arg0[0, 1] [20, 42] [1, 1] : memref<20x42xf32> to memref<20x42xf32, strided<[42, 1], offset: 1>>
  return %0 : memref<20x42xf32, strided<[42, 1], offset: 1>>
}
// CHECK-LABEL: func @no_fold_subview_with_non_zero_offset(
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview
//       CHECK:    return %[[SUBVIEW]]

// -----

func.func @no_fold_subview_with_non_unit_stride(%arg0 : memref<20x42xf32>) -> memref<20x42xf32, strided<[42, 2]>> {
  %0 = memref.subview %arg0[0, 0] [20, 42] [1, 2] : memref<20x42xf32> to memref<20x42xf32, strided<[42, 2]>>
  return %0 : memref<20x42xf32, strided<[42, 2]>>
}
// CHECK-LABEL: func @no_fold_subview_with_non_unit_stride(
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview
//       CHECK:    return %[[SUBVIEW]]

// -----

func.func @no_fold_dynamic_no_op_subview(%arg0 : memref<?x?xf32>) -> memref<?x?xf32, strided<[?, 1], offset: ?>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %2 = memref.subview %arg0[0, 0] [%0, %1] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
  return %2 : memref<?x?xf32, strided<[?, 1], offset: ?>>
}
// CHECK-LABEL: func @no_fold_dynamic_no_op_subview(
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview
//       CHECK:    return %[[SUBVIEW]]

// -----

func.func @atomicrmw_cast_fold(%arg0 : f32, %arg1 : memref<4xf32>, %c : index) {
  %v = memref.cast %arg1 : memref<4xf32> to memref<?xf32>
  %a = memref.atomic_rmw addf %arg0, %v[%c] : (f32, memref<?xf32>) -> f32
  return
}

// CHECK-LABEL: func @atomicrmw_cast_fold
// CHECK-NEXT: memref.atomic_rmw addf %arg0, %arg1[%arg2] : (f32, memref<4xf32>) -> f32

// -----

func.func @copy_of_cast(%m1: memref<?xf32>, %m2: memref<*xf32>) {
  %casted1 = memref.cast %m1 : memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
  %casted2 = memref.cast %m2 : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
  memref.copy %casted1, %casted2 : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
  return
}

// CHECK-LABEL: func @copy_of_cast(
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32>, %[[m2:.*]]: memref<*xf32>
//       CHECK:   %[[casted2:.*]] = memref.cast %[[m2]]
//       CHECK:   memref.copy %[[m1]], %[[casted2]]

// -----

func.func @self_copy(%m1: memref<?xf32>) {
  memref.copy %m1, %m1 : memref<?xf32> to memref<?xf32>
  return
}

// CHECK-LABEL: func @self_copy
//  CHECK-NEXT:   return

// -----

func.func @scopeMerge() {
  memref.alloca_scope {
    %cnt = "test.count"() : () -> index
    %a = memref.alloca(%cnt) : memref<?xi64>
    "test.use"(%a) : (memref<?xi64>) -> ()
  }
  return
}
// CHECK:   func @scopeMerge() {
// CHECK-NOT: alloca_scope
// CHECK:     %[[cnt:.+]] = "test.count"() : () -> index
// CHECK:     %[[alloc:.+]] = memref.alloca(%[[cnt]]) : memref<?xi64>
// CHECK:     "test.use"(%[[alloc]]) : (memref<?xi64>) -> ()
// CHECK:     return

func.func @scopeMerge2() {
  "test.region"() ({
    memref.alloca_scope {
      %cnt = "test.count"() : () -> index
      %a = memref.alloca(%cnt) : memref<?xi64>
      "test.use"(%a) : (memref<?xi64>) -> ()
    }
    "test.terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK:   func @scopeMerge2() {
// CHECK:     "test.region"() ({
// CHECK:       memref.alloca_scope {
// CHECK:         %[[cnt:.+]] = "test.count"() : () -> index
// CHECK:         %[[alloc:.+]] = memref.alloca(%[[cnt]]) : memref<?xi64>
// CHECK:         "test.use"(%[[alloc]]) : (memref<?xi64>) -> ()
// CHECK:       }
// CHECK:       "test.terminator"() : () -> ()
// CHECK:     }) : () -> ()
// CHECK:     return
// CHECK:   }

func.func @scopeMerge3() {
  %cnt = "test.count"() : () -> index
  "test.region"() ({
    memref.alloca_scope {
      %a = memref.alloca(%cnt) : memref<?xi64>
      "test.use"(%a) : (memref<?xi64>) -> ()
    }
    "test.terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK:   func @scopeMerge3() {
// CHECK:     %[[cnt:.+]] = "test.count"() : () -> index
// CHECK:     %[[alloc:.+]] = memref.alloca(%[[cnt]]) : memref<?xi64>
// CHECK:     "test.region"() ({
// CHECK:       memref.alloca_scope {
// CHECK:         "test.use"(%[[alloc]]) : (memref<?xi64>) -> ()
// CHECK:       }
// CHECK:       "test.terminator"() : () -> ()
// CHECK:     }) : () -> ()
// CHECK:     return
// CHECK:   }

func.func @scopeMerge4() {
  %cnt = "test.count"() : () -> index
  "test.region"() ({
    memref.alloca_scope {
      %a = memref.alloca(%cnt) : memref<?xi64>
      "test.use"(%a) : (memref<?xi64>) -> ()
    }
    "test.op"() : () -> ()
    "test.terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK:   func @scopeMerge4() {
// CHECK:     %[[cnt:.+]] = "test.count"() : () -> index
// CHECK:     "test.region"() ({
// CHECK:       memref.alloca_scope {
// CHECK:         %[[alloc:.+]] = memref.alloca(%[[cnt]]) : memref<?xi64>
// CHECK:         "test.use"(%[[alloc]]) : (memref<?xi64>) -> ()
// CHECK:       }
// CHECK:       "test.op"() : () -> ()
// CHECK:       "test.terminator"() : () -> ()
// CHECK:     }) : () -> ()
// CHECK:     return
// CHECK:   }

func.func @scopeMerge5() {
  "test.region"() ({
    memref.alloca_scope {
      affine.parallel (%arg) = (0) to (64) {
        %a = memref.alloca(%arg) : memref<?xi64>
        "test.use"(%a) : (memref<?xi64>) -> ()
      }
    }
    "test.op"() : () -> ()
    "test.terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK:   func @scopeMerge5() {
// CHECK:     "test.region"() ({
// CHECK:       affine.parallel (%[[cnt:.+]]) = (0) to (64) {
// CHECK:         %[[alloc:.+]] = memref.alloca(%[[cnt]]) : memref<?xi64>
// CHECK:         "test.use"(%[[alloc]]) : (memref<?xi64>) -> ()
// CHECK:       }
// CHECK:       "test.op"() : () -> ()
// CHECK:       "test.terminator"() : () -> ()
// CHECK:     }) : () -> ()
// CHECK:     return
// CHECK:   }

func.func @scopeInline(%arg : memref<index>) {
  %cnt = "test.count"() : () -> index
  "test.region"() ({
    memref.alloca_scope {
      memref.store %cnt, %arg[] : memref<index>
    }
    "test.terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK:   func @scopeInline
// CHECK-NOT:  memref.alloca_scope

// -----

// CHECK-LABEL: func @reinterpret_noop
//  CHECK-SAME: (%[[ARG:.*]]: memref<2x3x4xf32>)
//  CHECK-NEXT: return %[[ARG]]
func.func @reinterpret_noop(%arg : memref<2x3x4xf32>) -> memref<2x3x4xf32> {
  %0 = memref.reinterpret_cast %arg to offset: [0], sizes: [2, 3, 4], strides: [12, 4, 1] : memref<2x3x4xf32> to memref<2x3x4xf32>
  return %0 : memref<2x3x4xf32>
}

// -----

// CHECK-LABEL: func @reinterpret_of_reinterpret
//  CHECK-SAME: (%[[ARG:.*]]: memref<?xi8>, %[[SIZE1:.*]]: index, %[[SIZE2:.*]]: index)
//       CHECK: %[[RES:.*]] = memref.reinterpret_cast %[[ARG]] to offset: [0], sizes: [%[[SIZE2]]], strides: [1]
//       CHECK: return %[[RES]]
func.func @reinterpret_of_reinterpret(%arg : memref<?xi8>, %size1: index, %size2: index) -> memref<?xi8> {
  %0 = memref.reinterpret_cast %arg to offset: [0], sizes: [%size1], strides: [1] : memref<?xi8> to memref<?xi8>
  %1 = memref.reinterpret_cast %0 to offset: [0], sizes: [%size2], strides: [1] : memref<?xi8> to memref<?xi8>
  return %1 : memref<?xi8>
}

// -----

// CHECK-LABEL: func @reinterpret_of_cast
//  CHECK-SAME: (%[[ARG:.*]]: memref<?xi8>, %[[SIZE:.*]]: index)
//       CHECK: %[[RES:.*]] = memref.reinterpret_cast %[[ARG]] to offset: [0], sizes: [%[[SIZE]]], strides: [1]
//       CHECK: return %[[RES]]
func.func @reinterpret_of_cast(%arg : memref<?xi8>, %size: index) -> memref<?xi8> {
  %0 = memref.cast %arg : memref<?xi8> to memref<5xi8>
  %1 = memref.reinterpret_cast %0 to offset: [0], sizes: [%size], strides: [1] : memref<5xi8> to memref<?xi8>
  return %1 : memref<?xi8>
}

// -----

// CHECK-LABEL: func @reinterpret_of_subview
//  CHECK-SAME: (%[[ARG:.*]]: memref<?xi8>, %[[SIZE1:.*]]: index, %[[SIZE2:.*]]: index)
//       CHECK: %[[RES:.*]] = memref.reinterpret_cast %[[ARG]] to offset: [0], sizes: [%[[SIZE2]]], strides: [1]
//       CHECK: return %[[RES]]
func.func @reinterpret_of_subview(%arg : memref<?xi8>, %size1: index, %size2: index) -> memref<?xi8> {
  %0 = memref.subview %arg[0] [%size1] [1] : memref<?xi8> to memref<?xi8>
  %1 = memref.reinterpret_cast %0 to offset: [0], sizes: [%size2], strides: [1] : memref<?xi8> to memref<?xi8>
  return %1 : memref<?xi8>
}

// -----

// Check that a reinterpret cast of an equivalent extract strided metadata
// is canonicalized to a plain cast when the destination type is different
// than the type of the original memref.
// CHECK-LABEL: func @reinterpret_of_extract_strided_metadata_w_type_mistach
//  CHECK-SAME: (%[[ARG:.*]]: memref<8x2xf32>)
//       CHECK: %[[CAST:.*]] = memref.cast %[[ARG]] : memref<8x2xf32> to memref<?x?xf32,
//       CHECK: return %[[CAST]]
func.func @reinterpret_of_extract_strided_metadata_w_type_mistach(%arg0 : memref<8x2xf32>) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  %base, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<8x2xf32> -> memref<f32>, index, index, index, index, index
  %m2 = memref.reinterpret_cast %base to offset: [%offset], sizes: [%sizes#0, %sizes#1], strides: [%strides#0, %strides#1] : memref<f32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %m2 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}

// -----

// Similar to reinterpret_of_extract_strided_metadata_w_type_mistach except that
// we check that the match happen when the static information has been folded.
// E.g., in this case, we know that size of dim 0 is 8 and size of dim 1 is 2.
// So even if we don't use the values sizes#0, sizes#1, as long as they have the
// same constant value, the match is valid.
// CHECK-LABEL: func @reinterpret_of_extract_strided_metadata_w_constants
//  CHECK-SAME: (%[[ARG:.*]]: memref<8x2xf32>)
//       CHECK: %[[CAST:.*]] = memref.cast %[[ARG]] : memref<8x2xf32> to memref<?x?xf32,
//       CHECK: return %[[CAST]]
func.func @reinterpret_of_extract_strided_metadata_w_constants(%arg0 : memref<8x2xf32>) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  %base, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<8x2xf32> -> memref<f32>, index, index, index, index, index
  %c8 = arith.constant 8: index
  %m2 = memref.reinterpret_cast %base to offset: [0], sizes: [%c8, 2], strides: [2, %strides#1] : memref<f32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %m2 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}
// -----

// Check that a reinterpret cast of an equivalent extract strided metadata
// is completely removed when the original memref has the same type.
// CHECK-LABEL: func @reinterpret_of_extract_strided_metadata_same_type
//  CHECK-SAME: (%[[ARG:.*]]: memref<?x?xf32
//       CHECK: return %[[ARG]]
func.func @reinterpret_of_extract_strided_metadata_same_type(%arg0 : memref<?x?xf32, strided<[?,?], offset: ?>>) -> memref<?x?xf32, strided<[?,?], offset: ?>> {
  %base, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<?x?xf32, strided<[?,?], offset: ?>> -> memref<f32>, index, index, index, index, index
  %m2 = memref.reinterpret_cast %base to offset: [%offset], sizes: [%sizes#0, %sizes#1], strides: [%strides#0, %strides#1] : memref<f32> to memref<?x?xf32, strided<[?,?], offset:?>>
  return %m2 : memref<?x?xf32, strided<[?,?], offset:?>>
}

// -----

// Check that we don't simplify reinterpret cast of extract strided metadata
// when the strides don't match.
// CHECK-LABEL: func @reinterpret_of_extract_strided_metadata_w_different_stride
//  CHECK-SAME: (%[[ARG:.*]]: memref<8x2xf32>)
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG]]
//       CHECK: %[[RES:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[C0]]], sizes: [4, 2, 2], strides: [1, 1, %[[C1]]]
//       CHECK: return %[[RES]]
func.func @reinterpret_of_extract_strided_metadata_w_different_stride(%arg0 : memref<8x2xf32>) -> memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> {
  %base, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<8x2xf32> -> memref<f32>, index, index, index, index, index
  %m2 = memref.reinterpret_cast %base to offset: [%offset], sizes: [4, 2, 2], strides: [1, 1, %strides#1] : memref<f32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  return %m2 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
}
// -----

// Check that we don't simplify reinterpret cast of extract strided metadata
// when the offset doesn't match.
// CHECK-LABEL: func @reinterpret_of_extract_strided_metadata_w_different_offset
//  CHECK-SAME: (%[[ARG:.*]]: memref<8x2xf32>)
//   CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG]]
//       CHECK: %[[RES:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [1], sizes: [%[[C8]], %[[C2]]], strides: [%[[C2]], %[[C1]]]
//       CHECK: return %[[RES]]
func.func @reinterpret_of_extract_strided_metadata_w_different_offset(%arg0 : memref<8x2xf32>) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  %base, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<8x2xf32> -> memref<f32>, index, index, index, index, index
  %m2 = memref.reinterpret_cast %base to offset: [1], sizes: [%sizes#0, %sizes#1], strides: [%strides#0, %strides#1] : memref<f32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %m2 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}

// -----

func.func @canonicalize_rank_reduced_subview(%arg0 : memref<8x?xf32>,
    %arg1 : index) -> memref<?xf32, strided<[?], offset: ?>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.subview %arg0[%c0, %c0] [1, %arg1] [%c1, %c1] : memref<8x?xf32> to memref<?xf32, strided<[?], offset: ?>>
  return %0 :  memref<?xf32, strided<[?], offset: ?>>
}
//      CHECK: func @canonicalize_rank_reduced_subview
// CHECK-SAME:     %[[ARG0:.+]]: memref<8x?xf32>
// CHECK-SAME:     %[[ARG1:.+]]: index
//      CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][0, 0] [1, %[[ARG1]]] [1, 1]
// CHECK-SAME:       memref<8x?xf32> to memref<?xf32, strided<[1], offset: ?>>

// -----

// CHECK-LABEL: func @memref_realloc_dead
// CHECK-SAME: %[[SRC:[0-9a-z]+]]: memref<2xf32>
// CHECK-NOT: memref.realloc
// CHECK: return %[[SRC]]
func.func @memref_realloc_dead(%src : memref<2xf32>, %v : f32) -> memref<2xf32>{
  %0 = memref.realloc %src : memref<2xf32> to memref<4xf32>
  %i2 = arith.constant 2 : index
  memref.store %v, %0[%i2] : memref<4xf32>
  return %src : memref<2xf32>
}

// -----

// CHECK-LABEL: func @collapse_expand_fold_to_cast(
//  CHECK-SAME:     %[[m:.*]]: memref<?xf32, strided<[1]>, 3>
//       CHECK:   %[[casted:.*]] = memref.cast %[[m]] : memref<?xf32, strided<[1]>, 3> to memref<?xf32, 3
//       CHECK:   return %[[casted]]
func.func @collapse_expand_fold_to_cast(%m: memref<?xf32, strided<[1]>, 3>)
    -> (memref<?xf32, 3>)
{
  %0 = memref.expand_shape %m [[0, 1]]
      : memref<?xf32, strided<[1]>, 3> into memref<1x?xf32, 3>
  %1 = memref.collapse_shape %0 [[0, 1]]
      : memref<1x?xf32, 3> into memref<?xf32, 3>
  return %1 : memref<?xf32, 3>
}

// -----

// CHECK-LABEL: func @fold_trivial_subviews(
//  CHECK-SAME:     %[[m:.*]]: memref<?xf32, strided<[?], offset: ?>>
//       CHECK:   %[[subview:.*]] = memref.subview %[[m]][5]
//       CHECK:   return %[[subview]]
func.func @fold_trivial_subviews(%m: memref<?xf32, strided<[?], offset: ?>>,
                                 %sz: index)
    -> memref<?xf32, strided<[?], offset: ?>>
{
  %0 = memref.subview %m[5] [%sz] [1]
      : memref<?xf32, strided<[?], offset: ?>>
        to memref<?xf32, strided<[?], offset: ?>>
  %1 = memref.subview %0[0] [%sz] [1]
      : memref<?xf32, strided<[?], offset: ?>>
        to memref<?xf32, strided<[?], offset: ?>>
  return %1 : memref<?xf32, strided<[?], offset: ?>>
}

// -----

// CHECK-LABEL: func @load_store_nontemporal(
func.func @load_store_nontemporal(%input : memref<32xf32, affine_map<(d0) -> (d0)>>, %output : memref<32xf32, affine_map<(d0) -> (d0)>>) {
  %1 = arith.constant 7 : index
  // CHECK: memref.load %{{.*}}[%{{.*}}] {nontemporal = true} : memref<32xf32>
  %2 = memref.load %input[%1] {nontemporal = true} : memref<32xf32, affine_map<(d0) -> (d0)>>
  // CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}] {nontemporal = true} : memref<32xf32>
  memref.store %2, %output[%1] {nontemporal = true} : memref<32xf32, affine_map<(d0) -> (d0)>>
  func.return
}

// -----

// CHECK-LABEL: func @fold_trivial_memory_space_cast(
//  CHECK-SAME:     %[[arg:.*]]: memref<?xf32>
//       CHECK:   return %[[arg]]
func.func @fold_trivial_memory_space_cast(%arg : memref<?xf32>) -> memref<?xf32> {
  %0 = memref.memory_space_cast %arg : memref<?xf32> to memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

// CHECK-LABEL: func @fold_multiple_memory_space_cast(
//  CHECK-SAME:     %[[arg:.*]]: memref<?xf32>
//       CHECK:   %[[res:.*]] = memref.memory_space_cast %[[arg]] : memref<?xf32> to memref<?xf32, 2>
//       CHECK:   return %[[res]]
func.func @fold_multiple_memory_space_cast(%arg : memref<?xf32>) -> memref<?xf32, 2> {
  %0 = memref.memory_space_cast %arg : memref<?xf32> to memref<?xf32, 1>
  %1 = memref.memory_space_cast %0 : memref<?xf32, 1> to memref<?xf32, 2>
  return %1 : memref<?xf32, 2>
}

// -----

// CHECK-LABEL: func private @ub_negative_alloc_size
func.func private @ub_negative_alloc_size() -> memref<?x?x?xi1> {
  %idx1 = index.constant 1
  %c-2 = arith.constant -2 : index
  %c15 = arith.constant 15 : index
// CHECK:   %[[ALLOC:.*]] = memref.alloc(%c-2) : memref<15x?x1xi1>
  %alloc = memref.alloc(%c15, %c-2, %idx1) : memref<?x?x?xi1>
  return %alloc : memref<?x?x?xi1>
}

// -----

// CHECK-LABEL: func @subview_rank_reduction(
//  CHECK-SAME:     %[[arg0:.*]]: memref<1x384x384xf32>, %[[arg1:.*]]: index
func.func @subview_rank_reduction(%arg0: memref<1x384x384xf32>, %idx: index)
    -> memref<?x?xf32, strided<[384, 1], offset: ?>> {
  %c1 = arith.constant 1 : index
  // CHECK: %[[subview:.*]] = memref.subview %[[arg0]][0, %[[arg1]], %[[arg1]]] [1, 1, %[[arg1]]] [1, 1, 1] : memref<1x384x384xf32> to memref<1x?xf32, strided<[384, 1], offset: ?>>
  // CHECK: %[[cast:.*]] = memref.cast %[[subview]] : memref<1x?xf32, strided<[384, 1], offset: ?>> to memref<?x?xf32, strided<[384, 1], offset: ?>>
  %0 = memref.subview %arg0[0, %idx, %idx] [1, %c1, %idx] [1, 1, 1]
      : memref<1x384x384xf32> to memref<?x?xf32, strided<[384, 1], offset: ?>>
  // CHECK: return %[[cast]]
  return %0 : memref<?x?xf32, strided<[384, 1], offset: ?>>
}

// -----

// CHECK-LABEL: func @fold_double_transpose(
//  CHECK-SAME:     %[[arg0:.*]]: memref<1x2x3x4x5xf32>
func.func @fold_double_transpose(%arg0: memref<1x2x3x4x5xf32>) -> memref<5x3x2x4x1xf32, strided<[1, 20, 60, 5, 120]>> {
  // CHECK: %[[ONETRANSPOSE:.+]] = memref.transpose %[[arg0]] (d0, d1, d2, d3, d4) -> (d4, d2, d1, d3, d0)
  %0 = memref.transpose %arg0 (d0, d1, d2, d3, d4) -> (d1, d0, d4, d3, d2) : memref<1x2x3x4x5xf32> to memref<2x1x5x4x3xf32, strided<[60, 120, 1, 5, 20]>>
  %1 = memref.transpose %0 (d1, d0, d4, d3, d2) -> (d4, d2, d1, d3, d0) : memref<2x1x5x4x3xf32, strided<[60, 120, 1, 5, 20]>> to memref<5x3x2x4x1xf32, strided<[1, 20, 60, 5, 120]>>
  // CHECK: return %[[ONETRANSPOSE]]
  return %1 : memref<5x3x2x4x1xf32, strided<[1, 20, 60, 5, 120]>>
}

// -----

// CHECK-LABEL: func @fold_double_transpose2(
//  CHECK-SAME:     %[[arg0:.*]]: memref<1x2x3x4x5xf32>
func.func @fold_double_transpose2(%arg0: memref<1x2x3x4x5xf32>) -> memref<5x3x2x4x1xf32, strided<[1, 20, 60, 5, 120]>> {
  // CHECK: %[[ONETRANSPOSE:.+]] = memref.transpose %[[arg0]] (d0, d1, d2, d3, d4) -> (d4, d2, d1, d3, d0)
  %0 = memref.transpose %arg0 (d0, d1, d2, d3, d4) -> (d0, d1, d4, d3, d2) : memref<1x2x3x4x5xf32> to memref<1x2x5x4x3xf32, strided<[120, 60, 1, 5, 20]>>
  %1 = memref.transpose %0 (d0, d1, d4, d3, d2) -> (d4, d2, d1, d3, d0) : memref<1x2x5x4x3xf32, strided<[120, 60, 1, 5, 20]>> to memref<5x3x2x4x1xf32, strided<[1, 20, 60, 5, 120]>>
  // CHECK: return %[[ONETRANSPOSE]]
  return %1 : memref<5x3x2x4x1xf32, strided<[1, 20, 60, 5, 120]>>
}

// -----

// CHECK-LABEL: func @fold_identity_transpose(
//  CHECK-SAME:     %[[arg0:.*]]: memref<1x2x3x4x5xf32>
func.func @fold_identity_transpose(%arg0: memref<1x2x3x4x5xf32>) -> memref<1x2x3x4x5xf32> {
  %0 = memref.transpose %arg0 (d0, d1, d2, d3, d4) -> (d1, d0, d4, d3, d2) : memref<1x2x3x4x5xf32> to memref<2x1x5x4x3xf32, strided<[60, 120, 1, 5, 20]>>
  %1 = memref.transpose %0 (d1, d0, d4, d3, d2) -> (d0, d1, d2, d3, d4) : memref<2x1x5x4x3xf32, strided<[60, 120, 1, 5, 20]>> to memref<1x2x3x4x5xf32>
  // CHECK: return %[[arg0]]
  return %1 : memref<1x2x3x4x5xf32>
}

// -----

#transpose_map = affine_map<(d0, d1)[s0] -> (d0 + d1 * s0)>

// CHECK-LABEL: func @cannot_fold_transpose_cast(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?x4xf32>
func.func @cannot_fold_transpose_cast(%arg0: memref<?x4xf32>) -> memref<?x?xf32, #transpose_map> {
    // CHECK: %[[CAST:.*]] = memref.cast %[[arg0]] : memref<?x4xf32> to memref<?x?xf32>
    %cast = memref.cast %arg0 : memref<?x4xf32> to memref<?x?xf32>
    // CHECK: %[[TRANSPOSE:.*]] = memref.transpose %[[CAST]] (d0, d1) -> (d1, d0) : memref<?x?xf32> to memref<?x?xf32, #{{.*}}>
    %transpose = memref.transpose %cast (d0, d1) -> (d1, d0) : memref<?x?xf32> to memref<?x?xf32, #transpose_map>
    // CHECK: return %[[TRANSPOSE]]
    return %transpose : memref<?x?xf32, #transpose_map>
}
