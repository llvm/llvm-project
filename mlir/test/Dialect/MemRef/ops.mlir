// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>

// CHECK-LABEL: func @alloc() {
func.func @alloc() {
^bb0:
  // Test simple alloc.
  // CHECK: %{{.*}} = memref.alloc() : memref<1024x64xf32, 1>
  %0 = memref.alloc() : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>

  %c0 = "arith.constant"() {value = 0: index} : () -> index
  %c1 = "arith.constant"() {value = 1: index} : () -> index

  // Test alloc with dynamic dimensions.
  // CHECK: %{{.*}} = memref.alloc(%{{.*}}, %{{.*}}) : memref<?x?xf32, 1>
  %1 = memref.alloc(%c0, %c1) : memref<?x?xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>

  // Test alloc with no dynamic dimensions and one symbol.
  // CHECK: %{{.*}} = memref.alloc()[%{{.*}}] : memref<2x4xf32, #[[$MAP]], 1>
  %2 = memref.alloc()[%c0] : memref<2x4xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>

  // Test alloc with dynamic dimensions and one symbol.
  // CHECK: %{{.*}} = memref.alloc(%{{.*}})[%{{.*}}] : memref<2x?xf32, #[[$MAP]], 1>
  %3 = memref.alloc(%c1)[%c0] : memref<2x?xf32, affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>, 1>

  // Alloc with no mappings.
  // b/116054838 Parser crash while parsing ill-formed AllocOp
  // CHECK: %{{.*}} = memref.alloc() : memref<2xi32>
  %4 = memref.alloc() : memref<2 x i32>

  // CHECK:   return
  return
}

// CHECK-LABEL: func @alloca() {
func.func @alloca() {
^bb0:
  // Test simple alloc.
  // CHECK: %{{.*}} = memref.alloca() : memref<1024x64xf32, 1>
  %0 = memref.alloca() : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>

  %c0 = "arith.constant"() {value = 0: index} : () -> index
  %c1 = "arith.constant"() {value = 1: index} : () -> index

  // Test alloca with dynamic dimensions.
  // CHECK: %{{.*}} = memref.alloca(%{{.*}}, %{{.*}}) : memref<?x?xf32, 1>
  %1 = memref.alloca(%c0, %c1) : memref<?x?xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>

  // Test alloca with no dynamic dimensions and one symbol.
  // CHECK: %{{.*}} = memref.alloca()[%{{.*}}] : memref<2x4xf32, #[[$MAP]], 1>
  %2 = memref.alloca()[%c0] : memref<2x4xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>

  // Test alloca with dynamic dimensions and one symbol.
  // CHECK: %{{.*}} = memref.alloca(%{{.*}})[%{{.*}}] : memref<2x?xf32, #[[$MAP]], 1>
  %3 = memref.alloca(%c1)[%c0] : memref<2x?xf32, affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>, 1>

  // Alloca with no mappings, but with alignment.
  // CHECK: %{{.*}} = memref.alloca() {alignment = 64 : i64} : memref<2xi32>
  %4 = memref.alloca() {alignment = 64} : memref<2 x i32>

  return
}

// CHECK-LABEL: func @dealloc() {
func.func @dealloc() {
^bb0:
  // CHECK: %{{.*}} = memref.alloc() : memref<1024x64xf32>
  %0 = memref.alloc() : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 0>

  // CHECK: memref.dealloc %{{.*}} : memref<1024x64xf32>
  memref.dealloc %0 : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 0>
  return
}

// CHECK-LABEL: func @load_store
func.func @load_store() {
^bb0:
  // CHECK: %{{.*}} = memref.alloc() : memref<1024x64xf32, 1>
  %0 = memref.alloc() : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>

  %1 = arith.constant 0 : index
  %2 = arith.constant 1 : index

  // CHECK: %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x64xf32, 1>
  %3 = memref.load %0[%1, %2] : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>

  // CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x64xf32, 1>
  memref.store %3, %0[%1, %2] : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>

  return
}

// CHECK-LABEL: func @dma_ops()
func.func @dma_ops() {
  %c0 = arith.constant 0 : index
  %stride = arith.constant 32 : index
  %elt_per_stride = arith.constant 16 : index

  %A = memref.alloc() : memref<256 x f32, affine_map<(d0) -> (d0)>, 0>
  %Ah = memref.alloc() : memref<256 x f32, affine_map<(d0) -> (d0)>, 1>
  %tag = memref.alloc() : memref<1 x f32>

  %num_elements = arith.constant 256 : index

  memref.dma_start %A[%c0], %Ah[%c0], %num_elements, %tag[%c0] : memref<256 x f32>, memref<256 x f32, 1>, memref<1 x f32>
  memref.dma_wait %tag[%c0], %num_elements : memref<1 x f32>
  // CHECK: dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}, %{{.*}}[%{{.*}}] : memref<256xf32>, memref<256xf32, 1>, memref<1xf32>
  // CHECK-NEXT:  dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xf32>

  // DMA with strides
  memref.dma_start %A[%c0], %Ah[%c0], %num_elements, %tag[%c0], %stride, %elt_per_stride : memref<256 x f32>, memref<256 x f32, 1>, memref<1 x f32>
  memref.dma_wait %tag[%c0], %num_elements : memref<1 x f32>
  // CHECK-NEXT:  dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}, %{{.*}}[%{{.*}}], %{{.*}}, %{{.*}} : memref<256xf32>, memref<256xf32, 1>, memref<1xf32>
  // CHECK-NEXT:  dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xf32>

  return
}

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

// CHECK-LABEL: func @memref_cast(%arg0
func.func @memref_cast(%arg0: memref<4xf32>, %arg1 : memref<?xf32>, %arg2 : memref<64x16x4xf32, strided<[64, 4, 1], offset: 0>>) {
  // CHECK: memref.cast %{{.*}} : memref<4xf32> to memref<?xf32>
  %0 = memref.cast %arg0 : memref<4xf32> to memref<?xf32>

  // CHECK: memref.cast %{{.*}} : memref<?xf32> to memref<4xf32>
  %1 = memref.cast %arg1 : memref<?xf32> to memref<4xf32>

  // CHECK: memref.cast %{{.*}} : memref<64x16x4xf32, strided<[64, 4, 1]>> to memref<64x16x4xf32, strided<[?, ?, ?], offset: ?>>
  %2 = memref.cast %arg2 : memref<64x16x4xf32, strided<[64, 4, 1], offset: 0>> to memref<64x16x4xf32, strided<[?, ?, ?], offset: ?>>

  // CHECK: memref.cast {{%.*}} : memref<64x16x4xf32, strided<[?, ?, ?], offset: ?>> to memref<64x16x4xf32, strided<[64, 4, 1]>>
  %3 = memref.cast %2 : memref<64x16x4xf32, strided<[?, ?, ?], offset: ?>> to memref<64x16x4xf32, strided<[64, 4, 1], offset: 0>>

  // CHECK: memref.cast %{{.*}} : memref<4xf32> to memref<*xf32>
  %4 = memref.cast %1 : memref<4xf32> to memref<*xf32>

  // CHECK: memref.cast %{{.*}} : memref<*xf32> to memref<4xf32>
  %5 = memref.cast %4 : memref<*xf32> to memref<4xf32>
  return
}

// Check that unranked memrefs with non-default memory space roundtrip
// properly.
// CHECK-LABEL: @unranked_memref_roundtrip(memref<*xf32, 4>)
func.func private @unranked_memref_roundtrip(memref<*xf32, 4>)

// CHECK-LABEL: func @load_store_prefetch
func.func @load_store_prefetch(memref<4x4xi32>, index) {
^bb0(%0: memref<4x4xi32>, %1: index):
  // CHECK: %0 = memref.load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %2 = "memref.load"(%0, %1, %1) : (memref<4x4xi32>, index, index)->i32

  // CHECK: %{{.*}} = memref.load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %3 = memref.load %0[%1, %1] : memref<4x4xi32>

  // CHECK: memref.prefetch %arg0[%arg1, %arg1], write, locality<1>, data : memref<4x4xi32>
  memref.prefetch %0[%1, %1], write, locality<1>, data : memref<4x4xi32>

  // CHECK: memref.prefetch %arg0[%arg1, %arg1], read, locality<3>, instr : memref<4x4xi32>
  memref.prefetch %0[%1, %1], read, locality<3>, instr : memref<4x4xi32>

  return
}

// Test with zero-dimensional operands using no index in load/store.
// CHECK-LABEL: func @zero_dim_no_idx
func.func @zero_dim_no_idx(%arg0 : memref<i32>, %arg1 : memref<i32>, %arg2 : memref<i32>) {
  %0 = memref.load %arg0[] : memref<i32>
  memref.store %0, %arg1[] : memref<i32>
  return
  // CHECK: %0 = memref.load %{{.*}}[] : memref<i32>
  // CHECK: memref.store %{{.*}}, %{{.*}}[] : memref<i32>
}

// CHECK-LABEL: func @memref_view(%arg0
func.func @memref_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<2048xi8>
  // Test two dynamic sizes and dynamic offset.
  // CHECK: memref.view {{.*}} : memref<2048xi8> to memref<?x?xf32>
  %1 = memref.view %0[%arg2][%arg0, %arg1] : memref<2048xi8> to memref<?x?xf32>

  // Test one dynamic size and dynamic offset.
  // CHECK: memref.view {{.*}} : memref<2048xi8> to memref<4x?xf32>
  %3 = memref.view %0[%arg2][%arg1] : memref<2048xi8> to memref<4x?xf32>

  // Test static sizes and static offset.
  // CHECK: memref.view {{.*}} : memref<2048xi8> to memref<64x4xf32>
  %c0 = arith.constant 0: index
  %5 = memref.view %0[%c0][] : memref<2048xi8> to memref<64x4xf32>
  return
}

// CHECK-LABEL: func @assume_alignment
// CHECK-SAME: %[[MEMREF:.*]]: memref<4x4xf16>
func.func @assume_alignment(%0: memref<4x4xf16>) {
  // CHECK: memref.assume_alignment %[[MEMREF]], 16 : memref<4x4xf16>
  memref.assume_alignment %0, 16 : memref<4x4xf16>
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
    %arg7: memref<1x2049xi64, strided<[?, ?], offset: ?>>,
    %arg8: memref<1x1x1024xi8, strided<[40960, 4096, 1], offset: 0>>,
    %arg9: memref<24x1x1x1024xi8, strided<[40960, 40960, 4096, 1], offset: 0>>) {
  // Reshapes that collapse and expand back a contiguous buffer.
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<12x5xf32>
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] :
    memref<3x4x5xf32> into memref<12x5xf32>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]] output_shape [3, 4, 5]
//  CHECK-SAME:     memref<12x5xf32> into memref<3x4x5xf32>
  %r0 = memref.expand_shape %0 [[0, 1], [2]] output_shape [3, 4, 5] :
    memref<12x5xf32> into memref<3x4x5xf32>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0], [1, 2]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<3x20xf32>
  %1 = memref.collapse_shape %arg0 [[0], [1, 2]] :
    memref<3x4x5xf32> into memref<3x20xf32>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0], [1, 2]] output_shape [3, 4, 5]
//  CHECK-SAME:     memref<3x20xf32> into memref<3x4x5xf32>
  %r1 = memref.expand_shape %1 [[0], [1, 2]] output_shape [3, 4, 5] :
    memref<3x20xf32> into memref<3x4x5xf32>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1, 2]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<60xf32>
  %2 = memref.collapse_shape %arg0 [[0, 1, 2]] :
    memref<3x4x5xf32> into memref<60xf32>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1, 2]] output_shape [3, 4, 5]
//  CHECK-SAME:     memref<60xf32> into memref<3x4x5xf32>
  %r2 = memref.expand_shape %2 [[0, 1, 2]] output_shape [3, 4, 5] :
      memref<60xf32> into memref<3x4x5xf32>

//       CHECK:   memref.expand_shape {{.*}} [] output_shape [1, 1]
//  CHECK-SAME:     memref<f32> into memref<1x1xf32>
  %r5 = memref.expand_shape %arg5 [] output_shape [1, 1] :
      memref<f32> into memref<1x1xf32>

// Reshapes with a custom layout map.
//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0], [1, 2]] output_shape [30, 4, 5]
  %l0 = memref.expand_shape %arg3 [[0], [1, 2]] output_shape [30, 4, 5] :
      memref<30x20xf32, strided<[4000, 2], offset: 100>>
      into memref<30x4x5xf32, strided<[4000, 10, 2], offset: 100>>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]] output_shape [2, 15, 20]
  %l1 = memref.expand_shape %arg3 [[0, 1], [2]] output_shape [2, 15, 20] :
      memref<30x20xf32, strided<[4000, 2], offset: 100>>
      into memref<2x15x20xf32, strided<[60000, 4000, 2], offset: 100>>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0], [1, 2]] output_shape [1, 1, 5]
  %r4 = memref.expand_shape %arg4 [[0], [1, 2]] output_shape [1, 1, 5] :
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

    // %arg8: memref<1x1x1024xi8, strided<[40960, 4096, 1], offset: 0>>,
    // %arg9: memref<24x1x1x1024xi8, strided<[40960, 40960, 4096, 1], offset: 0>>) {

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1, 2]]
  %r8 = memref.collapse_shape %arg8 [[0, 1, 2]] :
      memref<1x1x1024xi8, strided<[40960, 4096, 1], offset: 0>> into
      memref<1024xi8, strided<[1], offset: 0>>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0], [1, 2, 3]]
  %r9 = memref.collapse_shape %arg9 [[0], [1, 2, 3]] :
      memref<24x1x1x1024xi8, strided<[40960, 40960, 4096, 1], offset: 0>> into
      memref<24x1024xi8, strided<[40960, 1], offset: 0>>

  // Reshapes that expand and collapse back a contiguous buffer with some 1's.
//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2], [3, 4]] output_shape [1, 3, 4, 1, 5]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<1x3x4x1x5xf32>
  %3 = memref.expand_shape %arg0 [[0, 1], [2], [3, 4]] output_shape [1, 3, 4, 1, 5]:
    memref<3x4x5xf32> into memref<1x3x4x1x5xf32>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2], [3, 4]]
//  CHECK-SAME:     memref<1x3x4x1x5xf32> into memref<3x4x5xf32>
  %r3 = memref.collapse_shape %3 [[0, 1], [2], [3, 4]] :
    memref<1x3x4x1x5xf32> into memref<3x4x5xf32>

  // Reshapes on tensors.
//       CHECK:   tensor.expand_shape {{.*}}: tensor<3x4x5xf32> into tensor<1x3x4x1x5xf32>
  %t0 = tensor.expand_shape %arg1 [[0, 1], [2], [3, 4]] output_shape [1, 3, 4, 1, 5] :
    tensor<3x4x5xf32> into tensor<1x3x4x1x5xf32>

//       CHECK:   tensor.collapse_shape {{.*}}: tensor<1x3x4x1x5xf32> into tensor<3x4x5xf32>
  %rt0 = tensor.collapse_shape %t0 [[0, 1], [2], [3, 4]] :
    tensor<1x3x4x1x5xf32> into tensor<3x4x5xf32>

//       CHECK:   tensor.dim %arg2, {{.*}} : tensor<3x?x5xf32>
//       CHECK:   tensor.expand_shape {{.*}}: tensor<3x?x5xf32> into tensor<1x3x?x1x5xf32>
  %c1 = arith.constant 1 : index
  %sz1 = tensor.dim %arg2, %c1 : tensor<3x?x5xf32>
  %t1 = tensor.expand_shape %arg2 [[0, 1], [2], [3, 4]] output_shape [1, 3, %sz1, 1, 5] :
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
         %arg3: memref<?x42xf32, strided<[42, 1], offset: 0>>,
         %arg4: index,
         %arg5: index,
         %arg6: index,
         %arg7: memref<4x?x4xf32>) {
//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?x?xf32> into memref<?x?xf32>
  %0 = memref.collapse_shape %arg0 [[0, 1], [2]] :
    memref<?x?x?xf32> into memref<?x?xf32>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]] output_shape [%arg4, 4, %arg5]
//  CHECK-SAME:     memref<?x?xf32> into memref<?x4x?xf32>
  %r0 = memref.expand_shape %0 [[0, 1], [2]] output_shape [%arg4, 4, %arg5] :
    memref<?x?xf32> into memref<?x4x?xf32>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?x?xf32, strided<[?, ?, 1]>> into memref<?x?xf32, strided<[?, 1]>>
  %1 = memref.collapse_shape %arg1 [[0, 1], [2]] :
    memref<?x?x?xf32, strided<[?, ?, 1], offset: 0>> into
    memref<?x?xf32, strided<[?, 1], offset: 0>>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]] output_shape [%arg4, 4, %arg5]
//  CHECK-SAME:     memref<?x?xf32, strided<[?, 1]>> into memref<?x4x?xf32, strided<[?, ?, 1]>>
  %r1 = memref.expand_shape %1 [[0, 1], [2]] output_shape [%arg4, 4, %arg5] :
    memref<?x?xf32, strided<[?, 1], offset: 0>> into
    memref<?x4x?xf32, strided<[?, ?, 1], offset: 0>>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1], [2]]
//  CHECK-SAME:     memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>> into memref<?x?xf32, strided<[?, 1], offset: ?>>
  %2 = memref.collapse_shape %arg2 [[0, 1], [2]] :
    memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>> into
    memref<?x?xf32, strided<[?, 1], offset: ?>>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2]] output_shape [%arg4, 4, %arg5]
//  CHECK-SAME:     memref<?x?xf32, strided<[?, 1], offset: ?>> into memref<?x4x?xf32, strided<[?, ?, 1], offset: ?>>
  %r2 = memref.expand_shape %2 [[0, 1], [2]] output_shape [%arg4, 4, %arg5] :
    memref<?x?xf32, strided<[?, 1], offset: ?>> into
    memref<?x4x?xf32, strided<[?, ?, 1], offset: ?>>

//       CHECK:   memref.collapse_shape {{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:     memref<?x42xf32, strided<[42, 1]>> into memref<?xf32, strided<[1]>>
  %3 = memref.collapse_shape %arg3 [[0, 1]] :
    memref<?x42xf32, strided<[42, 1], offset: 0>> into
    memref<?xf32, strided<[1]>>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1]] output_shape [%arg6, 42]
//  CHECK-SAME:     memref<?xf32, strided<[1]>> into memref<?x42xf32>
  %r3 = memref.expand_shape %3 [[0, 1]] output_shape [%arg6, 42] :
    memref<?xf32, strided<[1]>> into memref<?x42xf32>

//       CHECK:   memref.expand_shape {{.*}} {{\[}}[0, 1], [2], [3, 4]]
  %4 = memref.expand_shape %arg7 [[0, 1], [2], [3, 4]] output_shape [2, 2, %arg4, 2, 2]
        : memref<4x?x4xf32> into memref<2x2x?x2x2xf32>
  return
}

func.func @expand_collapse_shape_zero_dim(%arg0 : memref<1x1xf32>, %arg1 : memref<f32>)
    -> (memref<f32>, memref<1x1xf32>) {
  %0 = memref.collapse_shape %arg0 [] : memref<1x1xf32> into memref<f32>
  %1 = memref.expand_shape %0 [] output_shape [1, 1] : memref<f32> into memref<1x1xf32>
  return %0, %1 : memref<f32>, memref<1x1xf32>
}
// CHECK-LABEL: func @expand_collapse_shape_zero_dim
//       CHECK:   memref.collapse_shape %{{.*}} [] : memref<1x1xf32> into memref<f32>
//       CHECK:   memref.expand_shape %{{.*}} [] output_shape [1, 1] : memref<f32> into memref<1x1xf32>

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
    %m1: memref<4x5x6xf32, strided<[1, ?, 1000], offset: 0>>,
    %sz0: index,
    %sz1: index) {

  %r0 = memref.expand_shape %m0 [[0], [1, 2]] output_shape [%sz0, %sz1, 5] :
    memref<?x?xf32, strided<[1, 10], offset: 0>> into
    memref<?x?x5xf32, strided<[1, 50, 10], offset: 0>>
  %rr0 = memref.collapse_shape %r0 [[0], [1, 2]] :
    memref<?x?x5xf32, strided<[1, 50, 10], offset: 0>> into
    memref<?x?xf32, strided<[1, 10], offset: 0>>

  %r1 = memref.expand_shape %m1 [[0, 1], [2], [3, 4]] output_shape [2, 2, 5, 2, 3] :
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

// CHECK-LABEL: func @memref_memory_space_cast
func.func @memref_memory_space_cast(%src : memref<?xf32>) -> memref<?xf32, 1> {
  %dst = memref.memory_space_cast %src : memref<?xf32> to memref<?xf32, 1>
  return %dst : memref<?xf32, 1>
}

// CHECK-LABEL: func @memref_transpose_map
func.func @memref_transpose_map(%src : memref<?x?xf32>) -> memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>> {
  %dst = memref.transpose %src (i, j) -> (j, i) : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>>
  return %dst : memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>>
}
