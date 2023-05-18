// RUN: mlir-opt %s | FileCheck %s

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
