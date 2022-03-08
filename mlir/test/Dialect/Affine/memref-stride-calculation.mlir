// RUN: mlir-opt %s -pass-pipeline="builtin.func(test-memref-stride-calculation)" -o /dev/null | FileCheck %s

func @f(%0: index) {
// CHECK-LABEL: Testing: f
  %1 = memref.alloc() : memref<3x4x5xf32>
// CHECK: MemRefType offset: 0 strides: 20, 5, 1
  %2 = memref.alloc(%0) : memref<3x4x?xf32>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1
  %3 = memref.alloc(%0) : memref<3x?x5xf32>
// CHECK: MemRefType offset: 0 strides: ?, 5, 1
  %4 = memref.alloc(%0) : memref<?x4x5xf32>
// CHECK: MemRefType offset: 0 strides: 20, 5, 1
  %5 = memref.alloc(%0, %0) : memref<?x4x?xf32>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1
  %6 = memref.alloc(%0, %0, %0) : memref<?x?x?xf32>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1

  %11 = memref.alloc() : memref<3x4x5xf32, affine_map<(i, j, k)->(i, j, k)>>
// CHECK: MemRefType offset: 0 strides: 20, 5, 1
  %b11 = memref.alloc() : memref<3x4x5xf32, offset: 0, strides: [20, 5, 1]>
// CHECK: MemRefType offset: 0 strides: 20, 5, 1
  %12 = memref.alloc(%0) : memref<3x4x?xf32, affine_map<(i, j, k)->(i, j, k)>>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1
  %13 = memref.alloc(%0) : memref<3x?x5xf32, affine_map<(i, j, k)->(i, j, k)>>
// CHECK: MemRefType offset: 0 strides: ?, 5, 1
  %14 = memref.alloc(%0) : memref<?x4x5xf32, affine_map<(i, j, k)->(i, j, k)>>
// CHECK: MemRefType offset: 0 strides: 20, 5, 1
  %15 = memref.alloc(%0, %0) : memref<?x4x?xf32, affine_map<(i, j, k)->(i, j, k)>>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1
  %16 = memref.alloc(%0, %0, %0) : memref<?x?x?xf32, affine_map<(i, j, k)->(i, j, k)>>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1

  %21 = memref.alloc()[%0] : memref<3x4x5xf32, affine_map<(i, j, k)[M]->(32 * i + 16 * j + M * k + 1)>>
// CHECK: MemRefType offset: 1 strides: 32, 16, ?
  %22 = memref.alloc()[%0] : memref<3x4x5xf32, affine_map<(i, j, k)[M]->(32 * i + M * j + 16 * k + 3)>>
// CHECK: MemRefType offset: 3 strides: 32, ?, 16
  %b22 = memref.alloc(%0)[%0, %0] : memref<3x4x?xf32, offset: 0, strides: [?, ?, 1]>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1
  %23 = memref.alloc(%0)[%0] : memref<3x?x5xf32, affine_map<(i, j, k)[M]->(M * i + 32 * j + 16 * k + 7)>>
// CHECK: MemRefType offset: 7 strides: ?, 32, 16
  %b23 = memref.alloc(%0)[%0] : memref<3x?x5xf32, offset: 0, strides: [?, 5, 1]>
// CHECK: MemRefType offset: 0 strides: ?, 5, 1
  %24 = memref.alloc(%0)[%0] : memref<3x?x5xf32, affine_map<(i, j, k)[M]->(M * i + 32 * j + 16 * k + M)>>
// CHECK: MemRefType offset: ? strides: ?, 32, 16
  %b24 = memref.alloc(%0)[%0, %0] : memref<3x?x5xf32, offset: ?, strides: [?, 32, 16]>
// CHECK: MemRefType offset: ? strides: ?, 32, 16
  %25 = memref.alloc(%0, %0)[%0, %0] : memref<?x?x16xf32, affine_map<(i, j, k)[M, N]->(M * i + N * j + k + 1)>>
// CHECK: MemRefType offset: 1 strides: ?, ?, 1
  %b25 = memref.alloc(%0, %0)[%0, %0] : memref<?x?x16xf32, offset: 1, strides: [?, ?, 1]>
// CHECK: MemRefType offset: 1 strides: ?, ?, 1
  %26 = memref.alloc(%0)[] : memref<?xf32, affine_map<(i)[M]->(i)>>
// CHECK: MemRefType offset: 0 strides: 1
  %27 = memref.alloc()[%0] : memref<5xf32, affine_map<(i)[M]->(M)>>
// CHECK: MemRefType memref<5xf32, affine_map<(d0)[s0] -> (s0)>> cannot be converted to strided form
  %28 = memref.alloc()[%0] : memref<5xf32, affine_map<(i)[M]->(123)>>
// CHECK: MemRefType memref<5xf32, affine_map<(d0)[s0] -> (123)>> cannot be converted to strided form
  %29 = memref.alloc()[%0] : memref<f32, affine_map<()[M]->(M)>>
// CHECK: MemRefType offset: ? strides:
  %30 = memref.alloc()[%0] : memref<f32, affine_map<()[M]->(123)>>
// CHECK: MemRefType offset: 123 strides:

  %101 = memref.alloc() : memref<3x4x5xf32, affine_map<(i, j, k)->(i floordiv 4 + j + k)>>
// CHECK: MemRefType memref<3x4x5xf32, affine_map<(d0, d1, d2) -> (d0 floordiv 4 + d1 + d2)>> cannot be converted to strided form
  %102 = memref.alloc() : memref<3x4x5xf32, affine_map<(i, j, k)->(i ceildiv 4 + j + k)>>
// CHECK: MemRefType memref<3x4x5xf32, affine_map<(d0, d1, d2) -> (d0 ceildiv 4 + d1 + d2)>> cannot be converted to strided form
  %103 = memref.alloc() : memref<3x4x5xf32, affine_map<(i, j, k)->(i mod 4 + j + k)>>
// CHECK: MemRefType memref<3x4x5xf32, affine_map<(d0, d1, d2) -> (d0 mod 4 + d1 + d2)>> cannot be converted to strided form

  %200 = memref.alloc()[%0, %0, %0] : memref<3x4x5xf32, affine_map<(i, j, k)[M, N, K]->(M * i + N * i + N * j + K * k - (M + N - 20)* i)>>
  // CHECK: MemRefType offset: 0 strides: 20, ?, ?
  %201 = memref.alloc()[%0, %0, %0] : memref<3x4x5xf32, affine_map<(i, j, k)[M, N, K]->(M * i + N * i + N * K * j + K * K * k - (M + N - 20) * (i + 1))>>
  // CHECK: MemRefType offset: ? strides: 20, ?, ?
  %202 = memref.alloc()[%0, %0, %0] : memref<3x4x5xf32, affine_map<(i, j, k)[M, N, K]->(M * (i + 1) + j + k - M)>>
  // CHECK: MemRefType offset: 0 strides: ?, 1, 1
  %203 = memref.alloc()[%0, %0, %0] : memref<3x4x5xf32, affine_map<(i, j, k)[M, N, K]->(M + M * (i + N * (j + K * k)))>>
  // CHECK: MemRefType offset: ? strides: ?, ?, ?

  return
}
