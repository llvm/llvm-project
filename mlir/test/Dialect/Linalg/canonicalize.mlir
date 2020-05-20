// RUN: mlir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func @memref_cast(
func @memref_cast(%a: index, %b: index) -> memref<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c16 = constant 16 : index
  %1 = alloc (%b) : memref<?xi8>
  %2 = view %1[][] : memref<?xi8> to memref<16x16xf32>
  %3 = memref_cast %2 : memref<16x16xf32> to memref<?x?xf32>
  %r0 = linalg.range %c0:%c8:%c1 : !linalg.range

  // CHECK:  linalg.slice {{.*}} : memref<16x16xf32>, !linalg.range, !linalg.range, memref<?x?xf32>
  %4 = linalg.slice %3[%r0, %r0] : memref<?x?xf32>, !linalg.range, !linalg.range, memref<?x?xf32>

  // CHECK:  linalg.matmul{{.*}}: memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>
  linalg.matmul(%3, %3, %3) : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  return %4: memref<?x?xf32>
}
