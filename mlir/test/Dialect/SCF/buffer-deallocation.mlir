// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation \
// RUN:   -buffer-deallocation-simplification -split-input-file %s | FileCheck %s

func.func @parallel_insert_slice(%arg0: index) {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<2xf32>
  scf.forall (%arg1) in (%arg0) {
    %alloc0 = memref.alloc() : memref<2xf32>
    %0 = memref.load %alloc[%c0] : memref<2xf32>
    linalg.fill ins(%0 : f32) outs(%alloc0 : memref<2xf32>)
  }
  return
}

// CHECK-LABEL: func @parallel_insert_slice
//  CHECK-SAME: (%arg0: index)
//       CHECK: [[ALLOC0:%.+]] = memref.alloc(
//       CHECK: scf.forall
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc(
//       CHECK:   bufferization.dealloc ([[ALLOC1]] : memref<2xf32>) if (%true
//   CHECK-NOT: retain
//       CHECK: }
//       CHECK: bufferization.dealloc ([[ALLOC0]] : memref<2xf32>) if (%true
//   CHECK-NOT: retain
