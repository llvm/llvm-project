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

// -----

func.func @reduce(%buffer: memref<100xf32>) {
  %init = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%iv) = (%c0) to (%c1) step (%c1) init (%init) -> f32 {
    %elem_to_reduce = memref.load %buffer[%iv] : memref<100xf32>
    scf.reduce(%elem_to_reduce) : f32 {
      ^bb0(%lhs : f32, %rhs: f32):
        %alloc = memref.alloc() : memref<2xf32>
        memref.store %lhs, %alloc [%c0] : memref<2xf32>
        memref.store %rhs, %alloc [%c1] : memref<2xf32>
        %0 = memref.load %alloc[%c0] : memref<2xf32>
        %1 = memref.load %alloc[%c1] : memref<2xf32>
        %res = arith.addf %0, %1 : f32
        scf.reduce.return %res : f32
    }
  }
  func.return
}

// CHECK-LABEL: func @reduce
//       CHECK: scf.reduce
//       CHECK:   [[ALLOC:%.+]] = memref.alloc(
//       CHECK:   bufferization.dealloc ([[ALLOC]] :{{.*}}) if (%true
//       CHECK:   scf.reduce.return
