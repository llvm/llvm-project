// RUN: mlir-opt -buffer-loop-hoisting -split-input-file %s | FileCheck %s --implicit-check-not='memref.alloc('
// RUN: mlir-opt -buffer-loop-hoisting -buffer-loop-hoisting -split-input-file %s | FileCheck %s --implicit-check-not='memref.alloc('

// CHECK-LABEL: func.func @affine_parallel_alloc(
// CHECK: affine.parallel
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: memref.store {{.*}}, %[[ALLOC]]
// CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[ALLOC]]
// CHECK-NEXT: memref.store %[[VALUE]]
// CHECK: return
func.func @affine_parallel_alloc(%out: memref<2xindex>) {
  %c0 = arith.constant 0 : index
  affine.parallel (%i) = (0) to (2) {
    %buffer = memref.alloc() : memref<1xindex>
    memref.store %i, %buffer[%c0] : memref<1xindex>
    %value = memref.load %buffer[%c0] : memref<1xindex>
    memref.store %value, %out[%i] : memref<2xindex>
  }
  return
}
