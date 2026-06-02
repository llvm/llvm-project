// RUN: mlir-opt %s --affine-super-vectorize="virtual-vector-size=4" | FileCheck %s
// RUN: mlir-opt %s \
// RUN:   --convert-linalg-to-affine-loops \
// RUN:   --affine-loop-tile="tile-sizes=16,16,16" \
// RUN:   --enable-loopinterchange \
// RUN:   --affine-super-vectorize="virtual-vector-size=4" \
// RUN:   --canonicalize \
// RUN:   | FileCheck %s --check-prefix=MATMUL
// RUN: mlir-opt %s \
// RUN:   --affine-super-vectorize="virtual-vector-size=4" \
// RUN:   --convert-vector-to-llvm \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   | FileCheck %s --check-prefix=LLVM

// CHECK-LABEL: func.func @copy
// Verify that transfer_read and transfer_write carry {in_bounds = [true]} when
// the memref is static and its dimension is divisible by the vector width.
// CHECK:     vector.transfer_read {{.*}} {in_bounds = [true]} : memref<512x512xf32>, vector<4xf32>
// CHECK-NOT: vector.transfer_read
// CHECK:     vector.transfer_write {{.*}} {in_bounds = [true]} : vector<4xf32>, memref<512x512xf32>
// CHECK-NOT: vector.transfer_write

// LLVM-LABEL: llvm.func @copy
// Verify that in_bounds lowers to plain llvm.load/store, not masked intrinsics.
// LLVM:     llvm.load {{.*}} : !llvm.ptr -> vector<4xf32>
// LLVM:     llvm.store {{.*}} : vector<4xf32>, !llvm.ptr
// LLVM-NOT: llvm.intr.masked.load
// LLVM-NOT: llvm.intr.masked.store
func.func @copy(%A: memref<512x512xf32>, %B: memref<512x512xf32>) {
  affine.for %i = 0 to 512 {
    affine.for %j = 0 to 512 {
      %v = affine.load %A[%i, %j] : memref<512x512xf32>
      affine.store %v, %B[%i, %j] : memref<512x512xf32>
    }
  }
  return
}


// MATMUL-LABEL: func.func @matmul
// Verify all three transfer_read ops carry in_bounds=[true] after the full
// linalg-to-affine + tiling + vectorization pipeline.
// Without the fix only the broadcast A-read gets it; B and C do not.
// MATMUL-COUNT-3: vector.transfer_read {{.*}} {in_bounds = [true]
// MATMUL-NOT:     vector.transfer_read
func.func @matmul(%A: memref<512x512xf32>,
                  %B: memref<512x512xf32>,
                  %C: memref<512x512xf32>) {
  linalg.matmul ins(%A, %B : memref<512x512xf32>, memref<512x512xf32>)
               outs(%C : memref<512x512xf32>)
  return
}
