// RUN: mlir-opt %s --affine-super-vectorize="virtual-vector-size=4" | FileCheck %s --check-prefix=MASKED
// RUN: mlir-opt %s --affine-super-vectorize="virtual-vector-size=4" \
// RUN:   --eliminate-vector-masks --canonicalize | FileCheck %s
// RUN: mlir-opt %s \
// RUN:   --convert-linalg-to-affine-loops \
// RUN:   --affine-loop-tile="tile-sizes=16,16,16" \
// RUN:   --affine-super-vectorize="virtual-vector-size=4" \
// RUN:   --eliminate-vector-masks \
// RUN:   --canonicalize \
// RUN:   | FileCheck %s --check-prefix=MATMUL
// RUN: mlir-opt %s \
// RUN:   --affine-super-vectorize="virtual-vector-size=4" \
// RUN:   --eliminate-vector-masks --canonicalize \
// RUN:   --convert-vector-to-llvm='enable-gep-inbounds-nuw=1' \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   | FileCheck %s --check-prefix=LLVM

// The vectorizer masks each transfer by the memref bounds.
// MASKED-LABEL: func.func @copy
// MASKED:       vector.mask %{{.*}} { vector.transfer_read
// MASKED:       vector.mask %{{.*}} { vector.transfer_write

// CHECK-LABEL: func.func @copy
// When the memref is static and its dimension is divisible by the vector
// width, the masks are proven all-true and removed.
// CHECK:      %[[READ:.*]] = vector.transfer_read {{.*}} {in_bounds = [true]} : memref<512x512xf32>, vector<4xf32>
// CHECK-NEXT: vector.transfer_write %[[READ]], {{.*}} {in_bounds = [true]} : vector<4xf32>, memref<512x512xf32>{{$}}
// CHECK-NOT:  vector.transfer_read
// CHECK-NOT:  vector.transfer_write

// LLVM-LABEL: llvm.func @copy
// Verify that in_bounds lowers to plain llvm.load/store with inbounds|nuw GEP,
// not masked intrinsics, enabling LLVM to apply no-wrap optimizations.
// LLVM:     llvm.getelementptr inbounds|nuw {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// LLVM:     llvm.load {{.*}} : !llvm.ptr -> vector<4xf32>
// LLVM:     llvm.getelementptr inbounds|nuw {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
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
// Verify that all transfers are unmasked and carry in_bounds=[true] after the
// full linalg-to-affine + tiling + vectorization pipeline.
// MATMUL:      %[[A:.*]] = vector.transfer_read {{.*}} {in_bounds = [true], permutation_map = #{{.*}}} : memref<512x512xf32>, vector<4xf32>
// MATMUL-NEXT: %[[B:.*]] = vector.transfer_read {{.*}} {in_bounds = [true]} : memref<512x512xf32>, vector<4xf32>
// MATMUL-NEXT: %[[C:.*]] = vector.transfer_read {{.*}} {in_bounds = [true]} : memref<512x512xf32>, vector<4xf32>
// MATMUL-NEXT: %[[MUL:.*]] = arith.mulf %[[A]], %[[B]] : vector<4xf32>
// MATMUL-NEXT: %[[ADD:.*]] = arith.addf %[[C]], %[[MUL]] : vector<4xf32>
// MATMUL-NEXT: vector.transfer_write %[[ADD]], {{.*}} {in_bounds = [true]} : vector<4xf32>, memref<512x512xf32>{{$}}
// MATMUL-NOT:  vector.transfer_read
func.func @matmul(%A: memref<512x512xf32>,
                  %B: memref<512x512xf32>,
                  %C: memref<512x512xf32>) {
  linalg.matmul ins(%A, %B : memref<512x512xf32>, memref<512x512xf32>)
               outs(%C : memref<512x512xf32>)
  return
}
