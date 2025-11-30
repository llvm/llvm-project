// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion))' -split-input-file | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion{maximal mode=sibling}))' -split-input-file | FileCheck %s --check-prefix=SIBLING

// CHECK-LABEL: func.func @skip_fusing_mismatched_vectors
// CHECK: affine.for %{{.*}} = 0 to 8 {
// CHECK:   affine.vector_store {{.*}} : memref<64x512xf32>, vector<64x64xf32>
// CHECK: }
// CHECK: affine.for %{{.*}} = 0 to 8 {
// CHECK:   affine.vector_load {{.*}} : memref<64x512xf32>, vector<64x512xf32>
// CHECK: }
func.func @skip_fusing_mismatched_vectors(%a: memref<64x512xf32>, %b: memref<64x512xf32>, %c: memref<64x512xf32>, %d: memref<64x4096xf32>, %e: memref<64x4096xf32>) {
  affine.for %j = 0 to 8 {
    %lhs = affine.vector_load %a[0, %j * 64] : memref<64x512xf32>, vector<64x64xf32>
    %rhs = affine.vector_load %b[0, %j * 64] : memref<64x512xf32>, vector<64x64xf32>
    %res = arith.addf %lhs, %rhs : vector<64x64xf32>
    affine.vector_store %res, %c[0, %j * 64] : memref<64x512xf32>, vector<64x64xf32>
  }

  affine.for %j = 0 to 8 {
    %lhs = affine.vector_load %c[0, 0] : memref<64x512xf32>, vector<64x512xf32>
    %rhs = affine.vector_load %d[0, %j * 512] : memref<64x4096xf32>, vector<64x512xf32>
    %res = arith.subf %lhs, %rhs : vector<64x512xf32>
    affine.vector_store %res, %d[0, %j * 512] : memref<64x4096xf32>, vector<64x512xf32>
  }
  return
}

// -----

// CHECK-LABEL: func.func @vector_private_memref
// CHECK: memref.alloc() : memref<1x64xf32>
// CHECK-NOT: memref<1x1xf32>
// CHECK: affine.vector_store {{.*}} : memref<1x64xf32>, vector<64xf32>
func.func @vector_private_memref(%src: memref<10x64xf32>, %dst: memref<10x64xf32>) {
  %tmp = memref.alloc() : memref<10x64xf32>
  affine.for %i = 0 to 10 {
    %vec = affine.vector_load %src[%i, 0] : memref<10x64xf32>, vector<64xf32>
    affine.vector_store %vec, %tmp[%i, 0] : memref<10x64xf32>, vector<64xf32>
  }

  affine.for %i = 0 to 10 {
    %vec = affine.vector_load %tmp[%i, 0] : memref<10x64xf32>, vector<64xf32>
    affine.vector_store %vec, %dst[%i, 0] : memref<10x64xf32>, vector<64xf32>
  }
  return
}

// -----

// CHECK-LABEL: func.func @fuse_scalar_vector
// CHECK: %[[TMP:.*]] = memref.alloc() : memref<64xf32>
// CHECK: affine.for %[[I:.*]] = 0 to 16 {
// CHECK:   %[[S0:.*]] = affine.load %[[SRC:.*]][%[[I]] * 4] : memref<64xf32>
// CHECK:   affine.store %[[S0]], %[[TMP]][%[[I]] * 4] : memref<64xf32>
// CHECK:   %[[V:.*]] = affine.vector_load %[[TMP]][%[[I]] * 4] : memref<64xf32>, vector<4xf32>
// CHECK:   affine.vector_store %[[V]], %[[DST:.*]][%[[I]] * 4] : memref<64xf32>, vector<4xf32>
// CHECK: }
func.func @fuse_scalar_vector(%src: memref<64xf32>, %dst: memref<64xf32>) {
  %tmp = memref.alloc() : memref<64xf32>
  affine.for %i = 0 to 16 {
    %s0 = affine.load %src[%i * 4] : memref<64xf32>
    affine.store %s0, %tmp[%i * 4] : memref<64xf32>
    %s1 = affine.load %src[%i * 4 + 1] : memref<64xf32>
    affine.store %s1, %tmp[%i * 4 + 1] : memref<64xf32>
    %s2 = affine.load %src[%i * 4 + 2] : memref<64xf32>
    affine.store %s2, %tmp[%i * 4 + 2] : memref<64xf32>
    %s3 = affine.load %src[%i * 4 + 3] : memref<64xf32>
    affine.store %s3, %tmp[%i * 4 + 3] : memref<64xf32>
  }

  affine.for %i = 0 to 16 {
    %vec = affine.vector_load %tmp[%i * 4] : memref<64xf32>, vector<4xf32>
    affine.vector_store %vec, %dst[%i * 4] : memref<64xf32>, vector<4xf32>
  }
  memref.dealloc %tmp : memref<64xf32>
  return
}

// -----

// SIBLING-LABEL: func.func @sibling_vector_mismatch
// SIBLING: affine.for %{{.*}} = 0 to 10 {
// SIBLING:   affine.vector_load %{{.*}} : memref<10x16xf32>, vector<4xf32>
// SIBLING:   affine.vector_load %{{.*}} : memref<10x16xf32>, vector<8xf32>
// SIBLING:   affine.vector_load %{{.*}} : memref<10x16xf32>, vector<4xf32>
// SIBLING: }
func.func @sibling_vector_mismatch(%src: memref<10x16xf32>) {
  affine.for %i = 0 to 10 {
    %vec = affine.vector_load %src[%i, 0] : memref<10x16xf32>, vector<4xf32>
  }

  affine.for %i = 0 to 10 {
    %wide = affine.vector_load %src[%i, 8] : memref<10x16xf32>, vector<8xf32>
    %vec = affine.vector_load %src[%i, 0] : memref<10x16xf32>, vector<4xf32>
  }
  return
}
