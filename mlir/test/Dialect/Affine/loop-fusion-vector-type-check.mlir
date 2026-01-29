// RUN: mlir-opt --pass-pipeline='builtin.module(affine-loop-fusion)' %s | FileCheck %s

// Test that fusion is prevented when producer and consumer access different-sized vector types
// This is a regression test for issue #115849

// CHECK-LABEL: func.func @illegal_fusion_different_vector_sizes
func.func @illegal_fusion_different_vector_sizes(%a: memref<64x512xf32>, %b: memref<64x512xf32>, %c: memref<64x512xf32>, %d: memref<64x4096xf32>, %e: memref<64x4096xf32>) {
  // The two loops should NOT be fused because they access different vector sizes
  // First loop writes vector<64x64xf32>, second loop reads vector<64x512xf32>
  
  // CHECK: affine.for %[[IV1:.*]] = 0 to 8 {
  // CHECK:   affine.vector_store %{{.*}}, %{{.*}}[0, %[[IV1]] * 64] : memref<64x512xf32>, vector<64x64xf32>
  // CHECK: }
  // CHECK: affine.for %[[IV2:.*]] = 0 to 8 {
  // CHECK:   affine.vector_load %{{.*}}[0, 0] : memref<{{.*}}>, vector<64x512xf32>
  // CHECK:   affine.vector_store %{{.*}}, %{{.*}}[0, %[[IV2]] * 512] : memref<64x4096xf32>, vector<64x512xf32>
  // CHECK: }
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

  func.return
}

// Test that fusion still works when vector sizes match
// CHECK-LABEL: func.func @legal_fusion_same_vector_sizes
func.func @legal_fusion_same_vector_sizes(%a: memref<64x512xf32>, %b: memref<64x512xf32>, %c: memref<64x512xf32>) {
  // These loops should be fused because they use the same vector size
  
  // CHECK: affine.for %[[IV:.*]] = 0 to 8 {
  // CHECK:   arith.addf
  // CHECK:   affine.vector_store %{{.*}}, %{{.*}}[0, %[[IV]] * 64] : memref<64x512xf32>, vector<64x64xf32>
  // CHECK:   affine.vector_load %{{.*}}[0, %[[IV]] * 64] : memref<64x512xf32>, vector<64x64xf32>
  // CHECK:   arith.mulf
  // CHECK:   affine.vector_store %{{.*}}, %{{.*}}[0, %[[IV]] * 64] : memref<64x512xf32>, vector<64x64xf32>
  // CHECK: }
  // CHECK-NOT: affine.for
  affine.for %j = 0 to 8 {
    %lhs = affine.vector_load %a[0, %j * 64] : memref<64x512xf32>, vector<64x64xf32>
    %rhs = affine.vector_load %b[0, %j * 64] : memref<64x512xf32>, vector<64x64xf32>
    %res = arith.addf %lhs, %rhs : vector<64x64xf32>
    affine.vector_store %res, %c[0, %j * 64] : memref<64x512xf32>, vector<64x64xf32>
  }

  affine.for %j = 0 to 8 {
    %lhs = affine.vector_load %c[0, %j * 64] : memref<64x512xf32>, vector<64x64xf32>
    %rhs = affine.vector_load %b[0, %j * 64] : memref<64x512xf32>, vector<64x64xf32>
    %res = arith.mulf %lhs, %rhs : vector<64x64xf32>
    affine.vector_store %res, %c[0, %j * 64] : memref<64x512xf32>, vector<64x64xf32>
  }

  func.return
}