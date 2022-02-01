// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.generic fuse tile-sizes=0,0 run-enable-pass=false" -cse -split-input-file | FileCheck %s

builtin.func @no_fuse_gemm(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %fill = linalg.fill(%cst, %init) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
// CHECK-LABEL: @no_fuse_gemm(
//   CHECK-NOT:   scf.for
//       CHECK:   linalg.fill
//   CHECK-NOT:   scf.for
//       CHECK:   linalg.matmul
