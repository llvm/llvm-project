// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.matmul fuse tile-sizes=0,0,0 run-enable-pass=false" -split-input-file | FileCheck --check-prefix=MATMUL %s
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.elemwise_unary fuse tile-sizes=32,32,0 run-enable-pass=false" -split-input-file | FileCheck --check-prefix=UNARY %s

// MATMUL-LABEL: @tile_sizes_zero(
func.func @tile_sizes_zero(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>

  //   MATMUL-NOT:   scf.for
  //       MATMUL:   linalg.fill
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>

  //   MATMUL-NOT:   scf.for
  //       MATMUL:   linalg.matmul
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %result : tensor<?x?xf32>
}

// -----

// UNARY_LABEL: @shape_only(
func.func @shape_only(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32

  //       UNARY:   linalg.fill
  %0 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<?x?xf32>) -> tensor<?x?xf32>

  //       UNARY:   scf.for
  //       UNARY:     scf.for
  //   UNARY-NOT:       linalg.fill
  //       UNARY:       linalg.elemwise_unary
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
      ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}
