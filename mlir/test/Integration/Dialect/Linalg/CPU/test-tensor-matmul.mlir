// UNSUPPORTED: asan
// RUN: mlir-opt %s -test-transform-dialect-erase-schedule \
// RUN: -one-shot-bufferize="bufferize-function-boundaries" \
// RUN: -finalizing-bufferize -buffer-deallocation-pipeline -convert-bufferization-to-memref -convert-linalg-to-loops -convert-scf-to-cf \
// RUN: -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-scf-to-cf --finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -transform-interpreter -test-transform-dialect-erase-schedule \
// RUN: -one-shot-bufferize="bufferize-function-boundaries" \
// RUN: -finalizing-bufferize -convert-linalg-to-loops -convert-scf-to-cf -convert-scf-to-cf \
// RUN:  -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-scf-to-cf --finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils \
// RUN: | FileCheck %s

func.func @main() {
  %A = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %B = arith.constant dense<[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32>
  %C = arith.constant dense<1000.0> : tensor<2x4xf32>

  %D = linalg.matmul ins(%A, %B: tensor<2x3xf32>, tensor<3x4xf32>)
                     outs(%C: tensor<2x4xf32>) -> tensor<2x4xf32>

  %unranked = tensor.cast %D : tensor<2x4xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 2 offset = 0 sizes = [2, 4] strides = [4, 1] data =
  // CHECK-NEXT: [1038,   1044,   1050,   1056]
  // CHECK-NEXT: [1083,   1098,   1113,   1128]

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [1, 2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
