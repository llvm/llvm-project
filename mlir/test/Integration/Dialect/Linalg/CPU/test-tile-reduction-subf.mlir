// Checks that tiling a subtracting accumulation (`acc - x`) computes the same
// result as the untiled reduction. The first RUN line runs the untiled
// reduction, the second one tiles it first; both have to print the same values.

// RUN: mlir-opt %s -test-transform-dialect-erase-schedule \
// RUN: -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries" \
// RUN: -buffer-deallocation-pipeline -convert-bufferization-to-memref -convert-linalg-to-loops -convert-scf-to-cf \
// RUN: -expand-strided-metadata -lower-affine -convert-arith-to-llvm --finalize-memref-to-llvm -convert-func-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -transform-interpreter -test-transform-dialect-erase-schedule \
// RUN: -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries" \
// RUN: -buffer-deallocation-pipeline -convert-bufferization-to-memref -convert-linalg-to-loops -convert-scf-to-cf \
// RUN: -expand-strided-metadata -lower-affine -convert-arith-to-llvm --finalize-memref-to-llvm -convert-func-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %input = arith.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                                 [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]]>
      : tensor<2x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<2xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<2xf32>) -> tensor<2xf32>
  %res = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<2x8xf32>) outs(%init : tensor<2xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %sub = arith.subf %acc, %in : f32
    linalg.yield %sub : f32
  } -> tensor<2xf32>

  %buffer = bufferization.to_buffer %res : tensor<2xf32> to memref<2xf32>
  %cast = memref.cast %buffer : memref<2xf32> to memref<*xf32>
  call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    // A tile size that does not divide the reduction dimension, so that the
    // partial results are unevenly sized.
    %1, %2, %3, %loop = transform.structured.tile_reduction_using_for %0
        by tile_sizes = [0, 3]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                  !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// Both rows reduce to the negated sum of their elements. Merging the partial
// results with a subtraction instead of an addition would print [36, 360].
// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [2] strides = [1] data =
// CHECK-NEXT: [-36,  -360]
