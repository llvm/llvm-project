// RUN: mlir-opt %s -arith-bufferize -linalg-bufferize \
// RUN: -tensor-bufferize -func-bufferize -finalizing-bufferize -buffer-deallocation-pipeline -convert-linalg-to-loops \
// RUN: -convert-arith-to-llvm -convert-scf-to-cf -convert-cf-to-llvm --finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN: | FileCheck %s

func.func @foo() -> tensor<4xf32> {
  %0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @main() {
  %0 = call @foo() : () -> tensor<4xf32>

  // Instead of relying on tensor_store which introduces aliasing, we rely on
  // the conversion of printMemrefF32(tensor<*xf32>) to
  // printMemrefF32(memref<*xf32>).
  // Note that this is skipping a step and we would need at least some function
  // attribute to declare that this conversion is valid (e.g. when we statically
  // know that things will play nicely at the C ABI boundary).
  %unranked = tensor.cast %0 : tensor<4xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 1 offset = 0 sizes = [4] strides = [1] data =
  // CHECK-NEXT: [1, 2, 3, 4]

  return
}

// This gets converted to a function operating on memref<*xf32>.
// Note that this is skipping a step and we would need at least some function
// attribute to declare that this conversion is valid (e.g. when we statically
// know that things will play nicely at the C ABI boundary).
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
