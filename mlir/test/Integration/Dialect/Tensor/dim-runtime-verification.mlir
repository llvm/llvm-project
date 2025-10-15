// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -buffer-deallocation-pipeline=private-function-dynamic-ownership \
// RUN:     -test-cf-assert \
// RUN:     -convert-to-llvm | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%tlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -buffer-deallocation-pipeline=private-function-dynamic-ownership \
// RUN:     -test-cf-assert \
// RUN:     -convert-to-llvm="allow-pattern-rollback=0" \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%tlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

func.func @main() {
  %c4 = arith.constant 4 : index
  %tensor = tensor.empty() : tensor<1xf32>

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.dim %{{.*}}, %{{.*}} : tensor<1xf32>
  // CHECK-NEXT: ^ index is out of bounds
  // CHECK-NEXT: Location: loc({{.*}})
  %dim = tensor.dim %tensor, %c4 : tensor<1xf32>

  return
}
