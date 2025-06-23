// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -one-shot-bufferize \
// RUN:     -buffer-deallocation-pipeline \
// RUN:     -test-cf-assert \
// RUN:     -convert-to-llvm | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

func.func @main() {
  %c4 = arith.constant 4 : index
  %tensor = tensor.empty() : tensor<1xf32>

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "tensor.dim"(%{{.*}}, %{{.*}}) : (tensor<1xf32>, index) -> index
  // CHECK-NEXT: ^ index is out of bounds
  // CHECK-NEXT: Location: loc({{.*}})
  %dim = tensor.dim %tensor, %c4 : tensor<1xf32>

  return
}
