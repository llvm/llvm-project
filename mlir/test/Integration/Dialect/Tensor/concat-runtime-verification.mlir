// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -buffer-deallocation-pipeline=private-function-dynamic-ownership \
// RUN:     -test-cf-assert \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-to-llvm | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -buffer-deallocation-pipeline=private-function-dynamic-ownership \
// RUN:     -test-cf-assert \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-to-llvm="allow-pattern-rollback=0" \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

func.func private @concat(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c = tensor.concat dim(1) %a, %b : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %c : tensor<?x?xf32>
}

func.func private @concat_sum(%a: tensor<?x3xf32>, %b: tensor<?x3xf32>) -> tensor<5x3xf32> {
  %c = tensor.concat dim(0) %a, %b : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<5x3xf32>
  return %c : tensor<5x3xf32>
}

func.func @main() {
  %x = tensor.empty() : tensor<2x3xf32>
  %y = tensor.empty() : tensor<4x3xf32>
  %z = tensor.empty() : tensor<3x3xf32>
  %a = tensor.cast %x : tensor<2x3xf32> to tensor<?x?xf32>
  %b = tensor.cast %y : tensor<4x3xf32> to tensor<?x?xf32>
  %c = tensor.cast %z : tensor<3x3xf32> to tensor<?x?xf32>
  %a3 = tensor.cast %x : tensor<2x3xf32> to tensor<?x3xf32>
  %b3 = tensor.cast %z : tensor<3x3xf32> to tensor<?x3xf32>
  %y3 = tensor.cast %y : tensor<4x3xf32> to tensor<?x3xf32>

  // Non-concatenated dimension 0 mismatch: 2 vs 4.
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.concat dim(1) %{{.*}}, %{{.*}} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-NEXT: ^ input size mismatch along non-concatenated dimension 0
  // CHECK-NEXT: Location: loc({{.*}})
  %r1 = func.call @concat(%a, %b) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

  // Sum 2+4=6 does not equal static result size 5.
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.concat dim(0) %{{.*}}, %{{.*}} : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<5x3xf32>
  // CHECK-NEXT: ^ sum of input sizes along concatenated dimension does not match result size
  // CHECK-NEXT: Location: loc({{.*}})
  %r2 = func.call @concat_sum(%a3, %y3) : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<5x3xf32>

  // Matching inputs. No error.
  // CHECK-NOT: ERROR: Runtime op verification failed
  %r3 = func.call @concat(%a, %c) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

  // Sum 2+3=5 equals static result size 5. No error.
  // CHECK-NOT: ERROR: Runtime op verification failed
  %r4 = func.call @concat_sum(%a3, %b3) : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<5x3xf32>

  return
}