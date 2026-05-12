// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -buffer-deallocation-pipeline=private-function-dynamic-ownership \
// RUN:     -test-cf-assert \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-to-llvm | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%tlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -buffer-deallocation-pipeline=private-function-dynamic-ownership \
// RUN:     -test-cf-assert \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-to-llvm="allow-pattern-rollback=0" \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%tlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

func.func private @cast_to_static_dim(%t: tensor<?xf32>) -> tensor<10xf32> {
  %0 = tensor.cast %t : tensor<?xf32> to tensor<10xf32>
  return %0 : tensor<10xf32>
}

func.func private @cast_to_ranked(%t: tensor<*xf32>) -> tensor<f32> {
  %0 = tensor.cast %t : tensor<*xf32> to tensor<f32>
  return %0 : tensor<f32>
}

func.func private @valid_cast(%t: tensor<*xf32>) -> tensor<?xf32> {
  %0 = tensor.cast %t : tensor<*xf32> to tensor<?xf32>
  return %0 : tensor<?xf32>
}

func.func @main() {
  // All casts inside the called functions are invalid at runtime, except for
  // the last one.
  %alloc = tensor.empty() : tensor<5xf32>

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.cast %{{.*}} : tensor<?xf32> to tensor<10xf32>
  // CHECK-NEXT: ^ size mismatch of dim 0
  // CHECK-NEXT: Location: loc({{.*}})
  %1 = tensor.cast %alloc : tensor<5xf32> to tensor<?xf32>
  func.call @cast_to_static_dim(%1) : (tensor<?xf32>) -> (tensor<10xf32>)

  // CHECK-NEXT: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.cast %{{.*}} : tensor<*xf32> to tensor<f32>
  // CHECK-NEXT: ^ rank mismatch
  // CHECK-NEXT: Location: loc({{.*}})
  %3 = tensor.cast %alloc : tensor<5xf32> to tensor<*xf32>
  func.call @cast_to_ranked(%3) : (tensor<*xf32>) -> (tensor<f32>)

  // A last cast that actually succeeds.
  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @valid_cast(%3) : (tensor<*xf32>) -> (tensor<?xf32>)

  return
}
