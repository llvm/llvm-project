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

func.func @extract(%tensor: tensor<1xf32>, %index: index) {
    tensor.extract %tensor[%index] :  tensor<1xf32>
    return
}

func.func @extract_dynamic(%tensor: tensor<?xf32>, %index: index) {
    tensor.extract %tensor[%index] :  tensor<?xf32>
    return
}

func.func @extract_nd_dynamic(%tensor: tensor<?x?x?xf32>, %index0: index, %index1: index, %index2: index) {
    tensor.extract %tensor[%index0, %index1, %index2] :  tensor<?x?x?xf32>
    return
}

func.func @main() {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %n1 = arith.constant -1 : index
  %2 = arith.constant 2 : index
  %alloca_1 = tensor.empty() : tensor<1xf32>
  %alloc_1 = tensor.empty(%1) : tensor<?xf32>
  %alloc_2x2x2 = tensor.empty(%2, %2, %2) : tensor<?x?x?xf32>

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.extract %{{.*}}[%{{.*}}] : tensor<1xf32>
  // CHECK-NEXT: ^ out-of-bounds access
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @extract(%alloca_1, %1) : (tensor<1xf32>, index) -> ()

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.extract %{{.*}}[%{{.*}}] : tensor<?xf32>
  // CHECK-NEXT: ^ out-of-bounds access
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @extract_dynamic(%alloc_1, %1) : (tensor<?xf32>, index) -> ()

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.extract %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : tensor<?x?x?xf32>
  // CHECK-NEXT: ^ out-of-bounds access
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @extract_nd_dynamic(%alloc_2x2x2, %1, %n1, %0) : (tensor<?x?x?xf32>, index, index, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @extract(%alloca_1, %0) : (tensor<1xf32>, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @extract_dynamic(%alloc_1, %0) : (tensor<?xf32>, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @extract_nd_dynamic(%alloc_2x2x2, %1, %1, %0) : (tensor<?x?x?xf32>, index, index, index) -> ()

  return
}

