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

func.func private @reshape(%x: tensor<?xf32>, %n: index, %m: index) -> tensor<?x?xf32> {
  %shape = tensor.from_elements %n, %m : tensor<2xindex>
  %0 = tensor.reshape %x(%shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

func.func private @reshape_static_result(%x: tensor<?xf32>, %n: index, %m: index) -> tensor<6x?xf32> {
  %shape = tensor.from_elements %n, %m : tensor<2xindex>
  %0 = tensor.reshape %x(%shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<6x?xf32>
  return %0 : tensor<6x?xf32>
}

func.func @main() {
  %x = tensor.empty() : tensor<6xf32>
  %x_dyn = tensor.cast %x : tensor<6xf32> to tensor<?xf32>

  // Product 4*4=16 does not equal source size 6.
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.reshape %{{.*}}(%{{.*}}) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-NEXT: ^ product of shape operand values does not equal number of elements in source
  // CHECK-NEXT: Location: loc({{.*}})
  %c4 = arith.constant 4 : index
  %r1 = func.call @reshape(%x_dyn, %c4, %c4) : (tensor<?xf32>, index, index) -> tensor<?x?xf32>

  // Shape value 3 does not match static result size 6 (product 3*2=6 still
  // matches the source, so only the static-size assert fires).
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.reshape %{{.*}}(%{{.*}}) : (tensor<?xf32>, tensor<2xindex>) -> tensor<6x?xf32>
  // CHECK-NEXT: ^ shape operand value does not match static result size along dimension 0
  // CHECK-NEXT: Location: loc({{.*}})
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %r2 = func.call @reshape_static_result(%x_dyn, %c3, %c2) : (tensor<?xf32>, index, index) -> tensor<6x?xf32>

  // Product 3*2=6 equals source size 6. No error.
  // CHECK-NOT: ERROR: Runtime op verification failed
  %r3 = func.call @reshape(%x_dyn, %c3, %c2) : (tensor<?xf32>, index, index) -> tensor<?x?xf32>

  // Product 6*1=6 equals source size 6, and shape value 6 matches the static
  // result size along dimension 0. No error.
  // CHECK-NOT: ERROR: Runtime op verification failed
  %c6 = arith.constant 6 : index
  %c1 = arith.constant 1 : index
  %r4 = func.call @reshape_static_result(%x_dyn, %c6, %c1) : (tensor<?xf32>, index, index) -> tensor<6x?xf32>

  return
}
