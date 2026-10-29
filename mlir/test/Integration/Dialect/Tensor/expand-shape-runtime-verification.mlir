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

func.func private @expand(%a: tensor<?xf32>, %n: index) -> tensor<?x2xf32> {
  %e = tensor.expand_shape %a [[0, 1]] output_shape [%n, 2] : tensor<?xf32> into tensor<?x2xf32>
  return %e : tensor<?x2xf32>
}

func.func @main() {
  %a = tensor.empty() : tensor<6xf32>
  %a_dyn = tensor.cast %a : tensor<6xf32> to tensor<?xf32>

  // Product 5*2=10 does not equal input dim 6.
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: tensor.expand_shape %{{.*}} {{\[}}[0, 1]{{\]}} output_shape [%{{.*}}, 2] : tensor<?xf32> into tensor<?x2xf32>
  // CHECK-NEXT: ^ product of output dims in reassoc group does not equal input dim
  // CHECK-NEXT: Location: loc({{.*}})
  %n5 = arith.constant 5 : index
  %r1 = func.call @expand(%a_dyn, %n5) : (tensor<?xf32>, index) -> tensor<?x2xf32>

  // Product 3*2=6 equals input dim 6. No error.
  // CHECK-NOT: ERROR: Runtime op verification failed
  %n3 = arith.constant 3 : index
  %r2 = func.call @expand(%a_dyn, %n3) : (tensor<?xf32>, index) -> tensor<?x2xf32>

  return
}