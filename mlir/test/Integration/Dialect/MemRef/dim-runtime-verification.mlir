// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -expand-strided-metadata \
// RUN:     -test-cf-assert \
// RUN:     -convert-to-llvm | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -expand-strided-metadata \
// RUN:     -test-cf-assert \
// RUN:     -convert-to-llvm="allow-pattern-rollback=0" \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

func.func @main() {
  %c4 = arith.constant 4 : index
  %alloca = memref.alloca() : memref<1xf32>

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.dim"(%{{.*}}, %{{.*}}) : (memref<1xf32>, index) -> index
  // CHECK-NEXT: ^ index is out of bounds
  // CHECK-NEXT: Location: loc({{.*}})
  %dim = memref.dim %alloca, %c4 : memref<1xf32>

  return
}
