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

// Put memref.copy in a function, otherwise the memref.cast may fold.
func.func @memcpy_helper(%src: memref<?xf32>, %dest: memref<?xf32>) {
  memref.copy %src, %dest : memref<?xf32> to memref<?xf32>
  return
}

func.func @main() {
  %alloca1 = memref.alloca() : memref<4xf32>
  %alloca2 = memref.alloca() : memref<5xf32>
  %cast1 = memref.cast %alloca1 : memref<4xf32> to memref<?xf32>
  %cast2 = memref.cast %alloca2 : memref<5xf32> to memref<?xf32>

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: memref.copy %{{.*}}, %{{.*}} : memref<?xf32> to memref<?xf32>
  // CHECK-NEXT: ^ size of 0-th source/target dim does not match
  // CHECK-NEXT: Location: loc({{.*}})
  call @memcpy_helper(%cast1, %cast2) : (memref<?xf32>, memref<?xf32>) -> ()

  return
}
