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

func.func @cast_to_static_dim(%m: memref<?xf32>) -> memref<10xf32> {
  %0 = memref.cast %m : memref<?xf32> to memref<10xf32>
  return %0 : memref<10xf32>
}

func.func @cast_to_ranked(%m: memref<*xf32>) -> memref<f32> {
  %0 = memref.cast %m : memref<*xf32> to memref<f32>
  return %0 : memref<f32>
}

func.func @cast_to_static_strides(%m: memref<?xf32, strided<[?], offset: ?>>)
    -> memref<?xf32, strided<[9], offset: 5>> {
  %0 = memref.cast %m : memref<?xf32, strided<[?], offset: ?>>
                     to memref<?xf32, strided<[9], offset: 5>>
  return %0 : memref<?xf32, strided<[9], offset: 5>>
}

func.func @valid_cast(%m: memref<*xf32>) -> memref<?xf32> {
  %0 = memref.cast %m : memref<*xf32> to memref<?xf32>
  return %0 : memref<?xf32>
}

func.func @main() {
  // All casts inside the called functions are invalid at runtime, except for
  // the last one.
  %alloc = memref.alloc() : memref<5xf32>

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.cast"(%{{.*}}) : (memref<?xf32>) -> memref<10xf32>
  // CHECK-NEXT: ^ size mismatch of dim 0
  // CHECK-NEXT: Location: loc({{.*}})
  %1 = memref.cast %alloc : memref<5xf32> to memref<?xf32>
  func.call @cast_to_static_dim(%1) : (memref<?xf32>) -> (memref<10xf32>)

  // CHECK-NEXT: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.cast"(%{{.*}}) : (memref<*xf32>) -> memref<f32>
  // CHECK-NEXT: ^ rank mismatch
  // CHECK-NEXT: Location: loc({{.*}})
  %3 = memref.cast %alloc : memref<5xf32> to memref<*xf32>
  func.call @cast_to_ranked(%3) : (memref<*xf32>) -> (memref<f32>)

  // CHECK-NEXT: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.cast"(%{{.*}}) : (memref<?xf32, strided<[?], offset: ?>>) -> memref<?xf32, strided<[9], offset: 5>>
  // CHECK-NEXT: ^ offset mismatch
  // CHECK-NEXT: Location: loc({{.*}})

  // CHECK-NEXT: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.cast"(%{{.*}}) : (memref<?xf32, strided<[?], offset: ?>>) -> memref<?xf32, strided<[9], offset: 5>>
  // CHECK-NEXT: ^ stride mismatch of dim 0
  // CHECK-NEXT: Location: loc({{.*}})
  %4 = memref.cast %alloc
      : memref<5xf32> to memref<?xf32, strided<[?], offset: ?>>
  func.call @cast_to_static_strides(%4)
      : (memref<?xf32, strided<[?], offset: ?>>)
     -> (memref<?xf32, strided<[9], offset: 5>>)

  // A last cast that actually succeeds.
  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @valid_cast(%3) : (memref<*xf32>) -> (memref<?xf32>)

  memref.dealloc %alloc : memref<5xf32>

  return
}
