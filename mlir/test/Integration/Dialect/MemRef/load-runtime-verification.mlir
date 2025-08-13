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

func.func @load(%memref: memref<1xf32>, %index: index) {
    memref.load %memref[%index] :  memref<1xf32>
    return
}

func.func @load_dynamic(%memref: memref<?xf32>, %index: index) {
    memref.load %memref[%index] :  memref<?xf32>
    return
}

func.func @load_nd_dynamic(%memref: memref<?x?x?xf32>, %index0: index, %index1: index, %index2: index) {
    memref.load %memref[%index0, %index1, %index2] :  memref<?x?x?xf32>
    return
}

func.func @main() {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %n1 = arith.constant -1 : index
  %2 = arith.constant 2 : index
  %alloca_1 = memref.alloca() : memref<1xf32>
  %alloc_1 = memref.alloc(%1) : memref<?xf32>
  %alloc_2x2x2 = memref.alloc(%2, %2, %2) : memref<?x?x?xf32>

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.load"(%{{.*}}, %{{.*}}) : (memref<1xf32>, index) -> f32
  // CHECK-NEXT: ^ out-of-bounds access
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @load(%alloca_1, %1) : (memref<1xf32>, index) -> ()

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.load"(%{{.*}}, %{{.*}}) : (memref<?xf32>, index) -> f32
  // CHECK-NEXT: ^ out-of-bounds access
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @load_dynamic(%alloc_1, %1) : (memref<?xf32>, index) -> ()

  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.load"(%{{.*}}, %{{.*}}) : (memref<?x?x?xf32>, index, index, index) -> f32
  // CHECK-NEXT: ^ out-of-bounds access
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @load_nd_dynamic(%alloc_2x2x2, %1, %n1, %0) : (memref<?x?x?xf32>, index, index, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @load(%alloca_1, %0) : (memref<1xf32>, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @load_dynamic(%alloc_1, %0) : (memref<?xf32>, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @load_nd_dynamic(%alloc_2x2x2, %1, %1, %0) : (memref<?x?x?xf32>, index, index, index) -> ()

  memref.dealloc %alloc_1 : memref<?xf32>
  memref.dealloc %alloc_2x2x2 : memref<?x?x?xf32>

  return
}

