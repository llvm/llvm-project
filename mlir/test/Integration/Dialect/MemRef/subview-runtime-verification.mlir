// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -expand-strided-metadata \
// RUN:     -lower-affine \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -test-cf-assert \
// RUN:     -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

func.func @subview(%memref: memref<1xf32>, %offset: index) {
    memref.subview %memref[%offset] [1] [1] : 
        memref<1xf32> to 
        memref<1xf32, strided<[1], offset: ?>>
    return
}

func.func @subview_dynamic(%memref: memref<?x4xf32>, %offset: index, %size: index, %stride: index) {
    memref.subview %memref[%offset, 0] [%size, 4] [%stride, 1] : 
        memref<?x4xf32> to 
        memref<?x4xf32, strided<[?, 1], offset: ?>>
    return
}

func.func @subview_dynamic_rank_reduce(%memref: memref<?x4xf32>, %offset: index, %size: index, %stride: index) {
    memref.subview %memref[%offset, 0] [%size, 1] [%stride, 1] :
        memref<?x4xf32> to
        memref<?xf32, strided<[?], offset: ?>>
    return
}

func.func @main() {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %n1 = arith.constant -1 : index
  %4 = arith.constant 4 : index
  %5 = arith.constant 5 : index

  %alloca = memref.alloca() : memref<1xf32>
  %alloca_4 = memref.alloca() : memref<4x4xf32>
  %alloca_4_dyn = memref.cast %alloca_4 : memref<4x4xf32> to memref<?x4xf32>

  // Offset is out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.subview"
  // CHECK-NEXT: ^ subview is out-of-bounds of the base memref
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @subview_dynamic_rank_reduce(%alloca_4_dyn, %5, %5, %1) : (memref<?x4xf32>, index, index, index) -> ()

  // Offset is out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.subview"
  // CHECK-NEXT: ^ subview is out-of-bounds of the base memref
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @subview(%alloca, %1) : (memref<1xf32>, index) -> ()

  // Offset is out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.subview"
  // CHECK-NEXT: ^ subview is out-of-bounds of the base memref
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @subview(%alloca, %n1) : (memref<1xf32>, index) -> ()

  // Size is out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.subview"
  // CHECK-NEXT: ^ subview is out-of-bounds of the base memref
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @subview_dynamic(%alloca_4_dyn, %0, %5, %1) : (memref<?x4xf32>, index, index, index) -> ()

  // Stride is out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: "memref.subview"
  // CHECK-NEXT: ^ subview is out-of-bounds of the base memref
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @subview_dynamic(%alloca_4_dyn, %0, %4, %4) : (memref<?x4xf32>, index, index, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @subview(%alloca, %0) : (memref<1xf32>, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @subview_dynamic(%alloca_4_dyn, %0, %4, %1) : (memref<?x4xf32>, index, index, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @subview_dynamic_rank_reduce(%alloca_4_dyn, %0, %1, %0) : (memref<?x4xf32>, index, index, index) -> ()


  return
}
