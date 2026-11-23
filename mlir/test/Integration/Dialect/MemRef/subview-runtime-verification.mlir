// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -expand-strided-metadata \
// RUN:     -lower-affine \
// RUN:     -test-cf-assert \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-to-llvm | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -expand-strided-metadata \
// RUN:     -lower-affine \
// RUN:     -test-cf-assert \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-to-llvm="allow-pattern-rollback=0" \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
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

func.func @subview_zero_size_dim(%memref: memref<10x4x1xf32, strided<[?, ?, ?], offset: ?>>, 
                                 %dim_0: index, 
                                 %dim_1: index, 
                                 %dim_2: index) {
    %subview = memref.subview %memref[0, 0, 0] [%dim_0, %dim_1, %dim_2] [1, 1, 1] :
        memref<10x4x1xf32, strided<[?, ?, ?], offset: ?>> to
        memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
    return
}

func.func @subview_with_empty_slice(%memref: memref<10x4x1xf32, strided<[?, ?, ?], offset: ?>>, 
                                 %dim_0: index, 
                                 %dim_1: index, 
                                 %dim_2: index,
                                 %offset: index) {
    %subview = memref.subview %memref[%offset, 0, 0] [%dim_0, %dim_1, %dim_2] [1, 1, 1] :
        memref<10x4x1xf32, strided<[?, ?, ?], offset: ?>> to
        memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
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

  // Offset is out-of-bounds and slice runs out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, 1] [%{{.*}}, 1] : memref<?x4xf32> to memref<?xf32, strided<[?], offset: ?>>
  // CHECK-NEXT: ^ offset 0 is out-of-bounds
  // CHECK-NEXT: Location: loc({{.*}})
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, 1] [%{{.*}}, 1] : memref<?x4xf32> to memref<?xf32, strided<[?], offset: ?>>
  // CHECK-NEXT: ^ subview runs out-of-bounds along dimension 0
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @subview_dynamic_rank_reduce(%alloca_4_dyn, %5, %5, %1) : (memref<?x4xf32>, index, index, index) -> ()

  // Offset is out-of-bounds and slice runs out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: memref.subview %{{.*}}[%{{.*}}] [1] [1] : memref<1xf32> to memref<1xf32, strided<[1], offset: ?>>
  // CHECK-NEXT: ^ offset 0 is out-of-bounds
  // CHECK-NEXT: Location: loc({{.*}})
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: memref.subview %{{.*}}[%{{.*}}] [1] [1] : memref<1xf32> to memref<1xf32, strided<[1], offset: ?>>
  // CHECK-NEXT: ^ subview runs out-of-bounds along dimension 0
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @subview(%alloca, %1) : (memref<1xf32>, index) -> ()

  // Offset is out-of-bounds and slice runs out-of-bounds
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: memref.subview %{{.*}}[%{{.*}}] [1] [1] : memref<1xf32> to memref<1xf32, strided<[1], offset: ?>>
  // CHECK-NEXT: ^ offset 0 is out-of-bounds
  // CHECK-NEXT: Location: loc({{.*}})
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: memref.subview %{{.*}}[%{{.*}}] [1] [1] : memref<1xf32> to memref<1xf32, strided<[1], offset: ?>>
  // CHECK-NEXT: ^ subview runs out-of-bounds along dimension 0
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @subview(%alloca, %n1) : (memref<1xf32>, index) -> ()

  // Slice runs out-of-bounds due to size
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, 4] [%{{.*}}, 1] : memref<?x4xf32> to memref<?x4xf32, strided<[?, 1], offset: ?>>
  // CHECK-NEXT: ^ subview runs out-of-bounds along dimension 0
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @subview_dynamic(%alloca_4_dyn, %0, %5, %1) : (memref<?x4xf32>, index, index, index) -> ()

  // Slice runs out-of-bounds due to stride
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, 4] [%{{.*}}, 1] : memref<?x4xf32> to memref<?x4xf32, strided<[?, 1], offset: ?>>
  // CHECK-NEXT: ^ subview runs out-of-bounds along dimension 0
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @subview_dynamic(%alloca_4_dyn, %0, %4, %4) : (memref<?x4xf32>, index, index, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @subview(%alloca, %0) : (memref<1xf32>, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @subview_dynamic(%alloca_4_dyn, %0, %4, %1) : (memref<?x4xf32>, index, index, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @subview_dynamic_rank_reduce(%alloca_4_dyn, %0, %1, %0) : (memref<?x4xf32>, index, index, index) -> ()

  %alloca_10x4x1 = memref.alloca() : memref<10x4x1xf32>
  %alloca_10x4x1_dyn_stride = memref.cast %alloca_10x4x1 : memref<10x4x1xf32> to memref<10x4x1xf32, strided<[?, ?, ?], offset: ?>>
  // CHECK-NOT: ERROR: Runtime op verification failed
  %dim_0 = arith.constant 0 : index
  %dim_1 = arith.constant 4 : index
  %dim_2 = arith.constant 1 : index
  func.call @subview_zero_size_dim(%alloca_10x4x1_dyn_stride, %dim_0, %dim_1, %dim_2)
                                        : (memref<10x4x1xf32, strided<[?, ?, ?], offset: ?>>, index, index, index) -> ()

  // CHECK-NOT: ERROR: Runtime op verification failed
  %offset = arith.constant 10 : index
  func.call @subview_with_empty_slice(%alloca_10x4x1_dyn_stride, %dim_0, %dim_1, %dim_2, %offset)
                                        : (memref<10x4x1xf32, strided<[?, ?, ?], offset: ?>>, index, index, index, index) -> ()
  return
}
