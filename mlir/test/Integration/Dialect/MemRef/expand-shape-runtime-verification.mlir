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

// Dynamic source into dynamic target.
func.func @expand_shape_dynamic(%m: memref<?xf32>, %sz0: index, %sz1: index) {
  %0 = memref.expand_shape %m [[0, 1]] output_shape [%sz0, %sz1]
      : memref<?xf32> into memref<?x?xf32>
  return
}

// Dynamic source into mixed static/dynamic target.
func.func @expand_shape_mixed(%m: memref<?xf32>, %sz0: index) {
  %0 = memref.expand_shape %m [[0, 1]] output_shape [%sz0, 5]
      : memref<?xf32> into memref<?x5xf32>
  return
}

// Multiple reassociation groups: 2D -> 3D.
func.func @expand_shape_multi_group(%m: memref<?x?xf32>, %sz0: index, %sz1: index) {
  %0 = memref.expand_shape %m [[0], [1, 2]] output_shape [%sz0, %sz1, 4]
      : memref<?x?xf32> into memref<?x?x4xf32>
  return
}

func.func @main() {
  %alloca_10 = memref.alloca() : memref<10xf32>
  %alloca_10_dyn = memref.cast %alloca_10 : memref<10xf32> to memref<?xf32>

  %alloca_3x20 = memref.alloca() : memref<3x20xf32>
  %alloca_3x20_dyn = memref.cast %alloca_3x20 : memref<3x20xf32> to memref<?x?xf32>

  %2 = arith.constant 2 : index
  %3 = arith.constant 3 : index
  %4 = arith.constant 4 : index
  %5 = arith.constant 5 : index

  // Product 3*5=15 does not equal input dim 10.
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: memref.expand_shape %{{.*}} {{\[}}[0, 1]{{\]}} output_shape [%{{.*}}, %{{.*}}] : memref<?xf32> into memref<?x?xf32>
  // CHECK-NEXT: ^ product of output dims in reassoc group does not equal input dim
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @expand_shape_dynamic(%alloca_10_dyn, %3, %5)
      : (memref<?xf32>, index, index) -> ()

  // Product 4*5=20 does not equal input dim 10.
  //      CHECK: ERROR: Runtime op verification failed
  // CHECK-NEXT: memref.expand_shape %{{.*}} {{\[}}[0, 1]{{\]}} output_shape [%{{.*}}, 5] : memref<?xf32> into memref<?x5xf32>
  // CHECK-NEXT: ^ product of output dims in reassoc group does not equal input dim
  // CHECK-NEXT: Location: loc({{.*}})
  func.call @expand_shape_mixed(%alloca_10_dyn, %4)
      : (memref<?xf32>, index) -> ()

  // Product 2*5=10 equals input dim 10. No error.
  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @expand_shape_dynamic(%alloca_10_dyn, %2, %5)
      : (memref<?xf32>, index, index) -> ()

  // Product 2*5=10 equals input dim 10. No error.
  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @expand_shape_mixed(%alloca_10_dyn, %2)
      : (memref<?xf32>, index) -> ()

  // Group 0: dim 0 -> [dim 0] product 3, input dim 0 = 3 -> OK.
  // Group 1: dim 1 -> [dim 1, dim 2] product 5*4=20, input dim 1 = 20 -> OK.
  // CHECK-NOT: ERROR: Runtime op verification failed
  func.call @expand_shape_multi_group(%alloca_3x20_dyn, %3, %5)
      : (memref<?x?xf32>, index, index) -> ()

  return
}
