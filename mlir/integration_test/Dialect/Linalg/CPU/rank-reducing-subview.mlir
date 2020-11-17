// RUN: mlir-opt %s -convert-linalg-to-loops -convert-linalg-to-llvm | \
// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func private @print_memref_f32(memref<*xf32>)

func @main() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %f0 = constant 0.0 : f32
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32
  %f3 = constant 3.0 : f32
  %A = alloc(%c2, %c2) : memref<?x?xf32>
  store %f0, %A[%c0, %c0] : memref<?x?xf32>
  store %f1, %A[%c0, %c1] : memref<?x?xf32>
  store %f2, %A[%c1, %c0] : memref<?x?xf32>
  store %f3, %A[%c1, %c1] : memref<?x?xf32>
  %B = subview %A[%c1, 0][1, %c2][1, 1] : memref<?x?xf32> to memref<?xf32, offset: ?, strides: [1]>
  %C = subview %A[0, %c1][%c2, 1][1, 1] : memref<?x?xf32> to memref<?xf32, offset: ?, strides: [?]>
  %A_ = memref_cast %A : memref<?x?xf32> to memref<*xf32>
  call @print_memref_f32(%A_) : (memref<*xf32>) -> ()
  %B_ = memref_cast %B : memref<?xf32, offset: ?, strides: [1]> to memref<*xf32>
  call @print_memref_f32(%B_) : (memref<*xf32>) -> ()
  %C_ = memref_cast %C : memref<?xf32, offset: ?, strides: [?]> to memref<*xf32>
  call @print_memref_f32(%C_) : (memref<*xf32>) -> ()
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
// CHECK-NEXT:      [
// CHECK-SAME:  [0,   1],
// CHECK-NEXT:  [2,   3]
// CHECK-SAME: ]
// CHECK: [2,  3]
// CHECK: [1,  3]
