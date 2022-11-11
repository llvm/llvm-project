// RUN: mlir-opt %s -convert-linalg-to-loops -convert-scf-to-cf -convert-linalg-to-llvm -convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  %A = memref.alloc(%c2, %c2) : memref<?x?xf32>
  memref.store %f0, %A[%c0, %c0] : memref<?x?xf32>
  memref.store %f1, %A[%c0, %c1] : memref<?x?xf32>
  memref.store %f2, %A[%c1, %c0] : memref<?x?xf32>
  memref.store %f3, %A[%c1, %c1] : memref<?x?xf32>
  %B = memref.subview %A[%c1, 0][1, %c2][1, 1] : memref<?x?xf32> to memref<?xf32, strided<[1], offset: ?>>
  %C = memref.subview %A[0, %c1][%c2, 1][1, 1] : memref<?x?xf32> to memref<?xf32, strided<[?], offset: ?>>
  %A_ = memref.cast %A : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%A_) : (memref<*xf32>) -> ()
  %B_ = memref.cast %B : memref<?xf32, strided<[1], offset: ?>> to memref<*xf32>
  call @printMemrefF32(%B_) : (memref<*xf32>) -> ()
  %C_ = memref.cast %C : memref<?xf32, strided<[?], offset: ?>> to memref<*xf32>
  call @printMemrefF32(%C_) : (memref<*xf32>) -> ()
  memref.dealloc %A : memref<?x?xf32>
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
// CHECK-NEXT:      [
// CHECK-SAME:  [0,   1],
// CHECK-NEXT:  [2,   3]
// CHECK-SAME: ]
// CHECK: [2,  3]
// CHECK: [1,  3]
