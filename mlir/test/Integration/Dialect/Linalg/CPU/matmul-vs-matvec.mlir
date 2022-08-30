// RUN: mlir-opt %s -convert-linalg-to-loops -convert-scf-to-cf -convert-linalg-to-llvm -convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>)

func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>) -> (memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32
  %x = memref.dim %A, %c0 : memref<?x?xf32>
  %y = memref.dim %B, %c1 : memref<?x?xf32>
  %C = memref.alloc(%x, %y) : memref<?x?xf32>
  linalg.fill ins(%f0 : f32) outs(%C : memref<?x?xf32>)
  linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
                outs(%C: memref<?x?xf32>)
  return %C : memref<?x?xf32>
}

func.func @matvec(%A: memref<?x?xf32>, %B: memref<?x?xf32>) -> (memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32
  %m = memref.dim %A, %c0 : memref<?x?xf32>
  %x = memref.dim %A, %c1 : memref<?x?xf32>
  %n = memref.dim %B, %c1 : memref<?x?xf32>
  %C = memref.alloc(%m, %n) : memref<?x?xf32>
  linalg.fill ins(%f0 : f32) outs(%C : memref<?x?xf32>)
  scf.for %i = %c0 to %n step %c1 {
    %b = memref.subview %B[0, %i][%x, 1][1, 1] : memref<?x?xf32> to memref<?xf32, strided<[?], offset: ?>>
    %c = memref.subview %C[0, %i][%m, 1][1, 1] : memref<?x?xf32> to memref<?xf32, strided<[?], offset: ?>>
    linalg.matvec ins(%A, %b: memref<?x?xf32>, memref<?xf32, strided<[?], offset: ?>>)
                  outs(%c: memref<?xf32, strided<[?], offset: ?>>)
  }
  return %C : memref<?x?xf32>
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = arith.constant 5 : index
  %x = arith.constant 3 : index
  %n = arith.constant 2 : index
  %val1 = arith.constant 13.0 : f32
  %val2 = arith.constant 17.0 : f32
  %A = memref.alloc(%m, %x) : memref<?x?xf32>
  %B = memref.alloc(%x, %n) : memref<?x?xf32>
  linalg.fill ins(%val1 : f32) outs(%A : memref<?x?xf32>)
  linalg.fill ins(%val2 : f32) outs(%B : memref<?x?xf32>)
  memref.store %val1, %B[%c0, %c0] : memref<?x?xf32>
  %C1 = call @matmul(%A, %B) : (memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
  %C2 = call @matvec(%A, %B) : (memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
  scf.for %i = %c0 to %m step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %e1 = memref.load %C1[%i, %j] : memref<?x?xf32>
      %e2 = memref.load %C2[%i, %j] : memref<?x?xf32>
      %c = arith.cmpf oeq, %e1, %e2 : f32
      cf.assert %c, "Matmul does not produce same output as matvec"
    }
  }
  %C2_ = memref.cast %C2 : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%C2_) : (memref<*xf32>) -> ()
  memref.dealloc %C1 : memref<?x?xf32>
  memref.dealloc %C2 : memref<?x?xf32>
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [5, 2] strides = [2, 1] data =
// CHECK-NEXT:      [
// CHECK-SAME:  [611,   663],
// CHECK-NEXT:  [611,   663],
// CHECK-NEXT:  [611,   663],
// CHECK-NEXT:  [611,   663],
// CHECK-NEXT:  [611,   663]
// CHECK-SAME: ]
