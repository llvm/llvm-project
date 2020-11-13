// RUN: mlir-opt %s -convert-linalg-to-loops -convert-linalg-to-llvm | \
// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func private @print_memref_f32(memref<*xf32>)

func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>) -> (memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %f0 = constant 0.0 : f32
  %x = dim %A, %c0 : memref<?x?xf32>
  %y = dim %B, %c1 : memref<?x?xf32>
  %C = alloc(%x, %y) : memref<?x?xf32>
  linalg.fill(%C, %f0) : memref<?x?xf32>, f32
  linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
                outs(%C: memref<?x?xf32>)
  return %C : memref<?x?xf32>
}

func @matvec(%A: memref<?x?xf32>, %B: memref<?x?xf32>) -> (memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %f0 = constant 0.0 : f32
  %m = dim %A, %c0 : memref<?x?xf32>
  %x = dim %A, %c1 : memref<?x?xf32>
  %n = dim %B, %c1 : memref<?x?xf32>
  %C = alloc(%m, %n) : memref<?x?xf32>
  linalg.fill(%C, %f0) : memref<?x?xf32>, f32
  scf.for %i = %c0 to %n step %c1 {
    %b = subview %B[0, %i][%x, 1][1, 1] : memref<?x?xf32> to memref<?xf32, offset: ?, strides: [?]>
    %c = subview %C[0, %i][%m, 1][1, 1] : memref<?x?xf32> to memref<?xf32, offset: ?, strides: [?]>
    linalg.matvec ins(%A, %b: memref<?x?xf32>, memref<?xf32, offset: ?, strides: [?]>)
                  outs(%c: memref<?xf32, offset: ?, strides: [?]>)
  }
  return %C : memref<?x?xf32>
}

func @main() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %m = constant 5 : index
  %x = constant 3 : index
  %n = constant 2 : index
  %val1 = constant 13.0 : f32
  %val2 = constant 17.0 : f32
  %A = alloc(%m, %x) : memref<?x?xf32>
  %B = alloc(%x, %n) : memref<?x?xf32>
  linalg.fill(%A, %val1) : memref<?x?xf32>, f32
  linalg.fill(%B, %val2) : memref<?x?xf32>, f32
  store %val1, %B[%c0, %c0] : memref<?x?xf32>
  %C1 = call @matmul(%A, %B) : (memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
  %C2 = call @matvec(%A, %B) : (memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
  scf.for %i = %c0 to %m step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %e1 = load %C1[%i, %j] : memref<?x?xf32>
      %e2 = load %C2[%i, %j] : memref<?x?xf32>
      %c = cmpf "oeq", %e1, %e2 : f32
      assert %c, "Matmul does not produce same output as matvec"
    }
  }
  %C2_ = memref_cast %C2 : memref<?x?xf32> to memref<*xf32>
  call @print_memref_f32(%C2_) : (memref<*xf32>) -> ()
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
