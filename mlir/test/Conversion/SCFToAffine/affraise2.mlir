// RUN: mlir-opt --affine-cfg --raise-scf-to-affine %s | FileCheck %s

module {
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %19 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.if %12 {
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.muli %15, %c4 : index
      %17 = arith.divui %16, %c4 : index
      scf.for %arg2 = %c0 to %17 step %c1 {
        %20 = memref.load %19[%arg2] : memref<?xf32>
        memref.store %20, %18[%arg2] : memref<?xf32>
      }
    }
    return
  }
}

// CHECK:   func.func @main(%[[arg0:.+]]: i1, %[[arg1:.+]]: i32, %[[arg2:.+]]: memref<?xf32>, %[[arg3:.+]]: memref<?xf32>) {
// CHECK-NEXT:     %[[c4:.+]] = arith.constant 4 : index
// CHECK-NEXT:     %[[V0:.+]] = arith.index_cast %[[arg1]] : i32 to index
// CHECK-NEXT:     %[[V1:.+]] = arith.muli %[[V0]], %[[c4]] : index
// CHECK-NEXT:     scf.if %[[arg0]] {
// CHECK-NEXT:       %[[V2:.+]] = arith.divui %[[V1]], %[[c4]] : index
// CHECK-NEXT:       affine.for %[[arg4:.+]] = 0 to %[[V2]] {
// CHECK-NEXT:         %[[a:.+]] = memref.load %[[arg3]][%[[arg4]]] : memref<?xf32>
// CHECK-NEXT:         memref.store %[[a]], %[[arg2]][%[[arg4]]] : memref<?xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
