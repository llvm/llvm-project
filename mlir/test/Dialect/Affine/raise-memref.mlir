// RUN: mlir-opt %s -allow-unregistered-dialect -affine-raise-from-memref --canonicalize | FileCheck %s

// CHECK-LABEL:    func @reduce_window_max(
func.func @reduce_window_max() {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<1x8x8x64xf32>
  %1 = memref.alloc() : memref<1x18x18x64xf32>
  affine.for %arg0 = 0 to 1 {
    affine.for %arg1 = 0 to 8 {
      affine.for %arg2 = 0 to 8 {
        affine.for %arg3 = 0 to 64 {
          memref.store %cst, %0[%arg0, %arg1, %arg2, %arg3] : memref<1x8x8x64xf32>
        }
      }
    }
  }
  affine.for %arg0 = 0 to 1 {
    affine.for %arg1 = 0 to 8 {
      affine.for %arg2 = 0 to 8 {
        affine.for %arg3 = 0 to 64 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 3 {
              affine.for %arg6 = 0 to 3 {
                affine.for %arg7 = 0 to 1 {
                  %2 = memref.load %0[%arg0, %arg1, %arg2, %arg3] : memref<1x8x8x64xf32>
                  %21 = arith.addi %arg0, %arg4 : index
                  %22 = arith.constant 2 : index
                  %23 = arith.muli %arg1, %22 : index
                  %24 = arith.addi %23, %arg5 : index
                  %25 = arith.muli %arg2, %22 : index
                  %26 = arith.addi %25, %arg6 : index
                  %27 = arith.addi %arg3, %arg7 : index
                  %3 = memref.load %1[%21, %24, %26, %27] : memref<1x18x18x64xf32>
                  %4 = arith.cmpf ogt, %2, %3 : f32
                  %5 = arith.select %4, %2, %3 : f32
                  memref.store %5, %0[%arg0, %arg1, %arg2, %arg3] : memref<1x8x8x64xf32>
                }
              }
            }
          }
        }
      }
    }
  }
  return
}

// CHECK:        %[[cst:.*]] = arith.constant 0
// CHECK:        %[[v0:.*]] = memref.alloc() : memref<1x8x8x64xf32>
// CHECK:        %[[v1:.*]] = memref.alloc() : memref<1x18x18x64xf32>
// CHECK:        affine.for %[[arg0:.*]] =
// CHECK:          affine.for %[[arg1:.*]] =
// CHECK:            affine.for %[[arg2:.*]] =
// CHECK:              affine.for %[[arg3:.*]] =
// CHECK:                affine.store %[[cst]], %[[v0]][%[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]]] :
// CHECK:        affine.for %[[a0:.*]] =
// CHECK:          affine.for %[[a1:.*]] =
// CHECK:            affine.for %[[a2:.*]] =
// CHECK:              affine.for %[[a3:.*]] =
// CHECK:                affine.for %[[a4:.*]] =
// CHECK:                  affine.for %[[a5:.*]] =
// CHECK:                    affine.for %[[a6:.*]] =
// CHECK:                      affine.for %[[a7:.*]] =
// CHECK:                        %[[lhs:.*]] = affine.load %[[v0]][%[[a0]], %[[a1]], %[[a2]], %[[a3]]] :
// CHECK:                        %[[rhs:.*]] = affine.load %[[v1]][%[[a0]] + %[[a4]], %[[a1]] * 2 + %[[a5]], %[[a2]] * 2 + %[[a6]], %[[a3]] + %[[a7]]] :
// CHECK:                        %[[res:.*]] = arith.cmpf ogt, %[[lhs]], %[[rhs]] : f32
// CHECK:                        %[[sel:.*]] = arith.select %[[res]], %[[lhs]], %[[rhs]] : f32
// CHECK:                        affine.store %[[sel]], %[[v0]][%[[a0]], %[[a1]], %[[a2]], %[[a3]]] :

// CHECK-LABEL:    func @symbols(
func.func @symbols(%N : index) {
  %0 = memref.alloc() : memref<1024x1024xf32>
  %1 = memref.alloc() : memref<1024x1024xf32>
  %2 = memref.alloc() : memref<1024x1024xf32>
  %cst1 = arith.constant 1 : index
  %cst2 = arith.constant 2 : index
  affine.for %i = 0 to %N {
    affine.for %j = 0 to %N {
      %7 = memref.load %2[%i, %j] : memref<1024x1024xf32>
      %10 = affine.for %k = 0 to %N iter_args(%ax = %cst1) -> index {
        %12 = arith.muli %N, %cst2 : index
        %13 = arith.addi %12, %cst1 : index
        %14 = arith.addi %13, %j : index
        %5 = memref.load %0[%i, %12] : memref<1024x1024xf32>
        %6 = memref.load %1[%14, %j] : memref<1024x1024xf32>
        %8 = arith.mulf %5, %6 : f32
        %9 = arith.addf %7, %8 : f32
        %4 = arith.addi %N, %cst1 : index
        %11 = arith.addi %ax, %cst1 : index
        memref.store %9, %2[%i, %4] : memref<1024x1024xf32> // this uses an expression of the symbol
        memref.store %9, %2[%i, %11] : memref<1024x1024xf32> // this uses an iter_args and cannot be raised
        %something = "ab.v"() : () -> index
        memref.store %9, %2[%i, %something] : memref<1024x1024xf32> // this cannot be raised
        affine.yield %11 : index
      }
    }
  }
  return
}

// CHECK:          %[[cst1:.*]] = arith.constant 1 : index
// CHECK:          %[[v0:.*]] = memref.alloc() : memref<
// CHECK:          %[[v1:.*]] = memref.alloc() : memref<
// CHECK:          %[[v2:.*]] = memref.alloc() : memref<
// CHECK:          affine.for %[[a1:.*]] = 0 to %arg0 {
// CHECK:             affine.for %[[a2:.*]] = 0 to %arg0 {
// CHECK:                %[[lhs:.*]] = affine.load %{{.*}}[%[[a1]], %[[a2]]] : memref<1024x1024xf32>
// CHECK:                affine.for %[[a3:.*]] = 0 to %arg0 iter_args(%[[a4:.*]] = %[[cst1]]) -> (index) {
// CHECK:                  %[[lhs2:.*]] = affine.load %{{.*}}[%[[a1]], symbol(%arg0) * 2] :
// CHECK:                  %[[lhs3:.*]] = affine.load %{{.*}}[%[[a2]] + symbol(%arg0) * 2 + 1, %[[a2]]] :
// CHECK:                  %[[lhs4:.*]] = arith.mulf %[[lhs2]], %[[lhs3]]
// CHECK:                  %[[lhs5:.*]] = arith.addf %[[lhs]], %[[lhs4]]
// CHECK:                  %[[lhs6:.*]] = arith.addi %[[a4]], %[[cst1]]
// CHECK:                  affine.store %[[lhs5]], %{{.*}}[%[[a1]], symbol(%arg0) + 1] :
// CHECK:                  memref.store %[[lhs5]], %{{.*}}[%[[a1]], %[[lhs6]]] :
// CHECK:                  %[[lhs7:.*]] = "ab.v"
// CHECK:                  memref.store %[[lhs5]], %{{.*}}[%[[a1]], %[[lhs7]]] :
// CHECK:                  affine.yield %[[lhs6]]


// CHECK-LABEL:    func @non_affine(
func.func @non_affine(%N : index) {
  %2 = memref.alloc() : memref<1024x1024xf32>
  affine.for %i = 0 to %N {
    affine.for %j = 0 to %N {
      %ij = arith.muli %i, %j : index
      %7 = memref.load %2[%i, %ij] : memref<1024x1024xf32>
      memref.store %7, %2[%ij, %ij] : memref<1024x1024xf32>
    }
  }
  return
}

// CHECK:          affine.for %[[i:.*]] =
// CHECK:             affine.for %[[j:.*]] =
// CHECK:                  %[[ij:.*]] = arith.muli %[[i]], %[[j]]
// CHECK:                  %[[v:.*]] = memref.load %{{.*}}[%[[i]], %[[ij]]]
// CHECK:                  memref.store %[[v]], %{{.*}}[%[[ij]], %[[ij]]]
