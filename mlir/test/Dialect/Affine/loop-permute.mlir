// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=1,2,0" | FileCheck %s --check-prefix=CHECK-120
// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=1,0,2" | FileCheck %s --check-prefix=CHECK-102
// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=0,1,2" | FileCheck %s --check-prefix=CHECK-012
// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=0,2,1" | FileCheck %s --check-prefix=CHECK-021
// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=2,0,1" | FileCheck %s --check-prefix=CHECK-201
// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=2,1,0" | FileCheck %s --check-prefix=CHECK-210
// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=2,1,0 check-validity=1" | FileCheck %s --check-prefix=CHECK-210-VALID

// CHECK-120-LABEL: func @permute
func.func @permute(%U0 : index, %U1 : index, %U2 : index) {
  "abc"() : () -> ()
  affine.for %arg0 = 0 to %U0 {
    affine.for %arg1 = 0 to %U1 {
      affine.for %arg2 = 0 to %U2 {
        "foo"(%arg0, %arg1) : (index, index) -> ()
        "bar"(%arg2) : (index) -> ()
      }
    }
  }
  "xyz"() : () -> ()
  return
}
// CHECK-120:      "abc"
// CHECK-120-NEXT: affine.for
// CHECK-120-NEXT:   affine.for
// CHECK-120-NEXT:     affine.for
// CHECK-120-NEXT:       "foo"(%arg4, %arg5)
// CHECK-120-NEXT:       "bar"(%arg3)
// CHECK-120-NEXT:     }
// CHECK-120-NEXT:   }
// CHECK-120-NEXT: }
// CHECK-120-NEXT: "xyz"
// CHECK-120-NEXT: return

// CHECK-102:      "foo"(%arg4, %arg3)
// CHECK-102-NEXT: "bar"(%arg5)

// CHECK-012:      "foo"(%arg3, %arg4)
// CHECK-012-NEXT: "bar"(%arg5)

// CHECK-021:      "foo"(%arg3, %arg5)
// CHECK-021-NEXT: "bar"(%arg4)

// CHECK-210:      "foo"(%arg5, %arg4)
// CHECK-210-NEXT: "bar"(%arg3)

// CHECK-201:      "foo"(%arg5, %arg3)
// CHECK-201-NEXT: "bar"(%arg4)

// -----

// Tests that the permutation validation check utility conservatively returns false when the
// for loop has an iter_arg.

// CHECK-210-VALID-LABEL: func @check_validity_with_iter_args
// CHECK-210-VALID-SAME:    %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index
func.func @check_validity_with_iter_args(%U0 : index, %U1 : index, %U2 : index) {
  %buf = memref.alloc() : memref<100x100xf32>
  %cst = arith.constant 1.0 : f32
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index

  // Check that the loops are not permuted.
  // CHECK-210-VALID:       affine.for %{{.*}} = 0 to %[[ARG0]] {
  // CHECK-210-VALID-NEXT:    affine.for %{{.*}} = 0 to %[[ARG1]] {
  // CHECK-210-VALID-NEXT:      affine.for %{{.*}} = 0 to %[[ARG2]] iter_args(
  affine.for %arg0 = 0 to %U0 {
    affine.for %arg1 = 0 to %U1 {
      %res = affine.for %arg2 = 0 to %U2 iter_args(%iter1 = %cst) -> (f32) {
        %val = affine.load %buf[%arg0 + 10, %arg1 + 20] : memref<100x100xf32>
        %newVal = arith.addf %val, %cst : f32
        affine.store %newVal, %buf[%arg0 + 10, %arg1 + 20] : memref<100x100xf32>
        %newVal2 = arith.addf %newVal, %iter1 : f32
        affine.yield %iter1 : f32
      }
    }
  }
  return
}
