// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=1,2,0" | FileCheck %s --check-prefix=CHECK-120
// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=1,0,2" | FileCheck %s --check-prefix=CHECK-102
// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=0,1,2" | FileCheck %s --check-prefix=CHECK-012
// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=0,2,1" | FileCheck %s --check-prefix=CHECK-021
// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=2,0,1" | FileCheck %s --check-prefix=CHECK-201
// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-permutation="permutation-map=2,1,0" | FileCheck %s --check-prefix=CHECK-210

// CHECK-120-LABEL: func @permute
func @permute(%U0 : index, %U1 : index, %U2 : index) {
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
