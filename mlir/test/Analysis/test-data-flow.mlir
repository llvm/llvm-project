// RUN: mlir-opt -test-data-flow --allow-unregistered-dialect %s 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "loop-arg-pessimistic"
module attributes {test.name = "loop-arg-pessimistic"} {
  func.func @f() -> index {
    // CHECK: Visiting : %{{.*}} = arith.constant 0
    // CHECK-NEXT: Result 0 moved from uninitialized to 1
    %c0 = arith.constant 0 : index
    // CHECK: Visiting : %{{.*}} = arith.constant 1
    // CHECK-NEXT: Result 0 moved from uninitialized to 1
    %c1 = arith.constant 1 : index
    // CHECK: Visiting region branch op : %{{.*}} = scf.for
    // CHECK: Block argument 0 moved from uninitialized to 1
    %0 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0) -> index {
      // CHECK: Visiting : %{{.*}} = arith.addi %{{.*}}, %{{.*}}
      // CHECK-NEXT: Arg 0 : 1
      // CHECK-NEXT: Arg 1 : 1
      // CHECK-NEXT: Result 0 moved from uninitialized to 1
      %10 = arith.addi %arg1, %arg2 : index
      scf.yield %10 : index
    }
    return %0 : index
  }
}
