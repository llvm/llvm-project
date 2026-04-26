// RUN: mlir-opt %s -inline='default-pipeline=' | FileCheck %s
// RUN: mlir-opt %s --mlir-disable-threading -inline='default-pipeline=' | FileCheck %s

module {
  // CHECK-LABEL: func.func @parent1
  func.func @parent1(%arg0: i32) -> i32 {
    // CHECK: call @child
    %0 = call @child(%arg0) : (i32) -> i32
    return %0 : i32
  }

  // CHECK-LABEL: func.func @parent2
  func.func @parent2(%arg0: i32) -> i32 {
    // CHECK: call @child
    %0 = call @child(%arg0) : (i32) -> i32
    return %0 : i32
  }

  // CHECK-LABEL: func.func @child
  func.func @child(%arg0: i32) -> i32 {
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.cmpi sge, %arg0, %c10_i32 : i32
    %1 = scf.if %0 -> (i32) {
      scf.yield %arg0 : i32
    } else {
      %2 = arith.addi %arg0, %c1_i32 : i32
      // CHECK: call @parent1
      // CHECK: call @parent2
      %3 = func.call @parent1(%2) : (i32) -> i32
      %4 = func.call @parent2(%2) : (i32) -> i32
      %5 = arith.addi %3, %4 : i32
      scf.yield %5 : i32
    }
    return %1 : i32
  }
}
