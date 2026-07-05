// RUN: mlir-opt -split-input-file -control-flow-sink %s | FileCheck %s

// CHECK-LABEL: @test_scf_if_sink
// CHECK-SAME:  (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i32)
// CHECK: %[[V0:.*]] = scf.if %[[ARG0]]
// CHECK:   %[[V1:.*]] = arith.addi %[[ARG1]], %[[ARG1]]
// CHECK:   scf.yield %[[V1]]
// CHECK: else
// CHECK:   %[[V1:.*]] = arith.muli %[[ARG1]], %[[ARG1]]
// CHECK:   scf.yield %[[V1]]
// CHECK: return %[[V0]]
func.func @test_scf_if_sink(%arg0: i1, %arg1: i32) -> i32 {
  %0 = arith.addi %arg1, %arg1 : i32
  %1 = arith.muli %arg1, %arg1 : i32
  %result = scf.if %arg0 -> i32 {
    scf.yield %0 : i32
  } else {
    scf.yield %1 : i32
  }
  return %result : i32
}

// -----

func.func private @consume(i32) -> ()

// CHECK-LABEL: @test_scf_if_then_only_sink
// CHECK-SAME:  (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i32)
// CHECK: scf.if %[[ARG0]]
// CHECK:   %[[V0:.*]] = arith.addi %[[ARG1]], %[[ARG1]]
// CHECK:   call @consume(%[[V0]])
func.func @test_scf_if_then_only_sink(%arg0: i1, %arg1: i32) {
  %0 = arith.addi %arg1, %arg1 : i32
  scf.if %arg0 {
    func.call @consume(%0) : (i32) -> ()
    scf.yield
  }
  return
}

// -----

func.func private @consume(i32) -> ()

// CHECK-LABEL: @test_scf_if_double_sink
// CHECK-SAME:  (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i32)
// CHECK: scf.if %[[ARG0]]
// CHECK:   scf.if %[[ARG0]]
// CHECK:     %[[V0:.*]] = arith.addi %[[ARG1]], %[[ARG1]]
// CHECK:     call @consume(%[[V0]])
func.func @test_scf_if_double_sink(%arg0: i1, %arg1: i32) {
  %0 = arith.addi %arg1, %arg1 : i32
  scf.if %arg0 {
    scf.if %arg0 {
      func.call @consume(%0) : (i32) -> ()
      scf.yield
    }
  }
  return
}
