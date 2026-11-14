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

// -----

func.func private @consume(i32) -> ()

// CHECK-LABEL: @test_scf_execute_region_multiblock_sink
// CHECK-SAME:  (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
// CHECK: scf.execute_region
// CHECK-NEXT:   %[[V0:.*]] = arith.muli %[[ARG0]], %[[ARG1]]
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   call @consume(%[[V0]])
func.func @test_scf_execute_region_multiblock_sink(%arg0: i32, %arg1: i32) {
  %0 = arith.muli %arg0, %arg1 : i32
  scf.execute_region {
    cf.br ^bb1
  ^bb1:
    func.call @consume(%0) : (i32) -> ()
    scf.yield
  }
  return
}

// -----

func.func private @sink_i32(i32)

// CHECK-LABEL: @test_scf_if_sink_through_loop
// CHECK-SAME: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: index, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32)
// CHECK: scf.if %[[ARG0]]
// CHECK:   %[[V0:.*]] = arith.muli %[[ARG2]], %[[ARG3]]
// CHECK:   scf.for
// CHECK:     call @sink_i32(%[[V0]])

func.func @test_scf_if_sink_through_loop(
    %arg0: i1, %arg1: index, %arg2: i32, %arg3: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.muli %arg2, %arg3 : i32
  scf.if %arg0 {
    scf.for %arg4 = %c0 to %arg1 step %c1 {
      func.call @sink_i32(%0) : (i32) -> ()
    }
  }
  return
}

// CHECK-LABEL: @test_scf_if_sink_through_loop_with_external_use
// CHECK: %[[V0:.*]] = arith.muli %{{.*}}, %{{.*}}
// CHECK: scf.if
// CHECK:   scf.for
// CHECK:     call @sink_i32(%[[V0]])
// CHECK: call @sink_i32(%[[V0]])

func.func @test_scf_if_sink_through_loop_with_external_use(
    %arg0: i1, %arg1: index, %arg2: i32, %arg3: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.muli %arg2, %arg3 : i32
  scf.if %arg0 {
    scf.for %arg4 = %c0 to %arg1 step %c1 {
      func.call @sink_i32(%0) : (i32) -> ()
    }
  }
  func.call @sink_i32(%0) : (i32) -> ()
  return
}
