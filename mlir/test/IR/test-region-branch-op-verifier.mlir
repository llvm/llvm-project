// RUN: mlir-opt %s -split-input-file

func.func @test_ops_verify(%arg: i32) -> f32 {
  %0 = "test.constant"() { value = 5.3 : f32 } : () -> f32
  %1 = test.loop_block %arg : (i32) -> f32 {
  ^bb0(%arg1 : i32):
    test.loop_block_term iter %arg exit %0
  }
  return %1 : f32
}

// -----

func.func @test_no_terminator(%arg: index) {
  test.switch_with_no_break %arg
  case 0 {
  ^bb:
  }
  case 1 {
  ^bb:
  }
  return
}

// -----

// test.loop_block_term has two operands: iter (i32, passed back to the region)
// and exit (f32, passed to the parent). getMutableSuccessorOperands(parent)
// returns only the exit operand. The function returns f32, matching the exit
// operand type, so verification must succeed.
//
// A verifier using getNumOperands() instead would incorrectly report "has 2
// operands, but enclosing function returns 1".
func.func @func_with_region_branch_terminator(%arg: i32) -> f32 {
  %0 = "test.constant"() { value = 5.3 : f32 } : () -> f32
  test.loop_block_term iter %arg exit %0
}
