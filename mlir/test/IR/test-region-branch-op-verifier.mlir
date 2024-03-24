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
