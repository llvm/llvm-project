// RUN: mlir-opt %s

func.func @test_ops_verify(%arg: i32) -> f32 {
  %0 = "test.constant"() { value = 5.3 : f32 } : () -> f32
  %1 = test.loop_block %arg : (i32) -> f32 {
  ^bb0(%arg1 : i32):
    test.loop_block_term iter %arg exit %0
  }
  return %1 : f32
}
