// RUN: mlir-opt %s
// RUN: not mlir-opt %s -test-mlir-reducer -pass-test function-reducer

func.func @test() {
  "test.op_crash"() : () -> ()
  return
}
