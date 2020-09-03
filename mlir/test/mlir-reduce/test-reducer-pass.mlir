// RUN: mlir-opt %s
// RUN: not mlir-opt %s -test-mlir-reducer

func @test() {
  "test.crashOp"() : () -> ()
  return
}
