# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.builtin as builtin
import mlir.dialects.std as std


def run(f):
  print("\nTEST:", f.__name__)
  f()


# CHECK-LABEL: TEST: testBuildFuncOp
def testBuildFuncOp():
  ctx = Context()
  with Location.unknown(ctx) as loc:
    m = builtin.ModuleOp()

    f32 = F32Type.get()
    tensor_type = RankedTensorType.get((2, 3, 4), f32)
    with InsertionPoint.at_block_begin(m.body):
      func = builtin.FuncOp(name="some_func",
                            type=FunctionType.get(
                                inputs=[tensor_type, tensor_type],
                                results=[tensor_type]),
                            visibility="nested")
      # CHECK: Name is: "some_func"
      print("Name is: ", func.name)

      # CHECK: Type is: (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
      print("Type is: ", func.type)

      # CHECK: Visibility is: "nested"
      print("Visibility is: ", func.visibility)

      try:
        entry_block = func.entry_block
      except IndexError as e:
        # CHECK: External function does not have a body
        print(e)

      with InsertionPoint(func.add_entry_block()):
        std.ReturnOp([func.entry_block.arguments[0]])
        pass

      try:
        func.add_entry_block()
      except IndexError as e:
        # CHECK: The function already has an entry block!
        print(e)

      # Try the callback builder and passing type as tuple.
      func = builtin.FuncOp(name="some_other_func",
                            type=([tensor_type, tensor_type], [tensor_type]),
                            visibility="nested",
                            body_builder=lambda func: std.ReturnOp(
                                [func.entry_block.arguments[0]]))

  # CHECK: module  {
  # CHECK:  func nested @some_func(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  # CHECK:   return %arg0 : tensor<2x3x4xf32>
  # CHECK:  }
  # CHECK:  func nested @some_other_func(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  # CHECK:   return %arg0 : tensor<2x3x4xf32>
  # CHECK:  }
  print(m)


run(testBuildFuncOp)
