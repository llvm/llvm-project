# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.builtin as builtin
import mlir.dialects.std as std


def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f


# CHECK-LABEL: TEST: testFromPyFunc
@run
def testFromPyFunc():
  with Context() as ctx, Location.unknown() as loc:
    m = builtin.ModuleOp()
    f32 = F32Type.get()
    f64 = F64Type.get()
    with InsertionPoint(m.body):
      # CHECK-LABEL: func @unary_return(%arg0: f64) -> f64
      # CHECK: return %arg0 : f64
      @builtin.FuncOp.from_py_func(f64)
      def unary_return(a):
        return a

      # CHECK-LABEL: func @binary_return(%arg0: f32, %arg1: f64) -> (f32, f64)
      # CHECK: return %arg0, %arg1 : f32, f64
      @builtin.FuncOp.from_py_func(f32, f64)
      def binary_return(a, b):
        return a, b

      # CHECK-LABEL: func @none_return(%arg0: f32, %arg1: f64)
      # CHECK: return
      @builtin.FuncOp.from_py_func(f32, f64)
      def none_return(a, b):
        pass

      # CHECK-LABEL: func @call_unary
      # CHECK: %0 = call @unary_return(%arg0) : (f64) -> f64
      # CHECK: return %0 : f64
      @builtin.FuncOp.from_py_func(f64)
      def call_unary(a):
        return unary_return(a)

      # CHECK-LABEL: func @call_binary
      # CHECK: %0:2 = call @binary_return(%arg0, %arg1) : (f32, f64) -> (f32, f64)
      # CHECK: return %0#0, %0#1 : f32, f64
      @builtin.FuncOp.from_py_func(f32, f64)
      def call_binary(a, b):
        return binary_return(a, b)

      # CHECK-LABEL: func @call_none
      # CHECK: call @none_return(%arg0, %arg1) : (f32, f64) -> ()
      # CHECK: return
      @builtin.FuncOp.from_py_func(f32, f64)
      def call_none(a, b):
        return none_return(a, b)

      ## Variants and optional feature tests.
      # CHECK-LABEL: func @from_name_arg
      @builtin.FuncOp.from_py_func(f32, f64, name="from_name_arg")
      def explicit_name(a, b):
        return b

      @builtin.FuncOp.from_py_func(f32, f64)
      def positional_func_op(a, b, func_op):
        assert isinstance(func_op, builtin.FuncOp)
        return b

      @builtin.FuncOp.from_py_func(f32, f64)
      def kw_func_op(a, b=None, func_op=None):
        assert isinstance(func_op, builtin.FuncOp)
        return b

      @builtin.FuncOp.from_py_func(f32, f64)
      def kwargs_func_op(a, b=None, **kwargs):
        assert isinstance(kwargs["func_op"], builtin.FuncOp)
        return b

      # CHECK-LABEL: func @explicit_results(%arg0: f32, %arg1: f64) -> f64
      # CHECK: return %arg1 : f64
      @builtin.FuncOp.from_py_func(f32, f64, results=[f64])
      def explicit_results(a, b):
        std.ReturnOp([b])

  print(m)


# CHECK-LABEL: TEST: testFromPyFuncErrors
@run
def testFromPyFuncErrors():
  with Context() as ctx, Location.unknown() as loc:
    m = builtin.ModuleOp()
    f32 = F32Type.get()
    f64 = F64Type.get()
    with InsertionPoint(m.body):
      try:

        @builtin.FuncOp.from_py_func(f64, results=[f64])
        def unary_return(a):
          return a
      except AssertionError as e:
        # CHECK: Capturing a python function with explicit `results=` requires that the wrapped function returns None.
        print(e)


# CHECK-LABEL: TEST: testBuildFuncOp
@run
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


# CHECK-LABEL: TEST: testFuncArgumentAccess
@run
def testFuncArgumentAccess():
  with Context(), Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    f64 = F64Type.get()
    with InsertionPoint(module.body):
      func = builtin.FuncOp("some_func", ([f32, f32], [f32, f32]))
      with InsertionPoint(func.add_entry_block()):
        std.ReturnOp(func.arguments)
      func.arg_attrs = ArrayAttr.get([
          DictAttr.get({
              "foo": StringAttr.get("bar"),
              "baz": UnitAttr.get()
          }),
          DictAttr.get({"qux": ArrayAttr.get([])})
      ])
      func.result_attrs = ArrayAttr.get([
          DictAttr.get({"res1": FloatAttr.get(f32, 42.0)}),
          DictAttr.get({"res2": FloatAttr.get(f64, 256.0)})
      ])

      other = builtin.FuncOp("other_func", ([f32, f32], []))
      with InsertionPoint(other.add_entry_block()):
        std.ReturnOp([])
      other.arg_attrs = [
          DictAttr.get({"foo": StringAttr.get("qux")}),
          DictAttr.get()
      ]

  # CHECK: [{baz, foo = "bar"}, {qux = []}]
  print(func.arg_attrs)

  # CHECK: [{res1 = 4.200000e+01 : f32}, {res2 = 2.560000e+02 : f64}]
  print(func.result_attrs)

  # CHECK: func @some_func(
  # CHECK: %[[ARG0:.*]]: f32 {baz, foo = "bar"},
  # CHECK: %[[ARG1:.*]]: f32 {qux = []}) ->
  # CHECK: f32 {res1 = 4.200000e+01 : f32},
  # CHECK: f32 {res2 = 2.560000e+02 : f64})
  # CHECK: return %[[ARG0]], %[[ARG1]] : f32, f32
  #
  # CHECK: func @other_func(
  # CHECK: %{{.*}}: f32 {foo = "qux"},
  # CHECK: %{{.*}}: f32)
  print(module)
