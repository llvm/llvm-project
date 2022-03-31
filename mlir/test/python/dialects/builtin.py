# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.builtin as builtin
import mlir.dialects.func as func


def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f


# CHECK-LABEL: TEST: testFromPyFunc
@run
def testFromPyFunc():
  with Context() as ctx, Location.unknown() as loc:
    ctx.allow_unregistered_dialects = True
    m = builtin.ModuleOp()
    f32 = F32Type.get()
    f64 = F64Type.get()
    with InsertionPoint(m.body):
      # CHECK-LABEL: func @unary_return(%arg0: f64) -> f64
      # CHECK: return %arg0 : f64
      @func.FuncOp.from_py_func(f64)
      def unary_return(a):
        return a

      # CHECK-LABEL: func @binary_return(%arg0: f32, %arg1: f64) -> (f32, f64)
      # CHECK: return %arg0, %arg1 : f32, f64
      @func.FuncOp.from_py_func(f32, f64)
      def binary_return(a, b):
        return a, b

      # CHECK-LABEL: func @none_return(%arg0: f32, %arg1: f64)
      # CHECK: return
      @func.FuncOp.from_py_func(f32, f64)
      def none_return(a, b):
        pass

      # CHECK-LABEL: func @call_unary
      # CHECK: %0 = call @unary_return(%arg0) : (f64) -> f64
      # CHECK: return %0 : f64
      @func.FuncOp.from_py_func(f64)
      def call_unary(a):
        return unary_return(a)

      # CHECK-LABEL: func @call_binary
      # CHECK: %0:2 = call @binary_return(%arg0, %arg1) : (f32, f64) -> (f32, f64)
      # CHECK: return %0#0, %0#1 : f32, f64
      @func.FuncOp.from_py_func(f32, f64)
      def call_binary(a, b):
        return binary_return(a, b)

      # We expect coercion of a single result operation to a returned value.
      # CHECK-LABEL: func @single_result_op
      # CHECK: %0 = "custom.op1"() : () -> f32
      # CHECK: return %0 : f32
      @func.FuncOp.from_py_func()
      def single_result_op():
        return Operation.create("custom.op1", results=[f32])

      # CHECK-LABEL: func @call_none
      # CHECK: call @none_return(%arg0, %arg1) : (f32, f64) -> ()
      # CHECK: return
      @func.FuncOp.from_py_func(f32, f64)
      def call_none(a, b):
        return none_return(a, b)

      ## Variants and optional feature tests.
      # CHECK-LABEL: func @from_name_arg
      @func.FuncOp.from_py_func(f32, f64, name="from_name_arg")
      def explicit_name(a, b):
        return b

      @func.FuncOp.from_py_func(f32, f64)
      def positional_func_op(a, b, func_op):
        assert isinstance(func_op, func.FuncOp)
        return b

      @func.FuncOp.from_py_func(f32, f64)
      def kw_func_op(a, b=None, func_op=None):
        assert isinstance(func_op, func.FuncOp)
        return b

      @func.FuncOp.from_py_func(f32, f64)
      def kwargs_func_op(a, b=None, **kwargs):
        assert isinstance(kwargs["func_op"], func.FuncOp)
        return b

      # CHECK-LABEL: func @explicit_results(%arg0: f32, %arg1: f64) -> f64
      # CHECK: return %arg1 : f64
      @func.FuncOp.from_py_func(f32, f64, results=[f64])
      def explicit_results(a, b):
        func.ReturnOp([b])

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

        @func.FuncOp.from_py_func(f64, results=[f64])
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
      f = func.FuncOp(name="some_func",
                            type=FunctionType.get(
                                inputs=[tensor_type, tensor_type],
                                results=[tensor_type]),
                            visibility="nested")
      # CHECK: Name is: "some_func"
      print("Name is: ", f.name)

      # CHECK: Type is: (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
      print("Type is: ", f.type)

      # CHECK: Visibility is: "nested"
      print("Visibility is: ", f.visibility)

      try:
        entry_block = f.entry_block
      except IndexError as e:
        # CHECK: External function does not have a body
        print(e)

      with InsertionPoint(f.add_entry_block()):
        func.ReturnOp([f.entry_block.arguments[0]])
        pass

      try:
        f.add_entry_block()
      except IndexError as e:
        # CHECK: The function already has an entry block!
        print(e)

      # Try the callback builder and passing type as tuple.
      f = func.FuncOp(name="some_other_func",
                            type=([tensor_type, tensor_type], [tensor_type]),
                            visibility="nested",
                            body_builder=lambda f: func.ReturnOp(
                                [f.entry_block.arguments[0]]))

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
  with Context() as ctx, Location.unknown():
    ctx.allow_unregistered_dialects = True
    module = Module.create()
    f32 = F32Type.get()
    f64 = F64Type.get()
    with InsertionPoint(module.body):
      f = func.FuncOp("some_func", ([f32, f32], [f32, f32]))
      with InsertionPoint(f.add_entry_block()):
        func.ReturnOp(f.arguments)
      f.arg_attrs = ArrayAttr.get([
          DictAttr.get({
              "custom_dialect.foo": StringAttr.get("bar"),
              "custom_dialect.baz": UnitAttr.get()
          }),
          DictAttr.get({"custom_dialect.qux": ArrayAttr.get([])})
      ])
      f.result_attrs = ArrayAttr.get([
          DictAttr.get({"custom_dialect.res1": FloatAttr.get(f32, 42.0)}),
          DictAttr.get({"custom_dialect.res2": FloatAttr.get(f64, 256.0)})
      ])

      other = func.FuncOp("other_func", ([f32, f32], []))
      with InsertionPoint(other.add_entry_block()):
        func.ReturnOp([])
      other.arg_attrs = [
          DictAttr.get({"custom_dialect.foo": StringAttr.get("qux")}),
          DictAttr.get()
      ]

  # CHECK: [{custom_dialect.baz, custom_dialect.foo = "bar"}, {custom_dialect.qux = []}]
  print(f.arg_attrs)

  # CHECK: [{custom_dialect.res1 = 4.200000e+01 : f32}, {custom_dialect.res2 = 2.560000e+02 : f64}]
  print(f.result_attrs)

  # CHECK: func @some_func(
  # CHECK: %[[ARG0:.*]]: f32 {custom_dialect.baz, custom_dialect.foo = "bar"},
  # CHECK: %[[ARG1:.*]]: f32 {custom_dialect.qux = []}) ->
  # CHECK: f32 {custom_dialect.res1 = 4.200000e+01 : f32},
  # CHECK: f32 {custom_dialect.res2 = 2.560000e+02 : f64})
  # CHECK: return %[[ARG0]], %[[ARG1]] : f32, f32
  #
  # CHECK: func @other_func(
  # CHECK: %{{.*}}: f32 {custom_dialect.foo = "qux"},
  # CHECK: %{{.*}}: f32)
  print(module)
