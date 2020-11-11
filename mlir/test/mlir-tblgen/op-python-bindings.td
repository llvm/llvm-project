// RUN: mlir-tblgen -gen-python-op-bindings -bind-dialect=test -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

// CHECK: @_cext.register_dialect
// CHECK: class _Dialect(_ir.Dialect):
  // CHECK: DIALECT_NAMESPACE = "test"
  // CHECK: pass
def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "Test";
}
class TestOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

// CHECK: @_cext.register_operation(_Dialect)
// CHECK: class AttrSizedOperandsOp(_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.attr_sized_operands"
def AttrSizedOperandsOp : TestOp<"attr_sized_operands",
                                 [AttrSizedOperandSegments]> {
  // CHECK: @property
  // CHECK: def variadic1(self):
  // CHECK:   operand_range = _segmented_accessor(
  // CHECK:       self.operation.operands,
  // CHECK:       self.operation.attributes["operand_segment_sizes"], 0)
  // CHECK:   return operand_range
  //
  // CHECK: @property
  // CHECK: def non_variadic(self):
  // CHECK:   operand_range = _segmented_accessor(
  // CHECK:       self.operation.operands,
  // CHECK:       self.operation.attributes["operand_segment_sizes"], 1)
  // CHECK:   return operand_range[0]
  //
  // CHECK: @property
  // CHECK: def variadic2(self):
  // CHECK:   operand_range = _segmented_accessor(
  // CHECK:       self.operation.operands,
  // CHECK:       self.operation.attributes["operand_segment_sizes"], 2)
  // CHECK:   return operand_range[0] if len(operand_range) > 0 else None
  let arguments = (ins Variadic<AnyType>:$variadic1, AnyType:$non_variadic,
                   Optional<AnyType>:$variadic2);
}

// CHECK: @_cext.register_operation(_Dialect)
// CHECK: class AttrSizedResultsOp(_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.attr_sized_results"
def AttrSizedResultsOp : TestOp<"attr_sized_results",
                               [AttrSizedResultSegments]> {
  // CHECK: @property
  // CHECK: def variadic1(self):
  // CHECK:   result_range = _segmented_accessor(
  // CHECK:       self.operation.results,
  // CHECK:       self.operation.attributes["result_segment_sizes"], 0)
  // CHECK:   return result_range[0] if len(result_range) > 0 else None
  //
  // CHECK: @property
  // CHECK: def non_variadic(self):
  // CHECK:   result_range = _segmented_accessor(
  // CHECK:       self.operation.results,
  // CHECK:       self.operation.attributes["result_segment_sizes"], 1)
  // CHECK:   return result_range[0]
  //
  // CHECK: @property
  // CHECK: def variadic2(self):
  // CHECK:   result_range = _segmented_accessor(
  // CHECK:       self.operation.results,
  // CHECK:       self.operation.attributes["result_segment_sizes"], 2)
  // CHECK:   return result_range
  let results = (outs Optional<AnyType>:$variadic1, AnyType:$non_variadic,
                 Optional<AnyType>:$variadic2);
}

// CHECK: @_cext.register_operation(_Dialect)
// CHECK: class EmptyOp(_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.empty"
def EmptyOp : TestOp<"empty">;

// CHECK: @_cext.register_operation(_Dialect)
// CHECK: class MissingNamesOp(_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.missing_names"
def MissingNamesOp : TestOp<"missing_names"> {
  // CHECK: @property
  // CHECK: def f32(self):
  // CHECK:   return self.operation.operands[1]
  let arguments = (ins I32, F32:$f32, I64);

  // CHECK: @property
  // CHECK: def i32(self):
  // CHECK:   return self.operation.results[0]
  //
  // CHECK: @property
  // CHECK: def i64(self):
  // CHECK:   return self.operation.results[2]
  let results = (outs I32:$i32, F32, I64:$i64);
}

// CHECK: @_cext.register_operation(_Dialect)
// CHECK: class OneVariadicOperandOp(_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.one_variadic_operand"
def OneVariadicOperandOp : TestOp<"one_variadic_operand"> {
  // CHECK: @property
  // CHECK: def non_variadic(self):
  // CHECK:   return self.operation.operands[0]
  //
  // CHECK: @property
  // CHECK: def variadic(self):
  // CHECK:   variadic_group_length = len(self.operation.operands) - 2 + 1
  // CHECK:   return self.operation.operands[1:1 + variadic_group_length]
  let arguments = (ins AnyType:$non_variadic, Variadic<AnyType>:$variadic);
}

// CHECK: @_cext.register_operation(_Dialect)
// CHECK: class OneVariadicResultOp(_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.one_variadic_result"
def OneVariadicResultOp : TestOp<"one_variadic_result"> {
  // CHECK: @property
  // CHECK: def variadic(self):
  // CHECK:   variadic_group_length = len(self.operation.results) - 2 + 1
  // CHECK:   return self.operation.results[0:0 + variadic_group_length]
  //
  // CHECK: @property
  // CHECK: def non_variadic(self):
  // CHECK:   variadic_group_length = len(self.operation.results) - 2 + 1
  // CHECK:   return self.operation.results[1 + variadic_group_length - 1]
  let results = (outs Variadic<AnyType>:$variadic, AnyType:$non_variadic);
}

// CHECK: @_cext.register_operation(_Dialect)
// CHECK: class PythonKeywordOp(_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.python_keyword"
def PythonKeywordOp : TestOp<"python_keyword"> {
  // CHECK: @property
  // CHECK: def in_(self):
  // CHECK:   return self.operation.operands[0]
  let arguments = (ins AnyType:$in);
}

// CHECK: @_cext.register_operation(_Dialect)
// CHECK: class SameVariadicOperandSizeOp(_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.same_variadic_operand"
def SameVariadicOperandSizeOp : TestOp<"same_variadic_operand",
                                       [SameVariadicOperandSize]> {
  // CHECK: @property
  // CHECK: def variadic1(self):
  // CHECK:   start, pg = _equally_sized_accessor(operation.operands, 2, 0, 0)
  // CHECK:   return self.operation.operands[start:start + pg]
  //
  // CHECK: @property
  // CHECK: def non_variadic(self):
  // CHECK:   start, pg = _equally_sized_accessor(operation.operands, 2, 0, 1)
  // CHECK:   return self.operation.operands[start]
  //
  // CHECK: @property
  // CHECK: def variadic2(self):
  // CHECK:   start, pg = _equally_sized_accessor(operation.operands, 2, 1, 1)
  // CHECK:   return self.operation.operands[start:start + pg]
  let arguments = (ins Variadic<AnyType>:$variadic1, AnyType:$non_variadic,
                   Variadic<AnyType>:$variadic2);
}

// CHECK: @_cext.register_operation(_Dialect)
// CHECK: class SameVariadicResultSizeOp(_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.same_variadic_result"
def SameVariadicResultSizeOp : TestOp<"same_variadic_result",
                                      [SameVariadicResultSize]> {
  // CHECK: @property
  // CHECK: def variadic1(self):
  // CHECK:   start, pg = _equally_sized_accessor(operation.results, 2, 0, 0)
  // CHECK:   return self.operation.results[start:start + pg]
  //
  // CHECK: @property
  // CHECK: def non_variadic(self):
  // CHECK:   start, pg = _equally_sized_accessor(operation.results, 2, 0, 1)
  // CHECK:   return self.operation.results[start]
  //
  // CHECK: @property
  // CHECK: def variadic2(self):
  // CHECK:   start, pg = _equally_sized_accessor(operation.results, 2, 1, 1)
  // CHECK:   return self.operation.results[start:start + pg]
  let results = (outs Variadic<AnyType>:$variadic1, AnyType:$non_variadic,
                 Variadic<AnyType>:$variadic2);
}

// CHECK: @_cext.register_operation(_Dialect)
// CHECK: class SimpleOp(_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.simple"
def SimpleOp : TestOp<"simple"> {
  // CHECK: @property
  // CHECK: def i32(self):
  // CHECK:   return self.operation.operands[0]
  //
  // CHECK: @property
  // CHECK: def f32(self):
  // CHECK:   return self.operation.operands[1]
  let arguments = (ins I32:$i32, F32:$f32);

  // CHECK: @property
  // CHECK: def i64(self):
  // CHECK:   return self.operation.results[0]
  //
  // CHECK: @property
  // CHECK: def f64(self):
  // CHECK:   return self.operation.results[1]
  let results = (outs I64:$i64, F64:$f64);
}
