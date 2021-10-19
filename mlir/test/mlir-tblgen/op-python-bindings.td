// RUN: mlir-tblgen -gen-python-op-bindings -bind-dialect=test -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"
include "mlir/Bindings/Python/Attributes.td"

// CHECK: @_ods_cext.register_dialect
// CHECK: class _Dialect(_ods_ir.Dialect):
  // CHECK: DIALECT_NAMESPACE = "test"
  // CHECK: pass
def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "Test";
}
class TestOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class AttrSizedOperandsOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.attr_sized_operands"
// CHECK: _ODS_OPERAND_SEGMENTS = [-1,1,-1,]
def AttrSizedOperandsOp : TestOp<"attr_sized_operands",
                                 [AttrSizedOperandSegments]> {
  // CHECK: def __init__(self, variadic1, non_variadic, variadic2, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   operands.append(_get_op_results_or_values(variadic1))
  // CHECK:   operands.append(_get_op_result_or_value(non_variadic))
  // CHECK:   if variadic2 is not None: operands.append(_get_op_result_or_value(variadic2))
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def variadic1(self):
  // CHECK:   operand_range = _ods_segmented_accessor(
  // CHECK:       self.operation.operands,
  // CHECK:       self.operation.attributes["operand_segment_sizes"], 0)
  // CHECK:   return operand_range
  //
  // CHECK: @builtins.property
  // CHECK: def non_variadic(self):
  // CHECK:   operand_range = _ods_segmented_accessor(
  // CHECK:       self.operation.operands,
  // CHECK:       self.operation.attributes["operand_segment_sizes"], 1)
  // CHECK:   return operand_range[0]
  //
  // CHECK: @builtins.property
  // CHECK: def variadic2(self):
  // CHECK:   operand_range = _ods_segmented_accessor(
  // CHECK:       self.operation.operands,
  // CHECK:       self.operation.attributes["operand_segment_sizes"], 2)
  // CHECK:   return operand_range[0] if len(operand_range) > 0 else None
  let arguments = (ins Variadic<AnyType>:$variadic1, AnyType:$non_variadic,
                   Optional<AnyType>:$variadic2);
}

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class AttrSizedResultsOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.attr_sized_results"
// CHECK: _ODS_RESULT_SEGMENTS = [-1,1,-1,]
def AttrSizedResultsOp : TestOp<"attr_sized_results",
                               [AttrSizedResultSegments]> {
  // CHECK: def __init__(self, variadic1, non_variadic, variadic2, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   if variadic1 is not None: results.append(variadic1)
  // CHECK:   results.append(non_variadic)
  // CHECK:   if variadic2 is not None: results.append(variadic2)
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def variadic1(self):
  // CHECK:   result_range = _ods_segmented_accessor(
  // CHECK:       self.operation.results,
  // CHECK:       self.operation.attributes["result_segment_sizes"], 0)
  // CHECK:   return result_range[0] if len(result_range) > 0 else None
  //
  // CHECK: @builtins.property
  // CHECK: def non_variadic(self):
  // CHECK:   result_range = _ods_segmented_accessor(
  // CHECK:       self.operation.results,
  // CHECK:       self.operation.attributes["result_segment_sizes"], 1)
  // CHECK:   return result_range[0]
  //
  // CHECK: @builtins.property
  // CHECK: def variadic2(self):
  // CHECK:   result_range = _ods_segmented_accessor(
  // CHECK:       self.operation.results,
  // CHECK:       self.operation.attributes["result_segment_sizes"], 2)
  // CHECK:   return result_range
  let results = (outs Optional<AnyType>:$variadic1, AnyType:$non_variadic,
                 Optional<AnyType>:$variadic2);
}


// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class AttributedOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.attributed_op"
// CHECK-NOT: _ODS_OPERAND_SEGMENTS
// CHECK-NOT: _ODS_RESULT_SEGMENTS
def AttributedOp : TestOp<"attributed_op"> {
  // CHECK: def __init__(self, i32attr, optionalF32Attr, unitAttr, in_, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   attributes["i32attr"] = i32attr
  // CHECK:   if optionalF32Attr is not None: attributes["optionalF32Attr"] = optionalF32Attr
  // CHECK:   if bool(unitAttr): attributes["unitAttr"] = _ods_ir.UnitAttr.get(
  // CHECK:     _ods_get_default_loc_context(loc))
  // CHECK:   attributes["in"] = in_
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def i32attr(self):
  // CHECK:   return _ods_ir.IntegerAttr(self.operation.attributes["i32attr"])

  // CHECK: @builtins.property
  // CHECK: def optionalF32Attr(self):
  // CHECK:   if "optionalF32Attr" not in self.operation.attributes:
  // CHECK:     return None
  // CHECK:   return _ods_ir.FloatAttr(self.operation.attributes["optionalF32Attr"])

  // CHECK: @builtins.property
  // CHECK: def unitAttr(self):
  // CHECK:   return "unitAttr" in self.operation.attributes

  // CHECK: @builtins.property
  // CHECK: def in_(self):
  // CHECK:   return _ods_ir.IntegerAttr(self.operation.attributes["in"])
  let arguments = (ins I32Attr:$i32attr, OptionalAttr<F32Attr>:$optionalF32Attr,
                   UnitAttr:$unitAttr, I32Attr:$in);
}

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class AttributedOpWithOperands(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.attributed_op_with_operands"
// CHECK-NOT: _ODS_OPERAND_SEGMENTS
// CHECK-NOT: _ODS_RESULT_SEGMENTS
def AttributedOpWithOperands : TestOp<"attributed_op_with_operands"> {
  // CHECK: def __init__(self, _gen_arg_0, in_, _gen_arg_2, is_, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   operands.append(_get_op_result_or_value(_gen_arg_0))
  // CHECK:   operands.append(_get_op_result_or_value(_gen_arg_2))
  // CHECK:   if bool(in_): attributes["in"] = _ods_ir.UnitAttr.get(
  // CHECK:     _ods_get_default_loc_context(loc))
  // CHECK:   if is_ is not None: attributes["is"] = is_
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def in_(self):
  // CHECK:   return "in" in self.operation.attributes

  // CHECK: @builtins.property
  // CHECK: def is_(self):
  // CHECK:   if "is" not in self.operation.attributes:
  // CHECK:     return None
  // CHECK:   return _ods_ir.FloatAttr(self.operation.attributes["is"])
  let arguments = (ins I32, UnitAttr:$in, F32, OptionalAttr<F32Attr>:$is);
}


// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class EmptyOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.empty"
def EmptyOp : TestOp<"empty">;
  // CHECK: def __init__(self, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class MissingNamesOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.missing_names"
def MissingNamesOp : TestOp<"missing_names"> {
  // CHECK: def __init__(self, i32, _gen_res_1, i64, _gen_arg_0, f32, _gen_arg_2, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   results.append(i32)
  // CHECK:   results.append(_gen_res_1)
  // CHECK:   results.append(i64)
  // CHECK:   operands.append(_get_op_result_or_value(_gen_arg_0))
  // CHECK:   operands.append(_get_op_result_or_value(f32))
  // CHECK:   operands.append(_get_op_result_or_value(_gen_arg_2))
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def f32(self):
  // CHECK:   return self.operation.operands[1]
  let arguments = (ins I32, F32:$f32, I64);

  // CHECK: @builtins.property
  // CHECK: def i32(self):
  // CHECK:   return self.operation.results[0]
  //
  // CHECK: @builtins.property
  // CHECK: def i64(self):
  // CHECK:   return self.operation.results[2]
  let results = (outs I32:$i32, F32, I64:$i64);
}

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class OneVariadicOperandOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.one_variadic_operand"
// CHECK-NOT: _ODS_OPERAND_SEGMENTS
// CHECK-NOT: _ODS_RESULT_SEGMENTS
def OneVariadicOperandOp : TestOp<"one_variadic_operand"> {
  // CHECK: def __init__(self, non_variadic, variadic, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   operands.append(_get_op_result_or_value(non_variadic))
  // CHECK:   operands.extend(_get_op_results_or_values(variadic))
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def non_variadic(self):
  // CHECK:   return self.operation.operands[0]
  //
  // CHECK: @builtins.property
  // CHECK: def variadic(self):
  // CHECK:   _ods_variadic_group_length = len(self.operation.operands) - 2 + 1
  // CHECK:   return self.operation.operands[1:1 + _ods_variadic_group_length]
  let arguments = (ins AnyType:$non_variadic, Variadic<AnyType>:$variadic);
}

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class OneVariadicResultOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.one_variadic_result"
// CHECK-NOT: _ODS_OPERAND_SEGMENTS
// CHECK-NOT: _ODS_RESULT_SEGMENTS
def OneVariadicResultOp : TestOp<"one_variadic_result"> {
  // CHECK: def __init__(self, variadic, non_variadic, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   results.extend(variadic)
  // CHECK:   results.append(non_variadic)
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def variadic(self):
  // CHECK:   _ods_variadic_group_length = len(self.operation.results) - 2 + 1
  // CHECK:   return self.operation.results[0:0 + _ods_variadic_group_length]
  //
  // CHECK: @builtins.property
  // CHECK: def non_variadic(self):
  // CHECK:   _ods_variadic_group_length = len(self.operation.results) - 2 + 1
  // CHECK:   return self.operation.results[1 + _ods_variadic_group_length - 1]
  let results = (outs Variadic<AnyType>:$variadic, AnyType:$non_variadic);
}

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class PythonKeywordOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.python_keyword"
def PythonKeywordOp : TestOp<"python_keyword"> {
  // CHECK: def __init__(self, in_, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   operands.append(_get_op_result_or_value(in_))
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def in_(self):
  // CHECK:   return self.operation.operands[0]
  let arguments = (ins AnyType:$in);
}

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class SameVariadicOperandSizeOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.same_variadic_operand"
def SameVariadicOperandSizeOp : TestOp<"same_variadic_operand",
                                       [SameVariadicOperandSize]> {
  // CHECK: @builtins.property
  // CHECK: def variadic1(self):
  // CHECK:   start, pg = _ods_equally_sized_accessor(operation.operands, 2, 0, 0)
  // CHECK:   return self.operation.operands[start:start + pg]
  //
  // CHECK: @builtins.property
  // CHECK: def non_variadic(self):
  // CHECK:   start, pg = _ods_equally_sized_accessor(operation.operands, 2, 0, 1)
  // CHECK:   return self.operation.operands[start]
  //
  // CHECK: @builtins.property
  // CHECK: def variadic2(self):
  // CHECK:   start, pg = _ods_equally_sized_accessor(operation.operands, 2, 1, 1)
  // CHECK:   return self.operation.operands[start:start + pg]
  let arguments = (ins Variadic<AnyType>:$variadic1, AnyType:$non_variadic,
                   Variadic<AnyType>:$variadic2);
}

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class SameVariadicResultSizeOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.same_variadic_result"
def SameVariadicResultSizeOp : TestOp<"same_variadic_result",
                                      [SameVariadicResultSize]> {
  // CHECK: @builtins.property
  // CHECK: def variadic1(self):
  // CHECK:   start, pg = _ods_equally_sized_accessor(operation.results, 2, 0, 0)
  // CHECK:   return self.operation.results[start:start + pg]
  //
  // CHECK: @builtins.property
  // CHECK: def non_variadic(self):
  // CHECK:   start, pg = _ods_equally_sized_accessor(operation.results, 2, 0, 1)
  // CHECK:   return self.operation.results[start]
  //
  // CHECK: @builtins.property
  // CHECK: def variadic2(self):
  // CHECK:   start, pg = _ods_equally_sized_accessor(operation.results, 2, 1, 1)
  // CHECK:   return self.operation.results[start:start + pg]
  let results = (outs Variadic<AnyType>:$variadic1, AnyType:$non_variadic,
                 Variadic<AnyType>:$variadic2);
}

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class SimpleOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.simple"
def SimpleOp : TestOp<"simple"> {
  // CHECK: def __init__(self, i64, f64, i32, f32, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   results.append(i64)
  // CHECK:   results.append(f64)
  // CHECK:   operands.append(_get_op_result_or_value(i32))
  // CHECK:   operands.append(_get_op_result_or_value(f32))
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def i32(self):
  // CHECK:   return self.operation.operands[0]
  //
  // CHECK: @builtins.property
  // CHECK: def f32(self):
  // CHECK:   return self.operation.operands[1]
  let arguments = (ins I32:$i32, F32:$f32);

  // CHECK: @builtins.property
  // CHECK: def i64(self):
  // CHECK:   return self.operation.results[0]
  //
  // CHECK: @builtins.property
  // CHECK: def f64(self):
  // CHECK:   return self.operation.results[1]
  let results = (outs I64:$i64, F64:$f64);
}

// CHECK: class VariadicAndNormalRegionOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.variadic_and_normal_region"
def VariadicAndNormalRegionOp : TestOp<"variadic_and_normal_region"> {
  // CHECK:  def __init__(self, num_variadic, *, loc=None, ip=None):
  // CHECK:    operands = []
  // CHECK:    results = []
  // CHECK:    attributes = {}
  // CHECK:    regions = None
  // CHECK:    _ods_successors = None
  // CHECK:    regions = 2 + num_variadic
  // CHECK:    super().__init__(self.build_generic(
  // CHECK:      attributes=attributes, results=results, operands=operands,
  // CHECK:      successors=_ods_successors, regions=regions, loc=loc, ip=ip))
  let regions = (region AnyRegion:$region, AnyRegion, VariadicRegion<AnyRegion>:$variadic);

  // CHECK:  @builtins.property
  // CHECK:  def region():
  // CHECK:    return self.regions[0]

  // CHECK:  @builtins.property
  // CHECK:  def variadic():
  // CHECK:    return self.regions[2:]
}

// CHECK: class VariadicRegionOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.variadic_region"
def VariadicRegionOp : TestOp<"variadic_region"> {
  // CHECK:  def __init__(self, num_variadic, *, loc=None, ip=None):
  // CHECK:    operands = []
  // CHECK:    results = []
  // CHECK:    attributes = {}
  // CHECK:    regions = None
  // CHECK:    _ods_successors = None
  // CHECK:    regions = 0 + num_variadic
  // CHECK:    super().__init__(self.build_generic(
  // CHECK:      attributes=attributes, results=results, operands=operands,
  // CHECK:      successors=_ods_successors, regions=regions, loc=loc, ip=ip))
  let regions = (region VariadicRegion<AnyRegion>:$Variadic);

  // CHECK:  @builtins.property
  // CHECK:  def Variadic():
  // CHECK:    return self.regions[0:]
}

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class WithSuccessorsOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.with_successors"
def WithSuccessorsOp : TestOp<"with_successors"> {
  // CHECK-NOT:  _ods_successors = None
  // CHECK:      _ods_successors = []
  // CHECK-NEXT: _ods_successors.append(successor)
  // CHECK-NEXT: _ods_successors.extend(successors)
  let successors = (successor AnySuccessor:$successor,
                              VariadicSuccessor<AnySuccessor>:$successors);
}

