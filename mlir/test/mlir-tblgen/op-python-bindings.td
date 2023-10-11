// RUN: mlir-tblgen -gen-python-op-bindings -bind-dialect=test -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// CHECK: @_ods_cext.register_dialect
// CHECK: class _Dialect(_ods_ir.Dialect):
  // CHECK: DIALECT_NAMESPACE = "test"
  // CHECK: pass
def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "Test";
}

class TestOp<string mnemonic, list<Trait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class AttrSizedOperandsOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.attr_sized_operands"
// CHECK: _ODS_OPERAND_SEGMENTS = [-1,1,0,]
def AttrSizedOperandsOp : TestOp<"attr_sized_operands",
                                 [AttrSizedOperandSegments]> {
  // CHECK: def __init__(self, variadic1, non_variadic, *, variadic2=None, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   operands.append(_get_op_results_or_values(variadic1))
  // CHECK:   operands.append(_get_op_result_or_value(non_variadic))
  // CHECK:   operands.append(_get_op_result_or_value(variadic2) if variadic2 is not None else None)
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def variadic1(self):
  // CHECK:   operand_range = _ods_segmented_accessor(
  // CHECK:       self.operation.operands,
  // CHECK:       self.operation.attributes["operandSegmentSizes"], 0)
  // CHECK:   return operand_range
  // CHECK-NOT: if len(operand_range)
  //
  // CHECK: @builtins.property
  // CHECK: def non_variadic(self):
  // CHECK:   operand_range = _ods_segmented_accessor(
  // CHECK:       self.operation.operands,
  // CHECK:       self.operation.attributes["operandSegmentSizes"], 1)
  // CHECK:   return operand_range[0]
  //
  // CHECK: @builtins.property
  // CHECK: def variadic2(self):
  // CHECK:   operand_range = _ods_segmented_accessor(
  // CHECK:       self.operation.operands,
  // CHECK:       self.operation.attributes["operandSegmentSizes"], 2)
  // CHECK:   return operand_range[0] if len(operand_range) > 0 else None
  let arguments = (ins Variadic<AnyType>:$variadic1, AnyType:$non_variadic,
                   Optional<AnyType>:$variadic2);
}

// CHECK: def attr_sized_operands(variadic1, non_variadic, *, variadic2=None, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(AttrSizedOperandsOp(variadic1=variadic1, non_variadic=non_variadic, variadic2=variadic2, loc=loc, ip=ip)))

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class AttrSizedResultsOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.attr_sized_results"
// CHECK: _ODS_RESULT_SEGMENTS = [0,1,-1,]
def AttrSizedResultsOp : TestOp<"attr_sized_results",
                               [AttrSizedResultSegments]> {
  // CHECK: def __init__(self, variadic1, non_variadic, variadic2, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   if variadic1 is not None: results.append(variadic1)
  // CHECK:   results.append(non_variadic)
  // CHECK:   results.append(variadic2)
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def variadic1(self):
  // CHECK:   result_range = _ods_segmented_accessor(
  // CHECK:       self.operation.results,
  // CHECK:       self.operation.attributes["resultSegmentSizes"], 0)
  // CHECK:   return result_range[0] if len(result_range) > 0 else None
  //
  // CHECK: @builtins.property
  // CHECK: def non_variadic(self):
  // CHECK:   result_range = _ods_segmented_accessor(
  // CHECK:       self.operation.results,
  // CHECK:       self.operation.attributes["resultSegmentSizes"], 1)
  // CHECK:   return result_range[0]
  //
  // CHECK: @builtins.property
  // CHECK: def variadic2(self):
  // CHECK:   result_range = _ods_segmented_accessor(
  // CHECK:       self.operation.results,
  // CHECK:       self.operation.attributes["resultSegmentSizes"], 2)
  // CHECK:   return result_range
  // CHECK-NOT: if len(result_range)
  let results = (outs Optional<AnyType>:$variadic1, AnyType:$non_variadic,
                 Variadic<AnyType>:$variadic2);
}

// CHECK: def attr_sized_results(variadic1, non_variadic, variadic2, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(AttrSizedResultsOp(variadic1=variadic1, non_variadic=non_variadic, variadic2=variadic2, loc=loc, ip=ip)))


// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class AttributedOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.attributed_op"
// CHECK-NOT: _ODS_OPERAND_SEGMENTS
// CHECK-NOT: _ODS_RESULT_SEGMENTS
def AttributedOp : TestOp<"attributed_op"> {
  // CHECK: def __init__(self, i32attr, in_, *, optionalF32Attr=None, unitAttr=None, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   attributes["i32attr"] = (i32attr if (
  // CHECK-NEXT:   issubclass(type(i32attr), _ods_ir.Attribute) or
  // CHECK-NEXT:   not _ods_ir.AttrBuilder.contains('I32Attr')
  // CHECK-NEXT:   _ods_ir.AttrBuilder.get('I32Attr')(i32attr, context=_ods_context)
  // CHECK:   if optionalF32Attr is not None: attributes["optionalF32Attr"] = (optionalF32Attr
  // CHECK:   if bool(unitAttr): attributes["unitAttr"] = _ods_ir.UnitAttr.get(
  // CHECK:     _ods_get_default_loc_context(loc))
  // CHECK:   attributes["in"] = (in_
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def i32attr(self):
  // CHECK:   return self.operation.attributes["i32attr"]

  // CHECK: @builtins.property
  // CHECK: def optionalF32Attr(self):
  // CHECK:   if "optionalF32Attr" not in self.operation.attributes:
  // CHECK:     return None
  // CHECK:   return self.operation.attributes["optionalF32Attr"]

  // CHECK: @builtins.property
  // CHECK: def unitAttr(self):
  // CHECK:   return "unitAttr" in self.operation.attributes

  // CHECK: @builtins.property
  // CHECK: def in_(self):
  // CHECK:   return self.operation.attributes["in"]

  let arguments = (ins I32Attr:$i32attr, OptionalAttr<F32Attr>:$optionalF32Attr,
                   UnitAttr:$unitAttr, I32Attr:$in);
}

// CHECK: def attributed_op(i32attr, in_, *, optional_f32_attr=None, unit_attr=None, loc=None, ip=None)
// CHECK:     return _maybe_cast(_get_op_result_or_op_results(AttributedOp(i32attr=i32attr, in_=in_, optionalF32Attr=optional_f32_attr, unitAttr=unit_attr, loc=loc, ip=ip)))

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class AttributedOpWithOperands(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.attributed_op_with_operands"
// CHECK-NOT: _ODS_OPERAND_SEGMENTS
// CHECK-NOT: _ODS_RESULT_SEGMENTS
def AttributedOpWithOperands : TestOp<"attributed_op_with_operands"> {
  // CHECK: def __init__(self, _gen_arg_0, _gen_arg_2, *, in_=None, is_=None, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   operands.append(_get_op_result_or_value(_gen_arg_0))
  // CHECK:   operands.append(_get_op_result_or_value(_gen_arg_2))
  // CHECK:   if bool(in_): attributes["in"] = _ods_ir.UnitAttr.get(
  // CHECK:     _ods_get_default_loc_context(loc))
  // CHECK:   if is_ is not None: attributes["is"] = (is_
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
  // CHECK:   return self.operation.attributes["is"]
  let arguments = (ins I32, UnitAttr:$in, F32, OptionalAttr<F32Attr>:$is);
}

// CHECK: def attributed_op_with_operands(_gen_arg_0, _gen_arg_2, *, in_=None, is_=None, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(AttributedOpWithOperands(_gen_arg_0=_gen_arg_0, _gen_arg_2=_gen_arg_2, in_=in_, is_=is_, loc=loc, ip=ip)))

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class DefaultValuedAttrsOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.default_valued_attrs"
def DefaultValuedAttrsOp : TestOp<"default_valued_attrs"> {
  // CHECK: def __init__(self, *, arr=None, unsupported=None, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   if arr is not None: attributes["arr"] = (arr
  // CHECK:   if unsupported is not None: attributes["unsupported"] = (unsupported
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  let arguments = (ins DefaultValuedAttr<I64ArrayAttr, "{}">:$arr,
                       DefaultValuedAttr<I64ArrayAttr, "dummy_func()">:$unsupported);
  let results = (outs);
}

// CHECK: def default_valued_attrs(*, arr=None, unsupported=None, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(DefaultValuedAttrsOp(arr=arr, unsupported=unsupported, loc=loc, ip=ip)))

// CHECK-LABEL: OPERATION_NAME = "test.derive_result_types_op"
def DeriveResultTypesOp : TestOp<"derive_result_types_op", [FirstAttrDerivedResultType]> {
  // CHECK: def __init__(self, type_, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   _ods_result_type_source_attr = attributes["type"]
  // CHECK:   _ods_derived_result_type = (
  // CHECK:       _ods_ir.TypeAttr(_ods_result_type_source_attr).value
  // CHECK:       if _ods_ir.TypeAttr.isinstance(_ods_result_type_source_attr) else
  // CHECK:       _ods_result_type_source_attr.type)
  // CHECK:   results.extend([_ods_derived_result_type] * 2)
  let arguments = (ins TypeAttr:$type);
  let results = (outs AnyType:$res, AnyType);
}

// CHECK: def derive_result_types_op(type_, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(DeriveResultTypesOp(type_=type_, loc=loc, ip=ip)))

// CHECK-LABEL: OPERATION_NAME = "test.derive_result_types_variadic_op"
def DeriveResultTypesVariadicOp : TestOp<"derive_result_types_variadic_op", [FirstAttrDerivedResultType]> {
  // CHECK: def __init__(self, res, _gen_res_1, type_, *, loc=None, ip=None):
  let arguments = (ins TypeAttr:$type);
  let results = (outs AnyType:$res, Variadic<AnyType>);
}

// CHECK: def derive_result_types_variadic_op(res, _gen_res_1, type_, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(DeriveResultTypesVariadicOp(res=res, _gen_res_1=_gen_res_1, type_=type_, loc=loc, ip=ip)))

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

// CHECK: def empty(*, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(EmptyOp(loc=loc, ip=ip)))

// CHECK-LABEL: OPERATION_NAME = "test.infer_result_types_implied_op"
def InferResultTypesImpliedOp : TestOp<"infer_result_types_implied_op"> {
  // CHECK:  def __init__(self, *, loc=None, ip=None):
  // CHECK:    _ods_context = _ods_get_default_loc_context(loc)
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))
  let results = (outs I32:$i32, F32:$f32);
}

// CHECK: def infer_result_types_implied_op(*, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(InferResultTypesImpliedOp(loc=loc, ip=ip)))

// CHECK-LABEL: OPERATION_NAME = "test.infer_result_types_op"
def InferResultTypesOp : TestOp<"infer_result_types_op", [InferTypeOpInterface]> {
  // CHECK:  def __init__(self, *, loc=None, ip=None):
  // CHECK:    operands = []
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))
  let results = (outs AnyType, AnyType, AnyType);
}

// CHECK: def infer_result_types_op(*, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(InferResultTypesOp(loc=loc, ip=ip)))

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class MissingNamesOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.missing_names"
def MissingNamesOp : TestOp<"missing_names"> {
  // CHECK: def __init__(self, i32, _gen_res_1, i64, _gen_arg_0, f32, _gen_arg_2, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   operands.append(_get_op_result_or_value(_gen_arg_0))
  // CHECK:   operands.append(_get_op_result_or_value(f32))
  // CHECK:   operands.append(_get_op_result_or_value(_gen_arg_2))
  // CHECK:   results.append(i32)
  // CHECK:   results.append(_gen_res_1)
  // CHECK:   results.append(i64)
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
  let results = (outs I32:$i32, AnyFloat, I64:$i64);
}

// CHECK: def missing_names(i32, _gen_res_1, i64, _gen_arg_0, f32, _gen_arg_2, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(MissingNamesOp(i32=i32, _gen_res_1=_gen_res_1, i64=i64, _gen_arg_0=_gen_arg_0, f32=f32, _gen_arg_2=_gen_arg_2, loc=loc, ip=ip)))

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class OneOptionalOperandOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.one_optional_operand"
// CHECK-NOT: _ODS_OPERAND_SEGMENTS
// CHECK-NOT: _ODS_RESULT_SEGMENTS
def OneOptionalOperandOp : TestOp<"one_optional_operand"> {
  let arguments = (ins AnyType:$non_optional, Optional<AnyType>:$optional);
  // CHECK: def __init__(self, non_optional, *, optional=None, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   operands.append(_get_op_result_or_value(non_optional))
  // CHECK:   if optional is not None: operands.append(_get_op_result_or_value(optional))
  // CHECK:   _ods_successors = None
  // CHECK:   super().__init__(self.build_generic(
  // CHECK:     attributes=attributes, results=results, operands=operands,
  // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  // CHECK: @builtins.property
  // CHECK: def non_optional(self):
  // CHECK:   return self.operation.operands[0]

  // CHECK: @builtins.property
  // CHECK: def optional(self):
  // CHECK:   return None if len(self.operation.operands) < 2 else self.operation.operands[1]
}

// CHECK: def one_optional_operand(non_optional, *, optional=None, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(OneOptionalOperandOp(non_optional=non_optional, optional=optional, loc=loc, ip=ip)))

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

// CHECK: def one_variadic_operand(non_variadic, variadic, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(OneVariadicOperandOp(non_variadic=non_variadic, variadic=variadic, loc=loc, ip=ip)))

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

// CHECK: def one_variadic_result(variadic, non_variadic, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(OneVariadicResultOp(variadic=variadic, non_variadic=non_variadic, loc=loc, ip=ip)))

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

// CHECK: def python_keyword(in_, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(PythonKeywordOp(in_=in_, loc=loc, ip=ip)))

// CHECK-LABEL: OPERATION_NAME = "test.same_results"
def SameResultsOp : TestOp<"same_results", [SameOperandsAndResultType]> {
  // CHECK: def __init__(self, in1, in2, *, loc=None, ip=None):
  // CHECK: operands = []
  // CHECK: results = []
  // CHECK: operands.append
  // CHECK: results.extend([operands[0].type] * 1)
  let arguments = (ins AnyType:$in1, AnyType:$in2);
  let results = (outs AnyType:$res);
}

// CHECK: def same_results(in1, in2, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(SameResultsOp(in1=in1, in2=in2, loc=loc, ip=ip)))

// CHECK-LABEL: OPERATION_NAME = "test.same_results_variadic"
def SameResultsVariadicOp : TestOp<"same_results_variadic", [SameOperandsAndResultType]> {
  // CHECK: def __init__(self, res, in1, in2, *, loc=None, ip=None):
  let arguments = (ins AnyType:$in1, AnyType:$in2);
  let results = (outs Variadic<AnyType>:$res);
}

// CHECK: def same_results_variadic(res, in1, in2, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(SameResultsVariadicOp(res=res, in1=in1, in2=in2, loc=loc, ip=ip)))


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

// CHECK: def same_variadic_operand(variadic1, non_variadic, variadic2, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(SameVariadicOperandSizeOp(variadic1=variadic1, non_variadic=non_variadic, variadic2=variadic2, loc=loc, ip=ip)))

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

// CHECK: def same_variadic_result(variadic1, non_variadic, variadic2, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(SameVariadicResultSizeOp(variadic1=variadic1, non_variadic=non_variadic, variadic2=variadic2, loc=loc, ip=ip)))

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class SimpleOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.simple"
def SimpleOp : TestOp<"simple"> {
  // CHECK: def __init__(self, i64, f64, i32, f32, *, loc=None, ip=None):
  // CHECK:   operands = []
  // CHECK:   results = []
  // CHECK:   attributes = {}
  // CHECK:   regions = None
  // CHECK:   operands.append(_get_op_result_or_value(i32))
  // CHECK:   operands.append(_get_op_result_or_value(f32))
  // CHECK:   results.append(i64)
  // CHECK:   results.append(f64)
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
  let results = (outs I64:$i64, AnyFloat:$f64);
}

// CHECK: def simple(i64, f64, i32, f32, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(SimpleOp(i64=i64, f64=f64, i32=i32, f32=f32, loc=loc, ip=ip)))

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
  // CHECK:  def region(self):
  // CHECK:    return self.regions[0]

  // CHECK:  @builtins.property
  // CHECK:  def variadic(self):
  // CHECK:    return self.regions[2:]
}

// CHECK: def variadic_and_normal_region(num_variadic, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(VariadicAndNormalRegionOp(num_variadic=num_variadic, loc=loc, ip=ip)))

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
  // CHECK:  def Variadic(self):
  // CHECK:    return self.regions[0:]
}

// CHECK: def variadic_region(num_variadic, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(VariadicRegionOp(num_variadic=num_variadic, loc=loc, ip=ip)))

// CHECK: @_ods_cext.register_operation(_Dialect)
// CHECK: class WithSpecialCharactersOp(_ods_ir.OpView):
// CHECK-LABEL: OPERATION_NAME = "test.123with--special.characters"
def WithSpecialCharactersOp : TestOp<"123with--special.characters"> {
}

// CHECK: def _123with__special_characters(*, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(WithSpecialCharactersOp(loc=loc, ip=ip)))

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

// CHECK: def with_successors(successor, successors, *, loc=None, ip=None)
// CHECK:   return _maybe_cast(_get_op_result_or_op_results(WithSuccessorsOp(successor=successor, successors=successors, loc=loc, ip=ip)))