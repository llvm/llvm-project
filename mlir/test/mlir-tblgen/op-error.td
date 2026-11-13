// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR4 %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR5 %s 2>&1 | FileCheck --check-prefix=ERROR5 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR6 %s 2>&1 | FileCheck --check-prefix=ERROR6 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR7 %s 2>&1 | FileCheck --check-prefix=ERROR7 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR8 %s 2>&1 | FileCheck --check-prefix=ERROR8 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR9 %s 2>&1 | FileCheck --check-prefix=ERROR9 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR10 %s 2>&1 | FileCheck --check-prefix=ERROR10 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR11 %s 2>&1 | FileCheck --check-prefix=ERROR11 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR12 %s 2>&1 | FileCheck --check-prefix=ERROR12 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR13 %s 2>&1 | FileCheck --check-prefix=ERROR13 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR14 %s 2>&1 | FileCheck --check-prefix=ERROR14 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR15 %s 2>&1 | FileCheck --check-prefix=ERROR15 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR16 %s 2>&1 | FileCheck --check-prefix=ERROR16 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR17 %s 2>&1 | FileCheck --check-prefix=ERROR17 %s
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DERROR18 %s 2>&1 | FileCheck --check-prefix=ERROR18 %s
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DERROR19 %s 2>&1 | FileCheck --check-prefix=ERROR19 %s
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DERROR20 %s 2>&1 | FileCheck --check-prefix=ERROR20 %s
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DERROR21 %s 2>&1 | FileCheck --check-prefix=ERROR21 %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test_dialect";
}

#ifdef ERROR18
// ERROR18: error: cannot substitute '$_self' while formatting trait predicate: missing named binding
// ERROR18: note: template: {{.*}}$_self.getType() == $_op.getOperand(0).getType()
// ERROR18: note: failed occurrence starts at template byte offset
// ERROR18: note: while expanding operation 'test_dialect.unbound_self'
// ERROR18: note: available named substitutions: $_ctxt, $_op, $a, $b
// ERROR18: note: no positional substitutions were supplied
def UnboundSelfOp : Op<Test_Dialect, "unbound_self", [
    PredOpTrait<"same type",
        CPred<"$_self.getType() == $_op.getOperand(0).getType()">>]> {
  let arguments = (ins AnyType:$a, AnyType:$b);
}
#endif

#ifdef ERROR19
// ERROR19: error: cannot substitute '$missing' while formatting trait predicate: missing named binding
// ERROR19: note: cannot substitute '$9' at template byte offset {{[0-9]+}}: positional argument out of range
// ERROR19: note: cannot substitute '$missing' at template byte offset {{[0-9]+}}: missing named binding
// ERROR19: note: template: {{.*}}$missing && $9 && $missing
// ERROR19: note: while expanding operation 'test_dialect.unbound_multiple'
// ERROR19: note: available named substitutions: $_ctxt, $_op
// ERROR19: note: no positional substitutions were supplied
def UnboundMultipleOp : Op<Test_Dialect, "unbound_multiple", [
    PredOpTrait<"multiple failures", CPred<"$missing && $9 && $missing">>]>;
#endif

#ifdef ERROR20
// ERROR20: error: cannot substitute '$missing' while formatting trait predicate
// ERROR20: note: template:
// ERROR20: note: while expanding operation 'test_dialect.multiclass'
// ERROR20: note: instantiated from multiclass
// ERROR20: note: available named substitutions: $_ctxt, $_op
// ERROR20: note: no positional substitutions were supplied
multiclass UnboundTraitMulticlass {
  def Op : Op<Test_Dialect, "multiclass", [
      PredOpTrait<"missing", CPred<"$missing">>]>;
}
defm Broken : UnboundTraitMulticlass;
#endif

#ifdef ERROR21
// An inline attribute predicate binds $_self while checking that attribute.
// That temporary binding must not make $_self available to an operation trait.
// ERROR21: error: cannot substitute '$_self' while formatting trait predicate: missing named binding
// ERROR21: note: while expanding operation 'test_dialect.inline_attr'
// ERROR21: note: available named substitutions: $_ctxt, $_op, $input, $value
// ERROR21: note: no positional substitutions were supplied
def DependentAttr : Attr<CPred<"$_self && $input">, "dependent"> {
  let convertFromStorage = "$_self";
}
def InlineAttrOp : Op<Test_Dialect, "inline_attr", [
    PredOpTrait<"unbound self", CPred<"$_self">>]> {
  let arguments = (ins DependentAttr:$value, I32:$input);
}
#endif

#ifdef ERROR1
// ERROR1: error: expected 'ins'
def OpInsMissing : Op<Test_Dialect, "ins_missing"> {
  let builders = [
    OpBuilder<(outs)>
  ];
}
#endif

#ifdef ERROR2
// ERROR2: error: expected an argument with default value after other arguments with default values
def OpDefaultValueNotTrailing : Op<Test_Dialect, "default_value"> {
  let builders = [
    OpBuilder<(ins CArg<"int", "42">, "int")>
  ];
}
#endif

#ifdef ERROR3
// ERROR3: error: expected an argument with default value after other arguments with default values
def OpDefaultValueNotTrailing : Op<Test_Dialect, "default_value"> {
  let builders = [
    OpBuilder<(ins CArg<"int", "42">, CArg<"int">)>
  ];
}
#endif

#ifdef ERROR4
// ERROR4: error: op has a conflict with two operands having the same name 'tensor'
def OpWithDuplicatedArgNames : Op<Test_Dialect, "default_value"> {
  let arguments = (ins AnyTensor:$tensor, AnyTensor:$tensor);
}
#endif

#ifdef ERROR5
// ERROR5: error: op has a conflict with two results having the same name 'tensor'
def OpWithDuplicatedResultNames : Op<Test_Dialect, "default_value"> {
  let results = (outs AnyTensor:$tensor, AnyTensor:$tensor);
}
#endif

#ifdef ERROR6
// ERROR6: error: op has a conflict with operands and results both having an entry with the name 'tensor'
def OpWithDuplicatedArgResultNames : Op<Test_Dialect, "default_value"> {
  let arguments = (ins AnyTensor:$tensor);
  let results = (outs AnyTensor:$tensor);
}
#endif

#ifdef ERROR7
// ERROR7: error: op has a conflict with operands and regions both having an entry with the name 'tensor'
def OpWithDuplicatedArgResultNames : Op<Test_Dialect, "default_value"> {
  let arguments = (ins AnyTensor:$tensor);
  let regions = (region AnyRegion:$tensor);
}
#endif

#ifdef ERROR8
// ERROR8: error: op has a conflict with results and regions both having an entry with the name 'tensor'
def OpWithDuplicatedArgResultNames : Op<Test_Dialect, "default_value"> {
  let results = (outs AnyTensor:$tensor);
  let regions = (region AnyRegion:$tensor);
}
#endif

#ifdef ERROR9
// ERROR9: error: op has a conflict with operands and successors both having an entry with the name 'target'
def OpWithDuplicatedArgResultNames : Op<Test_Dialect, "default_value"> {
  let successors = (successor AnySuccessor:$target);
  let arguments = (ins AnyTensor:$target);
}
#endif

#ifdef ERROR10
// ERROR10: error: op has a conflict with results and successors both having an entry with the name 'target'
def OpWithDuplicatedArgResultNames : Op<Test_Dialect, "default_value"> {
  let successors = (successor AnySuccessor:$target);
  let results = (outs AnyTensor:$target);
}
#endif

#ifdef ERROR11
// ERROR11: error: op has a conflict with regions and successors both having an entry with the name 'target'
def OpWithDuplicatedArgResultNames : Op<Test_Dialect, "default_value"> {
  let successors = (successor AnySuccessor:$target);
  let regions = (region AnyRegion:$target);
}
#endif

#ifdef ERROR12
def OpTraitA : NativeOpTrait<"OpTraitA"> {}
def OpTraitB : NativeOpTrait<"OpTraitB", [OpTraitA]> {}

// ERROR12: error: OpTraitB requires OpTraitA to precede it in traits list
def OpTraitWithoutDependentTrait : Op<Test_Dialect, "default_value", [OpTraitB]> {}
#endif

#ifdef ERROR13
def OpTraitA : NativeOpTrait<"OpTraitA"> {}
def OpInterfaceB : OpInterface<"OpInterfaceB"> {
  let dependentTraits = [OpTraitA];
}

// ERROR13: error: OpInterfaceB::Trait requires OpTraitA to precede it in traits list
def OpInterfaceWithoutDependentTrait : Op<Test_Dialect, "default_value", [OpInterfaceB]> {}
#endif

#ifdef ERROR14
// ERROR14: error: property 'operandSegmentSizes' conflicts with the property generated by 'AttrSizedOperandSegments'
def OpWithOperandSegmentProperty : Op<Test_Dialect, "operand_segment_property",
    [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyType>:$lhs, Variadic<AnyType>:$rhs,
                       I32Prop:$operandSegmentSizes);
}
#endif

#ifdef ERROR15
// ERROR15: error: property 'operand_segment_sizes' conflicts with the property generated by 'AttrSizedOperandSegments'
def OpWithLegacyOperandSegmentProperty
    : Op<Test_Dialect, "legacy_operand_segment_property",
         [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyType>:$lhs, Variadic<AnyType>:$rhs,
                       I32Prop:$operand_segment_sizes);
}
#endif

#ifdef ERROR16
// ERROR16: error: property 'resultSegmentSizes' conflicts with the property generated by 'AttrSizedResultSegments'
def OpWithResultSegmentProperty : Op<Test_Dialect, "result_segment_property",
    [AttrSizedResultSegments]> {
  let arguments = (ins I32Prop:$resultSegmentSizes);
  let results = (outs Variadic<AnyType>:$lhs, Variadic<AnyType>:$rhs);
}
#endif

#ifdef ERROR17
// ERROR17: error: property 'result_segment_sizes' conflicts with the property generated by 'AttrSizedResultSegments'
def OpWithLegacyResultSegmentProperty
    : Op<Test_Dialect, "legacy_result_segment_property",
         [AttrSizedResultSegments]> {
  let arguments = (ins I32Prop:$result_segment_sizes);
  let results = (outs Variadic<AnyType>:$lhs, Variadic<AnyType>:$rhs);
}
#endif
