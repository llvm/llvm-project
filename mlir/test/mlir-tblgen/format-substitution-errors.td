// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DDEFAULT %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: mlir-tblgen -gen-op-defs -I %S/../../include -DSUMMARYVALID %s | FileCheck %s --check-prefix=SUMMARYVALID
// RUN: mlir-tblgen -gen-op-defs -I %S/../../include -DPROPVALID %s | FileCheck %s --check-prefix=PROPVALID
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DPROP %s 2>&1 | FileCheck %s --check-prefix=PROP
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DPROPSUM %s 2>&1 | FileCheck %s --check-prefix=PROPSUM
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DATTRSUM %s 2>&1 | FileCheck %s --check-prefix=ATTRSUM
// RUN: not mlir-tblgen -gen-typedef-defs -I %S/../../include -DASSEMBLY %s 2>&1 | FileCheck %s --check-prefix=ASSEMBLY
// RUN: not mlir-tblgen -gen-typedef-defs -I %S/../../include -DPARAM %s 2>&1 | FileCheck %s --check-prefix=PARAM
// RUN: not mlir-tblgen -gen-enum-defs -I %S/../../include -DENUM %s 2>&1 | FileCheck %s --check-prefix=ENUM
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DSHARED %s 2>&1 | FileCheck %s --check-prefix=SHARED
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DSUMMARY %s 2>&1 | FileCheck %s --check-prefix=SUMMARY
// RUN: not mlir-tblgen -gen-op-defs -I %S/../../include -DINLINE %s 2>&1 | FileCheck %s --check-prefix=INLINE
// RUN: not mlir-tblgen -gen-attrdef-defs -I %S/../../include -DATTR %s 2>&1 | FileCheck %s --check-prefix=ATTR
// RUN: not mlir-tblgen -gen-op-interface-decls -I %S/../../include -DINTERFACE %s 2>&1 | FileCheck %s --check-prefix=INTERFACE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Properties.td"
include "mlir/IR/EnumAttr.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

#ifdef SHARED
// SHARED: format-substitution-errors.td:[[#@LINE+5]]:5: error: cannot substitute '$_builder' while formatting shared type constraint
// SHARED: note: template: ($_builder.getType())
// SHARED: note: available named substitutions: $_op, $_self
// SHARED-NOT: <no-subst-found>
// A type constraint's shared verifier has no builder binding.
def MissingBuilder : Type<CPred<"$_builder.getType()">, "bad type">;
def SharedOp : Op<TestDialect, "shared"> {
  let arguments = (ins MissingBuilder:$input);
}
#endif

#ifdef SUMMARY
// SUMMARY: format-substitution-errors.td:[[#@LINE+5]]:5: error: cannot substitute '$missing' while formatting escaped summary expression
// SUMMARY: note: template: $missing
// SUMMARY: note: while formatting shared type constraint
// SUMMARY: note: available named substitutions: $_op, $_self
// SUMMARY-NOT: <no-subst-found>
def MissingSummary : Type<CPred<"true">, "bad {{$missing}} type">;
def SummaryOp : Op<TestDialect, "summary"> {
  let arguments = (ins MissingSummary:$input);
}
#endif

#ifdef INLINE
// A rejected speculative attribute probe is retried in the full inline context.
// INLINE: format-substitution-errors.td:[[#@LINE+4]]:5: error: cannot substitute '$absent' while formatting attribute predicate
// INLINE: format-substitution-errors.td:[[#@LINE+4]]:5: note: while expanding operation adaptor verifier
// INLINE: note: available named substitutions: $_self, $input, $value
// INLINE-NOT: <no-subst-found>
def InlineAttr : Attr<CPred<"$_self && $input && $absent">, "inline attr">;
def InlineOp : Op<TestDialect, "inline"> {
  let arguments = (ins AnyType:$input, InlineAttr:$value);
}
#endif

#ifdef ATTR
// ATTR: format-substitution-errors.td:[[#@LINE+4]]:5: error: cannot substitute '$missing' while formatting attribute/type template
// ATTR: note: template: $missing
// ATTR: note: available named substitutions: $cppClass
// ATTR-NOT: <no-subst-found>
def BadAttr : AttrDef<TestDialect, "Bad"> {
  let mnemonic = "bad";
  let extraClassDefinition = [{$missing}];
}
#endif

#ifdef INTERFACE
// INTERFACE: format-substitution-errors.td:[[#@LINE+4]]:5: error: cannot substitute '$missing' while formatting interface template
// INTERFACE: note: template: return $missing;
// INTERFACE: note: available named substitutions: $_op
// INTERFACE-NOT: <no-subst-found>
def BadInterface : OpInterface<"BadInterface"> {
  let methods = [InterfaceMethod<"bad default", "int", "test", (ins),
                                 [{}], [{return $missing;}]>];
}
#endif

#ifdef PROP
// The property probe cannot extract an operand-dependent predicate.
// PROP: format-substitution-errors.td:[[#@LINE+3]]:5: error: cannot substitute '$absent' while formatting property predicate
// PROP: note: while expanding operation adaptor verifier
// PROP: note: available named substitutions: $_self, $input
def InlineProp : Property<"int"> {
  let predicate = CPred<"$_self && $input && $absent">;
}
def PropOp : Op<TestDialect, "prop"> {
  let arguments = (ins AnyType:$input, InlineProp:$value);
}
#endif

#ifdef PROPSUM
// PROPSUM: format-substitution-errors.td:[[#@LINE+4]]:5: error: cannot substitute '$missing' while formatting escaped summary expression
// PROPSUM: note: template: $missing
// PROPSUM: note: while formatting property summary
// PROPSUM: note: while expanding operation adaptor verifier
def SummaryProp : Property<"int", "bad {{$missing}} property"> {
  let predicate = CPred<"$_self && $input">;
}
def PropSummaryOp : Op<TestDialect, "prop_summary"> {
  let arguments = (ins AnyType:$input, SummaryProp:$value);
}
#endif

#ifdef ATTRSUM
// ATTRSUM: format-substitution-errors.td:[[#@LINE+3]]:5: error: cannot substitute '$missing' while formatting escaped summary expression
// ATTRSUM: note: while formatting attribute summary
// ATTRSUM: note: while expanding operation adaptor verifier
def SummaryAttr : Attr<CPred<"$_self && $input">, "bad {{$missing}} attribute">;
def AttrSummaryOp : Op<TestDialect, "attr_summary"> {
  let arguments = (ins AnyType:$input, SummaryAttr:$value);
}
#endif

#ifdef ASSEMBLY
// ASSEMBLY: format-substitution-errors.td:[[#@LINE+6]]:5: error: cannot substitute '$missing' while formatting attribute/type assembly format
// ASSEMBLY: note: template: $missing
// ASSEMBLY: note: available named substitutions: $_builder, $_ctxt, $_loc, $_parser
def AssemblyParam : AttrOrTypeParameter<"int", "bad parser"> {
  let parser = "$missing";
}
def BadAssembly : TypeDef<TestDialect, "BadAssembly"> {
  let mnemonic = "assembly";
  let parameters = (ins AssemblyParam:$value);
  let assemblyFormat = "$value";
}
#endif

#ifdef PARAM
// PARAM: format-substitution-errors.td:[[#@LINE+3]]:5: error: cannot substitute '$_builder' while formatting parameter constraint predicate
// PARAM: format-substitution-errors.td:[[#@LINE+3]]:5: note: while expanding attribute/type verifier
// PARAM: note: available named substitutions: $_self
def ParamConstraint : Type<CPred<"$_builder.getType()">, "bad parameter">;
def BadParam : TypeDef<TestDialect, "BadParam"> {
  let mnemonic = "param";
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins ParamConstraint:$value);
}
#endif

#ifdef ENUM
// ENUM: format-substitution-errors.td:[[#@LINE+2]]:5: error: cannot substitute '$missing' while formatting enum attribute predicate
// ENUM: note: available named substitutions: $_self
def EnumBase : Attr<CPred<"$missing">, "bad enum"> {
  let storageType = "::mlir::IntegerAttr";
  let returnType = "int";
  let constBuilderCall = "build($0)";
  let valueType = I32;
}
def BadEnum : I32EnumAttr<"BadEnum", "bad enum", [I32EnumAttrCase<"A", 0>]> {
  let baseAttrClass = EnumBase;
}
#endif

#ifdef PROPVALID
// An operand-dependent property fails the restricted probe, then succeeds inline.
// PROPVALID: if (!((tblgen_value && (*this->getODSOperands(0).begin()))))
// PROPVALID-NOT: <no-subst-found>
def ValidInlineProp : Property<"int"> {
  let predicate = CPred<"$_self && $input">;
}
def ValidPropOp : Op<TestDialect, "valid_prop"> {
  let arguments = (ins AnyType:$input, ValidInlineProp:$value);
}
#endif

#ifdef SUMMARYVALID
// Both predicates fit a shared verifier, but their summaries need an operand.
// SUMMARYVALID: SummaryBindingsOpAdaptor::verify
// SUMMARYVALID: property 'prop' failed to satisfy constraint: needs ") << (*this->getODSOperands(0).begin())
// SUMMARYVALID: attribute 'attr' failed to satisfy constraint: needs ") << (*this->getODSOperands(0).begin())
// SUMMARYVALID-NOT: <no-subst-found>
def OperandSummaryProp : Property<"int", "needs {{$input}}"> {
  let predicate = CPred<"$_self">;
}
def OperandSummaryAttr : Attr<CPred<"$_self">, "needs {{$input}}">;
def SummaryBindingsOp : Op<TestDialect, "summary_bindings"> {
  let arguments = (ins AnyType:$input, OperandSummaryProp:$prop,
                       OperandSummaryAttr:$attr);
}
#endif

#ifdef DEFAULT
// DEFAULT: format-substitution-errors.td:[[#@LINE+3]]:5: error: cannot substitute '$missing' while formatting default value
// DEFAULT: note: template: $missing
// DEFAULT: format-substitution-errors.td:[[#@LINE+2]]:5: note: while expanding operation 'test.default_value'
def BadDefault : DefaultValuedAttr<I32Attr, "$missing">;
def DefaultOp : Op<TestDialect, "default_value"> {
  let arguments = (ins BadDefault:$value);
}
#endif
