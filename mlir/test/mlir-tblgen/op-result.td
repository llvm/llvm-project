// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Test_Dialect : Dialect {
  let name = "test";

  // TODO: Flip to _Prefixed
  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}
class NS_Op<string mnemonic, list<Trait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

def OpA : NS_Op<"one_normal_result_op", []> {
  let results = (outs I32:$result);
}

// CHECK-LABEL: void OpA::build
// CHECK:         ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands
// CHECK:         assert(resultTypes.size() == 1u && "mismatched number of return types");
// CHECK-NEXT:    odsState.addTypes(resultTypes);

def OpB : NS_Op<"same_input_output_type_op", [SameOperandsAndResultType]> {
  let arguments = (ins I32:$x);
  let results = (outs I32:$y);
}

// CHECK-LABEL: OpB definitions
// CHECK: void OpB::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type y, ::mlir::Value x)
// CHECK:   odsState.addTypes(y);
// CHECK: void OpB::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x)
// CHECK:   ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
// CHECK:   if (::mlir::succeeded(OpB::inferReturnTypes(odsBuilder.getContext(),
// CHECK:             odsState.location, odsState.operands,
// CHECK:             odsState.attributes.getDictionary(odsState.getContext()),
// CHECK:             /*regions=*/{}, inferredReturnTypes)))
// CHECK:     odsState.addTypes(inferredReturnTypes);

def OpC : NS_Op<"three_normal_result_op", []> {
  let results = (outs I32:$x, /*unnamed*/I32, I32:$z);
}

// CHECK-LABEL: OpC definitions
// CHECK:       void OpC::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type x, ::mlir::Type resultType1, ::mlir::Type z)
// CHECK-NEXT:   odsState.addTypes(x)
// CHECK-NEXT:   odsState.addTypes(resultType1)
// CHECK-NEXT:   odsState.addTypes(z)

// CHECK:      void OpC::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
// CHECK-NEXT:   assert(resultTypes.size() == 3u && "mismatched number of results");
// CHECK-NEXT:   odsState.addTypes(resultTypes);

def IntegerTypeAttr : TypeAttrBase<"IntegerType", "Integer type attribute">;
def OpD : NS_Op<"type_attr_as_result_type", [FirstAttrDerivedResultType]> {
  let arguments = (ins I32:$x, IntegerTypeAttr:$attr, F32Attr:$f32);
  let results = (outs AnyTensor:$y);
}

// CHECK-LABEL: OpD definitions
// CHECK: void OpD::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes)
// CHECK: odsState.addTypes({attr.getValue().cast<::mlir::TypeAttr>().getValue()});

def OpE : NS_Op<"value_attr_as_result_type", [FirstAttrDerivedResultType]> {
  let arguments = (ins I32:$x, F32Attr:$attr);
  let results = (outs AnyTensor:$y);
}

// CHECK-LABEL: OpE definitions
// CHECK: void OpE::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes)
// CHECK: odsState.addTypes({attr.getValue().getType()});

def OpF : NS_Op<"one_variadic_result_op", []> {
  let results = (outs Variadic<I32>:$x);
}

// CHECK-LABEL: void OpF::build
// CHECK-SAME:    ::mlir::TypeRange x
// CHECK-NOT:     assert
// CHECK:         odsState.addTypes(x);

def OpG : NS_Op<"one_normal_and_one_variadic_result_op", []> {

  let results = (outs I32:$x, Variadic<I32>:$y);
}

// CHECK-LABEL: OpG definitions

// CHECK:      void OpG::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type x, ::mlir::TypeRange y)
// CHECK-NEXT:   odsState.addTypes(x);
// CHECK-NEXT:   odsState.addTypes(y);

// CHECK:       void OpG::build
// CHECK:         ::mlir::TypeRange resultTypes
// CHECK:         assert(resultTypes.size() >= 1u && "mismatched number of return types");
// CHECK-NEXT:    odsState.addTypes(resultTypes);

def OpI : NS_Op<"mix_variadic_and_normal_results_op", [SameVariadicResultSize]> {
  let results = (outs Variadic<AnyTensor>:$output1, AnyTensor:$output2, Variadic<AnyTensor>:$output3);
}

// CHECK-LABEL: ::mlir::Operation::result_range OpI::output1
// CHECK-NEXT:    return getODSResults(0);

// CHECK-LABEL: ::mlir::Value OpI::output2
// CHECK-NEXT:    return *getODSResults(1).begin();

// CHECK-LABEL: OpI::build
// CHECK-NEXT:    odsState.addTypes(output1);
// CHECK-NEXT:    odsState.addTypes(output2);
// CHECK-NEXT:    odsState.addTypes(output3);

// Test that if the only operand is variadic, we access the first value in the
// pack to set result type
// ---
def OpK : NS_Op<"only_input_is_variadic_with_same_value_type_op", [SameOperandsAndResultType]> {
  let arguments = (ins Variadic<AnyTensor>:$input);
  let results = (outs AnyTensor:$result);
}

// CHECK-LABEL: OpK::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes)
// CHECK: odsState.addTypes({operands[0].getType()});

// Test with inferred shapes and interleaved with operands/attributes.
//
def OpL1 : NS_Op<"op_with_all_types_constraint",
    [AllTypesMatch<["a", "b"]>]> {
  let arguments = (ins I32Attr:$attr1, AnyType:$a);
  let results = (outs Res<AnyType, "output b", []>:$b);
}

// CHECK-LABEL: LogicalResult OpL1::inferReturnTypes
// CHECK-NOT: }
// CHECK: ::mlir::Type odsInferredType0 = operands[0].getType();
// CHECK: inferredReturnTypes[0] = odsInferredType0;

def OpL2 : NS_Op<"op_with_all_types_constraint",
    [AllTypesMatch<["c", "b"]>, AllTypesMatch<["a", "d"]>]> {
  let arguments = (ins I32Attr:$attr1, AnyType:$a, AnyType:$a2, AnyType:$c);
  let results = (outs Res<AnyType, "output b", []>:$b, AnyType:$d);
}

// CHECK-LABEL: LogicalResult OpL2::inferReturnTypes
// CHECK-NOT: }
// CHECK: ::mlir::Type odsInferredType0 = operands[2].getType();
// CHECK: ::mlir::Type odsInferredType1 = operands[0].getType();
// CHECK: inferredReturnTypes[0] = odsInferredType0;
// CHECK: inferredReturnTypes[1] = odsInferredType1;

def OpL3 : NS_Op<"op_with_all_types_constraint",
    [AllTypesMatch<["a", "b"]>]> {
  let arguments = (ins I32Attr:$a);
  let results = (outs AnyType:$b);
}

// CHECK-LABEL: LogicalResult OpL3::inferReturnTypes
// CHECK-NOT: }
// CHECK: ::mlir::Type odsInferredType0 = attributes.get("a").getType();
// CHECK: inferredReturnTypes[0] = odsInferredType0;
