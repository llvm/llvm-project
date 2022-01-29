// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s
// RUN: mlir-tblgen -gen-op-decls -op-include-regex="test.a_op" -I %S/../../include %s | FileCheck  %s --check-prefix=REDUCE_INC
// RUN: mlir-tblgen -gen-op-decls -op-exclude-regex="test.a_op" -I %S/../../include %s | FileCheck  %s --check-prefix=REDUCE_EXC

// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEFS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "NS";
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}
class NS_Op<string mnemonic, list<Trait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

// IsolatedFromAbove trait is included twice to ensure it gets uniqued during
// emission.
def NS_AOp : NS_Op<"a_op", [IsolatedFromAbove, IsolatedFromAbove]> {
  let arguments = (ins
    I32:$a,
    Variadic<F32>:$b,

    I32Attr:$attr1,
    OptionalAttr<F32Attr>:$attr2
  );

  let results = (outs
    I32:$r,
    Variadic<F32>:$s
  );

  let regions = (region
    AnyRegion:$someRegion,
    VariadicRegion<AnyRegion>:$someRegions
  );
  let builders = [OpBuilder<(ins "Value":$val)>,
                  OpBuilder<(ins CArg<"int", "0">:$integer)>];
  let parser = [{ foo }];
  let printer = [{ bar }];
  let verifier = [{ baz }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
    // Display a graph for debugging purposes.
    void displayGraph();
  }];
}

// CHECK-LABEL: NS::AOp declarations

// CHECK: class AOpAdaptor {
// CHECK: public:
// CHECK:   AOpAdaptor(::mlir::ValueRange values
// CHECK:   ::mlir::ValueRange getODSOperands(unsigned index);
// CHECK:   ::mlir::Value getA();
// CHECK:   ::mlir::ValueRange getB();
// CHECK:   ::mlir::IntegerAttr getAttr1Attr();
// CHECK:   uint32_t getAttr1();
// CHECK:   ::mlir::FloatAttr getAttr2Attr();
// CHECK:   ::llvm::Optional< ::llvm::APFloat > getAttr2();
// CHECK:   ::mlir::Region &getSomeRegion();
// CHECK:   ::mlir::RegionRange getSomeRegions();
// CHECK: private:
// CHECK:   ::mlir::ValueRange odsOperands;
// CHECK: };

// CHECK: class AOp : public ::mlir::Op<AOp, ::mlir::OpTrait::AtLeastNRegions<1>::Impl, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::IsIsolatedFromAbove
// CHECK-NOT: ::mlir::OpTrait::IsIsolatedFromAbove
// CHECK: public:
// CHECK:   using Op::Op;
// CHECK:   using Adaptor = AOpAdaptor;
// CHECK:   static constexpr ::llvm::StringLiteral getOperationName() {
// CHECK:     return ::llvm::StringLiteral("test.a_op");
// CHECK:   }
// CHECK:   ::mlir::Operation::operand_range getODSOperands(unsigned index);
// CHECK:   ::mlir::Value getA();
// CHECK:   ::mlir::Operation::operand_range getB();
// CHECK:   ::mlir::MutableOperandRange getAMutable();
// CHECK:   ::mlir::MutableOperandRange getBMutable();
// CHECK:   ::mlir::Operation::result_range getODSResults(unsigned index);
// CHECK:   ::mlir::Value getR();
// CHECK:   ::mlir::Region &getSomeRegion();
// CHECK:   ::mlir::MutableArrayRef<::mlir::Region> getSomeRegions();
// CHECK:   ::mlir::IntegerAttr getAttr1Attr()
// CHECK:   uint32_t getAttr1();
// CHECK:   ::mlir::FloatAttr getAttr2Attr()
// CHECK:   ::llvm::Optional< ::llvm::APFloat > getAttr2();
// CHECK:   ::mlir::Attribute removeAttr2Attr();
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value val);
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int integer = 0);
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type r, ::mlir::TypeRange s, ::mlir::Value a, ::mlir::ValueRange b, ::mlir::IntegerAttr attr1, /*optional*/::mlir::FloatAttr attr2, unsigned someRegionsCount)
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type r, ::mlir::TypeRange s, ::mlir::Value a, ::mlir::ValueRange b, uint32_t attr1, /*optional*/::mlir::FloatAttr attr2, unsigned someRegionsCount)
// CHECK:   static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions)
// CHECK:   static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
// CHECK:   void print(::mlir::OpAsmPrinter &p);
// CHECK:   ::mlir::LogicalResult verify();
// CHECK:   static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
// CHECK:   ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
// CHECK:   // Display a graph for debugging purposes.
// CHECK:   void displayGraph();
// CHECK: };

// DEFS-LABEL: NS::AOp definitions

// DEFS: AOpAdaptor::AOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions)
// DEFS: ::mlir::RegionRange AOpAdaptor::getRegions()
// DEFS: ::mlir::RegionRange AOpAdaptor::getSomeRegions()
// DEFS-NEXT: return odsRegions.drop_front(1);

// Check AttrSizedOperandSegments
// ---

def NS_AttrSizedOperandOp : NS_Op<"attr_sized_operands",
                                 [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<I32>:$a,
    Variadic<I32>:$b,
    I32:$c,
    Variadic<I32>:$d,
    I32ElementsAttr:$operand_segment_sizes
  );
}

// CHECK-LABEL: AttrSizedOperandOpAdaptor(
// CHECK-SAME:    ::mlir::ValueRange values
// CHECK-SAME:    ::mlir::DictionaryAttr attrs
// CHECK:  ::mlir::ValueRange getA();
// CHECK:  ::mlir::ValueRange getB();
// CHECK:  ::mlir::Value getC();
// CHECK:  ::mlir::ValueRange getD();
// CHECK:  ::mlir::DenseIntElementsAttr getOperandSegmentSizes();

// Check op trait for different number of operands
// ---

def NS_BOp : NS_Op<"op_with_no_operand", []> {
  let arguments = (ins);
}

// CHECK-LABEL: NS::BOp declarations
// CHECK: OpTrait::ZeroOperands

def NS_COp : NS_Op<"op_with_one_operand", []> {
  let arguments = (ins I32:$operand);
}

// CHECK-LABEL: NS::COp declarations
// CHECK: OpTrait::OneOperand

def NS_DOp : NS_Op<"op_with_two_operands", []> {
  let arguments = (ins I32:$input1, I32:$input2);
}

// CHECK-LABEL: NS::DOp declarations
// CHECK: OpTrait::NOperands<2>::Impl

def NS_EOp : NS_Op<"op_with_optionals", []> {
  let arguments = (ins Optional<I32>:$a);
  let results = (outs Optional<F32>:$b);
}

// CHECK-LABEL: NS::EOp declarations
// CHECK:   ::mlir::Value getA();
// CHECK:   ::mlir::MutableOperandRange getAMutable();
// CHECK:   ::mlir::Value getB();
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type b, /*optional*/::mlir::Value a)

// Check that all types match constraint results in generating builder.
// ---

def NS_FOp : NS_Op<"op_with_all_types_constraint",
    [AllTypesMatch<["a", "b"]>]> {
  let arguments = (ins AnyType:$a);
  let results = (outs Res<AnyType, "output b", []>:$b);
}

// CHECK-LABEL: class FOp :
// CHECK: static ::mlir::LogicalResult inferReturnTypes

def NS_GOp : NS_Op<"op_with_fixed_return_type", []> {
  let arguments = (ins AnyType:$a);
  let results = (outs I32:$b);
}

// CHECK-LABEL: class GOp :
// CHECK: static ::mlir::LogicalResult inferReturnTypes

// Check default value for collective params builder. Check that other builders
// are generated as well.
def NS_HCollectiveParamsOp : NS_Op<"op_collective_params", []> {
  let arguments = (ins AnyType:$a);
  let results = (outs AnyType:$b);
}

// CHECK_LABEL: class NS_HCollectiveParamsOp :
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type b, ::mlir::Value a);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a);
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {})

// Check suppression of "separate arg, separate result" build method for an op
// with single variadic arg and single variadic result (since it will be
// ambiguous with the collective params build method).
def NS_HCollectiveParamsSuppress0Op : NS_Op<"op_collective_suppress0", []> {
  let arguments = (ins Variadic<I32>:$a);
  let results = (outs Variadic<I32>:$b);
}

// CHECK_LABEL: class NS_HCollectiveParamsSuppress0Op :
// CHECK-NOT: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange b, ::mlir::ValueRange a);
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

// Check suppression of "separate arg, collective result" build method for an op
// with single variadic arg and non variadic result (since it will be
// ambiguous with the collective params build method).
def NS_HCollectiveParamsSuppress1Op : NS_Op<"op_collective_suppress1", []> {
  let arguments = (ins Variadic<I32>:$a);
  let results = (outs I32:$b);
}

// CHECK_LABEL: class NS_HCollectiveParamsSuppress1Op :
// CHECK-NOT: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange b, ::mlir::ValueRange a);
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

// Check suppression of "separate arg, collective result" build method for an op
// with single variadic arg and > 1 variadic result (since it will be
// ambiguous with the collective params build method). Note that "separate arg,
// separate result" build method should be generated in this case as its not
// ambiguous with the collective params build method.
def NS_HCollectiveParamsSuppress2Op : NS_Op<"op_collective_suppress2", [SameVariadicResultSize]> {
  let arguments = (ins Variadic<I32>:$a);
  let results = (outs Variadic<I32>:$b, Variadic<F32>:$c);
}
// CHECK_LABEL: class NS_HCollectiveParamsSuppress2Op :
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange b, ::mlir::TypeRange c, ::mlir::ValueRange a);
// CHECK-NOT: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange b, ::mlir::ValueRange a);
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

// Check default value of `attributes` for the `genUseOperandAsResultTypeCollectiveParamBuilder` builder
def NS_IOp : NS_Op<"op_with_same_operands_and_result_types_trait", [SameOperandsAndResultType]> {
  let arguments = (ins AnyType:$a, AnyType:$b);
  let results = (outs AnyType:$r);
}
// CHECK_LABEL: class NS_IOp :
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type r, ::mlir::Value a, ::mlir::Value b);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b);
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

// Check default value of `attributes` for the `genInferredTypeCollectiveParamBuilder` builder
def NS_JOp : NS_Op<"op_with_InferTypeOpInterface_interface", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyType:$a, AnyType:$b);
  let results = (outs AnyType:$r);
}
// CHECK_LABEL: class NS_JOp :
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type r, ::mlir::Value a, ::mlir::Value b);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b);
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

// Test usage of TraitList getting flattened during emission.
def NS_KOp : NS_Op<"k_op", [IsolatedFromAbove,
    TraitList<[DeclareOpInterfaceMethods<InferTypeOpInterface>]>]> {
  let arguments = (ins AnyType:$a, AnyType:$b);
}

// CHECK: class KOp : public ::mlir::Op<KOp,
// CHECK-SAME: ::mlir::OpTrait::IsIsolatedFromAbove, ::mlir::InferTypeOpInterface::Trait

// Check native OpTrait usage
// ---

def NS_TestTrait : NativeOpTrait<"TestTrait"> {
    let cppNamespace = "SomeNamespace";
}

def NS_KWithTraitOp : NS_Op<"KWithTrait", [NS_TestTrait]>;

// CHECK-LABEL: NS::KWithTraitOp declarations
// CHECK:       class KWithTraitOp : public ::mlir::Op<KWithTraitOp
// CHECK-SAME:      SomeNamespace::TestTrait

// Test that type defs have the proper namespaces when used as a constraint.
// ---

def Test_Dialect2 : Dialect {
  let name = "test";
  let cppNamespace = "::mlir::dialect2";
}
def TestDialect2Type : TypeDef<Test_Dialect2, "Dialect2Type">;

def NS_ResultWithDialectTypeOp : NS_Op<"op_with_dialect_type", []> {
  let results = (outs TestDialect2Type);
}

// CHECK-LABEL: NS::ResultWithDialectTypeOp declarations
// CHECK:     class ResultWithDialectTypeOp :
// CHECK-SAME: ::mlir::OpTrait::OneTypedResult<::mlir::dialect2::Dialect2TypeType>

// Check that default builders can be suppressed.
// ---

def NS_SkipDefaultBuildersOp : NS_Op<"skip_default_builders", []> {
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$val)>];
}

// CHECK-LABEL: NS::SkipDefaultBuildersOp declarations
// CHECK:     class SkipDefaultBuildersOp
// CHECK-NOT:   static void build(::mlir::Builder
// CHECK:       static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value

// Check leading underscore in op name
// ---

def NS__AOp : NS_Op<"_op_with_leading_underscore", []>;

// CHECK-LABEL: NS::_AOp declarations
// CHECK: class _AOp : public ::mlir::Op<_AOp

def _BOp : NS_Op<"_op_with_leading_underscore_and_no_namespace", []>;

// CHECK-LABEL: _BOp declarations
// CHECK: class _BOp : public ::mlir::Op<_BOp

// REDUCE_INC-LABEL: NS::AOp declarations
// REDUCE_INC-NOT: NS::BOp declarations

// REDUCE_EXC-NOT: NS::AOp declarations
// REDUCE_EXC-LABEL: NS::BOp declarations
