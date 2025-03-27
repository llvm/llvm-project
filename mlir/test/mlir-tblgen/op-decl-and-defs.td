// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s
// RUN: mlir-tblgen -gen-op-decls -op-include-regex="test.a_op" -I %S/../../include %s | FileCheck  %s --check-prefix=REDUCE_INC
// RUN: mlir-tblgen -gen-op-decls -op-exclude-regex="test.a_op" -I %S/../../include %s | FileCheck  %s --check-prefix=REDUCE_EXC

// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "NS";
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
    OptionalAttr<F32Attr>:$some_attr2
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
                  OpBuilder<(ins CArg<"int", "0">:$integer)>,
                  DeprecatedOpBuilder<"the deprecation message",
                            (ins "float":$something)>];
  let hasCustomAssemblyFormat = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Display a graph for debugging purposes.
    void displayGraph();
  }];
}

// CHECK-LABEL: NS::AOp declarations

// CHECK: namespace detail {
// CHECK: class AOpGenericAdaptorBase {
// CHECK: public:
// CHECK:   AOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions)
// CHECK:   AOpGenericAdaptorBase(AOp op);
// CHECK:   ::mlir::IntegerAttr getAttr1Attr() {
// CHECK:   uint32_t getAttr1();
// CHECK:   ::mlir::FloatAttr getSomeAttr2Attr() {
// CHECK:   ::std::optional< ::llvm::APFloat > getSomeAttr2();
// CHECK:   ::mlir::Region &getSomeRegion() {
// CHECK:   ::mlir::RegionRange getSomeRegions() {
// CHECK: };
// CHECK: }

// CHECK: template <typename RangeT>
// CHECK: class AOpGenericAdaptor : public detail::AOpGenericAdaptorBase {
// CHECK: public:
// CHECK:   AOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}
// CHECK:   AOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}
// CHECK:   AOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AOpGenericAdaptor(values, attrs, Properties{}, {}) {}
// CHECK:   AOpGenericAdaptor(RangeT values, const AOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}
// CHECK:   RangeT getODSOperands(unsigned index) {
// CHECK:   ValueT getA() {
// CHECK:   RangeT getB() {
// CHECK: private:
// CHECK:   RangeT odsOperands;
// CHECK: };

// CHECK: class AOpAdaptor : public AOpGenericAdaptor<::mlir::ValueRange> {
// CHECK: public:
// CHECK:   AOpAdaptor(AOp
// CHECK:   ::llvm::LogicalResult verify(
// CHECK: };

// CHECK: class AOp : public ::mlir::Op<AOp, ::mlir::OpTrait::AtLeastNRegions<1>::Impl, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::OpTrait::IsIsolatedFromAbove
// CHECK-NOT: ::mlir::OpTrait::IsIsolatedFromAbove
// CHECK: public:
// CHECK:   using Op::Op;
// CHECK:   using Adaptor = AOpAdaptor;
// CHECK:   static constexpr ::llvm::StringLiteral getOperationName() {
// CHECK:     return ::llvm::StringLiteral("test.a_op");
// CHECK:   }
// CHECK:   ::mlir::Operation::operand_range getODSOperands(unsigned index) {
// CHECK:   ::mlir::TypedValue<::mlir::IntegerType> getA() {
// CHECK:   ::mlir::Operation::operand_range getB() {
// CHECK:   ::mlir::OpOperand &getAMutable() {
// CHECK:   ::mlir::MutableOperandRange getBMutable();
// CHECK:   ::mlir::Operation::result_range getODSResults(unsigned index) {
// CHECK:   ::mlir::TypedValue<::mlir::IntegerType> getR() {
// CHECK:   ::mlir::Region &getSomeRegion() {
// CHECK:   ::mlir::MutableArrayRef<::mlir::Region> getSomeRegions() {
// CHECK:   ::mlir::IntegerAttr getAttr1Attr() {
// CHECK:   uint32_t getAttr1();
// CHECK:   ::mlir::FloatAttr getSomeAttr2Attr() {
// CHECK:   ::std::optional< ::llvm::APFloat > getSomeAttr2();
// CHECK:   ::mlir::Attribute removeSomeAttr2Attr() {
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value val);
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int integer = 0);
// CHECK{LITERAL}:   [[deprecated("the deprecation message")]]
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, float something);
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type r, ::mlir::TypeRange s, ::mlir::Value a, ::mlir::ValueRange b, ::mlir::IntegerAttr attr1, /*optional*/::mlir::FloatAttr some_attr2, unsigned someRegionsCount)
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::ValueRange b, ::mlir::IntegerAttr attr1, /*optional*/::mlir::FloatAttr some_attr2, unsigned someRegionsCount);
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type r, ::mlir::TypeRange s, ::mlir::Value a, ::mlir::ValueRange b, uint32_t attr1, /*optional*/::mlir::FloatAttr some_attr2, unsigned someRegionsCount)
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::ValueRange b, uint32_t attr1, /*optional*/::mlir::FloatAttr some_attr2, unsigned someRegionsCount);
// CHECK:   static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions)
// CHECK:   static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes, unsigned numRegions)
// CHECK:   static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
// CHECK:   void print(::mlir::OpAsmPrinter &p);
// CHECK:   ::llvm::LogicalResult verifyInvariants();
// CHECK:   static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
// CHECK:   ::llvm::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
// CHECK:   // Display a graph for debugging purposes.
// CHECK:   void displayGraph();
// CHECK: };

// DEFS-LABEL: NS::AOp definitions

// Check that `getAttrDictionary()` is used when not using properties.

// DECLS: ::mlir::RegionRange AOpGenericAdaptorBase::getSomeRegions()
// DECLS-NEXT: return odsRegions.drop_front(1);
// DECLS: ::mlir::RegionRange AOpGenericAdaptorBase::getRegions()

// Check AttrSizedOperandSegments
// ---

def NS_AttrSizedOperandOp : NS_Op<"attr_sized_operands",
                                 [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<I32>:$a,
    Variadic<I32>:$b,
    I32:$c,
    Variadic<I32>:$d
  );
}

// CHECK-LABEL: class AttrSizedOperandOpGenericAdaptorBase {
// CHECK: ::llvm::ArrayRef<int32_t> getOperandSegmentSizes() const

// CHECK-LABEL: AttrSizedOperandOpGenericAdaptor(
// CHECK-SAME:    RangeT values
// CHECK-SAME:    ::mlir::DictionaryAttr attrs
// CHECK:  RangeT getA() {
// CHECK:  RangeT getB() {
// CHECK:  ValueT getC() {
// CHECK:  RangeT getD() {

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
// CHECK:   ::mlir::TypedValue<::mlir::IntegerType> getA() {
// CHECK:   ::mlir::MutableOperandRange getAMutable();
// CHECK:   ::mlir::TypedValue<::mlir::FloatType> getB() {
// CHECK:   static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type b, /*optional*/::mlir::Value a)

// Check that all types match constraint results in generating builder.
// ---

def NS_FOp : NS_Op<"op_with_all_types_constraint",
    [AllTypesMatch<["a", "b"]>]> {
  let arguments = (ins AnyType:$a);
  let results = (outs Res<AnyType, "output b", []>:$b);
}

// CHECK-LABEL: class FOp :
// CHECK: static ::llvm::LogicalResult inferReturnTypes

// DEFS: void FOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a) {
// DEFS:   if (::mlir::succeeded(FOp::inferReturnTypes(odsBuilder.getContext(),
// DEFS:   else
// DEFS:     ::mlir::detail::reportFatalInferReturnTypesError(odsState);

def NS_GOp : NS_Op<"op_with_fixed_return_type", []> {
  let arguments = (ins AnyType:$a);
  let results = (outs I32:$b);
}

// CHECK-LABEL: class GOp :
// CHECK: static ::llvm::LogicalResult inferReturnTypes

// Check default value for collective params builder. Check that other builders
// are generated as well.
def NS_HCollectiveParamsOp : NS_Op<"op_collective_params", []> {
  let arguments = (ins AnyType:$a);
  let results = (outs AnyType:$b);
}

// CHECK-LABEL: class HCollectiveParamsOp :
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type b, ::mlir::Value a);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a);
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {})
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {})

// Check suppression of "separate arg, separate result" build method for an op
// with single variadic arg and single variadic result (since it will be
// ambiguous with the collective params build method).
def NS_HCollectiveParamsSuppress0Op : NS_Op<"op_collective_suppress0", []> {
  let arguments = (ins Variadic<I32>:$a);
  let results = (outs Variadic<I32>:$b);
}

// CHECK-LABEL: class HCollectiveParamsSuppress0Op :
// CHECK-NOT: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange b, ::mlir::ValueRange a);
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

// Check suppression of "separate arg, collective result" build method for an op
// with single variadic arg and non variadic result (since it will be
// ambiguous with the collective params build method).
def NS_HCollectiveParamsSuppress1Op : NS_Op<"op_collective_suppress1", []> {
  let arguments = (ins Variadic<I32>:$a);
  let results = (outs I32:$b);
}

// CHECK-LABEL: class HCollectiveParamsSuppress1Op :
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
// CHECK-LABEL: class HCollectiveParamsSuppress2Op :
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange b, ::mlir::TypeRange c, ::mlir::ValueRange a);
// CHECK-NOT: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange b, ::mlir::ValueRange a);
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

// Check default value of `attributes` for the `genUseOperandAsResultTypeCollectiveParamBuilder` builder
def NS_IOp : NS_Op<"op_with_same_operands_and_result_types_trait", [SameOperandsAndResultType]> {
  let arguments = (ins AnyType:$a, AnyType:$b);
  let results = (outs AnyType:$r);
}
// CHECK-LABEL: class IOp :
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type r, ::mlir::Value a, ::mlir::Value b);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b);
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});

// Check default value of `attributes` for the `genInferredTypeCollectiveParamBuilder` builder
def NS_JOp : NS_Op<"op_with_InferTypeOpInterface_interface", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyType:$a, AnyType:$b);
  let results = (outs AnyType:$r);
}
// CHECK-LABEL: class JOp :
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type r, ::mlir::Value a, ::mlir::Value b);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b);
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});

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

def NS_LOp : NS_Op<"op_with_same_operands_and_result_types_unwrapped_attr", [SameOperandsAndResultType]> {
  let arguments = (ins AnyType:$a, AnyType:$b, I32Attr:$attr1);
  let results = (outs AnyType:$r);
}
// CHECK-LABEL: class LOp :
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type r, ::mlir::Value a, ::mlir::Value b, ::mlir::IntegerAttr attr1);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::IntegerAttr attr1);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::IntegerAttr attr1);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type r, ::mlir::Value a, ::mlir::Value b, uint32_t attr1);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, uint32_t attr1);
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, uint32_t attr1);
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
// CHECK: static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
// CHECK: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});

def NS_MOp : NS_Op<"op_with_single_result_and_fold_adaptor_fold", []> {
  let results = (outs AnyType:$res);

  let hasFolder = 1;
}

// CHECK-LABEL: class MOp :
// CHECK: ::mlir::OpFoldResult fold(FoldAdaptor adaptor);

def NS_NOp : NS_Op<"op_with_properties", []> {
  let arguments = (ins Property<"unsigned">:$value);
}

// DEFS: NOpGenericAdaptorBase::NOpGenericAdaptorBase(NOp op) :
// DEFS-SAME: odsAttrs(op->getRawDictionaryAttrs())
// DEFS-SAME: odsOpName(op->getName())
// DEFS-SAME: properties(op.getProperties())
// DEFS-SAME: odsRegions(op->getRegions())

// Test that type defs have the proper namespaces when used as a constraint.
// ---

def Test_Dialect2 : Dialect {
  let name = "test";
  let cppNamespace = "::mlir::dialect2";
}
def TestDialect2Type : TypeDef<Test_Dialect2, "Dialect2Type"> {
  let typeName = "test.type";
}

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

def NS_VarOfVarOperandOp : NS_Op<"var_of_var_operand", []> {
  let arguments = (ins
    VariadicOfVariadic<F32, "var_size">:$var_of_var_attr,
    DenseI32ArrayAttr:$var_size
  );
}

// CHECK-LABEL: class VarOfVarOperandOpGenericAdaptor
// CHECK: public:
// CHECK: ::llvm::SmallVector<RangeT> getVarOfVarAttr() {


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
