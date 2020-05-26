// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck --dump-input-on-failure %s

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "NS";
}
class NS_Op<string mnemonic, list<OpTrait> traits> :
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
  let builders = [OpBuilder<"Value val">];
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

// CHECK: class AOp;

// CHECK-LABEL: NS::AOp declarations

// CHECK: class AOpOperandAdaptor {
// CHECK: public:
// CHECK:   AOpOperandAdaptor(ArrayRef<Value> values
// CHECK:   ArrayRef<Value> getODSOperands(unsigned index);
// CHECK:   Value a();
// CHECK:   ArrayRef<Value> b();
// CHECK:   IntegerAttr attr1();
// CHECL:   FloatAttr attr2();
// CHECK: private:
// CHECK:   ArrayRef<Value> odsOperands;
// CHECK: };

// CHECK: class AOp : public Op<AOp, OpTrait::AtLeastNRegions<1>::Impl, OpTrait::AtLeastNResults<1>::Impl, OpTrait::ZeroSuccessor, OpTrait::AtLeastNOperands<1>::Impl, OpTrait::IsIsolatedFromAbove
// CHECK-NOT: OpTrait::IsIsolatedFromAbove
// CHECK: public:
// CHECK:   using Op::Op;
// CHECK:   using OperandAdaptor = AOpOperandAdaptor;
// CHECK:   static StringRef getOperationName();
// CHECK:   Operation::operand_range getODSOperands(unsigned index);
// CHECK:   Value a();
// CHECK:   Operation::operand_range b();
// CHECK:   ::mlir::MutableOperandRange aMutable();
// CHECK:   ::mlir::MutableOperandRange bMutable();
// CHECK:   Operation::result_range getODSResults(unsigned index);
// CHECK:   Value r();
// CHECK:   Region &someRegion();
// CHECK:   MutableArrayRef<Region> someRegions();
// CHECK:   IntegerAttr attr1Attr()
// CHECK:   APInt attr1();
// CHECK:   FloatAttr attr2Attr()
// CHECK:   Optional< APFloat > attr2();
// CHECK:   static void build(Value val);
// CHECK:   static void build(OpBuilder &odsBuilder, OperationState &odsState, Type r, ArrayRef<Type> s, Value a, ValueRange b, IntegerAttr attr1, /*optional*/FloatAttr attr2, unsigned someRegionsCount)
// CHECK:   static void build(OpBuilder &odsBuilder, OperationState &odsState, Type r, ArrayRef<Type> s, Value a, ValueRange b, APInt attr1, /*optional*/FloatAttr attr2, unsigned someRegionsCount)
// CHECK:   static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes, unsigned numRegions)
// CHECK:   static ParseResult parse(OpAsmParser &parser, OperationState &result);
// CHECK:   void print(OpAsmPrinter &p);
// CHECK:   LogicalResult verify();
// CHECK:   static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);
// CHECK:   LogicalResult fold(ArrayRef<Attribute> operands, SmallVectorImpl<OpFoldResult> &results);
// CHECK:   // Display a graph for debugging purposes.
// CHECK:   void displayGraph();
// CHECK: };

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

// CHECK-LABEL: AttrSizedOperandOpOperandAdaptor(
// CHECK-SAME:    ArrayRef<Value> values
// CHECK-SAME:    DictionaryAttr attrs
// CHECK:  ArrayRef<Value> a();
// CHECK:  ArrayRef<Value> b();
// CHECK:  Value c();
// CHECK:  ArrayRef<Value> d();
// CHECK:  DenseIntElementsAttr operand_segment_sizes();

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
// CHECK:   Value a();
// CHECK:   ::mlir::MutableOperandRange aMutable();
// CHECK:   Value b();
// CHECK:   static void build(OpBuilder &odsBuilder, OperationState &odsState, /*optional*/Type b, /*optional*/Value a)

// Check that default builders can be suppressed.
// ---

def NS_SkipDefaultBuildersOp : NS_Op<"skip_default_builders", []> {
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"Value val">];
}

// CHECK-LABEL: NS::SkipDefaultBuildersOp declarations
// CHECK:     class SkipDefaultBuildersOp
// CHECK-NOT:   static void build(Builder
// CHECK:       static void build(Value

// Check leading underscore in op name
// ---

def NS__AOp : NS_Op<"_op_with_leading_underscore", []>;

// CHECK-LABEL: NS::_AOp declarations
// CHECK: class _AOp : public Op<_AOp

def _BOp : NS_Op<"_op_with_leading_underscore_and_no_namespace", []>;

// CHECK-LABEL: _BOp declarations
// CHECK: class _BOp : public Op<_BOp

