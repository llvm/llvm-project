// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL --dump-input-on-failure
// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF --dump-input-on-failure
// RUN: mlir-tblgen -print-records -I %S/../../include %s | FileCheck %s --check-prefix=RECORD --dump-input-on-failure

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}
class NS_Op<string mnemonic, list<OpTrait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

def SomeAttr : Attr<CPred<"some-condition">, "some attribute kind"> {
  let storageType = "some-attr-kind";
  let returnType = "some-return-type";
  let convertFromStorage = "$_self.some-convert-from-storage()";
  let constBuilderCall = "some-const-builder-call($_builder, $0)";
}

// Test required, optional, default-valued attributes
// ---

def AOp : NS_Op<"a_op", []> {
  let arguments = (ins
      SomeAttr:$aAttr,
      DefaultValuedAttr<SomeAttr, "4.2">:$bAttr,
      OptionalAttr<SomeAttr>:$cAttr
  );
}

// DEF-LABEL: AOp definitions

// Test verify method
// ---

// DEF:      LogicalResult AOpOperandAdaptor::verify
// DEF:      auto tblgen_aAttr = odsAttrs.get("aAttr");
// DEF-NEXT: if (!tblgen_aAttr) return emitError(loc, "'test.a_op' op ""requires attribute 'aAttr'");
// DEF:        if (!((some-condition))) return emitError(loc, "'test.a_op' op ""attribute 'aAttr' failed to satisfy constraint: some attribute kind");
// DEF:      auto tblgen_bAttr = odsAttrs.get("bAttr");
// DEF-NEXT: if (tblgen_bAttr) {
// DEF-NEXT:   if (!((some-condition))) return emitError(loc, "'test.a_op' op ""attribute 'bAttr' failed to satisfy constraint: some attribute kind");
// DEF:      auto tblgen_cAttr = odsAttrs.get("cAttr");
// DEF-NEXT: if (tblgen_cAttr) {
// DEF-NEXT:   if (!((some-condition))) return emitError(loc, "'test.a_op' op ""attribute 'cAttr' failed to satisfy constraint: some attribute kind");

// Test getter methods
// ---

// DEF:      some-attr-kind AOp::aAttrAttr()
// DEF-NEXT:   this->getAttr("aAttr").cast<some-attr-kind>()
// DEF:      some-return-type AOp::aAttr() {
// DEF-NEXT:   auto attr = aAttrAttr()
// DEF-NEXT:   return attr.some-convert-from-storage();

// DEF:      some-attr-kind AOp::bAttrAttr()
// DEF-NEXT:   return this->getAttr("bAttr").dyn_cast_or_null<some-attr-kind>()
// DEF:      some-return-type AOp::bAttr() {
// DEF-NEXT:   auto attr = bAttrAttr();
// DEF-NEXT:   if (!attr)
// DEF-NEXT:       return some-const-builder-call(mlir::Builder(this->getContext()), 4.2).some-convert-from-storage();
// DEF-NEXT:   return attr.some-convert-from-storage();

// DEF:      some-attr-kind AOp::cAttrAttr()
// DEF-NEXT:   return this->getAttr("cAttr").dyn_cast_or_null<some-attr-kind>()
// DEF:      Optional<some-return-type> AOp::cAttr() {
// DEF-NEXT:   auto attr = cAttrAttr()
// DEF-NEXT:   return attr ? Optional<some-return-type>(attr.some-convert-from-storage()) : (llvm::None);

// Test setter methods
// ---

// DEF:      void AOp::aAttrAttr(some-attr-kind attr) {
// DEF-NEXT:   this->getOperation()->setAttr("aAttr", attr);
// DEF:      void AOp::bAttrAttr(some-attr-kind attr) {
// DEF-NEXT:   this->getOperation()->setAttr("bAttr", attr);
// DEF:      void AOp::cAttrAttr(some-attr-kind attr) {
// DEF-NEXT:   this->getOperation()->setAttr("cAttr", attr);

// Test build methods
// ---

// DEF:      void AOp::build(
// DEF:        odsState.addAttribute("aAttr", aAttr);
// DEF:        odsState.addAttribute("bAttr", bAttr);
// DEF:        if (cAttr) {
// DEF-NEXT:     odsState.addAttribute("cAttr", cAttr);

// DEF:      void AOp::build(
// DEF:        some-return-type aAttr, some-return-type bAttr, /*optional*/some-attr-kind cAttr
// DEF:        odsState.addAttribute("aAttr", some-const-builder-call(odsBuilder, aAttr));

// DEF:      void AOp::build(
// DEF:        ArrayRef<NamedAttribute> attributes
// DEF:      odsState.addAttributes(attributes);

def SomeTypeAttr : TypeAttrBase<"SomeType", "some type attribute">;

def BOp : NS_Op<"b_op", []> {
  let arguments = (ins
    AnyAttr:$any_attr,
    BoolAttr:$bool_attr,
    I32Attr:$i32_attr,
    I64Attr:$i64_attr,
    F32Attr:$f32_attr,
    F64Attr:$f64_attr,
    StrAttr:$str_attr,
    ElementsAttr:$elements_attr,
    FlatSymbolRefAttr:$function_attr,
    SomeTypeAttr:$type_attr,
    ArrayAttr:$array_attr,
    TypedArrayAttrBase<SomeAttr, "SomeAttr array">:$some_attr_array,
    TypeAttr:$type_attr
  );
}


// Test common attribute kinds' constraints
// ---

// DEF-LABEL: BOpOperandAdaptor::verify
// DEF: if (!((true)))
// DEF: if (!((tblgen_bool_attr.isa<BoolAttr>())))
// DEF: if (!(((tblgen_i32_attr.isa<IntegerAttr>())) && ((tblgen_i32_attr.cast<IntegerAttr>().getType().isSignlessInteger(32)))))
// DEF: if (!(((tblgen_i64_attr.isa<IntegerAttr>())) && ((tblgen_i64_attr.cast<IntegerAttr>().getType().isSignlessInteger(64)))))
// DEF: if (!(((tblgen_f32_attr.isa<FloatAttr>())) && ((tblgen_f32_attr.cast<FloatAttr>().getType().isF32()))))
// DEF: if (!(((tblgen_f64_attr.isa<FloatAttr>())) && ((tblgen_f64_attr.cast<FloatAttr>().getType().isF64()))))
// DEF: if (!((tblgen_str_attr.isa<StringAttr>())))
// DEF: if (!((tblgen_elements_attr.isa<ElementsAttr>())))
// DEF: if (!((tblgen_function_attr.isa<FlatSymbolRefAttr>())))
// DEF: if (!(((tblgen_type_attr.isa<TypeAttr>())) && ((tblgen_type_attr.cast<TypeAttr>().getValue().isa<SomeType>()))))
// DEF: if (!((tblgen_array_attr.isa<ArrayAttr>())))
// DEF: if (!(((tblgen_some_attr_array.isa<ArrayAttr>())) && (llvm::all_of(tblgen_some_attr_array.cast<ArrayAttr>(), [](Attribute attr) { return (some-condition); }))))
// DEF: if (!(((tblgen_type_attr.isa<TypeAttr>())) && ((tblgen_type_attr.cast<TypeAttr>().getValue().isa<Type>()))))

// Test common attribute kind getters' return types
// ---

// DEF: Attribute BOp::any_attr()
// DEF: bool BOp::bool_attr()
// DEF: APInt BOp::i32_attr()
// DEF: APInt BOp::i64_attr()
// DEF: APFloat BOp::f32_attr()
// DEF: APFloat BOp::f64_attr()
// DEF: StringRef BOp::str_attr()
// DEF: ElementsAttr BOp::elements_attr()
// DEF: StringRef BOp::function_attr()
// DEF: SomeType BOp::type_attr()
// DEF: ArrayAttr BOp::array_attr()
// DEF: ArrayAttr BOp::some_attr_array()
// DEF: Type BOp::type_attr()

// Test building constant values for array attribute kinds
// ---

def COp : NS_Op<"c_op", []> {
  let arguments = (ins
    DefaultValuedAttr<I32ArrayAttr, "{1, 2}">:$i32_array_attr,
    DefaultValuedAttr<I64ArrayAttr, "{3, 4}">:$i64_array_attr,
    DefaultValuedAttr<F32ArrayAttr, "{5.f, 6.f}">:$f32_array_attr,
    DefaultValuedAttr<F64ArrayAttr, "{7., 8.}">:$f64_array_attr,
    DefaultValuedAttr<StrArrayAttr, "{\"a\", \"b\"}">:$str_array_attr
  );
}

// DEF-LABEL: COp definitions
// DEF: mlir::Builder(this->getContext()).getI32ArrayAttr({1, 2})
// DEF: mlir::Builder(this->getContext()).getI64ArrayAttr({3, 4})
// DEF: mlir::Builder(this->getContext()).getF32ArrayAttr({5.f, 6.f})
// DEF: mlir::Builder(this->getContext()).getF64ArrayAttr({7., 8.})
// DEF: mlir::Builder(this->getContext()).getStrArrayAttr({"a", "b"})


// Test builder method which takes unwrapped values for attributes
// ---

def I32Case5:  I32EnumAttrCase<"case5", 5>;
def I32Case10: I32EnumAttrCase<"case10", 10>;

def SomeI32Enum: I32EnumAttr<
  "SomeI32Enum", "", [I32Case5, I32Case10]>;

def DOp : NS_Op<"d_op", []> {
  let arguments = (ins
    I32Attr:$i32_attr,
    F64Attr:$f64_attr,
    StrAttr:$str_attr,
    BoolAttr:$bool_attr,
    SomeI32Enum:$enum_attr,
    DefaultValuedAttr<I32Attr, "42">:$dv_i32_attr,
    DefaultValuedAttr<F64Attr, "8.">:$dv_f64_attr,
    DefaultValuedAttr<StrAttr, "abc">:$dv_str_attr,
    DefaultValuedAttr<BoolAttr, "true">:$dv_bool_attr,
    DefaultValuedAttr<SomeI32Enum, "::SomeI32Enum::case5">:$dv_enum_attr
  );
}

// DECL-LABEL: DOp declarations
// DECL: static void build({{.*}}, APInt i32_attr, APFloat f64_attr, StringRef str_attr, bool bool_attr, ::SomeI32Enum enum_attr, APInt dv_i32_attr, APFloat dv_f64_attr, StringRef dv_str_attr = "abc", bool dv_bool_attr = true, ::SomeI32Enum dv_enum_attr = ::SomeI32Enum::case5)

// DEF-LABEL: DOp definitions
// DEF: odsState.addAttribute("str_attr", odsBuilder.getStringAttr(str_attr));
// DEF: odsState.addAttribute("dv_str_attr", odsBuilder.getStringAttr(dv_str_attr));

// Test derived type attr.
// ---
def DerivedTypeAttrOp : NS_Op<"derived_type_attr_op", []> {
  let results = (outs AnyTensor:$output);
  DerivedTypeAttr element_dtype = DerivedTypeAttr<"return output().getType();">;
}

// DECL: class DerivedTypeAttrOp : public Op
// DECL-SAME: DerivedAttributeOpInterface::Trait
// DECL: static bool isDerivedAttribute
// DEF: bool DerivedTypeAttrOp::isDerivedAttribute(StringRef name) {
// DEF:   if (name == "element_dtype") return true;
// DEF:   return false;
// DEF: }
// DEF: DerivedTypeAttrOp::materializeDerivedAttributes

// Test that only default valued attributes at the end of the arguments
// list get default values in the builder signature
// ---

def EOp : NS_Op<"e_op", []> {
  let arguments = (ins
    I32Attr:$i32_attr,
    DefaultValuedAttr<I32Attr, "42">:$dv_i32_attr,
    F64Attr:$f64_attr,
    DefaultValuedAttr<F64Attr, "8.">:$dv_f64_attr,
    StrAttr:$str_attr,
    DefaultValuedAttr<StrAttr, "abc">:$dv_str_attr,
    BoolAttr:$bool_attr,
    DefaultValuedAttr<BoolAttr, "true">:$dv_bool_attr,
    SomeI32Enum:$enum_attr,
    DefaultValuedAttr<SomeI32Enum, "::SomeI32Enum::case5">:$dv_enum_attr
  );
}

// DECL-LABEL: EOp declarations
// DECL: static void build({{.*}}, APInt i32_attr, APInt dv_i32_attr, APFloat f64_attr, APFloat dv_f64_attr, StringRef str_attr, StringRef dv_str_attr, bool bool_attr, bool dv_bool_attr, ::SomeI32Enum enum_attr, ::SomeI32Enum dv_enum_attr = ::SomeI32Enum::case5)

// Test mixing operands and attributes in arbitrary order
// ---

def MixOperandsAndAttrs : NS_Op<"mix_operands_and_attrs", []> {
  let arguments = (ins F32Attr:$attr, F32:$operand, F32Attr:$otherAttr, F32:$otherArg);
}

// DEF-LABEL: MixOperandsAndAttrs definitions
// DEF-DAG: Value MixOperandsAndAttrs::operand()
// DEF-DAG: Value MixOperandsAndAttrs::otherArg()
// DEF-DAG: void MixOperandsAndAttrs::build(OpBuilder &odsBuilder, OperationState &odsState, FloatAttr attr, Value operand, FloatAttr otherAttr, Value otherArg)
// DEF-DAG: APFloat MixOperandsAndAttrs::attr()
// DEF-DAG: APFloat MixOperandsAndAttrs::otherAttr()

// Test unit attributes.
// ---

def UnitAttrOp : NS_Op<"unit_attr_op", []> {
  let arguments = (ins UnitAttr:$attr);
}

// DEF-LABEL: UnitAttrOp definitions
// DEF: bool UnitAttrOp::attr() {
// DEF:   return {{.*}} != nullptr

// DEF: build(OpBuilder &odsBuilder, OperationState &odsState, /*optional*/UnitAttr attr)


// Test elementAttr field of TypedArrayAttr.
// ---

def SomeTypedArrayAttr : TypedArrayAttrBase<SomeAttr, "SomeAttr array">;

// RECORD-LABEL: def SomeTypedArrayAttr
// RECORD: Attr elementAttr = SomeAttr;
