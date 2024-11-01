// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF
// RUN: mlir-tblgen -print-records -I %S/../../include %s | FileCheck %s --check-prefix=RECORD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "foobar";
}
class NS_Op<string mnemonic, list<Trait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

def SomeAttr : Attr<CPred<"some-condition">, "some attribute kind"> {
  let storageType = "some-attr-kind";
  let returnType = "some-return-type";
  let convertFromStorage = "$_self.some-convert-from-storage()";
  let constBuilderCall = "some-const-builder-call($_builder, $0)";
}

def SomeAttrDef : AttrDef<Test_Dialect, "SomeAttr"> {
}


// Test required, optional, default-valued attributes
// ---

def AOp : NS_Op<"a_op", []> {
  let arguments = (ins
      SomeAttr:$aAttr,
      DefaultValuedAttr<SomeAttr, "4.2">:$bAttr,
      OptionalAttr<SomeAttr>:$cAttr,
      DefaultValuedOptionalAttr<SomeAttr, "4.2">:$dAttr
  );
}

// DECL-LABEL: AOp declarations

// Test attribute name methods
// ---

// DECL:      static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames()
// DECL-NEXT:   static ::llvm::StringRef attrNames[] =
// DECL-SAME:     {::llvm::StringRef("aAttr"), ::llvm::StringRef("bAttr"), ::llvm::StringRef("cAttr"), ::llvm::StringRef("dAttr")};
// DECL-NEXT:   return ::llvm::makeArrayRef(attrNames);

// DECL:      ::mlir::StringAttr getAAttrAttrName()
// DECL-NEXT:      return getAttributeNameForIndex(0);
// DECL:      ::mlir::StringAttr getAAttrAttrName(::mlir::OperationName name)
// DECL-NEXT:      return getAttributeNameForIndex(name, 0);

// DECL:      ::mlir::StringAttr getBAttrAttrName()
// DECL-NEXT:      return getAttributeNameForIndex(1);
// DECL:      ::mlir::StringAttr getBAttrAttrName(::mlir::OperationName name)
// DECL-NEXT:      return getAttributeNameForIndex(name, 1);

// DECL:      ::mlir::StringAttr getCAttrAttrName()
// DECL-NEXT:      return getAttributeNameForIndex(2);
// DECL:      ::mlir::StringAttr getCAttrAttrName(::mlir::OperationName name)
// DECL-NEXT:      return getAttributeNameForIndex(name, 2);

// DEF-LABEL: AOp definitions

// Test verify method
// ---

// DEF:      ::mlir::LogicalResult AOpAdaptor::verify
// DEF:      ::mlir::Attribute tblgen_aAttr;
// DEF-NEXT: while (true) {
// DEF-NEXT:   if (namedAttrIt == namedAttrRange.end())
// DEF-NEXT:     return emitError(loc, "'test.a_op' op ""requires attribute 'aAttr'");
// DEF-NEXT:   if (namedAttrIt->getName() == AOp::getAAttrAttrName(*odsOpName)) {
// DEF-NEXT:     tblgen_aAttr = namedAttrIt->getValue();
// DEF-NEXT:     break;
// DEF:      ::mlir::Attribute tblgen_bAttr;
// DEF-NEXT: ::mlir::Attribute tblgen_cAttr;
// DEF-NEXT: ::mlir::Attribute tblgen_dAttr;
// DEF-NEXT: while (true) {
// DEF-NEXT:   if (namedAttrIt == namedAttrRange.end())
// DEF-NEXT:     break;
// DEF:        if (namedAttrIt->getName() == AOp::getBAttrAttrName(*odsOpName))
// DEF-NEXT:     tblgen_bAttr = namedAttrIt->getValue();
// DEF:        if (namedAttrIt->getName() == AOp::getCAttrAttrName(*odsOpName))
// DEF-NEXT:     tblgen_cAttr = namedAttrIt->getValue();
// DEF:      if (tblgen_aAttr && !((some-condition)))
// DEF-NEXT:   return emitError(loc, "'test.a_op' op ""attribute 'aAttr' failed to satisfy constraint: some attribute kind");
// DEF:      if (tblgen_bAttr && !((some-condition)))
// DEF-NEXT:   return emitError(loc, "'test.a_op' op ""attribute 'bAttr' failed to satisfy constraint: some attribute kind");
// DEF:      if (tblgen_cAttr && !((some-condition)))
// DEF-NEXT:   return emitError(loc, "'test.a_op' op ""attribute 'cAttr' failed to satisfy constraint: some attribute kind");
// DEF:      if (tblgen_dAttr && !((some-condition)))
// DEF-NEXT:   return emitError(loc, "'test.a_op' op ""attribute 'dAttr' failed to satisfy constraint: some attribute kind");

// Test getter methods
// ---

// DEF:      some-attr-kind AOp::getAAttrAttr()
// DEF-NEXT:   ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAAttrAttrName()).cast<some-attr-kind>()
// DEF:      some-return-type AOp::getAAttr() {
// DEF-NEXT:   auto attr = getAAttrAttr()
// DEF-NEXT:   return attr.some-convert-from-storage();

// DEF:      some-attr-kind AOp::getBAttrAttr()
// DEF-NEXT:   ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getBAttrAttrName()).dyn_cast_or_null<some-attr-kind>()
// DEF:      some-return-type AOp::getBAttr() {
// DEF-NEXT:   auto attr = getBAttrAttr();
// DEF-NEXT:   return attr.some-convert-from-storage();

// DEF:      some-attr-kind AOp::getCAttrAttr()
// DEF-NEXT:   ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getCAttrAttrName()).dyn_cast_or_null<some-attr-kind>()
// DEF:      ::llvm::Optional<some-return-type> AOp::getCAttr() {
// DEF-NEXT:   auto attr = getCAttrAttr()
// DEF-NEXT:   return attr ? ::llvm::Optional<some-return-type>(attr.some-convert-from-storage()) : (::std::nullopt);

// DEF:      some-attr-kind AOp::getDAttrAttr()
// DEF-NEXT:   ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getDAttrAttrName()).dyn_cast_or_null<some-attr-kind>()
// DEF:      some-return-type AOp::getDAttr() {
// DEF-NEXT:   auto attr = getDAttrAttr();
// DEF-NEXT:   if (!attr)
// DEF-NEXT:       return some-const-builder-call(::mlir::Builder((*this)->getContext()), 4.2).some-convert-from-storage();
// DEF-NEXT:   return attr.some-convert-from-storage();

// Test setter methods
// ---

// DEF:      void AOp::setAAttrAttr(some-attr-kind attr) {
// DEF-NEXT:   (*this)->setAttr(getAAttrAttrName(), attr);
// DEF:      void AOp::setAAttr(some-return-type attrValue) {
// DEF-NEXT:   (*this)->setAttr(getAAttrAttrName(), some-const-builder-call(::mlir::Builder((*this)->getContext()), attrValue));
// DEF:      void AOp::setBAttrAttr(some-attr-kind attr) {
// DEF-NEXT:   (*this)->setAttr(getBAttrAttrName(), attr);
// DEF:      void AOp::setBAttr(some-return-type attrValue) {
// DEF-NEXT:   (*this)->setAttr(getBAttrAttrName(), some-const-builder-call(::mlir::Builder((*this)->getContext()), attrValue));
// DEF:      void AOp::setCAttrAttr(some-attr-kind attr) {
// DEF-NEXT:   (*this)->setAttr(getCAttrAttrName(), attr);
// DEF:      void AOp::setCAttr(::llvm::Optional<some-return-type> attrValue) {
// DEF-NEXT:   if (attrValue)
// DEF-NEXT:     return (*this)->setAttr(getCAttrAttrName(), some-const-builder-call(::mlir::Builder((*this)->getContext()), *attrValue));
// DEF-NEXT:   (*this)->removeAttr(getCAttrAttrName());

// Test remove methods
// ---

// DEF: ::mlir::Attribute AOp::removeCAttrAttr() {
// DEF-NEXT: return (*this)->removeAttr(getCAttrAttrName());

// Test build methods
// ---

// DEF:      void AOp::build(
// DEF:        odsState.addAttribute(getAAttrAttrName(odsState.name), aAttr);
// DEF:        odsState.addAttribute(getBAttrAttrName(odsState.name), bAttr);
// DEF:        if (cAttr) {
// DEF-NEXT:     odsState.addAttribute(getCAttrAttrName(odsState.name), cAttr);

// DEF:        odsState.addAttribute(getAAttrAttrName(odsState.name), some-const-builder-call(odsBuilder, aAttr));
// DEF-NEXT:   odsState.addAttribute(getBAttrAttrName(odsState.name), some-const-builder-call(odsBuilder, bAttr));
// DEF-NEXT:   if (cAttr) {
// DEF-NEXT:   odsState.addAttribute(getCAttrAttrName(odsState.name), cAttr);
// DEF-NEXT:   }
// DEF-NOT:    if (dAttr)
// DEF:        odsState.addAttribute(getDAttrAttrName(odsState.name), some-const-builder-call(odsBuilder, dAttr));

// DEF:      void AOp::build(
// DEF:        some-return-type aAttr, some-return-type bAttr, /*optional*/some-attr-kind cAttr
// DEF:        odsState.addAttribute(getAAttrAttrName(odsState.name), some-const-builder-call(odsBuilder, aAttr));

// DEF:      void AOp::build(
// DEF:        ::llvm::ArrayRef<::mlir::NamedAttribute> attributes
// DEF:      odsState.addAttributes(attributes);

// Test the above but with prefix.

def Test2_Dialect : Dialect {
  let name = "test2";
  let cppNamespace = "foobar2";
}
def AgetOp : Op<Test2_Dialect, "a_get_op", []> {
  let arguments = (ins
      SomeAttr:$aAttr,
      DefaultValuedOptionalAttr<SomeAttr, "4.2">:$bAttr,
      OptionalAttr<SomeAttr>:$cAttr
  );
}

// DECL-LABEL: AgetOp declarations

// Test attribute name methods
// ---

// DECL:      static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames()
// DECL-NEXT:   static ::llvm::StringRef attrNames[] =
// DECL-SAME:     {::llvm::StringRef("aAttr"), ::llvm::StringRef("bAttr"), ::llvm::StringRef("cAttr")};
// DECL-NEXT:   return ::llvm::makeArrayRef(attrNames);

// DECL:      ::mlir::StringAttr getAAttrAttrName()
// DECL-NEXT:      return getAttributeNameForIndex(0);
// DECL:      ::mlir::StringAttr getAAttrAttrName(::mlir::OperationName name)
// DECL-NEXT:      return getAttributeNameForIndex(name, 0);

// DECL:      ::mlir::StringAttr getBAttrAttrName()
// DECL-NEXT:      return getAttributeNameForIndex(1);
// DECL:      ::mlir::StringAttr getBAttrAttrName(::mlir::OperationName name)
// DECL-NEXT:      return getAttributeNameForIndex(name, 1);

// DECL:      ::mlir::StringAttr getCAttrAttrName()
// DECL-NEXT:      return getAttributeNameForIndex(2);
// DECL:      ::mlir::StringAttr getCAttrAttrName(::mlir::OperationName name)
// DECL-NEXT:      return getAttributeNameForIndex(name, 2);

// DEF-LABEL: AgetOp definitions

// Test verify method
// ---

// DEF:      ::mlir::LogicalResult AgetOpAdaptor::verify
// DEF:      ::mlir::Attribute tblgen_aAttr;
// DEF-NEXT: while (true)
// DEF:      ::mlir::Attribute tblgen_bAttr;
// DEF-NEXT: ::mlir::Attribute tblgen_cAttr;
// DEF-NEXT: while (true)
// DEF:      if (tblgen_aAttr && !((some-condition)))
// DEF-NEXT:   return emitError(loc, "'test2.a_get_op' op ""attribute 'aAttr' failed to satisfy constraint: some attribute kind");
// DEF:      if (tblgen_bAttr && !((some-condition)))
// DEF-NEXT:   return emitError(loc, "'test2.a_get_op' op ""attribute 'bAttr' failed to satisfy constraint: some attribute kind");
// DEF:      if (tblgen_cAttr && !((some-condition)))
// DEF-NEXT:   return emitError(loc, "'test2.a_get_op' op ""attribute 'cAttr' failed to satisfy constraint: some attribute kind");

// Test getter methods
// ---

// DEF:      some-attr-kind AgetOp::getAAttrAttr()
// DEF-NEXT:   ::mlir::impl::getAttrFromSortedRange({{.*}}).cast<some-attr-kind>()
// DEF:      some-return-type AgetOp::getAAttr() {
// DEF-NEXT:   auto attr = getAAttrAttr()
// DEF-NEXT:   return attr.some-convert-from-storage();

// DEF:      some-attr-kind AgetOp::getBAttrAttr()
// DEF-NEXT:   return ::mlir::impl::getAttrFromSortedRange({{.*}}).dyn_cast_or_null<some-attr-kind>()
// DEF:      some-return-type AgetOp::getBAttr() {
// DEF-NEXT:   auto attr = getBAttrAttr();
// DEF-NEXT:   if (!attr)
// DEF-NEXT:       return some-const-builder-call(::mlir::Builder((*this)->getContext()), 4.2).some-convert-from-storage();
// DEF-NEXT:   return attr.some-convert-from-storage();

// DEF:      some-attr-kind AgetOp::getCAttrAttr()
// DEF-NEXT:   return ::mlir::impl::getAttrFromSortedRange({{.*}}).dyn_cast_or_null<some-attr-kind>()
// DEF:      ::llvm::Optional<some-return-type> AgetOp::getCAttr() {
// DEF-NEXT:   auto attr = getCAttrAttr()
// DEF-NEXT:   return attr ? ::llvm::Optional<some-return-type>(attr.some-convert-from-storage()) : (::std::nullopt);

// Test setter methods
// ---

// DEF:      void AgetOp::setAAttrAttr(some-attr-kind attr) {
// DEF-NEXT:   (*this)->setAttr(getAAttrAttrName(), attr);
// DEF:      void AgetOp::setBAttrAttr(some-attr-kind attr) {
// DEF-NEXT:   (*this)->setAttr(getBAttrAttrName(), attr);
// DEF:      void AgetOp::setCAttrAttr(some-attr-kind attr) {
// DEF-NEXT:   (*this)->setAttr(getCAttrAttrName(), attr);

// Test remove methods
// ---

// DEF: ::mlir::Attribute AgetOp::removeCAttrAttr() {
// DEF-NEXT: return (*this)->removeAttr(getCAttrAttrName());

// Test build methods
// ---

// DEF:      void AgetOp::build(
// DEF:        odsState.addAttribute(getAAttrAttrName(odsState.name), aAttr);
// DEF:        odsState.addAttribute(getBAttrAttrName(odsState.name), bAttr);
// DEF:        if (cAttr) {
// DEF-NEXT:     odsState.addAttribute(getCAttrAttrName(odsState.name), cAttr);

// DEF:      void AgetOp::build(
// DEF:        some-return-type aAttr, /*optional*/some-return-type bAttr, /*optional*/some-attr-kind cAttr
// DEF:        odsState.addAttribute(getAAttrAttrName(odsState.name), some-const-builder-call(odsBuilder, aAttr));

// DEF:      void AgetOp::build(
// DEF:        ::llvm::ArrayRef<::mlir::NamedAttribute> attributes
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
    SomeTypeAttr:$some_type_attr,
    ArrayAttr:$array_attr,
    TypedArrayAttrBase<SomeAttr, "SomeAttr array">:$some_attr_array,
    TypeAttr:$type_attr
  );
}


// Test common attribute kinds' constraints
// ---

// DEF-LABEL: BOpAdaptor::verify
// DEF: if (tblgen_any_attr && !((true)))
// DEF: if (tblgen_bool_attr && !((tblgen_bool_attr.isa<::mlir::BoolAttr>())))
// DEF: if (tblgen_i32_attr && !(((tblgen_i32_attr.isa<::mlir::IntegerAttr>())) && ((tblgen_i32_attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
// DEF: if (tblgen_i64_attr && !(((tblgen_i64_attr.isa<::mlir::IntegerAttr>())) && ((tblgen_i64_attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
// DEF: if (tblgen_f32_attr && !(((tblgen_f32_attr.isa<::mlir::FloatAttr>())) && ((tblgen_f32_attr.cast<::mlir::FloatAttr>().getType().isF32()))))
// DEF: if (tblgen_f64_attr && !(((tblgen_f64_attr.isa<::mlir::FloatAttr>())) && ((tblgen_f64_attr.cast<::mlir::FloatAttr>().getType().isF64()))))
// DEF: if (tblgen_str_attr && !((tblgen_str_attr.isa<::mlir::StringAttr>())))
// DEF: if (tblgen_elements_attr && !((tblgen_elements_attr.isa<::mlir::ElementsAttr>())))
// DEF: if (tblgen_function_attr && !((tblgen_function_attr.isa<::mlir::FlatSymbolRefAttr>())))
// DEF: if (tblgen_some_type_attr && !(((tblgen_some_type_attr.isa<::mlir::TypeAttr>())) && ((tblgen_some_type_attr.cast<::mlir::TypeAttr>().getValue().isa<SomeType>()))))
// DEF: if (tblgen_array_attr && !((tblgen_array_attr.isa<::mlir::ArrayAttr>())))
// DEF: if (tblgen_some_attr_array && !(((tblgen_some_attr_array.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_some_attr_array.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((some-condition)); }))))
// DEF: if (tblgen_type_attr && !(((tblgen_type_attr.isa<::mlir::TypeAttr>())) && ((tblgen_type_attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))

// Test common attribute kind getters' return types
// ---

// DEF: ::mlir::Attribute BOp::getAnyAttr()
// DEF: bool BOp::getBoolAttr()
// DEF: uint32_t BOp::getI32Attr()
// DEF: uint64_t BOp::getI64Attr()
// DEF: ::llvm::APFloat BOp::getF32Attr()
// DEF: ::llvm::APFloat BOp::getF64Attr()
// DEF: ::llvm::StringRef BOp::getStrAttr()
// DEF: ::mlir::ElementsAttr BOp::getElementsAttr()
// DEF: ::llvm::StringRef BOp::getFunctionAttr()
// DEF: SomeType BOp::getSomeTypeAttr()
// DEF: ::mlir::ArrayAttr BOp::getArrayAttr()
// DEF: ::mlir::ArrayAttr BOp::getSomeAttrArray()
// DEF: ::mlir::Type BOp::getTypeAttr()

// Test building constant values for array attribute kinds
// ---

def COp : NS_Op<"c_op", []> {
  let arguments = (ins
    DefaultValuedOptionalAttr<I32ArrayAttr, "{1, 2}">:$i32_array_attr,
    DefaultValuedOptionalAttr<I64ArrayAttr, "{3, 4}">:$i64_array_attr,
    DefaultValuedOptionalAttr<F32ArrayAttr, "{5.f, 6.f}">:$f32_array_attr,
    DefaultValuedOptionalAttr<F64ArrayAttr, "{7., 8.}">:$f64_array_attr,
    DefaultValuedOptionalAttr<StrArrayAttr, "{\"a\", \"b\"}">:$str_array_attr
  );
}

// DEF-LABEL: COp definitions
// DEF: ::mlir::Builder((*this)->getContext()).getI32ArrayAttr({1, 2})
// DEF: ::mlir::Builder((*this)->getContext()).getI64ArrayAttr({3, 4})
// DEF: ::mlir::Builder((*this)->getContext()).getF32ArrayAttr({5.f, 6.f})
// DEF: ::mlir::Builder((*this)->getContext()).getF64ArrayAttr({7., 8.})
// DEF: ::mlir::Builder((*this)->getContext()).getStrArrayAttr({"a", "b"})


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
    DefaultValuedStrAttr<StrAttr, "abc">:$dv_str_attr,
    DefaultValuedAttr<BoolAttr, "true">:$dv_bool_attr,
    DefaultValuedAttr<SomeI32Enum, "::SomeI32Enum::case5">:$dv_enum_attr
  );
}

// DECL-LABEL: DOp declarations
// DECL: static void build({{.*}}, uint32_t i32_attr, ::llvm::APFloat f64_attr, ::llvm::StringRef str_attr, bool bool_attr, ::SomeI32Enum enum_attr, uint32_t dv_i32_attr, ::llvm::APFloat dv_f64_attr, ::llvm::StringRef dv_str_attr = "abc", bool dv_bool_attr = true, ::SomeI32Enum dv_enum_attr = ::SomeI32Enum::case5)

// DEF-LABEL: DOp definitions
// DEF: odsState.addAttribute(getStrAttrAttrName(odsState.name), odsBuilder.getStringAttr(str_attr));
// DEF: odsState.addAttribute(getDvStrAttrAttrName(odsState.name), odsBuilder.getStringAttr(dv_str_attr));

// Test derived type attr.
// ---
def DerivedTypeAttrOp : NS_Op<"derived_type_attr_op", []> {
  let results = (outs AnyTensor:$output);
  DerivedTypeAttr element_dtype = DerivedTypeAttr<"return output().getType();">;
}

// DECL: class DerivedTypeAttrOp : public ::mlir::Op
// DECL-SAME: DerivedAttributeOpInterface::Trait
// DECL: static bool isDerivedAttribute
// DEF: bool DerivedTypeAttrOp::isDerivedAttribute(::llvm::StringRef name) {
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
    DefaultValuedStrAttr<StrAttr, "abc">:$dv_str_attr,
    BoolAttr:$bool_attr,
    DefaultValuedAttr<BoolAttr, "true">:$dv_bool_attr,
    SomeI32Enum:$enum_attr,
    DefaultValuedAttr<SomeI32Enum, "::SomeI32Enum::case5">:$dv_enum_attr
  );
}

// DECL-LABEL: EOp declarations
// DECL: static void build({{.*}}, uint32_t i32_attr, uint32_t dv_i32_attr, ::llvm::APFloat f64_attr, ::llvm::APFloat dv_f64_attr, ::llvm::StringRef str_attr, ::llvm::StringRef dv_str_attr, bool bool_attr, bool dv_bool_attr, ::SomeI32Enum enum_attr, ::SomeI32Enum dv_enum_attr = ::SomeI32Enum::case5)


// Test proper namespacing for AttrDef
// ---

def NamespaceOp : NS_Op<"namespace_op", []> {
  let arguments = (ins
      SomeAttrDef:$AttrDef
  );
}
// DECL: NamespaceOp
// DECL: foobar::SomeAttrAttr getAttrDef()


// Test mixing operands and attributes in arbitrary order
// ---

def MixOperandsAndAttrs : NS_Op<"mix_operands_and_attrs", []> {
  let arguments = (ins F32Attr:$attr, F32:$operand, F32Attr:$otherAttr, F32:$otherArg);
}

def OpWithDefaultAndRegion : NS_Op<"default_with_region", []> {
  let arguments = (ins
          DefaultValuedAttr<BoolAttr, "true">:$dv_bool_attr
  );
  let regions = (region VariadicRegion<AnyRegion>:$region);
}

// We should not have a default attribute in this case.

// DECL-LABEL: OpWithDefaultAndRegion declarations
// DECL: static void build({{.*}}, bool dv_bool_attr, unsigned regionCount)

def OpWithDefaultAndSuccessor : NS_Op<"default_with_succ", []> {
  let arguments = (ins
          DefaultValuedAttr<BoolAttr, "true">:$dv_bool_attr
  );
  let successors = (successor VariadicSuccessor<AnySuccessor>:$succ);
}

// We should not have a default attribute in this case.

// DECL-LABEL: OpWithDefaultAndSuccessor declarations
// DECL: static void build({{.*}}, bool dv_bool_attr, ::mlir::BlockRange succ)

// DEF-LABEL: MixOperandsAndAttrs definitions
// DEF-DAG: ::mlir::TypedValue<::mlir::FloatType> MixOperandsAndAttrs::getOperand()
// DEF-DAG: ::mlir::TypedValue<::mlir::FloatType> MixOperandsAndAttrs::getOtherArg()
// DEF-DAG: void MixOperandsAndAttrs::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::FloatAttr attr, ::mlir::Value operand, ::mlir::FloatAttr otherAttr, ::mlir::Value otherArg)
// DEF-DAG: ::llvm::APFloat MixOperandsAndAttrs::getAttr()
// DEF-DAG: ::llvm::APFloat MixOperandsAndAttrs::getOtherAttr()

// Test unit attributes.
// ---

def UnitAttrOp : NS_Op<"unit_attr_op", []> {
  let arguments = (ins UnitAttr:$attr);
}

// DEF-LABEL: UnitAttrOp definitions
// DEF: bool UnitAttrOp::getAttr() {
// DEF:   return {{.*}} != nullptr

// DEF: ::mlir::Attribute UnitAttrOp::removeAttrAttr() {
// DEF-NEXT:   (*this)->removeAttr(getAttrAttrName());

// DEF: build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::UnitAttr attr)
// DEF: build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool attr)

// DECL-LABEL: UnitAttrOp declarations
// DECL-NOT: declarations
// DECL: build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool attr = false)


// Test elementAttr field of TypedArrayAttr.
// ---

def SomeTypedArrayAttr : TypedArrayAttrBase<SomeAttr, "SomeAttr array">;

// RECORD-LABEL: def SomeTypedArrayAttr
// RECORD: Attr elementAttr = SomeAttr;
