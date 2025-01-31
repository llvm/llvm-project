//===-- CommonAttrConstraints.td - Common Attr Constraints--*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains commonly used attr constraints.
//
//===----------------------------------------------------------------------===//

#ifndef COMMON_ATTR_CONSTRAINTS_TD
#define COMMON_ATTR_CONSTRAINTS_TD

include "mlir/IR/Constraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// Attribute definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base attribute definition

// Base class for all attributes.
class Attr<Pred condition, string summary = ""> :
    AttrConstraint<condition, summary> {
  code storageType = ?; // The backing mlir::Attribute type
  code returnType = ?;  // The underlying C++ value type

  // The call expression to convert from the storage type to the return
  // type. For example, an enum can be stored as an int but returned as an
  // enum class.
  //
  // Format: $_self will be expanded to the attribute.
  //
  // For example, `$_self.getValue().getSExtValue()` for `IntegerAttr val` will
  // expand to `getAttrOfType<IntegerAttr>("val").getValue().getSExtValue()`.
  code convertFromStorage = "$_self.getValue()";

  // The call expression to build an attribute from a constant value.
  //
  // Format: $0 will be expanded to the constant value of the attribute.
  //
  // For example, `$_builder.getStringAttr("$0")` for `StringAttr:"foo"` will
  // expand to `builder.getStringAttr("foo")`.
  string constBuilderCall = ?;

  // Default value for attribute.
  // Requires a constBuilderCall defined.
  string defaultValue = ?;

  // The value type of this attribute. This corresponds to the mlir::Type that
  // this attribute returns via `getType()`.
  Type valueType = ?;

  // Whether the attribute is optional. Typically requires a custom
  // convertFromStorage method to handle the case where the attribute is
  // not present.
  bit isOptional = 0;

  // What is the base-level Attr instantiation that this Attr is built upon.
  // Unset means this is a base-level Attr.
  //
  // This field is used by attribute wrapper classes (DefaultValuedAttr,
  // OptionalAttr, etc.) to retrieve the base-level attribute definition.
  // This can be used for getting its name; otherwise, we will see
  // "anonymous_<number>" as the attribute def name because of template
  // instantiation.
  // TOOD(b/132458159): deduplicate the fields in attribute wrapper classes.
  Attr baseAttr = ?;

  // The fully-qualified C++ namespace where the generated class lives.
  string cppNamespace = "";

  // The full description of this attribute.
  string description = "";
}

// An attribute of a specific dialect.
class DialectAttr<Dialect d, Pred condition, string summary = ""> :
    Attr<condition, summary> {
  Dialect dialect = d;
  let cppNamespace = d.cppNamespace;
}

//===----------------------------------------------------------------------===//
// Attribute modifier definition

// Decorates an attribute to have an (unvalidated) default value if not present.
class DefaultValuedAttr<Attr attr, string val> :
    Attr<attr.predicate, attr.summary> {
  // Construct this attribute with the input attribute and change only
  // the default value.
  // Note: this has to be kept up to date with Attr above.
  let storageType = attr.storageType;
  let returnType = attr.returnType;
  let convertFromStorage = attr.convertFromStorage;
  let constBuilderCall = attr.constBuilderCall;
  let defaultValue = val;
  let valueType = attr.valueType;

  let baseAttr = attr;
}

// Decorates an optional attribute to have an (unvalidated) default value
// return by ODS generated accessors if not present.
class DefaultValuedOptionalAttr<Attr attr, string val> :
    Attr<attr.predicate, attr.summary> {
  // Construct this attribute with the input attribute and change only
  // the default value.
  // Note: this has to be kept up to date with Attr above.
  let storageType = attr.storageType;
  let returnType = attr.returnType;
  let convertFromStorage = attr.convertFromStorage;
  let constBuilderCall = attr.constBuilderCall;
  let defaultValue = val;
  let valueType = attr.valueType;
  let isOptional = 1;

  let baseAttr = attr;
}

// Decorates an attribute as optional. The return type of the generated
// attribute accessor method will be Optional<>.
class OptionalAttr<Attr attr> : Attr<attr.predicate, attr.summary> {
  // Rewrite the attribute to be optional.
  // Note: this has to be kept up to date with Attr above.
  let storageType = attr.storageType;
  let returnType = "::std::optional<" # attr.returnType #">";
  let convertFromStorage = "$_self ? " # returnType # "(" #
                           attr.convertFromStorage # ") : (::std::nullopt)";
  let valueType = attr.valueType;
  let isOptional = 1;

  let baseAttr = attr;
}

// Default-valued string-based attribute. Wraps the default value in escaped
// quotes.
class DefaultValuedStrAttr<Attr attr, string val>
    : DefaultValuedAttr<attr, "\"" # val # "\"">;
class DefaultValuedOptionalStrAttr<Attr attr, string val>
    : DefaultValuedOptionalAttr<attr, "\"" # val # "\"">;

//===----------------------------------------------------------------------===//
// Primitive attribute kinds

// A generic attribute that must be constructed around a specific buildable type
// `attrValType`. Backed by MLIR attribute kind `attrKind`.
class TypedAttrBase<Type attrValType, string attrKind, Pred condition,
                    string descr> :
    Attr<condition, descr> {
  let constBuilderCall = "$_builder.get" # attrKind # "(" #
                         attrValType.builderCall # ", $0)";
  let storageType = "::mlir::" # attrKind;
  let valueType = attrValType;
}

// Any attribute.
def AnyAttr : Attr<CPred<"true">, "any attribute"> {
  let storageType = "::mlir::Attribute";
  let returnType = "::mlir::Attribute";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

// Any attribute from the given list
class AnyAttrOf<list<Attr> allowedAttrs, string summary = "",
                string cppType = "::mlir::Attribute",
                string fromStorage = "$_self"> : Attr<
    // Satisfy any of the allowed attribute's condition
    Or<!foreach(allowedattr, allowedAttrs, allowedattr.predicate)>,
    !if(!eq(summary, ""),
        !interleave(!foreach(t, allowedAttrs, t.summary), " or "),
        summary)> {
    let returnType = cppType;
    let convertFromStorage = fromStorage;
    list<Attr> allowedAttributes = allowedAttrs;
}

def LocationAttr : Attr<CPred<"::llvm::isa<::mlir::LocationAttr>($_self)">,
                        "location attribute">;

def BoolAttr : Attr<CPred<"::llvm::isa<::mlir::BoolAttr>($_self)">, "bool attribute"> {
  let storageType = [{ ::mlir::BoolAttr }];
  let returnType = [{ bool }];
  let valueType = I1;
  let constBuilderCall = "$_builder.getBoolAttr($0)";
}

// Index attribute.
def IndexAttr :
    TypedAttrBase<
      Index, "IntegerAttr",
      And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
           CPred<"::llvm::isa<::mlir::IndexType>(::llvm::cast<::mlir::IntegerAttr>($_self).getType())">]>,
      "index attribute"> {
  let returnType = [{ ::llvm::APInt }];
}

// Base class for any integer (regardless of signedness semantics) attributes
// of fixed width.
class AnyIntegerAttrBase<AnyI attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
           CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getType()."
                 "isInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ ::llvm::APInt }];
  let constBuilderCall = ?;
}

def AnyI1Attr  : AnyIntegerAttrBase<AnyI1,  "1-bit integer attribute">;
def AnyI8Attr  : AnyIntegerAttrBase<AnyI8,  "8-bit integer attribute">;
def AnyI16Attr : AnyIntegerAttrBase<AnyI16, "16-bit integer attribute">;
def AnyI32Attr : AnyIntegerAttrBase<AnyI32, "32-bit integer attribute">;
def AnyI64Attr : AnyIntegerAttrBase<AnyI64, "64-bit integer attribute">;

def APIntAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
                     "arbitrary integer attribute"> {
  let storageType = [{ ::mlir::IntegerAttr }];
  let returnType = [{ ::mlir::APInt }];
}

// Base class for signless integer attributes of fixed width.
class SignlessIntegerAttrBase<I attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
           CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getType()."
                 "isSignlessInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ ::llvm::APInt }];
}
// Base class for signless integer attributes of fixed width that have a
// corresponding C++ type.
class TypedSignlessIntegerAttrBase<I attrValType, string retType, string descr>
    : SignlessIntegerAttrBase<attrValType, descr> {
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().getZExtValue()";
}

def I1Attr  : TypedSignlessIntegerAttrBase<
    I1,  "bool",     "1-bit signless integer attribute">;
def I8Attr  : TypedSignlessIntegerAttrBase<
    I8,  "uint8_t",  "8-bit signless integer attribute">;
def I16Attr : TypedSignlessIntegerAttrBase<
    I16, "uint16_t", "16-bit signless integer attribute">;
def I32Attr : TypedSignlessIntegerAttrBase<
    I32, "uint32_t", "32-bit signless integer attribute">;
def I64Attr : TypedSignlessIntegerAttrBase<
    I64, "uint64_t", "64-bit signless integer attribute">;

// Base class for signed integer attributes of fixed width.
class SignedIntegerAttrBase<SI attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
           CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getType()."
                 "isSignedInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ ::llvm::APInt }];
}
// Base class for signed integer attributes of fixed width that have a
// corresponding C++ type.
class TypedSignedIntegerAttrBase<SI attrValType, string retType, string descr>
    : SignedIntegerAttrBase<attrValType, descr> {
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().getSExtValue()";
}

def SI1Attr  : TypedSignedIntegerAttrBase<
    SI1,  "bool",    "1-bit signed integer attribute">;
def SI8Attr  : TypedSignedIntegerAttrBase<
    SI8,  "int8_t",  "8-bit signed integer attribute">;
def SI16Attr : TypedSignedIntegerAttrBase<
    SI16, "int16_t", "16-bit signed integer attribute">;
def SI32Attr : TypedSignedIntegerAttrBase<
    SI32, "int32_t", "32-bit signed integer attribute">;
def SI64Attr : TypedSignedIntegerAttrBase<
    SI64, "int64_t", "64-bit signed integer attribute">;

// Base class for unsigned integer attributes of fixed width.
class UnsignedIntegerAttrBase<UI attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
           CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getType()."
                 "isUnsignedInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ ::llvm::APInt }];
}
// Base class for unsigned integer attributes of fixed width that have a
// corresponding C++ type.
class TypedUnsignedIntegerAttrBase<UI attrValType, string retType, string descr>
    : UnsignedIntegerAttrBase<attrValType, descr> {
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().getZExtValue()";
}

def UI1Attr  : TypedUnsignedIntegerAttrBase<
    UI1,  "bool",     "1-bit unsigned integer attribute">;
def UI8Attr  : TypedUnsignedIntegerAttrBase<
    UI8,  "uint8_t",  "8-bit unsigned integer attribute">;
def UI16Attr : TypedUnsignedIntegerAttrBase<
    UI16, "uint16_t", "16-bit unsigned integer attribute">;
def UI32Attr : TypedUnsignedIntegerAttrBase<
    UI32, "uint32_t", "32-bit unsigned integer attribute">;
def UI64Attr : TypedUnsignedIntegerAttrBase<
    UI64, "uint64_t", "64-bit unsigned integer attribute">;

// Base class for float attributes of fixed width.
class FloatAttrBase<F attrValType, string descr> :
    TypedAttrBase<attrValType, "FloatAttr",
              And<[CPred<"::llvm::isa<::mlir::FloatAttr>($_self)">,
                     CPred<"::llvm::cast<::mlir::FloatAttr>($_self).getType().isF" #
                           attrValType.bitwidth # "()">]>,
              descr> {
  let returnType = [{ ::llvm::APFloat }];
}

def F32Attr : FloatAttrBase<F32, "32-bit float attribute">;
def F64Attr : FloatAttrBase<F64, "64-bit float attribute">;

// An attribute backed by a string type.
class StringBasedAttr<Pred condition, string descr> : Attr<condition, descr> {
  let constBuilderCall = "$_builder.getStringAttr($0)";
  let storageType = [{ ::mlir::StringAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
}

def StrAttrPred : CPred<"::llvm::isa<::mlir::StringAttr>($_self)">;

def StrAttr : StringBasedAttr<StrAttrPred, "string attribute">;

// A string attribute that represents the name of a symbol.
def SymbolNameAttr : StringBasedAttr<StrAttrPred, "string attribute">;

// String attribute that has a specific value type.
class TypedStrAttr<Type ty>
    : StringBasedAttr<And<[StrAttrPred,
        SubstLeaves<"$_self", "::mlir::cast<StringAttr>($_self).getType()",
                    ty.predicate>]>,
        "string attribute of " # ty.summary> {
  let valueType = ty;
}

// Base class for attributes containing types. Example:
//   def IntTypeAttr : TypeAttrBase<"IntegerType", "integer type attribute">
// defines a type attribute containing an integer type.
class TypeAttrBase<string retType, string summary,
                        Pred typePred = CPred<"true">> :
    Attr<And<[
      CPred<"::llvm::isa<::mlir::TypeAttr>($_self)">,
      CPred<"::llvm::isa<" # retType # ">(::llvm::cast<::mlir::TypeAttr>($_self).getValue())">,
      SubstLeaves<"$_self",
                    "::llvm::cast<::mlir::TypeAttr>($_self).getValue()", typePred>]>,
    summary> {
  let storageType = [{ ::mlir::TypeAttr }];
  let returnType = retType;
  let valueType = NoneType;
  let convertFromStorage = "::llvm::cast<" # retType # ">($_self.getValue())";
}

def TypeAttr : TypeAttrBase<"::mlir::Type", "any type attribute"> {
  let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

class TypeAttrOf<Type ty>
   : TypeAttrBase<ty.cppType, "type attribute of " # ty.summary,
                    ty.predicate> {
  let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

// The mere presence of unit attributes has a meaning.  Therefore, unit
// attributes are always treated as optional and accessors to them return
// "true" if the attribute is present and "false" otherwise.
def UnitAttr : Attr<CPred<"::llvm::isa<::mlir::UnitAttr>($_self)">, "unit attribute"> {
  let storageType = [{ ::mlir::UnitAttr }];
  let constBuilderCall = "(($0) ? $_builder.getUnitAttr() : nullptr)";
  let convertFromStorage = "$_self != nullptr";
  let returnType = "bool";
  let defaultValue = "false";
  let valueType = NoneType;
  let isOptional = 1;
}

//===----------------------------------------------------------------------===//
// Composite attribute kinds

class DictionaryAttrBase<Pred condition, string summary> :
    Attr<condition, summary> {
  let storageType = [{ ::mlir::DictionaryAttr }];
  let constBuilderCall = "$_builder.getDictionaryAttr($0)";
  let returnType = [{ ::mlir::DictionaryAttr }];
  let valueType = NoneType;
  let convertFromStorage = "$_self";
}

def DictionaryAttr
    : DictionaryAttrBase<CPred<"::llvm::isa<::mlir::DictionaryAttr>($_self)">,
                               "dictionary of named attribute values">;

class ElementsAttrBase<Pred condition, string summary> :
    Attr<condition, summary> {
  let storageType = [{ ::mlir::ElementsAttr }];
  let returnType = [{ ::mlir::ElementsAttr }];
  let convertFromStorage = "$_self";
}

def ElementsAttr : ElementsAttrBase<CPred<"::llvm::isa<::mlir::ElementsAttr>($_self)">,
                                    "constant vector/tensor attribute">;

class IntElementsAttrBase<Pred condition, string summary> :
    ElementsAttrBase<And<[CPred<"::llvm::isa<::mlir::DenseIntElementsAttr>($_self)">,
                          condition]>,
                     summary> {
  let storageType = [{ ::mlir::DenseIntElementsAttr }];
  let returnType = [{ ::mlir::DenseIntElementsAttr }];

  let convertFromStorage = "$_self";
}

class DenseArrayAttrBase<string denseAttrName, string cppType, string summaryName> :
    ElementsAttrBase<CPred<"::llvm::isa<::mlir::" # denseAttrName # ">($_self)">,
                     summaryName # " dense array attribute"> {
  let storageType = "::mlir::" # denseAttrName;
  let returnType = "::llvm::ArrayRef<" # cppType # ">";
  let constBuilderCall = "$_builder.get" # denseAttrName # "($0)";
}
def DenseBoolArrayAttr : DenseArrayAttrBase<"DenseBoolArrayAttr", "bool", "i1">;
def DenseI8ArrayAttr : DenseArrayAttrBase<"DenseI8ArrayAttr", "int8_t", "i8">;
def DenseI16ArrayAttr : DenseArrayAttrBase<"DenseI16ArrayAttr", "int16_t", "i16">;
def DenseI32ArrayAttr : DenseArrayAttrBase<"DenseI32ArrayAttr", "int32_t", "i32">;
def DenseI64ArrayAttr : DenseArrayAttrBase<"DenseI64ArrayAttr", "int64_t", "i64">;
def DenseF32ArrayAttr : DenseArrayAttrBase<"DenseF32ArrayAttr", "float", "f32">;
def DenseF64ArrayAttr : DenseArrayAttrBase<"DenseF64ArrayAttr", "double", "f64">;

def IndexElementsAttr
    : IntElementsAttrBase<CPred<[{::llvm::cast<::mlir::DenseIntElementsAttr>($_self)
                                      .getType()
                                      .getElementType()
                                      .isIndex()}]>,
                          "index elements attribute">;

def AnyIntElementsAttr : IntElementsAttrBase<CPred<"true">, "integer elements attribute">;

class IntElementsAttrOf<int width> : IntElementsAttrBase<
  CPred<"::llvm::cast<::mlir::DenseIntElementsAttr>($_self).getType()."
        "getElementType().isInteger(" # width # ")">,
  width # "-bit integer elements attribute">;

def AnyI32ElementsAttr : IntElementsAttrOf<32>;
def AnyI64ElementsAttr : IntElementsAttrOf<64>;

class SignlessIntElementsAttr<int width> : IntElementsAttrBase<
  CPred<"::llvm::cast<::mlir::DenseIntElementsAttr>($_self).getType()."
        "getElementType().isSignlessInteger(" # width # ")">,
  width # "-bit signless integer elements attribute"> {

  // Note that this is only constructing scalar elements attribute.
  let constBuilderCall = "::llvm::cast<::mlir::DenseIntElementsAttr>("
  "::mlir::DenseElementsAttr::get("
    "::mlir::RankedTensorType::get({}, $_builder.getIntegerType(" # width # ")), "
    "::llvm::ArrayRef($0)))";
}

def I32ElementsAttr : SignlessIntElementsAttr<32>;
def I64ElementsAttr : SignlessIntElementsAttr<64>;

// A `width`-bit signless integer elements attribute. The attribute should be
// ranked and has a shape as specified in `dims`.
class RankedSignlessIntElementsAttr<int width, list<int> dims> :
    SignlessIntElementsAttr<width> {
  // Check that this has the specified shape.
  let predicate = And<[
    SignlessIntElementsAttr<width>.predicate,
    CPred<"::llvm::cast<::mlir::DenseIntElementsAttr>($_self).getType().getShape() == "
        "::mlir::ArrayRef<int64_t>({" # !interleave(dims, ", ") # "})">]>;

  let summary = width # "-bit signless int elements attribute of shape [" #
                !interleave(dims, ", ") # "]";

  let constBuilderCall = "::mlir::DenseIntElementsAttr::get("
    "::mlir::RankedTensorType::get({" # !interleave(dims, ", ") #
    "}, $_builder.getIntegerType(" # width # ")), ::llvm::ArrayRef($0))";
}

class RankedI32ElementsAttr<list<int> dims> :
    RankedSignlessIntElementsAttr<32, dims>;
class RankedI64ElementsAttr<list<int> dims> :
    RankedSignlessIntElementsAttr<64, dims>;

class FloatElementsAttr<int width> : ElementsAttrBase<
  CPred<"::llvm::isa<::mlir::DenseFPElementsAttr>($_self) &&"
      "::llvm::cast<::mlir::DenseElementsAttr>($_self).getType()."
      "getElementType().isF" # width # "()">,
  width # "-bit float elements attribute"> {

  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  // Note that this is only constructing scalar elements attribute.
  let constBuilderCall = "::mlir::DenseElementsAttr::get("
    "::mlir::RankedTensorType::get({}, $_builder.getF" # width # "Type()),"
    "::llvm::ArrayRef($0))";
  let convertFromStorage = "$_self";
}

def F64ElementsAttr : FloatElementsAttr<64>;

// A `width`-bit floating point elements attribute. The attribute should be
// ranked and has a shape as specified in `dims`.
class RankedFloatElementsAttr<int width, list<int> dims> : ElementsAttrBase<
  CPred<"::llvm::isa<::mlir::DenseFPElementsAttr>($_self) &&"
      "::llvm::cast<::mlir::DenseFPElementsAttr>($_self).getType()."
      "getElementType().isF" # width # "() && "
      // Check that this is ranked and has the specified shape.
      "::llvm::cast<::mlir::DenseFPElementsAttr>($_self).getType().hasRank() && "
      "::llvm::cast<::mlir::DenseFPElementsAttr>($_self).getType().getShape() == "
      "::mlir::ArrayRef<int64_t>({" # !interleave(dims, ", ") # "})">,
  width # "-bit float elements attribute of shape [" #
  !interleave(dims, ", ") # "]"> {

  let storageType = [{ ::mlir::DenseFPElementsAttr }];
  let returnType = [{ ::mlir::DenseFPElementsAttr }];

  let constBuilderCall = "::llvm::cast<::mlir::DenseFPElementsAttr>("
  "::mlir::DenseElementsAttr::get("
    "::mlir::RankedTensorType::get({" # !interleave(dims, ", ") #
    "}, $_builder.getF" # width # "Type()), "
    "::llvm::ArrayRef($0)))";
  let convertFromStorage = "$_self";
}

class RankedF32ElementsAttr<list<int> dims> : RankedFloatElementsAttr<32, dims>;
class RankedF64ElementsAttr<list<int> dims> : RankedFloatElementsAttr<64, dims>;

def StringElementsAttr : ElementsAttrBase<
  CPred<"::llvm::isa<::mlir::DenseStringElementsAttr>($_self)" >,
  "string elements attribute"> {

  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  let convertFromStorage = "$_self";
}

// Attributes containing affine maps.
def AffineMapAttr : Attr<
CPred<"::llvm::isa<::mlir::AffineMapAttr>($_self)">, "AffineMap attribute"> {
  let storageType = [{::mlir::AffineMapAttr }];
  let returnType = [{ ::mlir::AffineMap }];
  let valueType = Index;
  let constBuilderCall = "::mlir::AffineMapAttr::get($0)";
}

// Attributes containing integer sets.
def IntegerSetAttr : Attr<
CPred<"::llvm::isa<::mlir::IntegerSetAttr>($_self)">, "IntegerSet attribute"> {
  let storageType = [{::mlir::IntegerSetAttr }];
  let returnType = [{ ::mlir::IntegerSet }];
  let valueType = NoneType;
  let constBuilderCall = "::mlir::IntegerSetAttr::get($0)";
}

// Base class for array attributes.
class ArrayAttrBase<Pred condition, string summary> : Attr<condition, summary> {
  let storageType = [{ ::mlir::ArrayAttr }];
  let returnType = [{ ::mlir::ArrayAttr }];
  let valueType = NoneType;
  let convertFromStorage = "$_self";
  let constBuilderCall = "$_builder.getArrayAttr($0)";
}

def ArrayAttr : ArrayAttrBase<CPred<"::llvm::isa<::mlir::ArrayAttr>($_self)">,
                              "array attribute">;

// Base class for array attributes whose elements are of the same kind.
// `element` specifies the element attribute kind stored in this array.
class TypedArrayAttrBase<Attr element, string summary>: ArrayAttrBase<
    And<[
      // Guarantee this is an ArrayAttr first
      CPred<"::llvm::isa<::mlir::ArrayAttr>($_self)">,
      // Guarantee all elements satisfy the constraints from `element`
      Concat<"::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self), "
                            "[&](::mlir::Attribute attr) { return attr && (",
                               SubstLeaves<"$_self", "attr", element.predicate>,
                            "); })">]>,
    summary> {

  Attr elementAttr = element;
}

def LocationArrayAttr : TypedArrayAttrBase<LocationAttr,
                                           "location array attribute">;

def AffineMapArrayAttr : TypedArrayAttrBase<AffineMapAttr,
                                      "AffineMap array attribute"> {
  let constBuilderCall = "$_builder.getAffineMapArrayAttr($0)";
}

def BoolArrayAttr : TypedArrayAttrBase<BoolAttr,
                                      "1-bit boolean array attribute"> {
  let constBuilderCall = "$_builder.getBoolArrayAttr($0)";
}
def I32ArrayAttr : TypedArrayAttrBase<I32Attr,
                                      "32-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getI32ArrayAttr($0)";
}
def I64ArrayAttr : TypedArrayAttrBase<I64Attr,
                                      "64-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getI64ArrayAttr($0)";
}
// Variant of I64ArrayAttr whose user accessor is SmallVector<in64_t>.
def I64SmallVectorArrayAttr :
    TypedArrayAttrBase<I64Attr, "64-bit integer array attribute"> {
  let returnType = [{ ::llvm::SmallVector<int64_t, 8> }];
  let convertFromStorage = [{
    llvm::to_vector<4>(
      llvm::map_range($_self.getAsRange<mlir::IntegerAttr>(),
      [](mlir::IntegerAttr attr) { return attr.getInt(); }));
  }];
  let constBuilderCall = "$_builder.getI64ArrayAttr($0)";
}
def F32ArrayAttr : TypedArrayAttrBase<F32Attr, "32-bit float array attribute"> {
  let constBuilderCall = "$_builder.getF32ArrayAttr($0)";
}
def F64ArrayAttr : TypedArrayAttrBase<F64Attr, "64-bit float array attribute"> {
  let constBuilderCall = "$_builder.getF64ArrayAttr($0)";
}
def StrArrayAttr : TypedArrayAttrBase<StrAttr, "string array attribute"> {
  let constBuilderCall = "$_builder.getStrArrayAttr($0)";
}
def TypeArrayAttr : TypedArrayAttrBase<TypeAttr, "type array attribute"> {
  let constBuilderCall = "$_builder.getTypeArrayAttr($0)";
}
def IndexListArrayAttr :
  TypedArrayAttrBase<I64ArrayAttr, "Array of 64-bit integer array attributes">;
def DictArrayAttr :
  TypedArrayAttrBase<DictionaryAttr, "Array of dictionary attributes">;

// Attributes containing symbol references.
def SymbolRefAttr : Attr<CPred<"::llvm::isa<::mlir::SymbolRefAttr>($_self)">,
                        "symbol reference attribute"> {
  let storageType = [{ ::mlir::SymbolRefAttr }];
  let returnType = [{ ::mlir::SymbolRefAttr }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self";
}

def FlatSymbolRefAttr : Attr<CPred<"::llvm::isa<::mlir::FlatSymbolRefAttr>($_self)">,
                                   "flat symbol reference attribute"> {
  let storageType = [{ ::mlir::FlatSymbolRefAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self.getValue()";
}

def SymbolRefArrayAttr :
  TypedArrayAttrBase<SymbolRefAttr, "symbol ref array attribute"> {
  let constBuilderCall = ?;
}

def FlatSymbolRefArrayAttr :
  TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = ?;
}

//===----------------------------------------------------------------------===//
// Derive attribute kinds

// DerivedAttr are attributes whose value is computed from properties
// of the operation. They do not require additional storage and are
// materialized as needed.
// Note: All derived attributes should be materializable as an Attribute. E.g.,
// do not use DerivedAttr for things that could not have been stored as
// Attribute.
//
class DerivedAttr<code ret, code b, code convert = ""> :
    Attr<CPred<"true">, "derived attribute"> {
  let returnType = ret;
  code body = b;

  // Specify how to convert from the derived attribute to an attribute.
  //
  // ## Special placeholders
  //
  // Special placeholders can be used to refer to entities during conversion:
  //
  // * `$_builder` will be replaced by a mlir::Builder instance.
  // * `$_ctxt` will be replaced by the MLIRContext* instance.
  // * `$_self` will be replaced with the derived attribute (value produces
  //    `returnType`).
  let convertFromStorage = convert;
}

// Derived attribute that returns a mlir::Type.
class DerivedTypeAttr<code body> : DerivedAttr<"::mlir::Type", body> {
  let convertFromStorage = "::mlir::TypeAttr::get($_self)";
}

//===----------------------------------------------------------------------===//
// Constant attribute kinds

// Represents a constant attribute of specific Attr type. A constant
// attribute can be specified only of attributes that have a constant
// builder call defined. The constant value is specified as a string.
//
// If used as a constraint, it generates a matcher on a constant attribute by
// using the constant value builder of the attribute and the value.
class ConstantAttr<Attr attribute, string val> : AttrConstraint<
    CPred<"$_self == " # !subst("$0", val, attribute.constBuilderCall)>,
    "constant attribute " # val> {
  Attr attr = attribute;
  string value = val;
}

class ConstF32Attr<string val> : ConstantAttr<F32Attr, val>;
def ConstBoolAttrFalse : ConstantAttr<BoolAttr, "false">;
def ConstBoolAttrTrue : ConstantAttr<BoolAttr, "true">;
def ConstUnitAttr : ConstantAttr<UnitAttr, "true">;

// Constant string-based attribute. Wraps the desired string in escaped quotes.
class ConstantStrAttr<Attr attribute, string val>
    : ConstantAttr<attribute, "\"" # val # "\"">;

//===----------------------------------------------------------------------===//
// Common attribute constraints
//===----------------------------------------------------------------------===//

// A general mechanism to further confine the given `attr` with all the
// `constraints`. This allows to compose complex constraints out of a series
// of more primitive ones.
class ConfinedAttr<Attr attr, list<AttrConstraint> constraints> : Attr<
    And<!listconcat([attr.predicate],
                      !foreach(pred, constraints, pred.predicate))>,
    !foldl(/*init*/attr.summary, /*list*/constraints,
           prev, cur, prev # " " # cur.summary)> {
  let storageType = attr.storageType;
  let returnType = attr.returnType;
  let convertFromStorage = attr.convertFromStorage;
  let constBuilderCall = attr.constBuilderCall;
  let defaultValue = attr.defaultValue;
  let valueType = attr.valueType;
  let isOptional = attr.isOptional;

  let baseAttr = attr;

  list<AttrConstraint> attrConstraints = constraints;
}

// An AttrConstraint that holds if all attr constraints specified in
// 'constraints' hold.
class AllAttrOf<list<AttrConstraint> constraints> : AttrConstraint<
    And<!listconcat([!head(constraints).predicate],
                    !foreach(pred, !tail(constraints), pred.predicate))>,
    !interleave(!foreach(con, constraints, con.summary), " and ")> {
}

class IntNEQValue<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getInt() != " # n>,
    "whose value is not " # n>;

class IntMinValue<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getInt() >= " # n>,
    "whose minimum value is " # n>;

class IntMaxValue<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getInt() <= " # n>,
    "whose maximum value is " # n>;

def IntNonNegative : AttrConstraint<
    CPred<"!::llvm::cast<::mlir::IntegerAttr>($_self).getValue().isNegative()">,
    "whose value is non-negative">;

def IntPositive : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getValue().isStrictlyPositive()">,
    "whose value is positive">;

class ArrayMaxCount<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() <= " # n>,
    "with at most " # n # " elements">;

class ArrayMinCount<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() >= " # n>,
    "with at least " # n # " elements">;

class ArrayCount<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() == " #n>,
    "with exactly " # n # " elements">;

class DenseArrayCount<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::DenseArrayAttr>($_self).size() == " #n>,
    "with exactly " # n # " elements">;

class DenseArrayMaxCt<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::DenseArrayAttr>($_self).size() <= " # n>,
    "with at most " # n # " elements">;

class DenseArrayMinCt<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::DenseArrayAttr>($_self).size() >= " # n>,
    "with at least " # n # " elements">;

class DenseArrayStrictlyPositive<DenseArrayAttrBase arrayType> : AttrConstraint<
  CPred<"::llvm::all_of(::llvm::cast<" # arrayType #">($_self).asArrayRef(), "
                        "[&](auto v) { return v > 0; })">,
  "whose value is positive">;

class DenseArrayNonNegative<DenseArrayAttrBase arrayType> : AttrConstraint<
  CPred<"::llvm::all_of(::llvm::cast<" # arrayType #">($_self).asArrayRef(), "
                        "[&](auto v) { return v >= 0; })">,
  "whose value is non-negative">;

class DenseArraySorted<DenseArrayAttrBase arrayType> : AttrConstraint<
    CPred<"llvm::is_sorted(::llvm::cast<" # arrayType # ">($_self).asArrayRef())">,
    "should be in non-decreasing order">;

class DenseArrayStrictlySorted<DenseArrayAttrBase arrayType> : AttrConstraint<
    And<[
      CPred<"llvm::is_sorted(::llvm::cast<" # arrayType # ">($_self).asArrayRef())">,
      // Check that no two adjacent elements are the same.
      CPred<"[](" # arrayType.returnType # " a) {\n"
        "return std::adjacent_find(std::begin(a), std::end(a)) == "
        "std::end(a);\n"
        "}(::llvm::cast<" # arrayType # ">($_self).asArrayRef())"
      >]>,
    "should be in increasing order">;

class IntArrayNthElemEq<int index, int value> : AttrConstraint<
    And<[
      CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() > " # index>,
      CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<::mlir::ArrayAttr>($_self)["
            # index # "]).getInt() == "  # value>
       ]>,
    "whose " # index # "-th element must be " # value>;

class IntArrayNthElemMinValue<int index, int min> : AttrConstraint<
    And<[
      CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() > " # index>,
      CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<::mlir::ArrayAttr>($_self)["
            # index # "]).getInt() >= " # min>
        ]>,
    "whose " # index # "-th element must be at least " # min>;

class IntArrayNthElemMaxValue<int index, int max> : AttrConstraint<
    And<[
      CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() > " # index>,
      CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<::mlir::ArrayAttr>($_self)["
            # index # "]).getInt() <= " # max>
        ]>,
    "whose " # index # "-th element must be at most " # max>;

class IntArrayNthElemInRange<int index, int min, int max> : AttrConstraint<
    And<[
      CPred<"::llvm::cast<::mlir::ArrayAttr>($_self).size() > " # index>,
      CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<::mlir::ArrayAttr>($_self)["
            # index # "]).getInt() >= " # min>,
      CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<::mlir::ArrayAttr>($_self)["
            # index # "]).getInt() <= " # max>
        ]>,
    "whose " # index # "-th element must be at least " # min # " and at most " # max>;

def IsNullAttr : AttrConstraint<
    CPred<"!$_self">, "empty attribute (for optional attributes)">;

#endif // COMMON_ATTR_CONSTRAINTS_TD
