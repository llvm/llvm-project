//===- BuiltinDialectBytecode.cpp - Builtin Bytecode Implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BuiltinDialectBytecode.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

namespace {
namespace builtin_encoding {
/// This enum contains marker codes used to indicate which attribute is
/// currently being decoded, and how it should be decoded. The order of these
/// codes should generally be unchanged, as any changes will inevitably break
/// compatibility with older bytecode.
enum AttributeCode {
  ///   ArrayAttr {
  ///     elements: Attribute[]
  ///   }
  ///
  kArrayAttr = 0,

  ///   DictionaryAttr {
  ///     attrs: <StringAttr, Attribute>[]
  ///   }
  kDictionaryAttr = 1,

  ///   StringAttr {
  ///     value: string
  ///   }
  kStringAttr = 2,

  ///   StringAttrWithType {
  ///     value: string,
  ///     type: Type
  ///   }
  /// A variant of StringAttr with a type.
  kStringAttrWithType = 3,

  ///   FlatSymbolRefAttr {
  ///     rootReference: StringAttr
  ///   }
  /// A variant of SymbolRefAttr with no leaf references.
  kFlatSymbolRefAttr = 4,

  ///   SymbolRefAttr {
  ///     rootReference: StringAttr,
  ///     leafReferences: FlatSymbolRefAttr[]
  ///   }
  kSymbolRefAttr = 5,

  ///   TypeAttr {
  ///     value: Type
  ///   }
  kTypeAttr = 6,

  ///   UnitAttr {
  ///   }
  kUnitAttr = 7,

  ///   IntegerAttr {
  ///     type: Type
  ///     value: APInt,
  ///   }
  kIntegerAttr = 8,

  ///   FloatAttr {
  ///     type: FloatType
  ///     value: APFloat
  ///   }
  kFloatAttr = 9,

  ///   CallSiteLoc {
  ///    callee: LocationAttr,
  ///    caller: LocationAttr
  ///   }
  kCallSiteLoc = 10,

  ///   FileLineColLoc {
  ///     file: StringAttr,
  ///     line: varint,
  ///     column: varint
  ///   }
  kFileLineColLoc = 11,

  ///   FusedLoc {
  ///     locations: LocationAttr[]
  ///   }
  kFusedLoc = 12,

  ///   FusedLocWithMetadata {
  ///     locations: LocationAttr[],
  ///     metadata: Attribute
  ///   }
  /// A variant of FusedLoc with metadata.
  kFusedLocWithMetadata = 13,

  ///   NameLoc {
  ///     name: StringAttr,
  ///     childLoc: LocationAttr
  ///   }
  kNameLoc = 14,

  ///   UnknownLoc {
  ///   }
  kUnknownLoc = 15,

  ///   DenseResourceElementsAttr {
  ///     type: Type,
  ///     handle: ResourceHandle
  ///   }
  kDenseResourceElementsAttr = 16,
};

/// This enum contains marker codes used to indicate which type is currently
/// being decoded, and how it should be decoded. The order of these codes should
/// generally be unchanged, as any changes will inevitably break compatibility
/// with older bytecode.
enum TypeCode {
  ///   IntegerType {
  ///     widthAndSignedness: varint // (width << 2) | (signedness)
  ///   }
  ///
  kIntegerType = 0,

  ///   IndexType {
  ///   }
  ///
  kIndexType = 1,

  ///   FunctionType {
  ///     inputs: Type[],
  ///     results: Type[]
  ///   }
  ///
  kFunctionType = 2,

  ///   BFloat16Type {
  ///   }
  ///
  kBFloat16Type = 3,

  ///   Float16Type {
  ///   }
  ///
  kFloat16Type = 4,

  ///   Float32Type {
  ///   }
  ///
  kFloat32Type = 5,

  ///   Float64Type {
  ///   }
  ///
  kFloat64Type = 6,

  ///   Float80Type {
  ///   }
  ///
  kFloat80Type = 7,

  ///   Float128Type {
  ///   }
  ///
  kFloat128Type = 8,

  ///   ComplexType {
  ///     elementType: Type
  ///   }
  ///
  kComplexType = 9,

  ///   MemRefType {
  ///     shape: svarint[],
  ///     elementType: Type,
  ///     layout: Attribute
  ///   }
  ///
  kMemRefType = 10,

  ///   MemRefTypeWithMemSpace {
  ///     memorySpace: Attribute,
  ///     shape: svarint[],
  ///     elementType: Type,
  ///     layout: Attribute
  ///   }
  /// Variant of MemRefType with non-default memory space.
  kMemRefTypeWithMemSpace = 11,

  ///   NoneType {
  ///   }
  ///
  kNoneType = 12,

  ///   RankedTensorType {
  ///     shape: svarint[],
  ///     elementType: Type,
  ///   }
  ///
  kRankedTensorType = 13,

  ///   RankedTensorTypeWithEncoding {
  ///     encoding: Attribute,
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  /// Variant of RankedTensorType with an encoding.
  kRankedTensorTypeWithEncoding = 14,

  ///   TupleType {
  ///     elementTypes: Type[]
  ///   }
  kTupleType = 15,

  ///   UnrankedMemRefType {
  ///     shape: svarint[]
  ///   }
  ///
  kUnrankedMemRefType = 16,

  ///   UnrankedMemRefTypeWithMemSpace {
  ///     memorySpace: Attribute,
  ///     shape: svarint[]
  ///   }
  /// Variant of UnrankedMemRefType with non-default memory space.
  kUnrankedMemRefTypeWithMemSpace = 17,

  ///   UnrankedTensorType {
  ///     elementType: Type
  ///   }
  ///
  kUnrankedTensorType = 18,

  ///   VectorType {
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  ///
  kVectorType = 19,

  ///   VectorTypeWithScalableDims {
  ///     numScalableDims: varint,
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  /// Variant of VectorType with scalable dimensions.
  kVectorTypeWithScalableDims = 20,
};

} // namespace builtin_encoding
} // namespace

//===----------------------------------------------------------------------===//
// BuiltinDialectBytecodeInterface
//===----------------------------------------------------------------------===//

namespace {
/// This class implements the bytecode interface for the builtin dialect.
struct BuiltinDialectBytecodeInterface : public BytecodeDialectInterface {
  BuiltinDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  Attribute readAttribute(DialectBytecodeReader &reader) const override;
  ArrayAttr readArrayAttr(DialectBytecodeReader &reader) const;
  DenseResourceElementsAttr
  readDenseResourceElementsAttr(DialectBytecodeReader &reader) const;
  DictionaryAttr readDictionaryAttr(DialectBytecodeReader &reader) const;
  FloatAttr readFloatAttr(DialectBytecodeReader &reader) const;
  IntegerAttr readIntegerAttr(DialectBytecodeReader &reader) const;
  StringAttr readStringAttr(DialectBytecodeReader &reader, bool hasType) const;
  SymbolRefAttr readSymbolRefAttr(DialectBytecodeReader &reader,
                                  bool hasNestedRefs) const;
  TypeAttr readTypeAttr(DialectBytecodeReader &reader) const;

  LocationAttr readCallSiteLoc(DialectBytecodeReader &reader) const;
  LocationAttr readFileLineColLoc(DialectBytecodeReader &reader) const;
  LocationAttr readFusedLoc(DialectBytecodeReader &reader,
                            bool hasMetadata) const;
  LocationAttr readNameLoc(DialectBytecodeReader &reader) const;

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override;
  void write(ArrayAttr attr, DialectBytecodeWriter &writer) const;
  void write(DenseResourceElementsAttr attr,
             DialectBytecodeWriter &writer) const;
  void write(DictionaryAttr attr, DialectBytecodeWriter &writer) const;
  void write(IntegerAttr attr, DialectBytecodeWriter &writer) const;
  void write(FloatAttr attr, DialectBytecodeWriter &writer) const;
  void write(StringAttr attr, DialectBytecodeWriter &writer) const;
  void write(SymbolRefAttr attr, DialectBytecodeWriter &writer) const;
  void write(TypeAttr attr, DialectBytecodeWriter &writer) const;

  void write(CallSiteLoc attr, DialectBytecodeWriter &writer) const;
  void write(FileLineColLoc attr, DialectBytecodeWriter &writer) const;
  void write(FusedLoc attr, DialectBytecodeWriter &writer) const;
  void write(NameLoc attr, DialectBytecodeWriter &writer) const;
  LogicalResult write(OpaqueLoc attr, DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Types

  Type readType(DialectBytecodeReader &reader) const override;
  ComplexType readComplexType(DialectBytecodeReader &reader) const;
  IntegerType readIntegerType(DialectBytecodeReader &reader) const;
  FunctionType readFunctionType(DialectBytecodeReader &reader) const;
  MemRefType readMemRefType(DialectBytecodeReader &reader,
                            bool hasMemSpace) const;
  RankedTensorType readRankedTensorType(DialectBytecodeReader &reader,
                                        bool hasEncoding) const;
  TupleType readTupleType(DialectBytecodeReader &reader) const;
  UnrankedMemRefType readUnrankedMemRefType(DialectBytecodeReader &reader,
                                            bool hasMemSpace) const;
  UnrankedTensorType
  readUnrankedTensorType(DialectBytecodeReader &reader) const;
  VectorType readVectorType(DialectBytecodeReader &reader,
                            bool hasScalableDims) const;

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override;
  void write(ComplexType type, DialectBytecodeWriter &writer) const;
  void write(IntegerType type, DialectBytecodeWriter &writer) const;
  void write(FunctionType type, DialectBytecodeWriter &writer) const;
  void write(MemRefType type, DialectBytecodeWriter &writer) const;
  void write(RankedTensorType type, DialectBytecodeWriter &writer) const;
  void write(TupleType type, DialectBytecodeWriter &writer) const;
  void write(UnrankedMemRefType type, DialectBytecodeWriter &writer) const;
  void write(UnrankedTensorType type, DialectBytecodeWriter &writer) const;
  void write(VectorType type, DialectBytecodeWriter &writer) const;
};
} // namespace

void builtin_dialect_detail::addBytecodeInterface(BuiltinDialect *dialect) {
  dialect->addInterfaces<BuiltinDialectBytecodeInterface>();
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

Attribute BuiltinDialectBytecodeInterface::readAttribute(
    DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Attribute();
  switch (code) {
  case builtin_encoding::kArrayAttr:
    return readArrayAttr(reader);
  case builtin_encoding::kDictionaryAttr:
    return readDictionaryAttr(reader);
  case builtin_encoding::kStringAttr:
    return readStringAttr(reader, /*hasType=*/false);
  case builtin_encoding::kStringAttrWithType:
    return readStringAttr(reader, /*hasType=*/true);
  case builtin_encoding::kFlatSymbolRefAttr:
    return readSymbolRefAttr(reader, /*hasNestedRefs=*/false);
  case builtin_encoding::kSymbolRefAttr:
    return readSymbolRefAttr(reader, /*hasNestedRefs=*/true);
  case builtin_encoding::kTypeAttr:
    return readTypeAttr(reader);
  case builtin_encoding::kUnitAttr:
    return UnitAttr::get(getContext());
  case builtin_encoding::kIntegerAttr:
    return readIntegerAttr(reader);
  case builtin_encoding::kFloatAttr:
    return readFloatAttr(reader);
  case builtin_encoding::kCallSiteLoc:
    return readCallSiteLoc(reader);
  case builtin_encoding::kFileLineColLoc:
    return readFileLineColLoc(reader);
  case builtin_encoding::kFusedLoc:
    return readFusedLoc(reader, /*hasMetadata=*/false);
  case builtin_encoding::kFusedLocWithMetadata:
    return readFusedLoc(reader, /*hasMetadata=*/true);
  case builtin_encoding::kNameLoc:
    return readNameLoc(reader);
  case builtin_encoding::kUnknownLoc:
    return UnknownLoc::get(getContext());
  case builtin_encoding::kDenseResourceElementsAttr:
    return readDenseResourceElementsAttr(reader);
  default:
    reader.emitError() << "unknown builtin attribute code: " << code;
    return Attribute();
  }
}

LogicalResult BuiltinDialectBytecodeInterface::writeAttribute(
    Attribute attr, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<ArrayAttr, DenseResourceElementsAttr, DictionaryAttr, FloatAttr,
            IntegerAttr, StringAttr, SymbolRefAttr, TypeAttr>([&](auto attr) {
        write(attr, writer);
        return success();
      })
      .Case<CallSiteLoc, FileLineColLoc, FusedLoc, NameLoc>([&](auto attr) {
        write(attr, writer);
        return success();
      })
      .Case([&](OpaqueLoc attr) { return write(attr, writer); })
      .Case([&](UnitAttr) {
        writer.writeVarInt(builtin_encoding::kUnitAttr);
        return success();
      })
      .Case([&](UnknownLoc) {
        writer.writeVarInt(builtin_encoding::kUnknownLoc);
        return success();
      })
      .Default([&](Attribute) { return failure(); });
}

//===----------------------------------------------------------------------===//
// ArrayAttr

ArrayAttr BuiltinDialectBytecodeInterface::readArrayAttr(
    DialectBytecodeReader &reader) const {
  SmallVector<Attribute> elements;
  if (failed(reader.readAttributes(elements)))
    return ArrayAttr();
  return ArrayAttr::get(getContext(), elements);
}

void BuiltinDialectBytecodeInterface::write(
    ArrayAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kArrayAttr);
  writer.writeAttributes(attr.getValue());
}

//===----------------------------------------------------------------------===//
// DenseResourceElementsAttr

DenseResourceElementsAttr
BuiltinDialectBytecodeInterface::readDenseResourceElementsAttr(
    DialectBytecodeReader &reader) const {
  ShapedType type;
  if (failed(reader.readType(type)))
    return DenseResourceElementsAttr();

  FailureOr<DenseResourceElementsHandle> handle =
      reader.readResourceHandle<DenseResourceElementsHandle>();
  if (failed(handle))
    return DenseResourceElementsAttr();

  return DenseResourceElementsAttr::get(type, *handle);
}

void BuiltinDialectBytecodeInterface::write(
    DenseResourceElementsAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kDenseResourceElementsAttr);
  writer.writeType(attr.getType());
  writer.writeResourceHandle(attr.getRawHandle());
}

//===----------------------------------------------------------------------===//
// DictionaryAttr

DictionaryAttr BuiltinDialectBytecodeInterface::readDictionaryAttr(
    DialectBytecodeReader &reader) const {
  auto readNamedAttr = [&]() -> FailureOr<NamedAttribute> {
    StringAttr name;
    Attribute value;
    if (failed(reader.readAttribute(name)) ||
        failed(reader.readAttribute(value)))
      return failure();
    return NamedAttribute(name, value);
  };
  SmallVector<NamedAttribute> attrs;
  if (failed(reader.readList(attrs, readNamedAttr)))
    return DictionaryAttr();
  return DictionaryAttr::get(getContext(), attrs);
}

void BuiltinDialectBytecodeInterface::write(
    DictionaryAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kDictionaryAttr);
  writer.writeList(attr.getValue(), [&](NamedAttribute attr) {
    writer.writeAttribute(attr.getName());
    writer.writeAttribute(attr.getValue());
  });
}

//===----------------------------------------------------------------------===//
// FloatAttr

FloatAttr BuiltinDialectBytecodeInterface::readFloatAttr(
    DialectBytecodeReader &reader) const {
  FloatType type;
  if (failed(reader.readType(type)))
    return FloatAttr();
  FailureOr<APFloat> value =
      reader.readAPFloatWithKnownSemantics(type.getFloatSemantics());
  if (failed(value))
    return FloatAttr();
  return FloatAttr::get(type, *value);
}

void BuiltinDialectBytecodeInterface::write(
    FloatAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kFloatAttr);
  writer.writeType(attr.getType());
  writer.writeAPFloatWithKnownSemantics(attr.getValue());
}

//===----------------------------------------------------------------------===//
// IntegerAttr

IntegerAttr BuiltinDialectBytecodeInterface::readIntegerAttr(
    DialectBytecodeReader &reader) const {
  Type type;
  if (failed(reader.readType(type)))
    return IntegerAttr();

  // Extract the value storage width from the type.
  unsigned bitWidth;
  if (auto intType = type.dyn_cast<IntegerType>()) {
    bitWidth = intType.getWidth();
  } else if (type.isa<IndexType>()) {
    bitWidth = IndexType::kInternalStorageBitWidth;
  } else {
    reader.emitError()
        << "expected integer or index type for IntegerAttr, but got: " << type;
    return IntegerAttr();
  }

  FailureOr<APInt> value = reader.readAPIntWithKnownWidth(bitWidth);
  if (failed(value))
    return IntegerAttr();
  return IntegerAttr::get(type, *value);
}

void BuiltinDialectBytecodeInterface::write(
    IntegerAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kIntegerAttr);
  writer.writeType(attr.getType());
  writer.writeAPIntWithKnownWidth(attr.getValue());
}

//===----------------------------------------------------------------------===//
// StringAttr

StringAttr
BuiltinDialectBytecodeInterface::readStringAttr(DialectBytecodeReader &reader,
                                                bool hasType) const {
  StringRef string;
  if (failed(reader.readString(string)))
    return StringAttr();

  // Read the type if present.
  Type type;
  if (!hasType)
    type = NoneType::get(getContext());
  else if (failed(reader.readType(type)))
    return StringAttr();
  return StringAttr::get(string, type);
}

void BuiltinDialectBytecodeInterface::write(
    StringAttr attr, DialectBytecodeWriter &writer) const {
  // We only encode the type if it isn't NoneType, which is significantly less
  // common.
  Type type = attr.getType();
  if (!type.isa<NoneType>()) {
    writer.writeVarInt(builtin_encoding::kStringAttrWithType);
    writer.writeOwnedString(attr.getValue());
    writer.writeType(type);
    return;
  }
  writer.writeVarInt(builtin_encoding::kStringAttr);
  writer.writeOwnedString(attr.getValue());
}

//===----------------------------------------------------------------------===//
// SymbolRefAttr

SymbolRefAttr BuiltinDialectBytecodeInterface::readSymbolRefAttr(
    DialectBytecodeReader &reader, bool hasNestedRefs) const {
  StringAttr rootReference;
  if (failed(reader.readAttribute(rootReference)))
    return SymbolRefAttr();
  SmallVector<FlatSymbolRefAttr> nestedReferences;
  if (hasNestedRefs && failed(reader.readAttributes(nestedReferences)))
    return SymbolRefAttr();
  return SymbolRefAttr::get(rootReference, nestedReferences);
}

void BuiltinDialectBytecodeInterface::write(
    SymbolRefAttr attr, DialectBytecodeWriter &writer) const {
  ArrayRef<FlatSymbolRefAttr> nestedRefs = attr.getNestedReferences();
  writer.writeVarInt(nestedRefs.empty() ? builtin_encoding::kFlatSymbolRefAttr
                                        : builtin_encoding::kSymbolRefAttr);

  writer.writeAttribute(attr.getRootReference());
  if (!nestedRefs.empty())
    writer.writeAttributes(nestedRefs);
}

//===----------------------------------------------------------------------===//
// TypeAttr

TypeAttr BuiltinDialectBytecodeInterface::readTypeAttr(
    DialectBytecodeReader &reader) const {
  Type type;
  if (failed(reader.readType(type)))
    return TypeAttr();
  return TypeAttr::get(type);
}

void BuiltinDialectBytecodeInterface::write(
    TypeAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kTypeAttr);
  writer.writeType(attr.getValue());
}

//===----------------------------------------------------------------------===//
// CallSiteLoc

LocationAttr BuiltinDialectBytecodeInterface::readCallSiteLoc(
    DialectBytecodeReader &reader) const {
  LocationAttr callee, caller;
  if (failed(reader.readAttribute(callee)) ||
      failed(reader.readAttribute(caller)))
    return LocationAttr();
  return CallSiteLoc::get(callee, caller);
}

void BuiltinDialectBytecodeInterface::write(
    CallSiteLoc attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kCallSiteLoc);
  writer.writeAttribute(attr.getCallee());
  writer.writeAttribute(attr.getCaller());
}

//===----------------------------------------------------------------------===//
// FileLineColLoc

LocationAttr BuiltinDialectBytecodeInterface::readFileLineColLoc(
    DialectBytecodeReader &reader) const {
  StringAttr filename;
  uint64_t line, column;
  if (failed(reader.readAttribute(filename)) ||
      failed(reader.readVarInt(line)) || failed(reader.readVarInt(column)))
    return LocationAttr();
  return FileLineColLoc::get(filename, line, column);
}

void BuiltinDialectBytecodeInterface::write(
    FileLineColLoc attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kFileLineColLoc);
  writer.writeAttribute(attr.getFilename());
  writer.writeVarInt(attr.getLine());
  writer.writeVarInt(attr.getColumn());
}

//===----------------------------------------------------------------------===//
// FusedLoc

LocationAttr
BuiltinDialectBytecodeInterface::readFusedLoc(DialectBytecodeReader &reader,
                                              bool hasMetadata) const {
  // Parse the child locations.
  auto readLoc = [&]() -> FailureOr<Location> {
    LocationAttr locAttr;
    if (failed(reader.readAttribute(locAttr)))
      return failure();
    return Location(locAttr);
  };
  SmallVector<Location> locations;
  if (failed(reader.readList(locations, readLoc)))
    return LocationAttr();

  // Parse the metadata if present.
  Attribute metadata;
  if (hasMetadata && failed(reader.readAttribute(metadata)))
    return LocationAttr();

  return FusedLoc::get(locations, metadata, getContext());
}

void BuiltinDialectBytecodeInterface::write(
    FusedLoc attr, DialectBytecodeWriter &writer) const {
  if (Attribute metadata = attr.getMetadata()) {
    writer.writeVarInt(builtin_encoding::kFusedLocWithMetadata);
    writer.writeAttributes(attr.getLocations());
    writer.writeAttribute(metadata);
  } else {
    writer.writeVarInt(builtin_encoding::kFusedLoc);
    writer.writeAttributes(attr.getLocations());
  }
}

//===----------------------------------------------------------------------===//
// NameLoc

LocationAttr BuiltinDialectBytecodeInterface::readNameLoc(
    DialectBytecodeReader &reader) const {
  StringAttr name;
  LocationAttr childLoc;
  if (failed(reader.readAttribute(name)) ||
      failed(reader.readAttribute(childLoc)))
    return LocationAttr();
  return NameLoc::get(name, childLoc);
}

void BuiltinDialectBytecodeInterface::write(
    NameLoc attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kNameLoc);
  writer.writeAttribute(attr.getName());
  writer.writeAttribute(attr.getChildLoc());
}

//===----------------------------------------------------------------------===//
// OpaqueLoc

LogicalResult
BuiltinDialectBytecodeInterface::write(OpaqueLoc attr,
                                       DialectBytecodeWriter &writer) const {
  // We can't encode an OpaqueLoc directly given that it is in-memory only, so
  // encode the fallback instead.
  return writeAttribute(attr.getFallbackLocation(), writer);
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

Type BuiltinDialectBytecodeInterface::readType(
    DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Type();
  switch (code) {
  case builtin_encoding::kIntegerType:
    return readIntegerType(reader);
  case builtin_encoding::kIndexType:
    return IndexType::get(getContext());
  case builtin_encoding::kFunctionType:
    return readFunctionType(reader);
  case builtin_encoding::kBFloat16Type:
    return BFloat16Type::get(getContext());
  case builtin_encoding::kFloat16Type:
    return Float16Type::get(getContext());
  case builtin_encoding::kFloat32Type:
    return Float32Type::get(getContext());
  case builtin_encoding::kFloat64Type:
    return Float64Type::get(getContext());
  case builtin_encoding::kFloat80Type:
    return Float80Type::get(getContext());
  case builtin_encoding::kFloat128Type:
    return Float128Type::get(getContext());
  case builtin_encoding::kComplexType:
    return readComplexType(reader);
  case builtin_encoding::kMemRefType:
    return readMemRefType(reader, /*hasMemSpace=*/false);
  case builtin_encoding::kMemRefTypeWithMemSpace:
    return readMemRefType(reader, /*hasMemSpace=*/true);
  case builtin_encoding::kNoneType:
    return NoneType::get(getContext());
  case builtin_encoding::kRankedTensorType:
    return readRankedTensorType(reader, /*hasEncoding=*/false);
  case builtin_encoding::kRankedTensorTypeWithEncoding:
    return readRankedTensorType(reader, /*hasEncoding=*/true);
  case builtin_encoding::kTupleType:
    return readTupleType(reader);
  case builtin_encoding::kUnrankedMemRefType:
    return readUnrankedMemRefType(reader, /*hasMemSpace=*/false);
  case builtin_encoding::kUnrankedMemRefTypeWithMemSpace:
    return readUnrankedMemRefType(reader, /*hasMemSpace=*/true);
  case builtin_encoding::kUnrankedTensorType:
    return readUnrankedTensorType(reader);
  case builtin_encoding::kVectorType:
    return readVectorType(reader, /*hasScalableDims=*/false);
  case builtin_encoding::kVectorTypeWithScalableDims:
    return readVectorType(reader, /*hasScalableDims=*/true);

  default:
    reader.emitError() << "unknown builtin type code: " << code;
    return Type();
  }
}

LogicalResult BuiltinDialectBytecodeInterface::writeType(
    Type type, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<ComplexType, IntegerType, FunctionType, MemRefType,
            RankedTensorType, TupleType, UnrankedMemRefType, UnrankedTensorType,
            VectorType>([&](auto type) {
        write(type, writer);
        return success();
      })
      .Case([&](IndexType) {
        return writer.writeVarInt(builtin_encoding::kIndexType), success();
      })
      .Case([&](BFloat16Type) {
        return writer.writeVarInt(builtin_encoding::kBFloat16Type), success();
      })
      .Case([&](Float16Type) {
        return writer.writeVarInt(builtin_encoding::kFloat16Type), success();
      })
      .Case([&](Float32Type) {
        return writer.writeVarInt(builtin_encoding::kFloat32Type), success();
      })
      .Case([&](Float64Type) {
        return writer.writeVarInt(builtin_encoding::kFloat64Type), success();
      })
      .Case([&](Float80Type) {
        return writer.writeVarInt(builtin_encoding::kFloat80Type), success();
      })
      .Case([&](Float128Type) {
        return writer.writeVarInt(builtin_encoding::kFloat128Type), success();
      })
      .Case([&](NoneType) {
        return writer.writeVarInt(builtin_encoding::kNoneType), success();
      })
      .Default([&](Type) { return failure(); });
}

//===----------------------------------------------------------------------===//
// ComplexType

ComplexType BuiltinDialectBytecodeInterface::readComplexType(
    DialectBytecodeReader &reader) const {
  Type elementType;
  if (failed(reader.readType(elementType)))
    return ComplexType();
  return ComplexType::get(elementType);
}

void BuiltinDialectBytecodeInterface::write(
    ComplexType type, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kComplexType);
  writer.writeType(type.getElementType());
}

//===----------------------------------------------------------------------===//
// IntegerType

IntegerType BuiltinDialectBytecodeInterface::readIntegerType(
    DialectBytecodeReader &reader) const {
  uint64_t encoding;
  if (failed(reader.readVarInt(encoding)))
    return IntegerType();
  return IntegerType::get(
      getContext(), encoding >> 2,
      static_cast<IntegerType::SignednessSemantics>(encoding & 0x3));
}

void BuiltinDialectBytecodeInterface::write(
    IntegerType type, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kIntegerType);
  writer.writeVarInt((type.getWidth() << 2) | type.getSignedness());
}

//===----------------------------------------------------------------------===//
// FunctionType

FunctionType BuiltinDialectBytecodeInterface::readFunctionType(
    DialectBytecodeReader &reader) const {
  SmallVector<Type> inputs, results;
  if (failed(reader.readTypes(inputs)) || failed(reader.readTypes(results)))
    return FunctionType();
  return FunctionType::get(getContext(), inputs, results);
}

void BuiltinDialectBytecodeInterface::write(
    FunctionType type, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kFunctionType);
  writer.writeTypes(type.getInputs());
  writer.writeTypes(type.getResults());
}

//===----------------------------------------------------------------------===//
// MemRefType

MemRefType
BuiltinDialectBytecodeInterface::readMemRefType(DialectBytecodeReader &reader,
                                                bool hasMemSpace) const {
  Attribute memorySpace;
  if (hasMemSpace && failed(reader.readAttribute(memorySpace)))
    return MemRefType();
  SmallVector<int64_t> shape;
  Type elementType;
  MemRefLayoutAttrInterface layout;
  if (failed(reader.readSignedVarInts(shape)) ||
      failed(reader.readType(elementType)) ||
      failed(reader.readAttribute(layout)))
    return MemRefType();
  return MemRefType::get(shape, elementType, layout, memorySpace);
}

void BuiltinDialectBytecodeInterface::write(
    MemRefType type, DialectBytecodeWriter &writer) const {
  if (Attribute memSpace = type.getMemorySpace()) {
    writer.writeVarInt(builtin_encoding::kMemRefTypeWithMemSpace);
    writer.writeAttribute(memSpace);
  } else {
    writer.writeVarInt(builtin_encoding::kMemRefType);
  }
  writer.writeSignedVarInts(type.getShape());
  writer.writeType(type.getElementType());
  writer.writeAttribute(type.getLayout());
}

//===----------------------------------------------------------------------===//
// RankedTensorType

RankedTensorType BuiltinDialectBytecodeInterface::readRankedTensorType(
    DialectBytecodeReader &reader, bool hasEncoding) const {
  Attribute encoding;
  if (hasEncoding && failed(reader.readAttribute(encoding)))
    return RankedTensorType();
  SmallVector<int64_t> shape;
  Type elementType;
  if (failed(reader.readSignedVarInts(shape)) ||
      failed(reader.readType(elementType)))
    return RankedTensorType();
  return RankedTensorType::get(shape, elementType, encoding);
}

void BuiltinDialectBytecodeInterface::write(
    RankedTensorType type, DialectBytecodeWriter &writer) const {
  if (Attribute encoding = type.getEncoding()) {
    writer.writeVarInt(builtin_encoding::kRankedTensorTypeWithEncoding);
    writer.writeAttribute(encoding);
  } else {
    writer.writeVarInt(builtin_encoding::kRankedTensorType);
  }
  writer.writeSignedVarInts(type.getShape());
  writer.writeType(type.getElementType());
}

//===----------------------------------------------------------------------===//
// TupleType

TupleType BuiltinDialectBytecodeInterface::readTupleType(
    DialectBytecodeReader &reader) const {
  SmallVector<Type> elements;
  if (failed(reader.readTypes(elements)))
    return TupleType();
  return TupleType::get(getContext(), elements);
}

void BuiltinDialectBytecodeInterface::write(
    TupleType type, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kTupleType);
  writer.writeTypes(type.getTypes());
}

//===----------------------------------------------------------------------===//
// UnrankedMemRefType

UnrankedMemRefType BuiltinDialectBytecodeInterface::readUnrankedMemRefType(
    DialectBytecodeReader &reader, bool hasMemSpace) const {
  Attribute memorySpace;
  if (hasMemSpace && failed(reader.readAttribute(memorySpace)))
    return UnrankedMemRefType();
  Type elementType;
  if (failed(reader.readType(elementType)))
    return UnrankedMemRefType();
  return UnrankedMemRefType::get(elementType, memorySpace);
}

void BuiltinDialectBytecodeInterface::write(
    UnrankedMemRefType type, DialectBytecodeWriter &writer) const {
  if (Attribute memSpace = type.getMemorySpace()) {
    writer.writeVarInt(builtin_encoding::kUnrankedMemRefTypeWithMemSpace);
    writer.writeAttribute(memSpace);
  } else {
    writer.writeVarInt(builtin_encoding::kUnrankedMemRefType);
  }
  writer.writeType(type.getElementType());
}

//===----------------------------------------------------------------------===//
// UnrankedTensorType

UnrankedTensorType BuiltinDialectBytecodeInterface::readUnrankedTensorType(
    DialectBytecodeReader &reader) const {
  Type elementType;
  if (failed(reader.readType(elementType)))
    return UnrankedTensorType();
  return UnrankedTensorType::get(elementType);
}

void BuiltinDialectBytecodeInterface::write(
    UnrankedTensorType type, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kUnrankedTensorType);
  writer.writeType(type.getElementType());
}

//===----------------------------------------------------------------------===//
// VectorType

VectorType
BuiltinDialectBytecodeInterface::readVectorType(DialectBytecodeReader &reader,
                                                bool hasScalableDims) const {
  uint64_t numScalableDims = 0;
  if (hasScalableDims && failed(reader.readVarInt(numScalableDims)))
    return VectorType();
  SmallVector<int64_t> shape;
  Type elementType;
  if (failed(reader.readSignedVarInts(shape)) ||
      failed(reader.readType(elementType)))
    return VectorType();
  return VectorType::get(shape, elementType, numScalableDims);
}

void BuiltinDialectBytecodeInterface::write(
    VectorType type, DialectBytecodeWriter &writer) const {
  if (unsigned numScalableDims = type.getNumScalableDims()) {
    writer.writeVarInt(builtin_encoding::kVectorTypeWithScalableDims);
    writer.writeVarInt(numScalableDims);
  } else {
    writer.writeVarInt(builtin_encoding::kVectorType);
  }
  writer.writeSignedVarInts(type.getShape());
  writer.writeType(type.getElementType());
}
