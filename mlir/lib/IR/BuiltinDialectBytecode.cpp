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
  DictionaryAttr readDictionaryAttr(DialectBytecodeReader &reader) const;
  FloatAttr readFloatAttr(DialectBytecodeReader &reader) const;
  IntegerAttr readIntegerAttr(DialectBytecodeReader &reader) const;
  StringAttr readStringAttr(DialectBytecodeReader &reader, bool hasType) const;
  SymbolRefAttr readSymbolRefAttr(DialectBytecodeReader &reader,
                                  bool hasNestedRefs) const;
  TypeAttr readTypeAttr(DialectBytecodeReader &reader) const;

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override;
  void write(ArrayAttr attr, DialectBytecodeWriter &writer) const;
  void write(DictionaryAttr attr, DialectBytecodeWriter &writer) const;
  void write(IntegerAttr attr, DialectBytecodeWriter &writer) const;
  void write(FloatAttr attr, DialectBytecodeWriter &writer) const;
  void write(StringAttr attr, DialectBytecodeWriter &writer) const;
  void write(SymbolRefAttr attr, DialectBytecodeWriter &writer) const;
  void write(TypeAttr attr, DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Types

  Type readType(DialectBytecodeReader &reader) const override;
  IntegerType readIntegerType(DialectBytecodeReader &reader) const;
  FunctionType readFunctionType(DialectBytecodeReader &reader) const;

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override;
  void write(IntegerType type, DialectBytecodeWriter &writer) const;
  void write(FunctionType type, DialectBytecodeWriter &writer) const;
};
} // namespace

void builtin_dialect_detail::addBytecodeInterface(BuiltinDialect *dialect) {
  dialect->addInterfaces<BuiltinDialectBytecodeInterface>();
}

//===----------------------------------------------------------------------===//
// Attributes: Reader

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
  default:
    reader.emitError() << "unknown builtin attribute code: " << code;
    return Attribute();
  }
}

ArrayAttr BuiltinDialectBytecodeInterface::readArrayAttr(
    DialectBytecodeReader &reader) const {
  SmallVector<Attribute> elements;
  if (failed(reader.readAttributes(elements)))
    return ArrayAttr();
  return ArrayAttr::get(getContext(), elements);
}

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

TypeAttr BuiltinDialectBytecodeInterface::readTypeAttr(
    DialectBytecodeReader &reader) const {
  Type type;
  if (failed(reader.readType(type)))
    return TypeAttr();
  return TypeAttr::get(type);
}

//===----------------------------------------------------------------------===//
// Attributes: Writer

LogicalResult BuiltinDialectBytecodeInterface::writeAttribute(
    Attribute attr, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<ArrayAttr, DictionaryAttr, FloatAttr, IntegerAttr, StringAttr,
            SymbolRefAttr, TypeAttr>([&](auto attr) {
        write(attr, writer);
        return success();
      })
      .Case([&](UnitAttr) {
        writer.writeVarInt(builtin_encoding::kUnitAttr);
        return success();
      })
      .Default([&](Attribute) { return failure(); });
}

void BuiltinDialectBytecodeInterface::write(
    ArrayAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kArrayAttr);
  writer.writeAttributes(attr.getValue());
}

void BuiltinDialectBytecodeInterface::write(
    DictionaryAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kDictionaryAttr);
  writer.writeList(attr.getValue(), [&](NamedAttribute attr) {
    writer.writeAttribute(attr.getName());
    writer.writeAttribute(attr.getValue());
  });
}

void BuiltinDialectBytecodeInterface::write(
    FloatAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kFloatAttr);
  writer.writeType(attr.getType());
  writer.writeAPFloatWithKnownSemantics(attr.getValue());
}

void BuiltinDialectBytecodeInterface::write(
    IntegerAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kIntegerAttr);
  writer.writeType(attr.getType());
  writer.writeAPIntWithKnownWidth(attr.getValue());
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

void BuiltinDialectBytecodeInterface::write(
    SymbolRefAttr attr, DialectBytecodeWriter &writer) const {
  ArrayRef<FlatSymbolRefAttr> nestedRefs = attr.getNestedReferences();
  writer.writeVarInt(nestedRefs.empty() ? builtin_encoding::kFlatSymbolRefAttr
                                        : builtin_encoding::kSymbolRefAttr);

  writer.writeAttribute(attr.getRootReference());
  if (!nestedRefs.empty())
    writer.writeAttributes(nestedRefs);
}

void BuiltinDialectBytecodeInterface::write(
    TypeAttr attr, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kTypeAttr);
  writer.writeType(attr.getValue());
}

//===----------------------------------------------------------------------===//
// Types: Reader

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
  default:
    reader.emitError() << "unknown builtin type code: " << code;
    return Type();
  }
}

IntegerType BuiltinDialectBytecodeInterface::readIntegerType(
    DialectBytecodeReader &reader) const {
  uint64_t encoding;
  if (failed(reader.readVarInt(encoding)))
    return IntegerType();
  return IntegerType::get(
      getContext(), encoding >> 2,
      static_cast<IntegerType::SignednessSemantics>(encoding & 0x3));
}

FunctionType BuiltinDialectBytecodeInterface::readFunctionType(
    DialectBytecodeReader &reader) const {
  SmallVector<Type> inputs, results;
  if (failed(reader.readTypes(inputs)) || failed(reader.readTypes(results)))
    return FunctionType();
  return FunctionType::get(getContext(), inputs, results);
}

//===----------------------------------------------------------------------===//
// Types: Writer

LogicalResult BuiltinDialectBytecodeInterface::writeType(
    Type type, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<IntegerType, FunctionType>([&](auto type) {
        write(type, writer);
        return success();
      })
      .Case([&](IndexType) {
        return writer.writeVarInt(builtin_encoding::kIndexType), success();
      })
      .Default([&](Type) { return failure(); });
}

void BuiltinDialectBytecodeInterface::write(
    IntegerType type, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kIntegerType);
  writer.writeVarInt((type.getWidth() << 2) | type.getSignedness());
}

void BuiltinDialectBytecodeInterface::write(
    FunctionType type, DialectBytecodeWriter &writer) const {
  writer.writeVarInt(builtin_encoding::kFunctionType);
  writer.writeTypes(type.getInputs());
  writer.writeTypes(type.getResults());
}
