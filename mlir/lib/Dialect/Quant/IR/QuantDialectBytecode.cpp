//===- QuantDialectBytecode.cpp - Quant Bytecode Implementation
//------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QuantDialectBytecode.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::quant;

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

namespace {
namespace quant_encoding {
/// This enum contains marker codes used to indicate which type is currently
/// being decoded, and how it should be decoded. The order of these codes should
/// generally be unchanged, as any changes will inevitably break compatibility
/// with older bytecode.
enum TypeCode {
  ///   AnyQuantizedType {
  ///     flags: varint
  ///     storageType: Type
  ///     storageTypeMin: svarint
  ///     storageTypeMax: svarint
  ///   }
  ///
  kAnyQuantizedType = 1,

  ///   AnyQuantizedType {
  ///     flags: varint
  ///     storageType: Type
  ///     expressedType: Type
  ///     storageTypeMin: svarint
  ///     storageTypeMax: svarint
  ///   }
  ///
  kAnyQuantizedTypeWithExpressedType = 2,

  ///   CalibratedQuantizedType {
  ///     expressedType: Type
  ///     min: APFloat
  ///     max: APFloat
  ///   }
  ///
  kCalibratedQuantizedType = 3,

  ///   UniformQuantizedType {
  ///     flags: varint
  ///     storageType: Type
  ///     expressedType: Type
  ///     scale: APFloat
  ///     zeroPoint: svarint
  ///     storageTypeMin: svarint
  ///     storageTypeMax: svarint
  ///   }
  ///
  kUniformQuantizedType = 4,

  ///   UniformQuantizedPerAxisType {
  ///     flags: varint
  ///     storageType: Type
  ///     expressedType: Type
  ///     quantizedDimension: varint
  ///     storageTypeMin: svarint
  ///     storageTypeMax: svarint
  ///     scale: APFloat[]
  ///     zeroPoint: svarint[]
  ///   }
  ///
  kUniformQuantizedPerAxisType = 5,
};

} // namespace quant_encoding
} // namespace

//===----------------------------------------------------------------------===//
// QuantDialectBytecodeInterface
//===----------------------------------------------------------------------===//

namespace {
/// This class implements the bytecode interface for the Quant dialect.
struct QuantDialectBytecodeInterface : public BytecodeDialectInterface {
  QuantDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Types

  Type readType(DialectBytecodeReader &reader) const override;
  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override;

  AnyQuantizedType readAnyQuantizedType(bool withExpressedType,
                                        DialectBytecodeReader &reader) const;
  void write(AnyQuantizedType type, DialectBytecodeWriter &writer) const;

  CalibratedQuantizedType
  readCalibratedQuantizedType(DialectBytecodeReader &reader) const;
  void write(CalibratedQuantizedType type, DialectBytecodeWriter &writer) const;

  UniformQuantizedType
  readUniformQuantizedType(DialectBytecodeReader &reader) const;
  void write(UniformQuantizedType type, DialectBytecodeWriter &writer) const;

  UniformQuantizedPerAxisType
  readUniformQuantizedPerAxisType(DialectBytecodeReader &reader) const;
  void write(UniformQuantizedPerAxisType type,
             DialectBytecodeWriter &writer) const;
};
} // namespace

void quant::detail::addBytecodeInterface(QuantizationDialect *dialect) {
  dialect->addInterfaces<QuantDialectBytecodeInterface>();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

Type QuantDialectBytecodeInterface::readType(
    DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Type();

  switch (code) {
  case quant_encoding::kAnyQuantizedType:
    return readAnyQuantizedType(/*withExpressedType=*/false, reader);
  case quant_encoding::kAnyQuantizedTypeWithExpressedType:
    return readAnyQuantizedType(/*withExpressedType=*/true, reader);
  case quant_encoding::kCalibratedQuantizedType:
    return readCalibratedQuantizedType(reader);
  case quant_encoding::kUniformQuantizedType:
    return readUniformQuantizedType(reader);
  case quant_encoding::kUniformQuantizedPerAxisType:
    return readUniformQuantizedPerAxisType(reader);

  default:
    reader.emitError() << "unknown builtin type code: " << code;
    return Type();
  }
}

LogicalResult
QuantDialectBytecodeInterface::writeType(Type type,
                                         DialectBytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<AnyQuantizedType, CalibratedQuantizedType, UniformQuantizedType>(
          [&](auto attr) { return write(attr, writer), success(); })
      .Default([&](Type) { return failure(); });
}

AnyQuantizedType QuantDialectBytecodeInterface::readAnyQuantizedType(
    bool withExpressedType, DialectBytecodeReader &reader) const {
  uint64_t flags;
  Type storageType, expressedType;
  int64_t storageTypeMin, storageTypeMax;
  if (failed(reader.readVarInt(flags)) ||
      failed(reader.readType(storageType)) ||
      (withExpressedType && failed(reader.readType(expressedType))) ||
      failed(reader.readSignedVarInt(storageTypeMin)) ||
      failed(reader.readSignedVarInt(storageTypeMax)))
    return reader.emitError("invalid AnyQuantizedType"), AnyQuantizedType();
  return AnyQuantizedType::get(flags, storageType, expressedType,
                               storageTypeMin, storageTypeMax);
}
void QuantDialectBytecodeInterface::write(AnyQuantizedType type,
                                          DialectBytecodeWriter &writer) const {
  if (type.getExpressedType())
    writer.writeVarInt(quant_encoding::kAnyQuantizedTypeWithExpressedType);
  else
    writer.writeVarInt(quant_encoding::kAnyQuantizedType);

  writer.writeVarInt(type.getFlags());
  writer.writeType(type.getStorageType());
  if (type.getExpressedType())
    writer.writeType(type.getExpressedType());
  writer.writeSignedVarInt(type.getStorageTypeMin());
  writer.writeSignedVarInt(type.getStorageTypeMax());
}

CalibratedQuantizedType
QuantDialectBytecodeInterface::readCalibratedQuantizedType(
    DialectBytecodeReader &reader) const {
  Type expressedType;
  FailureOr<APFloat> min, max;
  if (failed(reader.readType(expressedType)) ||
      failed(min = reader.readAPFloatWithKnownSemantics(
                 llvm::APFloat::IEEEdouble())) ||
      failed(max = reader.readAPFloatWithKnownSemantics(
                 llvm::APFloat::IEEEdouble())))
    return reader.emitError("invalid CalibratedQuantizedType"),
           CalibratedQuantizedType();
  return CalibratedQuantizedType::get(expressedType,
                                      min.value().convertToDouble(),
                                      max.value().convertToDouble());
}
void QuantDialectBytecodeInterface::write(CalibratedQuantizedType type,
                                          DialectBytecodeWriter &writer) const {
  writer.writeVarInt(quant_encoding::kCalibratedQuantizedType);
  writer.writeType(type.getExpressedType());
  writer.writeAPFloatWithKnownSemantics(APFloat(type.getMin()));
  writer.writeAPFloatWithKnownSemantics(APFloat(type.getMax()));
}

UniformQuantizedType QuantDialectBytecodeInterface::readUniformQuantizedType(
    DialectBytecodeReader &reader) const {
  uint64_t flags;
  Type storageType, expressedType;
  FailureOr<APFloat> scale;
  int64_t zeroPoint, storageTypeMin, storageTypeMax;
  if (failed(reader.readVarInt(flags)) ||
      failed(reader.readType(storageType)) ||
      failed(reader.readType(expressedType)) ||
      failed(scale = reader.readAPFloatWithKnownSemantics(
                 llvm::APFloat::IEEEdouble())) ||
      failed(reader.readSignedVarInt(zeroPoint)) ||
      failed(reader.readSignedVarInt(storageTypeMin)) ||
      failed(reader.readSignedVarInt(storageTypeMax)))
    return reader.emitError("invalid UniformQuantizedType"),
           UniformQuantizedType();
  return UniformQuantizedType::get(flags, storageType, expressedType,
                                   scale.value().convertToDouble(), zeroPoint,
                                   storageTypeMin, storageTypeMax);
}
void QuantDialectBytecodeInterface::write(UniformQuantizedType type,
                                          DialectBytecodeWriter &writer) const {
  writer.writeVarInt(quant_encoding::kUniformQuantizedType);
  writer.writeVarInt(type.getFlags());
  writer.writeType(type.getStorageType());
  writer.writeType(type.getExpressedType());
  writer.writeAPFloatWithKnownSemantics(APFloat(type.getScale()));
  writer.writeSignedVarInt(type.getZeroPoint());
  writer.writeSignedVarInt(type.getStorageTypeMin());
  writer.writeSignedVarInt(type.getStorageTypeMax());
}

UniformQuantizedPerAxisType
QuantDialectBytecodeInterface::readUniformQuantizedPerAxisType(
    DialectBytecodeReader &reader) const {
  uint64_t flags;
  Type storageType, expressedType;
  SmallVector<double> scales;
  SmallVector<int64_t> zeroPoints;
  uint64_t quantizedDimension;
  int64_t storageTypeMin, storageTypeMax;

  auto scalesRead = [&](double &val) -> LogicalResult {
    FailureOr<APFloat> fl =
        reader.readAPFloatWithKnownSemantics(APFloat::IEEEdouble());
    if (succeeded(fl)) {
      val = fl.value().convertToDouble();
      return success();
    }
    return failure();
  };

  if (failed(reader.readVarInt(flags)) ||
      failed(reader.readType(storageType)) ||
      failed(reader.readType(expressedType)) ||
      failed(reader.readList(scales, scalesRead)) ||
      failed(reader.readSignedVarInts(zeroPoints)) ||
      failed(reader.readVarInt(quantizedDimension)) ||
      failed(reader.readSignedVarInt(storageTypeMin)) ||
      failed(reader.readSignedVarInt(storageTypeMax)))
    return reader.emitError("invalid UniformQuantizedPerAxisType"),
           UniformQuantizedPerAxisType();
  return UniformQuantizedPerAxisType::get(
      flags, storageType, expressedType, scales, zeroPoints,
      (int32_t)quantizedDimension, storageTypeMin, storageTypeMax);
}
void QuantDialectBytecodeInterface::write(UniformQuantizedPerAxisType type,
                                          DialectBytecodeWriter &writer) const {
  writer.writeVarInt(quant_encoding::kUniformQuantizedType);
  writer.writeVarInt(type.getFlags());
  writer.writeType(type.getStorageType());
  writer.writeType(type.getExpressedType());
  writer.writeList(type.getScales(), [&](double val) {
    writer.writeAPFloatWithKnownSemantics(APFloat(val));
  });
  writer.writeSignedVarInts(type.getZeroPoints());
  writer.writeVarInt(type.getQuantizedDimension());
  writer.writeSignedVarInt(type.getStorageTypeMin());
  writer.writeSignedVarInt(type.getStorageTypeMax());
}
