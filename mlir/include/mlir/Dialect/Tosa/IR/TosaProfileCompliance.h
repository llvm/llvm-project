//===- TosaProfileCompliance.h - Tosa Profile-based Compliance Validation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_TRANSFORMS_TOSAPROFILECOMPILANCE_H
#define MLIR_DIALECT_TOSA_TRANSFORMS_TOSAPROFILECOMPILANCE_H

#include <cstdint>
#include <optional>
#include <type_traits>

#include "mlir/Dialect/Tosa/IR/TargetEnv.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace tosa {
//===----------------------------------------------------------------------===//
// Type Compilance Definition
//===----------------------------------------------------------------------===//

enum class ComplianceTypeKind : uint8_t {
  None = 0,
  Integer,
  BFloat16,
  Float16,
  Float32,
  Float8E4M3FN,
  Float8E5M2,
  Float6E2M3FN,
  Float6E3M2FN,
  Float4E2M1FN,
  Float8E8M0FNU,
  MxInt8,
  BlockScaled,
  Unknown,
};

struct TypeInfo {
  constexpr TypeInfo()
      : typeKind(ComplianceTypeKind::None), bitWidth(0),
        valueTypeKind(ComplianceTypeKind::None),
        scaleTypeKind(ComplianceTypeKind::None), blockShape(std::nullopt) {}

  constexpr TypeInfo(ComplianceTypeKind typeKind, uint32_t bitWidth)
      : typeKind(typeKind), bitWidth(bitWidth),
        valueTypeKind(ComplianceTypeKind::None),
        scaleTypeKind(ComplianceTypeKind::None), blockShape(std::nullopt) {}

  constexpr TypeInfo(ComplianceTypeKind typeKind, uint32_t bitWidth,
                     ComplianceTypeKind valueTypeKind,
                     ComplianceTypeKind scaleTypeKind,
                     tosa::BlockShape blockShape)
      : typeKind(typeKind), bitWidth(bitWidth), valueTypeKind(valueTypeKind),
        scaleTypeKind(scaleTypeKind), blockShape(blockShape) {}

  ComplianceTypeKind typeKind;
  uint32_t bitWidth;
  ComplianceTypeKind valueTypeKind;
  ComplianceTypeKind scaleTypeKind;
  std::optional<tosa::BlockShape> blockShape;
};

enum CheckCondition {
  invalid,
  // Valid when any of the profile (extension) requirement is meet.
  anyOf,
  // Valid when all of the profile (extension) requirement are meet.
  allOf
};

template <typename T, size_t Capacity>
struct ComplianceInlineArray {
  T storage[Capacity]{};
  uint8_t numElements = 0;

  constexpr ComplianceInlineArray() = default;

  template <typename... Ts,
            typename = std::enable_if_t<(sizeof...(Ts) >= 1) &&
                                        (sizeof...(Ts) <= Capacity) &&
                                        (std::is_convertible_v<Ts, T> && ...)>>
  constexpr ComplianceInlineArray(Ts... elems)
      : storage{static_cast<T>(elems)...},
        numElements(static_cast<uint8_t>(sizeof...(Ts))) {}

  constexpr ArrayRef<T> asArrayRef() const {
    return ArrayRef<T>(storage, numElements);
  }
  constexpr operator ArrayRef<T>() const { return asArrayRef(); }
  constexpr bool empty() const { return numElements == 0; }
  constexpr size_t size() const { return numElements; }
};

template <typename T>
struct OpComplianceInfo {
  llvm::StringLiteral opName;
  // Certain operations require multiple modes enabled.
  // e.g. cast bf16 to fp8e4m3 requires EXT-BF16 and EXT-FP8E4M3.
  ComplianceInlineArray<T, 4> mode;
  ComplianceInlineArray<TypeInfo, 7> operandTypes;
  SpecificationVersion specVersion = SpecificationVersion::V_1_0;
  CheckCondition condition = CheckCondition::anyOf;
};

//===----------------------------------------------------------------------===//
// Tosa Profile And Extension Information Depot
//===----------------------------------------------------------------------===//

class ProfileInfoDepot {
public:
  ProfileInfoDepot(Operation *op) {
    if (failed(populatationDispatch(op)))
      op->emitOpError() << "fail to populate the profile info\n";
  }

  void addType(Type t) { tyInfo.push_back(convertTypeToInfo(t)); }
  void addValue(Value v) { tyInfo.push_back(convertValueToInfo(v)); }
  SmallVector<TypeInfo> getInfo() { return tyInfo; }

private:
  static ComplianceTypeKind getComplianceTypeKind(Type type) {
    if (isa<IntegerType>(type))
      return ComplianceTypeKind::Integer;
    if (isa<BFloat16Type>(type))
      return ComplianceTypeKind::BFloat16;
    if (isa<Float16Type>(type))
      return ComplianceTypeKind::Float16;
    if (isa<Float32Type>(type))
      return ComplianceTypeKind::Float32;
    if (isa<Float8E4M3FNType>(type))
      return ComplianceTypeKind::Float8E4M3FN;
    if (isa<Float8E5M2Type>(type))
      return ComplianceTypeKind::Float8E5M2;
    if (isa<Float6E2M3FNType>(type))
      return ComplianceTypeKind::Float6E2M3FN;
    if (isa<Float6E3M2FNType>(type))
      return ComplianceTypeKind::Float6E3M2FN;
    if (isa<Float4E2M1FNType>(type))
      return ComplianceTypeKind::Float4E2M1FN;
    if (isa<Float8E8M0FNUType>(type))
      return ComplianceTypeKind::Float8E8M0FNU;
    if (isa<tosa::mxint8Type>(type))
      return ComplianceTypeKind::MxInt8;
    if (isa<tosa::BlockScaledType>(type))
      return ComplianceTypeKind::BlockScaled;
    return ComplianceTypeKind::Unknown;
  }

  TypeInfo convertTypeToInfo(Type type) {
    if (auto blockScaledTy = dyn_cast<tosa::BlockScaledType>(type)) {
      Type valueTy = blockScaledTy.getValueType();
      Type scaleTy = blockScaledTy.getScaleType();
      return {ComplianceTypeKind::BlockScaled, tosa::getBitWidth(valueTy),
              getComplianceTypeKind(valueTy), getComplianceTypeKind(scaleTy),
              blockScaledTy.getBlockShape()};
    }
    return {getComplianceTypeKind(type), tosa::getBitWidth(type)};
  }

  TypeInfo convertValueToInfo(Value value) {
    return convertTypeToInfo(getElementTypeOrSelf(value.getType()));
  }

  LogicalResult populatationDispatch(Operation *op);

  // Add input operands and output results to the profile type info list
  LogicalResult populateProfileInfo(ValueRange operands, ValueRange results);

  // Base
  template <typename T>
  LogicalResult populateProfileInfo(T op) {
    return op->emitOpError()
           << "profile requirement for this op has not been defined";
  }
  // For conv2d, conv3d, transpose_conv2d, and depthwise_conv2d.
  template <typename T>
  LogicalResult populateProfileInfoConv(T op);

  // For reshape, slice, tile, and transpose.
  template <typename T>
  LogicalResult populateProfileInfoDataLayout(T op);

private:
  SmallVector<TypeInfo> tyInfo;
};

//===----------------------------------------------------------------------===//
// Tosa Profile And Extension Compliance Checker
//===----------------------------------------------------------------------===//

class TosaProfileCompliance {
public:
  explicit TosaProfileCompliance();

  // Accessor of the compliance info for an operation.
  template <typename T>
  ArrayRef<OpComplianceInfo<T>> getProfileComplianceInfos(StringRef opName) {
    // Only profile and extension compliance info are provided.
    return {};
  }

  // Verify if the operation is allowed to be executed in the given target
  // environment.
  LogicalResult checkProfile(Operation *op, const tosa::TargetEnv &targetEnv);
  LogicalResult checkExtension(Operation *op, const tosa::TargetEnv &targetEnv);
  LogicalResult checkInvalid(Operation *op);

  template <typename T>
  LogicalResult checkProfileOrExtension(Operation *op,
                                        const tosa::TargetEnv &targetEnv);

  bool isSameTypeInfo(TypeInfo a, TypeInfo b) {
    if (a.typeKind == ComplianceTypeKind::Unknown ||
        b.typeKind == ComplianceTypeKind::Unknown)
      return false;
    return a.typeKind == b.typeKind && a.bitWidth == b.bitWidth &&
           a.valueTypeKind == b.valueTypeKind &&
           a.scaleTypeKind == b.scaleTypeKind && a.blockShape == b.blockShape;
  }

  // Find the required profiles or extensions from the compliance info according
  // to the operand type combination.
  template <typename T>
  SmallVector<OpComplianceInfo<T>>
  findMatchedEntries(Operation *op, ArrayRef<OpComplianceInfo<T>> compInfo);

  // Debug utilites.
  template <typename T>
  SmallVector<StringRef> stringifyProfile(ArrayRef<T> profiles);

  template <typename T>
  SmallVector<StringRef>
  stringifyProfile(const SmallVector<ArrayRef<T>> &profileSet);

  static llvm::SmallString<32> stringifyTypeInfo(const TypeInfo &typeInfo);

private:
  template <typename T>
  FailureOr<SmallVector<OpComplianceInfo<T>>>
  getOperatorMatchedEntries(Operation *op);
};
} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_TRANSFORMS_TOSAPROFILECOMPILANCE_H
