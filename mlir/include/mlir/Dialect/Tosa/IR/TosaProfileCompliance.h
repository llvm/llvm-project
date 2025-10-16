//===- TosaProfileCompliance.h - Tosa Profile-based Compliance Validation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_TRANSFORMS_TOSAPROFILECOMPILANCE_H
#define MLIR_DIALECT_TOSA_TRANSFORMS_TOSAPROFILECOMPILANCE_H

#include <unordered_map>

#include "mlir/Dialect/Tosa/IR/TargetEnv.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Support/TypeID.h"

using namespace mlir;
using namespace mlir::tosa;

//===----------------------------------------------------------------------===//
// Type Compilance Definition
//===----------------------------------------------------------------------===//

typedef struct {
  mlir::TypeID typeID;
  uint32_t bitWidth;
} TypeInfo;

enum CheckCondition {
  invalid,
  // Valid when any of the profile (extension) requirement is meet.
  anyOf,
  // Valid when all of the profile (extension) requirement are meet.
  allOf
};

using VersionedTypeInfo =
    std::pair<SmallVector<TypeInfo>, SpecificationVersion>;

template <typename T>
struct OpComplianceInfo {
  // Certain operations require multiple modes enabled.
  // e.g. cast bf16 to fp8e4m3 requires EXT-BF16 and EXT-FP8E4M3.
  SmallVector<T> mode;
  SmallVector<VersionedTypeInfo> operandTypeInfoSet;
  CheckCondition condition = CheckCondition::anyOf;
};

using OperationProfileComplianceMap =
    std::unordered_map<std::string, SmallVector<OpComplianceInfo<Profile>>>;
using OperationExtensionComplianceMap =
    std::unordered_map<std::string, SmallVector<OpComplianceInfo<Extension>>>;

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
  TypeInfo convertTypeToInfo(Type type) {
    return {type.getTypeID(), type.getIntOrFloatBitWidth()};
  }

  TypeInfo convertValueToInfo(Value value) {
    return convertTypeToInfo(getElementTypeOrSelf(value.getType()));
  }

  LogicalResult populatationDispatch(Operation *op);

  LogicalResult populateProfileInfo(ValueRange operands, Value output);

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

  // Accessor of the compliance info map.
  template <typename T>
  std::unordered_map<std::string, SmallVector<OpComplianceInfo<T>>>
  getProfileComplianceMap() {
    // Only profile and extension compliance info are provided.
    return {};
  }

  // Verify if the operation is allowed to be executed in the given target
  // environment.
  LogicalResult checkProfile(Operation *op, const tosa::TargetEnv &targetEnv);
  LogicalResult checkExtension(Operation *op, const tosa::TargetEnv &targetEnv);
  LogicalResult checkInvalid(Operation *op);

  template <typename T>
  LogicalResult checkProfileOrExtension(
      Operation *op, const tosa::TargetEnv &targetEnv,
      const SmallVector<ArrayRef<T>> &specDefinedProfileSet);

  bool isSameTypeInfo(TypeInfo a, TypeInfo b) {
    return a.typeID == b.typeID && a.bitWidth == b.bitWidth;
  }

  // Find the required profiles or extensions from the compliance info according
  // to the operand type combination.
  template <typename T>
  OpComplianceInfo<T>
  findMatchedEntry(Operation *op, SmallVector<OpComplianceInfo<T>> compInfo);

  SmallVector<Profile> getCooperativeProfiles(Extension ext) {
    switch (ext) {
    case Extension::int16:
    case Extension::int4:
    case Extension::doubleround:
    case Extension::inexactround:
      return {Profile::pro_int};
    case Extension::bf16:
    case Extension::fp8e4m3:
    case Extension::fp8e5m2:
    case Extension::fft:
      return {Profile::pro_fp};
    case Extension::variable:
    case Extension::controlflow:
    case Extension::dynamic:
      return {Profile::pro_fp, Profile::pro_int};
    case Extension::none:
      return {};
    };
    llvm_unreachable("bad Extension type");
  }

  // Debug utilites.
  template <typename T>
  SmallVector<StringRef> stringifyProfile(ArrayRef<T> profiles);

  template <typename T>
  SmallVector<StringRef>
  stringifyProfile(const SmallVector<ArrayRef<T>> &profileSet);

  static llvm::SmallString<7> stringifyTypeInfo(const TypeInfo &typeInfo);

private:
  template <typename T>
  FailureOr<OpComplianceInfo<T>> getOperatorDefinition(Operation *op);

  OperationProfileComplianceMap profileComplianceMap;
  OperationExtensionComplianceMap extensionComplianceMap;
};

#endif // MLIR_DIALECT_TOSA_TRANSFORMS_TOSAPROFILECOMPILANCE_H
