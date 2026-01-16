//===-------------- TosaTarget.cpp - TOSA Target utilities ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TargetEnv.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace tosa {

llvm::SmallString<4> stringifyVersion(TosaSpecificationVersion version) {
  return llvm::formatv("{0}.{1}", version.getMajor(), version.getMinor());
}

TosaSpecificationVersion getMinVersion(const Profile &profile) {
  switch (profile) {
  case Profile::pro_int:
  case Profile::pro_fp:
    return TosaSpecificationVersion(1, 0);
  case Profile::none:
    return TosaSpecificationVersion(0, 0);
  }
  llvm_unreachable("Unknown TOSA profile");
}

TosaSpecificationVersion getMinVersion(const Extension &extension) {
  switch (extension) {
  case Extension::int16:
  case Extension::int4:
  case Extension::bf16:
  case Extension::fp8e4m3:
  case Extension::fp8e5m2:
  case Extension::fft:
  case Extension::variable:
  case Extension::controlflow:
  case Extension::doubleround:
  case Extension::inexactround:
  case Extension::dynamic:
    return TosaSpecificationVersion(1, 0);
  case Extension::mxfp:
  case Extension::int64:
  case Extension::mxfp_conv:
  case Extension::shape:
    return TosaSpecificationVersion(1, 1);
  case Extension::none:
    return TosaSpecificationVersion(0, 0);
  }
  llvm_unreachable("Unknown TOSA extension");
}

TosaSpecificationVersion getMinVersion(const Level &level) {
  switch (level) {
  case Level::eightK:
  case Level::none:
    return TosaSpecificationVersion(1, 0);
  }
  llvm_unreachable("Unknown TOSA level");
}

FailureOr<TargetEnv>
TargetEnv::createTargetEnvFromAttr(TargetEnvAttr targetAttr,
                                   Location targetEnvAttrLoc) {
  if (failed(verifyTargetInformation(targetAttr, targetEnvAttrLoc)))
    return failure();

  return TargetEnv(targetAttr.getSpecificationVersion(), targetAttr.getLevel(),
                   targetAttr.getProfiles(), targetAttr.getExtensions());
}

LogicalResult TargetEnv::verifyTargetInformation(TargetEnvAttr targetAttr,
                                                 Location targetAttrLoc) {
  TosaSpecificationVersion targetVersion(targetAttr.getSpecificationVersion());

  const auto isCompatibleWithTargetVersion =
      [&](const auto &targetEnum, Location targetAttrLoc,
          StringRef enumName) -> LogicalResult {
    const TosaSpecificationVersion minRequiredVersion =
        getMinVersion(targetEnum);
    if (!targetVersion.isBackwardsCompatibleWith(minRequiredVersion))
      return emitError(targetAttrLoc, enumName)
             << " '" << stringifyEnum(targetEnum)
             << "' is not compatible with the target version "
             << stringifyVersion(targetVersion)
             << ", minimum required version is "
             << stringifyVersion(minRequiredVersion);
    return success();
  };

  for (const auto &profile : targetAttr.getProfiles())
    if (failed(
            isCompatibleWithTargetVersion(profile, targetAttrLoc, "profile")))
      return failure();
  for (const auto &extension : targetAttr.getExtensions())
    if (failed(isCompatibleWithTargetVersion(extension, targetAttrLoc,
                                             "extension")))
      return failure();
  if (failed(isCompatibleWithTargetVersion(targetAttr.getLevel(), targetAttrLoc,
                                           "level")))
    return failure();

  return success();
}

TargetEnvAttr lookupTargetEnv(Operation *op) {
  while (op) {
    op = SymbolTable::getNearestSymbolTable(op);
    if (!op)
      break;

    if (auto attr = op->getAttrOfType<TargetEnvAttr>(TargetEnvAttr::name))
      return attr;

    op = op->getParentOp();
  }

  return {};
}

TargetEnvAttr getDefaultTargetEnv(MLIRContext *context) {
  return TargetEnvAttr::get(context, SpecificationVersion::V_1_0, Level::eightK,
                            {Profile::pro_int, Profile::pro_fp}, {});
}

TargetEnvAttr lookupTargetEnvOrDefault(Operation *op) {
  if (auto attr = lookupTargetEnv(op))
    return attr;

  return getDefaultTargetEnv(op->getContext());
}

} // namespace tosa
} // namespace mlir
