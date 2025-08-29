//===- TargetEnv.h - Tosa target environment utilities ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for Tosa target environment (implementation).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_IR_TARGETENV_H
#define MLIR_DIALECT_TOSA_IR_TARGETENV_H

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace tosa {

struct TosaLevel {
  int32_t MAX_RANK = 0;
  int32_t MAX_KERNEL = 0;
  int32_t MAX_STRIDE = 0;
  int32_t MAX_SCALE = 0;
  int32_t MAX_LOG2_SIZE = 0;
  int32_t MAX_NESTING = 0;
  int32_t MAX_TENSOR_LIST_SIZE = 0;

  bool operator==(const TosaLevel &rhs) {
    return MAX_RANK == rhs.MAX_RANK && MAX_KERNEL == rhs.MAX_KERNEL &&
           MAX_STRIDE == rhs.MAX_STRIDE && MAX_SCALE == rhs.MAX_SCALE &&
           MAX_LOG2_SIZE == rhs.MAX_LOG2_SIZE &&
           MAX_NESTING == rhs.MAX_NESTING &&
           MAX_TENSOR_LIST_SIZE == rhs.MAX_TENSOR_LIST_SIZE;
  }
};

static constexpr TosaLevel TOSA_LEVEL_EIGHTK = {6, 8192, 8192, 256, 31, 6, 64};
static constexpr TosaLevel TOSA_LEVEL_NONE = {32, 2147483647, 2147483647, 2048,
                                              63, 256,        256};

TargetEnvAttr lookupTargetEnv(Operation *op);
TargetEnvAttr getDefaultTargetEnv(MLIRContext *context);

/// Queries the target environment recursively from enclosing symbol table ops
/// containing the given `op` or returns the default target environment as
/// returned by getDefaultTargetEnv() if not provided.
TargetEnvAttr lookupTargetEnvOrDefault(Operation *op);

/// A thin wrapper around the SpecificationVersion enum to represent
/// and provide utilities around the TOSA specification version.
class TosaSpecificationVersion {
public:
  TosaSpecificationVersion(uint32_t major, uint32_t minor)
      : majorVersion(major), minorVersion(minor) {}
  TosaSpecificationVersion(SpecificationVersion version)
      : TosaSpecificationVersion(fromVersionEnum(version)) {}

  bool isBackwardsCompatibleWith(TosaSpecificationVersion baseVersion) const {
    return this->majorVersion == baseVersion.majorVersion &&
           this->minorVersion >= baseVersion.minorVersion;
  }

  uint32_t getMajor() const { return majorVersion; }
  uint32_t getMinor() const { return minorVersion; }

private:
  uint32_t majorVersion = 0;
  uint32_t minorVersion = 0;

  static TosaSpecificationVersion
  fromVersionEnum(SpecificationVersion version) {
    switch (version) {
    case SpecificationVersion::V_1_0:
      return TosaSpecificationVersion(1, 0);
    case SpecificationVersion::V_1_1_DRAFT:
      return TosaSpecificationVersion(1, 1);
    }
    llvm_unreachable("Unknown TOSA version");
  }
};

llvm::SmallString<4> stringifyVersion(TosaSpecificationVersion version);

/// This class represents the capability enabled in the target implementation
/// such as profile, extension, and level. It's a wrapper class around
/// tosa::TargetEnvAttr.
class TargetEnv {
public:
  TargetEnv() {}
  explicit TargetEnv(SpecificationVersion specificationVersion, Level level,
                     const ArrayRef<Profile> &profiles,
                     const ArrayRef<Extension> &extensions)
      : specificationVersion(specificationVersion), level(level) {
    enabledProfiles.insert_range(profiles);
    enabledExtensions.insert_range(extensions);
  }

  explicit TargetEnv(TargetEnvAttr targetAttr)
      : TargetEnv(targetAttr.getSpecificationVersion(), targetAttr.getLevel(),
                  targetAttr.getProfiles(), targetAttr.getExtensions()) {}

  void addProfile(Profile p) { enabledProfiles.insert(p); }
  void addExtension(Extension e) { enabledExtensions.insert(e); }

  SpecificationVersion getSpecVersion() const { return specificationVersion; }

  TosaLevel getLevel() const {
    if (level == Level::eightK)
      return TOSA_LEVEL_EIGHTK;
    else if (level == Level::none)
      return TOSA_LEVEL_NONE;
    else
      llvm_unreachable("Unknown TOSA level");
  };

  // Returns true if the given profile is allowed.
  bool allows(Profile prof) const { return enabledProfiles.count(prof) != 0; }

  bool allowsAnyOf(ArrayRef<Profile> profs) const {
    return llvm::any_of(profs, [&](Profile prof) { return allows(prof); });
  }

  bool allowsAllOf(ArrayRef<Profile> profs) const {
    return llvm::all_of(profs, [&](Profile prof) { return allows(prof); });
  }

  // Returns true if the given extension is allowed.
  bool allows(Extension ext) const { return enabledExtensions.count(ext) != 0; }

  bool allowsAnyOf(ArrayRef<Extension> exts) const {
    return llvm::any_of(exts, [&](Extension ext) { return allows(ext); });
  }

  bool allowsAllOf(ArrayRef<Extension> exts) const {
    return llvm::all_of(exts, [&](Extension ext) { return allows(ext); });
  }

private:
  SpecificationVersion specificationVersion;
  Level level;
  llvm::SmallSet<Profile, 3> enabledProfiles;
  llvm::SmallSet<Extension, 13> enabledExtensions;
};

} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_IR_TARGETENV_H
