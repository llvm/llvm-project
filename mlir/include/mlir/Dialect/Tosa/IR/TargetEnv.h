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

/// This class represents the capability enabled in the target implementation
/// such as profile, extension, and level.
class TargetEnv {
public:
  TargetEnv() {}
  explicit TargetEnv(const SmallVectorImpl<Profile> &profiles,
                     const SmallVectorImpl<Extension> &extensions) {
    enabledProfiles.insert_range(profiles);

    enabledExtensions.insert_range(extensions);
  }

  void addProfile(Profile p) { enabledProfiles.insert(p); }
  void addExtension(Extension e) { enabledExtensions.insert(e); }

  // TODO implement the following utilities.
  // Version getSpecVersion() const;
  // TosaLevel getLevel() const;

  // Returns true if the given profile is allowed.
  bool allows(Profile prof) const { return enabledProfiles.count(prof) != 0; }

  bool allowsAnyOf(ArrayRef<Profile> profs) const {
    const auto *chosen = llvm::find_if(
        profs, [this](tosa::Profile prof) { return allows(prof); });
    return chosen != profs.end() ? true : false;
  }

  bool allowsAllOf(ArrayRef<Profile> profs) const {
    bool is_allowed = true;
    llvm::for_each(profs,
                   [&](tosa::Profile prof) { is_allowed &= allows(prof); });
    return is_allowed;
  }

  // Returns true if the given extension is allowed.
  bool allows(Extension ext) const { return enabledExtensions.count(ext) != 0; }

  bool allowsAnyOf(ArrayRef<Extension> exts) const {
    const auto *chosen = llvm::find_if(
        exts, [this](tosa::Extension ext) { return allows(ext); });
    return chosen != exts.end() ? true : false;
  }

  bool allowsAllOf(ArrayRef<Extension> exts) const {
    bool is_allowed = true;
    llvm::for_each(exts,
                   [&](tosa::Extension ext) { is_allowed &= allows(ext); });
    return is_allowed;
  }

private:
  llvm::SmallSet<Profile, 3> enabledProfiles;
  llvm::SmallSet<Extension, 8> enabledExtensions;
};

} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_IR_TARGETENV_H
