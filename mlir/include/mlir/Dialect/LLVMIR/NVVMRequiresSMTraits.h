//===--- NVVMRequiresSMTraits.h - NVVM Requires SM Traits -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op traits for the NVVM Dialect in MLIR
//
//===----------------------------------------------------------------------===//

#ifndef NVVM_DIALECT_NVVM_IR_NVVMREQUIRESSMTRAITS_H_
#define NVVM_DIALECT_NVVM_IR_NVVMREQUIRESSMTRAITS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {

namespace NVVM {

// Struct to store and check compatibility of SM versions.
struct NVVMCheckSMVersion {
  static constexpr char kArchAcceleratedSuffix = 'a';
  static constexpr char kFamilySpecificSuffix = 'f';

  // List of supported full SM versions.
  // This is used to check compatibility with a target SM version.
  // The full SM version is encoded as SM * 10 + ArchSuffixOffset where:
  // - SM is the SM version (e.g., 100)
  // - ArchSuffixOffset is 0 for base, 2 for family-specific, and 3 for
  //   architecture-accelerated
  //
  // For example, sm_100 is encoded as 1000 (100 * 10 + 0), sm_100f is encoded
  // as 1002 (100 * 10 + 2) and sm_100a is encoded as 1003 (100 * 10 + 3).
  llvm::SmallVector<unsigned> fullSmVersionList;

  template <typename... Versions>
  NVVMCheckSMVersion(Versions... fullSmVersions)
      : fullSmVersionList({fullSmVersions...}) {}

  bool isCompatibleWith(const unsigned &targetFullSmVersion) const {
    return llvm::any_of(
        fullSmVersionList, [&](const unsigned &requiredFullSmVersion) {
          if (hasArchAcceleratedFeatures(requiredFullSmVersion))
            return hasArchAcceleratedFeatures(targetFullSmVersion) &&
                   (getSMVersion(targetFullSmVersion) ==
                    getSMVersion(requiredFullSmVersion));

          if (hasFamilySpecificFeatures(requiredFullSmVersion))
            return hasFamilySpecificFeatures(targetFullSmVersion) &&
                   (getSMFamily(targetFullSmVersion) ==
                    getSMFamily(requiredFullSmVersion)) &&
                   (getSMVersion(targetFullSmVersion) >=
                    getSMVersion(requiredFullSmVersion));

          return targetFullSmVersion >= requiredFullSmVersion;
        });
  }

  // Parses an SM version string and returns an equivalent full SM version
  // integer.
  static unsigned getTargetFullSmVersionFromStr(StringRef smVersionString) {
    bool isAA = smVersionString.back() == kArchAcceleratedSuffix;
    bool isFS = smVersionString.back() == kFamilySpecificSuffix;

    unsigned smVersion;
    smVersionString.drop_front(3)
        .take_while([](char c) { return llvm::isDigit(c); })
        .getAsInteger(10, smVersion);

    return smVersion * 10 + (isAA ? 3 : 0) + (isFS ? 2 : 0);
  }

  static bool isMinimumSMVersion(unsigned fullSmVersion) {
    return getSMVersion(fullSmVersion) >= 20;
  }

private:
  static bool hasFamilySpecificFeatures(unsigned fullSmVersion) {
    return (fullSmVersion % 10) >= 2;
  }

  static bool hasArchAcceleratedFeatures(unsigned fullSmVersion) {
    return (fullSmVersion % 10) == 3;
  }

  static unsigned getSMVersion(unsigned fullSmVersion) {
    return fullSmVersion / 10;
  }

  static unsigned getSMFamily(unsigned fullSmVersion) {
    return fullSmVersion / 100;
  }
};

} // namespace NVVM
} // namespace mlir

#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h.inc"

namespace mlir {

namespace OpTrait {

template <unsigned... FullSMVersions>
class NVVMRequiresSM {
public:
  template <typename ConcreteOp>
  class Impl
      : public OpTrait::TraitBase<ConcreteOp,
                                  NVVMRequiresSM<FullSMVersions...>::Impl>,
        public mlir::NVVM::RequiresSMInterface::Trait<ConcreteOp> {
  public:
    NVVM::NVVMCheckSMVersion getRequiredMinSMVersion() const {
      return NVVM::NVVMCheckSMVersion(FullSMVersions...);
    }
  };
};
} // namespace OpTrait
} // namespace mlir
#endif // NVVM_DIALECT_NVVM_IR_NVVMREQUIRESSMTRAITS_H_
