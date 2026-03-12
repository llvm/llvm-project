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
  struct SMVersion {
    unsigned version;
    // Set to true if the SM version is accelerated (e.g., sm_90a).
    bool archAccelerated;
    // Set to true if the SM version is family-specific (e.g., sm_100f).
    bool familySpecific;

    unsigned getSmFamilyVersion() const { return version / 10; }

    bool hasFamilySpecificFeatures() const {
      return familySpecific || archAccelerated;
    }
  };

  // List of SM versions.
  // Typically only has one version except for cases where multiple
  // arch-accelerated versions are supported.
  // For example, tcgen05.shift is supported on sm_100a, sm_101a, and sm_103a.
  llvm::SmallVector<SMVersion, 1> smVersionList;

  template <typename... Versions>
  NVVMCheckSMVersion(bool archAccelerated, bool familySpecific,
                     Versions... smVersions)
      : smVersionList({SMVersion{static_cast<unsigned>(smVersions),
                                 archAccelerated, familySpecific}...}) {
    assert(
        !(archAccelerated && familySpecific) &&
        "archAccelerated and familySpecific cannot be true at the same time!");
  }

  bool isCompatibleWith(const NVVMCheckSMVersion &targetSM) const {
    assert(targetSM.smVersionList.size() == 1 &&
           "target SM version list must be a single version!");

    SMVersion targetSMVersion = targetSM.smVersionList[0];

    return llvm::any_of(smVersionList, [&](const SMVersion &RequiredSMVersion) {
      if (RequiredSMVersion.archAccelerated) {
        return targetSMVersion.archAccelerated &&
               (RequiredSMVersion.version == targetSMVersion.version);
      } else if (RequiredSMVersion.familySpecific) {
        return targetSMVersion.hasFamilySpecificFeatures() &&
               (RequiredSMVersion.getSmFamilyVersion() ==
                targetSMVersion.getSmFamilyVersion()) &&
               (targetSMVersion.version >= RequiredSMVersion.version);
      } else {
        return targetSMVersion.version >= RequiredSMVersion.version;
      }
    });
  }

  bool isMinimumSMVersion() const { return smVersionList[0].version >= 20; }

  // Parses an SM version string and returns an equivalent NVVMCheckSMVersion
  // object.
  static NVVMCheckSMVersion
  getTargetSMVersionFromStr(StringRef smVersionString) {
    bool isAA = smVersionString.back() == 'a';
    bool isFS = smVersionString.back() == 'f';

    int smVersionInt;
    smVersionString.drop_front(3)
        .take_while([](char c) { return llvm::isDigit(c); })
        .getAsInteger(10, smVersionInt);

    return NVVMCheckSMVersion(isAA, isFS, smVersionInt);
  }

  NVVMCheckSMVersion &append(const NVVMCheckSMVersion &other) {
    smVersionList.append(other.smVersionList);
    return *this;
  }
};

} // namespace NVVM
} // namespace mlir

#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h.inc"

namespace mlir {

namespace OpTrait {

template <int MinVersion>
class NVVMRequiresSM {
public:
  template <typename ConcreteOp>
  class Impl
      : public OpTrait::TraitBase<ConcreteOp, NVVMRequiresSM<MinVersion>::Impl>,
        public mlir::NVVM::RequiresSMInterface::Trait<ConcreteOp> {
  public:
    NVVM::NVVMCheckSMVersion getRequiredMinSMVersion() const {
      return NVVM::NVVMCheckSMVersion(false, false, MinVersion);
    }
  };
};

template <int... SMVersions>
class NVVMRequiresSMa {
public:
  template <typename ConcreteOp>
  class Impl : public OpTrait::TraitBase<ConcreteOp,
                                         NVVMRequiresSMa<SMVersions...>::Impl>,
               public mlir::NVVM::RequiresSMInterface::Trait<ConcreteOp> {
  public:
    NVVM::NVVMCheckSMVersion getRequiredMinSMVersion() const {
      return NVVM::NVVMCheckSMVersion(true, false, SMVersions...);
    }
  };

  static NVVM::NVVMCheckSMVersion getRequiredMinSMVersion() {
    return NVVM::NVVMCheckSMVersion(true, false, SMVersions...);
  }
};

template <int... SMVersions>
class NVVMRequiresSMf {
public:
  template <typename ConcreteOp>
  class Impl : public OpTrait::TraitBase<ConcreteOp,
                                         NVVMRequiresSMf<SMVersions...>::Impl>,
               public mlir::NVVM::RequiresSMInterface::Trait<ConcreteOp> {
  public:
    NVVM::NVVMCheckSMVersion getRequiredMinSMVersion() const {
      return NVVM::NVVMCheckSMVersion(false, true, SMVersions...);
    }
  };

  static NVVM::NVVMCheckSMVersion getRequiredMinSMVersion() {
    return NVVM::NVVMCheckSMVersion(false, true, SMVersions...);
  }
};

template <typename T, typename U>
class NVVMRequiresSMaOrf {
public:
  template <typename ConcreteOp>
  class Impl
      : public OpTrait::TraitBase<ConcreteOp, NVVMRequiresSMaOrf<T, U>::Impl>,
        public mlir::NVVM::RequiresSMInterface::Trait<ConcreteOp> {
  public:
    NVVM::NVVMCheckSMVersion getRequiredMinSMVersion() const {
      auto result = T::getRequiredMinSMVersion();
      result.append(U::getRequiredMinSMVersion());
      return result;
    }
  };
};
} // namespace OpTrait
} // namespace mlir
#endif // NVVM_DIALECT_NVVM_IR_NVVMREQUIRESSMTRAITS_H_
