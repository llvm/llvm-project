//===--- NVVMTraits.h - NVVM Traits -----------------------------*- C++ -*-===//
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

#ifndef NVVM_DIALECT_NVVM_IR_NVVMTRAITS_H_
#define NVVM_DIALECT_NVVM_IR_NVVMTRAITS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {

namespace NVVM {

// Structure to store and check compatibility of SM versions.
struct NVVMCheckSMVersion {
  int archVersion;

  // Set to true if the SM version is accelerated (e.g., sm_90a).
  bool archAccelerated;

  // Set to true if the target SM version must match exactly
  // (both archVersion and archAccelerated).
  // For example, sm_90a with exactMatch = false will also match
  // sm_100a, sm_120a, etc.
  bool exactMatch;

  NVVMCheckSMVersion()
      : archVersion(0), archAccelerated(false), exactMatch(false) {}
  NVVMCheckSMVersion(StringRef smVersion, bool exactMatch = false)
      : exactMatch(exactMatch) {
    parse(smVersion);
  }
  NVVMCheckSMVersion(int archVersion, bool archAccelerated, bool exactMatch)
      : archVersion(archVersion), archAccelerated(archAccelerated),
        exactMatch(exactMatch) {}

  // Parses the SM version string and sets the archVersion (as an integer)
  // and the archAccelerated flag.
  void parse(StringRef smVersion) {
    archAccelerated = (smVersion.back() == 'a');
    smVersion.drop_front(3)
        .take_while([](char c) { return llvm::isDigit(c); })
        .getAsInteger(10, archVersion);
  }

  bool isCompatible(const NVVMCheckSMVersion &targetSM) const {
    if (exactMatch)
      return (*this) == targetSM;

    return archAccelerated
               ? archVersion <= targetSM.archVersion && targetSM.archAccelerated
               : archVersion <= targetSM.archVersion;
  }

  bool operator==(const NVVMCheckSMVersion &other) const {
    return archVersion == other.archVersion &&
           archAccelerated == other.archAccelerated;
  }
};
} // namespace NVVM
} // namespace mlir

#include "mlir/Dialect/LLVMIR/NVVMTraits.h.inc"

namespace mlir {

namespace OpTrait {

template <int MinVersion, bool ArchAccelerated = false, bool ExactMatch = false>
class NVVMRequiresSM {
public:
  template <typename ConcreteOp>
  class Impl
      : public OpTrait::TraitBase<
            ConcreteOp,
            NVVMRequiresSM<MinVersion, ArchAccelerated, ExactMatch>::Impl>,
        public mlir::NVVM::RequiresSMInterface::Trait<ConcreteOp> {
  public:
    const NVVM::NVVMCheckSMVersion getRequiredMinSMVersion() const {
      return NVVM::NVVMCheckSMVersion(MinVersion, ArchAccelerated, ExactMatch);
    }
  };
};
} // namespace OpTrait
} // namespace mlir
#endif // NVVM_DIALECT_NVVM_IR_NVVMTRAITS_H_
