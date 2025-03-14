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

struct NVVMCheckSMVersion {
  int archVersion;
  bool archAccelerated;

  NVVMCheckSMVersion() {}
  NVVMCheckSMVersion(StringRef smVersion) { parse(smVersion); }
  NVVMCheckSMVersion(int archVersion, bool archAccelerated)
      : archVersion(archVersion), archAccelerated(archAccelerated) {}

  // Parses the SM version string and sets the archVersion (integer) and
  // the archAccelerated flag.
  void parse(StringRef smVersion) {
    archAccelerated = (smVersion.back() == 'a');
    smVersion.drop_front(3)
        .take_while([](char c) { return llvm::isDigit(c); })
        .getAsInteger(10, archVersion);
  }

  bool isCompatible(const NVVMCheckSMVersion &targetSM) const {
    // for arch-conditional SMs, they should exactly match to be valid
    if (archAccelerated || targetSM.archAccelerated)
      return (*this) == targetSM;

    return archVersion <= targetSM.archVersion;
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

template <int Version, bool ArchAccelerated = false>
class NVVMRequiresSM {
public:
  template <typename ConcreteOp>
  class Impl : public OpTrait::TraitBase<
                   ConcreteOp, NVVMRequiresSM<Version, ArchAccelerated>::Impl>,
               public mlir::NVVM::RequiresSMInterface::Trait<ConcreteOp> {
  public:
    const NVVM::NVVMCheckSMVersion getRequiredMinSMVersion() const {
      return NVVM::NVVMCheckSMVersion(Version, ArchAccelerated);
    }
  };
};
} // namespace OpTrait
} // namespace mlir
#endif // NVVM_DIALECT_NVVM_IR_NVVMTRAITS_H_
