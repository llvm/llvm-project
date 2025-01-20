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
  std::string archString;

  NVVMCheckSMVersion() {}
  NVVMCheckSMVersion(StringRef SMVersion) : archString(SMVersion) {
    parse(SMVersion);
  }
  NVVMCheckSMVersion(int archVersion, bool archAccelerated)
      : archVersion(archVersion), archAccelerated(archAccelerated) {
    archString = (llvm::Twine("sm_") + llvm::Twine(archVersion) +
                  (archAccelerated ? "a" : "\0"))
                     .str();
  }

  const StringRef getArchString() const { return archString; }

  // Parses the SM version string and sets the archVersion (integer) and
  // the archAccelerated flag.
  void parse(StringRef SMVersion) {
    archAccelerated = (SMVersion.back() == 'a');
    SMVersion.drop_front(3)
        .take_while([](char c) { return llvm::isDigit(c); })
        .getAsInteger(10, archVersion);
  }

  bool isCompatible(const NVVMCheckSMVersion &TargetSM) const {
    // for arch-conditional SMs, they should exactly match to be valid
    if (archAccelerated || TargetSM.archAccelerated)
      return (*this) == TargetSM;

    return archVersion <= TargetSM.archVersion;
  }

  bool operator==(const NVVMCheckSMVersion &Other) const {
    return archVersion == Other.archVersion &&
           archAccelerated == Other.archAccelerated;
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
