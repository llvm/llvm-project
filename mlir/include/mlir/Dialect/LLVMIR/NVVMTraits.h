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
#include "llvm/ADT/StringExtras.h"

namespace mlir {

namespace NVVM {

struct NVVMCheckSMVersion {
  int ArchVersion;
  bool ArchAccelerated;
  std::string ArchString;

  NVVMCheckSMVersion() {}
  NVVMCheckSMVersion(StringRef SMVersion) : ArchString(SMVersion) {
    parse(SMVersion);
  }
  NVVMCheckSMVersion(int ArchVersion, bool ArchAccelerated)
      : ArchVersion(ArchVersion), ArchAccelerated(ArchAccelerated) {
    ArchString = (llvm::Twine("sm_") + llvm::Twine(ArchVersion) +
                  (ArchAccelerated ? "a" : "\0"))
                     .str();
  }

  const StringRef getArchString() const { return ArchString; }

  void parse(StringRef SMVersion) {
    ArchAccelerated = (SMVersion.back() == 'a');
    SMVersion.drop_front(3)
        .take_while([](char c) { return llvm::isDigit(c); })
        .getAsInteger(10, ArchVersion);
  }

  bool isCompatible(const NVVMCheckSMVersion &TargetSM) const {
    // for arch-conditional SMs, they should exactly match to be valid
    if (ArchAccelerated || TargetSM.ArchAccelerated)
      return (*this) == TargetSM;

    return ArchVersion <= TargetSM.ArchVersion;
  }

  bool operator==(const NVVMCheckSMVersion &Other) const {
    return ArchVersion == Other.ArchVersion &&
           ArchAccelerated == Other.ArchAccelerated;
  }
};

llvm::SmallVector<NVVMCheckSMVersion> getTargetSMVersions(Operation *op);

LogicalResult
verifyOpSMRequirements(Operation *op,
                       llvm::SmallVector<NVVMCheckSMVersion> TargetSMVersions,
                       NVVMCheckSMVersion RequiredSMVersion);
} // namespace NVVM

namespace OpTrait {

template <int Version, bool ArchAccelerated = false>
class NVVMRequiresSM {
public:
  template <typename ConcreteOp>
  class Impl : public OpTrait::TraitBase<
                   ConcreteOp, NVVMRequiresSM<Version, ArchAccelerated>::Impl> {
  public:
    static LogicalResult verifyTrait(Operation *op) {
      NVVM::NVVMCheckSMVersion RequiredSMVersion(Version, ArchAccelerated);
      llvm::SmallVector<NVVM::NVVMCheckSMVersion> TargetSMVersions =
          NVVM::getTargetSMVersions(op);

      return NVVM::verifyOpSMRequirements(op, TargetSMVersions,
                                          RequiredSMVersion);
    }
  };
};
} // namespace OpTrait
} // namespace mlir
#endif // NVVM_DIALECT_NVVM_IR_NVVMTRAITS_H_
