//===--- AArch64MachineModuleInfo.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AArch64 Machine Module Info.
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64MACHINEMODULEINFO_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64MACHINEMODULEINFO_H

#include "llvm/CodeGen/MachineModuleInfoImpls.h"

namespace llvm {

class AArch64MachineModuleInfo final : public MachineModuleInfoELF {
  /// HasSignedPersonality is true if the corresponding IR module has the
  /// "ptrauth-sign-personality" flag set to 1.
  bool HasSignedPersonality = false;

public:
  AArch64MachineModuleInfo(const MachineModuleInfo &);

  bool hasSignedPersonality() const { return HasSignedPersonality; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_AARCH64MACHINEMODULEINFO_H
