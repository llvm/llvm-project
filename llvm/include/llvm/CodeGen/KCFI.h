//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the MachineKCFI class, which is a
/// Machine Pass that implements kernel control flow integrity.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_KCFI_H
#define LLVM_CODEGEN_KCFI_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class MachineKCFIPass : public PassInfoMixin<MachineKCFIPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_CODEGEN_KCFI_H
