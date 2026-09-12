//===-- llvm/CodeGen/ResetMachineFunctionPass.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_RESETMACHINEFUNCTIONPASS_H
#define LLVM_CODEGEN_RESETMACHINEFUNCTIONPASS_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class ResetMachineFunctionPass
    : public RequiredPassInfoMixin<ResetMachineFunctionPass> {
  /// Tells whether or not this pass should emit a fallback
  /// diagnostic when it resets a function.
  bool EmitFallbackDiag;
  /// Whether we should abort immediately instead of resetting the function.
  bool AbortOnFailedISel;

public:
  ResetMachineFunctionPass(bool EmitFallbackDiag = false,
                           bool AbortOnFailedISel = false)
      : EmitFallbackDiag(EmitFallbackDiag),
        AbortOnFailedISel(AbortOnFailedISel) {}

  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_CODEGEN_RESETMACHINEFUNCTIONPASS_H
