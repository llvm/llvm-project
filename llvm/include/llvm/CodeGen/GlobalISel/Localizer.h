//== llvm/CodeGen/GlobalISel/Localizer.h - Localizer -------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file describes the interface of the Localizer pass.
/// This pass moves/duplicates constant-like instructions close to their uses.
/// Its primarily goal is to workaround the deficiencies of the fast register
/// allocator.
/// With GlobalISel constants are all materialized in the entry block of
/// a function. However, the fast allocator cannot rematerialize constants and
/// has a lot more live-ranges to deal with and will most likely end up
/// spilling a lot.
/// By pushing the constants close to their use, we only create small
/// live-ranges.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_LOCALIZER_H
#define LLVM_CODEGEN_GLOBALISEL_LOCALIZER_H

#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
// Forward declarations.
class AnalysisUsage;

/// This pass implements the localization mechanism described at the
/// top of this file. One specificity of the implementation is that
/// it will materialize one and only one instance of a constant per
/// basic block, thus enabling reuse of that constant within that block.
/// Moreover, it only materializes constants in blocks where they
/// are used. PHI uses are considered happening at the end of the
/// related predecessor.
class LLVM_ABI LocalizerLegacy : public MachineFunctionPass {
public:
  static char ID;

  LocalizerLegacy();

  StringRef getPassName() const override { return "Localizer"; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;
};

class LocalizerPass : public RequiredPassInfoMixin<LocalizerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }
};

} // End namespace llvm.

#endif
