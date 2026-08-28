//===- llvm/CodeGen/GlobalISel/IRTranslator.h - IRTranslator ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares the IRTranslator pass.
/// This pass is responsible for translating LLVM IR into MachineInstr.
/// It uses target hooks to lower the ABI but aside from that, the pass
/// generated code is generic. This is the default translator used for
/// GlobalISel.
///
/// \todo Replace the comments with actual doxygen comments.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_IRTRANSLATOR_H
#define LLVM_CODEGEN_GLOBALISEL_IRTRANSLATOR_H

#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CodeGen.h"
#include <memory>

namespace llvm {

class IRTranslatorImpl;

// Technically the pass should run on an hypothetical MachineModule,
// since it should translate Global into some sort of MachineGlobal.
// The MachineGlobal should ultimately just be a transfer of ownership of
// the interesting bits that are relevant to represent a global value.
// That being said, we could investigate what would it cost to just duplicate
// the information from the LLVM IR.
// The idea is that ultimately we would be able to free up the memory used
// by the LLVM IR as soon as the translation is over.
class LLVM_ABI IRTranslatorLegacy : public MachineFunctionPass {
public:
  static char ID;
  IRTranslatorLegacy(CodeGenOptLevel OptLevel = CodeGenOptLevel::None);
  ~IRTranslatorLegacy() override;

  StringRef getPassName() const override { return "IRTranslator"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  CodeGenOptLevel OptLevel;
  std::unique_ptr<IRTranslatorImpl> Impl;
};

class IRTranslatorPass : public RequiredPassInfoMixin<IRTranslatorPass> {
  std::unique_ptr<IRTranslatorImpl> Impl;

public:
  IRTranslatorPass(CodeGenOptLevel OptLevel);
  ~IRTranslatorPass();
  IRTranslatorPass(IRTranslatorPass &&);

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_IRTRANSLATOR_H
