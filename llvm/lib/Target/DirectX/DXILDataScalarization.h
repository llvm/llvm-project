//===- DXILDataScalarization.h - Prepare LLVM Module for DXIL Data
//Legalization----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------------===//
#ifndef LLVM_TARGET_DIRECTX_DXILDATASCALARIZATION_H
#define LLVM_TARGET_DIRECTX_DXILDATASCALARIZATION_H

#include "DXILResource.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

/// A pass thattransforms Vectors to Arrays
class DXILDataScalarization : public PassInfoMixin<DXILDataScalarization> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

class DXILDataScalarizationLegacy : public ModulePass {

public:
  bool runOnModule(Module &M) override;
  DXILDataScalarizationLegacy() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  static char ID; // Pass identification.
};
} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILDATASCALARIZATION_H
