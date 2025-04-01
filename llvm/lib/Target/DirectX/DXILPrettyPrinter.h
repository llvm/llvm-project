//===- DXILPrettyPrinter.h - Print resources for textual DXIL ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file This file contains a pass for pretty printing DXIL metadata into IR
// comments when printing assembly output.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_DIRECTX_DXILPRETTYPRINTER_H
#define LLVM_TARGET_DIRECTX_DXILPRETTYPRINTER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// A pass that prints resources in a format suitable for textual DXIL.
class DXILPrettyPrinterPass : public PassInfoMixin<DXILPrettyPrinterPass> {
  raw_ostream &OS;

public:
  explicit DXILPrettyPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILPRETTYPRINTER_H
