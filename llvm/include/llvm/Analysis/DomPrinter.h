//===-- DomPrinter.h - Dom printer external interface ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines external functions that can be called to explicitly
// instantiate the dominance tree printer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMPRINTER_H
#define LLVM_ANALYSIS_DOMPRINTER_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class DomTreePrinterPass : public PassInfoMixin<DomTreePrinterPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class DomTreeOnlyPrinterPass : public PassInfoMixin<DomTreeOnlyPrinterPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
} // namespace llvm

namespace llvm {
  class FunctionPass;
  FunctionPass *createDomPrinterPass();
  FunctionPass *createDomOnlyPrinterPass();
  FunctionPass *createDomViewerPass();
  FunctionPass *createDomOnlyViewerPass();
  FunctionPass *createPostDomPrinterPass();
  FunctionPass *createPostDomOnlyPrinterPass();
  FunctionPass *createPostDomViewerPass();
  FunctionPass *createPostDomOnlyViewerPass();
} // End llvm namespace

#endif
