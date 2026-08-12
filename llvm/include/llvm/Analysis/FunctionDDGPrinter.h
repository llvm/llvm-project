//===- llvm/Analysis/FunctionDDGPrinter.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a function-scoped DOT printer for the Data Dependence
// Graph, the function-scope counterpart to the loop-scoped
// llvm::DDGDotPrinterPass declared in DDGPrinter.h.
//
// Motivation: the loop-scoped DDG printer only covers code that has been
// recognised as a loop nest. For visualising the dependence structure of
// straight-line or cross-loop code -- for example SLP-style vector code or
// fully unrolled kernels -- a function-wide view is required. This printer
// builds a DataDependenceGraph for the whole function and renders it with the
// existing DDG DOTGraphTraits, so loop- and function-scope graphs look alike.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_FUNCTIONDDGPRINTER_H
#define LLVM_ANALYSIS_FUNCTIONDDGPRINTER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Function;

/// DOT printer pass for function-scoped data dependences.
///
/// Counterpart to DDGDotPrinterPass at function (rather than loop) scope. It
/// builds a DataDependenceGraph for the whole function and writes it to a
/// single DOT file named "<prefix>.<function-name>.dot", reusing the shared
/// DDG DOTGraphTraits; -dot-function-ddg-only selects the simplified rendering.
class FunctionDDGDotPrinterPass
    : public PassInfoMixin<FunctionDDGDotPrinterPass> {
public:
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_FUNCTIONDDGPRINTER_H
