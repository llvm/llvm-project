//===- DRFScopedNoAlias.h - DRF-based scoped-noalias metadata ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Adds scoped-noalias metadata to memory accesses based on Tapir's parallel
// control flow constructs and the assumption that the function is data-race
// free.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_DRFSCOPEDNOALIASPASS_H
#define LLVM_TRANSFORMS_TAPIR_DRFSCOPEDNOALIASPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// The DRF-Based Scoped-Noalias Pass.
struct DRFScopedNoAliasPass : public PassInfoMixin<DRFScopedNoAliasPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

}

#endif
