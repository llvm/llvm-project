//===- DRFScopedNoAlias.h - DRF-based scoped-noalias metadata ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
