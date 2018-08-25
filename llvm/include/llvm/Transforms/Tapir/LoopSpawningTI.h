//===---- LoopSpawning.h - Spawn loop iterations efficiently ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass modifies Tapir loops to spawn their iterations efficiently.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H
#define LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H

#include "llvm/IR/PassManager.h"

namespace llvm {
/// The LoopSpawning Pass.
struct LoopSpawningPass : public PassInfoMixin<LoopSpawningPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
}

#endif // LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H
