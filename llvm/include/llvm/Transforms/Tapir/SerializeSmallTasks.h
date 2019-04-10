//===- SerializeSmallTasks.h - Serialize small Tapir tasks ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_SERIALIZESMALLTASKS_H_
#define LLVM_TRANSFORMS_TAPIR_SERIALIZESMALLTASKS_H_

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

/// Pass to serialize small Tapir tasks, whose work is too little to overcome
/// the overhead of a spawn.
class SerializeSmallTasksPass : public PassInfoMixin<SerializeSmallTasksPass> {
public:
  explicit SerializeSmallTasksPass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_SERIALIZESMALLTASKS_H_
