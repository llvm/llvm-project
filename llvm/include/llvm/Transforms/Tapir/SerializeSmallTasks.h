//===- SerializeSmallTasks.h - Serialize small Tapir tasks ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
