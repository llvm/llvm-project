//==- include/llvm/CodeGen/AssignSectionsToGlobals.h -------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef LLVM_CODEGEN_ASSIGNSECTIONSTOGLOBALS_H
#define LLVM_CODEGEN_ASSIGNSECTIONSTOGLOBALS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class AssignSectionsToGlobalsPass
    : public PassInfoMixin<AssignSectionsToGlobalsPass> {
  TargetMachine *TM;

public:
  AssignSectionsToGlobalsPass(TargetMachine *tm) : TM(tm) {}
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};
} // end namespace llvm

#endif // LLVM_CODEGEN_ASSIGNSECTIONSTOGLOBALS_H
