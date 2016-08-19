//===-- NameAnonFunctions.h - Anonymous Function Naming Pass ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements naming anonymous function to make sure they can be
// referred to by ThinLTO.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_NAMEANONFUNCTIONS_H
#define LLVM_TRANSFORMS_UTILS_NAMEANONFUNCTIONS_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Simple pass that provides a name to every anonymous function.
class NameAnonFunctionPass : public PassInfoMixin<NameAnonFunctionPass> {
public:
  NameAnonFunctionPass() {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
}

#endif // LLVM_TRANSFORMS_UTILS_NAMEANONFUNCTIONS_H
