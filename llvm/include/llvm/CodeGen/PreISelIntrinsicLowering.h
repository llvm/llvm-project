//===--- PreISelIntrinsicLowering.h - Pre-ISel intrinsic lowering pass ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements IR lowering for the llvm.load.relative intrinsic.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CODEGEN_PREISELINTRINSICLOWERING_H
#define LLVM_CODEGEN_PREISELINTRINSICLOWERING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

struct PreISelIntrinsicLoweringPass
    : PassInfoMixin<PreISelIntrinsicLoweringPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
}

#endif // LLVM_CODEGEN_PREISELINTRINSICLOWERING_H
