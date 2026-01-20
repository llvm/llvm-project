//===- LowFatSanitizer.cpp - LowFat Pointer Bounds Checking ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/LowFatSanitizer.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "lowfat"

LowFatSanitizerPass::LowFatSanitizerPass(const LowFatSanitizerOptions &Options) {}

PreservedAnalyses LowFatSanitizerPass::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  LLVM_DEBUG(dbgs() << "[LowFat] Running on module: " << M.getName() << "\n");

  return PreservedAnalyses::all();
}
