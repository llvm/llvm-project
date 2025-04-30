//===- NextSiliconImportRecursion.cpp - NS recursive-import inference pass ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/NextSiliconImportRecursion.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "next-silicon-import-recursion"

/// Searches for any ns location pragma in the function and returns true if it
/// found any. Returns false otherwise.
static bool hasLocationPragma(Function &F, LoopInfo &LI) {
  if (F.hasFnAttribute("ns-location"))
    return true;
  return llvm::any_of(LI.getLoopsInPreorder(), [](Loop *L) {
    return findStringMetadataForLoop(L, "ns.loop.location").has_value();
  });
}

PreservedAnalyses
NextSiliconImportRecursionPass::run(Function &F, FunctionAnalysisManager &FAM) {
  auto &LI = FAM.getResult<LoopAnalysis>(F);
  if (!hasLocationPragma(F, LI))
    return PreservedAnalyses::all();

  // Assign the the max integer value, as a location pragma is recursive.
  F.addFnAttr("ns-import-recursion");
  return PreservedAnalyses::none();
}
