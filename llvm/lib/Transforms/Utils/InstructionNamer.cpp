//===- InstructionNamer.cpp - Give anonymous instructions names -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a little utility pass that gives instructions names, this is mostly
// useful when diffing the effect of an optimization because deleting an
// unnamed instruction can change all other instruction numbering, making the
// diff very noisy.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/InstructionNamer.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include <cstdint>
#include <string>

using namespace llvm;

static void nameInstructions(Function &F, uint64_t *NextID = nullptr) {
  auto getName = [NextID](StringRef Prefix) -> std::string {
    if (!NextID)
      return Prefix.str();
    return (Twine(Prefix) + "." + Twine((*NextID)++)).str();
  };

  for (Argument &Arg : F.args()) {
    if (!Arg.hasName())
      Arg.setName(getName("arg"));
  }

  for (BasicBlock &BB : F) {
    if (!BB.hasName())
      BB.setName(getName("bb"));

    for (Instruction &I : BB) {
      if (!I.hasName() && !I.getType()->isVoidTy())
        I.setName(getName("i"));
    }
  }
}

PreservedAnalyses InstructionNamerPass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  nameInstructions(F);
  return PreservedAnalyses::all();
}

static void nameIRUnit(IRUnitRef IR, uint64_t &NextID) {
  if (const auto *M = dyn_cast<Module>(IR)) {
    for (Function &F : *const_cast<Module *>(M))
      nameInstructions(F, &NextID);
  } else if (const auto *F = dyn_cast<Function>(IR)) {
    nameInstructions(*const_cast<Function *>(F), &NextID);
  } else if (const auto *C = dyn_cast<LazyCallGraph::SCC>(IR)) {
    for (const LazyCallGraph::Node &N : *C)
      nameInstructions(N.getFunction(), &NextID);
  } else if (const auto *L = dyn_cast<Loop>(IR)) {
    nameInstructions(*L->getHeader()->getParent(), &NextID);
  }
}

void InstructionNamerPass::registerCallbacks(PassInstrumentationCallbacks &PIC,
                                             uint64_t &NextID) {
  // The symbol table only detects collisions with live values. Keep an ID
  // across callbacks so deleting a value does not reuse its generated ID.
  PIC.registerBeforeNonSkippedPassCallback(
      [&NextID](StringRef, IRUnitRef IR) { nameIRUnit(IR, NextID); });
  PIC.registerAfterPassCallback(
      [&NextID](StringRef, IRUnitRef IR, const PreservedAnalyses &) {
        nameIRUnit(IR, NextID);
      });
}
