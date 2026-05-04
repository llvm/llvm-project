//===-- EJitAotModulePass.cpp - EmbeddedJIT AOT Coordinator --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  PASS5: Late-stage AOT pipeline coordinator. Runs PASS2→PASS3→PASS4
//  in order (PASS1 is an independent early pass), manages
//  PreservedAnalyses composition, and performs diagnostic consistency
//  checks on ejit metadata.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/EmbeddedJIT/EJitPasses.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::ejit;

namespace {

static bool hasMDStringEntry(const MDNode *Node, StringRef Name) {
  if (!Node)
    return false;
  for (const MDOperand &Op : Node->operands()) {
    auto *Sub = dyn_cast<MDNode>(Op.get());
    if (!Sub || Sub->getNumOperands() == 0)
      continue;
    if (auto *S = dyn_cast<MDString>(Sub->getOperand(0)))
      if (S->getString() == Name)
        return true;
  }
  return false;
}

static bool hasAnyEjitMetadata(Module &M) {
  for (Function &F : M.functions())
    if (F.hasMetadata(MD_EJIT_METADATA))
      return true;
  for (GlobalVariable &GV : M.globals())
    if (GV.hasMetadata(MD_EJIT_METADATA))
      return true;
  return false;
}

static void runDiagnosticCheck(Module &M) {
  // For each ejit_entry function, check that referenced period arrays
  // are declared in the metadata.
  for (Function &F : M.functions()) {
    MDNode *MD = F.getMetadata(MD_EJIT_METADATA);
    if (!hasMDStringEntry(MD, TAG_EJIT_ENTRY))
      continue;

    // Collect declared period dependencies from metadata
    SmallVector<std::string, 4> DeclaredPeriods;
    for (const MDOperand &Op : MD->operands()) {
      auto *Sub = dyn_cast<MDNode>(Op.get());
      if (!Sub || Sub->getNumOperands() < 2)
        continue;
      if (auto *Tag = dyn_cast<MDString>(Sub->getOperand(0))) {
        if (Tag->getString() == TAG_EJIT_PERIOD_ARR_IND)
          if (auto *PN = dyn_cast<MDString>(Sub->getOperand(1)))
            DeclaredPeriods.push_back(PN->getString().str());
      }
    }

    // Check for referenced period arrays not in declared periods
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        for (Value *Op : I.operands()) {
          Value *V = Op->stripPointerCasts();
          if (auto *GV = dyn_cast<GlobalVariable>(V)) {
            MDNode *GMD = GV->getMetadata(MD_EJIT_METADATA);
            if (!GMD)
              continue;
            for (const MDOperand &GMOp : GMD->operands()) {
              auto *Sub = dyn_cast<MDNode>(GMOp.get());
              if (!Sub || Sub->getNumOperands() < 2)
                continue;
              if (auto *Tag = dyn_cast<MDString>(Sub->getOperand(0))) {
                if (Tag->getString() == TAG_EJIT_PERIOD_ARR) {
                  auto *PN = dyn_cast<MDString>(Sub->getOperand(1));
                  if (PN && !is_contained(DeclaredPeriods, PN->getString())) {
                    errs() << "EJit warning: function '" << F.getName()
                           << "' references ejit_period_arr '" << PN->getString()
                           << "' but it is not declared via ejit_period_arr_ind\n";
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

} // anonymous namespace

PreservedAnalyses
EJitAotModulePass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!hasAnyEjitMetadata(M))
    return PreservedAnalyses::all();

  // Run sub-passes in order: PASS2 → PASS3 → PASS4
  // (PASS1 is an independent early pass, not part of this pipeline)
  PreservedAnalyses PA = PreservedAnalyses::none();

  PA.intersect(EJitRegisterPeriodPass().run(M, AM));
  PA.intersect(EJitWrapperGenPass().run(M, AM));
  PA.intersect(EJitPeriodHandlerPass().run(M, AM));

  // Diagnostic consistency check (warnings only, no error)
  runDiagnosticCheck(M);

  return PA;
}
