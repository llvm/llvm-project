//===- InternalizeLinkOnceODR.cpp - Clone linkonce_odr functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/InternalizeLinkOnceODR.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Comdat.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "internalize-linkonce-odr"

STATISTIC(NumInternalized, "Number of linkonce_odr functions made internal");
STATISTIC(NumCloned, "Number of internal clones of linkonce_odr functions");
STATISTIC(NumCallsRedirected, "Number of direct calls redirected to a clone");

enum class LinkOnceODRInternalization { Off, LikelyModuleLocal, All };

static cl::opt<LinkOnceODRInternalization> EnableLinkOnceODRInternalization(
    "enable-linkonce-odr-internalization", cl::Hidden,
    cl::init(LinkOnceODRInternalization::Off),
    cl::desc("Clone internal copies of linkonce_odr functions"),
    cl::values(clEnumValN(LinkOnceODRInternalization::Off, "off", "never"),
               clEnumValN(LinkOnceODRInternalization::LikelyModuleLocal,
                          "likely-module-local",
                          "when TU-local hint is present"),
               clEnumValN(LinkOnceODRInternalization::All, "all",
                          "all linkonce_odr functions")));

static bool isCandidate(const Function &F) {
  if (F.isDeclaration() || !F.hasLinkOnceODRLinkage())
    return false;
  return EnableLinkOnceODRInternalization == LinkOnceODRInternalization::All ||
         F.hasFnAttribute("frontend-hint-likely-module-local");
}

static bool isDirectCall(const Use &U) {
  const auto *CB = dyn_cast<CallBase>(U.getUser());
  return CB && CB->isCallee(&U);
}

// Whether a copy of F can be made. Duplicating it must not duplicate anything
// that has to be unique, like labels of block addresses or of inline asm.
static bool isCloneable(const Function &F) {
  if (F.hasFnAttribute(Attribute::NoDuplicate))
    return false;
  for (const BasicBlock &BB : F)
    if (BB.hasAddressTaken())
      return false;
  for (const Instruction &I : instructions(F))
    if (const auto *CB = dyn_cast<CallBase>(&I))
      if (CB->isInlineAsm() || CB->cannotDuplicate())
        return false;
  return true;
}

PreservedAnalyses InternalizeLinkOnceODRPass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  if (EnableLinkOnceODRInternalization == LinkOnceODRInternalization::Off)
    return PreservedAnalyses::all();

  SmallVector<Function *, 16> Candidates;
  for (Function &F : M)
    if (isCandidate(F))
      Candidates.push_back(&F);

  bool Changed = false;
  for (Function *F : Candidates) {
    bool HasDirectCalls = false, HasOtherUses = false;
    for (const Use &U : F->uses()) {
      if (isDirectCall(U))
        HasDirectCalls = true;
      else
        HasOtherUses = true;
    }
    if (!HasDirectCalls)
      continue;

    // (1) Make local in-place when allowed (only direct calls, alone in comdat)
    const Comdat *C = F->getComdat();
    if (!HasOtherUses && (!C || C->getUsers().size() == 1)) {
      F->setLinkage(GlobalValue::InternalLinkage);
      F->setComdat(nullptr);
      ++NumInternalized;
      Changed = true;
      continue;
    }

    if (!isCloneable(*F))
      continue;

    // (2) Clone and redirect (direct calls only)
    ValueToValueMapTy VMap;
    Function *Clone = CloneFunction(F, VMap);
    Clone->setName(F->getName() + ".internal");
    Clone->setLinkage(GlobalValue::InternalLinkage);
    Clone->setComdat(nullptr);
    Clone->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    for (Use &U : make_early_inc_range(F->uses())) {
      if (isDirectCall(U)) {
        U.set(Clone);
        ++NumCallsRedirected;
      }
    }
    ++NumCloned;
    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
