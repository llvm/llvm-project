//===- SymbolTracker.cpp - Symbol Tracker ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SymbolTracker is an LLVM pass that embeds global variables in the IR
// module to allow a runtime program analysis tool to match IR symbols
// with native symbols in a target-independent way.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SymbolTracker.h"
#include "llvm/IR/Comdat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "symbol-tracker"

static cl::opt<bool>
    EnableBBSymbolTrackers("enable-bb-symbol-trackers", cl::Hidden,
                           cl::init(false),
                           cl::desc("Enable BasicBlock symbol trackers"));

static constexpr StringLiteral GVTrackerPrefix = "__irsym_";
static constexpr StringLiteral BBTrackerPrefix = "__irbb_";
static constexpr StringLiteral BBTrackerNameSep = "_LBB";

using ThunksMap =
    DenseMap<std::pair<FunctionType *, AttributeList>, Function *>;

static GlobalVariable *CreateSymbolTracker(const Twine &Name,
                                           Constant *Initializer,
                                           StringRef Prefix) {
  assert(Initializer != nullptr && "No initializer provided");

  Type *Ty = Initializer->getType();
  const Twine TrackSymName = Twine(Prefix) + Name;
  GlobalVariable *NewGV = new GlobalVariable(
      Ty, /* isConstant */ true, GlobalVariable::ExternalLinkage, Initializer,
      TrackSymName, GlobalValue::NotThreadLocal);
  NewGV->setDSOLocal(true);
  NewGV->setVisibility(GlobalValue::HiddenVisibility);
  NewGV->setSection(".irsymtrack");
  return NewGV;
}

static GlobalVariable *CreateGlobalValueSymbolTracker(GlobalValue &GV) {
  const StringRef Name = GV.getName();

  if (GV.isThreadLocal() || Name.empty() || Name.starts_with("llvm."))
    return nullptr;

  return CreateSymbolTracker(Name, &GV, GVTrackerPrefix);
}

static GlobalVariable *CreateBasicBlockSymbolTracker(size_t i, BasicBlock &BB) {
  if (!EnableBBSymbolTrackers)
    return nullptr;

  if (i == 0)
    return nullptr;

  StringRef FuncName = BB.getParent()->getName();
  BlockAddress *BA = BlockAddress::get(&BB);
  assert(BA != nullptr && "Failed getting BlockAddress!");

  return CreateSymbolTracker(FuncName + BBTrackerNameSep + Twine(i), BA,
                             BBTrackerPrefix);
}

static void EnsureFunctionCallingConvC(Function &F) {
  // We want to change the calling convention of all functions we process to
  // CallingConv::C so that they have a documented stable ABI.
  static constexpr auto NewCC = CallingConv::C;
  const auto CurCC = F.getCallingConv();

  switch (CurCC) {
  case NewCC:
    return;
  case CallingConv::Fast:
  case CallingConv::Cold:
    break;
  default:
    errs() << "WARNING: Skipping unexpected CallingConv '" << CurCC << "'\n";
    return;
  }

  F.setCallingConv(NewCC);
  for (User *U : F.users()) {
    if (auto *CB = dyn_cast<CallBase>(U)) {
      CB->setCallingConv(NewCC);
    }
  }
}

static bool runImpl(Module &M) {
  if (M.getContext().shouldDiscardValueNames()) {
    // Cannot persist value names for BasicBlocks
    LLVM_DEBUG(dbgs() << "WARNING: Cannot embed symbol trackers in value-name-"
                      << "discarding LLVMContext\n");
    return false;
  }

  std::vector<GlobalVariable *> TrackerGVs;
  for (GlobalValue &GV : M.global_values()) {
    if (GlobalVariable *TrackerGV = CreateGlobalValueSymbolTracker(GV))
      TrackerGVs.push_back(TrackerGV);
    else
      continue;

    if (Function *F = dyn_cast<Function>(&GV)) {
      // Function implementation will not be emitted into the object file,
      // no point in adding symbol trackers for its BasicBlocks.
      if (F->hasAvailableExternallyLinkage())
        continue;

      EnsureFunctionCallingConvC(*F);

      size_t BBIndex = 0;
      for (auto &BB : *F) {
        if (auto *TrackerGV = CreateBasicBlockSymbolTracker(BBIndex, BB))
          TrackerGVs.push_back(TrackerGV);
        BBIndex++;
      }
    }
  }

  for (GlobalVariable *GV : TrackerGVs)
    M.insertGlobalVariable(GV);

  return !TrackerGVs.empty();
}

PreservedAnalyses EmbedSymbolTrackersPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  if (!runImpl(M))
    return PreservedAnalyses::all();

  // Be conservative for now, optimize later if necessary
  return PreservedAnalyses::none();
}
