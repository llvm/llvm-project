//===- PGOFlowVerify.cpp - PGO flow verification -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After-pass hook and standalone pass. Flag behavior is on the cl::opt
// definitions below.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/PGOFlowVerify.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <numeric>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "verify-pgo-flow"

static cl::opt<bool> VerifyPGOFlow(
    "verify-pgo-flow", cl::init(false), cl::Hidden,
    cl::desc("Run InstrProf flow checks after IR-changing passes"));

static cl::opt<bool> VerifyPGOFlowPrintDiagnostics(
    "verify-pgo-flow-print-diagnostics", cl::init(true), cl::Hidden,
    cl::desc("Print verify-pgo-flow banners and findings to stderr"));

static cl::opt<bool>
    VerifyPGOFlowFatal("verify-pgo-flow-fatal", cl::init(false), cl::Hidden,
                       cl::desc("Abort after a flow-check finding"));

static void printVerifyBanner(StringRef PassName, bool Skipped) {
  if (!VerifyPGOFlowPrintDiagnostics)
    return;
  errs() << "*** PGO Flow Verification After " << PassName
         << (Skipped ? " (Skipped)" : "") << " ***\n";
}

bool PGOFlowVerifier::isHookEnabled() { return VerifyPGOFlow; }

static void emitPGOFlowDiagnostic(const Function *F, StringRef RemarkName,
                                  const Twine &Msg) {
  if (!F)
    return;
  std::string Text = Msg.str();
  if (VerifyPGOFlowPrintDiagnostics)
    errs() << "PGOFlowVerify[" << RemarkName << "] " << F->getName() << ": "
           << Text << "\n";
  if (VerifyPGOFlowFatal)
    report_fatal_error(Twine("PGOFlowVerify[") + RemarkName + "] " +
                           F->getName() + ": " + Text,
                       /*gen_crash_diag=*/false);
}

void PGOFlowVerifier::registerCallbacks(PassInstrumentationCallbacks &PIC) {
  if (!VerifyPGOFlow) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: hook not registered "
                         "(-verify-pgo-flow is off)\n");
    return;
  }

  LLVM_DEBUG(dbgs() << "PGOFlowVerifier: registering after-pass hook\n");
  PIC.registerAfterPassCallback([this](StringRef PassName, IRUnitRef IR,
                                       const PreservedAnalyses &PA) {
    // Same ignore list as print-changed. Adaptors would re-walk the
    // function after every nested loop pass.
    static const std::vector<StringRef> Ignored = {"PassManager",
                                                   "PassAdaptor",
                                                   "AnalysisManagerProxy",
                                                   "DevirtSCCRepeatedPass",
                                                   "ModuleInlinerWrapperPass",
                                                   "VerifierPass",
                                                   "PrintModulePass",
                                                   "PrintMIRPass",
                                                   "PrintMIRPreparePass",
                                                   "RequireAnalysisPass",
                                                   "InvalidateAnalysisPass"};
    if (isSpecialPass(PassName, Ignored)) {
      LLVM_DEBUG(dbgs() << "PGOFlowVerifier: after " << PassName
                        << " (skip, ignored pass)\n");
      return;
    }
    bool Changed = !PA.areAllPreserved();
    LLVM_DEBUG(
        dbgs() << "PGOFlowVerifier: after " << PassName
               << (Changed ? " (walk)\n" : " (skip, PA all preserved)\n"));
    printVerifyBanner(PassName, /*Skipped=*/!Changed);
    if (!Changed)
      return;
    runAfterPass(PassName, IR);
  });
}

void PGOFlowVerifier::invalidateFunctionFrequencyCache(IRUnitRef IR) {
  if (isa<Module>(IR)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: clear block-freq cache (module)\n");
    FunctionBlockFreqInfoCache.clear();
    return;
  }
  if (const auto *F = dyn_cast<Function>(IR)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: drop block-freq cache for '"
                      << F->getName() << "'\n");
    FunctionBlockFreqInfoCache.erase(F);
    return;
  }
  if (const auto *C = dyn_cast<LazyCallGraph::SCC>(IR)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: drop block-freq cache for SCC\n");
    for (const LazyCallGraph::Node &N : *C)
      FunctionBlockFreqInfoCache.erase(&N.getFunction());
    return;
  }
  if (const auto *L = dyn_cast<Loop>(IR)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: drop block-freq cache for loop\n");
    if (L->getHeader())
      FunctionBlockFreqInfoCache.erase(L->getHeader()->getParent());
    return;
  }
  LLVM_DEBUG(
      dbgs() << "PGOFlowVerifier: clear block-freq cache (unhandled IR)\n");
  FunctionBlockFreqInfoCache.clear();
}

void PGOFlowVerifier::runAfterPass(StringRef PassID, IRUnitRef IR) {
  invalidateFunctionFrequencyCache(IR);
  if (const auto *M = dyn_cast<Module>(IR)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: module IR after " << PassID << "\n");
    runAfterPass(M);
  } else if (const auto *F = dyn_cast<Function>(IR)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: function IR '" << F->getName()
                      << "' after " << PassID << "\n");
    runAfterPass(F);
  } else if (const auto *C = dyn_cast<LazyCallGraph::SCC>(IR)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: SCC IR after " << PassID << "\n");
    runAfterPass(C);
  } else if (const auto *L = dyn_cast<Loop>(IR)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: loop IR after " << PassID << "\n");
    runAfterPass(L);
  } else {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: unhandled IR unit after " << PassID
                      << "\n");
  }
}

void PGOFlowVerifier::runAfterPass(const Module *M) {
  if (!M)
    return;
  if (!hasInstrProfUseSummary(M)) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: skip module '" << M->getName()
                      << "' (no InstrProf use-phase summary)\n");
    return;
  }
  for (const Function &F : *M)
    runAfterPass(&F);
}

void PGOFlowVerifier::runAfterPass(const Function *F) {
  if (!F || F->isDeclaration())
    return;
  if (!hasInstrProfUseSummary(F->getParent())) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: skip '" << F->getName()
                      << "' (no InstrProf use-phase summary)\n");
    return;
  }
  computeBlockFrequencies(F);
  validateBlockFrequencies(F);
}

void PGOFlowVerifier::runAfterPass(const LazyCallGraph::SCC *C) {
  if (!C)
    return;
  for (const LazyCallGraph::Node &N : *C)
    runAfterPass(&N.getFunction());
}

void PGOFlowVerifier::runAfterPass(const Loop *L) {
  if (!L)
    return;
  runAfterPass(L->getHeader()->getParent());
}

void PGOFlowVerifier::computeBlockFrequencies(const Function *F) {
  if (!F)
    return;

  AllBlockFreqInfo AllFreqInfo;
  for (const BasicBlock &BB : *F) {
    AllFreqInfo[&BB].NumUnknownIn = pred_size(&BB);
    AllFreqInfo[&BB].NumUnknownOut = succ_size(&BB);
  }

  AllFreqInfo[&F->getEntryBlock()].NumUnknownIn = 1;
  if (std::optional<uint64_t> Count = F->getEntryCount()) {
    AllFreqInfo[&F->getEntryBlock()].SumIn = *Count;
    AllFreqInfo[&F->getEntryBlock()].NumUnknownIn = 0;
    if (*Count == 0) {
      for (const BasicBlock &BB : *F) {
        AllFreqInfo[&BB].NumUnknownIn = 0;
        AllFreqInfo[&BB].SumIn = 0;
        AllFreqInfo[&BB].NumUnknownOut = 0;
        AllFreqInfo[&BB].SumOut = 0;
      }
      FunctionBlockFreqInfoCache[F] = std::move(AllFreqInfo);
      return;
    } else if (const Instruction *EntryTerm =
                   F->getEntryBlock().getTerminator();
               EntryTerm && EntryTerm->getNumSuccessors() == 0) {
      AllFreqInfo[&F->getEntryBlock()].SumOut = *Count;
      AllFreqInfo[&F->getEntryBlock()].NumUnknownOut = 0;
    }
  }

  SmallVector<const BasicBlock *, 16> Worklist;
  SmallPtrSet<const BasicBlock *, 16> InWorklist;
  auto Enqueue = [&](const BasicBlock *BB) {
    if (BB && InWorklist.insert(BB).second)
      Worklist.push_back(BB);
  };

  // Unreachable blocks (including cyclic leftover SCCs) are not live flow.
  SmallPtrSet<const BasicBlock *, 16> Reachable;
  SmallVector<const BasicBlock *, 16> ReachWork;
  ReachWork.push_back(&F->getEntryBlock());
  Reachable.insert(&F->getEntryBlock());
  while (!ReachWork.empty()) {
    const BasicBlock *BB = ReachWork.pop_back_val();
    const Instruction *Term = BB->getTerminator();
    if (!Term)
      continue;
    for (unsigned I = 0, E = Term->getNumSuccessors(); I < E; ++I) {
      const BasicBlock *Succ = Term->getSuccessor(I);
      if (Reachable.insert(Succ).second)
        ReachWork.push_back(Succ);
    }
  }

  auto HasCountTypeOutgoing = [&](const BasicBlock *BB) {
    const Instruction *Term = BB->getTerminator();
    if (!Term || Term->getNumSuccessors() == 0)
      return false;
    if (hasBranchWeightOrigin(*Term) ||
        hasExplicitlyUnknownBranchWeights(*Term))
      return false;
    MDNode *WeightMD = getValidBranchWeightMDNode(*Term);
    if (!WeightMD)
      return false;
    SmallVector<uint64_t, 8> Weights;
    extractFromBranchWeightMD64(WeightMD, Weights);
    return Weights.size() == Term->getNumSuccessors();
  };

  auto ShouldProcess = [&](const BasicBlock *BB) {
    if (!BB || !Reachable.contains(BB))
      return false;
    const BlockFreqInfo &Info = AllFreqInfo[BB];
    if (Info.NumUnknownIn == 0)
      return true;
    // Live flow already credited, but a backedge is still unknown. Apply
    // count-type outs now. Do not early-walk unweighted blocks or leftover
    // !prof on a 0-in path.
    return Info.SumIn > 0 && HasCountTypeOutgoing(BB);
  };

  auto ReleaseEdge = [&](const BasicBlock *Succ, uint64_t Add) {
    if (!Succ)
      return;
    BlockFreqInfo &SuccInfo = AllFreqInfo[Succ];
    if (SuccInfo.NumUnknownIn > 0)
      SuccInfo.NumUnknownIn--;
    if (Add)
      SuccInfo.SumIn += Add;
    if (ShouldProcess(Succ))
      Enqueue(Succ);
  };

  for (const BasicBlock &BB : *F) {
    if (Reachable.contains(&BB))
      continue;
    BlockFreqInfo &Info = AllFreqInfo[&BB];
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: skip leftover branch weights on "
                      << "unreachable block " << BB.getName() << " in '"
                      << F->getName() << "'\n");
    Info.NumUnknownIn = 0;
    Info.SumIn = 0;
    Info.NumUnknownOut = 0;
    Info.SumOut = 0;
    const Instruction *Term = BB.getTerminator();
    if (!Term)
      continue;
    // Drop dead preds from live successors so conservation can still run.
    for (unsigned I = 0, E = Term->getNumSuccessors(); I < E; ++I) {
      const BasicBlock *Succ = Term->getSuccessor(I);
      if (Succ && Reachable.contains(Succ))
        ReleaseEdge(Succ, 0);
    }
  }

  auto ProcessResolved = [&](const BasicBlock *BB) {
    const Instruction *Term = BB->getTerminator();
    if (!Term)
      return;
    BlockFreqInfo &Info = AllFreqInfo[BB];
    if (isa<ReturnInst>(Term)) {
      if (Info.NumUnknownIn != 0)
        return;
      Info.SumOut = Info.SumIn;
      Info.NumUnknownOut = 0;
      return;
    }
    // Known-zero incoming: do not extract or apply leftover branch weights.
    if (Info.NumUnknownIn == 0 && Info.SumIn == 0) {
      LLVM_DEBUG(dbgs() << "PGOFlowVerifier: skip leftover branch weights on "
                        << "unreachable/zero-count block " << BB->getName()
                        << " in '" << F->getName() << "'\n");
      Info.NumUnknownOut = 0;
      Info.SumOut = 0;
      for (unsigned I = 0, E = Term->getNumSuccessors(); I < E; ++I)
        ReleaseEdge(Term->getSuccessor(I), 0);
      return;
    }
    // Count-type InstrProf branch_weights only. !"expected" (llvm.expect)
    // origins are probabilities, not execution counts. Explicitly unknown
    // weights are not counts either. Leave NumUnknownIn/NumUnknownOut as-is.
    if (hasBranchWeightOrigin(*Term) ||
        hasExplicitlyUnknownBranchWeights(*Term)) {
      LLVM_DEBUG(dbgs() << "PGOFlowVerifier: skip non-count branch weights in '"
                        << F->getName() << "' block " << BB->getName() << "\n");
      return;
    }
    SmallVector<uint32_t, 8> Weights32;
    if (extractBranchWeights(*Term, Weights32) &&
        Weights32.size() == Term->getNumSuccessors()) {
      // Outs already closed (weights applied while a backedge was unknown).
      if (Info.NumUnknownOut == 0)
        return;
      // No live flow yet.
      if (Info.SumIn == 0)
        return;
      SmallVector<uint64_t, 8> Weights(Weights32.begin(), Weights32.end());
      Info.NumUnknownOut = 0;
      Info.SumOut =
          std::accumulate(Weights.begin(), Weights.end(), uint64_t(0));
      for (unsigned I = 0, E = Term->getNumSuccessors(); I < E; ++I)
        ReleaseEdge(Term->getSuccessor(I), Weights[I]);
      return;
    }
    if (Info.NumUnknownIn != 0)
      return;
    if (Info.NumUnknownOut == 1 && Term->getNumSuccessors() == 1) {
      ReleaseEdge(Term->getSuccessor(0), Info.SumIn);
      Info.NumUnknownOut = 0;
      Info.SumOut = Info.SumIn;
    }
  };

  // Count-type weights can close outs before every pred is known (live
  // loops). Unweighted copy and 0-in leftover skip still wait for pred close
  // so BB-list order cannot apply !prof on a child listed before a 0-edge.
  for (const BasicBlock &BB : *F)
    if (ShouldProcess(&BB))
      Enqueue(&BB);
  while (!Worklist.empty()) {
    const BasicBlock *BB = Worklist.pop_back_val();
    ProcessResolved(BB);
  }

  FunctionBlockFreqInfoCache[F] = std::move(AllFreqInfo);
}

void PGOFlowVerifier::validateBlockFrequencies(const Function *F) {
  if (!F)
    return;
  auto CachedIt = FunctionBlockFreqInfoCache.find(F);
  if (CachedIt == FunctionBlockFreqInfoCache.end()) {
    LLVM_DEBUG(dbgs() << "PGOFlowVerifier: no block-freq cache for '"
                      << F->getName() << "'\n");
    return;
  }

  const AllBlockFreqInfo &AllFreqInfo = CachedIt->second;
  for (const BasicBlock &BB : *F) {
    const Instruction *Term = BB.getTerminator();
    if (!Term || Term->getNumSuccessors() == 0)
      continue;
    auto It = AllFreqInfo.find(&BB);
    if (It == AllFreqInfo.end())
      continue;
    const BlockFreqInfo &Info = It->second;
    if (Info.NumUnknownIn == 0 && Info.NumUnknownOut == 0 &&
        Info.SumIn != Info.SumOut)
      emitPGOFlowDiagnostic(F, "BlockFrequencyMismatch",
                            Twine("block ") + BB.getName() +
                                ": incoming=" + Twine(Info.SumIn) +
                                " vs outgoing=" + Twine(Info.SumOut));
    else if (Info.NumUnknownIn != 0 || Info.NumUnknownOut != 0)
      LLVM_DEBUG(dbgs() << "PGOFlowVerifier: unknown edges in '" << F->getName()
                        << "' block " << BB.getName()
                        << " in=" << Info.NumUnknownIn
                        << " out=" << Info.NumUnknownOut << "\n");
  }
}

bool PGOFlowVerifier::hasInstrProfUseSummary(const Module *M) const {
  if (!M)
    return false;
  Metadata *SummaryMD = M->getProfileSummary(/*IsCS=*/false);
  if (!SummaryMD)
    return false;
  std::unique_ptr<ProfileSummary> PS(ProfileSummary::getFromMD(SummaryMD));
  return PS && (PS->getKind() == ProfileSummary::PSK_Instr ||
                PS->getKind() == ProfileSummary::PSK_CSInstr);
}

PreservedAnalyses PGOFlowVerifierPass::run(Module &M,
                                           ModuleAnalysisManager &MAM) {
  (void)MAM;
  LLVM_DEBUG(dbgs() << "PGOFlowVerifier: pipeline pass (module)\n");
  printVerifyBanner("verify-pgo-flow", /*Skipped=*/false);
  Verifier.runAfterPass("verify-pgo-flow", M);
  return PreservedAnalyses::all();
}

PreservedAnalyses PGOFlowVerifierPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  (void)FAM;
  LLVM_DEBUG(dbgs() << "PGOFlowVerifier: pipeline pass (function "
                    << F.getName() << ")\n");
  printVerifyBanner("verify-pgo-flow", /*Skipped=*/false);
  Verifier.runAfterPass("verify-pgo-flow", F);
  return PreservedAnalyses::all();
}
