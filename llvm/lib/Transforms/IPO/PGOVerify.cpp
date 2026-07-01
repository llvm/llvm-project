//===- PGOVerify.cpp - PGO Verification ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IPGOVerifier currently provides registration-only diagnostics for
// pass-instrumentation tracing under `-verify-ipgo`.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/PGOVerify.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <numeric>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "verify-ipgo"

static cl::opt<bool> VerifyIPGOPrintDiagnostics(
    "verify-ipgo-print-diagnostics", cl::init(true), cl::Hidden,
    cl::desc("Print verify-ipgo diagnostics to stderr"));

static cl::opt<bool>
    VerifyIPGO("verify-ipgo", cl::init(false), cl::Hidden,
               cl::desc("Enable Instrumented PGO verification"));

static cl::list<std::string>
  VerifyIPGOFuncList("verify-ipgo-funcs", cl::Hidden,
             cl::desc("Comma-separated list of functions to verify"));

/// Emit PGO verification diagnostics with structured formatting.
///
/// \param F Function being verified.
/// \param RemarkName Diagnostic remark identifier.
/// \param Msg Error/diagnostic message.
static void emitPGOVerifyDiagnostic(const Function *F, StringRef RemarkName,
                                    const Twine &Msg) {
  std::string MsgText = Msg.str();
  if (VerifyIPGOPrintDiagnostics)
    errs() << "PGOVerify[" << RemarkName << "] " << F->getName() << ": "
           << MsgText << "\n";
  LLVM_DEBUG(dbgs() << "PGOVerify[" << RemarkName << "] " << F->getName()
                    << ": " << MsgText << "\n");
}

bool IPGOVerifier::shouldVerifyFunction(const Function *F) const {
  if (!F || F->isDeclaration())
    return false;

  if (F->hasAvailableExternallyLinkage())
    return false;

  // Cache command-line function filters.
  static const DenseSet<StringRef> FuncFilter = [] {
    DenseSet<StringRef> S;
    for (const auto &Func : VerifyIPGOFuncList)
      S.insert(Func);
    return S;
  }();

  return FuncFilter.empty() || FuncFilter.count(F->getName());
}

/// Register post-pass diagnostic callbacks for `-verify-ipgo`.
///
/// \param PIC Pass instrumentation callback registry.
void IPGOVerifier::registerCallbacks(PassInstrumentationCallbacks &PIC) {
  if (!VerifyIPGO)
    return;

  PIC.registerAfterPassCallback(
      [this](StringRef PassName, Any IR, const PreservedAnalyses &PA) {
        bool IsChanged = !PA.areAllPreserved();

        StringRef Skipped = IsChanged ? "" : " (Skipped)";
        if (VerifyIPGOPrintDiagnostics)
          errs() << "*** IPGO Verification After " << PassName << Skipped
                 << " ***\n";
        LLVM_DEBUG(dbgs() << "\n*** IPGO Verification After " << PassName
                          << Skipped << " ***\n");
        if (!IsChanged) {
          // Pass made no IR changes; skip verification.
          return;
        }

        runAfterPass(PassName, IR);
      });
}

/// Dispatch post-pass handling for supported IR unit kinds.
///
/// \param PassID Name of the pass that completed.
/// \param IR IR unit received from pass instrumentation callbacks.
void IPGOVerifier::runAfterPass(StringRef PassID, Any IR) {
  (void)PassID;

  // Drop cached per-function state for the IR unit that just changed before
  // rebuilding or rechecking any derived block-frequency information.
  invalidateFunctionFrequencyCache(IR);

  if (const auto *M = any_cast<const Module *>(&IR))
    runAfterPass(*M);
  else if (const auto *F = any_cast<const Function *>(&IR)) {
    // The verifier does not mutate IR, but the handler API is function-based,
    // so adapt the callback payload here.
    auto *NonConstF = const_cast<Function *>(*F);
    runAfterPass(NonConstF);
  } else if (const auto *C = any_cast<const LazyCallGraph::SCC *>(&IR))
    runAfterPass(*C);
  else if (const auto *L = any_cast<const Loop *>(&IR))
    runAfterPass(*L);
  else {
    return;
  }
}

void IPGOVerifier::invalidateFunctionFrequencyCache(Any IR) {
  if (const auto *M = any_cast<const Module *>(&IR)) {
    (void)M;
    // Module passes can invalidate frequency state for any contained function.
    FunctionBlockFreqInfoCache.clear();
    LLVM_DEBUG(dbgs() << "PGOVerify cache invalidated: module\n");
    return;
  }

  if (const auto *F = any_cast<const Function *>(&IR)) {
    FunctionBlockFreqInfoCache.erase(*F);
    LLVM_DEBUG(dbgs() << "PGOVerify cache invalidated: function\n");
    return;
  }

  if (const auto *C = any_cast<const LazyCallGraph::SCC *>(&IR)) {
    for (const LazyCallGraph::Node &N : **C)
      FunctionBlockFreqInfoCache.erase(&N.getFunction());
    LLVM_DEBUG(dbgs() << "PGOVerify cache invalidated: scc\n");
    return;
  }

  if (const auto *L = any_cast<const Loop *>(&IR)) {
    FunctionBlockFreqInfoCache.erase((*L)->getHeader()->getParent());
    LLVM_DEBUG(dbgs() << "PGOVerify cache invalidated: loop\n");
    return;
  }

  FunctionBlockFreqInfoCache.clear();
  LLVM_DEBUG(dbgs() << "PGOVerify cache invalidated: unknown\n");
}

/// Delegate module callback handling to the function handler.
///
/// \param M Module callback payload.
void IPGOVerifier::runAfterPass(const Module *M) {
  // Run Use-phase checks only when an InstrProf use summary is present.
  if (M->getProfileSummary(/*IsCS=*/true))
    return;
  if (!hasInstrProfUseSummary(M)) {
    for (const Function &F : *M) {
      if (F.isDeclaration())
        continue;
      verifyGenPhase(&F);
    }
    return;
  }
  // First build frequency cache for all non-declaration functions so caller
  // information is available regardless of function order in the module.
  for (const Function &F : *M) {
    if (F.isDeclaration())
      continue;
    // use BFI's non-synthetic per-block profile
    // counts as the primary overflow signal.
    auto *NonConstF = const_cast<Function *>(&F);
    DominatorTree DT(*NonConstF);
    LoopInfo LI(DT);
    BranchProbabilityInfo BPI(*NonConstF, LI, nullptr, &DT, nullptr);
    BlockFrequencyInfo BFI(*NonConstF, BPI, LI);

    computeBlockFrequencies(&F, BFI);
  }

  // Then run validations using the populated cache.
  for (const Function &F : *M) {
    if (!shouldVerifyFunction(&F))
      continue;
    validateBlockFrequencies(&F);
    validateEntryCountAgainstCallerSum(&F);
  }
}

/// Per-function post-pass handler.
///
/// \param F Function callback payload.
void IPGOVerifier::runAfterPass(Function *F) {
  if (!F || F->isDeclaration() || !F->getParent())
    return;

  if (!shouldVerifyFunction(F))
    return;

  // Run Use-phase checks only when an InstrProf use summary is present.
  if (F->getParent()->getProfileSummary(/*IsCS=*/true))
    return;
  if (!hasInstrProfUseSummary(F->getParent())) {
    // Run Gen-phase checks (no dependencies on other passes).
    verifyGenPhase(F);
    return;
  }

  // Rebuild the minimal local analysis stack here so verification can query
  // non-synthetic block profile counts after each pass callback.
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  BranchProbabilityInfo BPI(*F, LI, nullptr, &DT, nullptr);
  BlockFrequencyInfo BFI(*F, BPI, LI);

  computeBlockFrequencies(F, BFI);
  validateBlockFrequencies(F);
  validateEntryCountAgainstCallerSum(F);
}

/// Delegate SCC callback handling to the function handler.
///
/// \param C SCC callback payload.
void IPGOVerifier::runAfterPass(const LazyCallGraph::SCC *C) {
  for (const LazyCallGraph::Node &N : *C)
    runAfterPass(&N.getFunction());
}

/// Delegate loop callback handling to the containing function handler.
///
/// \param L Loop callback payload.
void IPGOVerifier::runAfterPass(const Loop *L) {
  runAfterPass(L->getHeader()->getParent());
}

/// Compute and cache per-block flow state for verifier checks.
///
/// This seeds block-local incoming and outgoing totals from profile metadata,
/// then iteratively propagates any facts that become forced by CFG structure
/// until no additional block flow state can be resolved.
///
///
/// \param F Function whose basic blocks are being analyzed.
/// \param BFI BlockFrequencyInfo used to query non-synthetic profile counts
///            and to detect cases where strict verification would be unsafe.
void IPGOVerifier::computeBlockFrequencies(const Function *F,
                                           const BlockFrequencyInfo &BFI) {
  // Skip strict flow checks when local profile counts can overflow uint32.
  if (hasFunctionLocalCountOverflow(F, BFI)) {
    FunctionBlockFreqInfoCache[F] = AllBlockFreqInfo();
    return;
  }

  AllBlockFreqInfo AllFreqInfo;

  for (const BasicBlock &BB : *F) {
    // Start with all predecessor and successor contributions unknown, then
    // refine each block as profile metadata or structural rules provide facts.
    AllFreqInfo[&BB].numUnknownIn = llvm::pred_size(&BB);
    AllFreqInfo[&BB].numUnknownOut = llvm::succ_size(&BB);
    AllFreqInfo[&BB].sumIn = 0;
    AllFreqInfo[&BB].sumOut = 0;
  }

  // Model the function entry as an external incoming edge so the entry count
  // can seed flow-conservation reasoning like any other known predecessor.
  AllFreqInfo[&F->getEntryBlock()].numUnknownIn = 1;

  if (auto Count = F->getEntryCount()) {
    AllFreqInfo[&F->getEntryBlock()].sumIn = Count->getCount();
    AllFreqInfo[&F->getEntryBlock()].numUnknownIn = 0;
    if (Count->getCount() == 0) {
      // A zero entry count forces every reachable block contribution to zero,
      // which avoids leaving unknown edges behind in dead-profile functions.
      for (const BasicBlock &BB : *F) {
        AllFreqInfo[&BB].numUnknownIn = 0;
        AllFreqInfo[&BB].sumIn = 0;
        AllFreqInfo[&BB].numUnknownOut = 0;
        AllFreqInfo[&BB].sumOut = 0;
      }
    } else {
      const Instruction *Term = F->getEntryBlock().getTerminator();
      if (Term && (Term->getNumSuccessors() == 0)) {
        AllFreqInfo[&F->getEntryBlock()].sumOut = Count->getCount();
        AllFreqInfo[&F->getEntryBlock()].numUnknownOut = 0;
      }
    }
  }

  for (const BasicBlock &BB : *F) {
    SmallVector<uint64_t> Weights;
    const Instruction *Term = BB.getTerminator();
    if (!Term)
      continue;

    if (isa<ReturnInst>(Term) && AllFreqInfo[&BB].numUnknownIn == 0) {
      AllFreqInfo[&BB].sumOut = AllFreqInfo[&BB].sumIn;
      AllFreqInfo[&BB].numUnknownOut = 0;
      continue;
    }

    if (MDNode *Prof = Term->getMetadata(LLVMContext::MD_prof)) {
      if (Prof->getNumOperands() > 1) {
        for (unsigned I = 1; I < Prof->getNumOperands(); ++I) {
          auto *CI = mdconst::dyn_extract<ConstantInt>(Prof->getOperand(I));
          if (!CI) {
            // Ignore malformed weight metadata and leave the block unresolved.
            Weights.clear();
            break;
          }
          Weights.push_back(CI->getZExtValue());
        }
      }
    }

    if (Weights.empty())
      continue;

    if (Weights.size() != Term->getNumSuccessors())
      continue;

    // Explicit successor weights fully determine the outgoing total for this
    // terminator and contribute known incoming counts to each successor.
    for (unsigned I = 0; I < Term->getNumSuccessors(); ++I) {
      if (AllFreqInfo[Term->getSuccessor(I)].numUnknownIn > 0)
        AllFreqInfo[Term->getSuccessor(I)].numUnknownIn--;
      AllFreqInfo[Term->getSuccessor(I)].sumIn += Weights[I];
    }
    AllFreqInfo[&BB].numUnknownOut = 0;
    AllFreqInfo[&BB].sumOut =
        std::accumulate(Weights.begin(), Weights.end(), uint64_t(0));
  }

  bool Changed = false;
  do {
    Changed = false;
    for (const BasicBlock &BB : *F) {
      const Instruction *Term = BB.getTerminator();
      if (!Term)
        continue;

      // Once a block is known to receive zero flow, every still-unknown exit
      // edge from that block can also be fixed to zero.
      if (AllFreqInfo[&BB].numUnknownIn == 0 && AllFreqInfo[&BB].sumIn == 0 &&
          AllFreqInfo[&BB].numUnknownOut > 0) {
        for (unsigned I = 0; I < Term->getNumSuccessors(); ++I)
          if (AllFreqInfo[Term->getSuccessor(I)].numUnknownIn > 0)
            AllFreqInfo[Term->getSuccessor(I)].numUnknownIn--;
        AllFreqInfo[&BB].numUnknownOut = 0;
        AllFreqInfo[&BB].sumOut = 0;

        Changed = true;
        continue;
      }

      // A single unresolved successor on a single-successor terminator must
      // carry the entire incoming flow for this block.
      if (AllFreqInfo[&BB].numUnknownIn == 0 &&
          AllFreqInfo[&BB].numUnknownOut == 1) {
        if (Term->getNumSuccessors() > 1)
          continue;

        for (unsigned I = 0; I < Term->getNumSuccessors(); ++I) {
          if (AllFreqInfo[Term->getSuccessor(I)].numUnknownIn > 0)
            AllFreqInfo[Term->getSuccessor(I)].numUnknownIn--;
          AllFreqInfo[Term->getSuccessor(I)].sumIn += AllFreqInfo[&BB].sumIn;
        }
        AllFreqInfo[&BB].numUnknownOut = 0;
        AllFreqInfo[&BB].sumOut = AllFreqInfo[&BB].sumIn;

        Changed = true;
      }
    }
  } while (Changed);

  FunctionBlockFreqInfoCache[F] = AllFreqInfo;
}

const IPGOVerifier::AllBlockFreqInfo *
IPGOVerifier::getCachedBlockFreqInfo(const Function *F) const {
  auto It = FunctionBlockFreqInfoCache.find(F);
  if (It == FunctionBlockFreqInfoCache.end())
    return nullptr;
  return &It->second;
}

bool IPGOVerifier::hasFunctionLocalCountOverflow(const Function *F,
                                           const BlockFrequencyInfo &BFI) const {
  constexpr uint64_t UInt32Max = std::numeric_limits<uint32_t>::max();

  if (auto EntryCount = F->getEntryCount();
      EntryCount && EntryCount->getCount() > UInt32Max)
    return true;


  bool HasUnknownBlockCount = false;
  for (const BasicBlock &BB : *F) {
    if (&BB == &F->getEntryBlock())
      continue;

    // Only trust non-synthetic counts here; synthetic counts may already be
    // inferred from the CFG and would circularly justify verifier results.
    if (std::optional<uint64_t> Count =
            BFI.getBlockProfileCount(&BB, /*AllowSynthetic=*/false);
        Count && *Count > UInt32Max)
      return true;
    else if (!Count)
      HasUnknownBlockCount = true;
  }

  // Conservative fallback when some per-block counts are unavailable.
  if (HasUnknownBlockCount) {
    if (const Module *M = F->getParent()) {
      // The profile summary gives an upper bound when local block counts are
      // missing, allowing the verifier to avoid strict checks near overflow.
      Metadata *SummaryMD = M->getProfileSummary(/*IsCS=*/false);
      if (SummaryMD) {
        std::unique_ptr<ProfileSummary> PS(
            ProfileSummary::getFromMD(SummaryMD));
        if (PS && PS->getMaxInternalCount() > UInt32Max)
          return true;
      }
    }
  }

  return false;
}

/// Validate instrumentation-generation phase invariants.
void IPGOVerifier::verifyGenPhase(const Function *F) {
  // Validate instrprof_increment names against the containing function.
  auto *IncIntrinsic = Intrinsic::getOrInsertDeclaration(
      const_cast<Module *>(F->getParent()), Intrinsic::instrprof_increment);

  if (IncIntrinsic) {
    for (User *U : IncIntrinsic->users()) {
      auto *Instr = dyn_cast<InstrProfCntrInstBase>(U);
      if (!Instr || Instr->getFunction() != F)
        continue;

      StringRef Prefix = getInstrProfNameVarPrefix();
      StringRef ProfiledName =
          cast<InstrProfInstBase>(Instr)->getName()->getName().substr(
              Prefix.size());

      if (!ProfiledName.ends_with(F->getName())) {
        LLVM_DEBUG(dbgs() << "PGOVerify# Intrinsic name mismatch in function "
                          << F->getName() << ": ");
        LLVM_DEBUG(Instr->print(dbgs()));
        LLVM_DEBUG(dbgs() << "\n");
        emitPGOVerifyDiagnostic(F, "IntrinsicNameMismatch",
                                "Intrinsic name mismatch: profiling " +
                                    ProfiledName.str() + " instead of " +
                                    F->getName().str());
      }
    }
  }

  // Validate counter-global loads against the containing function.
  for (auto &GV : const_cast<Module *>(F->getParent())->globals()) {

    StringRef Prefix = getInstrProfCountersVarPrefix();
    if (!GV.getName().starts_with(Prefix))
      continue;

    StringRef CounterName = GV.getName().substr(Prefix.size());

    for (const User *U : GV.users()) {
      auto *LI = dyn_cast<LoadInst>(U);
      if (!LI || LI->getFunction() != F)
        continue;

      if (!CounterName.contains(F->getName())) {
        LLVM_DEBUG(dbgs() << "PGOVerify# Counter load mismatch in function "
                          << F->getName() << ": ");
        LLVM_DEBUG(LI->print(dbgs()));
        LLVM_DEBUG(dbgs() << "\n");
        emitPGOVerifyDiagnostic(F, "CounterLoadMismatch",
                                "Counter variable mismatch: loading " +
                                    CounterName.str() + " instead of " +
                                    F->getName().str());
      }
    }
  }
}

bool IPGOVerifier::hasInstrProfUseSummary(const Module *M) const {
  if (!M)
    return false;

  Metadata *SummaryMD = M->getProfileSummary(/*IsCS=*/false);
  if (!SummaryMD)
    return false;

  std::unique_ptr<ProfileSummary> PS(ProfileSummary::getFromMD(SummaryMD));
  return PS && PS->getKind() == ProfileSummary::PSK_Instr;
}

/// Validate flow conservation where both sides are known.
void IPGOVerifier::validateBlockFrequencies(const Function *F) {
  auto CachedIt = FunctionBlockFreqInfoCache.find(F);
  if (CachedIt == FunctionBlockFreqInfoCache.end())
    return;
  const AllBlockFreqInfo &AllFreqInfo = CachedIt->second;

  for (const BasicBlock &BB : *F) {
    const Instruction *Term = BB.getTerminator();
    if (!Term)
      continue;
    if (Term->getNumSuccessors() == 0)
      continue;

    auto It = AllFreqInfo.find(&BB);
    if (It == AllFreqInfo.end())
      continue;

    const BlockFreqInfo &Info = It->second;
    // Only diagnose hard mismatches once both sides are fully known; otherwise
    // leave the block as debug-only inconclusive state.
    if (Info.numUnknownIn == 0 && Info.numUnknownOut == 0 &&
        Info.sumIn != Info.sumOut) {
      if (VerifyIPGOPrintDiagnostics)
        errs() << "PGOVerify# Block frequency mismatch in function "
               << F->getName() << ", block " << BB.getName()
               << ":  Incoming=" << Info.sumIn
               << ":  Outgoing=" << Info.sumOut << "\n";
      LLVM_DEBUG(dbgs() << "PGOVerify# Block frequency mismatch in function "
                        << F->getName() << ", block " << BB.getName()
                        << ":  Incoming=" << Info.sumIn
                        << ":  Outgoing=" << Info.sumOut << "\n");
    } else if (Info.numUnknownIn != 0 || Info.numUnknownOut != 0) {
      LLVM_DEBUG(
          dbgs() << "PGOVerify# Not able to determine Block frequency for "
                 << F->getName() << ", block " << BB.getName() << "\n");
    }
  }
}

void IPGOVerifier::validateEntryCountAgainstCallerSum(const Function *F) {
  // Skip main - it is the program entry point with no in-module callers.
  if (F->getName() == "main")
    return;

  auto MaybeEntryCount = F->getEntryCount();
  if (!MaybeEntryCount)
    return;
  uint64_t EntryCount = MaybeEntryCount->getCount();

  uint64_t Sum = 0;
  bool IsRecursive = false;
  bool HasAnyDirectCallsite = false;
  bool HasUnknownCallsiteCount = false;

  // Walk the use-def chain of F: only CallBase uses where F is the callee
  // are direct calls.  This avoids a costly triple-nested module scan.
  for (const User *U : F->users()) {
    const auto *CB = dyn_cast<CallBase>(U);
    // Skip non-call uses (e.g. address-taken, bitcast passed as argument).
    if (!CB || CB->getCalledOperand() != F)
      continue;

    const BasicBlock *BB = CB->getParent();
    if (!BB)
      continue;
    const Function *CallerFunc = BB->getParent();
    if (!CallerFunc)
      continue;

    // Detect recursion: callee is also the caller.
    if (CallerFunc == F)
      IsRecursive = true;

    HasAnyDirectCallsite = true;
    uint64_t CallsiteCount = 0;
    bool HasKnownCount = extractProfTotalWeight(*CB, CallsiteCount);

    // Fall back to cached caller block frequency when direct callsite
    // metadata is unavailable.
    if (!HasKnownCount) {
      const AllBlockFreqInfo *CallerFreq = getCachedBlockFreqInfo(CallerFunc);
      if (CallerFreq) {
        auto CallerBBIt = CallerFreq->find(BB);
        if (CallerBBIt != CallerFreq->end() &&
            CallerBBIt->second.numUnknownIn == 0) {
          CallsiteCount = CallerBBIt->second.sumIn;
          HasKnownCount = true;
        }
      }
    }

    if (!HasKnownCount) {
      HasUnknownCallsiteCount = true;
      continue;
    }

    if (CallsiteCount > std::numeric_limits<uint64_t>::max() - Sum)
      Sum = std::numeric_limits<uint64_t>::max();
    else
      Sum += CallsiteCount;
  }

  // Require complete direct-caller count visibility to avoid false positives.
  if (!HasAnyDirectCallsite || HasUnknownCallsiteCount)
    return;

  if (EntryCount == Sum)
    return;

  if (IsRecursive) {
    LLVM_DEBUG(dbgs() << "PGOVerify# EntryCount mismatch in RECURSIVE function "
                      << F->getName() << " Entry=" << EntryCount
                      << " CallerSiteSum=" << Sum
                      << " (unreliable for recursion)\n");
    emitPGOVerifyDiagnostic(
        F, "EntryCountMismatch",
        "EntryCount mismatch (recursive function): entry=" +
            std::to_string(EntryCount) + " vs caller-sum=" +
            std::to_string(Sum));
  } else {
    LLVM_DEBUG(dbgs() << "PGOVerify# Entry count mismatch in function "
                      << F->getName() << ":  Entry=" << EntryCount
                      << ":  CallerSum=" << Sum << "\n");
    emitPGOVerifyDiagnostic(F, "EntryCountMismatch",
                            "Entry count mismatch: entry=" +
                                std::to_string(EntryCount) +
                                " vs caller-sum=" + std::to_string(Sum));
  }
}
