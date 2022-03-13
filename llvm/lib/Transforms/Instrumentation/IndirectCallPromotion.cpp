//===- IndirectCallPromotion.cpp - Optimizations based on value profiling -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the transformation that promotes indirect calls to
// conditional direct calls when the indirect-call value profile metadata is
// available.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/IndirectCallPromotionAnalysis.h"
#include "llvm/Analysis/IndirectCallVisitor.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "llvm/Transforms/Utils/CallPromotionUtils.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "pgo-icall-prom"

STATISTIC(NumOfPGOICallPromotion, "Number of indirect call promotions.");
STATISTIC(NumOfPGOICallsites, "Number of indirect call candidate sites.");

// Command line option to disable indirect-call promotion with the default as
// false. This is for debug purpose.
static cl::opt<bool> DisableICP("disable-icp", cl::init(false), cl::Hidden,
                                cl::desc("Disable indirect call promotion"));

// Set the cutoff value for the promotion. If the value is other than 0, we
// stop the transformation once the total number of promotions equals the cutoff
// value.
// For debug use only.
static cl::opt<unsigned>
    ICPCutOff("icp-cutoff", cl::init(0), cl::Hidden, cl::ZeroOrMore,
              cl::desc("Max number of promotions for this compilation"));

// If ICPCSSkip is non zero, the first ICPCSSkip callsites will be skipped.
// For debug use only.
static cl::opt<unsigned>
    ICPCSSkip("icp-csskip", cl::init(0), cl::Hidden, cl::ZeroOrMore,
              cl::desc("Skip Callsite up to this number for this compilation"));

// Set if the pass is called in LTO optimization. The difference for LTO mode
// is the pass won't prefix the source module name to the internal linkage
// symbols.
static cl::opt<bool> ICPLTOMode("icp-lto", cl::init(false), cl::Hidden,
                                cl::desc("Run indirect-call promotion in LTO "
                                         "mode"));

// Set if the pass is called in SamplePGO mode. The difference for SamplePGO
// mode is it will add prof metadatato the created direct call.
static cl::opt<bool>
    ICPSamplePGOMode("icp-samplepgo", cl::init(false), cl::Hidden,
                     cl::desc("Run indirect-call promotion in SamplePGO mode"));

// If the option is set to true, only call instructions will be considered for
// transformation -- invoke instructions will be ignored.
static cl::opt<bool>
    ICPCallOnly("icp-call-only", cl::init(false), cl::Hidden,
                cl::desc("Run indirect-call promotion for call instructions "
                         "only"));

// If the option is set to true, only invoke instructions will be considered for
// transformation -- call instructions will be ignored.
static cl::opt<bool> ICPInvokeOnly("icp-invoke-only", cl::init(false),
                                   cl::Hidden,
                                   cl::desc("Run indirect-call promotion for "
                                            "invoke instruction only"));

// Dump the function level IR if the transformation happened in this
// function. For debug use only.
static cl::opt<bool>
    ICPDUMPAFTER("icp-dumpafter", cl::init(false), cl::Hidden,
                 cl::desc("Dump IR after transformation happens"));

namespace {

class PGOIndirectCallPromotionLegacyPass : public ModulePass {
public:
  static char ID;

  PGOIndirectCallPromotionLegacyPass(bool InLTO = false, bool SamplePGO = false)
      : ModulePass(ID), InLTO(InLTO), SamplePGO(SamplePGO) {
    initializePGOIndirectCallPromotionLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
  }

  StringRef getPassName() const override { return "PGOIndirectCallPromotion"; }

private:
  bool runOnModule(Module &M) override;

  // If this pass is called in LTO. We need to special handling the PGOFuncName
  // for the static variables due to LTO's internalization.
  bool InLTO;

  // If this pass is called in SamplePGO. We need to add the prof metadata to
  // the promoted direct call.
  bool SamplePGO;
};

} // end anonymous namespace

char PGOIndirectCallPromotionLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(PGOIndirectCallPromotionLegacyPass, "pgo-icall-prom",
                      "Use PGO instrumentation profile to promote indirect "
                      "calls to direct calls.",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_END(PGOIndirectCallPromotionLegacyPass, "pgo-icall-prom",
                    "Use PGO instrumentation profile to promote indirect "
                    "calls to direct calls.",
                    false, false)

ModulePass *llvm::createPGOIndirectCallPromotionLegacyPass(bool InLTO,
                                                           bool SamplePGO) {
  return new PGOIndirectCallPromotionLegacyPass(InLTO, SamplePGO);
}

namespace {

// The class for main data structure to promote indirect calls to conditional
// direct calls.
class ICallPromotionFunc {
private:
  Function &F;
  Module *M;

  // Symtab that maps indirect call profile values to function names and
  // defines.
  InstrProfSymtab *Symtab;

  bool SamplePGO;

  OptimizationRemarkEmitter &ORE;

  // A struct that records the direct target and it's call count.
  struct PromotionCandidate {
    Function *TargetFunction;
    uint64_t Count;

    PromotionCandidate(Function *F, uint64_t C) : TargetFunction(F), Count(C) {}
  };

  // Check if the indirect-call call site should be promoted. Return the number
  // of promotions. Inst is the candidate indirect call, ValueDataRef
  // contains the array of value profile data for profiled targets,
  // TotalCount is the total profiled count of call executions, and
  // NumCandidates is the number of candidate entries in ValueDataRef.
  std::vector<PromotionCandidate> getPromotionCandidatesForCallSite(
      const CallBase &CB, const ArrayRef<InstrProfValueData> &ValueDataRef,
      uint64_t TotalCount, uint32_t NumCandidates);

  // Promote a list of targets for one indirect-call callsite. Return
  // the number of promotions.
  uint32_t tryToPromote(CallBase &CB,
                        const std::vector<PromotionCandidate> &Candidates,
                        uint64_t &TotalCount);

public:
  ICallPromotionFunc(Function &Func, Module *Modu, InstrProfSymtab *Symtab,
                     bool SamplePGO, OptimizationRemarkEmitter &ORE)
      : F(Func), M(Modu), Symtab(Symtab), SamplePGO(SamplePGO), ORE(ORE) {}
  ICallPromotionFunc(const ICallPromotionFunc &) = delete;
  ICallPromotionFunc &operator=(const ICallPromotionFunc &) = delete;

  bool processFunction(ProfileSummaryInfo *PSI);
};

} // end anonymous namespace

// Indirect-call promotion heuristic. The direct targets are sorted based on
// the count. Stop at the first target that is not promoted.
std::vector<ICallPromotionFunc::PromotionCandidate>
ICallPromotionFunc::getPromotionCandidatesForCallSite(
    const CallBase &CB, const ArrayRef<InstrProfValueData> &ValueDataRef,
    uint64_t TotalCount, uint32_t NumCandidates) {
  std::vector<PromotionCandidate> Ret;

  LLVM_DEBUG(dbgs() << " \nWork on callsite #" << NumOfPGOICallsites << CB
                    << " Num_targets: " << ValueDataRef.size()
                    << " Num_candidates: " << NumCandidates << "\n");
  NumOfPGOICallsites++;
  if (ICPCSSkip != 0 && NumOfPGOICallsites <= ICPCSSkip) {
    LLVM_DEBUG(dbgs() << " Skip: User options.\n");
    return Ret;
  }

  for (uint32_t I = 0; I < NumCandidates; I++) {
    uint64_t Count = ValueDataRef[I].Count;
    assert(Count <= TotalCount);
    (void)TotalCount;
    uint64_t Target = ValueDataRef[I].Value;
    LLVM_DEBUG(dbgs() << " Candidate " << I << " Count=" << Count
                      << "  Target_func: " << Target << "\n");

    if (ICPInvokeOnly && isa<CallInst>(CB)) {
      LLVM_DEBUG(dbgs() << " Not promote: User options.\n");
      ORE.emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UserOptions", &CB)
               << " Not promote: User options";
      });
      break;
    }
    if (ICPCallOnly && isa<InvokeInst>(CB)) {
      LLVM_DEBUG(dbgs() << " Not promote: User option.\n");
      ORE.emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UserOptions", &CB)
               << " Not promote: User options";
      });
      break;
    }
    if (ICPCutOff != 0 && NumOfPGOICallPromotion >= ICPCutOff) {
      LLVM_DEBUG(dbgs() << " Not promote: Cutoff reached.\n");
      ORE.emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "CutOffReached", &CB)
               << " Not promote: Cutoff reached";
      });
      break;
    }

    // Don't promote if the symbol is not defined in the module. This avoids
    // creating a reference to a symbol that doesn't exist in the module
    // This can happen when we compile with a sample profile collected from
    // one binary but used for another, which may have profiled targets that
    // aren't used in the new binary. We might have a declaration initially in
    // the case where the symbol is globally dead in the binary and removed by
    // ThinLTO.
    Function *TargetFunction = Symtab->getFunction(Target);
    if (TargetFunction == nullptr || TargetFunction->isDeclaration()) {
      LLVM_DEBUG(dbgs() << " Not promote: Cannot find the target\n");
      ORE.emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnableToFindTarget", &CB)
               << "Cannot promote indirect call: target with md5sum "
               << ore::NV("target md5sum", Target) << " not found";
      });
      break;
    }

    const char *Reason = nullptr;
    if (!isLegalToPromote(CB, TargetFunction, &Reason)) {
      using namespace ore;

      ORE.emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "UnableToPromote", &CB)
               << "Cannot promote indirect call to "
               << NV("TargetFunction", TargetFunction) << " with count of "
               << NV("Count", Count) << ": " << Reason;
      });
      break;
    }

    Ret.push_back(PromotionCandidate(TargetFunction, Count));
    TotalCount -= Count;
  }
  return Ret;
}

CallBase &llvm::pgo::promoteIndirectCall(CallBase &CB, Function *DirectCallee,
                                         uint64_t Count, uint64_t TotalCount,
                                         bool AttachProfToDirectCall,
                                         OptimizationRemarkEmitter *ORE) {

  uint64_t ElseCount = TotalCount - Count;
  uint64_t MaxCount = (Count >= ElseCount ? Count : ElseCount);
  uint64_t Scale = calculateCountScale(MaxCount);
  MDBuilder MDB(CB.getContext());
  MDNode *BranchWeights = MDB.createBranchWeights(
      scaleBranchCount(Count, Scale), scaleBranchCount(ElseCount, Scale));

  CallBase &NewInst =
      promoteCallWithIfThenElse(CB, DirectCallee, BranchWeights);

  if (AttachProfToDirectCall) {
    MDBuilder MDB(NewInst.getContext());
    NewInst.setMetadata(
        LLVMContext::MD_prof,
        MDB.createBranchWeights({static_cast<uint32_t>(Count)}));
  }

  using namespace ore;

  if (ORE)
    ORE->emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "Promoted", &CB)
             << "Promote indirect call to " << NV("DirectCallee", DirectCallee)
             << " with count " << NV("Count", Count) << " out of "
             << NV("TotalCount", TotalCount);
    });
  return NewInst;
}

// Promote indirect-call to conditional direct-call for one callsite.
uint32_t ICallPromotionFunc::tryToPromote(
    CallBase &CB, const std::vector<PromotionCandidate> &Candidates,
    uint64_t &TotalCount) {
  uint32_t NumPromoted = 0;

  for (auto &C : Candidates) {
    uint64_t Count = C.Count;
    pgo::promoteIndirectCall(CB, C.TargetFunction, Count, TotalCount, SamplePGO,
                             &ORE);
    assert(TotalCount >= Count);
    TotalCount -= Count;
    NumOfPGOICallPromotion++;
    NumPromoted++;
  }
  return NumPromoted;
}

// Traverse all the indirect-call callsite and get the value profile
// annotation to perform indirect-call promotion.
bool ICallPromotionFunc::processFunction(ProfileSummaryInfo *PSI) {
  bool Changed = false;
  ICallPromotionAnalysis ICallAnalysis;
  for (auto *CB : findIndirectCalls(F)) {
    uint32_t NumVals, NumCandidates;
    uint64_t TotalCount;
    auto ICallProfDataRef = ICallAnalysis.getPromotionCandidatesForInstruction(
        CB, NumVals, TotalCount, NumCandidates);
    if (!NumCandidates ||
        (PSI && PSI->hasProfileSummary() && !PSI->isHotCount(TotalCount)))
      continue;
    auto PromotionCandidates = getPromotionCandidatesForCallSite(
        *CB, ICallProfDataRef, TotalCount, NumCandidates);
    uint32_t NumPromoted = tryToPromote(*CB, PromotionCandidates, TotalCount);
    if (NumPromoted == 0)
      continue;

    Changed = true;
    // Adjust the MD.prof metadata. First delete the old one.
    CB->setMetadata(LLVMContext::MD_prof, nullptr);
    // If all promoted, we don't need the MD.prof metadata.
    if (TotalCount == 0 || NumPromoted == NumVals)
      continue;
    // Otherwise we need update with the un-promoted records back.
    annotateValueSite(*M, *CB, ICallProfDataRef.slice(NumPromoted), TotalCount,
                      IPVK_IndirectCallTarget, NumCandidates);
  }
  return Changed;
}

// A wrapper function that does the actual work.
static bool promoteIndirectCalls(Module &M, ProfileSummaryInfo *PSI,
                                 bool InLTO, bool SamplePGO,
                                 ModuleAnalysisManager *AM = nullptr) {
  if (DisableICP)
    return false;
  InstrProfSymtab Symtab;
  if (Error E = Symtab.create(M, InLTO)) {
    std::string SymtabFailure = toString(std::move(E));
    M.getContext().emitError("Failed to create symtab: " + SymtabFailure);
    return false;
  }
  bool Changed = false;
  for (auto &F : M) {
    if (F.isDeclaration() || F.hasOptNone())
      continue;

    std::unique_ptr<OptimizationRemarkEmitter> OwnedORE;
    OptimizationRemarkEmitter *ORE;
    if (AM) {
      auto &FAM =
          AM->getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
      ORE = &FAM.getResult<OptimizationRemarkEmitterAnalysis>(F);
    } else {
      OwnedORE = std::make_unique<OptimizationRemarkEmitter>(&F);
      ORE = OwnedORE.get();
    }

    ICallPromotionFunc ICallPromotion(F, &M, &Symtab, SamplePGO, *ORE);
    bool FuncChanged = ICallPromotion.processFunction(PSI);
    if (ICPDUMPAFTER && FuncChanged) {
      LLVM_DEBUG(dbgs() << "\n== IR Dump After =="; F.print(dbgs()));
      LLVM_DEBUG(dbgs() << "\n");
    }
    Changed |= FuncChanged;
    if (ICPCutOff != 0 && NumOfPGOICallPromotion >= ICPCutOff) {
      LLVM_DEBUG(dbgs() << " Stop: Cutoff reached.\n");
      break;
    }
  }
  return Changed;
}

bool PGOIndirectCallPromotionLegacyPass::runOnModule(Module &M) {
  ProfileSummaryInfo *PSI =
      &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();

  // Command-line option has the priority for InLTO.
  return promoteIndirectCalls(M, PSI, InLTO | ICPLTOMode,
                              SamplePGO | ICPSamplePGOMode);
}

PreservedAnalyses PGOIndirectCallPromotion::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  ProfileSummaryInfo *PSI = &AM.getResult<ProfileSummaryAnalysis>(M);

  if (!promoteIndirectCalls(M, PSI, InLTO | ICPLTOMode,
                            SamplePGO | ICPSamplePGOMode, &AM))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
