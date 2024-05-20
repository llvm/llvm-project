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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/IndirectCallPromotionAnalysis.h"
#include "llvm/Analysis/IndirectCallVisitor.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TypeMetadataUtils.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Value.h"
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

extern cl::opt<unsigned> MaxNumVTableAnnotations;

// Command line option to disable indirect-call promotion with the default as
// false. This is for debug purpose.
static cl::opt<bool> DisableICP("disable-icp", cl::init(false), cl::Hidden,
                                cl::desc("Disable indirect call promotion"));

// Set the cutoff value for the promotion. If the value is other than 0, we
// stop the transformation once the total number of promotions equals the cutoff
// value.
// For debug use only.
static cl::opt<unsigned>
    ICPCutOff("icp-cutoff", cl::init(0), cl::Hidden,
              cl::desc("Max number of promotions for this compilation"));

// If ICPCSSkip is non zero, the first ICPCSSkip callsites will be skipped.
// For debug use only.
static cl::opt<unsigned>
    ICPCSSkip("icp-csskip", cl::init(0), cl::Hidden,
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

// This option is meant to be used by LLVM regression test and test the
// transformation that compares vtables.
// TODO: ICP pass will do cost-benefit analysis between function-based
// comparison and vtable-based comparison and choose one of the two
// transformations.
static cl::opt<bool> ICPEnableVTableCmp(
    "icp-enable-vtable-cmp", cl::init(false), cl::Hidden,
    cl::desc("If ThinLTO and WPD is enabled and this option is true, "
             "indirect-call promotion pass will compare vtables rather than "
             "functions for speculative devirtualization of virtual calls."
             " If set to false, indirect-call promotion pass will always "
             "compare functions."));

namespace {

using VTableAddressPointOffsetValMap =
    SmallDenseMap<const GlobalVariable *, SmallDenseMap<int, Constant *, 4>, 8>;

// A struct to collect type information for a virtual call site.
struct VirtualCallSiteInfo {
  // The offset from the address point to virtual function in the vtable.
  uint64_t FunctionOffset;
  // The instruction that computes the address point of vtable.
  Instruction *VPtr;
  // The compatible type used in LLVM type intrinsics.
  StringRef CompatibleTypeStr;
};

// The key is a virtual call, and value is its type information.
using VirtualCallSiteTypeInfoMap =
    SmallDenseMap<const CallBase *, VirtualCallSiteInfo, 8>;

// Given the list of compatible type metadata for a vtable and one specified
// type, returns the address point offset of the type if any.
static std::optional<uint64_t>
getCompatibleTypeOffset(const ArrayRef<MDNode *> &Types,
                        StringRef CompatibleType) {
  if (Types.empty()) {
    return std::nullopt;
  }
  std::optional<uint64_t> Offset;
  // find the offset where type string is equal to the one in llvm.type.test
  // intrinsic
  for (MDNode *Type : Types) {
    auto TypeIDMetadata = Type->getOperand(1).get();
    if (auto *TypeId = dyn_cast<MDString>(TypeIDMetadata)) {
      StringRef TypeStr = TypeId->getString();
      if (TypeStr != CompatibleType) {
        continue;
      }
      Offset = cast<ConstantInt>(
                   cast<ConstantAsMetadata>(Type->getOperand(0))->getValue())
                   ->getZExtValue();
      break;
    }
  }
  return Offset;
}

// Returns a constant representing the vtable's address point specified by the
// offset.
static Constant *getVTableAddressPointOffset(GlobalVariable *VTable,
                                             uint32_t AddressPointOffset) {
  Module &M = *VTable->getParent();
  LLVMContext &Context = M.getContext();
  assert(AddressPointOffset <
             M.getDataLayout().getTypeAllocSize(VTable->getValueType()) &&
         "Out-of-bound access");

  return ConstantExpr::getInBoundsGetElementPtr(
      Type::getInt8Ty(Context), VTable,
      llvm::ConstantInt::get(Type::getInt32Ty(Context), AddressPointOffset));
}

// Promote indirect calls to conditional direct calls, keeping track of
// thresholds.
class IndirectCallPromoter {
private:
  Function &F;
  Module &M;

  // Symtab that maps indirect call profile values to function names and
  // defines.
  InstrProfSymtab *const Symtab;

  const bool SamplePGO;

  // A map from a virtual call to its type information.
  const VirtualCallSiteTypeInfoMap &VirtualCSInfo;

  VTableAddressPointOffsetValMap &VTableAddressPointOffsetVal;

  OptimizationRemarkEmitter &ORE;

  // A struct that records the direct target and it's call count.
  struct PromotionCandidate {
    Function *const TargetFunction;
    const uint64_t Count;

    uint64_t FunctionOffset;

    SmallVector<std::pair<uint64_t, uint64_t>, 2> VTableGUIDAndCounts;

    SmallVector<Constant *, 2> AddressPoints;

    PromotionCandidate(Function *F, uint64_t C) : TargetFunction(F), Count(C) {}
  };

  using VTableGUIDCountsMap = SmallDenseMap<uint64_t, uint64_t, 4>;

  // Check if the indirect-call call site should be promoted. Return the number
  // of promotions. Inst is the candidate indirect call, ValueDataRef
  // contains the array of value profile data for profiled targets,
  // TotalCount is the total profiled count of call executions, and
  // NumCandidates is the number of candidate entries in ValueDataRef.
  std::vector<PromotionCandidate> getPromotionCandidatesForCallSite(
      const CallBase &CB, const ArrayRef<InstrProfValueData> &ValueDataRef,
      uint64_t TotalCount, uint32_t NumCandidates,
      VTableGUIDCountsMap &VTableGUIDCounts);

  // Promote a list of targets for one indirect-call callsite by comparing
  // indirect callee with functions. Returns true if there are IR
  // transformations and false otherwise.
  bool tryToPromoteWithFuncCmp(
      CallBase &CB, const std::vector<PromotionCandidate> &Candidates,
      uint64_t TotalCount, ArrayRef<InstrProfValueData> ICallProfDataRef,
      uint32_t NumCandidates);

  bool tryToPromoteWithVTableCmp(
      CallBase &CB, const std::vector<PromotionCandidate> &Candidates,
      uint64_t TotalFuncCount, uint32_t NumCandidates,
      MutableArrayRef<InstrProfValueData> ICallProfDataRef,
      VTableGUIDCountsMap &VTableGUIDCounts);

  void
  tryGetVTableInfos(const CallBase &CB,
                    const SmallDenseMap<Function *, int, 4> &CalleeIndexMap,
                    VTableGUIDCountsMap &VTableGUIDCounts,
                    std::vector<PromotionCandidate> &Candidates);

  Constant *getOrCreateVTableAddressPointVar(GlobalVariable *GV,
                                             uint64_t AddressPointOffset);

  bool isProfitableToCompareVTables(
      const std::vector<PromotionCandidate> &Candidates, uint64_t TotalCount);

public:
  IndirectCallPromoter(
      Function &Func, Module &M, InstrProfSymtab *Symtab, bool SamplePGO,
      const VirtualCallSiteTypeInfoMap &VirtualCSInfo,
      VTableAddressPointOffsetValMap &VTableAddressPointOffsetVal,
      OptimizationRemarkEmitter &ORE)
      : F(Func), M(M), Symtab(Symtab), SamplePGO(SamplePGO),
        VirtualCSInfo(VirtualCSInfo),
        VTableAddressPointOffsetVal(VTableAddressPointOffsetVal), ORE(ORE) {}
  IndirectCallPromoter(const IndirectCallPromoter &) = delete;
  IndirectCallPromoter &operator=(const IndirectCallPromoter &) = delete;

  bool processFunction(ProfileSummaryInfo *PSI);
};

} // end anonymous namespace

// Indirect-call promotion heuristic. The direct targets are sorted based on
// the count. Stop at the first target that is not promoted.
std::vector<IndirectCallPromoter::PromotionCandidate>
IndirectCallPromoter::getPromotionCandidatesForCallSite(
    const CallBase &CB, const ArrayRef<InstrProfValueData> &ValueDataRef,
    uint64_t TotalCount, uint32_t NumCandidates,
    VTableGUIDCountsMap &VTableGUIDCounts) {
  std::vector<PromotionCandidate> Ret;

  SmallDenseMap<Function *, int, 4> CalleeIndexMap;

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

    CalleeIndexMap[TargetFunction] = Ret.size();
    Ret.push_back(PromotionCandidate(TargetFunction, Count));

    TotalCount -= Count;
  }

  if (!ICPEnableVTableCmp)
    return Ret;

  tryGetVTableInfos(CB, CalleeIndexMap, VTableGUIDCounts, Ret);

  return Ret;
}

Constant *IndirectCallPromoter::getOrCreateVTableAddressPointVar(
    GlobalVariable *GV, uint64_t AddressPointOffset) {
  Constant *Var = VTableAddressPointOffsetVal[GV][AddressPointOffset];
  if (Var != nullptr)
    return Var;
  Constant *Ret = getVTableAddressPointOffset(GV, AddressPointOffset);
  VTableAddressPointOffsetVal[GV][AddressPointOffset] = Ret;
  return Ret;
}

void IndirectCallPromoter::tryGetVTableInfos(
    const CallBase &CB, const SmallDenseMap<Function *, int, 4> &CalleeIndexMap,
    VTableGUIDCountsMap &GUIDCountsMap,
    std::vector<PromotionCandidate> &Candidates) {
  if (!ICPEnableVTableCmp)
    return;

  auto Iter = VirtualCSInfo.find(&CB);
  if (Iter == VirtualCSInfo.end())
    return;

  auto &VirtualCallInfo = Iter->second;

  uint32_t ActualNumValueData = 0;

  uint64_t TotalVTableCount = 0;
  auto VTableValueDataArray = getValueProfDataFromInst(
      *VirtualCallInfo.VPtr, IPVK_VTableTarget, MaxNumVTableAnnotations,
      ActualNumValueData, TotalVTableCount);

  if (VTableValueDataArray.get() == nullptr)
    return;

  SmallVector<MDNode *, 2> Types; // type metadata associated with a vtable.
  // Compute the functions and counts from by each vtable.
  for (size_t j = 0; j < ActualNumValueData; j++) {
    uint64_t VTableVal = VTableValueDataArray[j].Value;
    GUIDCountsMap[VTableVal] = VTableValueDataArray[j].Count;
    GlobalVariable *VTableVariable = Symtab->getGlobalVariable(VTableVal);
    if (!VTableVariable) {
      LLVM_DEBUG(dbgs() << "\tCannot find vtable definition for " << VTableVal
                        << "\n");
      continue;
    }

    Types.clear();
    VTableVariable->getMetadata(LLVMContext::MD_type, Types);
    std::optional<uint64_t> MaybeAddressPointOffset =
        getCompatibleTypeOffset(Types, VirtualCallInfo.CompatibleTypeStr);
    if (!MaybeAddressPointOffset)
      continue;

    const uint64_t AddressPointOffset = *MaybeAddressPointOffset;

    Function *Callee = nullptr;

    std::tie(Callee, std::ignore) = getFunctionAtVTableOffset(
        VTableVariable, AddressPointOffset + VirtualCallInfo.FunctionOffset,
        *(F.getParent()));
    if (!Callee)
      continue;

    auto CalleeIndexIter = CalleeIndexMap.find(Callee);
    if (CalleeIndexIter == CalleeIndexMap.end())
      continue;

    auto &Candidate = Candidates[CalleeIndexIter->second];
    Candidate.VTableGUIDAndCounts.push_back(
        {VTableVal, VTableValueDataArray[j].Count});
    Candidate.AddressPoints.push_back(
        getOrCreateVTableAddressPointVar(VTableVariable, AddressPointOffset));
  }
}

static MDNode *getBranchWeights(LLVMContext &Context, uint64_t IfCount,
                                uint64_t ElseCount) {
  MDBuilder MDB(Context);
  uint64_t Scale = calculateCountScale(std::max(IfCount, ElseCount));
  return MDB.createBranchWeights(scaleBranchCount(IfCount, Scale),
                                 scaleBranchCount(ElseCount, Scale));
}

CallBase &llvm::pgo::promoteIndirectCall(CallBase &CB, Function *DirectCallee,
                                         uint64_t Count, uint64_t TotalCount,
                                         bool AttachProfToDirectCall,
                                         OptimizationRemarkEmitter *ORE) {
  MDNode *BranchWeights =
      getBranchWeights(CB.getContext(), Count, TotalCount - Count);

  CallBase &NewInst =
      promoteCallWithIfThenElse(CB, DirectCallee, BranchWeights);

  if (AttachProfToDirectCall)
    setBranchWeights(NewInst, {static_cast<uint32_t>(Count)});

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
bool IndirectCallPromoter::tryToPromoteWithFuncCmp(
    CallBase &CB, const std::vector<PromotionCandidate> &Candidates,
    uint64_t TotalCount, ArrayRef<InstrProfValueData> ICallProfDataRef,
    uint32_t NumCandidates) {
  uint32_t NumPromoted = 0;

  for (const auto &C : Candidates) {
    uint64_t Count = C.Count;
    pgo::promoteIndirectCall(CB, C.TargetFunction, Count, TotalCount, SamplePGO,
                             &ORE);
    assert(TotalCount >= Count);
    TotalCount -= Count;
    NumOfPGOICallPromotion++;
    NumPromoted++;
  }

  const bool Changed = (NumPromoted != 0);

  if (Changed) {
    CB.setMetadata(LLVMContext::MD_prof, nullptr);

    if (TotalCount != 0)
      annotateValueSite(*F.getParent(), CB, ICallProfDataRef.slice(NumPromoted),
                        TotalCount, IPVK_IndirectCallTarget, NumCandidates);
  }

  return Changed;
}

bool IndirectCallPromoter::tryToPromoteWithVTableCmp(
    CallBase &CB, const std::vector<PromotionCandidate> &Candidates,
    uint64_t TotalFuncCount, uint32_t NumCandidates,
    MutableArrayRef<InstrProfValueData> ICallProfDataRef,
    VTableGUIDCountsMap &VTableGUIDCounts) {
  Instruction *VPtr = VirtualCSInfo.at(&CB).VPtr;

  SmallVector<int, 4> PromotedFuncCount;
  for (const auto &Candidate : Candidates) {
    uint64_t IfCount = 0;
    // FIXME: Skip vtables with cold count in the comparison.
    for (auto &[GUID, Count] : Candidate.VTableGUIDAndCounts) {
      IfCount += Count;
      VTableGUIDCounts[GUID] -= Count;
    }

    promoteCallWithVTableCmp(
        CB, VPtr, Candidate.TargetFunction, Candidate.AddressPoints,
        getBranchWeights(CB.getContext(), IfCount, TotalFuncCount - IfCount));

    PromotedFuncCount.push_back(IfCount);

    TotalFuncCount -= IfCount;
    NumOfPGOICallPromotion++;
  }

  if (PromotedFuncCount.empty())
    return false;

  // A comparator that sorts value profile data descendingly.
  auto Cmp = [](const InstrProfValueData &LHS, const InstrProfValueData &RHS) {
    return LHS.Count > RHS.Count;
  };

  CB.setMetadata(LLVMContext::MD_prof, nullptr);
  // Update indirect call value profiles if total count of the call site is not
  // zero.
  if (TotalFuncCount != 0) {
    for (size_t I = 0; I < PromotedFuncCount.size(); I++)
      ICallProfDataRef[I].Count -= PromotedFuncCount[I];

    llvm::sort(ICallProfDataRef.begin(), ICallProfDataRef.end(), Cmp);

    // Locate the first <target, count> pair where the count is zero or less.
    auto UB = llvm::upper_bound(
        ICallProfDataRef, 0U,
        [](uint64_t Count, const InstrProfValueData &ProfData) {
          return ProfData.Count <= Count;
        });

    ArrayRef<InstrProfValueData> VDs(ICallProfDataRef.begin(), UB);
    annotateValueSite(M, CB, VDs, TotalFuncCount, IPVK_IndirectCallTarget,
                      NumCandidates);
  }

  VPtr->setMetadata(LLVMContext::MD_prof, nullptr);
  std::vector<InstrProfValueData> VTableValueProfiles;
  uint64_t TotalVTableCount = 0;
  for (auto [GUID, Count] : VTableGUIDCounts) {
    if (Count == 0)
      continue;

    VTableValueProfiles.push_back({GUID, Count});
    TotalVTableCount += Count;
  }
  llvm::sort(VTableValueProfiles, Cmp);

  annotateValueSite(M, *VPtr, VTableValueProfiles, TotalVTableCount,
                    IPVK_VTableTarget, VTableValueProfiles.size());

  // Update vtable profile metadata
  return true;
}

// Traverse all the indirect-call callsite and get the value profile
// annotation to perform indirect-call promotion.
bool IndirectCallPromoter::processFunction(ProfileSummaryInfo *PSI) {
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
    VTableGUIDCountsMap VTableGUIDCounts;
    auto PromotionCandidates = getPromotionCandidatesForCallSite(
        *CB, ICallProfDataRef, TotalCount, NumCandidates, VTableGUIDCounts);

    if (isProfitableToCompareVTables(PromotionCandidates, TotalCount))
      Changed |= tryToPromoteWithVTableCmp(*CB, PromotionCandidates, TotalCount,
                                           NumCandidates, ICallProfDataRef,
                                           VTableGUIDCounts);
    else
      Changed |= tryToPromoteWithFuncCmp(*CB, PromotionCandidates, TotalCount,
                                         ICallProfDataRef, NumCandidates);
  }
  return Changed;
}

bool IndirectCallPromoter::isProfitableToCompareVTables(
    const std::vector<PromotionCandidate> &Candidates, uint64_t TotalCount) {
  if (!ICPEnableVTableCmp)
    return false;

  // FIXME: Implement cost-benefit analysis in a follow-up change.
  return true;
}

static void
computeVirtualCallSiteTypeInfoMap(Module &M, ModuleAnalysisManager &MAM,
                                  VirtualCallSiteTypeInfoMap &VirtualCSInfo) {
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupDomTree = [&FAM](Function &F) -> DominatorTree & {
    return FAM.getResult<DominatorTreeAnalysis>(F);
  };

  // Right now only llvm.type.test is used to find out virtual call sites.
  // With ThinLTO and whole-program-devirtualization, llvm.type.test and
  // llvm.public.type.test are emitted, and llvm.public.type.test is either
  // refined to llvm.type.test or dropped before indirect-call-promotion pass.
  //
  // FIXME: For fullLTO with VFE, `llvm.type.checked.load intrinsic` is emitted.
  // Find out virtual calls by looking at users of llvm.type.checked.load in
  // that case.
  Function *TypeTestFunc =
      M.getFunction(Intrinsic::getName(Intrinsic::type_test));

  if (!TypeTestFunc || TypeTestFunc->use_empty())
    return;

  // Iterate all type.test calls and find all indirect calls.
  for (Use &U : llvm::make_early_inc_range(TypeTestFunc->uses())) {
    auto *CI = dyn_cast<CallInst>(U.getUser());
    if (!CI)
      continue;

    auto *TypeMDVal = cast<MetadataAsValue>(CI->getArgOperand(1));
    if (!TypeMDVal)
      continue;

    auto *CompatibleTypeId = dyn_cast<MDString>(TypeMDVal->getMetadata());
    if (!CompatibleTypeId)
      continue;

    StringRef CompatibleTypeStr = CompatibleTypeId->getString();

    // Find out all devirtualizable call sites given a llvm.type.test intrinsic
    // call.
    SmallVector<DevirtCallSite, 1> DevirtCalls;
    SmallVector<CallInst *, 1> Assumes;
    auto &DT = LookupDomTree(*CI->getFunction());
    findDevirtualizableCallsForTypeTest(DevirtCalls, Assumes, CI, DT);

    // type-id, offset from the address point
    // combined with type metadata to compute function offset
    for (auto &DevirtCall : DevirtCalls) {
      CallBase &CB = DevirtCall.CB;
      // This is the offset from the address point offset to the virtual
      // function.
      uint64_t Offset = DevirtCall.Offset;

      // Given an indirect call, try find the instruction which loads a pointer
      // to virtual table.
      Instruction *VTablePtr =
          PGOIndirectCallVisitor::tryGetVTableInstruction(&CB);

      if (!VTablePtr)
        continue;

      VirtualCSInfo[&CB] = {Offset, VTablePtr, CompatibleTypeStr};
    }
  }
}

// A wrapper function that does the actual work.
static bool promoteIndirectCalls(Module &M, ProfileSummaryInfo *PSI, bool InLTO,
                                 bool SamplePGO, ModuleAnalysisManager &MAM) {
  if (DisableICP)
    return false;
  InstrProfSymtab Symtab;
  if (Error E = Symtab.create(M, InLTO)) {
    std::string SymtabFailure = toString(std::move(E));
    M.getContext().emitError("Failed to create symtab: " + SymtabFailure);
    return false;
  }
  bool Changed = false;
  VirtualCallSiteTypeInfoMap VirtualCSInfo;

  computeVirtualCallSiteTypeInfoMap(M, MAM, VirtualCSInfo);

  // This map records states across functions in an LLVM IR module.
  // IndirectCallPromoter processes one
  // function at a time and updates this map with new entries the first time
  // the entry is needed in the module; the subsequent functions could re-use
  // map entries inserted when processing prior functions.
  VTableAddressPointOffsetValMap VTableAddressPointOffsetVal;

  for (auto &F : M) {
    if (F.isDeclaration() || F.hasOptNone())
      continue;

    auto &FAM =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(F);

    IndirectCallPromoter CallPromoter(F, M, &Symtab, SamplePGO, VirtualCSInfo,
                                      VTableAddressPointOffsetVal, ORE);
    bool FuncChanged = CallPromoter.processFunction(PSI);
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

PreservedAnalyses PGOIndirectCallPromotion::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  ProfileSummaryInfo *PSI = &MAM.getResult<ProfileSummaryAnalysis>(M);

  if (!promoteIndirectCalls(M, PSI, InLTO | ICPLTOMode,
                            SamplePGO | ICPSamplePGOMode, MAM))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
