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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
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
#include <optional>
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

static cl::opt<bool> EnableVTableProm("enable-vtable-prom", cl::init(false),
                                      cl::Hidden,
                                      cl::desc("Enable vtable prom"));

static cl::opt<int>
    MaxNumAdditionalOffset("max-num-additional-offset", cl::init(0), cl::Hidden,
                           cl::desc("The max number of additional offset"));

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

// namespace {

// Promote indirect calls to conditional direct calls, keeping track of
// thresholds.
class IndirectCallPromoter {
public:
  struct VirtualCallInfo {
    uint64_t
        Offset; // The byte offset from address point to the virtual function.
    Instruction *I;              // The vptr instruction
    StringRef CompatibleTypeStr; // Compatible type str
    Instruction *TypeTestInstr;  // The type.test intrinsic
  };

private:
  // 24 is the maximum number of counters per instrumented value.
  static constexpr int MaxNumVTableToConsider = 24;
  Function &F;

  // Symtab that maps indirect call profile values to function names and
  // defines.
  InstrProfSymtab *const Symtab;

  const DenseMap<const CallBase *, VirtualCallInfo> &CBToVirtualCallInfoMap;

  const bool SamplePGO;

  OptimizationRemarkEmitter &ORE;

  // A struct that records the direct target and it's call count.
  struct PromotionCandidate {
    Function *const TargetFunction;
    const uint64_t Count;

    PromotionCandidate(Function *F, uint64_t C) : TargetFunction(F), Count(C) {}
  };

  // A helper function that transforms CB (indirect call) to a conditional call
  // to TargetFunction.
  // Inputs:
  // - VTableIndices collects the indices of elements in VTableCandidates whose
  // function is the TargetFunction.
  // - VTableOffsetToValueMap
  //   The key is address point offset, and value is the offset variable.
  // Outputs:
  // - TotalVTableCount is updated to subtract the count of TargetFunction.
  // - VTablePromotedSet adds TargetFunction into the set.
  // Returns the promoted direct call instruction.
  CallBase &promoteIndirectCallBasedOnVTable(
      CallBase &CB, Function *TargetFunction,
      const SmallVector<VTableCandidate> &VTableCandidates,
      const std::vector<int> &VTableIndices,
      const std::unordered_map<int, Value *> &VTableOffsetToValueMap,
      uint64_t &TotalVTableCount,
      SmallPtrSet<Function *, 4> &VTablePromotedSet);

  struct PerFunctionCandidateInfo {
    std::vector<int> VTableIndices;
    SetVector<int> Offsets;
  };

  // Does cost benefit analysis between comparing functions and comparing
  // vtables. Returns true if comparing vtable is more efficient and false
  // otherwise.
  bool shouldCompareVTable(
      CallBase &CB, const std::vector<PromotionCandidate> &Candidates,
      const SmallVector<VTableCandidate> &VTableCandidates,
      std::vector<PerFunctionCandidateInfo> &PerFunctionCandiateInfo);

  // Check if the indirect-call call site should be promoted. Return the number
  // of promotions. Inst is the candidate indirect call, ValueDataRef
  // contains the array of value profile data for profiled targets,
  // TotalCount is the total profiled count of call executions, and
  // NumCandidates is the number of candidate entries in ValueDataRef.
  std::vector<PromotionCandidate> getPromotionCandidatesForCallSite(
      const CallBase &CB, const ArrayRef<InstrProfValueData> &ValueDataRef,
      uint64_t TotalCount, uint32_t NumCandidates);

  uint32_t promoteIndirectCallsByComparingFunctions(
      CallBase &CB, const std::vector<PromotionCandidate> &Candidates,
      uint64_t &TotalCount, bool AttachProfToDirectCall,
      OptimizationRemarkEmitter *ORE);

  // Promote a list of targets for one indirect-call callsite. Return
  // the number of promotions.
  uint32_t tryToPromote(CallBase &CB,
                        const std::vector<PromotionCandidate> &Candidates,
                        uint64_t &TotalCount,
                        const SmallVector<VTableCandidate> &VTableCandidates,
                        uint64_t &TotalVTableCount);

  // For indirect call 'CB', find the list of vtable candidates where callees
  // are loaded from. Returns false if the callee is not loaded from virtual
  // tables.
  bool getVTableCandidates(CallBase *CB,
                           SmallVector<VTableCandidate> &VTableCandidates,
                           uint64_t &TotalVTableCount);

public:
  IndirectCallPromoter(
      Function &Func, InstrProfSymtab *Symtab, bool SamplePGO,
      const DenseMap<const CallBase *, VirtualCallInfo> &CBToVirtualCallInfoMap,
      OptimizationRemarkEmitter &ORE)
      : F(Func), Symtab(Symtab), CBToVirtualCallInfoMap(CBToVirtualCallInfoMap),
        SamplePGO(SamplePGO), ORE(ORE) {}
  IndirectCallPromoter(const IndirectCallPromoter &) = delete;
  IndirectCallPromoter &operator=(const IndirectCallPromoter &) = delete;

  bool processFunction(ProfileSummaryInfo *PSI);
};

//} // end anonymous namespace

static std::optional<uint64_t>
getCompatibleTypeOffset(const SmallVector<MDNode *, 2> &Types,
                        StringRef CompatibleType) {
  std::optional<uint64_t> Offset = std::nullopt;
  for (MDNode *Type : Types) {
    auto TypeIDMetadata = Type->getOperand(1).get();
    if (auto *TypeId = dyn_cast<MDString>(TypeIDMetadata)) {
      if (TypeId->getString() != CompatibleType) {
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

static Function *getFunctionAtVTableOffset(GlobalVariable *GV, uint64_t Offset,
                                           Module &M) {
  Constant *Ptr = getPointerAtOffset(GV->getInitializer(), Offset, M, GV);
  if (!Ptr)
    return nullptr;

  auto C = Ptr->stripPointerCasts();
  auto Fn = dyn_cast<Function>(C);
  auto A = dyn_cast<GlobalAlias>(C);
  if (!Fn && A)
    Fn = dyn_cast<Function>(A->getAliasee());
  return Fn;
}

bool IndirectCallPromoter::getVTableCandidates(
    CallBase *CB, SmallVector<VTableCandidate> &VTableCandidates,
    uint64_t &TotalVTableCount) {
  VTableCandidates.clear();
  // CB doesn't have virtual call info. This is possible, for example,
  // when the indirect callee is a function pointer.
  auto VirtualCallInfoIter = CBToVirtualCallInfoMap.find(CB);
  if (VirtualCallInfoIter == CBToVirtualCallInfoMap.end())
    return false;

  auto &VirtualCallInfo = VirtualCallInfoIter->second;

  Instruction *VTablePtr = VirtualCallInfo.I;
  StringRef CompatibleTypeStr = VirtualCallInfo.CompatibleTypeStr;

  std::unique_ptr<InstrProfValueData[]> VTableArray =
      std::make_unique<InstrProfValueData[]>(MaxNumVTableToConsider);
  uint32_t ActualNumValueData = 0;
  // Find out all vtables with callees in candidate sets, and their counts.
  bool Res = getValueProfDataFromInst(*VTablePtr, IPVK_VTableTarget,
                                      MaxNumVTableToConsider, VTableArray.get(),
                                      ActualNumValueData, TotalVTableCount);
  if (!Res || ActualNumValueData == 0)
    return false;

  SmallVector<MDNode *, 2> Types; // type metadata associated with a vtable.

  // Compute the functions and counts contributed by each vtable.
  for (uint32_t j = 0; j < ActualNumValueData; j++) {
    const uint64_t VTableVal = VTableArray[j].Value;
    GlobalVariable *VTableVariable = Symtab->getGlobalVariable(VTableVal);
    if (!VTableVariable) {
      LLVM_DEBUG(dbgs() << "No vtable definition for " << VTableVal
                        << " from callsite " << (*CB) << "\n");
      continue;
    }

    Types.clear();
    VTableVariable->getMetadata(LLVMContext::MD_type, Types);
    std::optional<uint64_t> MaybeOffset =
        getCompatibleTypeOffset(Types, CompatibleTypeStr);
    if (!MaybeOffset) {
      LLVM_DEBUG(dbgs() << "Cannot compute compatible type offset "
                        << CompatibleTypeStr << "\t" << *VTableVariable
                        << "\n");
      continue;
    }

    const uint64_t FuncByteOffset = (*MaybeOffset) + VirtualCallInfo.Offset;
    Function *Callee = getFunctionAtVTableOffset(VTableVariable, FuncByteOffset,
                                                 *(F.getParent()));
    if (!Callee) {
      LLVM_DEBUG(dbgs() << "Cannot find callee at offset " << FuncByteOffset
                        << " in vtable " << *VTableVariable << "\n");
      continue;
    }

    VTableCandidates.push_back({VTablePtr, VTableVariable, *MaybeOffset, Callee,
                                VTableArray[j].Count});
  }

  sort(VTableCandidates.begin(), VTableCandidates.end(),
       [](const VTableCandidate &LHS, const VTableCandidate &RHS) {
         return LHS.VTableValCount > RHS.VTableValCount;
       });

  return true;
}

CallBase &IndirectCallPromoter::promoteIndirectCallBasedOnVTable(
    CallBase &CB, Function *TargetFunction,
    const SmallVector<VTableCandidate> &VTableCandidates,
    const std::vector<int> &VTableIndices,
    const std::unordered_map<int /*address-point-offset*/, Value *>
        &VTableOffsetToValueMap,
    uint64_t &TotalVTableCount, SmallPtrSet<Function *, 4> &VTablePromotedSet) {
  uint64_t IfCount = 0;
  for (auto Index : VTableIndices) {
    IfCount += VTableCandidates[Index].VTableValCount;
  }
  uint64_t ElseCount = TotalVTableCount - IfCount;
  uint64_t MaxCount = (IfCount >= ElseCount ? IfCount : ElseCount);
  uint64_t Scale = calculateCountScale(MaxCount);
  MDBuilder MDB(CB.getContext());
  MDNode *BranchWeights = MDB.createBranchWeights(
      scaleBranchCount(IfCount, Scale), scaleBranchCount(ElseCount, Scale));
  uint64_t SumPromotedVTableCount = 0;
  CallBase &NewInst = promoteIndirectCallWithVTableInfo(
      CB, TargetFunction, VTableCandidates, VTableIndices,
      VTableOffsetToValueMap, SumPromotedVTableCount, BranchWeights);
  TotalVTableCount -= SumPromotedVTableCount;
  VTablePromotedSet.insert(TargetFunction);

  promoteCall(NewInst, TargetFunction, nullptr, true);
  return NewInst;
}

bool IndirectCallPromoter::shouldCompareVTable(
    CallBase &CB, const std::vector<PromotionCandidate> &Candidates,
    const SmallVector<VTableCandidate> &VTableCandidates,
    std::vector<PerFunctionCandidateInfo> &PerFunctionCandidateInfo) {
  if (!EnableVTableProm)
    return false;

  assert(PerFunctionCandidateInfo.empty() &&
         "Expect empty PerFunctionCandidateInfo");
  PerFunctionCandidateInfo.resize(Candidates.size());
  SmallDenseMap<Function *, int, 4> FunctionToIndexMap;
  for (int i = 0, size = Candidates.size(); i < size; i++) {
    auto &Candidate = Candidates[i];
    assert(FunctionToIndexMap.find(Candidate.TargetFunction) ==
               FunctionToIndexMap.end() &&
           "Expect unique functions");
    FunctionToIndexMap[Candidate.TargetFunction] = i;
  }
  for (int i = 0, size = VTableCandidates.size(); i < size; i++) {
    VTableCandidate C = VTableCandidates[i];
    auto iter = FunctionToIndexMap.find(C.TargetFunction);
    if (iter == FunctionToIndexMap.end())
      continue;

    PerFunctionCandidateInfo[iter->second].VTableIndices.push_back(i);
  }

  auto computeOffsets =
      [&VTableCandidates](const std::vector<int> &VTableIndices,
                          SetVector<int> &Offsets) {
        for (auto Index : VTableIndices) {
          Offsets.insert(VTableCandidates[Index].AddressPointOffset);
        }
      };

  for (auto &CandidateInfo : PerFunctionCandidateInfo) {
    computeOffsets(CandidateInfo.VTableIndices, CandidateInfo.Offsets);
  }

  int Offset = -1;
  bool EachCandiateFuncUniqueVTable = true;
  bool AllVTablesHaveSameOffset = true;
  for (int i = 0, size = PerFunctionCandidateInfo.size(); i < size; i++) {
    if (PerFunctionCandidateInfo[i].VTableIndices.size() != 1)
      EachCandiateFuncUniqueVTable = false;

    if (PerFunctionCandidateInfo[i].Offsets.size() != 1) {
      AllVTablesHaveSameOffset = false;
    } else {
      if (Offset == -1) {
        Offset = PerFunctionCandidateInfo[i].Offsets[0];
      } else if (Offset != PerFunctionCandidateInfo[i].Offsets[0]) {
        AllVTablesHaveSameOffset = false;
      }
    }
  }

  if (!AllVTablesHaveSameOffset || !EachCandiateFuncUniqueVTable) {
    return false;
  }

  return true;
}

// Indirect-call promotion heuristic. The direct targets are sorted based on
// the count. Stop at the first target that is not promoted.
std::vector<IndirectCallPromoter::PromotionCandidate>
IndirectCallPromoter::getPromotionCandidatesForCallSite(
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

uint32_t IndirectCallPromoter::promoteIndirectCallsByComparingFunctions(
    CallBase &CB, const std::vector<PromotionCandidate> &Candidates,
    uint64_t &TotalCount, bool AttachProfToDirectCall,
    OptimizationRemarkEmitter *ORE) {
  uint32_t NumPromoted = 0;

  for (const auto &C : Candidates) {
    uint64_t Count = C.Count;
    pgo::promoteIndirectCall(CB, C.TargetFunction, Count, TotalCount,
                             AttachProfToDirectCall, ORE);
    assert(TotalCount >= Count);
    TotalCount -= Count;
    NumOfPGOICallPromotion++;
    NumPromoted++;
  }
  return NumPromoted;
}

// Promote indirect-call to conditional direct-call for one callsite.
uint32_t IndirectCallPromoter::tryToPromote(
    CallBase &CB, const std::vector<PromotionCandidate> &Candidates,
    uint64_t &TotalCount, const SmallVector<VTableCandidate> &VTableCandidates,
    uint64_t &TotalVTableCount) {
  LLVM_DEBUG(dbgs() << "Try to promote callsite " << CB << "\n  with "
                    << Candidates.size() << " function candidates and "
                    << VTableCandidates.size() << " vtable candidates\n");

  std::vector<PerFunctionCandidateInfo> PerFunctionCandidateInfo;

  const bool compareVTable = shouldCompareVTable(
      CB, Candidates, VTableCandidates, PerFunctionCandidateInfo);

  if (!compareVTable) {
    LLVM_DEBUG(dbgs() << "\tCompare functions for callsite " << CB << "\n");

    return promoteIndirectCallsByComparingFunctions(CB, Candidates, TotalCount,
                                                    SamplePGO, &ORE);
  }

  LLVM_DEBUG(dbgs() << "\tCompare virtual table addresses for callsite " << CB
                    << "\n");

  auto VirtualCallInfoIter = CBToVirtualCallInfoMap.find(&CB);

  assert(VirtualCallInfoIter != CBToVirtualCallInfoMap.end() &&
         "Expect each virtual call to have an entry in map");

  // assert all vtables have the same offset
  IRBuilder<> Builder(VirtualCallInfoIter->second.TypeTestInstr);

  Value *CastedVTableInstr = Builder.CreatePtrToInt(
      VTableCandidates[PerFunctionCandidateInfo[0].VTableIndices[0]]
          .VTableInstr,
      Builder.getInt64Ty());

  Value *ValueObject =
      Builder.CreateNUWSub(CastedVTableInstr,
                           Builder.getInt64(static_cast<uint64_t>(
                               PerFunctionCandidateInfo[0].Offsets[0])),
                           "vtable_object", false /* AllowFold */
      );

  std::unordered_map<int, Value *> OffsetToValueMap;
  OffsetToValueMap[PerFunctionCandidateInfo[0].Offsets[0]] = ValueObject;

  SmallPtrSet<Function *, 4> PromotedFunctionSet;

  for (int i = 0, size = Candidates.size(); i < size; i++) {
    promoteIndirectCallBasedOnVTable(
        CB, Candidates[i].TargetFunction, VTableCandidates,
        PerFunctionCandidateInfo[i].VTableIndices, OffsetToValueMap,
        TotalVTableCount, PromotedFunctionSet);
  }

  assert(PromotedFunctionSet.size() == Candidates.size() &&
         "All functions should be promotable if cost-benefit analysis decides "
         "to compare vtables");

  return Candidates.size();
}

// Traverse all the indirect-call callsite and get the value profile
// annotation to perform indirect-call promotion.
bool IndirectCallPromoter::processFunction(ProfileSummaryInfo *PSI) {
  bool Changed = false;
  ICallPromotionAnalysis ICallAnalysis;
  SmallVector<VTableCandidate> VTableCandidates;

  for (auto *CB : findIndirectCalls(F)) {
    uint32_t NumVals, NumCandidates;
    uint64_t TotalCount;
    auto ICallProfDataRef = ICallAnalysis.getPromotionCandidatesForInstruction(
        CB, NumVals, TotalCount, NumCandidates);
    if (!NumCandidates ||
        (PSI && PSI->hasProfileSummary() && !PSI->isHotCount(TotalCount)))
      continue;
    auto FunctionCandidates = getPromotionCandidatesForCallSite(
        *CB, ICallProfDataRef, TotalCount, NumCandidates);

    uint64_t TotalVTableCount = 0;
    if (!getVTableCandidates(CB, VTableCandidates, TotalVTableCount)) {
      VTableCandidates.clear();
    }

    // get the vtable set for each target value.
    // for target values with only one vtable, compare vtable.
    uint32_t NumPromoted = tryToPromote(*CB, FunctionCandidates, TotalCount,
                                        VTableCandidates, TotalVTableCount);
    if (NumPromoted == 0)
      continue;

    // FIXME: Update vtable prof metadata.
    Changed = true;
    // Adjust the MD.prof metadata. First delete the old one.
    CB->setMetadata(LLVMContext::MD_prof, nullptr);
    // If all promoted, we don't need the MD.prof metadata.
    if (TotalCount == 0 || NumPromoted == NumVals)
      continue;
    // Otherwise we need update with the un-promoted records back.
    annotateValueSite(*F.getParent(), *CB, ICallProfDataRef.slice(NumPromoted),
                      TotalCount, IPVK_IndirectCallTarget, NumCandidates);
  }
  return Changed;
}

static void buildCBToVirtualCallInfoMap(
    Module &M, function_ref<DominatorTree &(Function &)> LookupDomTree,
    DenseMap<const CallBase *, IndirectCallPromoter::VirtualCallInfo>
        &CBToVirtualCallInfoMap) {
  Function *TypeTestFunc =
      M.getFunction(Intrinsic::getName(Intrinsic::type_test));
  if (!TypeTestFunc || TypeTestFunc->use_empty())
    return;

  SmallVector<DevirtCallSite, 1> DevirtCalls;
  SmallVector<CallInst *, 1> Assumes;
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
    DevirtCalls.clear();
    Assumes.clear();

    auto &DT = LookupDomTree(*CI->getFunction());

    findDevirtualizableCallsForTypeTest(DevirtCalls, Assumes, CI, DT);

    for (auto &DevirtCall : DevirtCalls) {
      CallBase &CB = DevirtCall.CB;
      uint64_t Offset = DevirtCall.Offset;

      Instruction *VTableInstr =
          PGOIndirectCallVisitor::getAnnotatedVTableInstruction(&CB);

      if (!VTableInstr)
        continue;

      CBToVirtualCallInfoMap[&CB] = {Offset, VTableInstr, CompatibleTypeStr,
                                     dyn_cast<Instruction>(CI)};
    }
  }
}

// A wrapper function that does the actual work.
static bool promoteIndirectCalls(Module &M, ProfileSummaryInfo *PSI, bool InLTO,
                                 bool SamplePGO, ModuleAnalysisManager &MAM) {
  if (DisableICP)
    return false;

  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupDomTree = [&FAM](Function &F) -> DominatorTree & {
    return FAM.getResult<DominatorTreeAnalysis>(F);
  };

  InstrProfSymtab Symtab;
  if (Error E = Symtab.create(M, InLTO)) {
    std::string SymtabFailure = toString(std::move(E));
    M.getContext().emitError("Failed to create symtab: " + SymtabFailure);
    return false;
  }

  // Keys are indirect calls that call virtual function and is the subset of all
  // indirect calls.
  DenseMap<const CallBase *, IndirectCallPromoter::VirtualCallInfo>
      CBToVirtualCallInfoMap;

  buildCBToVirtualCallInfoMap(M, LookupDomTree, CBToVirtualCallInfoMap);

  bool Changed = false;
  for (auto &F : M) {
    if (F.isDeclaration() || F.hasOptNone())
      continue;

    auto &FAM =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(F);

    IndirectCallPromoter CallPromoter(F, &Symtab, SamplePGO,
                                      CBToVirtualCallInfoMap, ORE);
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
