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
#include "llvm/IR/DebugInfo.h"
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
#include "llvm/Transforms/Utils/Local.h"
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
static cl::opt<bool> ICPEnableVTableCmp(
    "icp-enable-vtable-cmp", cl::init(false), cl::Hidden,
    cl::desc("If ThinLTO and WPD is enabled and this option is true, "
             "indirect-call promotion pass will compare vtables rather than "
             "functions for speculative devirtualization of virtual calls."
             " If set to false, indirect-call promotion pass will always "
             "compare functions."));

static cl::opt<float>
    ICPVTableCountPercentage("icp-vtable-count-percentage", cl::init(0.99),
                             cl::Hidden,
                             cl::desc("Percentage of vtable count to compare"));

static cl::opt<int> ICPNumAdditionalVTableLast(
    "icp-num-additional-vtable-last", cl::init(0), cl::Hidden,
    cl::desc("The number of additional instruction for the last candidate"));

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

// Find the offset where type string is `CompatibleType`.
static std::optional<uint64_t>
getCompatibleTypeOffset(const GlobalVariable &VTableVar,
                        StringRef CompatibleType) {
  SmallVector<MDNode *, 2> Types; // type metadata associated with a vtable.
  VTableVar.getMetadata(LLVMContext::MD_type, Types);

  for (MDNode *Type : Types)
    if (auto *TypeId = dyn_cast<MDString>(Type->getOperand(1).get());
        TypeId && TypeId->getString() == CompatibleType)

      return cast<ConstantInt>(
                 cast<ConstantAsMetadata>(Type->getOperand(0))->getValue())
          ->getZExtValue();

  return std::nullopt;
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

// Returns the basic block in which `Inst` by `Use`.
static BasicBlock *getUserBasicBlock(Instruction *Inst, unsigned int OperandNo,
                                     Instruction *UserInst) {
  if (PHINode *PN = dyn_cast<PHINode>(UserInst))
    return PN->getIncomingBlock(
        PHINode::getIncomingValueNumForOperand(OperandNo));

  return UserInst->getParent();
}

// `DestBB` is a suitable basic block to sink `Inst` into when the following
// conditions are true:
// 1) `Inst->getParent()` is the sole predecessor of `DestBB`. This way `DestBB`
//    is dominated by `Inst->getParent()` and we don't need to sink across a
//    critical edge.
// 2) `Inst` have users and all users are in `DestBB`.
static bool isDestBBSuitableForSink(Instruction *Inst, BasicBlock *DestBB) {
  BasicBlock *BB = Inst->getParent();
  assert(Inst->getParent() != DestBB &&
         BB->getTerminator()->getNumSuccessors() == 2 &&
         "Caller should guarantee");
  // Do not sink across a critical edge for simplicity.
  if (DestBB->getUniquePredecessor() != BB)
    return false;

  // Now we know BB dominates DestBB.
  BasicBlock *UserBB = nullptr;
  for (Use &Use : Inst->uses()) {
    User *User = Use.getUser();
    // Do checked cast since IR verifier guarantees that the user of an
    // instruction must be an instruction. See `Verifier::visitInstruction`.
    Instruction *UserInst = cast<Instruction>(User);
    // We can sink debug or pseudo instructions together with Inst.
    if (UserInst->isDebugOrPseudoInst())
      continue;
    UserBB = getUserBasicBlock(Inst, Use.getOperandNo(), UserInst);
    // Do not sink if Inst is used in a basic block that is not DestBB.
    // TODO: Sink to the common dominator of all user blocks.
    if (UserBB != DestBB)
      return false;
  }
  return UserBB != nullptr;
}

// For the virtual call dispatch sequence, try to sink vtable load instructions
// to the cold indirect call fallback.
static bool tryToSinkInstruction(Instruction *I, BasicBlock *DestBlock) {
  assert(!I->isTerminator());
  if (!isDestBBSuitableForSink(I, DestBlock))
    return false;

  assert(DestBlock->getUniquePredecessor() == I->getParent());

  // Do not move control-flow-involving, volatile loads, vaarg, etc.
  // Do not sink static or dynamic alloca instructions. Static allocas must
  // remain in the entry block, and dynamic allocas must not be sunk in between
  // a stacksave / stackrestore pair, which would incorrectly shorten its
  // lifetime.
  if (isa<PHINode>(I) || I->isEHPad() || I->mayThrow() || !I->willReturn() ||
      isa<AllocaInst>(I))
    return false;

  // Do not sink convergent call instructions.
  if (const auto *C = dyn_cast<CallBase>(I))
    if (C->isInlineAsm() || C->cannotMerge() || C->isConvergent())
      return false;

  // Do not move an instruction that may write to memory.
  if (I->mayWriteToMemory())
    return false;

  // We can only sink load instructions if there is nothing between the load and
  // the end of block that could change the value.
  if (I->mayReadFromMemory()) {
    // We know that SrcBlock is the unique predecessor of DestBlock.
    for (BasicBlock::iterator Scan = std::next(I->getIterator()),
                              E = I->getParent()->end();
         Scan != E; ++Scan)
      if (Scan->mayWriteToMemory())
        return false;
  }

  BasicBlock::iterator InsertPos = DestBlock->getFirstInsertionPt();
  I->moveBefore(*DestBlock, InsertPos);

  // TODO: Sink debug intrinsic users of I to 'DestBlock'.
  // 'InstCombinerImpl::tryToSinkInstructionDbgValues' and
  // 'InstCombinerImpl::tryToSinkInstructionDbgVariableRecords' already have
  // the core logic to do this.
  return true;
}

// Try to sink instructions after VPtr to the indirect call fallback.
// Returns the number of sunk IR instructions.
static int tryToSinkInstructions(Instruction *VPtr,
                                 BasicBlock *IndirectCallBB) {
  BasicBlock *OriginalBB = VPtr->getParent();

  int SinkCount = 0;
  // FIXME: Find a way to bail out of the loop.
  for (Instruction &I :
       llvm::make_early_inc_range(llvm::drop_begin(llvm::reverse(*OriginalBB))))
    if (tryToSinkInstruction(&I, IndirectCallBB))
      SinkCount++;

  return SinkCount;
}

// Promote indirect calls to conditional direct calls, keeping track of
// thresholds.
class IndirectCallPromoter {
private:
  Function &F;
  Module &M;

  ProfileSummaryInfo *PSI = nullptr;

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

    // The byte offset of TargetFunction starting from the vtable address point.
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
      uint64_t TotalCount, uint32_t NumCandidates);

  // Promote a list of targets for one indirect-call callsite by comparing
  // indirect callee with functions. Returns true if there are IR
  // transformations and false otherwise.
  bool tryToPromoteWithFuncCmp(
      CallBase &CB, Instruction *VPtr,
      const std::vector<PromotionCandidate> &Candidates, uint64_t TotalCount,
      ArrayRef<InstrProfValueData> ICallProfDataRef, uint32_t NumCandidates,
      VTableGUIDCountsMap &VTableGUIDCounts);

  // Promote a list of targets for one indirect call by comparing vtables with
  // functions. Returns true if there are IR transformations and false
  // otherwise.
  bool tryToPromoteWithVTableCmp(
      CallBase &CB, Instruction *VPtr,
      const std::vector<PromotionCandidate> &Candidates,
      uint64_t TotalFuncCount, uint32_t NumCandidates,
      MutableArrayRef<InstrProfValueData> ICallProfDataRef,
      VTableGUIDCountsMap &VTableGUIDCounts);

  // Returns true if it's profitable to compare vtables.
  bool isProfitableToCompareVTables(
      const std::vector<PromotionCandidate> &Candidates, uint64_t TotalCount);

  // Populate `VTableGUIDCounts` vtable GUIDs and their counts and each
  // candidate with vtable information. Returns the vtable instruction if not
  // null.
  Instruction *computeVTableInfos(const CallBase *CB,
                                  VTableGUIDCountsMap &VTableGUIDCounts,
                                  std::vector<PromotionCandidate> &Candidates);

  Constant *getOrCreateVTableAddressPointVar(GlobalVariable *GV,
                                             uint64_t AddressPointOffset);

  void updateFuncValueProfiles(CallBase &CB, ArrayRef<InstrProfValueData> VDs,
                               uint64_t Sum, uint32_t MaxMDCount);

  void updateVPtrValueProfiles(Instruction *VPtr,
                               VTableGUIDCountsMap &VTableGUIDCounts);

public:
  IndirectCallPromoter(
      Function &Func, Module &M, ProfileSummaryInfo *PSI,
      InstrProfSymtab *Symtab, bool SamplePGO,
      const VirtualCallSiteTypeInfoMap &VirtualCSInfo,
      VTableAddressPointOffsetValMap &VTableAddressPointOffsetVal,
      OptimizationRemarkEmitter &ORE)
      : F(Func), M(M), PSI(PSI), Symtab(Symtab), SamplePGO(SamplePGO),
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

Constant *IndirectCallPromoter::getOrCreateVTableAddressPointVar(
    GlobalVariable *GV, uint64_t AddressPointOffset) {
  auto [Iter, Inserted] =
      VTableAddressPointOffsetVal[GV].try_emplace(AddressPointOffset, nullptr);
  if (Inserted)
    Iter->second = getVTableAddressPointOffset(GV, AddressPointOffset);
  return Iter->second;
}

Instruction *IndirectCallPromoter::computeVTableInfos(
    const CallBase *CB, VTableGUIDCountsMap &GUIDCountsMap,
    std::vector<PromotionCandidate> &Candidates) {
  if (!ICPEnableVTableCmp)
    return nullptr;

  // Only virtual calls have virtual call site info.
  auto Iter = VirtualCSInfo.find(CB);
  if (Iter == VirtualCSInfo.end())
    return nullptr;

  const auto &VirtualCallInfo = Iter->second;
  Instruction *VPtr = VirtualCallInfo.VPtr;

  SmallDenseMap<Function *, int, 4> CalleeIndexMap;
  for (size_t I = 0; I < Candidates.size(); I++)
    CalleeIndexMap[Candidates[I].TargetFunction] = I;

  uint32_t ActualNumValueData = 0;
  uint64_t TotalVTableCount = 0;
  auto VTableValueDataArray = getValueProfDataFromInst(
      *VirtualCallInfo.VPtr, IPVK_VTableTarget, MaxNumVTableAnnotations,
      ActualNumValueData, TotalVTableCount);
  if (VTableValueDataArray.get() == nullptr)
    return VPtr;

  // Compute the functions and counts from by each vtable.
  for (size_t j = 0; j < ActualNumValueData; j++) {
    uint64_t VTableVal = VTableValueDataArray[j].Value;
    GUIDCountsMap[VTableVal] = VTableValueDataArray[j].Count;
    GlobalVariable *VTableVar = Symtab->getGlobalVariable(VTableVal);
    if (!VTableVar) {
      LLVM_DEBUG(dbgs() << "\tCannot find vtable definition for " << VTableVal
                        << "; maybe the vtable isn't imported\n");
      continue;
    }

    std::optional<uint64_t> MaybeAddressPointOffset =
        getCompatibleTypeOffset(*VTableVar, VirtualCallInfo.CompatibleTypeStr);
    if (!MaybeAddressPointOffset)
      continue;

    const uint64_t AddressPointOffset = *MaybeAddressPointOffset;

    Function *Callee = nullptr;
    std::tie(Callee, std::ignore) = getFunctionAtVTableOffset(
        VTableVar, AddressPointOffset + VirtualCallInfo.FunctionOffset, M);
    if (!Callee)
      continue;
    auto CalleeIndexIter = CalleeIndexMap.find(Callee);
    if (CalleeIndexIter == CalleeIndexMap.end())
      continue;

    auto &Candidate = Candidates[CalleeIndexIter->second];
    Candidate.VTableGUIDAndCounts.push_back(
        {VTableVal, VTableValueDataArray[j].Count});
    Candidate.AddressPoints.push_back(
        getOrCreateVTableAddressPointVar(VTableVar, AddressPointOffset));
  }

  return VPtr;
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
    CallBase &CB, Instruction *VPtr,
    const std::vector<PromotionCandidate> &Candidates, uint64_t TotalCount,
    ArrayRef<InstrProfValueData> ICallProfDataRef, uint32_t NumCandidates,
    VTableGUIDCountsMap &VTableGUIDCounts) {
  uint32_t NumPromoted = 0;

  for (const auto &C : Candidates) {
    uint64_t FuncCount = C.Count;
    pgo::promoteIndirectCall(CB, C.TargetFunction, FuncCount, TotalCount,
                             SamplePGO, &ORE);
    assert(TotalCount >= FuncCount);
    TotalCount -= FuncCount;
    NumOfPGOICallPromotion++;
    NumPromoted++;

    if (!ICPEnableVTableCmp || C.VTableGUIDAndCounts.empty())
      continue;

    // Update VTableGUIDCounts
    uint64_t SumVTableCount = 0;
    for (const auto &[GUID, VTableCount] : C.VTableGUIDAndCounts)
      SumVTableCount += VTableCount;

    for (const auto &[GUID, VTableCount] : C.VTableGUIDAndCounts) {
      APInt APFuncCount((unsigned)128, FuncCount, false /*signed*/);
      APFuncCount *= VTableCount;
      VTableGUIDCounts[GUID] -= APFuncCount.udiv(SumVTableCount).getZExtValue();
    }
  }
  if (NumPromoted == 0)
    return false;

  assert(NumPromoted <= ICallProfDataRef.size() &&
         "Number of promoted functions should not be greater than the number "
         "of values in profile metadata");

  // Update value profiles on the indirect call.
  // TODO: Handle profile update properly when Clang `-fstrict-vtable-pointers`
  // is enabled and a vtable is used to load multiple virtual functions.
  updateFuncValueProfiles(CB, ICallProfDataRef.slice(NumPromoted), TotalCount,
                          NumCandidates);
  // Update value profiles on the vtable pointer if it exists.
  if (VPtr)
    updateVPtrValueProfiles(VPtr, VTableGUIDCounts);
  return true;
}

void IndirectCallPromoter::updateFuncValueProfiles(
    CallBase &CB, ArrayRef<InstrProfValueData> CallVDs, uint64_t TotalCount,
    uint32_t MaxMDCount) {
  // First clear the existing !prof.
  CB.setMetadata(LLVMContext::MD_prof, nullptr);
  // Annotate the remaining value profiles if counter is not zero.
  if (TotalCount != 0)
    annotateValueSite(M, CB, CallVDs, TotalCount, IPVK_IndirectCallTarget,
                      MaxMDCount);
}

void IndirectCallPromoter::updateVPtrValueProfiles(
    Instruction *VPtr, VTableGUIDCountsMap &VTableGUIDCounts) {
  VPtr->setMetadata(LLVMContext::MD_prof, nullptr);
  std::vector<InstrProfValueData> VTableValueProfiles;
  uint64_t TotalVTableCount = 0;
  for (auto [GUID, Count] : VTableGUIDCounts) {
    if (Count == 0)
      continue;

    VTableValueProfiles.push_back({GUID, Count});
    TotalVTableCount += Count;
  }
  llvm::sort(VTableValueProfiles,
             [](const InstrProfValueData &LHS, const InstrProfValueData &RHS) {
               return LHS.Count > RHS.Count;
             });

  annotateValueSite(M, *VPtr, VTableValueProfiles, TotalVTableCount,
                    IPVK_VTableTarget, VTableValueProfiles.size());
}

bool IndirectCallPromoter::tryToPromoteWithVTableCmp(
    CallBase &CB, Instruction *VPtr,
    const std::vector<PromotionCandidate> &Candidates, uint64_t TotalFuncCount,
    uint32_t NumCandidates,
    MutableArrayRef<InstrProfValueData> ICallProfDataRef,
    VTableGUIDCountsMap &VTableGUIDCounts) {
  SmallVector<uint64_t, 4> PromotedFuncCount;
  // TODO: Explain the branch accuracy (-fstrict-vtable-pointer) with a
  // compiler-rt test.
  for (const auto &Candidate : Candidates) {
    uint64_t IfCount = 0;
    for (auto &[GUID, Count] : Candidate.VTableGUIDAndCounts) {
      IfCount += Count;
      VTableGUIDCounts[GUID] -= Count;
    }

    BasicBlock *OriginalBB = CB.getParent();
    promoteCallWithVTableCmp(
        CB, VPtr, Candidate.TargetFunction, Candidate.AddressPoints,
        getBranchWeights(CB.getContext(), IfCount, TotalFuncCount - IfCount));

    int SinkCount = tryToSinkInstructions(
        PromotedFuncCount.empty() ? VPtr : OriginalBB->getFirstNonPHI(),
        CB.getParent());

    ORE.emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "Promoted", &CB)
             << "Promote indirect call to "
             << ore::NV("DirectCallee", Candidate.TargetFunction)
             << " with count " << ore::NV("Count", Candidate.Count)
             << " out of " << ore::NV("TotalCount", TotalFuncCount)
             << ", compare "
             << ore::NV("VTable", Candidate.VTableGUIDAndCounts.size())
             << " vtables and sink " << ore::NV("SinkCount", SinkCount)
             << " instructions";
    });

    PromotedFuncCount.push_back(IfCount);

    TotalFuncCount -= IfCount;
    NumOfPGOICallPromotion++;
  }

  if (PromotedFuncCount.empty())
    return false;

  // Update value profiles for 'CB' and 'VPtr', assuming that each 'CB' has a
  // a distinct 'VPtr'.
  // TODO: Handle profile update properly when Clang `-fstrict-vtable-pointers`
  // is enabled and a vtable is used to load multiple virtual functions.
  for (size_t I = 0; I < PromotedFuncCount.size(); I++)
    ICallProfDataRef[I].Count -=
        std::max(PromotedFuncCount[I], ICallProfDataRef[I].Count);
  // Sort value profiles by count in descending order.
  llvm::sort(ICallProfDataRef.begin(), ICallProfDataRef.end(),
             [](const InstrProfValueData &LHS, const InstrProfValueData &RHS) {
               return LHS.Count > RHS.Count;
             });
  // Drop the <target-value, count> pair if count is not greater than zero.
  ArrayRef<InstrProfValueData> VDs(
      ICallProfDataRef.begin(),
      llvm::upper_bound(ICallProfDataRef, 0U,
                        [](uint64_t Count, const InstrProfValueData &ProfData) {
                          return ProfData.Count <= Count;
                        }));
  updateFuncValueProfiles(CB, VDs, TotalFuncCount, NumCandidates);
  updateVPtrValueProfiles(VPtr, VTableGUIDCounts);
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

    auto PromotionCandidates = getPromotionCandidatesForCallSite(
        *CB, ICallProfDataRef, TotalCount, NumCandidates);

    VTableGUIDCountsMap VTableGUIDCounts;
    Instruction *VPtr =
        computeVTableInfos(CB, VTableGUIDCounts, PromotionCandidates);

    if (isProfitableToCompareVTables(PromotionCandidates, TotalCount))
      Changed |= tryToPromoteWithVTableCmp(*CB, VPtr, PromotionCandidates,
                                           TotalCount, NumCandidates,
                                           ICallProfDataRef, VTableGUIDCounts);
    else
      Changed |= tryToPromoteWithFuncCmp(*CB, VPtr, PromotionCandidates,
                                         TotalCount, ICallProfDataRef,
                                         NumCandidates, VTableGUIDCounts);
  }
  return Changed;
}

// TODO: Returns false if the function addressing and vtable load instructions
// cannot sink to indirect fallback.
bool IndirectCallPromoter::isProfitableToCompareVTables(
    const std::vector<PromotionCandidate> &Candidates, uint64_t TotalCount) {
  if (!ICPEnableVTableCmp || Candidates.empty())
    return false;
  uint64_t RemainingVTableCount = TotalCount;
  for (size_t I = 0; I < Candidates.size(); I++) {
    auto &Candidate = Candidates[I];
    uint64_t VTableSumCount = 0;
    for (auto &[GUID, Count] : Candidate.VTableGUIDAndCounts)
      VTableSumCount += Count;

    if (VTableSumCount < Candidate.Count * ICPVTableCountPercentage)
      return false;

    RemainingVTableCount -= Candidate.Count;

    int NumAdditionalVTable = 0;
    if (I == Candidates.size() - 1)
      NumAdditionalVTable = ICPNumAdditionalVTableLast;

    int ActualNumAdditionalInst = Candidate.AddressPoints.size() - 1;
    if (ActualNumAdditionalInst > NumAdditionalVTable) {
      return false;
    }
  }

  // If the indirect fallback is not cold, don't compare vtables.
  if (PSI && PSI->hasProfileSummary() &&
      !PSI->isColdCount(RemainingVTableCount))
    return false;

  return true;
}

static void
computeVirtualCallSiteTypeInfoMap(Module &M, ModuleAnalysisManager &MAM,
                                  VirtualCallSiteTypeInfoMap &VirtualCSInfo) {
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupDomTree = [&FAM](Function &F) -> DominatorTree & {
    return FAM.getResult<DominatorTreeAnalysis>(F);
  };

  auto compute = [&](Function *Func) {
    if (!Func || Func->use_empty())
      return;
    // Iterate all type.test calls and find all indirect calls.
    // TODO: Add llvm.public.type.test
    for (Use &U : llvm::make_early_inc_range(Func->uses())) {
      auto *CI = dyn_cast<CallInst>(U.getUser());
      if (!CI)
        continue;
      auto *TypeMDVal = cast<MetadataAsValue>(CI->getArgOperand(1));
      if (!TypeMDVal)
        continue;
      auto *CompatibleTypeId = dyn_cast<MDString>(TypeMDVal->getMetadata());
      if (!CompatibleTypeId)
        continue;

      // Find out all devirtualizable call sites given a llvm.type.test
      // intrinsic call.
      SmallVector<DevirtCallSite, 1> DevirtCalls;
      SmallVector<CallInst *, 1> Assumes;
      auto &DT = LookupDomTree(*CI->getFunction());
      findDevirtualizableCallsForTypeTest(DevirtCalls, Assumes, CI, DT);

      // type-id, offset from the address point
      // combined with type metadata to compute function offset
      for (auto &DevirtCall : DevirtCalls) {
        CallBase &CB = DevirtCall.CB;
        // Given an indirect call, try find the instruction which loads a
        // pointer to virtual table.
        Instruction *VTablePtr =
            PGOIndirectCallVisitor::tryGetVTableInstruction(&CB);
        if (!VTablePtr)
          continue;
        VirtualCSInfo[&CB] = {DevirtCall.Offset, VTablePtr,
                              CompatibleTypeId->getString()};
      }
    }
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

  compute(TypeTestFunc);

  Function *PublicTypeTestFunc =
      M.getFunction(Intrinsic::getName(Intrinsic::public_type_test));
  compute(PublicTypeTestFunc);
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

    IndirectCallPromoter CallPromoter(F, M, PSI, &Symtab, SamplePGO,
                                      VirtualCSInfo,
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
