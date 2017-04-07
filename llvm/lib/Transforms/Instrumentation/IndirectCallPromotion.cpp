//===-- IndirectCallPromotion.cpp - Optimizations based on value profiling ===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IndirectCallPromotionAnalysis.h"
#include "llvm/Analysis/IndirectCallSiteVisitor.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/PassSupport.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/PGOInstrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <cassert>
#include <cstdint>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "pgo-icall-prom"

STATISTIC(NumOfPGOICallPromotion, "Number of indirect call promotions.");
STATISTIC(NumOfPGOICallsites, "Number of indirect call candidate sites.");
STATISTIC(NumOfPGOMemOPOpt, "Number of memop intrinsics optimized.");
STATISTIC(NumOfPGOMemOPAnnotate, "Number of memop intrinsics annotated.");

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
              cl::desc("Max number of promotions for this compilaiton"));

// If ICPCSSkip is non zero, the first ICPCSSkip callsites will be skipped.
// For debug use only.
static cl::opt<unsigned>
    ICPCSSkip("icp-csskip", cl::init(0), cl::Hidden, cl::ZeroOrMore,
              cl::desc("Skip Callsite up to this number for this compilaiton"));

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

// The minimum call count to optimize memory intrinsic calls.
static cl::opt<unsigned>
    MemOPCountThreshold("pgo-memop-count-threshold", cl::Hidden, cl::ZeroOrMore,
                        cl::init(1000),
                        cl::desc("The minimum count to optimize memory "
                                 "intrinsic calls"));

// Command line option to disable memory intrinsic optimization. The default is
// false. This is for debug purpose.
static cl::opt<bool> DisableMemOPOPT("disable-memop-opt", cl::init(false),
                                     cl::Hidden, cl::desc("Disable optimize"));

// The percent threshold to optimize memory intrinsic calls.
static cl::opt<unsigned>
    MemOPPercentThreshold("pgo-memop-percent-threshold", cl::init(40),
                          cl::Hidden, cl::ZeroOrMore,
                          cl::desc("The percentage threshold for the "
                                   "memory intrinsic calls optimization"));

// Maximum number of versions for optimizing memory intrinsic call.
static cl::opt<unsigned>
    MemOPMaxVersion("pgo-memop-max-version", cl::init(3), cl::Hidden,
                    cl::ZeroOrMore,
                    cl::desc("The max version for the optimized memory "
                             " intrinsic calls"));

// Scale the counts from the annotation using the BB count value.
static cl::opt<bool>
    MemOPScaleCount("pgo-memop-scale-count", cl::init(true), cl::Hidden,
                    cl::desc("Scale the memop size counts using the basic "
                             " block count value"));

// This option sets the rangge of precise profile memop sizes.
extern cl::opt<std::string> MemOPSizeRange;

// This option sets the value that groups large memop sizes
extern cl::opt<unsigned> MemOPSizeLarge;

namespace {
class PGOIndirectCallPromotionLegacyPass : public ModulePass {
public:
  static char ID;

  PGOIndirectCallPromotionLegacyPass(bool InLTO = false, bool SamplePGO = false)
      : ModulePass(ID), InLTO(InLTO), SamplePGO(SamplePGO) {
    initializePGOIndirectCallPromotionLegacyPassPass(
        *PassRegistry::getPassRegistry());
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

class PGOMemOPSizeOptLegacyPass : public FunctionPass {
public:
  static char ID;

  PGOMemOPSizeOptLegacyPass() : FunctionPass(ID) {
    initializePGOMemOPSizeOptLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "PGOMemOPSize"; }

private:
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
} // end anonymous namespace

char PGOIndirectCallPromotionLegacyPass::ID = 0;
INITIALIZE_PASS(PGOIndirectCallPromotionLegacyPass, "pgo-icall-prom",
                "Use PGO instrumentation profile to promote indirect calls to "
                "direct calls.",
                false, false)

ModulePass *llvm::createPGOIndirectCallPromotionLegacyPass(bool InLTO,
                                                           bool SamplePGO) {
  return new PGOIndirectCallPromotionLegacyPass(InLTO, SamplePGO);
}

char PGOMemOPSizeOptLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(PGOMemOPSizeOptLegacyPass, "pgo-memop-opt",
                      "Optimize memory intrinsic using its size value profile",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(PGOMemOPSizeOptLegacyPass, "pgo-memop-opt",
                    "Optimize memory intrinsic using its size value profile",
                    false, false)

FunctionPass *llvm::createPGOMemOPSizeOptLegacyPass() {
  return new PGOMemOPSizeOptLegacyPass();
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

  // Test if we can legally promote this direct-call of Target.
  bool isPromotionLegal(Instruction *Inst, uint64_t Target, Function *&F,
                        const char **Reason = nullptr);

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
      Instruction *Inst, const ArrayRef<InstrProfValueData> &ValueDataRef,
      uint64_t TotalCount, uint32_t NumCandidates);

  // Promote a list of targets for one indirect-call callsite. Return
  // the number of promotions.
  uint32_t tryToPromote(Instruction *Inst,
                        const std::vector<PromotionCandidate> &Candidates,
                        uint64_t &TotalCount);

  // Noncopyable
  ICallPromotionFunc(const ICallPromotionFunc &other) = delete;
  ICallPromotionFunc &operator=(const ICallPromotionFunc &other) = delete;

public:
  ICallPromotionFunc(Function &Func, Module *Modu, InstrProfSymtab *Symtab,
                     bool SamplePGO)
      : F(Func), M(Modu), Symtab(Symtab), SamplePGO(SamplePGO) {}

  bool processFunction();
};
} // end anonymous namespace

bool llvm::isLegalToPromote(Instruction *Inst, Function *F,
                            const char **Reason) {
  // Check the return type.
  Type *CallRetType = Inst->getType();
  if (!CallRetType->isVoidTy()) {
    Type *FuncRetType = F->getReturnType();
    if (FuncRetType != CallRetType &&
        !CastInst::isBitCastable(FuncRetType, CallRetType)) {
      if (Reason)
        *Reason = "Return type mismatch";
      return false;
    }
  }

  // Check if the arguments are compatible with the parameters
  FunctionType *DirectCalleeType = F->getFunctionType();
  unsigned ParamNum = DirectCalleeType->getFunctionNumParams();
  CallSite CS(Inst);
  unsigned ArgNum = CS.arg_size();

  if (ParamNum != ArgNum && !DirectCalleeType->isVarArg()) {
    if (Reason)
      *Reason = "The number of arguments mismatch";
    return false;
  }

  for (unsigned I = 0; I < ParamNum; ++I) {
    Type *PTy = DirectCalleeType->getFunctionParamType(I);
    Type *ATy = CS.getArgument(I)->getType();
    if (PTy == ATy)
      continue;
    if (!CastInst::castIsValid(Instruction::BitCast, CS.getArgument(I), PTy)) {
      if (Reason)
        *Reason = "Argument type mismatch";
      return false;
    }
  }

  DEBUG(dbgs() << " #" << NumOfPGOICallPromotion << " Promote the icall to "
               << F->getName() << "\n");
  return true;
}

bool ICallPromotionFunc::isPromotionLegal(Instruction *Inst, uint64_t Target,
                                          Function *&TargetFunction,
                                          const char **Reason) {
  TargetFunction = Symtab->getFunction(Target);
  if (TargetFunction == nullptr) {
    *Reason = "Cannot find the target";
    return false;
  }
  return isLegalToPromote(Inst, TargetFunction, Reason);
}

// Indirect-call promotion heuristic. The direct targets are sorted based on
// the count. Stop at the first target that is not promoted.
std::vector<ICallPromotionFunc::PromotionCandidate>
ICallPromotionFunc::getPromotionCandidatesForCallSite(
    Instruction *Inst, const ArrayRef<InstrProfValueData> &ValueDataRef,
    uint64_t TotalCount, uint32_t NumCandidates) {
  std::vector<PromotionCandidate> Ret;

  DEBUG(dbgs() << " \nWork on callsite #" << NumOfPGOICallsites << *Inst
               << " Num_targets: " << ValueDataRef.size()
               << " Num_candidates: " << NumCandidates << "\n");
  NumOfPGOICallsites++;
  if (ICPCSSkip != 0 && NumOfPGOICallsites <= ICPCSSkip) {
    DEBUG(dbgs() << " Skip: User options.\n");
    return Ret;
  }

  for (uint32_t I = 0; I < NumCandidates; I++) {
    uint64_t Count = ValueDataRef[I].Count;
    assert(Count <= TotalCount);
    uint64_t Target = ValueDataRef[I].Value;
    DEBUG(dbgs() << " Candidate " << I << " Count=" << Count
                 << "  Target_func: " << Target << "\n");

    if (ICPInvokeOnly && dyn_cast<CallInst>(Inst)) {
      DEBUG(dbgs() << " Not promote: User options.\n");
      break;
    }
    if (ICPCallOnly && dyn_cast<InvokeInst>(Inst)) {
      DEBUG(dbgs() << " Not promote: User option.\n");
      break;
    }
    if (ICPCutOff != 0 && NumOfPGOICallPromotion >= ICPCutOff) {
      DEBUG(dbgs() << " Not promote: Cutoff reached.\n");
      break;
    }
    Function *TargetFunction = nullptr;
    const char *Reason = nullptr;
    if (!isPromotionLegal(Inst, Target, TargetFunction, &Reason)) {
      StringRef TargetFuncName = Symtab->getFuncName(Target);
      DEBUG(dbgs() << " Not promote: " << Reason << "\n");
      emitOptimizationRemarkMissed(
          F.getContext(), "pgo-icall-prom", F, Inst->getDebugLoc(),
          Twine("Cannot promote indirect call to ") +
              (TargetFuncName.empty() ? Twine(Target) : Twine(TargetFuncName)) +
              Twine(" with count of ") + Twine(Count) + ": " + Reason);
      break;
    }
    Ret.push_back(PromotionCandidate(TargetFunction, Count));
    TotalCount -= Count;
  }
  return Ret;
}

// Create a diamond structure for If_Then_Else. Also update the profile
// count. Do the fix-up for the invoke instruction.
static void createIfThenElse(Instruction *Inst, Function *DirectCallee,
                             uint64_t Count, uint64_t TotalCount,
                             BasicBlock **DirectCallBB,
                             BasicBlock **IndirectCallBB,
                             BasicBlock **MergeBB) {
  CallSite CS(Inst);
  Value *OrigCallee = CS.getCalledValue();

  IRBuilder<> BBBuilder(Inst);
  LLVMContext &Ctx = Inst->getContext();
  Value *BCI1 =
      BBBuilder.CreateBitCast(OrigCallee, Type::getInt8PtrTy(Ctx), "");
  Value *BCI2 =
      BBBuilder.CreateBitCast(DirectCallee, Type::getInt8PtrTy(Ctx), "");
  Value *PtrCmp = BBBuilder.CreateICmpEQ(BCI1, BCI2, "");

  uint64_t ElseCount = TotalCount - Count;
  uint64_t MaxCount = (Count >= ElseCount ? Count : ElseCount);
  uint64_t Scale = calculateCountScale(MaxCount);
  MDBuilder MDB(Inst->getContext());
  MDNode *BranchWeights = MDB.createBranchWeights(
      scaleBranchCount(Count, Scale), scaleBranchCount(ElseCount, Scale));
  TerminatorInst *ThenTerm, *ElseTerm;
  SplitBlockAndInsertIfThenElse(PtrCmp, Inst, &ThenTerm, &ElseTerm,
                                BranchWeights);
  *DirectCallBB = ThenTerm->getParent();
  (*DirectCallBB)->setName("if.true.direct_targ");
  *IndirectCallBB = ElseTerm->getParent();
  (*IndirectCallBB)->setName("if.false.orig_indirect");
  *MergeBB = Inst->getParent();
  (*MergeBB)->setName("if.end.icp");

  // Special handing of Invoke instructions.
  InvokeInst *II = dyn_cast<InvokeInst>(Inst);
  if (!II)
    return;

  // We don't need branch instructions for invoke.
  ThenTerm->eraseFromParent();
  ElseTerm->eraseFromParent();

  // Add jump from Merge BB to the NormalDest. This is needed for the newly
  // created direct invoke stmt -- as its NormalDst will be fixed up to MergeBB.
  BranchInst::Create(II->getNormalDest(), *MergeBB);
}

// Find the PHI in BB that have the CallResult as the operand.
static bool getCallRetPHINode(BasicBlock *BB, Instruction *Inst) {
  BasicBlock *From = Inst->getParent();
  for (auto &I : *BB) {
    PHINode *PHI = dyn_cast<PHINode>(&I);
    if (!PHI)
      continue;
    int IX = PHI->getBasicBlockIndex(From);
    if (IX == -1)
      continue;
    Value *V = PHI->getIncomingValue(IX);
    if (dyn_cast<Instruction>(V) == Inst)
      return true;
  }
  return false;
}

// This method fixes up PHI nodes in BB where BB is the UnwindDest of an
// invoke instruction. In BB, there may be PHIs with incoming block being
// OrigBB (the MergeBB after if-then-else splitting). After moving the invoke
// instructions to its own BB, OrigBB is no longer the predecessor block of BB.
// Instead two new predecessors are added: IndirectCallBB and DirectCallBB,
// so the PHI node's incoming BBs need to be fixed up accordingly.
static void fixupPHINodeForUnwind(Instruction *Inst, BasicBlock *BB,
                                  BasicBlock *OrigBB,
                                  BasicBlock *IndirectCallBB,
                                  BasicBlock *DirectCallBB) {
  for (auto &I : *BB) {
    PHINode *PHI = dyn_cast<PHINode>(&I);
    if (!PHI)
      continue;
    int IX = PHI->getBasicBlockIndex(OrigBB);
    if (IX == -1)
      continue;
    Value *V = PHI->getIncomingValue(IX);
    PHI->addIncoming(V, IndirectCallBB);
    PHI->setIncomingBlock(IX, DirectCallBB);
  }
}

// This method fixes up PHI nodes in BB where BB is the NormalDest of an
// invoke instruction. In BB, there may be PHIs with incoming block being
// OrigBB (the MergeBB after if-then-else splitting). After moving the invoke
// instructions to its own BB, a new incoming edge will be added to the original
// NormalDstBB from the IndirectCallBB.
static void fixupPHINodeForNormalDest(Instruction *Inst, BasicBlock *BB,
                                      BasicBlock *OrigBB,
                                      BasicBlock *IndirectCallBB,
                                      Instruction *NewInst) {
  for (auto &I : *BB) {
    PHINode *PHI = dyn_cast<PHINode>(&I);
    if (!PHI)
      continue;
    int IX = PHI->getBasicBlockIndex(OrigBB);
    if (IX == -1)
      continue;
    Value *V = PHI->getIncomingValue(IX);
    if (dyn_cast<Instruction>(V) == Inst) {
      PHI->setIncomingBlock(IX, IndirectCallBB);
      PHI->addIncoming(NewInst, OrigBB);
      continue;
    }
    PHI->addIncoming(V, IndirectCallBB);
  }
}

// Add a bitcast instruction to the direct-call return value if needed.
static Instruction *insertCallRetCast(const Instruction *Inst,
                                      Instruction *DirectCallInst,
                                      Function *DirectCallee) {
  if (Inst->getType()->isVoidTy())
    return DirectCallInst;

  Type *CallRetType = Inst->getType();
  Type *FuncRetType = DirectCallee->getReturnType();
  if (FuncRetType == CallRetType)
    return DirectCallInst;

  BasicBlock *InsertionBB;
  if (CallInst *CI = dyn_cast<CallInst>(DirectCallInst))
    InsertionBB = CI->getParent();
  else
    InsertionBB = (dyn_cast<InvokeInst>(DirectCallInst))->getNormalDest();

  return (new BitCastInst(DirectCallInst, CallRetType, "",
                          InsertionBB->getTerminator()));
}

// Create a DirectCall instruction in the DirectCallBB.
// Parameter Inst is the indirect-call (invoke) instruction.
// DirectCallee is the decl of the direct-call (invoke) target.
// DirecallBB is the BB that the direct-call (invoke) instruction is inserted.
// MergeBB is the bottom BB of the if-then-else-diamond after the
// transformation. For invoke instruction, the edges from DirectCallBB and
// IndirectCallBB to MergeBB are removed before this call (during
// createIfThenElse).
static Instruction *createDirectCallInst(const Instruction *Inst,
                                         Function *DirectCallee,
                                         BasicBlock *DirectCallBB,
                                         BasicBlock *MergeBB) {
  Instruction *NewInst = Inst->clone();
  if (CallInst *CI = dyn_cast<CallInst>(NewInst)) {
    CI->setCalledFunction(DirectCallee);
    CI->mutateFunctionType(DirectCallee->getFunctionType());
  } else {
    // Must be an invoke instruction. Direct invoke's normal destination is
    // fixed up to MergeBB. MergeBB is the place where return cast is inserted.
    // Also since IndirectCallBB does not have an edge to MergeBB, there is no
    // need to insert new PHIs into MergeBB.
    InvokeInst *II = dyn_cast<InvokeInst>(NewInst);
    assert(II);
    II->setCalledFunction(DirectCallee);
    II->mutateFunctionType(DirectCallee->getFunctionType());
    II->setNormalDest(MergeBB);
  }

  DirectCallBB->getInstList().insert(DirectCallBB->getFirstInsertionPt(),
                                     NewInst);

  // Clear the value profile data.
  NewInst->setMetadata(LLVMContext::MD_prof, nullptr);
  CallSite NewCS(NewInst);
  FunctionType *DirectCalleeType = DirectCallee->getFunctionType();
  unsigned ParamNum = DirectCalleeType->getFunctionNumParams();
  for (unsigned I = 0; I < ParamNum; ++I) {
    Type *ATy = NewCS.getArgument(I)->getType();
    Type *PTy = DirectCalleeType->getParamType(I);
    if (ATy != PTy) {
      BitCastInst *BI = new BitCastInst(NewCS.getArgument(I), PTy, "", NewInst);
      NewCS.setArgument(I, BI);
    }
  }

  return insertCallRetCast(Inst, NewInst, DirectCallee);
}

// Create a PHI to unify the return values of calls.
static void insertCallRetPHI(Instruction *Inst, Instruction *CallResult,
                             Function *DirectCallee) {
  if (Inst->getType()->isVoidTy())
    return;

  BasicBlock *RetValBB = CallResult->getParent();

  BasicBlock *PHIBB;
  if (InvokeInst *II = dyn_cast<InvokeInst>(CallResult))
    RetValBB = II->getNormalDest();

  PHIBB = RetValBB->getSingleSuccessor();
  if (getCallRetPHINode(PHIBB, Inst))
    return;

  PHINode *CallRetPHI = PHINode::Create(Inst->getType(), 0);
  PHIBB->getInstList().push_front(CallRetPHI);
  Inst->replaceAllUsesWith(CallRetPHI);
  CallRetPHI->addIncoming(Inst, Inst->getParent());
  CallRetPHI->addIncoming(CallResult, RetValBB);
}

// This function does the actual indirect-call promotion transformation:
// For an indirect-call like:
//     Ret = (*Foo)(Args);
// It transforms to:
//     if (Foo == DirectCallee)
//        Ret1 = DirectCallee(Args);
//     else
//        Ret2 = (*Foo)(Args);
//     Ret = phi(Ret1, Ret2);
// It adds type casts for the args do not match the parameters and the return
// value. Branch weights metadata also updated.
// If \p AttachProfToDirectCall is true, a prof metadata is attached to the
// new direct call to contain \p Count. This is used by SamplePGO inliner to
// check callsite hotness.
// Returns the promoted direct call instruction.
Instruction *llvm::promoteIndirectCall(Instruction *Inst,
                                       Function *DirectCallee, uint64_t Count,
                                       uint64_t TotalCount,
                                       bool AttachProfToDirectCall) {
  assert(DirectCallee != nullptr);
  BasicBlock *BB = Inst->getParent();
  // Just to suppress the non-debug build warning.
  (void)BB;
  DEBUG(dbgs() << "\n\n== Basic Block Before ==\n");
  DEBUG(dbgs() << *BB << "\n");

  BasicBlock *DirectCallBB, *IndirectCallBB, *MergeBB;
  createIfThenElse(Inst, DirectCallee, Count, TotalCount, &DirectCallBB,
                   &IndirectCallBB, &MergeBB);

  Instruction *NewInst =
      createDirectCallInst(Inst, DirectCallee, DirectCallBB, MergeBB);

  if (AttachProfToDirectCall) {
    SmallVector<uint32_t, 1> Weights;
    Weights.push_back(Count);
    MDBuilder MDB(NewInst->getContext());
    dyn_cast<Instruction>(NewInst->stripPointerCasts())
        ->setMetadata(LLVMContext::MD_prof, MDB.createBranchWeights(Weights));
  }

  // Move Inst from MergeBB to IndirectCallBB.
  Inst->removeFromParent();
  IndirectCallBB->getInstList().insert(IndirectCallBB->getFirstInsertionPt(),
                                       Inst);

  if (InvokeInst *II = dyn_cast<InvokeInst>(Inst)) {
    // At this point, the original indirect invoke instruction has the original
    // UnwindDest and NormalDest. For the direct invoke instruction, the
    // NormalDest points to MergeBB, and MergeBB jumps to the original
    // NormalDest. MergeBB might have a new bitcast instruction for the return
    // value. The PHIs are with the original NormalDest. Since we now have two
    // incoming edges to NormalDest and UnwindDest, we have to do some fixups.
    //
    // UnwindDest will not use the return value. So pass nullptr here.
    fixupPHINodeForUnwind(Inst, II->getUnwindDest(), MergeBB, IndirectCallBB,
                          DirectCallBB);
    // We don't need to update the operand from NormalDest for DirectCallBB.
    // Pass nullptr here.
    fixupPHINodeForNormalDest(Inst, II->getNormalDest(), MergeBB,
                              IndirectCallBB, NewInst);
  }

  insertCallRetPHI(Inst, NewInst, DirectCallee);

  DEBUG(dbgs() << "\n== Basic Blocks After ==\n");
  DEBUG(dbgs() << *BB << *DirectCallBB << *IndirectCallBB << *MergeBB << "\n");

  emitOptimizationRemark(
      BB->getContext(), "pgo-icall-prom", *BB->getParent(), Inst->getDebugLoc(),
      Twine("Promote indirect call to ") + DirectCallee->getName() +
          " with count " + Twine(Count) + " out of " + Twine(TotalCount));
  return NewInst;
}

// Promote indirect-call to conditional direct-call for one callsite.
uint32_t ICallPromotionFunc::tryToPromote(
    Instruction *Inst, const std::vector<PromotionCandidate> &Candidates,
    uint64_t &TotalCount) {
  uint32_t NumPromoted = 0;

  for (auto &C : Candidates) {
    uint64_t Count = C.Count;
    promoteIndirectCall(Inst, C.TargetFunction, Count, TotalCount, SamplePGO);
    assert(TotalCount >= Count);
    TotalCount -= Count;
    NumOfPGOICallPromotion++;
    NumPromoted++;
  }
  return NumPromoted;
}

// Traverse all the indirect-call callsite and get the value profile
// annotation to perform indirect-call promotion.
bool ICallPromotionFunc::processFunction() {
  bool Changed = false;
  ICallPromotionAnalysis ICallAnalysis;
  for (auto &I : findIndirectCallSites(F)) {
    uint32_t NumVals, NumCandidates;
    uint64_t TotalCount;
    auto ICallProfDataRef = ICallAnalysis.getPromotionCandidatesForInstruction(
        I, NumVals, TotalCount, NumCandidates);
    if (!NumCandidates)
      continue;
    auto PromotionCandidates = getPromotionCandidatesForCallSite(
        I, ICallProfDataRef, TotalCount, NumCandidates);
    uint32_t NumPromoted = tryToPromote(I, PromotionCandidates, TotalCount);
    if (NumPromoted == 0)
      continue;

    Changed = true;
    // Adjust the MD.prof metadata. First delete the old one.
    I->setMetadata(LLVMContext::MD_prof, nullptr);
    // If all promoted, we don't need the MD.prof metadata.
    if (TotalCount == 0 || NumPromoted == NumVals)
      continue;
    // Otherwise we need update with the un-promoted records back.
    annotateValueSite(*M, *I, ICallProfDataRef.slice(NumPromoted), TotalCount,
                      IPVK_IndirectCallTarget, NumCandidates);
  }
  return Changed;
}

// A wrapper function that does the actual work.
static bool promoteIndirectCalls(Module &M, bool InLTO, bool SamplePGO) {
  if (DisableICP)
    return false;
  InstrProfSymtab Symtab;
  Symtab.create(M, InLTO);
  bool Changed = false;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    if (F.hasFnAttribute(Attribute::OptimizeNone))
      continue;
    ICallPromotionFunc ICallPromotion(F, &M, &Symtab, SamplePGO);
    bool FuncChanged = ICallPromotion.processFunction();
    if (ICPDUMPAFTER && FuncChanged) {
      DEBUG(dbgs() << "\n== IR Dump After =="; F.print(dbgs()));
      DEBUG(dbgs() << "\n");
    }
    Changed |= FuncChanged;
    if (ICPCutOff != 0 && NumOfPGOICallPromotion >= ICPCutOff) {
      DEBUG(dbgs() << " Stop: Cutoff reached.\n");
      break;
    }
  }
  return Changed;
}

bool PGOIndirectCallPromotionLegacyPass::runOnModule(Module &M) {
  // Command-line option has the priority for InLTO.
  return promoteIndirectCalls(M, InLTO | ICPLTOMode,
                              SamplePGO | ICPSamplePGOMode);
}

PreservedAnalyses PGOIndirectCallPromotion::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  if (!promoteIndirectCalls(M, InLTO | ICPLTOMode,
                            SamplePGO | ICPSamplePGOMode))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

namespace {
class MemOPSizeOpt : public InstVisitor<MemOPSizeOpt> {
public:
  MemOPSizeOpt(Function &Func, BlockFrequencyInfo &BFI)
      : Func(Func), BFI(BFI), Changed(false) {
    ValueDataArray =
        llvm::make_unique<InstrProfValueData[]>(MemOPMaxVersion + 2);
    // Get the MemOPSize range information from option MemOPSizeRange,
    getMemOPSizeRangeFromOption(MemOPSizeRange, PreciseRangeStart,
                                PreciseRangeLast);
  }
  bool isChanged() const { return Changed; }
  void perform() {
    WorkList.clear();
    visit(Func);

    for (auto &MI : WorkList) {
      ++NumOfPGOMemOPAnnotate;
      if (perform(MI)) {
        Changed = true;
        ++NumOfPGOMemOPOpt;
        DEBUG(dbgs() << "MemOP calls: " << MI->getCalledFunction()->getName()
                     << "is Transformed.\n");
      }
    }
  }

  void visitMemIntrinsic(MemIntrinsic &MI) {
    Value *Length = MI.getLength();
    // Not perform on constant length calls.
    if (dyn_cast<ConstantInt>(Length))
      return;
    WorkList.push_back(&MI);
  }

private:
  Function &Func;
  BlockFrequencyInfo &BFI;
  bool Changed;
  std::vector<MemIntrinsic *> WorkList;
  // Start of the previse range.
  int64_t PreciseRangeStart;
  // Last value of the previse range.
  int64_t PreciseRangeLast;
  // The space to read the profile annotation.
  std::unique_ptr<InstrProfValueData[]> ValueDataArray;
  bool perform(MemIntrinsic *MI);

  // This kind shows which group the value falls in. For PreciseValue, we have
  // the profile count for that value. LargeGroup groups the values that are in
  // range [LargeValue, +inf). NonLargeGroup groups the rest of values.
  enum MemOPSizeKind { PreciseValue, NonLargeGroup, LargeGroup };

  MemOPSizeKind getMemOPSizeKind(int64_t Value) const {
    if (Value == MemOPSizeLarge && MemOPSizeLarge != 0)
      return LargeGroup;
    if (Value == PreciseRangeLast + 1)
      return NonLargeGroup;
    return PreciseValue;
  }
};

static const char *getMIName(const MemIntrinsic *MI) {
  switch (MI->getIntrinsicID()) {
  case Intrinsic::memcpy:
    return "memcpy";
  case Intrinsic::memmove:
    return "memmove";
  case Intrinsic::memset:
    return "memset";
  default:
    return "unknown";
  }
}

static bool isProfitable(uint64_t Count, uint64_t TotalCount) {
  assert(Count <= TotalCount);
  if (Count < MemOPCountThreshold)
    return false;
  if (Count < TotalCount * MemOPPercentThreshold / 100)
    return false;
  return true;
}

static inline uint64_t getScaledCount(uint64_t Count, uint64_t Num,
                                      uint64_t Denom) {
  if (!MemOPScaleCount)
    return Count;
  bool Overflowed;
  uint64_t ScaleCount = SaturatingMultiply(Count, Num, &Overflowed);
  return ScaleCount / Denom;
}

bool MemOPSizeOpt::perform(MemIntrinsic *MI) {
  assert(MI);
  if (MI->getIntrinsicID() == Intrinsic::memmove)
    return false;

  uint32_t NumVals, MaxNumPromotions = MemOPMaxVersion + 2;
  uint64_t TotalCount;
  if (!getValueProfDataFromInst(*MI, IPVK_MemOPSize, MaxNumPromotions,
                                ValueDataArray.get(), NumVals, TotalCount))
    return false;

  uint64_t ActualCount = TotalCount;
  uint64_t SavedTotalCount = TotalCount;
  if (MemOPScaleCount) {
    auto BBEdgeCount = BFI.getBlockProfileCount(MI->getParent());
    if (!BBEdgeCount)
      return false;
    ActualCount = *BBEdgeCount;
  }

  if (ActualCount < MemOPCountThreshold)
    return false;

  ArrayRef<InstrProfValueData> VDs(ValueDataArray.get(), NumVals);
  TotalCount = ActualCount;
  if (MemOPScaleCount)
    DEBUG(dbgs() << "Scale counts: numberator = " << ActualCount
                 << " denominator = " << SavedTotalCount << "\n");

  // Keeping track of the count of the default case:
  uint64_t RemainCount = TotalCount;
  SmallVector<uint64_t, 16> SizeIds;
  SmallVector<uint64_t, 16> CaseCounts;
  uint64_t MaxCount = 0;
  unsigned Version = 0;
  // Default case is in the front -- save the slot here.
  CaseCounts.push_back(0);
  for (auto &VD : VDs) {
    int64_t V = VD.Value;
    uint64_t C = VD.Count;
    if (MemOPScaleCount)
      C = getScaledCount(C, ActualCount, SavedTotalCount);

    // Only care precise value here.
    if (getMemOPSizeKind(V) != PreciseValue)
      continue;

    // ValueCounts are sorted on the count. Break at the first un-profitable
    // value.
    if (!isProfitable(C, RemainCount))
      break;

    SizeIds.push_back(V);
    CaseCounts.push_back(C);
    if (C > MaxCount)
      MaxCount = C;

    assert(RemainCount >= C);
    RemainCount -= C;

    if (++Version > MemOPMaxVersion && MemOPMaxVersion != 0)
      break;
  }

  if (Version == 0)
    return false;

  CaseCounts[0] = RemainCount;
  if (RemainCount > MaxCount)
    MaxCount = RemainCount;

  uint64_t SumForOpt = TotalCount - RemainCount;
  DEBUG(dbgs() << "Read one memory intrinsic profile: " << SumForOpt << " vs "
               << TotalCount << "\n");
  DEBUG(
      for (auto &VD
           : VDs) { dbgs() << "  (" << VD.Value << "," << VD.Count << ")\n"; });

  DEBUG(dbgs() << "Optimize one memory intrinsic call to " << Version
               << " Versions\n");

  // mem_op(..., size)
  // ==>
  // switch (size) {
  //   case s1:
  //      mem_op(..., s1);
  //      goto merge_bb;
  //   case s2:
  //      mem_op(..., s2);
  //      goto merge_bb;
  //   ...
  //   default:
  //      mem_op(..., size);
  //      goto merge_bb;
  // }
  // merge_bb:

  BasicBlock *BB = MI->getParent();
  DEBUG(dbgs() << "\n\n== Basic Block Before ==\n");
  DEBUG(dbgs() << *BB << "\n");

  BasicBlock *DefaultBB = SplitBlock(BB, MI);
  BasicBlock::iterator It(*MI);
  ++It;
  assert(It != DefaultBB->end());
  BasicBlock *MergeBB = SplitBlock(DefaultBB, &(*It));
  DefaultBB->setName("MemOP.Default");
  MergeBB->setName("MemOP.Merge");

  auto &Ctx = Func.getContext();
  IRBuilder<> IRB(BB);
  BB->getTerminator()->eraseFromParent();
  Value *SizeVar = MI->getLength();
  SwitchInst *SI = IRB.CreateSwitch(SizeVar, DefaultBB, SizeIds.size());

  // Clear the value profile data.
  MI->setMetadata(LLVMContext::MD_prof, nullptr);

  DEBUG(dbgs() << "\n\n== Basic Block After==\n");

  for (uint64_t SizeId : SizeIds) {
    ConstantInt *CaseSizeId = ConstantInt::get(Type::getInt64Ty(Ctx), SizeId);
    BasicBlock *CaseBB = BasicBlock::Create(
        Ctx, Twine("MemOP.Case.") + Twine(SizeId), &Func, DefaultBB);
    Instruction *NewInst = MI->clone();
    // Fix the argument.
    dyn_cast<MemIntrinsic>(NewInst)->setLength(CaseSizeId);
    CaseBB->getInstList().push_back(NewInst);
    IRBuilder<> IRBCase(CaseBB);
    IRBCase.CreateBr(MergeBB);
    SI->addCase(CaseSizeId, CaseBB);
    DEBUG(dbgs() << *CaseBB << "\n");
  }
  setProfMetadata(Func.getParent(), SI, CaseCounts, MaxCount);

  DEBUG(dbgs() << *BB << "\n");
  DEBUG(dbgs() << *DefaultBB << "\n");
  DEBUG(dbgs() << *MergeBB << "\n");

  emitOptimizationRemark(Func.getContext(), "memop-opt", Func,
                         MI->getDebugLoc(),
                         Twine("optimize ") + getMIName(MI) + " with count " +
                             Twine(SumForOpt) + " out of " + Twine(TotalCount) +
                             " for " + Twine(Version) + " versions");

  return true;
}
} // namespace

static bool PGOMemOPSizeOptImpl(Function &F, BlockFrequencyInfo &BFI) {
  if (DisableMemOPOPT)
    return false;

  if (F.hasFnAttribute(Attribute::OptimizeForSize))
    return false;
  MemOPSizeOpt MemOPSizeOpt(F, BFI);
  MemOPSizeOpt.perform();
  return MemOPSizeOpt.isChanged();
}

bool PGOMemOPSizeOptLegacyPass::runOnFunction(Function &F) {
  BlockFrequencyInfo &BFI =
      getAnalysis<BlockFrequencyInfoWrapperPass>().getBFI();
  return PGOMemOPSizeOptImpl(F, BFI);
}

namespace llvm {
char &PGOMemOPSizeOptID = PGOMemOPSizeOptLegacyPass::ID;

PreservedAnalyses PGOMemOPSizeOpt::run(Function &F,
                                       FunctionAnalysisManager &FAM) {
  auto &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);
  bool Changed = PGOMemOPSizeOptImpl(F, BFI);
  if (!Changed)
    return PreservedAnalyses::all();
  auto  PA = PreservedAnalyses();
  PA.preserve<GlobalsAA>();
  return PA;
}
} // namespace llvm
