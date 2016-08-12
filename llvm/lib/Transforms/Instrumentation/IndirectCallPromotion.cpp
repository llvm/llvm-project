//===-- IndirectCallPromotion.cpp - Promote indirect calls to direct calls ===//
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

  PGOIndirectCallPromotionLegacyPass(bool InLTO = false)
      : ModulePass(ID), InLTO(InLTO) {
    initializePGOIndirectCallPromotionLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  const char *getPassName() const override {
    return "PGOIndirectCallPromotion";
  }

private:
  bool runOnModule(Module &M) override;

  // If this pass is called in LTO. We need to special handling the PGOFuncName
  // for the static variables due to LTO's internalization.
  bool InLTO;
};
} // end anonymous namespace

char PGOIndirectCallPromotionLegacyPass::ID = 0;
INITIALIZE_PASS(PGOIndirectCallPromotionLegacyPass, "pgo-icall-prom",
                "Use PGO instrumentation profile to promote indirect calls to "
                "direct calls.",
                false, false)

ModulePass *llvm::createPGOIndirectCallPromotionLegacyPass(bool InLTO) {
  return new PGOIndirectCallPromotionLegacyPass(InLTO);
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

  enum TargetStatus {
    OK,                   // Should be able to promote.
    NotAvailableInModule, // Cannot find the target in current module.
    ReturnTypeMismatch,   // Return type mismatch b/w target and indirect-call.
    NumArgsMismatch,      // Number of arguments does not match.
    ArgTypeMismatch       // Type mismatch in the arguments (cannot bitcast).
  };

  // Test if we can legally promote this direct-call of Target.
  TargetStatus isPromotionLegal(Instruction *Inst, uint64_t Target,
                                Function *&F);

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

  // Main function that transforms Inst (either a indirect-call instruction, or
  // an invoke instruction , to a conditional call to F. This is like:
  //     if (Inst.CalledValue == F)
  //        F(...);
  //     else
  //        Inst(...);
  //     end
  // TotalCount is the profile count value that the instruction executes.
  // Count is the profile count value that F is the target function.
  // These two values are being used to update the branch weight.
  void promote(Instruction *Inst, Function *F, uint64_t Count,
               uint64_t TotalCount);

  // Promote a list of targets for one indirect-call callsite. Return
  // the number of promotions.
  uint32_t tryToPromote(Instruction *Inst,
                        const std::vector<PromotionCandidate> &Candidates,
                        uint64_t &TotalCount);

  static const char *StatusToString(const TargetStatus S) {
    switch (S) {
    case OK:
      return "OK to promote";
    case NotAvailableInModule:
      return "Cannot find the target";
    case ReturnTypeMismatch:
      return "Return type mismatch";
    case NumArgsMismatch:
      return "The number of arguments mismatch";
    case ArgTypeMismatch:
      return "Argument Type mismatch";
    }
    llvm_unreachable("Should not reach here");
  }

  // Noncopyable
  ICallPromotionFunc(const ICallPromotionFunc &other) = delete;
  ICallPromotionFunc &operator=(const ICallPromotionFunc &other) = delete;

public:
  ICallPromotionFunc(Function &Func, Module *Modu, InstrProfSymtab *Symtab)
      : F(Func), M(Modu), Symtab(Symtab) {
  }

  bool processFunction();
};
} // end anonymous namespace

ICallPromotionFunc::TargetStatus
ICallPromotionFunc::isPromotionLegal(Instruction *Inst, uint64_t Target,
                                     Function *&TargetFunction) {
  Function *DirectCallee = Symtab->getFunction(Target);
  if (DirectCallee == nullptr)
    return NotAvailableInModule;
  // Check the return type.
  Type *CallRetType = Inst->getType();
  if (!CallRetType->isVoidTy()) {
    Type *FuncRetType = DirectCallee->getReturnType();
    if (FuncRetType != CallRetType &&
        !CastInst::isBitCastable(FuncRetType, CallRetType))
      return ReturnTypeMismatch;
  }

  // Check if the arguments are compatible with the parameters
  FunctionType *DirectCalleeType = DirectCallee->getFunctionType();
  unsigned ParamNum = DirectCalleeType->getFunctionNumParams();
  CallSite CS(Inst);
  unsigned ArgNum = CS.arg_size();

  if (ParamNum != ArgNum && !DirectCalleeType->isVarArg())
    return NumArgsMismatch;

  for (unsigned I = 0; I < ParamNum; ++I) {
    Type *PTy = DirectCalleeType->getFunctionParamType(I);
    Type *ATy = CS.getArgument(I)->getType();
    if (PTy == ATy)
      continue;
    if (!CastInst::castIsValid(Instruction::BitCast, CS.getArgument(I), PTy))
      return ArgTypeMismatch;
  }

  DEBUG(dbgs() << " #" << NumOfPGOICallPromotion << " Promote the icall to "
               << Symtab->getFuncName(Target) << "\n");
  TargetFunction = DirectCallee;
  return OK;
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
    TargetStatus Status = isPromotionLegal(Inst, Target, TargetFunction);
    if (Status != OK) {
      StringRef TargetFuncName = Symtab->getFuncName(Target);
      const char *Reason = StatusToString(Status);
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
void ICallPromotionFunc::promote(Instruction *Inst, Function *DirectCallee,
                                 uint64_t Count, uint64_t TotalCount) {
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
      F.getContext(), "pgo-icall-prom", F, Inst->getDebugLoc(),
      Twine("Promote indirect call to ") + DirectCallee->getName() +
          " with count " + Twine(Count) + " out of " + Twine(TotalCount));
}

// Promote indirect-call to conditional direct-call for one callsite.
uint32_t ICallPromotionFunc::tryToPromote(
    Instruction *Inst, const std::vector<PromotionCandidate> &Candidates,
    uint64_t &TotalCount) {
  uint32_t NumPromoted = 0;

  for (auto &C : Candidates) {
    uint64_t Count = C.Count;
    promote(Inst, C.TargetFunction, Count, TotalCount);
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
static bool promoteIndirectCalls(Module &M, bool InLTO) {
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
    ICallPromotionFunc ICallPromotion(F, &M, &Symtab);
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
  return promoteIndirectCalls(M, InLTO | ICPLTOMode);
}

PreservedAnalyses PGOIndirectCallPromotion::run(Module &M, ModuleAnalysisManager &AM) {
  if (!promoteIndirectCalls(M, InLTO | ICPLTOMode))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
