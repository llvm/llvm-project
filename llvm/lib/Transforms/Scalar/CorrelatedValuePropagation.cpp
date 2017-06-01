//===- CorrelatedValuePropagation.cpp - Propagate CFG-derived info --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Correlated Value Propagation pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

#define DEBUG_TYPE "correlated-value-propagation"

STATISTIC(NumPhis,      "Number of phis propagated");
STATISTIC(NumSelects,   "Number of selects propagated");
STATISTIC(NumMemAccess, "Number of memory access targets propagated");
STATISTIC(NumCmps,      "Number of comparisons propagated");
STATISTIC(NumReturns,   "Number of return values propagated");
STATISTIC(NumDeadCases, "Number of switch cases removed");
STATISTIC(NumSDivs,     "Number of sdiv converted to udiv");
STATISTIC(NumAShrs,     "Number of ashr converted to lshr");
STATISTIC(NumSRems,     "Number of srem converted to urem");

static cl::opt<bool> DontProcessAdds("cvp-dont-process-adds", cl::init(true));

namespace {
  class CorrelatedValuePropagation : public FunctionPass {
  public:
    static char ID;
    CorrelatedValuePropagation(): FunctionPass(ID) {
     initializeCorrelatedValuePropagationPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<LazyValueInfoWrapperPass>();
      AU.addPreserved<GlobalsAAWrapperPass>();
    }
  };
}

char CorrelatedValuePropagation::ID = 0;
INITIALIZE_PASS_BEGIN(CorrelatedValuePropagation, "correlated-propagation",
                "Value Propagation", false, false)
INITIALIZE_PASS_DEPENDENCY(LazyValueInfoWrapperPass)
INITIALIZE_PASS_END(CorrelatedValuePropagation, "correlated-propagation",
                "Value Propagation", false, false)

// Public interface to the Value Propagation pass
Pass *llvm::createCorrelatedValuePropagationPass() {
  return new CorrelatedValuePropagation();
}

static bool processSelect(SelectInst *S, LazyValueInfo *LVI) {
  if (S->getType()->isVectorTy()) return false;
  if (isa<Constant>(S->getOperand(0))) return false;

  Constant *C = LVI->getConstant(S->getOperand(0), S->getParent(), S);
  if (!C) return false;

  ConstantInt *CI = dyn_cast<ConstantInt>(C);
  if (!CI) return false;

  Value *ReplaceWith = S->getOperand(1);
  Value *Other = S->getOperand(2);
  if (!CI->isOne()) std::swap(ReplaceWith, Other);
  if (ReplaceWith == S) ReplaceWith = UndefValue::get(S->getType());

  S->replaceAllUsesWith(ReplaceWith);
  S->eraseFromParent();

  ++NumSelects;

  return true;
}

static bool processPHI(PHINode *P, LazyValueInfo *LVI,
                       const SimplifyQuery &SQ) {
  bool Changed = false;

  BasicBlock *BB = P->getParent();
  for (unsigned i = 0, e = P->getNumIncomingValues(); i < e; ++i) {
    Value *Incoming = P->getIncomingValue(i);
    if (isa<Constant>(Incoming)) continue;

    Value *V = LVI->getConstantOnEdge(Incoming, P->getIncomingBlock(i), BB, P);

    // Look if the incoming value is a select with a scalar condition for which
    // LVI can tells us the value. In that case replace the incoming value with
    // the appropriate value of the select. This often allows us to remove the
    // select later.
    if (!V) {
      SelectInst *SI = dyn_cast<SelectInst>(Incoming);
      if (!SI) continue;

      Value *Condition = SI->getCondition();
      if (!Condition->getType()->isVectorTy()) {
        if (Constant *C = LVI->getConstantOnEdge(
                Condition, P->getIncomingBlock(i), BB, P)) {
          if (C->isOneValue()) {
            V = SI->getTrueValue();
          } else if (C->isZeroValue()) {
            V = SI->getFalseValue();
          }
          // Once LVI learns to handle vector types, we could also add support
          // for vector type constants that are not all zeroes or all ones.
        }
      }

      // Look if the select has a constant but LVI tells us that the incoming
      // value can never be that constant. In that case replace the incoming
      // value with the other value of the select. This often allows us to
      // remove the select later.
      if (!V) {
        Constant *C = dyn_cast<Constant>(SI->getFalseValue());
        if (!C) continue;

        if (LVI->getPredicateOnEdge(ICmpInst::ICMP_EQ, SI, C,
              P->getIncomingBlock(i), BB, P) !=
            LazyValueInfo::False)
          continue;
        V = SI->getTrueValue();
      }

      DEBUG(dbgs() << "CVP: Threading PHI over " << *SI << '\n');
    }

    P->setIncomingValue(i, V);
    Changed = true;
  }

  if (Value *V = SimplifyInstruction(P, SQ)) {
    P->replaceAllUsesWith(V);
    P->eraseFromParent();
    Changed = true;
  }

  if (Changed)
    ++NumPhis;

  return Changed;
}

static bool processMemAccess(Instruction *I, LazyValueInfo *LVI) {
  Value *Pointer = nullptr;
  if (LoadInst *L = dyn_cast<LoadInst>(I))
    Pointer = L->getPointerOperand();
  else
    Pointer = cast<StoreInst>(I)->getPointerOperand();

  if (isa<Constant>(Pointer)) return false;

  Constant *C = LVI->getConstant(Pointer, I->getParent(), I);
  if (!C) return false;

  ++NumMemAccess;
  I->replaceUsesOfWith(Pointer, C);
  return true;
}

/// See if LazyValueInfo's ability to exploit edge conditions or range
/// information is sufficient to prove this comparison. Even for local
/// conditions, this can sometimes prove conditions instcombine can't by
/// exploiting range information.
static bool processCmp(CmpInst *C, LazyValueInfo *LVI) {
  Value *Op0 = C->getOperand(0);
  Constant *Op1 = dyn_cast<Constant>(C->getOperand(1));
  if (!Op1) return false;

  // As a policy choice, we choose not to waste compile time on anything where
  // the comparison is testing local values.  While LVI can sometimes reason
  // about such cases, it's not its primary purpose.  We do make sure to do
  // the block local query for uses from terminator instructions, but that's
  // handled in the code for each terminator.
  auto *I = dyn_cast<Instruction>(Op0);
  if (I && I->getParent() == C->getParent())
    return false;

  LazyValueInfo::Tristate Result =
    LVI->getPredicateAt(C->getPredicate(), Op0, Op1, C);
  if (Result == LazyValueInfo::Unknown) return false;

  ++NumCmps;
  if (Result == LazyValueInfo::True)
    C->replaceAllUsesWith(ConstantInt::getTrue(C->getContext()));
  else
    C->replaceAllUsesWith(ConstantInt::getFalse(C->getContext()));
  C->eraseFromParent();

  return true;
}

/// Simplify a switch instruction by removing cases which can never fire. If the
/// uselessness of a case could be determined locally then constant propagation
/// would already have figured it out. Instead, walk the predecessors and
/// statically evaluate cases based on information available on that edge. Cases
/// that cannot fire no matter what the incoming edge can safely be removed. If
/// a case fires on every incoming edge then the entire switch can be removed
/// and replaced with a branch to the case destination.
static bool processSwitch(SwitchInst *SI, LazyValueInfo *LVI) {
  Value *Cond = SI->getCondition();
  BasicBlock *BB = SI->getParent();

  // If the condition was defined in same block as the switch then LazyValueInfo
  // currently won't say anything useful about it, though in theory it could.
  if (isa<Instruction>(Cond) && cast<Instruction>(Cond)->getParent() == BB)
    return false;

  // If the switch is unreachable then trying to improve it is a waste of time.
  pred_iterator PB = pred_begin(BB), PE = pred_end(BB);
  if (PB == PE) return false;

  // Analyse each switch case in turn.  This is done in reverse order so that
  // removing a case doesn't cause trouble for the iteration.
  bool Changed = false;
  for (auto CI = SI->case_begin(), CE = SI->case_end(); CI != CE;) {
    ConstantInt *Case = CI->getCaseValue();

    // Check to see if the switch condition is equal to/not equal to the case
    // value on every incoming edge, equal/not equal being the same each time.
    LazyValueInfo::Tristate State = LazyValueInfo::Unknown;
    for (pred_iterator PI = PB; PI != PE; ++PI) {
      // Is the switch condition equal to the case value?
      LazyValueInfo::Tristate Value = LVI->getPredicateOnEdge(CmpInst::ICMP_EQ,
                                                              Cond, Case, *PI,
                                                              BB, SI);
      // Give up on this case if nothing is known.
      if (Value == LazyValueInfo::Unknown) {
        State = LazyValueInfo::Unknown;
        break;
      }

      // If this was the first edge to be visited, record that all other edges
      // need to give the same result.
      if (PI == PB) {
        State = Value;
        continue;
      }

      // If this case is known to fire for some edges and known not to fire for
      // others then there is nothing we can do - give up.
      if (Value != State) {
        State = LazyValueInfo::Unknown;
        break;
      }
    }

    if (State == LazyValueInfo::False) {
      // This case never fires - remove it.
      CI->getCaseSuccessor()->removePredecessor(BB);
      CI = SI->removeCase(CI);
      CE = SI->case_end();

      // The condition can be modified by removePredecessor's PHI simplification
      // logic.
      Cond = SI->getCondition();

      ++NumDeadCases;
      Changed = true;
      continue;
    }
    if (State == LazyValueInfo::True) {
      // This case always fires.  Arrange for the switch to be turned into an
      // unconditional branch by replacing the switch condition with the case
      // value.
      SI->setCondition(Case);
      NumDeadCases += SI->getNumCases();
      Changed = true;
      break;
    }

    // Increment the case iterator sense we didn't delete it.
    ++CI;
  }

  if (Changed)
    // If the switch has been simplified to the point where it can be replaced
    // by a branch then do so now.
    ConstantFoldTerminator(BB);

  return Changed;
}

/// Infer nonnull attributes for the arguments at the specified callsite.
static bool processCallSite(CallSite CS, LazyValueInfo *LVI) {
  SmallVector<unsigned, 4> ArgNos;
  unsigned ArgNo = 0;

  for (Value *V : CS.args()) {
    PointerType *Type = dyn_cast<PointerType>(V->getType());
    // Try to mark pointer typed parameters as non-null.  We skip the
    // relatively expensive analysis for constants which are obviously either
    // null or non-null to start with.
    if (Type && !CS.paramHasAttr(ArgNo, Attribute::NonNull) &&
        !isa<Constant>(V) && 
        LVI->getPredicateAt(ICmpInst::ICMP_EQ, V,
                            ConstantPointerNull::get(Type),
                            CS.getInstruction()) == LazyValueInfo::False)
      ArgNos.push_back(ArgNo);
    ArgNo++;
  }

  assert(ArgNo == CS.arg_size() && "sanity check");

  if (ArgNos.empty())
    return false;

  AttributeList AS = CS.getAttributes();
  LLVMContext &Ctx = CS.getInstruction()->getContext();
  AS = AS.addParamAttribute(Ctx, ArgNos,
                            Attribute::get(Ctx, Attribute::NonNull));
  CS.setAttributes(AS);

  return true;
}

// Helper function to rewrite srem and sdiv. As a policy choice, we choose not
// to waste compile time on anything where the operands are local defs.  While
// LVI can sometimes reason about such cases, it's not its primary purpose.
static bool hasLocalDefs(BinaryOperator *SDI) {
  for (Value *O : SDI->operands()) {
    auto *I = dyn_cast<Instruction>(O);
    if (I && I->getParent() == SDI->getParent())
      return true;
  }
  return false;
}

static bool hasPositiveOperands(BinaryOperator *SDI, LazyValueInfo *LVI) {
  Constant *Zero = ConstantInt::get(SDI->getType(), 0);
  for (Value *O : SDI->operands()) {
    auto Result = LVI->getPredicateAt(ICmpInst::ICMP_SGE, O, Zero, SDI);
    if (Result != LazyValueInfo::True)
      return false;
  }
  return true;
}

static bool processSRem(BinaryOperator *SDI, LazyValueInfo *LVI) {
  if (SDI->getType()->isVectorTy() || hasLocalDefs(SDI) ||
      !hasPositiveOperands(SDI, LVI))
    return false;

  ++NumSRems;
  auto *BO = BinaryOperator::CreateURem(SDI->getOperand(0), SDI->getOperand(1),
                                        SDI->getName(), SDI);
  SDI->replaceAllUsesWith(BO);
  SDI->eraseFromParent();
  return true;
}

/// See if LazyValueInfo's ability to exploit edge conditions or range
/// information is sufficient to prove the both operands of this SDiv are
/// positive.  If this is the case, replace the SDiv with a UDiv. Even for local
/// conditions, this can sometimes prove conditions instcombine can't by
/// exploiting range information.
static bool processSDiv(BinaryOperator *SDI, LazyValueInfo *LVI) {
  if (SDI->getType()->isVectorTy() || hasLocalDefs(SDI) ||
      !hasPositiveOperands(SDI, LVI))
    return false;

  ++NumSDivs;
  auto *BO = BinaryOperator::CreateUDiv(SDI->getOperand(0), SDI->getOperand(1),
                                        SDI->getName(), SDI);
  BO->setIsExact(SDI->isExact());
  SDI->replaceAllUsesWith(BO);
  SDI->eraseFromParent();

  return true;
}

static bool processAShr(BinaryOperator *SDI, LazyValueInfo *LVI) {
  if (SDI->getType()->isVectorTy() || hasLocalDefs(SDI))
    return false;

  Constant *Zero = ConstantInt::get(SDI->getType(), 0);
  if (LVI->getPredicateAt(ICmpInst::ICMP_SGE, SDI->getOperand(0), Zero, SDI) !=
      LazyValueInfo::True)
    return false;

  ++NumAShrs;
  auto *BO = BinaryOperator::CreateLShr(SDI->getOperand(0), SDI->getOperand(1),
                                        SDI->getName(), SDI);
  BO->setIsExact(SDI->isExact());
  SDI->replaceAllUsesWith(BO);
  SDI->eraseFromParent();

  return true;
}

static bool processAdd(BinaryOperator *AddOp, LazyValueInfo *LVI) {
  typedef OverflowingBinaryOperator OBO;

  if (DontProcessAdds)
    return false;

  if (AddOp->getType()->isVectorTy() || hasLocalDefs(AddOp))
    return false;

  bool NSW = AddOp->hasNoSignedWrap();
  bool NUW = AddOp->hasNoUnsignedWrap();
  if (NSW && NUW)
    return false;

  BasicBlock *BB = AddOp->getParent();

  Value *LHS = AddOp->getOperand(0);
  Value *RHS = AddOp->getOperand(1);

  ConstantRange LRange = LVI->getConstantRange(LHS, BB, AddOp);

  // Initialize RRange only if we need it. If we know that guaranteed no wrap
  // range for the given LHS range is empty don't spend time calculating the
  // range for the RHS.
  Optional<ConstantRange> RRange;
  auto LazyRRange = [&] () {
      if (!RRange)
        RRange = LVI->getConstantRange(RHS, BB, AddOp);
      return RRange.getValue();
  };

  bool Changed = false;
  if (!NUW) {
    ConstantRange NUWRange = ConstantRange::makeGuaranteedNoWrapRegion(
        BinaryOperator::Add, LRange, OBO::NoUnsignedWrap);
    if (!NUWRange.isEmptySet()) {
      bool NewNUW = NUWRange.contains(LazyRRange());
      AddOp->setHasNoUnsignedWrap(NewNUW);
      Changed |= NewNUW;
    }
  }
  if (!NSW) {
    ConstantRange NSWRange = ConstantRange::makeGuaranteedNoWrapRegion(
        BinaryOperator::Add, LRange, OBO::NoSignedWrap);
    if (!NSWRange.isEmptySet()) {
      bool NewNSW = NSWRange.contains(LazyRRange());
      AddOp->setHasNoSignedWrap(NewNSW);
      Changed |= NewNSW;
    }
  }

  return Changed;
}

static Constant *getConstantAt(Value *V, Instruction *At, LazyValueInfo *LVI) {
  if (Constant *C = LVI->getConstant(V, At->getParent(), At))
    return C;

  // TODO: The following really should be sunk inside LVI's core algorithm, or
  // at least the outer shims around such.
  auto *C = dyn_cast<CmpInst>(V);
  if (!C) return nullptr;

  Value *Op0 = C->getOperand(0);
  Constant *Op1 = dyn_cast<Constant>(C->getOperand(1));
  if (!Op1) return nullptr;
  
  LazyValueInfo::Tristate Result =
    LVI->getPredicateAt(C->getPredicate(), Op0, Op1, At);
  if (Result == LazyValueInfo::Unknown)
    return nullptr;
  
  return (Result == LazyValueInfo::True) ?
    ConstantInt::getTrue(C->getContext()) :
    ConstantInt::getFalse(C->getContext());
}

static bool runImpl(Function &F, LazyValueInfo *LVI, const SimplifyQuery &SQ) {
  bool FnChanged = false;
  // Visiting in a pre-order depth-first traversal causes us to simplify early
  // blocks before querying later blocks (which require us to analyze early
  // blocks).  Eagerly simplifying shallow blocks means there is strictly less
  // work to do for deep blocks.  This also means we don't visit unreachable
  // blocks. 
  for (BasicBlock *BB : depth_first(&F.getEntryBlock())) {
    bool BBChanged = false;
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
      Instruction *II = &*BI++;
      switch (II->getOpcode()) {
      case Instruction::Select:
        BBChanged |= processSelect(cast<SelectInst>(II), LVI);
        break;
      case Instruction::PHI:
        BBChanged |= processPHI(cast<PHINode>(II), LVI, SQ);
        break;
      case Instruction::ICmp:
      case Instruction::FCmp:
        BBChanged |= processCmp(cast<CmpInst>(II), LVI);
        break;
      case Instruction::Load:
      case Instruction::Store:
        BBChanged |= processMemAccess(II, LVI);
        break;
      case Instruction::Call:
      case Instruction::Invoke:
        BBChanged |= processCallSite(CallSite(II), LVI);
        break;
      case Instruction::SRem:
        BBChanged |= processSRem(cast<BinaryOperator>(II), LVI);
        break;
      case Instruction::SDiv:
        BBChanged |= processSDiv(cast<BinaryOperator>(II), LVI);
        break;
      case Instruction::AShr:
        BBChanged |= processAShr(cast<BinaryOperator>(II), LVI);
        break;
      case Instruction::Add:
        BBChanged |= processAdd(cast<BinaryOperator>(II), LVI);
        break;
      }
    }

    Instruction *Term = BB->getTerminator();
    switch (Term->getOpcode()) {
    case Instruction::Switch:
      BBChanged |= processSwitch(cast<SwitchInst>(Term), LVI);
      break;
    case Instruction::Ret: {
      auto *RI = cast<ReturnInst>(Term);
      // Try to determine the return value if we can.  This is mainly here to
      // simplify the writing of unit tests, but also helps to enable IPO by
      // constant folding the return values of callees.
      auto *RetVal = RI->getReturnValue();
      if (!RetVal) break; // handle "ret void"
      if (isa<Constant>(RetVal)) break; // nothing to do
      if (auto *C = getConstantAt(RetVal, RI, LVI)) {
        ++NumReturns;
        RI->replaceUsesOfWith(RetVal, C);
        BBChanged = true;        
      }
    }
    };

    FnChanged |= BBChanged;
  }

  return FnChanged;
}

bool CorrelatedValuePropagation::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  LazyValueInfo *LVI = &getAnalysis<LazyValueInfoWrapperPass>().getLVI();
  return runImpl(F, LVI, getBestSimplifyQuery(*this, F));
}

PreservedAnalyses
CorrelatedValuePropagationPass::run(Function &F, FunctionAnalysisManager &AM) {

  LazyValueInfo *LVI = &AM.getResult<LazyValueAnalysis>(F);
  bool Changed = runImpl(F, LVI, getBestSimplifyQuery(AM, F));

  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<GlobalsAA>();
  return PA;
}
