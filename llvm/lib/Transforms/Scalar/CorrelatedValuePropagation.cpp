//===- CorrelatedValuePropagation.cpp - Propagate CFG-derived info --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Correlated Value Propagation pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "correlated-value-propagation"

STATISTIC(NumPhis,      "Number of phis propagated");
STATISTIC(NumPhiCommon, "Number of phis deleted via common incoming value");
STATISTIC(NumSelects,   "Number of selects propagated");
STATISTIC(NumMemAccess, "Number of memory access targets propagated");
STATISTIC(NumCmps,      "Number of comparisons propagated");
STATISTIC(NumReturns,   "Number of return values propagated");
STATISTIC(NumDeadCases, "Number of switch cases removed");
STATISTIC(NumSDivs,     "Number of sdiv converted to udiv");
STATISTIC(NumUDivs,     "Number of udivs whose width was decreased");
STATISTIC(NumAShrs,     "Number of ashr converted to lshr");
STATISTIC(NumSRems,     "Number of srem converted to urem");
STATISTIC(NumSExt,      "Number of sext converted to zext");
STATISTIC(NumAnd,       "Number of ands removed");
STATISTIC(NumNW,        "Number of no-wrap deductions");
STATISTIC(NumNSW,       "Number of no-signed-wrap deductions");
STATISTIC(NumNUW,       "Number of no-unsigned-wrap deductions");
STATISTIC(NumAddNW,     "Number of no-wrap deductions for add");
STATISTIC(NumAddNSW,    "Number of no-signed-wrap deductions for add");
STATISTIC(NumAddNUW,    "Number of no-unsigned-wrap deductions for add");
STATISTIC(NumSubNW,     "Number of no-wrap deductions for sub");
STATISTIC(NumSubNSW,    "Number of no-signed-wrap deductions for sub");
STATISTIC(NumSubNUW,    "Number of no-unsigned-wrap deductions for sub");
STATISTIC(NumMulNW,     "Number of no-wrap deductions for mul");
STATISTIC(NumMulNSW,    "Number of no-signed-wrap deductions for mul");
STATISTIC(NumMulNUW,    "Number of no-unsigned-wrap deductions for mul");
STATISTIC(NumShlNW,     "Number of no-wrap deductions for shl");
STATISTIC(NumShlNSW,    "Number of no-signed-wrap deductions for shl");
STATISTIC(NumShlNUW,    "Number of no-unsigned-wrap deductions for shl");
STATISTIC(NumOverflows, "Number of overflow checks removed");
STATISTIC(NumSaturating,
    "Number of saturating arithmetics converted to normal arithmetics");

static cl::opt<bool> DontAddNoWrapFlags("cvp-dont-add-nowrap-flags", cl::init(false));

namespace {

  class CorrelatedValuePropagation : public FunctionPass {
  public:
    static char ID;

    CorrelatedValuePropagation(): FunctionPass(ID) {
     initializeCorrelatedValuePropagationPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LazyValueInfoWrapperPass>();
      AU.addPreserved<GlobalsAAWrapperPass>();
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addPreserved<LazyValueInfoWrapperPass>();
    }
  };

} // end anonymous namespace

char CorrelatedValuePropagation::ID = 0;

INITIALIZE_PASS_BEGIN(CorrelatedValuePropagation, "correlated-propagation",
                "Value Propagation", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
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

  Constant *C = LVI->getConstant(S->getCondition(), S->getParent(), S);
  if (!C) return false;

  ConstantInt *CI = dyn_cast<ConstantInt>(C);
  if (!CI) return false;

  Value *ReplaceWith = S->getTrueValue();
  Value *Other = S->getFalseValue();
  if (!CI->isOne()) std::swap(ReplaceWith, Other);
  if (ReplaceWith == S) ReplaceWith = UndefValue::get(S->getType());

  S->replaceAllUsesWith(ReplaceWith);
  S->eraseFromParent();

  ++NumSelects;

  return true;
}

/// Try to simplify a phi with constant incoming values that match the edge
/// values of a non-constant value on all other edges:
/// bb0:
///   %isnull = icmp eq i8* %x, null
///   br i1 %isnull, label %bb2, label %bb1
/// bb1:
///   br label %bb2
/// bb2:
///   %r = phi i8* [ %x, %bb1 ], [ null, %bb0 ]
/// -->
///   %r = %x
static bool simplifyCommonValuePhi(PHINode *P, LazyValueInfo *LVI,
                                   DominatorTree *DT) {
  // Collect incoming constants and initialize possible common value.
  SmallVector<std::pair<Constant *, unsigned>, 4> IncomingConstants;
  Value *CommonValue = nullptr;
  for (unsigned i = 0, e = P->getNumIncomingValues(); i != e; ++i) {
    Value *Incoming = P->getIncomingValue(i);
    if (auto *IncomingConstant = dyn_cast<Constant>(Incoming)) {
      IncomingConstants.push_back(std::make_pair(IncomingConstant, i));
    } else if (!CommonValue) {
      // The potential common value is initialized to the first non-constant.
      CommonValue = Incoming;
    } else if (Incoming != CommonValue) {
      // There can be only one non-constant common value.
      return false;
    }
  }

  if (!CommonValue || IncomingConstants.empty())
    return false;

  // The common value must be valid in all incoming blocks.
  BasicBlock *ToBB = P->getParent();
  if (auto *CommonInst = dyn_cast<Instruction>(CommonValue))
    if (!DT->dominates(CommonInst, ToBB))
      return false;

  // We have a phi with exactly 1 variable incoming value and 1 or more constant
  // incoming values. See if all constant incoming values can be mapped back to
  // the same incoming variable value.
  for (auto &IncomingConstant : IncomingConstants) {
    Constant *C = IncomingConstant.first;
    BasicBlock *IncomingBB = P->getIncomingBlock(IncomingConstant.second);
    if (C != LVI->getConstantOnEdge(CommonValue, IncomingBB, ToBB, P))
      return false;
  }

  // All constant incoming values map to the same variable along the incoming
  // edges of the phi. The phi is unnecessary. However, we must drop all
  // poison-generating flags to ensure that no poison is propagated to the phi
  // location by performing this substitution.
  // Warning: If the underlying analysis changes, this may not be enough to
  //          guarantee that poison is not propagated.
  // TODO: We may be able to re-infer flags by re-analyzing the instruction.
  if (auto *CommonInst = dyn_cast<Instruction>(CommonValue))
    CommonInst->dropPoisonGeneratingFlags();
  P->replaceAllUsesWith(CommonValue);
  P->eraseFromParent();
  ++NumPhiCommon;
  return true;
}

static bool processPHI(PHINode *P, LazyValueInfo *LVI, DominatorTree *DT,
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

      LLVM_DEBUG(dbgs() << "CVP: Threading PHI over " << *SI << '\n');
    }

    P->setIncomingValue(i, V);
    Changed = true;
  }

  if (Value *V = SimplifyInstruction(P, SQ)) {
    P->replaceAllUsesWith(V);
    P->eraseFromParent();
    Changed = true;
  }

  if (!Changed)
    Changed = simplifyCommonValuePhi(P, LVI, DT);

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
static bool processCmp(CmpInst *Cmp, LazyValueInfo *LVI) {
  Value *Op0 = Cmp->getOperand(0);
  auto *C = dyn_cast<Constant>(Cmp->getOperand(1));
  if (!C)
    return false;

  // As a policy choice, we choose not to waste compile time on anything where
  // the comparison is testing local values.  While LVI can sometimes reason
  // about such cases, it's not its primary purpose.  We do make sure to do
  // the block local query for uses from terminator instructions, but that's
  // handled in the code for each terminator. As an exception, we allow phi
  // nodes, for which LVI can thread the condition into predecessors.
  auto *I = dyn_cast<Instruction>(Op0);
  if (I && I->getParent() == Cmp->getParent() && !isa<PHINode>(I))
    return false;

  LazyValueInfo::Tristate Result =
      LVI->getPredicateAt(Cmp->getPredicate(), Op0, C, Cmp);
  if (Result == LazyValueInfo::Unknown)
    return false;

  ++NumCmps;
  Constant *TorF = ConstantInt::get(Type::getInt1Ty(Cmp->getContext()), Result);
  Cmp->replaceAllUsesWith(TorF);
  Cmp->eraseFromParent();
  return true;
}

/// Simplify a switch instruction by removing cases which can never fire. If the
/// uselessness of a case could be determined locally then constant propagation
/// would already have figured it out. Instead, walk the predecessors and
/// statically evaluate cases based on information available on that edge. Cases
/// that cannot fire no matter what the incoming edge can safely be removed. If
/// a case fires on every incoming edge then the entire switch can be removed
/// and replaced with a branch to the case destination.
static bool processSwitch(SwitchInst *I, LazyValueInfo *LVI,
                          DominatorTree *DT) {
  DomTreeUpdater DTU(*DT, DomTreeUpdater::UpdateStrategy::Lazy);
  Value *Cond = I->getCondition();
  BasicBlock *BB = I->getParent();

  // If the condition was defined in same block as the switch then LazyValueInfo
  // currently won't say anything useful about it, though in theory it could.
  if (isa<Instruction>(Cond) && cast<Instruction>(Cond)->getParent() == BB)
    return false;

  // If the switch is unreachable then trying to improve it is a waste of time.
  pred_iterator PB = pred_begin(BB), PE = pred_end(BB);
  if (PB == PE) return false;

  // Analyse each switch case in turn.
  bool Changed = false;
  DenseMap<BasicBlock*, int> SuccessorsCount;
  for (auto *Succ : successors(BB))
    SuccessorsCount[Succ]++;

  { // Scope for SwitchInstProfUpdateWrapper. It must not live during
    // ConstantFoldTerminator() as the underlying SwitchInst can be changed.
    SwitchInstProfUpdateWrapper SI(*I);

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
        BasicBlock *Succ = CI->getCaseSuccessor();
        Succ->removePredecessor(BB);
        CI = SI.removeCase(CI);
        CE = SI->case_end();

        // The condition can be modified by removePredecessor's PHI simplification
        // logic.
        Cond = SI->getCondition();

        ++NumDeadCases;
        Changed = true;
        if (--SuccessorsCount[Succ] == 0)
          DTU.applyUpdatesPermissive({{DominatorTree::Delete, BB, Succ}});
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

      // Increment the case iterator since we didn't delete it.
      ++CI;
    }
  }

  if (Changed)
    // If the switch has been simplified to the point where it can be replaced
    // by a branch then do so now.
    ConstantFoldTerminator(BB, /*DeleteDeadConditions = */ false,
                           /*TLI = */ nullptr, &DTU);
  return Changed;
}

// See if we can prove that the given binary op intrinsic will not overflow.
static bool willNotOverflow(BinaryOpIntrinsic *BO, LazyValueInfo *LVI) {
  ConstantRange LRange = LVI->getConstantRange(
      BO->getLHS(), BO->getParent(), BO);
  ConstantRange RRange = LVI->getConstantRange(
      BO->getRHS(), BO->getParent(), BO);
  ConstantRange NWRegion = ConstantRange::makeGuaranteedNoWrapRegion(
      BO->getBinaryOp(), RRange, BO->getNoWrapKind());
  return NWRegion.contains(LRange);
}

static void setDeducedOverflowingFlags(Value *V, Instruction::BinaryOps Opcode,
                                       bool NewNSW, bool NewNUW) {
  Statistic *OpcNW, *OpcNSW, *OpcNUW;
  switch (Opcode) {
  case Instruction::Add:
    OpcNW = &NumAddNW;
    OpcNSW = &NumAddNSW;
    OpcNUW = &NumAddNUW;
    break;
  case Instruction::Sub:
    OpcNW = &NumSubNW;
    OpcNSW = &NumSubNSW;
    OpcNUW = &NumSubNUW;
    break;
  case Instruction::Mul:
    OpcNW = &NumMulNW;
    OpcNSW = &NumMulNSW;
    OpcNUW = &NumMulNUW;
    break;
  case Instruction::Shl:
    OpcNW = &NumShlNW;
    OpcNSW = &NumShlNSW;
    OpcNUW = &NumShlNUW;
    break;
  default:
    llvm_unreachable("Will not be called with other binops");
  }

  auto *Inst = dyn_cast<Instruction>(V);
  if (NewNSW) {
    ++NumNW;
    ++*OpcNW;
    ++NumNSW;
    ++*OpcNSW;
    if (Inst)
      Inst->setHasNoSignedWrap();
  }
  if (NewNUW) {
    ++NumNW;
    ++*OpcNW;
    ++NumNUW;
    ++*OpcNUW;
    if (Inst)
      Inst->setHasNoUnsignedWrap();
  }
}

static bool processBinOp(BinaryOperator *BinOp, LazyValueInfo *LVI);

// Rewrite this with.overflow intrinsic as non-overflowing.
static void processOverflowIntrinsic(WithOverflowInst *WO, LazyValueInfo *LVI) {
  IRBuilder<> B(WO);
  Instruction::BinaryOps Opcode = WO->getBinaryOp();
  bool NSW = WO->isSigned();
  bool NUW = !WO->isSigned();

  Value *NewOp =
      B.CreateBinOp(Opcode, WO->getLHS(), WO->getRHS(), WO->getName());
  setDeducedOverflowingFlags(NewOp, Opcode, NSW, NUW);

  StructType *ST = cast<StructType>(WO->getType());
  Constant *Struct = ConstantStruct::get(ST,
      { UndefValue::get(ST->getElementType(0)),
        ConstantInt::getFalse(ST->getElementType(1)) });
  Value *NewI = B.CreateInsertValue(Struct, NewOp, 0);
  WO->replaceAllUsesWith(NewI);
  WO->eraseFromParent();
  ++NumOverflows;

  // See if we can infer the other no-wrap too.
  if (auto *BO = dyn_cast<BinaryOperator>(NewOp))
    processBinOp(BO, LVI);
}

static void processSaturatingInst(SaturatingInst *SI, LazyValueInfo *LVI) {
  Instruction::BinaryOps Opcode = SI->getBinaryOp();
  bool NSW = SI->isSigned();
  bool NUW = !SI->isSigned();
  BinaryOperator *BinOp = BinaryOperator::Create(
      Opcode, SI->getLHS(), SI->getRHS(), SI->getName(), SI);
  BinOp->setDebugLoc(SI->getDebugLoc());
  setDeducedOverflowingFlags(BinOp, Opcode, NSW, NUW);

  SI->replaceAllUsesWith(BinOp);
  SI->eraseFromParent();
  ++NumSaturating;

  // See if we can infer the other no-wrap too.
  if (auto *BO = dyn_cast<BinaryOperator>(BinOp))
    processBinOp(BO, LVI);
}

/// Infer nonnull attributes for the arguments at the specified callsite.
static bool processCallSite(CallSite CS, LazyValueInfo *LVI) {
  SmallVector<unsigned, 4> ArgNos;
  unsigned ArgNo = 0;

  if (auto *WO = dyn_cast<WithOverflowInst>(CS.getInstruction())) {
    if (WO->getLHS()->getType()->isIntegerTy() && willNotOverflow(WO, LVI)) {
      processOverflowIntrinsic(WO, LVI);
      return true;
    }
  }

  if (auto *SI = dyn_cast<SaturatingInst>(CS.getInstruction())) {
    if (SI->getType()->isIntegerTy() && willNotOverflow(SI, LVI)) {
      processSaturatingInst(SI, LVI);
      return true;
    }
  }

  // Deopt bundle operands are intended to capture state with minimal
  // perturbance of the code otherwise.  If we can find a constant value for
  // any such operand and remove a use of the original value, that's
  // desireable since it may allow further optimization of that value (e.g. via
  // single use rules in instcombine).  Since deopt uses tend to,
  // idiomatically, appear along rare conditional paths, it's reasonable likely
  // we may have a conditional fact with which LVI can fold.   
  if (auto DeoptBundle = CS.getOperandBundle(LLVMContext::OB_deopt)) {
    bool Progress = false;
    for (const Use &ConstU : DeoptBundle->Inputs) {
      Use &U = const_cast<Use&>(ConstU);
      Value *V = U.get();
      if (V->getType()->isVectorTy()) continue;
      if (isa<Constant>(V)) continue;

      Constant *C = LVI->getConstant(V, CS.getParent(), CS.getInstruction());
      if (!C) continue;
      U.set(C);
      Progress = true;
    }
    if (Progress)
      return true;
  }

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

static bool hasPositiveOperands(BinaryOperator *SDI, LazyValueInfo *LVI) {
  Constant *Zero = ConstantInt::get(SDI->getType(), 0);
  for (Value *O : SDI->operands()) {
    auto Result = LVI->getPredicateAt(ICmpInst::ICMP_SGE, O, Zero, SDI);
    if (Result != LazyValueInfo::True)
      return false;
  }
  return true;
}

/// Try to shrink a udiv/urem's width down to the smallest power of two that's
/// sufficient to contain its operands.
static bool processUDivOrURem(BinaryOperator *Instr, LazyValueInfo *LVI) {
  assert(Instr->getOpcode() == Instruction::UDiv ||
         Instr->getOpcode() == Instruction::URem);
  if (Instr->getType()->isVectorTy())
    return false;

  // Find the smallest power of two bitwidth that's sufficient to hold Instr's
  // operands.
  auto OrigWidth = Instr->getType()->getIntegerBitWidth();
  ConstantRange OperandRange(OrigWidth, /*isFullSet=*/false);
  for (Value *Operand : Instr->operands()) {
    OperandRange = OperandRange.unionWith(
        LVI->getConstantRange(Operand, Instr->getParent()));
  }
  // Don't shrink below 8 bits wide.
  unsigned NewWidth = std::max<unsigned>(
      PowerOf2Ceil(OperandRange.getUnsignedMax().getActiveBits()), 8);
  // NewWidth might be greater than OrigWidth if OrigWidth is not a power of
  // two.
  if (NewWidth >= OrigWidth)
    return false;

  ++NumUDivs;
  IRBuilder<> B{Instr};
  auto *TruncTy = Type::getIntNTy(Instr->getContext(), NewWidth);
  auto *LHS = B.CreateTruncOrBitCast(Instr->getOperand(0), TruncTy,
                                     Instr->getName() + ".lhs.trunc");
  auto *RHS = B.CreateTruncOrBitCast(Instr->getOperand(1), TruncTy,
                                     Instr->getName() + ".rhs.trunc");
  auto *BO = B.CreateBinOp(Instr->getOpcode(), LHS, RHS, Instr->getName());
  auto *Zext = B.CreateZExt(BO, Instr->getType(), Instr->getName() + ".zext");
  if (auto *BinOp = dyn_cast<BinaryOperator>(BO))
    if (BinOp->getOpcode() == Instruction::UDiv)
      BinOp->setIsExact(Instr->isExact());

  Instr->replaceAllUsesWith(Zext);
  Instr->eraseFromParent();
  return true;
}

static bool processSRem(BinaryOperator *SDI, LazyValueInfo *LVI) {
  if (SDI->getType()->isVectorTy() || !hasPositiveOperands(SDI, LVI))
    return false;

  ++NumSRems;
  auto *BO = BinaryOperator::CreateURem(SDI->getOperand(0), SDI->getOperand(1),
                                        SDI->getName(), SDI);
  BO->setDebugLoc(SDI->getDebugLoc());
  SDI->replaceAllUsesWith(BO);
  SDI->eraseFromParent();

  // Try to process our new urem.
  processUDivOrURem(BO, LVI);

  return true;
}

/// See if LazyValueInfo's ability to exploit edge conditions or range
/// information is sufficient to prove the both operands of this SDiv are
/// positive.  If this is the case, replace the SDiv with a UDiv. Even for local
/// conditions, this can sometimes prove conditions instcombine can't by
/// exploiting range information.
static bool processSDiv(BinaryOperator *SDI, LazyValueInfo *LVI) {
  if (SDI->getType()->isVectorTy() || !hasPositiveOperands(SDI, LVI))
    return false;

  ++NumSDivs;
  auto *BO = BinaryOperator::CreateUDiv(SDI->getOperand(0), SDI->getOperand(1),
                                        SDI->getName(), SDI);
  BO->setDebugLoc(SDI->getDebugLoc());
  BO->setIsExact(SDI->isExact());
  SDI->replaceAllUsesWith(BO);
  SDI->eraseFromParent();

  // Try to simplify our new udiv.
  processUDivOrURem(BO, LVI);

  return true;
}

static bool processAShr(BinaryOperator *SDI, LazyValueInfo *LVI) {
  if (SDI->getType()->isVectorTy())
    return false;

  Constant *Zero = ConstantInt::get(SDI->getType(), 0);
  if (LVI->getPredicateAt(ICmpInst::ICMP_SGE, SDI->getOperand(0), Zero, SDI) !=
      LazyValueInfo::True)
    return false;

  ++NumAShrs;
  auto *BO = BinaryOperator::CreateLShr(SDI->getOperand(0), SDI->getOperand(1),
                                        SDI->getName(), SDI);
  BO->setDebugLoc(SDI->getDebugLoc());
  BO->setIsExact(SDI->isExact());
  SDI->replaceAllUsesWith(BO);
  SDI->eraseFromParent();

  return true;
}

static bool processSExt(SExtInst *SDI, LazyValueInfo *LVI) {
  if (SDI->getType()->isVectorTy())
    return false;

  Value *Base = SDI->getOperand(0);

  Constant *Zero = ConstantInt::get(Base->getType(), 0);
  if (LVI->getPredicateAt(ICmpInst::ICMP_SGE, Base, Zero, SDI) !=
      LazyValueInfo::True)
    return false;

  ++NumSExt;
  auto *ZExt =
      CastInst::CreateZExtOrBitCast(Base, SDI->getType(), SDI->getName(), SDI);
  ZExt->setDebugLoc(SDI->getDebugLoc());
  SDI->replaceAllUsesWith(ZExt);
  SDI->eraseFromParent();

  return true;
}

static bool processBinOp(BinaryOperator *BinOp, LazyValueInfo *LVI) {
  using OBO = OverflowingBinaryOperator;

  if (DontAddNoWrapFlags)
    return false;

  if (BinOp->getType()->isVectorTy())
    return false;

  bool NSW = BinOp->hasNoSignedWrap();
  bool NUW = BinOp->hasNoUnsignedWrap();
  if (NSW && NUW)
    return false;

  BasicBlock *BB = BinOp->getParent();

  Instruction::BinaryOps Opcode = BinOp->getOpcode();
  Value *LHS = BinOp->getOperand(0);
  Value *RHS = BinOp->getOperand(1);

  ConstantRange LRange = LVI->getConstantRange(LHS, BB, BinOp);
  ConstantRange RRange = LVI->getConstantRange(RHS, BB, BinOp);

  bool Changed = false;
  bool NewNUW = false, NewNSW = false;
  if (!NUW) {
    ConstantRange NUWRange = ConstantRange::makeGuaranteedNoWrapRegion(
        Opcode, RRange, OBO::NoUnsignedWrap);
    NewNUW = NUWRange.contains(LRange);
    Changed |= NewNUW;
  }
  if (!NSW) {
    ConstantRange NSWRange = ConstantRange::makeGuaranteedNoWrapRegion(
        Opcode, RRange, OBO::NoSignedWrap);
    NewNSW = NSWRange.contains(LRange);
    Changed |= NewNSW;
  }

  setDeducedOverflowingFlags(BinOp, Opcode, NewNSW, NewNUW);

  return Changed;
}

static bool processAnd(BinaryOperator *BinOp, LazyValueInfo *LVI) {
  if (BinOp->getType()->isVectorTy())
    return false;

  // Pattern match (and lhs, C) where C includes a superset of bits which might
  // be set in lhs.  This is a common truncation idiom created by instcombine.
  BasicBlock *BB = BinOp->getParent();
  Value *LHS = BinOp->getOperand(0);
  ConstantInt *RHS = dyn_cast<ConstantInt>(BinOp->getOperand(1));
  if (!RHS || !RHS->getValue().isMask())
    return false;

  ConstantRange LRange = LVI->getConstantRange(LHS, BB, BinOp);
  if (!LRange.getUnsignedMax().ule(RHS->getValue()))
    return false;

  BinOp->replaceAllUsesWith(LHS);
  BinOp->eraseFromParent();
  NumAnd++;
  return true;
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

static bool runImpl(Function &F, LazyValueInfo *LVI, DominatorTree *DT,
                    const SimplifyQuery &SQ) {
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
        BBChanged |= processPHI(cast<PHINode>(II), LVI, DT, SQ);
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
      case Instruction::UDiv:
      case Instruction::URem:
        BBChanged |= processUDivOrURem(cast<BinaryOperator>(II), LVI);
        break;
      case Instruction::AShr:
        BBChanged |= processAShr(cast<BinaryOperator>(II), LVI);
        break;
      case Instruction::SExt:
        BBChanged |= processSExt(cast<SExtInst>(II), LVI);
        break;
      case Instruction::Add:
      case Instruction::Sub:
      case Instruction::Mul:
      case Instruction::Shl:
        BBChanged |= processBinOp(cast<BinaryOperator>(II), LVI);
        break;
      case Instruction::And:
        BBChanged |= processAnd(cast<BinaryOperator>(II), LVI);
        break;
      }
    }

    Instruction *Term = BB->getTerminator();
    switch (Term->getOpcode()) {
    case Instruction::Switch:
      BBChanged |= processSwitch(cast<SwitchInst>(Term), LVI, DT);
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
    }

    FnChanged |= BBChanged;
  }

  return FnChanged;
}

bool CorrelatedValuePropagation::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  LazyValueInfo *LVI = &getAnalysis<LazyValueInfoWrapperPass>().getLVI();
  DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  return runImpl(F, LVI, DT, getBestSimplifyQuery(*this, F));
}

PreservedAnalyses
CorrelatedValuePropagationPass::run(Function &F, FunctionAnalysisManager &AM) {
  LazyValueInfo *LVI = &AM.getResult<LazyValueAnalysis>(F);
  DominatorTree *DT = &AM.getResult<DominatorTreeAnalysis>(F);

  bool Changed = runImpl(F, LVI, DT, getBestSimplifyQuery(AM, F));

  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<GlobalsAA>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<LazyValueAnalysis>();
  return PA;
}
