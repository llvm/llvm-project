//===-- LoopUtils.cpp - Loop Utility functions -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines common loop utility functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "loop-utils"

bool RecurrenceDescriptor::areAllUsesIn(Instruction *I,
                                        SmallPtrSetImpl<Instruction *> &Set) {
  for (User::op_iterator Use = I->op_begin(), E = I->op_end(); Use != E; ++Use)
    if (!Set.count(dyn_cast<Instruction>(*Use)))
      return false;
  return true;
}

bool RecurrenceDescriptor::isIntegerRecurrenceKind(RecurrenceKind Kind) {
  switch (Kind) {
  default:
    break;
  case RK_IntegerAdd:
  case RK_IntegerMult:
  case RK_IntegerOr:
  case RK_IntegerAnd:
  case RK_IntegerXor:
  case RK_IntegerMinMax:
    return true;
  }
  return false;
}

bool RecurrenceDescriptor::isFloatingPointRecurrenceKind(RecurrenceKind Kind) {
  return (Kind != RK_NoRecurrence) && !isIntegerRecurrenceKind(Kind);
}

bool RecurrenceDescriptor::isArithmeticRecurrenceKind(RecurrenceKind Kind) {
  switch (Kind) {
  default:
    break;
  case RK_IntegerAdd:
  case RK_IntegerMult:
  case RK_FloatAdd:
  case RK_FloatMult:
    return true;
  }
  return false;
}

Instruction *
RecurrenceDescriptor::lookThroughAnd(PHINode *Phi, Type *&RT,
                                     SmallPtrSetImpl<Instruction *> &Visited,
                                     SmallPtrSetImpl<Instruction *> &CI) {
  if (!Phi->hasOneUse())
    return Phi;

  const APInt *M = nullptr;
  Instruction *I, *J = cast<Instruction>(Phi->use_begin()->getUser());

  // Matches either I & 2^x-1 or 2^x-1 & I. If we find a match, we update RT
  // with a new integer type of the corresponding bit width.
  if (match(J, m_CombineOr(m_And(m_Instruction(I), m_APInt(M)),
                           m_And(m_APInt(M), m_Instruction(I))))) {
    int32_t Bits = (*M + 1).exactLogBase2();
    if (Bits > 0) {
      RT = IntegerType::get(Phi->getContext(), Bits);
      Visited.insert(Phi);
      CI.insert(J);
      return J;
    }
  }
  return Phi;
}

bool RecurrenceDescriptor::getSourceExtensionKind(
    Instruction *Start, Instruction *Exit, Type *RT, bool &IsSigned,
    SmallPtrSetImpl<Instruction *> &Visited,
    SmallPtrSetImpl<Instruction *> &CI) {

  SmallVector<Instruction *, 8> Worklist;
  bool FoundOneOperand = false;
  unsigned DstSize = RT->getPrimitiveSizeInBits();
  Worklist.push_back(Exit);

  // Traverse the instructions in the reduction expression, beginning with the
  // exit value.
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    for (Use &U : I->operands()) {

      // Terminate the traversal if the operand is not an instruction, or we
      // reach the starting value.
      Instruction *J = dyn_cast<Instruction>(U.get());
      if (!J || J == Start)
        continue;

      // Otherwise, investigate the operation if it is also in the expression.
      if (Visited.count(J)) {
        Worklist.push_back(J);
        continue;
      }

      // If the operand is not in Visited, it is not a reduction operation, but
      // it does feed into one. Make sure it is either a single-use sign- or
      // zero-extend instruction.
      CastInst *Cast = dyn_cast<CastInst>(J);
      bool IsSExtInst = isa<SExtInst>(J);
      if (!Cast || !Cast->hasOneUse() || !(isa<ZExtInst>(J) || IsSExtInst))
        return false;

      // Ensure the source type of the extend is no larger than the reduction
      // type. It is not necessary for the types to be identical.
      unsigned SrcSize = Cast->getSrcTy()->getPrimitiveSizeInBits();
      if (SrcSize > DstSize)
        return false;

      // Furthermore, ensure that all such extends are of the same kind.
      if (FoundOneOperand) {
        if (IsSigned != IsSExtInst)
          return false;
      } else {
        FoundOneOperand = true;
        IsSigned = IsSExtInst;
      }

      // Lastly, if the source type of the extend matches the reduction type,
      // add the extend to CI so that we can avoid accounting for it in the
      // cost model.
      if (SrcSize == DstSize)
        CI.insert(Cast);
    }
  }
  return true;
}

bool RecurrenceDescriptor::AddReductionVar(PHINode *Phi, RecurrenceKind Kind,
                                           Loop *TheLoop, bool HasFunNoNaNAttr,
                                           RecurrenceDescriptor &RedDes) {
  if (Phi->getNumIncomingValues() != 2)
    return false;

  // Reduction variables are only found in the loop header block.
  if (Phi->getParent() != TheLoop->getHeader())
    return false;

  // Obtain the reduction start value from the value that comes from the loop
  // preheader.
  Value *RdxStart = Phi->getIncomingValueForBlock(TheLoop->getLoopPreheader());

  // ExitInstruction is the single value which is used outside the loop.
  // We only allow for a single reduction value to be used outside the loop.
  // This includes users of the reduction, variables (which form a cycle
  // which ends in the phi node).
  Instruction *ExitInstruction = nullptr;
  // Indicates that we found a reduction operation in our scan.
  bool FoundReduxOp = false;

  // We start with the PHI node and scan for all of the users of this
  // instruction. All users must be instructions that can be used as reduction
  // variables (such as ADD). We must have a single out-of-block user. The cycle
  // must include the original PHI.
  bool FoundStartPHI = false;

  // To recognize min/max patterns formed by a icmp select sequence, we store
  // the number of instruction we saw from the recognized min/max pattern,
  //  to make sure we only see exactly the two instructions.
  unsigned NumCmpSelectPatternInst = 0;
  InstDesc ReduxDesc(false, nullptr);

  // Data used for determining if the recurrence has been type-promoted.
  Type *RecurrenceType = Phi->getType();
  SmallPtrSet<Instruction *, 4> CastInsts;
  Instruction *Start = Phi;
  bool IsSigned = false;

  SmallPtrSet<Instruction *, 8> VisitedInsts;
  SmallVector<Instruction *, 8> Worklist;

  // Return early if the recurrence kind does not match the type of Phi. If the
  // recurrence kind is arithmetic, we attempt to look through AND operations
  // resulting from the type promotion performed by InstCombine.  Vector
  // operations are not limited to the legal integer widths, so we may be able
  // to evaluate the reduction in the narrower width.
  if (RecurrenceType->isFloatingPointTy()) {
    if (!isFloatingPointRecurrenceKind(Kind))
      return false;
  } else {
    if (!isIntegerRecurrenceKind(Kind))
      return false;
    if (isArithmeticRecurrenceKind(Kind))
      Start = lookThroughAnd(Phi, RecurrenceType, VisitedInsts, CastInsts);
  }

  Worklist.push_back(Start);
  VisitedInsts.insert(Start);

  // A value in the reduction can be used:
  //  - By the reduction:
  //      - Reduction operation:
  //        - One use of reduction value (safe).
  //        - Multiple use of reduction value (not safe).
  //      - PHI:
  //        - All uses of the PHI must be the reduction (safe).
  //        - Otherwise, not safe.
  //  - By one instruction outside of the loop (safe).
  //  - By further instructions outside of the loop (not safe).
  //  - By an instruction that is not part of the reduction (not safe).
  //    This is either:
  //      * An instruction type other than PHI or the reduction operation.
  //      * A PHI in the header other than the initial PHI.
  while (!Worklist.empty()) {
    Instruction *Cur = Worklist.back();
    Worklist.pop_back();

    // No Users.
    // If the instruction has no users then this is a broken chain and can't be
    // a reduction variable.
    if (Cur->use_empty())
      return false;

    bool IsAPhi = isa<PHINode>(Cur);

    // A header PHI use other than the original PHI.
    if (Cur != Phi && IsAPhi && Cur->getParent() == Phi->getParent())
      return false;

    // Reductions of instructions such as Div, and Sub is only possible if the
    // LHS is the reduction variable.
    if (!Cur->isCommutative() && !IsAPhi && !isa<SelectInst>(Cur) &&
        !isa<ICmpInst>(Cur) && !isa<FCmpInst>(Cur) &&
        !VisitedInsts.count(dyn_cast<Instruction>(Cur->getOperand(0))))
      return false;

    // Any reduction instruction must be of one of the allowed kinds. We ignore
    // the starting value (the Phi or an AND instruction if the Phi has been
    // type-promoted).
    if (Cur != Start) {
      ReduxDesc = isRecurrenceInstr(Cur, Kind, ReduxDesc, HasFunNoNaNAttr);
      if (!ReduxDesc.isRecurrence())
        return false;
    }

    // A reduction operation must only have one use of the reduction value.
    if (!IsAPhi && Kind != RK_IntegerMinMax && Kind != RK_FloatMinMax &&
        hasMultipleUsesOf(Cur, VisitedInsts))
      return false;

    // All inputs to a PHI node must be a reduction value.
    if (IsAPhi && Cur != Phi && !areAllUsesIn(Cur, VisitedInsts))
      return false;

    if (Kind == RK_IntegerMinMax &&
        (isa<ICmpInst>(Cur) || isa<SelectInst>(Cur)))
      ++NumCmpSelectPatternInst;
    if (Kind == RK_FloatMinMax && (isa<FCmpInst>(Cur) || isa<SelectInst>(Cur)))
      ++NumCmpSelectPatternInst;

    // Check  whether we found a reduction operator.
    FoundReduxOp |= !IsAPhi && Cur != Start;

    // Process users of current instruction. Push non-PHI nodes after PHI nodes
    // onto the stack. This way we are going to have seen all inputs to PHI
    // nodes once we get to them.
    SmallVector<Instruction *, 8> NonPHIs;
    SmallVector<Instruction *, 8> PHIs;
    for (User *U : Cur->users()) {
      Instruction *UI = cast<Instruction>(U);

      // Check if we found the exit user.
      BasicBlock *Parent = UI->getParent();
      if (!TheLoop->contains(Parent)) {
        // Exit if you find multiple outside users or if the header phi node is
        // being used. In this case the user uses the value of the previous
        // iteration, in which case we would loose "VF-1" iterations of the
        // reduction operation if we vectorize.
        if (ExitInstruction != nullptr || Cur == Phi)
          return false;

        // The instruction used by an outside user must be the last instruction
        // before we feed back to the reduction phi. Otherwise, we loose VF-1
        // operations on the value.
        if (std::find(Phi->op_begin(), Phi->op_end(), Cur) == Phi->op_end())
          return false;

        ExitInstruction = Cur;
        continue;
      }

      // Process instructions only once (termination). Each reduction cycle
      // value must only be used once, except by phi nodes and min/max
      // reductions which are represented as a cmp followed by a select.
      InstDesc IgnoredVal(false, nullptr);
      if (VisitedInsts.insert(UI).second) {
        if (isa<PHINode>(UI))
          PHIs.push_back(UI);
        else
          NonPHIs.push_back(UI);
      } else if (!isa<PHINode>(UI) &&
                 ((!isa<FCmpInst>(UI) && !isa<ICmpInst>(UI) &&
                   !isa<SelectInst>(UI)) ||
                  !isMinMaxSelectCmpPattern(UI, IgnoredVal).isRecurrence()))
        return false;

      // Remember that we completed the cycle.
      if (UI == Phi)
        FoundStartPHI = true;
    }
    Worklist.append(PHIs.begin(), PHIs.end());
    Worklist.append(NonPHIs.begin(), NonPHIs.end());
  }

  // This means we have seen one but not the other instruction of the
  // pattern or more than just a select and cmp.
  if ((Kind == RK_IntegerMinMax || Kind == RK_FloatMinMax) &&
      NumCmpSelectPatternInst != 2)
    return false;

  if (!FoundStartPHI || !FoundReduxOp || !ExitInstruction)
    return false;

  // If we think Phi may have been type-promoted, we also need to ensure that
  // all source operands of the reduction are either SExtInsts or ZEstInsts. If
  // so, we will be able to evaluate the reduction in the narrower bit width.
  if (Start != Phi)
    if (!getSourceExtensionKind(Start, ExitInstruction, RecurrenceType,
                                IsSigned, VisitedInsts, CastInsts))
      return false;

  // We found a reduction var if we have reached the original phi node and we
  // only have a single instruction with out-of-loop users.

  // The ExitInstruction(Instruction which is allowed to have out-of-loop users)
  // is saved as part of the RecurrenceDescriptor.

  // Save the description of this reduction variable.
  RecurrenceDescriptor RD(
      RdxStart, ExitInstruction, Kind, ReduxDesc.getMinMaxKind(),
      ReduxDesc.getUnsafeAlgebraInst(), RecurrenceType, IsSigned, CastInsts);
  RedDes = RD;

  return true;
}

/// Returns true if the instruction is a Select(ICmp(X, Y), X, Y) instruction
/// pattern corresponding to a min(X, Y) or max(X, Y).
RecurrenceDescriptor::InstDesc
RecurrenceDescriptor::isMinMaxSelectCmpPattern(Instruction *I, InstDesc &Prev) {

  assert((isa<ICmpInst>(I) || isa<FCmpInst>(I) || isa<SelectInst>(I)) &&
         "Expect a select instruction");
  Instruction *Cmp = nullptr;
  SelectInst *Select = nullptr;

  // We must handle the select(cmp()) as a single instruction. Advance to the
  // select.
  if ((Cmp = dyn_cast<ICmpInst>(I)) || (Cmp = dyn_cast<FCmpInst>(I))) {
    if (!Cmp->hasOneUse() || !(Select = dyn_cast<SelectInst>(*I->user_begin())))
      return InstDesc(false, I);
    return InstDesc(Select, Prev.getMinMaxKind());
  }

  // Only handle single use cases for now.
  if (!(Select = dyn_cast<SelectInst>(I)))
    return InstDesc(false, I);
  if (!(Cmp = dyn_cast<ICmpInst>(I->getOperand(0))) &&
      !(Cmp = dyn_cast<FCmpInst>(I->getOperand(0))))
    return InstDesc(false, I);
  if (!Cmp->hasOneUse())
    return InstDesc(false, I);

  Value *CmpLeft;
  Value *CmpRight;

  // Look for a min/max pattern.
  if (m_UMin(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_UIntMin);
  else if (m_UMax(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_UIntMax);
  else if (m_SMax(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_SIntMax);
  else if (m_SMin(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_SIntMin);
  else if (m_OrdFMin(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_FloatMin);
  else if (m_OrdFMax(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_FloatMax);
  else if (m_UnordFMin(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_FloatMin);
  else if (m_UnordFMax(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_FloatMax);

  return InstDesc(false, I);
}

RecurrenceDescriptor::InstDesc
RecurrenceDescriptor::isRecurrenceInstr(Instruction *I, RecurrenceKind Kind,
                                        InstDesc &Prev, bool HasFunNoNaNAttr) {
  bool FP = I->getType()->isFloatingPointTy();
  Instruction *UAI = Prev.getUnsafeAlgebraInst();
  if (!UAI && FP && !I->hasUnsafeAlgebra())
    UAI = I; // Found an unsafe (unvectorizable) algebra instruction.

  switch (I->getOpcode()) {
  default:
    return InstDesc(false, I);
  case Instruction::PHI:
    return InstDesc(I, Prev.getMinMaxKind(), Prev.getUnsafeAlgebraInst());
  case Instruction::Sub:
  case Instruction::Add:
    return InstDesc(Kind == RK_IntegerAdd, I);
  case Instruction::Mul:
    return InstDesc(Kind == RK_IntegerMult, I);
  case Instruction::And:
    return InstDesc(Kind == RK_IntegerAnd, I);
  case Instruction::Or:
    return InstDesc(Kind == RK_IntegerOr, I);
  case Instruction::Xor:
    return InstDesc(Kind == RK_IntegerXor, I);
  case Instruction::FMul:
    return InstDesc(Kind == RK_FloatMult, I, UAI);
  case Instruction::FSub:
  case Instruction::FAdd:
    return InstDesc(Kind == RK_FloatAdd, I, UAI);
  case Instruction::FCmp:
  case Instruction::ICmp:
  case Instruction::Select:
    if (Kind != RK_IntegerMinMax &&
        (!HasFunNoNaNAttr || Kind != RK_FloatMinMax))
      return InstDesc(false, I);
    return isMinMaxSelectCmpPattern(I, Prev);
  }
}

bool RecurrenceDescriptor::hasMultipleUsesOf(
    Instruction *I, SmallPtrSetImpl<Instruction *> &Insts) {
  unsigned NumUses = 0;
  for (User::op_iterator Use = I->op_begin(), E = I->op_end(); Use != E;
       ++Use) {
    if (Insts.count(dyn_cast<Instruction>(*Use)))
      ++NumUses;
    if (NumUses > 1)
      return true;
  }

  return false;
}
bool RecurrenceDescriptor::isReductionPHI(PHINode *Phi, Loop *TheLoop,
                                          RecurrenceDescriptor &RedDes) {

  BasicBlock *Header = TheLoop->getHeader();
  Function &F = *Header->getParent();
  bool HasFunNoNaNAttr =
      F.getFnAttribute("no-nans-fp-math").getValueAsString() == "true";

  if (AddReductionVar(Phi, RK_IntegerAdd, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found an ADD reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_IntegerMult, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found a MUL reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_IntegerOr, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found an OR reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_IntegerAnd, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found an AND reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_IntegerXor, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found a XOR reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_IntegerMinMax, TheLoop, HasFunNoNaNAttr,
                      RedDes)) {
    DEBUG(dbgs() << "Found a MINMAX reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_FloatMult, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found an FMult reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_FloatAdd, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found an FAdd reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_FloatMinMax, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found an float MINMAX reduction PHI." << *Phi << "\n");
    return true;
  }
  // Not a reduction of known type.
  return false;
}

bool RecurrenceDescriptor::isFirstOrderRecurrence(PHINode *Phi, Loop *TheLoop,
                                                  DominatorTree *DT) {

  // Ensure the phi node is in the loop header and has two incoming values.
  if (Phi->getParent() != TheLoop->getHeader() ||
      Phi->getNumIncomingValues() != 2)
    return false;

  // Ensure the loop has a preheader and a single latch block. The loop
  // vectorizer will need the latch to set up the next iteration of the loop.
  auto *Preheader = TheLoop->getLoopPreheader();
  auto *Latch = TheLoop->getLoopLatch();
  if (!Preheader || !Latch)
    return false;

  // Ensure the phi node's incoming blocks are the loop preheader and latch.
  if (Phi->getBasicBlockIndex(Preheader) < 0 ||
      Phi->getBasicBlockIndex(Latch) < 0)
    return false;

  // Get the previous value. The previous value comes from the latch edge while
  // the initial value comes form the preheader edge.
  auto *Previous = dyn_cast<Instruction>(Phi->getIncomingValueForBlock(Latch));
  if (!Previous || !TheLoop->contains(Previous) || isa<PHINode>(Previous))
    return false;

  // Ensure every user of the phi node is dominated by the previous value. The
  // dominance requirement ensures the loop vectorizer will not need to
  // vectorize the initial value prior to the first iteration of the loop.
  for (User *U : Phi->users())
    if (auto *I = dyn_cast<Instruction>(U))
      if (!DT->dominates(Previous, I))
        return false;

  return true;
}

/// This function returns the identity element (or neutral element) for
/// the operation K.
Constant *RecurrenceDescriptor::getRecurrenceIdentity(RecurrenceKind K,
                                                      Type *Tp) {
  switch (K) {
  case RK_IntegerXor:
  case RK_IntegerAdd:
  case RK_IntegerOr:
    // Adding, Xoring, Oring zero to a number does not change it.
    return ConstantInt::get(Tp, 0);
  case RK_IntegerMult:
    // Multiplying a number by 1 does not change it.
    return ConstantInt::get(Tp, 1);
  case RK_IntegerAnd:
    // AND-ing a number with an all-1 value does not change it.
    return ConstantInt::get(Tp, -1, true);
  case RK_FloatMult:
    // Multiplying a number by 1 does not change it.
    return ConstantFP::get(Tp, 1.0L);
  case RK_FloatAdd:
    // Adding zero to a number does not change it.
    return ConstantFP::get(Tp, 0.0L);
  default:
    llvm_unreachable("Unknown recurrence kind");
  }
}

/// This function translates the recurrence kind to an LLVM binary operator.
unsigned RecurrenceDescriptor::getRecurrenceBinOp(RecurrenceKind Kind) {
  switch (Kind) {
  case RK_IntegerAdd:
    return Instruction::Add;
  case RK_IntegerMult:
    return Instruction::Mul;
  case RK_IntegerOr:
    return Instruction::Or;
  case RK_IntegerAnd:
    return Instruction::And;
  case RK_IntegerXor:
    return Instruction::Xor;
  case RK_FloatMult:
    return Instruction::FMul;
  case RK_FloatAdd:
    return Instruction::FAdd;
  case RK_IntegerMinMax:
    return Instruction::ICmp;
  case RK_FloatMinMax:
    return Instruction::FCmp;
  default:
    llvm_unreachable("Unknown recurrence operation");
  }
}

Value *RecurrenceDescriptor::createMinMaxOp(IRBuilder<> &Builder,
                                            MinMaxRecurrenceKind RK,
                                            Value *Left, Value *Right) {
  CmpInst::Predicate P = CmpInst::ICMP_NE;
  switch (RK) {
  default:
    llvm_unreachable("Unknown min/max recurrence kind");
  case MRK_UIntMin:
    P = CmpInst::ICMP_ULT;
    break;
  case MRK_UIntMax:
    P = CmpInst::ICMP_UGT;
    break;
  case MRK_SIntMin:
    P = CmpInst::ICMP_SLT;
    break;
  case MRK_SIntMax:
    P = CmpInst::ICMP_SGT;
    break;
  case MRK_FloatMin:
    P = CmpInst::FCMP_OLT;
    break;
  case MRK_FloatMax:
    P = CmpInst::FCMP_OGT;
    break;
  }

  // We only match FP sequences with unsafe algebra, so we can unconditionally
  // set it on any generated instructions.
  IRBuilder<>::FastMathFlagGuard FMFG(Builder);
  FastMathFlags FMF;
  FMF.setUnsafeAlgebra();
  Builder.setFastMathFlags(FMF);

  Value *Cmp;
  if (RK == MRK_FloatMin || RK == MRK_FloatMax)
    Cmp = Builder.CreateFCmp(P, Left, Right, "rdx.minmax.cmp");
  else
    Cmp = Builder.CreateICmp(P, Left, Right, "rdx.minmax.cmp");

  Value *Select = Builder.CreateSelect(Cmp, Left, Right, "rdx.minmax.select");
  return Select;
}

InductionDescriptor::InductionDescriptor(Value *Start, InductionKind K,
                                         const SCEV *Step)
  : StartValue(Start), IK(K), Step(Step) {
  assert(IK != IK_NoInduction && "Not an induction");

  // Start value type should match the induction kind and the value
  // itself should not be null.
  assert(StartValue && "StartValue is null");
  assert((IK != IK_PtrInduction || StartValue->getType()->isPointerTy()) &&
         "StartValue is not a pointer for pointer induction");
  assert((IK != IK_IntInduction || StartValue->getType()->isIntegerTy()) &&
         "StartValue is not an integer for integer induction");

  // Check the Step Value. It should be non-zero integer value.
  assert((!getConstIntStepValue() || !getConstIntStepValue()->isZero()) &&
         "Step value is zero");

  assert((IK != IK_PtrInduction || getConstIntStepValue()) &&
         "Step value should be constant for pointer induction");
  assert(Step->getType()->isIntegerTy() && "StepValue is not an integer");
}

int InductionDescriptor::getConsecutiveDirection() const {
  ConstantInt *ConstStep = getConstIntStepValue();
  if (ConstStep && (ConstStep->isOne() || ConstStep->isMinusOne()))
    return ConstStep->getSExtValue();
  return 0;
}

ConstantInt *InductionDescriptor::getConstIntStepValue() const {
  if (isa<SCEVConstant>(Step))
    return dyn_cast<ConstantInt>(cast<SCEVConstant>(Step)->getValue());
  return nullptr;
}

Value *InductionDescriptor::transform(IRBuilder<> &B, Value *Index,
                                      ScalarEvolution *SE,
                                      const DataLayout& DL) const {

  SCEVExpander Exp(*SE, DL, "induction");
  switch (IK) {
  case IK_IntInduction: {
    assert(Index->getType() == StartValue->getType() &&
           "Index type does not match StartValue type");

    // FIXME: Theoretically, we can call getAddExpr() of ScalarEvolution
    // and calculate (Start + Index * Step) for all cases, without
    // special handling for "isOne" and "isMinusOne".
    // But in the real life the result code getting worse. We mix SCEV
    // expressions and ADD/SUB operations and receive redundant
    // intermediate values being calculated in different ways and
    // Instcombine is unable to reduce them all.

    if (getConstIntStepValue() &&
        getConstIntStepValue()->isMinusOne())
      return B.CreateSub(StartValue, Index);
    if (getConstIntStepValue() &&
        getConstIntStepValue()->isOne())
      return B.CreateAdd(StartValue, Index);
    const SCEV *S = SE->getAddExpr(SE->getSCEV(StartValue),
                                   SE->getMulExpr(Step, SE->getSCEV(Index)));
    return Exp.expandCodeFor(S, StartValue->getType(), &*B.GetInsertPoint());
  }
  case IK_PtrInduction: {
    assert(Index->getType() == Step->getType() &&
           "Index type does not match StepValue type");
    assert(isa<SCEVConstant>(Step) &&
           "Expected constant step for pointer induction");
    const SCEV *S = SE->getMulExpr(SE->getSCEV(Index), Step);
    Index = Exp.expandCodeFor(S, Index->getType(), &*B.GetInsertPoint());
    return B.CreateGEP(nullptr, StartValue, Index);
  }
  case IK_NoInduction:
    return nullptr;
  }
  llvm_unreachable("invalid enum");
}

bool InductionDescriptor::isInductionPHI(PHINode *Phi,
                                         PredicatedScalarEvolution &PSE,
                                         InductionDescriptor &D,
                                         bool Assume) {
  Type *PhiTy = Phi->getType();
  // We only handle integer and pointer inductions variables.
  if (!PhiTy->isIntegerTy() && !PhiTy->isPointerTy())
    return false;

  const SCEV *PhiScev = PSE.getSCEV(Phi);
  const auto *AR = dyn_cast<SCEVAddRecExpr>(PhiScev);

  // We need this expression to be an AddRecExpr.
  if (Assume && !AR)
    AR = PSE.getAsAddRec(Phi);

  if (!AR) {
    DEBUG(dbgs() << "LV: PHI is not a poly recurrence.\n");
    return false;
  }

  return isInductionPHI(Phi, PSE.getSE(), D, AR);
}

bool InductionDescriptor::isInductionPHI(PHINode *Phi,
                                         ScalarEvolution *SE,
                                         InductionDescriptor &D,
                                         const SCEV *Expr) {
  Type *PhiTy = Phi->getType();
  // We only handle integer and pointer inductions variables.
  if (!PhiTy->isIntegerTy() && !PhiTy->isPointerTy())
    return false;

  // Check that the PHI is consecutive.
  const SCEV *PhiScev = Expr ? Expr : SE->getSCEV(Phi);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PhiScev);

  if (!AR) {
    DEBUG(dbgs() << "LV: PHI is not a poly recurrence.\n");
    return false;
  }

  assert(AR->getLoop()->getHeader() == Phi->getParent() &&
         "PHI is an AddRec for a different loop?!");
  Value *StartValue =
    Phi->getIncomingValueForBlock(AR->getLoop()->getLoopPreheader());
  const SCEV *Step = AR->getStepRecurrence(*SE);
  // Calculate the pointer stride and check if it is consecutive.
  // The stride may be a constant or a loop invariant integer value.
  const SCEVConstant *ConstStep = dyn_cast<SCEVConstant>(Step);
  if (!ConstStep && !SE->isLoopInvariant(Step, AR->getLoop()))
    return false;

  if (PhiTy->isIntegerTy()) {
    D = InductionDescriptor(StartValue, IK_IntInduction, Step);
    return true;
  }

  assert(PhiTy->isPointerTy() && "The PHI must be a pointer");
  // Pointer induction should be a constant.
  if (!ConstStep)
    return false;

  ConstantInt *CV = ConstStep->getValue();
  Type *PointerElementType = PhiTy->getPointerElementType();
  // The pointer stride cannot be determined if the pointer element type is not
  // sized.
  if (!PointerElementType->isSized())
    return false;

  const DataLayout &DL = Phi->getModule()->getDataLayout();
  int64_t Size = static_cast<int64_t>(DL.getTypeAllocSize(PointerElementType));
  if (!Size)
    return false;

  int64_t CVSize = CV->getSExtValue();
  if (CVSize % Size)
    return false;
  auto *StepValue = SE->getConstant(CV->getType(), CVSize / Size,
                                    true /* signed */);
  D = InductionDescriptor(StartValue, IK_PtrInduction, StepValue);
  return true;
}

/// \brief Returns the instructions that use values defined in the loop.
SmallVector<Instruction *, 8> llvm::findDefsUsedOutsideOfLoop(Loop *L) {
  SmallVector<Instruction *, 8> UsedOutside;

  for (auto *Block : L->getBlocks())
    // FIXME: I believe that this could use copy_if if the Inst reference could
    // be adapted into a pointer.
    for (auto &Inst : *Block) {
      auto Users = Inst.users();
      if (std::any_of(Users.begin(), Users.end(), [&](User *U) {
            auto *Use = cast<Instruction>(U);
            return !L->contains(Use->getParent());
          }))
        UsedOutside.push_back(&Inst);
    }

  return UsedOutside;
}

void llvm::getLoopAnalysisUsage(AnalysisUsage &AU) {
  // By definition, all loop passes need the LoopInfo analysis and the
  // Dominator tree it depends on. Because they all participate in the loop
  // pass manager, they must also preserve these.
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();

  // We must also preserve LoopSimplify and LCSSA. We locally access their IDs
  // here because users shouldn't directly get them from this header.
  extern char &LoopSimplifyID;
  extern char &LCSSAID;
  AU.addRequiredID(LoopSimplifyID);
  AU.addPreservedID(LoopSimplifyID);
  AU.addRequiredID(LCSSAID);
  AU.addPreservedID(LCSSAID);

  // Loop passes are designed to run inside of a loop pass manager which means
  // that any function analyses they require must be required by the first loop
  // pass in the manager (so that it is computed before the loop pass manager
  // runs) and preserved by all loop pasess in the manager. To make this
  // reasonably robust, the set needed for most loop passes is maintained here.
  // If your loop pass requires an analysis not listed here, you will need to
  // carefully audit the loop pass manager nesting structure that results.
  AU.addRequired<AAResultsWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<BasicAAWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
  AU.addPreserved<SCEVAAWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addPreserved<ScalarEvolutionWrapperPass>();
}

/// Manually defined generic "LoopPass" dependency initialization. This is used
/// to initialize the exact set of passes from above in \c
/// getLoopAnalysisUsage. It can be used within a loop pass's initialization
/// with:
///
///   INITIALIZE_PASS_DEPENDENCY(LoopPass)
///
/// As-if "LoopPass" were a pass.
void llvm::initializeLoopPassPass(PassRegistry &Registry) {
  INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
  INITIALIZE_PASS_DEPENDENCY(LCSSAWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(BasicAAWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(SCEVAAWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
}

/// \brief Find string metadata for loop
///
/// If it has a value (e.g. {"llvm.distribute", 1} return the value as an
/// operand or null otherwise.  If the string metadata is not found return
/// Optional's not-a-value.
Optional<const MDOperand *> llvm::findStringMetadataForLoop(Loop *TheLoop,
                                                            StringRef Name) {
  MDNode *LoopID = TheLoop->getLoopID();
  // Return none if LoopID is false.
  if (!LoopID)
    return None;

  // First operand should refer to the loop id itself.
  assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
  assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

  // Iterate over LoopID operands and look for MDString Metadata
  for (unsigned i = 1, e = LoopID->getNumOperands(); i < e; ++i) {
    MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
    if (!MD)
      continue;
    MDString *S = dyn_cast<MDString>(MD->getOperand(0));
    if (!S)
      continue;
    // Return true if MDString holds expected MetaData.
    if (Name.equals(S->getString()))
      switch (MD->getNumOperands()) {
      case 1:
        return nullptr;
      case 2:
        return &MD->getOperand(1);
      default:
        llvm_unreachable("loop metadata has 0 or 1 operand");
      }
  }
  return None;
}

/// Returns true if the instruction in a loop is guaranteed to execute at least
/// once.
bool llvm::isGuaranteedToExecute(const Instruction &Inst,
                                 const DominatorTree *DT, const Loop *CurLoop,
                                 const LoopSafetyInfo *SafetyInfo) {
  // We have to check to make sure that the instruction dominates all
  // of the exit blocks.  If it doesn't, then there is a path out of the loop
  // which does not execute this instruction, so we can't hoist it.

  // If the instruction is in the header block for the loop (which is very
  // common), it is always guaranteed to dominate the exit blocks.  Since this
  // is a common case, and can save some work, check it now.
  if (Inst.getParent() == CurLoop->getHeader())
    // If there's a throw in the header block, we can't guarantee we'll reach
    // Inst.
    return !SafetyInfo->HeaderMayThrow;

  // Somewhere in this loop there is an instruction which may throw and make us
  // exit the loop.
  if (SafetyInfo->MayThrow)
    return false;

  // Get the exit blocks for the current loop.
  SmallVector<BasicBlock *, 8> ExitBlocks;
  CurLoop->getExitBlocks(ExitBlocks);

  // Verify that the block dominates each of the exit blocks of the loop.
  for (BasicBlock *ExitBlock : ExitBlocks)
    if (!DT->dominates(Inst.getParent(), ExitBlock))
      return false;

  // As a degenerate case, if the loop is statically infinite then we haven't
  // proven anything since there are no exit blocks.
  if (ExitBlocks.empty())
    return false;

  // FIXME: In general, we have to prove that the loop isn't an infinite loop.
  // See http::llvm.org/PR24078 .  (The "ExitBlocks.empty()" check above is
  // just a special case of this.)
  return true;
}
