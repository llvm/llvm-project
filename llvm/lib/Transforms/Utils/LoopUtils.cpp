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

#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

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
  if (match(J, m_c_And(m_Instruction(I), m_APInt(M)))) {
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
  //  - By instructions outside of the loop (safe).
  //      * One value may have several outside users, but all outside
  //        uses must be of the same value.
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
        // If we already know this instruction is used externally, move on to
        // the next user.
        if (ExitInstruction == Cur)
          continue;

        // Exit if you find multiple values used outside or if the header phi
        // node is being used. In this case the user uses the value of the
        // previous iteration, in which case we would loose "VF-1" iterations of
        // the reduction operation if we vectorize.
        if (ExitInstruction != nullptr || Cur == Phi)
          return false;

        // The instruction used by an outside user must be the last instruction
        // before we feed back to the reduction phi. Otherwise, we loose VF-1
        // operations on the value.
        if (!is_contained(Phi->operands(), Cur))
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
  if (!UAI && FP && !I->isFast())
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

bool RecurrenceDescriptor::isFirstOrderRecurrence(
    PHINode *Phi, Loop *TheLoop,
    DenseMap<Instruction *, Instruction *> &SinkAfter, DominatorTree *DT) {

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
  if (!Previous || !TheLoop->contains(Previous) || isa<PHINode>(Previous) ||
      SinkAfter.count(Previous)) // Cannot rely on dominance due to motion.
    return false;

  // Ensure every user of the phi node is dominated by the previous value.
  // The dominance requirement ensures the loop vectorizer will not need to
  // vectorize the initial value prior to the first iteration of the loop.
  // TODO: Consider extending this sinking to handle other kinds of instructions
  // and expressions, beyond sinking a single cast past Previous.
  if (Phi->hasOneUse()) {
    auto *I = Phi->user_back();
    if (I->isCast() && (I->getParent() == Phi->getParent()) && I->hasOneUse() &&
        DT->dominates(Previous, I->user_back())) {
      if (!DT->dominates(Previous, I)) // Otherwise we're good w/o sinking.
        SinkAfter[I] = Previous;
      return true;
    }
  }

  for (User *U : Phi->users())
    if (auto *I = dyn_cast<Instruction>(U)) {
      if (!DT->dominates(Previous, I))
        return false;
    }

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

  // We only match FP sequences that are 'fast', so we can unconditionally
  // set it on any generated instructions.
  IRBuilder<>::FastMathFlagGuard FMFG(Builder);
  FastMathFlags FMF;
  FMF.setFast();
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
                                         const SCEV *Step, BinaryOperator *BOp,
                                         SmallVectorImpl<Instruction *> *Casts)
  : StartValue(Start), IK(K), Step(Step), InductionBinOp(BOp) {
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
  assert((IK == IK_FpInduction || Step->getType()->isIntegerTy()) &&
         "StepValue is not an integer");

  assert((IK != IK_FpInduction || Step->getType()->isFloatingPointTy()) &&
         "StepValue is not FP for FpInduction");
  assert((IK != IK_FpInduction || (InductionBinOp &&
          (InductionBinOp->getOpcode() == Instruction::FAdd ||
           InductionBinOp->getOpcode() == Instruction::FSub))) &&
         "Binary opcode should be specified for FP induction");

  if (Casts) {
    for (auto &Inst : *Casts) {
      RedundantCasts.push_back(Inst);
    }
  }
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
  assert(Index->getType() == Step->getType() &&
         "Index type does not match StepValue type");
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
    assert(isa<SCEVConstant>(Step) &&
           "Expected constant step for pointer induction");
    const SCEV *S = SE->getMulExpr(SE->getSCEV(Index), Step);
    Index = Exp.expandCodeFor(S, Index->getType(), &*B.GetInsertPoint());
    return B.CreateGEP(nullptr, StartValue, Index);
  }
  case IK_FpInduction: {
    assert(Step->getType()->isFloatingPointTy() && "Expected FP Step value");
    assert(InductionBinOp &&
           (InductionBinOp->getOpcode() == Instruction::FAdd ||
            InductionBinOp->getOpcode() == Instruction::FSub) &&
           "Original bin op should be defined for FP induction");

    Value *StepValue = cast<SCEVUnknown>(Step)->getValue();

    // Floating point operations had to be 'fast' to enable the induction.
    FastMathFlags Flags;
    Flags.setFast();

    Value *MulExp = B.CreateFMul(StepValue, Index);
    if (isa<Instruction>(MulExp))
      // We have to check, the MulExp may be a constant.
      cast<Instruction>(MulExp)->setFastMathFlags(Flags);

    Value *BOp = B.CreateBinOp(InductionBinOp->getOpcode() , StartValue,
                               MulExp, "induction");
    if (isa<Instruction>(BOp))
      cast<Instruction>(BOp)->setFastMathFlags(Flags);

    return BOp;
  }
  case IK_NoInduction:
    return nullptr;
  }
  llvm_unreachable("invalid enum");
}

bool InductionDescriptor::isFPInductionPHI(PHINode *Phi, const Loop *TheLoop,
                                           ScalarEvolution *SE,
                                           InductionDescriptor &D) {

  // Here we only handle FP induction variables.
  assert(Phi->getType()->isFloatingPointTy() && "Unexpected Phi type");

  if (TheLoop->getHeader() != Phi->getParent())
    return false;

  // The loop may have multiple entrances or multiple exits; we can analyze
  // this phi if it has a unique entry value and a unique backedge value.
  if (Phi->getNumIncomingValues() != 2)
    return false;
  Value *BEValue = nullptr, *StartValue = nullptr;
  if (TheLoop->contains(Phi->getIncomingBlock(0))) {
    BEValue = Phi->getIncomingValue(0);
    StartValue = Phi->getIncomingValue(1);
  } else {
    assert(TheLoop->contains(Phi->getIncomingBlock(1)) &&
           "Unexpected Phi node in the loop");
    BEValue = Phi->getIncomingValue(1);
    StartValue = Phi->getIncomingValue(0);
  }

  BinaryOperator *BOp = dyn_cast<BinaryOperator>(BEValue);
  if (!BOp)
    return false;

  Value *Addend = nullptr;
  if (BOp->getOpcode() == Instruction::FAdd) {
    if (BOp->getOperand(0) == Phi)
      Addend = BOp->getOperand(1);
    else if (BOp->getOperand(1) == Phi)
      Addend = BOp->getOperand(0);
  } else if (BOp->getOpcode() == Instruction::FSub)
    if (BOp->getOperand(0) == Phi)
      Addend = BOp->getOperand(1);

  if (!Addend)
    return false;

  // The addend should be loop invariant
  if (auto *I = dyn_cast<Instruction>(Addend))
    if (TheLoop->contains(I))
      return false;

  // FP Step has unknown SCEV
  const SCEV *Step = SE->getUnknown(Addend);
  D = InductionDescriptor(StartValue, IK_FpInduction, Step, BOp);
  return true;
}

/// This function is called when we suspect that the update-chain of a phi node
/// (whose symbolic SCEV expression sin \p PhiScev) contains redundant casts, 
/// that can be ignored. (This can happen when the PSCEV rewriter adds a runtime 
/// predicate P under which the SCEV expression for the phi can be the 
/// AddRecurrence \p AR; See createAddRecFromPHIWithCast). We want to find the 
/// cast instructions that are involved in the update-chain of this induction. 
/// A caller that adds the required runtime predicate can be free to drop these 
/// cast instructions, and compute the phi using \p AR (instead of some scev 
/// expression with casts).
///
/// For example, without a predicate the scev expression can take the following
/// form:
///      (Ext ix (Trunc iy ( Start + i*Step ) to ix) to iy)
///
/// It corresponds to the following IR sequence:
/// %for.body:
///   %x = phi i64 [ 0, %ph ], [ %add, %for.body ]
///   %casted_phi = "ExtTrunc i64 %x"
///   %add = add i64 %casted_phi, %step
///
/// where %x is given in \p PN,
/// PSE.getSCEV(%x) is equal to PSE.getSCEV(%casted_phi) under a predicate,
/// and the IR sequence that "ExtTrunc i64 %x" represents can take one of
/// several forms, for example, such as:
///   ExtTrunc1:    %casted_phi = and  %x, 2^n-1
/// or:
///   ExtTrunc2:    %t = shl %x, m
///                 %casted_phi = ashr %t, m
///
/// If we are able to find such sequence, we return the instructions
/// we found, namely %casted_phi and the instructions on its use-def chain up
/// to the phi (not including the phi).
static bool getCastsForInductionPHI(PredicatedScalarEvolution &PSE,
                                    const SCEVUnknown *PhiScev,
                                    const SCEVAddRecExpr *AR,
                                    SmallVectorImpl<Instruction *> &CastInsts) {

  assert(CastInsts.empty() && "CastInsts is expected to be empty.");
  auto *PN = cast<PHINode>(PhiScev->getValue());
  assert(PSE.getSCEV(PN) == AR && "Unexpected phi node SCEV expression");
  const Loop *L = AR->getLoop();

  // Find any cast instructions that participate in the def-use chain of 
  // PhiScev in the loop.
  // FORNOW/TODO: We currently expect the def-use chain to include only
  // two-operand instructions, where one of the operands is an invariant.
  // createAddRecFromPHIWithCasts() currently does not support anything more
  // involved than that, so we keep the search simple. This can be
  // extended/generalized as needed.

  auto getDef = [&](const Value *Val) -> Value * {
    const BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Val);
    if (!BinOp)
      return nullptr;
    Value *Op0 = BinOp->getOperand(0);
    Value *Op1 = BinOp->getOperand(1);
    Value *Def = nullptr;
    if (L->isLoopInvariant(Op0))
      Def = Op1;
    else if (L->isLoopInvariant(Op1))
      Def = Op0;
    return Def;
  };

  // Look for the instruction that defines the induction via the
  // loop backedge.
  BasicBlock *Latch = L->getLoopLatch();
  if (!Latch)
    return false;
  Value *Val = PN->getIncomingValueForBlock(Latch);
  if (!Val)
    return false;

  // Follow the def-use chain until the induction phi is reached.
  // If on the way we encounter a Value that has the same SCEV Expr as the
  // phi node, we can consider the instructions we visit from that point
  // as part of the cast-sequence that can be ignored.
  bool InCastSequence = false;
  auto *Inst = dyn_cast<Instruction>(Val);
  while (Val != PN) {
    // If we encountered a phi node other than PN, or if we left the loop,
    // we bail out.
    if (!Inst || !L->contains(Inst)) {
      return false;
    }
    auto *AddRec = dyn_cast<SCEVAddRecExpr>(PSE.getSCEV(Val));
    if (AddRec && PSE.areAddRecsEqualWithPreds(AddRec, AR))
      InCastSequence = true;
    if (InCastSequence) {
      // Only the last instruction in the cast sequence is expected to have
      // uses outside the induction def-use chain.
      if (!CastInsts.empty())
        if (!Inst->hasOneUse())
          return false;
      CastInsts.push_back(Inst);
    }
    Val = getDef(Val);
    if (!Val)
      return false;
    Inst = dyn_cast<Instruction>(Val);
  }

  return InCastSequence;
}

bool InductionDescriptor::isInductionPHI(PHINode *Phi, const Loop *TheLoop,
                                         PredicatedScalarEvolution &PSE,
                                         InductionDescriptor &D,
                                         bool Assume) {
  Type *PhiTy = Phi->getType();

  // Handle integer and pointer inductions variables.
  // Now we handle also FP induction but not trying to make a
  // recurrent expression from the PHI node in-place.

  if (!PhiTy->isIntegerTy() && !PhiTy->isPointerTy() &&
      !PhiTy->isFloatTy() && !PhiTy->isDoubleTy() && !PhiTy->isHalfTy())
    return false;

  if (PhiTy->isFloatingPointTy())
    return isFPInductionPHI(Phi, TheLoop, PSE.getSE(), D);

  const SCEV *PhiScev = PSE.getSCEV(Phi);
  const auto *AR = dyn_cast<SCEVAddRecExpr>(PhiScev);

  // We need this expression to be an AddRecExpr.
  if (Assume && !AR)
    AR = PSE.getAsAddRec(Phi);

  if (!AR) {
    DEBUG(dbgs() << "LV: PHI is not a poly recurrence.\n");
    return false;
  }

  // Record any Cast instructions that participate in the induction update
  const auto *SymbolicPhi = dyn_cast<SCEVUnknown>(PhiScev);
  // If we started from an UnknownSCEV, and managed to build an addRecurrence
  // only after enabling Assume with PSCEV, this means we may have encountered
  // cast instructions that required adding a runtime check in order to
  // guarantee the correctness of the AddRecurence respresentation of the
  // induction.
  if (PhiScev != AR && SymbolicPhi) {
    SmallVector<Instruction *, 2> Casts;
    if (getCastsForInductionPHI(PSE, SymbolicPhi, AR, Casts))
      return isInductionPHI(Phi, TheLoop, PSE.getSE(), D, AR, &Casts);
  }

  return isInductionPHI(Phi, TheLoop, PSE.getSE(), D, AR);
}

bool InductionDescriptor::isInductionPHI(
    PHINode *Phi, const Loop *TheLoop, ScalarEvolution *SE,
    InductionDescriptor &D, const SCEV *Expr,
    SmallVectorImpl<Instruction *> *CastsToIgnore) {
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

  if (AR->getLoop() != TheLoop) {
    // FIXME: We should treat this as a uniform. Unfortunately, we
    // don't currently know how to handled uniform PHIs.
    DEBUG(dbgs() << "LV: PHI is a recurrence with respect to an outer loop.\n");
    return false;
  }

  Value *StartValue =
    Phi->getIncomingValueForBlock(AR->getLoop()->getLoopPreheader());
  const SCEV *Step = AR->getStepRecurrence(*SE);
  // Calculate the pointer stride and check if it is consecutive.
  // The stride may be a constant or a loop invariant integer value.
  const SCEVConstant *ConstStep = dyn_cast<SCEVConstant>(Step);
  if (!ConstStep && !SE->isLoopInvariant(Step, TheLoop))
    return false;

  if (PhiTy->isIntegerTy()) {
    D = InductionDescriptor(StartValue, IK_IntInduction, Step, /*BOp=*/ nullptr,
                            CastsToIgnore);
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

bool llvm::formDedicatedExitBlocks(Loop *L, DominatorTree *DT, LoopInfo *LI,
                                   bool PreserveLCSSA) {
  bool Changed = false;

  // We re-use a vector for the in-loop predecesosrs.
  SmallVector<BasicBlock *, 4> InLoopPredecessors;

  auto RewriteExit = [&](BasicBlock *BB) {
    assert(InLoopPredecessors.empty() &&
           "Must start with an empty predecessors list!");
    auto Cleanup = make_scope_exit([&] { InLoopPredecessors.clear(); });

    // See if there are any non-loop predecessors of this exit block and
    // keep track of the in-loop predecessors.
    bool IsDedicatedExit = true;
    for (auto *PredBB : predecessors(BB))
      if (L->contains(PredBB)) {
        if (isa<IndirectBrInst>(PredBB->getTerminator()))
          // We cannot rewrite exiting edges from an indirectbr.
          return false;

        InLoopPredecessors.push_back(PredBB);
      } else {
        IsDedicatedExit = false;
      }

    assert(!InLoopPredecessors.empty() && "Must have *some* loop predecessor!");

    // Nothing to do if this is already a dedicated exit.
    if (IsDedicatedExit)
      return false;

    auto *NewExitBB = SplitBlockPredecessors(
        BB, InLoopPredecessors, ".loopexit", DT, LI, PreserveLCSSA);

    if (!NewExitBB)
      DEBUG(dbgs() << "WARNING: Can't create a dedicated exit block for loop: "
                   << *L << "\n");
    else
      DEBUG(dbgs() << "LoopSimplify: Creating dedicated exit block "
                   << NewExitBB->getName() << "\n");
    return true;
  };

  // Walk the exit blocks directly rather than building up a data structure for
  // them, but only visit each one once.
  SmallPtrSet<BasicBlock *, 4> Visited;
  for (auto *BB : L->blocks())
    for (auto *SuccBB : successors(BB)) {
      // We're looking for exit blocks so skip in-loop successors.
      if (L->contains(SuccBB))
        continue;

      // Visit each exit block exactly once.
      if (!Visited.insert(SuccBB).second)
        continue;

      Changed |= RewriteExit(SuccBB);
    }

  return Changed;
}

/// \brief Returns the instructions that use values defined in the loop.
SmallVector<Instruction *, 8> llvm::findDefsUsedOutsideOfLoop(Loop *L) {
  SmallVector<Instruction *, 8> UsedOutside;

  for (auto *Block : L->getBlocks())
    // FIXME: I believe that this could use copy_if if the Inst reference could
    // be adapted into a pointer.
    for (auto &Inst : *Block) {
      auto Users = Inst.users();
      if (any_of(Users, [&](User *U) {
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
  // This is used in the LPPassManager to perform LCSSA verification on passes
  // which preserve lcssa form
  AU.addRequired<LCSSAVerificationPass>();
  AU.addPreserved<LCSSAVerificationPass>();

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

/// Does a BFS from a given node to all of its children inside a given loop.
/// The returned vector of nodes includes the starting point.
SmallVector<DomTreeNode *, 16>
llvm::collectChildrenInLoop(DomTreeNode *N, const Loop *CurLoop) {
  SmallVector<DomTreeNode *, 16> Worklist;
  auto AddRegionToWorklist = [&](DomTreeNode *DTN) {
    // Only include subregions in the top level loop.
    BasicBlock *BB = DTN->getBlock();
    if (CurLoop->contains(BB))
      Worklist.push_back(DTN);
  };

  AddRegionToWorklist(N);

  for (size_t I = 0; I < Worklist.size(); I++)
    for (DomTreeNode *Child : Worklist[I]->getChildren())
      AddRegionToWorklist(Child);

  return Worklist;
}

void llvm::deleteDeadLoop(Loop *L, DominatorTree *DT = nullptr,
                          ScalarEvolution *SE = nullptr,
                          LoopInfo *LI = nullptr) {
  assert((!DT || L->isLCSSAForm(*DT)) && "Expected LCSSA!");
  auto *Preheader = L->getLoopPreheader();
  assert(Preheader && "Preheader should exist!");

  // Now that we know the removal is safe, remove the loop by changing the
  // branch from the preheader to go to the single exit block.
  //
  // Because we're deleting a large chunk of code at once, the sequence in which
  // we remove things is very important to avoid invalidation issues.

  // Tell ScalarEvolution that the loop is deleted. Do this before
  // deleting the loop so that ScalarEvolution can look at the loop
  // to determine what it needs to clean up.
  if (SE)
    SE->forgetLoop(L);

  auto *ExitBlock = L->getUniqueExitBlock();
  assert(ExitBlock && "Should have a unique exit block!");
  assert(L->hasDedicatedExits() && "Loop should have dedicated exits!");

  auto *OldBr = dyn_cast<BranchInst>(Preheader->getTerminator());
  assert(OldBr && "Preheader must end with a branch");
  assert(OldBr->isUnconditional() && "Preheader must have a single successor");
  // Connect the preheader to the exit block. Keep the old edge to the header
  // around to perform the dominator tree update in two separate steps
  // -- #1 insertion of the edge preheader -> exit and #2 deletion of the edge
  // preheader -> header.
  //
  //
  // 0.  Preheader          1.  Preheader           2.  Preheader
  //        |                    |   |                   |
  //        V                    |   V                   |
  //      Header <--\            | Header <--\           | Header <--\
  //       |  |     |            |  |  |     |           |  |  |     |
  //       |  V     |            |  |  V     |           |  |  V     |
  //       | Body --/            |  | Body --/           |  | Body --/
  //       V                     V  V                    V  V
  //      Exit                   Exit                    Exit
  //
  // By doing this is two separate steps we can perform the dominator tree
  // update without using the batch update API.
  //
  // Even when the loop is never executed, we cannot remove the edge from the
  // source block to the exit block. Consider the case where the unexecuted loop
  // branches back to an outer loop. If we deleted the loop and removed the edge
  // coming to this inner loop, this will break the outer loop structure (by
  // deleting the backedge of the outer loop). If the outer loop is indeed a
  // non-loop, it will be deleted in a future iteration of loop deletion pass.
  IRBuilder<> Builder(OldBr);
  Builder.CreateCondBr(Builder.getFalse(), L->getHeader(), ExitBlock);
  // Remove the old branch. The conditional branch becomes a new terminator.
  OldBr->eraseFromParent();

  // Rewrite phis in the exit block to get their inputs from the Preheader
  // instead of the exiting block.
  for (PHINode &P : ExitBlock->phis()) {
    // Set the zero'th element of Phi to be from the preheader and remove all
    // other incoming values. Given the loop has dedicated exits, all other
    // incoming values must be from the exiting blocks.
    int PredIndex = 0;
    P.setIncomingBlock(PredIndex, Preheader);
    // Removes all incoming values from all other exiting blocks (including
    // duplicate values from an exiting block).
    // Nuke all entries except the zero'th entry which is the preheader entry.
    // NOTE! We need to remove Incoming Values in the reverse order as done
    // below, to keep the indices valid for deletion (removeIncomingValues
    // updates getNumIncomingValues and shifts all values down into the operand
    // being deleted).
    for (unsigned i = 0, e = P.getNumIncomingValues() - 1; i != e; ++i)
      P.removeIncomingValue(e - i, false);

    assert((P.getNumIncomingValues() == 1 &&
            P.getIncomingBlock(PredIndex) == Preheader) &&
           "Should have exactly one value and that's from the preheader!");
  }

  // Disconnect the loop body by branching directly to its exit.
  Builder.SetInsertPoint(Preheader->getTerminator());
  Builder.CreateBr(ExitBlock);
  // Remove the old branch.
  Preheader->getTerminator()->eraseFromParent();

  if (DT) {
    // Update the dominator tree by informing it about the new edge from the
    // preheader to the exit.
    DT->insertEdge(Preheader, ExitBlock);
    // Inform the dominator tree about the removed edge.
    DT->deleteEdge(Preheader, L->getHeader());
  }

  // Remove the block from the reference counting scheme, so that we can
  // delete it freely later.
  for (auto *Block : L->blocks())
    Block->dropAllReferences();

  if (LI) {
    // Erase the instructions and the blocks without having to worry
    // about ordering because we already dropped the references.
    // NOTE: This iteration is safe because erasing the block does not remove
    // its entry from the loop's block list.  We do that in the next section.
    for (Loop::block_iterator LpI = L->block_begin(), LpE = L->block_end();
         LpI != LpE; ++LpI)
      (*LpI)->eraseFromParent();

    // Finally, the blocks from loopinfo.  This has to happen late because
    // otherwise our loop iterators won't work.

    SmallPtrSet<BasicBlock *, 8> blocks;
    blocks.insert(L->block_begin(), L->block_end());
    for (BasicBlock *BB : blocks)
      LI->removeBlock(BB);

    // The last step is to update LoopInfo now that we've eliminated this loop.
    LI->erase(L);
  }
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

Optional<unsigned> llvm::getLoopEstimatedTripCount(Loop *L) {
  // Only support loops with a unique exiting block, and a latch.
  if (!L->getExitingBlock())
    return None;

  // Get the branch weights for the the loop's backedge.
  BranchInst *LatchBR =
      dyn_cast<BranchInst>(L->getLoopLatch()->getTerminator());
  if (!LatchBR || LatchBR->getNumSuccessors() != 2)
    return None;

  assert((LatchBR->getSuccessor(0) == L->getHeader() ||
          LatchBR->getSuccessor(1) == L->getHeader()) &&
         "At least one edge out of the latch must go to the header");

  // To estimate the number of times the loop body was executed, we want to
  // know the number of times the backedge was taken, vs. the number of times
  // we exited the loop.
  uint64_t TrueVal, FalseVal;
  if (!LatchBR->extractProfMetadata(TrueVal, FalseVal))
    return None;

  if (!TrueVal || !FalseVal)
    return 0;

  // Divide the count of the backedge by the count of the edge exiting the loop,
  // rounding to nearest.
  if (LatchBR->getSuccessor(0) == L->getHeader())
    return (TrueVal + (FalseVal / 2)) / FalseVal;
  else
    return (FalseVal + (TrueVal / 2)) / TrueVal;
}

/// \brief Adds a 'fast' flag to floating point operations.
static Value *addFastMathFlag(Value *V) {
  if (isa<FPMathOperator>(V)) {
    FastMathFlags Flags;
    Flags.setFast();
    cast<Instruction>(V)->setFastMathFlags(Flags);
  }
  return V;
}

// Helper to generate a log2 shuffle reduction.
Value *
llvm::getShuffleReduction(IRBuilder<> &Builder, Value *Src, unsigned Op,
                          RecurrenceDescriptor::MinMaxRecurrenceKind MinMaxKind,
                          ArrayRef<Value *> RedOps) {
  unsigned VF = Src->getType()->getVectorNumElements();
  // VF is a power of 2 so we can emit the reduction using log2(VF) shuffles
  // and vector ops, reducing the set of values being computed by half each
  // round.
  assert(isPowerOf2_32(VF) &&
         "Reduction emission only supported for pow2 vectors!");
  Value *TmpVec = Src;
  SmallVector<Constant *, 32> ShuffleMask(VF, nullptr);
  for (unsigned i = VF; i != 1; i >>= 1) {
    // Move the upper half of the vector to the lower half.
    for (unsigned j = 0; j != i / 2; ++j)
      ShuffleMask[j] = Builder.getInt32(i / 2 + j);

    // Fill the rest of the mask with undef.
    std::fill(&ShuffleMask[i / 2], ShuffleMask.end(),
              UndefValue::get(Builder.getInt32Ty()));

    Value *Shuf = Builder.CreateShuffleVector(
        TmpVec, UndefValue::get(TmpVec->getType()),
        ConstantVector::get(ShuffleMask), "rdx.shuf");

    if (Op != Instruction::ICmp && Op != Instruction::FCmp) {
      // Floating point operations had to be 'fast' to enable the reduction.
      TmpVec = addFastMathFlag(Builder.CreateBinOp((Instruction::BinaryOps)Op,
                                                   TmpVec, Shuf, "bin.rdx"));
    } else {
      assert(MinMaxKind != RecurrenceDescriptor::MRK_Invalid &&
             "Invalid min/max");
      TmpVec = RecurrenceDescriptor::createMinMaxOp(Builder, MinMaxKind, TmpVec,
                                                    Shuf);
    }
    if (!RedOps.empty())
      propagateIRFlags(TmpVec, RedOps);
  }
  // The result is in the first element of the vector.
  return Builder.CreateExtractElement(TmpVec, Builder.getInt32(0));
}

/// Create a simple vector reduction specified by an opcode and some
/// flags (if generating min/max reductions).
Value *llvm::createSimpleTargetReduction(
    IRBuilder<> &Builder, const TargetTransformInfo *TTI, unsigned Opcode,
    Value *Src, TargetTransformInfo::ReductionFlags Flags,
    ArrayRef<Value *> RedOps) {
  assert(isa<VectorType>(Src->getType()) && "Type must be a vector");

  Value *ScalarUdf = UndefValue::get(Src->getType()->getVectorElementType());
  std::function<Value*()> BuildFunc;
  using RD = RecurrenceDescriptor;
  RD::MinMaxRecurrenceKind MinMaxKind = RD::MRK_Invalid;
  // TODO: Support creating ordered reductions.
  FastMathFlags FMFFast;
  FMFFast.setFast();

  switch (Opcode) {
  case Instruction::Add:
    BuildFunc = [&]() { return Builder.CreateAddReduce(Src); };
    break;
  case Instruction::Mul:
    BuildFunc = [&]() { return Builder.CreateMulReduce(Src); };
    break;
  case Instruction::And:
    BuildFunc = [&]() { return Builder.CreateAndReduce(Src); };
    break;
  case Instruction::Or:
    BuildFunc = [&]() { return Builder.CreateOrReduce(Src); };
    break;
  case Instruction::Xor:
    BuildFunc = [&]() { return Builder.CreateXorReduce(Src); };
    break;
  case Instruction::FAdd:
    BuildFunc = [&]() {
      auto Rdx = Builder.CreateFAddReduce(ScalarUdf, Src);
      cast<CallInst>(Rdx)->setFastMathFlags(FMFFast);
      return Rdx;
    };
    break;
  case Instruction::FMul:
    BuildFunc = [&]() {
      auto Rdx = Builder.CreateFMulReduce(ScalarUdf, Src);
      cast<CallInst>(Rdx)->setFastMathFlags(FMFFast);
      return Rdx;
    };
    break;
  case Instruction::ICmp:
    if (Flags.IsMaxOp) {
      MinMaxKind = Flags.IsSigned ? RD::MRK_SIntMax : RD::MRK_UIntMax;
      BuildFunc = [&]() {
        return Builder.CreateIntMaxReduce(Src, Flags.IsSigned);
      };
    } else {
      MinMaxKind = Flags.IsSigned ? RD::MRK_SIntMin : RD::MRK_UIntMin;
      BuildFunc = [&]() {
        return Builder.CreateIntMinReduce(Src, Flags.IsSigned);
      };
    }
    break;
  case Instruction::FCmp:
    if (Flags.IsMaxOp) {
      MinMaxKind = RD::MRK_FloatMax;
      BuildFunc = [&]() { return Builder.CreateFPMaxReduce(Src, Flags.NoNaN); };
    } else {
      MinMaxKind = RD::MRK_FloatMin;
      BuildFunc = [&]() { return Builder.CreateFPMinReduce(Src, Flags.NoNaN); };
    }
    break;
  default:
    llvm_unreachable("Unhandled opcode");
    break;
  }
  if (TTI->useReductionIntrinsic(Opcode, Src->getType(), Flags))
    return BuildFunc();
  return getShuffleReduction(Builder, Src, Opcode, MinMaxKind, RedOps);
}

/// Create a vector reduction using a given recurrence descriptor.
Value *llvm::createTargetReduction(IRBuilder<> &B,
                                   const TargetTransformInfo *TTI,
                                   RecurrenceDescriptor &Desc, Value *Src,
                                   bool NoNaN) {
  // TODO: Support in-order reductions based on the recurrence descriptor.
  using RD = RecurrenceDescriptor;
  RD::RecurrenceKind RecKind = Desc.getRecurrenceKind();
  TargetTransformInfo::ReductionFlags Flags;
  Flags.NoNaN = NoNaN;
  switch (RecKind) {
  case RD::RK_FloatAdd:
    return createSimpleTargetReduction(B, TTI, Instruction::FAdd, Src, Flags);
  case RD::RK_FloatMult:
    return createSimpleTargetReduction(B, TTI, Instruction::FMul, Src, Flags);
  case RD::RK_IntegerAdd:
    return createSimpleTargetReduction(B, TTI, Instruction::Add, Src, Flags);
  case RD::RK_IntegerMult:
    return createSimpleTargetReduction(B, TTI, Instruction::Mul, Src, Flags);
  case RD::RK_IntegerAnd:
    return createSimpleTargetReduction(B, TTI, Instruction::And, Src, Flags);
  case RD::RK_IntegerOr:
    return createSimpleTargetReduction(B, TTI, Instruction::Or, Src, Flags);
  case RD::RK_IntegerXor:
    return createSimpleTargetReduction(B, TTI, Instruction::Xor, Src, Flags);
  case RD::RK_IntegerMinMax: {
    RD::MinMaxRecurrenceKind MMKind = Desc.getMinMaxRecurrenceKind();
    Flags.IsMaxOp = (MMKind == RD::MRK_SIntMax || MMKind == RD::MRK_UIntMax);
    Flags.IsSigned = (MMKind == RD::MRK_SIntMax || MMKind == RD::MRK_SIntMin);
    return createSimpleTargetReduction(B, TTI, Instruction::ICmp, Src, Flags);
  }
  case RD::RK_FloatMinMax: {
    Flags.IsMaxOp = Desc.getMinMaxRecurrenceKind() == RD::MRK_FloatMax;
    return createSimpleTargetReduction(B, TTI, Instruction::FCmp, Src, Flags);
  }
  default:
    llvm_unreachable("Unhandled RecKind");
  }
}

void llvm::propagateIRFlags(Value *I, ArrayRef<Value *> VL, Value *OpValue) {
  auto *VecOp = dyn_cast<Instruction>(I);
  if (!VecOp)
    return;
  auto *Intersection = (OpValue == nullptr) ? dyn_cast<Instruction>(VL[0])
                                            : dyn_cast<Instruction>(OpValue);
  if (!Intersection)
    return;
  const unsigned Opcode = Intersection->getOpcode();
  VecOp->copyIRFlags(Intersection);
  for (auto *V : VL) {
    auto *Instr = dyn_cast<Instruction>(V);
    if (!Instr)
      continue;
    if (OpValue == nullptr || Opcode == Instr->getOpcode())
      VecOp->andIRFlags(V);
  }
}
