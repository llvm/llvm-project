//===- InstructionCombining.cpp - Combine multiple instructions -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// InstructionCombining - Combine instructions to form fewer, simple
// instructions.  This pass does not modify the CFG This pass is where algebraic
// simplification happens.
//
// This pass combines things like:
//    %Y = add int %X, 1
//    %Z = add int %Y, 1
// into:
//    %Z = add int %X, 2
//
// This is a simple worklist driven algorithm.
//
// This pass guarantees that the following canonicalizations are performed on
// the program:
//    1. If a binary operator has a constant operand, it is moved to the RHS
//    2. Bitwise operators with constant operands are always grouped so that
//       shifts are performed first, then or's, then and's, then xor's.
//    3. SetCC instructions are converted from <,>,<=,>= to ==,!= if possible
//    4. All SetCC instructions on boolean values are replaced with logical ops
//    5. add X, X is represented as (X*2) => (X << 1)
//    6. Multiplies with a power-of-two constant argument are transformed into
//       shifts.
//    N. This list is incomplete
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instcombine"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Pass.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/PatternMatch.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include <algorithm>
using namespace llvm;
using namespace llvm::PatternMatch;

namespace {
  Statistic<> NumCombined ("instcombine", "Number of insts combined");
  Statistic<> NumConstProp("instcombine", "Number of constant folds");
  Statistic<> NumDeadInst ("instcombine", "Number of dead inst eliminated");

  class InstCombiner : public FunctionPass,
                       public InstVisitor<InstCombiner, Instruction*> {
    // Worklist of all of the instructions that need to be simplified.
    std::vector<Instruction*> WorkList;
    TargetData *TD;

    /// AddUsersToWorkList - When an instruction is simplified, add all users of
    /// the instruction to the work lists because they might get more simplified
    /// now.
    ///
    void AddUsersToWorkList(Instruction &I) {
      for (Value::use_iterator UI = I.use_begin(), UE = I.use_end();
           UI != UE; ++UI)
        WorkList.push_back(cast<Instruction>(*UI));
    }

    /// AddUsesToWorkList - When an instruction is simplified, add operands to
    /// the work lists because they might get more simplified now.
    ///
    void AddUsesToWorkList(Instruction &I) {
      for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
        if (Instruction *Op = dyn_cast<Instruction>(I.getOperand(i)))
          WorkList.push_back(Op);
    }

    // removeFromWorkList - remove all instances of I from the worklist.
    void removeFromWorkList(Instruction *I);
  public:
    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
      AU.setPreservesCFG();
    }

    TargetData &getTargetData() const { return *TD; }

    // Visitation implementation - Implement instruction combining for different
    // instruction types.  The semantics are as follows:
    // Return Value:
    //    null        - No change was made
    //     I          - Change was made, I is still valid, I may be dead though
    //   otherwise    - Change was made, replace I with returned instruction
    //   
    Instruction *visitAdd(BinaryOperator &I);
    Instruction *visitSub(BinaryOperator &I);
    Instruction *visitMul(BinaryOperator &I);
    Instruction *visitDiv(BinaryOperator &I);
    Instruction *visitRem(BinaryOperator &I);
    Instruction *visitAnd(BinaryOperator &I);
    Instruction *visitOr (BinaryOperator &I);
    Instruction *visitXor(BinaryOperator &I);
    Instruction *visitSetCondInst(BinaryOperator &I);
    Instruction *visitShiftInst(ShiftInst &I);
    Instruction *visitCastInst(CastInst &CI);
    Instruction *visitSelectInst(SelectInst &CI);
    Instruction *visitCallInst(CallInst &CI);
    Instruction *visitInvokeInst(InvokeInst &II);
    Instruction *visitPHINode(PHINode &PN);
    Instruction *visitGetElementPtrInst(GetElementPtrInst &GEP);
    Instruction *visitAllocationInst(AllocationInst &AI);
    Instruction *visitFreeInst(FreeInst &FI);
    Instruction *visitLoadInst(LoadInst &LI);
    Instruction *visitBranchInst(BranchInst &BI);
    Instruction *visitSwitchInst(SwitchInst &SI);

    // visitInstruction - Specify what to return for unhandled instructions...
    Instruction *visitInstruction(Instruction &I) { return 0; }

  private:
    Instruction *visitCallSite(CallSite CS);
    bool transformConstExprCastCall(CallSite CS);

  public:
    // InsertNewInstBefore - insert an instruction New before instruction Old
    // in the program.  Add the new instruction to the worklist.
    //
    Value *InsertNewInstBefore(Instruction *New, Instruction &Old) {
      assert(New && New->getParent() == 0 &&
             "New instruction already inserted into a basic block!");
      BasicBlock *BB = Old.getParent();
      BB->getInstList().insert(&Old, New);  // Insert inst
      WorkList.push_back(New);              // Add to worklist
      return New;
    }

    // ReplaceInstUsesWith - This method is to be used when an instruction is
    // found to be dead, replacable with another preexisting expression.  Here
    // we add all uses of I to the worklist, replace all uses of I with the new
    // value, then return I, so that the inst combiner will know that I was
    // modified.
    //
    Instruction *ReplaceInstUsesWith(Instruction &I, Value *V) {
      AddUsersToWorkList(I);         // Add all modified instrs to worklist
      if (&I != V) {
        I.replaceAllUsesWith(V);
        return &I;
      } else {
        // If we are replacing the instruction with itself, this must be in a
        // segment of unreachable code, so just clobber the instruction.
        I.replaceAllUsesWith(Constant::getNullValue(I.getType()));
        return &I;
      }
    }

    // EraseInstFromFunction - When dealing with an instruction that has side
    // effects or produces a void value, we can't rely on DCE to delete the
    // instruction.  Instead, visit methods should return the value returned by
    // this function.
    Instruction *EraseInstFromFunction(Instruction &I) {
      assert(I.use_empty() && "Cannot erase instruction that is used!");
      AddUsesToWorkList(I);
      removeFromWorkList(&I);
      I.getParent()->getInstList().erase(&I);
      return 0;  // Don't do anything with FI
    }


  private:
    /// InsertOperandCastBefore - This inserts a cast of V to DestTy before the
    /// InsertBefore instruction.  This is specialized a bit to avoid inserting
    /// casts that are known to not do anything...
    ///
    Value *InsertOperandCastBefore(Value *V, const Type *DestTy,
                                   Instruction *InsertBefore);

    // SimplifyCommutative - This performs a few simplifications for commutative
    // operators...
    bool SimplifyCommutative(BinaryOperator &I);

    Instruction *OptAndOp(Instruction *Op, ConstantIntegral *OpRHS,
                          ConstantIntegral *AndRHS, BinaryOperator &TheAnd);
  };

  RegisterOpt<InstCombiner> X("instcombine", "Combine redundant instructions");
}

// getComplexity:  Assign a complexity or rank value to LLVM Values...
//   0 -> Constant, 1 -> Other, 2 -> Argument, 2 -> Unary, 3 -> OtherInst
static unsigned getComplexity(Value *V) {
  if (isa<Instruction>(V)) {
    if (BinaryOperator::isNeg(V) || BinaryOperator::isNot(V))
      return 2;
    return 3;
  }
  if (isa<Argument>(V)) return 2;
  return isa<Constant>(V) ? 0 : 1;
}

// isOnlyUse - Return true if this instruction will be deleted if we stop using
// it.
static bool isOnlyUse(Value *V) {
  return V->hasOneUse() || isa<Constant>(V);
}

// getPromotedType - Return the specified type promoted as it would be to pass
// though a va_arg area...
static const Type *getPromotedType(const Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::SByteTyID:
  case Type::ShortTyID:  return Type::IntTy;
  case Type::UByteTyID:
  case Type::UShortTyID: return Type::UIntTy;
  case Type::FloatTyID:  return Type::DoubleTy;
  default:               return Ty;
  }
}

// SimplifyCommutative - This performs a few simplifications for commutative
// operators:
//
//  1. Order operands such that they are listed from right (least complex) to
//     left (most complex).  This puts constants before unary operators before
//     binary operators.
//
//  2. Transform: (op (op V, C1), C2) ==> (op V, (op C1, C2))
//  3. Transform: (op (op V1, C1), (op V2, C2)) ==> (op (op V1, V2), (op C1,C2))
//
bool InstCombiner::SimplifyCommutative(BinaryOperator &I) {
  bool Changed = false;
  if (getComplexity(I.getOperand(0)) < getComplexity(I.getOperand(1)))
    Changed = !I.swapOperands();
  
  if (!I.isAssociative()) return Changed;
  Instruction::BinaryOps Opcode = I.getOpcode();
  if (BinaryOperator *Op = dyn_cast<BinaryOperator>(I.getOperand(0)))
    if (Op->getOpcode() == Opcode && isa<Constant>(Op->getOperand(1))) {
      if (isa<Constant>(I.getOperand(1))) {
        Constant *Folded = ConstantExpr::get(I.getOpcode(),
                                             cast<Constant>(I.getOperand(1)),
                                             cast<Constant>(Op->getOperand(1)));
        I.setOperand(0, Op->getOperand(0));
        I.setOperand(1, Folded);
        return true;
      } else if (BinaryOperator *Op1=dyn_cast<BinaryOperator>(I.getOperand(1)))
        if (Op1->getOpcode() == Opcode && isa<Constant>(Op1->getOperand(1)) &&
            isOnlyUse(Op) && isOnlyUse(Op1)) {
          Constant *C1 = cast<Constant>(Op->getOperand(1));
          Constant *C2 = cast<Constant>(Op1->getOperand(1));

          // Fold (op (op V1, C1), (op V2, C2)) ==> (op (op V1, V2), (op C1,C2))
          Constant *Folded = ConstantExpr::get(I.getOpcode(), C1, C2);
          Instruction *New = BinaryOperator::create(Opcode, Op->getOperand(0),
                                                    Op1->getOperand(0),
                                                    Op1->getName(), &I);
          WorkList.push_back(New);
          I.setOperand(0, New);
          I.setOperand(1, Folded);
          return true;
        }      
    }
  return Changed;
}

// dyn_castNegVal - Given a 'sub' instruction, return the RHS of the instruction
// if the LHS is a constant zero (which is the 'negate' form).
//
static inline Value *dyn_castNegVal(Value *V) {
  if (BinaryOperator::isNeg(V))
    return BinaryOperator::getNegArgument(cast<BinaryOperator>(V));

  // Constants can be considered to be negated values if they can be folded...
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getNeg(C);
  return 0;
}

static inline Value *dyn_castNotVal(Value *V) {
  if (BinaryOperator::isNot(V))
    return BinaryOperator::getNotArgument(cast<BinaryOperator>(V));

  // Constants can be considered to be not'ed values...
  if (ConstantIntegral *C = dyn_cast<ConstantIntegral>(V))
    return ConstantExpr::getNot(C);
  return 0;
}

// dyn_castFoldableMul - If this value is a multiply that can be folded into
// other computations (because it has a constant operand), return the
// non-constant operand of the multiply.
//
static inline Value *dyn_castFoldableMul(Value *V) {
  if (V->hasOneUse() && V->getType()->isInteger())
    if (Instruction *I = dyn_cast<Instruction>(V))
      if (I->getOpcode() == Instruction::Mul)
        if (isa<Constant>(I->getOperand(1)))
          return I->getOperand(0);
  return 0;
}

// Log2 - Calculate the log base 2 for the specified value if it is exactly a
// power of 2.
static unsigned Log2(uint64_t Val) {
  assert(Val > 1 && "Values 0 and 1 should be handled elsewhere!");
  unsigned Count = 0;
  while (Val != 1) {
    if (Val & 1) return 0;    // Multiple bits set?
    Val >>= 1;
    ++Count;
  }
  return Count;
}


/// AssociativeOpt - Perform an optimization on an associative operator.  This
/// function is designed to check a chain of associative operators for a
/// potential to apply a certain optimization.  Since the optimization may be
/// applicable if the expression was reassociated, this checks the chain, then
/// reassociates the expression as necessary to expose the optimization
/// opportunity.  This makes use of a special Functor, which must define
/// 'shouldApply' and 'apply' methods.
///
template<typename Functor>
Instruction *AssociativeOpt(BinaryOperator &Root, const Functor &F) {
  unsigned Opcode = Root.getOpcode();
  Value *LHS = Root.getOperand(0);

  // Quick check, see if the immediate LHS matches...
  if (F.shouldApply(LHS))
    return F.apply(Root);

  // Otherwise, if the LHS is not of the same opcode as the root, return.
  Instruction *LHSI = dyn_cast<Instruction>(LHS);
  while (LHSI && LHSI->getOpcode() == Opcode && LHSI->hasOneUse()) {
    // Should we apply this transform to the RHS?
    bool ShouldApply = F.shouldApply(LHSI->getOperand(1));

    // If not to the RHS, check to see if we should apply to the LHS...
    if (!ShouldApply && F.shouldApply(LHSI->getOperand(0))) {
      cast<BinaryOperator>(LHSI)->swapOperands();   // Make the LHS the RHS
      ShouldApply = true;
    }

    // If the functor wants to apply the optimization to the RHS of LHSI,
    // reassociate the expression from ((? op A) op B) to (? op (A op B))
    if (ShouldApply) {
      BasicBlock *BB = Root.getParent();
      
      // Now all of the instructions are in the current basic block, go ahead
      // and perform the reassociation.
      Instruction *TmpLHSI = cast<Instruction>(Root.getOperand(0));

      // First move the selected RHS to the LHS of the root...
      Root.setOperand(0, LHSI->getOperand(1));

      // Make what used to be the LHS of the root be the user of the root...
      Value *ExtraOperand = TmpLHSI->getOperand(1);
      if (&Root == TmpLHSI) {
        Root.replaceAllUsesWith(Constant::getNullValue(TmpLHSI->getType()));
        return 0;
      }
      Root.replaceAllUsesWith(TmpLHSI);          // Users now use TmpLHSI
      TmpLHSI->setOperand(1, &Root);             // TmpLHSI now uses the root
      TmpLHSI->getParent()->getInstList().remove(TmpLHSI);
      BasicBlock::iterator ARI = &Root; ++ARI;
      BB->getInstList().insert(ARI, TmpLHSI);    // Move TmpLHSI to after Root
      ARI = Root;

      // Now propagate the ExtraOperand down the chain of instructions until we
      // get to LHSI.
      while (TmpLHSI != LHSI) {
        Instruction *NextLHSI = cast<Instruction>(TmpLHSI->getOperand(0));
        // Move the instruction to immediately before the chain we are
        // constructing to avoid breaking dominance properties.
        NextLHSI->getParent()->getInstList().remove(NextLHSI);
        BB->getInstList().insert(ARI, NextLHSI);
        ARI = NextLHSI;

        Value *NextOp = NextLHSI->getOperand(1);
        NextLHSI->setOperand(1, ExtraOperand);
        TmpLHSI = NextLHSI;
        ExtraOperand = NextOp;
      }
      
      // Now that the instructions are reassociated, have the functor perform
      // the transformation...
      return F.apply(Root);
    }
    
    LHSI = dyn_cast<Instruction>(LHSI->getOperand(0));
  }
  return 0;
}


// AddRHS - Implements: X + X --> X << 1
struct AddRHS {
  Value *RHS;
  AddRHS(Value *rhs) : RHS(rhs) {}
  bool shouldApply(Value *LHS) const { return LHS == RHS; }
  Instruction *apply(BinaryOperator &Add) const {
    return new ShiftInst(Instruction::Shl, Add.getOperand(0),
                         ConstantInt::get(Type::UByteTy, 1));
  }
};

// AddMaskingAnd - Implements (A & C1)+(B & C2) --> (A & C1)|(B & C2)
//                 iff C1&C2 == 0
struct AddMaskingAnd {
  Constant *C2;
  AddMaskingAnd(Constant *c) : C2(c) {}
  bool shouldApply(Value *LHS) const {
    ConstantInt *C1;
    return match(LHS, m_And(m_Value(), m_ConstantInt(C1))) && 
           ConstantExpr::getAnd(C1, C2)->isNullValue();
  }
  Instruction *apply(BinaryOperator &Add) const {
    return BinaryOperator::createOr(Add.getOperand(0), Add.getOperand(1));
  }
};

static Value *FoldOperationIntoSelectOperand(Instruction &BI, Value *SO,
                                             InstCombiner *IC) {
  // Figure out if the constant is the left or the right argument.
  bool ConstIsRHS = isa<Constant>(BI.getOperand(1));
  Constant *ConstOperand = cast<Constant>(BI.getOperand(ConstIsRHS));

  if (Constant *SOC = dyn_cast<Constant>(SO)) {
    if (ConstIsRHS)
      return ConstantExpr::get(BI.getOpcode(), SOC, ConstOperand);
    return ConstantExpr::get(BI.getOpcode(), ConstOperand, SOC);
  }

  Value *Op0 = SO, *Op1 = ConstOperand;
  if (!ConstIsRHS)
    std::swap(Op0, Op1);
  Instruction *New;
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(&BI))
    New = BinaryOperator::create(BO->getOpcode(), Op0, Op1);
  else if (ShiftInst *SI = dyn_cast<ShiftInst>(&BI))
    New = new ShiftInst(SI->getOpcode(), Op0, Op1);
  else {
    assert(0 && "Unknown binary instruction type!");
    abort();
  }
  return IC->InsertNewInstBefore(New, BI);
}

// FoldBinOpIntoSelect - Given an instruction with a select as one operand and a
// constant as the other operand, try to fold the binary operator into the
// select arguments.
static Instruction *FoldBinOpIntoSelect(Instruction &BI, SelectInst *SI,
                                        InstCombiner *IC) {
  // Don't modify shared select instructions
  if (!SI->hasOneUse()) return 0;
  Value *TV = SI->getOperand(1);
  Value *FV = SI->getOperand(2);

  if (isa<Constant>(TV) || isa<Constant>(FV)) {
    Value *SelectTrueVal = FoldOperationIntoSelectOperand(BI, TV, IC);
    Value *SelectFalseVal = FoldOperationIntoSelectOperand(BI, FV, IC);

    return new SelectInst(SI->getCondition(), SelectTrueVal,
                          SelectFalseVal);
  }
  return 0;
}

Instruction *InstCombiner::visitAdd(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
    // X + 0 --> X
    if (!I.getType()->isFloatingPoint() && // -0 + +0 = +0, so it's not a noop
        RHSC->isNullValue())
      return ReplaceInstUsesWith(I, LHS);
    
    // X + (signbit) --> X ^ signbit
    if (ConstantInt *CI = dyn_cast<ConstantInt>(RHSC)) {
      unsigned NumBits = CI->getType()->getPrimitiveSize()*8;
      uint64_t Val = CI->getRawValue() & (1ULL << NumBits)-1;
      if (Val == (1ULL << NumBits-1))
        return BinaryOperator::createXor(LHS, RHS);
    }
  }

  // X + X --> X << 1
  if (I.getType()->isInteger()) {
    if (Instruction *Result = AssociativeOpt(I, AddRHS(RHS))) return Result;
  }

  // -A + B  -->  B - A
  if (Value *V = dyn_castNegVal(LHS))
    return BinaryOperator::createSub(RHS, V);

  // A + -B  -->  A - B
  if (!isa<Constant>(RHS))
    if (Value *V = dyn_castNegVal(RHS))
      return BinaryOperator::createSub(LHS, V);

  // X*C + X --> X * (C+1)
  if (dyn_castFoldableMul(LHS) == RHS) {
    Constant *CP1 =
      ConstantExpr::getAdd(
                        cast<Constant>(cast<Instruction>(LHS)->getOperand(1)),
                        ConstantInt::get(I.getType(), 1));
    return BinaryOperator::createMul(RHS, CP1);
  }

  // X + X*C --> X * (C+1)
  if (dyn_castFoldableMul(RHS) == LHS) {
    Constant *CP1 =
      ConstantExpr::getAdd(
                        cast<Constant>(cast<Instruction>(RHS)->getOperand(1)),
                        ConstantInt::get(I.getType(), 1));
    return BinaryOperator::createMul(LHS, CP1);
  }

  // (A & C1)+(B & C2) --> (A & C1)|(B & C2) iff C1&C2 == 0
  ConstantInt *C2;
  if (match(RHS, m_And(m_Value(), m_ConstantInt(C2))))
    if (Instruction *R = AssociativeOpt(I, AddMaskingAnd(C2))) return R;

  if (ConstantInt *CRHS = dyn_cast<ConstantInt>(RHS)) {
    Value *X;
    if (match(LHS, m_Not(m_Value(X)))) {   // ~X + C --> (C-1) - X
      Constant *C= ConstantExpr::getSub(CRHS, ConstantInt::get(I.getType(), 1));
      return BinaryOperator::createSub(C, X);
    }

    // Try to fold constant add into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(LHS))
      if (Instruction *R = FoldBinOpIntoSelect(I, SI, this))
        return R;
  }

  return Changed ? &I : 0;
}

// isSignBit - Return true if the value represented by the constant only has the
// highest order bit set.
static bool isSignBit(ConstantInt *CI) {
  unsigned NumBits = CI->getType()->getPrimitiveSize()*8;
  return (CI->getRawValue() & ~(-1LL << NumBits)) == (1ULL << (NumBits-1));
}

static unsigned getTypeSizeInBits(const Type *Ty) {
  return Ty == Type::BoolTy ? 1 : Ty->getPrimitiveSize()*8;
}

/// RemoveNoopCast - Strip off nonconverting casts from the value.
///
static Value *RemoveNoopCast(Value *V) {
  if (CastInst *CI = dyn_cast<CastInst>(V)) {
    const Type *CTy = CI->getType();
    const Type *OpTy = CI->getOperand(0)->getType();
    if (CTy->isInteger() && OpTy->isInteger()) {
      if (CTy->getPrimitiveSize() == OpTy->getPrimitiveSize())
        return RemoveNoopCast(CI->getOperand(0));
    } else if (isa<PointerType>(CTy) && isa<PointerType>(OpTy))
      return RemoveNoopCast(CI->getOperand(0));
  }
  return V;
}

Instruction *InstCombiner::visitSub(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Op0 == Op1)         // sub X, X  -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // If this is a 'B = x-(-A)', change to B = x+A...
  if (Value *V = dyn_castNegVal(Op1))
    return BinaryOperator::createAdd(Op0, V);

  if (ConstantInt *C = dyn_cast<ConstantInt>(Op0)) {
    // Replace (-1 - A) with (~A)...
    if (C->isAllOnesValue())
      return BinaryOperator::createNot(Op1);

    // C - ~X == X + (1+C)
    Value *X;
    if (match(Op1, m_Not(m_Value(X))))
      return BinaryOperator::createAdd(X,
                    ConstantExpr::getAdd(C, ConstantInt::get(I.getType(), 1)));
    // -((uint)X >> 31) -> ((int)X >> 31)
    // -((int)X >> 31) -> ((uint)X >> 31)
    if (C->isNullValue()) {
      Value *NoopCastedRHS = RemoveNoopCast(Op1);
      if (ShiftInst *SI = dyn_cast<ShiftInst>(NoopCastedRHS))
        if (SI->getOpcode() == Instruction::Shr)
          if (ConstantUInt *CU = dyn_cast<ConstantUInt>(SI->getOperand(1))) {
            const Type *NewTy;
            if (SI->getType()->isSigned())
              NewTy = SI->getType()->getUnsignedVersion();
            else
              NewTy = SI->getType()->getSignedVersion();
            // Check to see if we are shifting out everything but the sign bit.
            if (CU->getValue() == SI->getType()->getPrimitiveSize()*8-1) {
              // Ok, the transformation is safe.  Insert a cast of the incoming
              // value, then the new shift, then the new cast.
              Instruction *FirstCast = new CastInst(SI->getOperand(0), NewTy,
                                                 SI->getOperand(0)->getName());
              Value *InV = InsertNewInstBefore(FirstCast, I);
              Instruction *NewShift = new ShiftInst(Instruction::Shr, FirstCast,
                                                    CU, SI->getName());
              if (NewShift->getType() == I.getType())
                return NewShift;
              else {
                InV = InsertNewInstBefore(NewShift, I);
                return new CastInst(NewShift, I.getType());
              }
            }
          }
    }

    // Try to fold constant sub into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldBinOpIntoSelect(I, SI, this))
        return R;
  }

  if (BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1))
    if (Op1I->hasOneUse()) {
      // Replace (x - (y - z)) with (x + (z - y)) if the (y - z) subexpression
      // is not used by anyone else...
      //
      if (Op1I->getOpcode() == Instruction::Sub &&
          !Op1I->getType()->isFloatingPoint()) {
        // Swap the two operands of the subexpr...
        Value *IIOp0 = Op1I->getOperand(0), *IIOp1 = Op1I->getOperand(1);
        Op1I->setOperand(0, IIOp1);
        Op1I->setOperand(1, IIOp0);
        
        // Create the new top level add instruction...
        return BinaryOperator::createAdd(Op0, Op1);
      }

      // Replace (A - (A & B)) with (A & ~B) if this is the only use of (A&B)...
      //
      if (Op1I->getOpcode() == Instruction::And &&
          (Op1I->getOperand(0) == Op0 || Op1I->getOperand(1) == Op0)) {
        Value *OtherOp = Op1I->getOperand(Op1I->getOperand(0) == Op0);

        Value *NewNot =
          InsertNewInstBefore(BinaryOperator::createNot(OtherOp, "B.not"), I);
        return BinaryOperator::createAnd(Op0, NewNot);
      }

      // X - X*C --> X * (1-C)
      if (dyn_castFoldableMul(Op1I) == Op0) {
        Constant *CP1 =
          ConstantExpr::getSub(ConstantInt::get(I.getType(), 1),
                         cast<Constant>(cast<Instruction>(Op1)->getOperand(1)));
        assert(CP1 && "Couldn't constant fold 1-C?");
        return BinaryOperator::createMul(Op0, CP1);
      }
    }

  // X*C - X --> X * (C-1)
  if (dyn_castFoldableMul(Op0) == Op1) {
    Constant *CP1 =
     ConstantExpr::getSub(cast<Constant>(cast<Instruction>(Op0)->getOperand(1)),
                        ConstantInt::get(I.getType(), 1));
    assert(CP1 && "Couldn't constant fold C - 1?");
    return BinaryOperator::createMul(Op1, CP1);
  }

  return 0;
}

/// isSignBitCheck - Given an exploded setcc instruction, return true if it is
/// really just returns true if the most significant (sign) bit is set.
static bool isSignBitCheck(unsigned Opcode, Value *LHS, ConstantInt *RHS) {
  if (RHS->getType()->isSigned()) {
    // True if source is LHS < 0 or LHS <= -1
    return Opcode == Instruction::SetLT && RHS->isNullValue() ||
           Opcode == Instruction::SetLE && RHS->isAllOnesValue();
  } else {
    ConstantUInt *RHSC = cast<ConstantUInt>(RHS);
    // True if source is LHS > 127 or LHS >= 128, where the constants depend on
    // the size of the integer type.
    if (Opcode == Instruction::SetGE)
      return RHSC->getValue() == 1ULL<<(RHS->getType()->getPrimitiveSize()*8-1);
    if (Opcode == Instruction::SetGT)
      return RHSC->getValue() ==
        (1ULL << (RHS->getType()->getPrimitiveSize()*8-1))-1;
  }
  return false;
}

Instruction *InstCombiner::visitMul(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0);

  // Simplify mul instructions with a constant RHS...
  if (Constant *Op1 = dyn_cast<Constant>(I.getOperand(1))) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {

      // ((X << C1)*C2) == (X * (C2 << C1))
      if (ShiftInst *SI = dyn_cast<ShiftInst>(Op0))
        if (SI->getOpcode() == Instruction::Shl)
          if (Constant *ShOp = dyn_cast<Constant>(SI->getOperand(1)))
            return BinaryOperator::createMul(SI->getOperand(0),
                                             ConstantExpr::getShl(CI, ShOp));
      
      if (CI->isNullValue())
        return ReplaceInstUsesWith(I, Op1);  // X * 0  == 0
      if (CI->equalsInt(1))                  // X * 1  == X
        return ReplaceInstUsesWith(I, Op0);
      if (CI->isAllOnesValue())              // X * -1 == 0 - X
        return BinaryOperator::createNeg(Op0, I.getName());

      int64_t Val = (int64_t)cast<ConstantInt>(CI)->getRawValue();
      if (uint64_t C = Log2(Val))            // Replace X*(2^C) with X << C
        return new ShiftInst(Instruction::Shl, Op0,
                             ConstantUInt::get(Type::UByteTy, C));
    } else if (ConstantFP *Op1F = dyn_cast<ConstantFP>(Op1)) {
      if (Op1F->isNullValue())
        return ReplaceInstUsesWith(I, Op1);

      // "In IEEE floating point, x*1 is not equivalent to x for nans.  However,
      // ANSI says we can drop signals, so we can do this anyway." (from GCC)
      if (Op1F->getValue() == 1.0)
        return ReplaceInstUsesWith(I, Op0);  // Eliminate 'mul double %X, 1.0'
    }

    // Try to fold constant mul into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldBinOpIntoSelect(I, SI, this))
        return R;
  }

  if (Value *Op0v = dyn_castNegVal(Op0))     // -X * -Y = X*Y
    if (Value *Op1v = dyn_castNegVal(I.getOperand(1)))
      return BinaryOperator::createMul(Op0v, Op1v);

  // If one of the operands of the multiply is a cast from a boolean value, then
  // we know the bool is either zero or one, so this is a 'masking' multiply.
  // See if we can simplify things based on how the boolean was originally
  // formed.
  CastInst *BoolCast = 0;
  if (CastInst *CI = dyn_cast<CastInst>(I.getOperand(0)))
    if (CI->getOperand(0)->getType() == Type::BoolTy)
      BoolCast = CI;
  if (!BoolCast)
    if (CastInst *CI = dyn_cast<CastInst>(I.getOperand(1)))
      if (CI->getOperand(0)->getType() == Type::BoolTy)
        BoolCast = CI;
  if (BoolCast) {
    if (SetCondInst *SCI = dyn_cast<SetCondInst>(BoolCast->getOperand(0))) {
      Value *SCIOp0 = SCI->getOperand(0), *SCIOp1 = SCI->getOperand(1);
      const Type *SCOpTy = SCIOp0->getType();

      // If the setcc is true iff the sign bit of X is set, then convert this
      // multiply into a shift/and combination.
      if (isa<ConstantInt>(SCIOp1) &&
          isSignBitCheck(SCI->getOpcode(), SCIOp0, cast<ConstantInt>(SCIOp1))) {
        // Shift the X value right to turn it into "all signbits".
        Constant *Amt = ConstantUInt::get(Type::UByteTy,
                                          SCOpTy->getPrimitiveSize()*8-1);
        if (SCIOp0->getType()->isUnsigned()) {
          const Type *NewTy = SCIOp0->getType()->getSignedVersion();
          SCIOp0 = InsertNewInstBefore(new CastInst(SCIOp0, NewTy,
                                                    SCIOp0->getName()), I);
        }

        Value *V =
          InsertNewInstBefore(new ShiftInst(Instruction::Shr, SCIOp0, Amt,
                                            BoolCast->getOperand(0)->getName()+
                                            ".mask"), I);

        // If the multiply type is not the same as the source type, sign extend
        // or truncate to the multiply type.
        if (I.getType() != V->getType())
          V = InsertNewInstBefore(new CastInst(V, I.getType(), V->getName()),I);
        
        Value *OtherOp = Op0 == BoolCast ? I.getOperand(1) : Op0;
        return BinaryOperator::createAnd(V, OtherOp);
      }
    }
  }

  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitDiv(BinaryOperator &I) {
  if (ConstantInt *RHS = dyn_cast<ConstantInt>(I.getOperand(1))) {
    // div X, 1 == X
    if (RHS->equalsInt(1))
      return ReplaceInstUsesWith(I, I.getOperand(0));

    // div X, -1 == -X
    if (RHS->isAllOnesValue())
      return BinaryOperator::createNeg(I.getOperand(0));

    // Check to see if this is an unsigned division with an exact power of 2,
    // if so, convert to a right shift.
    if (ConstantUInt *C = dyn_cast<ConstantUInt>(RHS))
      if (uint64_t Val = C->getValue())    // Don't break X / 0
        if (uint64_t C = Log2(Val))
          return new ShiftInst(Instruction::Shr, I.getOperand(0),
                               ConstantUInt::get(Type::UByteTy, C));
  }

  // 0 / X == 0, we don't need to preserve faults!
  if (ConstantInt *LHS = dyn_cast<ConstantInt>(I.getOperand(0)))
    if (LHS->equalsInt(0))
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  return 0;
}


Instruction *InstCombiner::visitRem(BinaryOperator &I) {
  if (I.getType()->isSigned())
    if (Value *RHSNeg = dyn_castNegVal(I.getOperand(1)))
      if (!isa<ConstantSInt>(RHSNeg) ||
          cast<ConstantSInt>(RHSNeg)->getValue() > 0) {
        // X % -Y -> X % Y
        AddUsesToWorkList(I);
        I.setOperand(1, RHSNeg);
        return &I;
      }

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(I.getOperand(1))) {
    if (RHS->equalsInt(1))  // X % 1 == 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

    // Check to see if this is an unsigned remainder with an exact power of 2,
    // if so, convert to a bitwise and.
    if (ConstantUInt *C = dyn_cast<ConstantUInt>(RHS))
      if (uint64_t Val = C->getValue())    // Don't break X % 0 (divide by zero)
        if (!(Val & (Val-1)))              // Power of 2
          return BinaryOperator::createAnd(I.getOperand(0),
                                        ConstantUInt::get(I.getType(), Val-1));
  }

  // 0 % X == 0, we don't need to preserve faults!
  if (ConstantInt *LHS = dyn_cast<ConstantInt>(I.getOperand(0)))
    if (LHS->equalsInt(0))
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  return 0;
}

// isMaxValueMinusOne - return true if this is Max-1
static bool isMaxValueMinusOne(const ConstantInt *C) {
  if (const ConstantUInt *CU = dyn_cast<ConstantUInt>(C)) {
    // Calculate -1 casted to the right type...
    unsigned TypeBits = C->getType()->getPrimitiveSize()*8;
    uint64_t Val = ~0ULL;                // All ones
    Val >>= 64-TypeBits;                 // Shift out unwanted 1 bits...
    return CU->getValue() == Val-1;
  }

  const ConstantSInt *CS = cast<ConstantSInt>(C);
  
  // Calculate 0111111111..11111
  unsigned TypeBits = C->getType()->getPrimitiveSize()*8;
  int64_t Val = INT64_MAX;             // All ones
  Val >>= 64-TypeBits;                 // Shift out unwanted 1 bits...
  return CS->getValue() == Val-1;
}

// isMinValuePlusOne - return true if this is Min+1
static bool isMinValuePlusOne(const ConstantInt *C) {
  if (const ConstantUInt *CU = dyn_cast<ConstantUInt>(C))
    return CU->getValue() == 1;

  const ConstantSInt *CS = cast<ConstantSInt>(C);
  
  // Calculate 1111111111000000000000 
  unsigned TypeBits = C->getType()->getPrimitiveSize()*8;
  int64_t Val = -1;                    // All ones
  Val <<= TypeBits-1;                  // Shift over to the right spot
  return CS->getValue() == Val+1;
}

// isOneBitSet - Return true if there is exactly one bit set in the specified
// constant.
static bool isOneBitSet(const ConstantInt *CI) {
  uint64_t V = CI->getRawValue();
  return V && (V & (V-1)) == 0;
}

/// getSetCondCode - Encode a setcc opcode into a three bit mask.  These bits
/// are carefully arranged to allow folding of expressions such as:
///
///      (A < B) | (A > B) --> (A != B)
///
/// Bit value '4' represents that the comparison is true if A > B, bit value '2'
/// represents that the comparison is true if A == B, and bit value '1' is true
/// if A < B.
///
static unsigned getSetCondCode(const SetCondInst *SCI) {
  switch (SCI->getOpcode()) {
    // False -> 0
  case Instruction::SetGT: return 1;
  case Instruction::SetEQ: return 2;
  case Instruction::SetGE: return 3;
  case Instruction::SetLT: return 4;
  case Instruction::SetNE: return 5;
  case Instruction::SetLE: return 6;
    // True -> 7
  default:
    assert(0 && "Invalid SetCC opcode!");
    return 0;
  }
}

/// getSetCCValue - This is the complement of getSetCondCode, which turns an
/// opcode and two operands into either a constant true or false, or a brand new
/// SetCC instruction.
static Value *getSetCCValue(unsigned Opcode, Value *LHS, Value *RHS) {
  switch (Opcode) {
  case 0: return ConstantBool::False;
  case 1: return new SetCondInst(Instruction::SetGT, LHS, RHS);
  case 2: return new SetCondInst(Instruction::SetEQ, LHS, RHS);
  case 3: return new SetCondInst(Instruction::SetGE, LHS, RHS);
  case 4: return new SetCondInst(Instruction::SetLT, LHS, RHS);
  case 5: return new SetCondInst(Instruction::SetNE, LHS, RHS);
  case 6: return new SetCondInst(Instruction::SetLE, LHS, RHS);
  case 7: return ConstantBool::True;
  default: assert(0 && "Illegal SetCCCode!"); return 0;
  }
}

// FoldSetCCLogical - Implements (setcc1 A, B) & (setcc2 A, B) --> (setcc3 A, B)
struct FoldSetCCLogical {
  InstCombiner &IC;
  Value *LHS, *RHS;
  FoldSetCCLogical(InstCombiner &ic, SetCondInst *SCI)
    : IC(ic), LHS(SCI->getOperand(0)), RHS(SCI->getOperand(1)) {}
  bool shouldApply(Value *V) const {
    if (SetCondInst *SCI = dyn_cast<SetCondInst>(V))
      return (SCI->getOperand(0) == LHS && SCI->getOperand(1) == RHS ||
              SCI->getOperand(0) == RHS && SCI->getOperand(1) == LHS);
    return false;
  }
  Instruction *apply(BinaryOperator &Log) const {
    SetCondInst *SCI = cast<SetCondInst>(Log.getOperand(0));
    if (SCI->getOperand(0) != LHS) {
      assert(SCI->getOperand(1) == LHS);
      SCI->swapOperands();  // Swap the LHS and RHS of the SetCC
    }

    unsigned LHSCode = getSetCondCode(SCI);
    unsigned RHSCode = getSetCondCode(cast<SetCondInst>(Log.getOperand(1)));
    unsigned Code;
    switch (Log.getOpcode()) {
    case Instruction::And: Code = LHSCode & RHSCode; break;
    case Instruction::Or:  Code = LHSCode | RHSCode; break;
    case Instruction::Xor: Code = LHSCode ^ RHSCode; break;
    default: assert(0 && "Illegal logical opcode!"); return 0;
    }

    Value *RV = getSetCCValue(Code, LHS, RHS);
    if (Instruction *I = dyn_cast<Instruction>(RV))
      return I;
    // Otherwise, it's a constant boolean value...
    return IC.ReplaceInstUsesWith(Log, RV);
  }
};


// OptAndOp - This handles expressions of the form ((val OP C1) & C2).  Where
// the Op parameter is 'OP', OpRHS is 'C1', and AndRHS is 'C2'.  Op is
// guaranteed to be either a shift instruction or a binary operator.
Instruction *InstCombiner::OptAndOp(Instruction *Op,
                                    ConstantIntegral *OpRHS,
                                    ConstantIntegral *AndRHS,
                                    BinaryOperator &TheAnd) {
  Value *X = Op->getOperand(0);
  Constant *Together = 0;
  if (!isa<ShiftInst>(Op))
    Together = ConstantExpr::getAnd(AndRHS, OpRHS);

  switch (Op->getOpcode()) {
  case Instruction::Xor:
    if (Together->isNullValue()) {
      // (X ^ C1) & C2 --> (X & C2) iff (C1&C2) == 0
      return BinaryOperator::createAnd(X, AndRHS);
    } else if (Op->hasOneUse()) {
      // (X ^ C1) & C2 --> (X & C2) ^ (C1&C2)
      std::string OpName = Op->getName(); Op->setName("");
      Instruction *And = BinaryOperator::createAnd(X, AndRHS, OpName);
      InsertNewInstBefore(And, TheAnd);
      return BinaryOperator::createXor(And, Together);
    }
    break;
  case Instruction::Or:
    // (X | C1) & C2 --> X & C2 iff C1 & C1 == 0
    if (Together->isNullValue())
      return BinaryOperator::createAnd(X, AndRHS);
    else {
      if (Together == AndRHS) // (X | C) & C --> C
        return ReplaceInstUsesWith(TheAnd, AndRHS);
      
      if (Op->hasOneUse() && Together != OpRHS) {
        // (X | C1) & C2 --> (X | (C1&C2)) & C2
        std::string Op0Name = Op->getName(); Op->setName("");
        Instruction *Or = BinaryOperator::createOr(X, Together, Op0Name);
        InsertNewInstBefore(Or, TheAnd);
        return BinaryOperator::createAnd(Or, AndRHS);
      }
    }
    break;
  case Instruction::Add:
    if (Op->hasOneUse()) {
      // Adding a one to a single bit bit-field should be turned into an XOR
      // of the bit.  First thing to check is to see if this AND is with a
      // single bit constant.
      uint64_t AndRHSV = cast<ConstantInt>(AndRHS)->getRawValue();

      // Clear bits that are not part of the constant.
      AndRHSV &= (1ULL << AndRHS->getType()->getPrimitiveSize()*8)-1;

      // If there is only one bit set...
      if (isOneBitSet(cast<ConstantInt>(AndRHS))) {
        // Ok, at this point, we know that we are masking the result of the
        // ADD down to exactly one bit.  If the constant we are adding has
        // no bits set below this bit, then we can eliminate the ADD.
        uint64_t AddRHS = cast<ConstantInt>(OpRHS)->getRawValue();
            
        // Check to see if any bits below the one bit set in AndRHSV are set.
        if ((AddRHS & (AndRHSV-1)) == 0) {
          // If not, the only thing that can effect the output of the AND is
          // the bit specified by AndRHSV.  If that bit is set, the effect of
          // the XOR is to toggle the bit.  If it is clear, then the ADD has
          // no effect.
          if ((AddRHS & AndRHSV) == 0) { // Bit is not set, noop
            TheAnd.setOperand(0, X);
            return &TheAnd;
          } else {
            std::string Name = Op->getName(); Op->setName("");
            // Pull the XOR out of the AND.
            Instruction *NewAnd = BinaryOperator::createAnd(X, AndRHS, Name);
            InsertNewInstBefore(NewAnd, TheAnd);
            return BinaryOperator::createXor(NewAnd, AndRHS);
          }
        }
      }
    }
    break;

  case Instruction::Shl: {
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!
    //
    Constant *AllOne = ConstantIntegral::getAllOnesValue(AndRHS->getType());
    Constant *CI = ConstantExpr::getAnd(AndRHS,
                                        ConstantExpr::getShl(AllOne, OpRHS));
    if (CI != AndRHS) {
      TheAnd.setOperand(1, CI);
      return &TheAnd;
    }
    break;
  } 
  case Instruction::Shr:
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!  This only applies to
    // unsigned shifts, because a signed shr may bring in set bits!
    //
    if (AndRHS->getType()->isUnsigned()) {
      Constant *AllOne = ConstantIntegral::getAllOnesValue(AndRHS->getType());
      Constant *CI = ConstantExpr::getAnd(AndRHS,
                                          ConstantExpr::getShr(AllOne, OpRHS));
      if (CI != AndRHS) {
        TheAnd.setOperand(1, CI);
        return &TheAnd;
      }
    }
    break;
  }
  return 0;
}


Instruction *InstCombiner::visitAnd(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // and X, X = X   and X, 0 == 0
  if (Op0 == Op1 || Op1 == Constant::getNullValue(I.getType()))
    return ReplaceInstUsesWith(I, Op1);

  // and X, -1 == X
  if (ConstantIntegral *RHS = dyn_cast<ConstantIntegral>(Op1)) {
    if (RHS->isAllOnesValue())
      return ReplaceInstUsesWith(I, Op0);

    // Optimize a variety of ((val OP C1) & C2) combinations...
    if (isa<BinaryOperator>(Op0) || isa<ShiftInst>(Op0)) {
      Instruction *Op0I = cast<Instruction>(Op0);
      Value *X = Op0I->getOperand(0);
      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1)))
        if (Instruction *Res = OptAndOp(Op0I, Op0CI, RHS, I))
          return Res;
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldBinOpIntoSelect(I, SI, this))
        return R;
  }

  Value *Op0NotVal = dyn_castNotVal(Op0);
  Value *Op1NotVal = dyn_castNotVal(Op1);

  if (Op0NotVal == Op1 || Op1NotVal == Op0)  // A & ~A  == ~A & A == 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // (~A & ~B) == (~(A | B)) - De Morgan's Law
  if (Op0NotVal && Op1NotVal && isOnlyUse(Op0) && isOnlyUse(Op1)) {
    Instruction *Or = BinaryOperator::createOr(Op0NotVal, Op1NotVal,
                                               I.getName()+".demorgan");
    InsertNewInstBefore(Or, I);
    return BinaryOperator::createNot(Or);
  }

  // (setcc1 A, B) & (setcc2 A, B) --> (setcc3 A, B)
  if (SetCondInst *RHS = dyn_cast<SetCondInst>(I.getOperand(1)))
    if (Instruction *R = AssociativeOpt(I, FoldSetCCLogical(*this, RHS)))
      return R;

  return Changed ? &I : 0;
}



Instruction *InstCombiner::visitOr(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // or X, X = X   or X, 0 == X
  if (Op0 == Op1 || Op1 == Constant::getNullValue(I.getType()))
    return ReplaceInstUsesWith(I, Op0);

  // or X, -1 == -1
  if (ConstantIntegral *RHS = dyn_cast<ConstantIntegral>(Op1)) {
    if (RHS->isAllOnesValue())
      return ReplaceInstUsesWith(I, Op1);

    ConstantInt *C1; Value *X;
    // (X & C1) | C2 --> (X | C2) & (C1|C2)
    if (match(Op0, m_And(m_Value(X), m_ConstantInt(C1))) && isOnlyUse(Op0)) {
      std::string Op0Name = Op0->getName(); Op0->setName("");
      Instruction *Or = BinaryOperator::createOr(X, RHS, Op0Name);
      InsertNewInstBefore(Or, I);
      return BinaryOperator::createAnd(Or, ConstantExpr::getOr(RHS, C1));
    }

    // (X ^ C1) | C2 --> (X | C2) ^ (C1&~C2)
    if (match(Op0, m_Xor(m_Value(X), m_ConstantInt(C1))) && isOnlyUse(Op0)) {
      std::string Op0Name = Op0->getName(); Op0->setName("");
      Instruction *Or = BinaryOperator::createOr(X, RHS, Op0Name);
      InsertNewInstBefore(Or, I);
      return BinaryOperator::createXor(Or,
                 ConstantExpr::getAnd(C1, ConstantExpr::getNot(RHS)));
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldBinOpIntoSelect(I, SI, this))
        return R;
  }

  // (A & C1)|(A & C2) == A & (C1|C2)
  Value *A, *B; ConstantInt *C1, *C2;
  if (match(Op0, m_And(m_Value(A), m_ConstantInt(C1))) &&
      match(Op1, m_And(m_Value(B), m_ConstantInt(C2))) && A == B)
    return BinaryOperator::createAnd(A, ConstantExpr::getOr(C1, C2));

  if (match(Op0, m_Not(m_Value(A)))) {   // ~A | Op1
    if (A == Op1)   // ~A | A == -1
      return ReplaceInstUsesWith(I, 
                                ConstantIntegral::getAllOnesValue(I.getType()));
  } else {
    A = 0;
  }

  if (match(Op1, m_Not(m_Value(B)))) {   // Op0 | ~B
    if (Op0 == B)
      return ReplaceInstUsesWith(I, 
                                ConstantIntegral::getAllOnesValue(I.getType()));

    // (~A | ~B) == (~(A & B)) - De Morgan's Law
    if (A && isOnlyUse(Op0) && isOnlyUse(Op1)) {
      Value *And = InsertNewInstBefore(BinaryOperator::createAnd(A, B,
                                              I.getName()+".demorgan"), I);
      return BinaryOperator::createNot(And);
    }
  }

  // (setcc1 A, B) | (setcc2 A, B) --> (setcc3 A, B)
  if (SetCondInst *RHS = dyn_cast<SetCondInst>(I.getOperand(1)))
    if (Instruction *R = AssociativeOpt(I, FoldSetCCLogical(*this, RHS)))
      return R;

  return Changed ? &I : 0;
}

// XorSelf - Implements: X ^ X --> 0
struct XorSelf {
  Value *RHS;
  XorSelf(Value *rhs) : RHS(rhs) {}
  bool shouldApply(Value *LHS) const { return LHS == RHS; }
  Instruction *apply(BinaryOperator &Xor) const {
    return &Xor;
  }
};


Instruction *InstCombiner::visitXor(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // xor X, X = 0, even if X is nested in a sequence of Xor's.
  if (Instruction *Result = AssociativeOpt(I, XorSelf(Op1))) {
    assert(Result == &I && "AssociativeOpt didn't work?");
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }

  if (ConstantIntegral *RHS = dyn_cast<ConstantIntegral>(Op1)) {
    // xor X, 0 == X
    if (RHS->isNullValue())
      return ReplaceInstUsesWith(I, Op0);

    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      // xor (setcc A, B), true = not (setcc A, B) = setncc A, B
      if (SetCondInst *SCI = dyn_cast<SetCondInst>(Op0I))
        if (RHS == ConstantBool::True && SCI->hasOneUse())
          return new SetCondInst(SCI->getInverseCondition(),
                                 SCI->getOperand(0), SCI->getOperand(1));

      // ~(c-X) == X-c-1 == X+(-c-1)
      if (Op0I->getOpcode() == Instruction::Sub && RHS->isAllOnesValue())
        if (Constant *Op0I0C = dyn_cast<Constant>(Op0I->getOperand(0))) {
          Constant *NegOp0I0C = ConstantExpr::getNeg(Op0I0C);
          Constant *ConstantRHS = ConstantExpr::getSub(NegOp0I0C,
                                              ConstantInt::get(I.getType(), 1));
          return BinaryOperator::createAdd(Op0I->getOperand(1), ConstantRHS);
        }

      // ~(~X & Y) --> (X | ~Y)
      if (Op0I->getOpcode() == Instruction::And && RHS->isAllOnesValue()) {
        if (dyn_castNotVal(Op0I->getOperand(1))) Op0I->swapOperands();
        if (Value *Op0NotVal = dyn_castNotVal(Op0I->getOperand(0))) {
          Instruction *NotY =
            BinaryOperator::createNot(Op0I->getOperand(1), 
                                      Op0I->getOperand(1)->getName()+".not");
          InsertNewInstBefore(NotY, I);
          return BinaryOperator::createOr(Op0NotVal, NotY);
        }
      }
          
      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1)))
        switch (Op0I->getOpcode()) {
        case Instruction::Add:
          // ~(X-c) --> (-c-1)-X
          if (RHS->isAllOnesValue()) {
            Constant *NegOp0CI = ConstantExpr::getNeg(Op0CI);
            return BinaryOperator::createSub(
                           ConstantExpr::getSub(NegOp0CI,
                                             ConstantInt::get(I.getType(), 1)),
                                          Op0I->getOperand(0));
          }
          break;
        case Instruction::And:
          // (X & C1) ^ C2 --> (X & C1) | C2 iff (C1&C2) == 0
          if (ConstantExpr::getAnd(RHS, Op0CI)->isNullValue())
            return BinaryOperator::createOr(Op0, RHS);
          break;
        case Instruction::Or:
          // (X | C1) ^ C2 --> (X | C1) & ~C2 iff (C1&C2) == C2
          if (ConstantExpr::getAnd(RHS, Op0CI) == RHS)
            return BinaryOperator::createAnd(Op0, ConstantExpr::getNot(RHS));
          break;
        default: break;
        }
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldBinOpIntoSelect(I, SI, this))
        return R;
  }

  if (Value *X = dyn_castNotVal(Op0))   // ~A ^ A == -1
    if (X == Op1)
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));

  if (Value *X = dyn_castNotVal(Op1))   // A ^ ~A == -1
    if (X == Op0)
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));

  if (Instruction *Op1I = dyn_cast<Instruction>(Op1))
    if (Op1I->getOpcode() == Instruction::Or) {
      if (Op1I->getOperand(0) == Op0) {              // B^(B|A) == (A|B)^B
        cast<BinaryOperator>(Op1I)->swapOperands();
        I.swapOperands();
        std::swap(Op0, Op1);
      } else if (Op1I->getOperand(1) == Op0) {       // B^(A|B) == (A|B)^B
        I.swapOperands();
        std::swap(Op0, Op1);
      }      
    } else if (Op1I->getOpcode() == Instruction::Xor) {
      if (Op0 == Op1I->getOperand(0))                        // A^(A^B) == B
        return ReplaceInstUsesWith(I, Op1I->getOperand(1));
      else if (Op0 == Op1I->getOperand(1))                   // A^(B^A) == B
        return ReplaceInstUsesWith(I, Op1I->getOperand(0));
    }

  if (Instruction *Op0I = dyn_cast<Instruction>(Op0))
    if (Op0I->getOpcode() == Instruction::Or && Op0I->hasOneUse()) {
      if (Op0I->getOperand(0) == Op1)                // (B|A)^B == (A|B)^B
        cast<BinaryOperator>(Op0I)->swapOperands();
      if (Op0I->getOperand(1) == Op1) {              // (A|B)^B == A & ~B
        Value *NotB = InsertNewInstBefore(BinaryOperator::createNot(Op1,
                                                     Op1->getName()+".not"), I);
        return BinaryOperator::createAnd(Op0I->getOperand(0), NotB);
      }
    } else if (Op0I->getOpcode() == Instruction::Xor) {
      if (Op1 == Op0I->getOperand(0))                        // (A^B)^A == B
        return ReplaceInstUsesWith(I, Op0I->getOperand(1));
      else if (Op1 == Op0I->getOperand(1))                   // (B^A)^A == B
        return ReplaceInstUsesWith(I, Op0I->getOperand(0));
    }

  // (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
  Value *A, *B; ConstantInt *C1, *C2;
  if (match(Op0, m_And(m_Value(A), m_ConstantInt(C1))) &&
      match(Op1, m_And(m_Value(B), m_ConstantInt(C2))) &&
      ConstantExpr::getAnd(C1, C2)->isNullValue())
    return BinaryOperator::createOr(Op0, Op1);

  // (setcc1 A, B) ^ (setcc2 A, B) --> (setcc3 A, B)
  if (SetCondInst *RHS = dyn_cast<SetCondInst>(I.getOperand(1)))
    if (Instruction *R = AssociativeOpt(I, FoldSetCCLogical(*this, RHS)))
      return R;

  return Changed ? &I : 0;
}

// AddOne, SubOne - Add or subtract a constant one from an integer constant...
static Constant *AddOne(ConstantInt *C) {
  return ConstantExpr::getAdd(C, ConstantInt::get(C->getType(), 1));
}
static Constant *SubOne(ConstantInt *C) {
  return ConstantExpr::getSub(C, ConstantInt::get(C->getType(), 1));
}

// isTrueWhenEqual - Return true if the specified setcondinst instruction is
// true when both operands are equal...
//
static bool isTrueWhenEqual(Instruction &I) {
  return I.getOpcode() == Instruction::SetEQ ||
         I.getOpcode() == Instruction::SetGE ||
         I.getOpcode() == Instruction::SetLE;
}

Instruction *InstCombiner::visitSetCondInst(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  const Type *Ty = Op0->getType();

  // setcc X, X
  if (Op0 == Op1)
    return ReplaceInstUsesWith(I, ConstantBool::get(isTrueWhenEqual(I)));

  // setcc <global/alloca*>, 0 - Global/Stack value addresses are never null!
  if (isa<ConstantPointerNull>(Op1) && 
      (isa<GlobalValue>(Op0) || isa<AllocaInst>(Op0)))
    return ReplaceInstUsesWith(I, ConstantBool::get(!isTrueWhenEqual(I)));


  // setcc's with boolean values can always be turned into bitwise operations
  if (Ty == Type::BoolTy) {
    // If this is <, >, or !=, we can change this into a simple xor instruction
    if (!isTrueWhenEqual(I))
      return BinaryOperator::createXor(Op0, Op1);

    // Otherwise we need to make a temporary intermediate instruction and insert
    // it into the instruction stream.  This is what we are after:
    //
    //  seteq bool %A, %B -> ~(A^B)
    //  setle bool %A, %B -> ~A | B
    //  setge bool %A, %B -> A | ~B
    //
    if (I.getOpcode() == Instruction::SetEQ) {  // seteq case
      Instruction *Xor = BinaryOperator::createXor(Op0, Op1, I.getName()+"tmp");
      InsertNewInstBefore(Xor, I);
      return BinaryOperator::createNot(Xor);
    }

    // Handle the setXe cases...
    assert(I.getOpcode() == Instruction::SetGE ||
           I.getOpcode() == Instruction::SetLE);

    if (I.getOpcode() == Instruction::SetGE)
      std::swap(Op0, Op1);                   // Change setge -> setle

    // Now we just have the SetLE case.
    Instruction *Not = BinaryOperator::createNot(Op0, I.getName()+"tmp");
    InsertNewInstBefore(Not, I);
    return BinaryOperator::createOr(Not, Op1);
  }

  // See if we are doing a comparison between a constant and an instruction that
  // can be folded into the comparison.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    if (Instruction *LHSI = dyn_cast<Instruction>(Op0))
      if (LHSI->hasOneUse())
        switch (LHSI->getOpcode()) {
        case Instruction::And:
          if (isa<ConstantInt>(LHSI->getOperand(1)) &&
              LHSI->getOperand(0)->hasOneUse()) {
            // If this is: (X >> C1) & C2 != C3 (where any shift and any compare
            // could exist), turn it into (X & (C2 << C1)) != (C3 << C1).  This
            // happens a LOT in code produced by the C front-end, for bitfield
            // access.
            ShiftInst *Shift = dyn_cast<ShiftInst>(LHSI->getOperand(0));
            ConstantUInt *ShAmt;
            ShAmt = Shift ? dyn_cast<ConstantUInt>(Shift->getOperand(1)) : 0;
            ConstantInt *AndCST = cast<ConstantInt>(LHSI->getOperand(1));
            const Type *Ty = LHSI->getType();
                  
            // We can fold this as long as we can't shift unknown bits
            // into the mask.  This can only happen with signed shift
            // rights, as they sign-extend.
            if (ShAmt) {
              bool CanFold = Shift->getOpcode() != Instruction::Shr ||
                             Shift->getType()->isUnsigned();
              if (!CanFold) {
                // To test for the bad case of the signed shr, see if any
                // of the bits shifted in could be tested after the mask.
                Constant *OShAmt = ConstantUInt::get(Type::UByteTy, 
                                   Ty->getPrimitiveSize()*8-ShAmt->getValue());
                Constant *ShVal = 
                 ConstantExpr::getShl(ConstantInt::getAllOnesValue(Ty), OShAmt);
                if (ConstantExpr::getAnd(ShVal, AndCST)->isNullValue())
                  CanFold = true;
              }

              if (CanFold) {
                unsigned ShiftOp = Shift->getOpcode() == Instruction::Shl
                  ? Instruction::Shr : Instruction::Shl;
                Constant *NewCst = ConstantExpr::get(ShiftOp, CI, ShAmt);

                // Check to see if we are shifting out any of the bits being
                // compared.
                if (ConstantExpr::get(Shift->getOpcode(), NewCst, ShAmt) != CI){
                  // If we shifted bits out, the fold is not going to work out.
                  // As a special case, check to see if this means that the
                  // result is always true or false now.
                  if (I.getOpcode() == Instruction::SetEQ)
                    return ReplaceInstUsesWith(I, ConstantBool::False);
                  if (I.getOpcode() == Instruction::SetNE)
                    return ReplaceInstUsesWith(I, ConstantBool::True);
                } else {
                  I.setOperand(1, NewCst);
                  LHSI->setOperand(1, ConstantExpr::get(ShiftOp, AndCST,ShAmt));
                  LHSI->setOperand(0, Shift->getOperand(0));
                  WorkList.push_back(Shift); // Shift is dead.
                  AddUsesToWorkList(I);
                  return &I;
                }
              }
            }
          }
          break;
        case Instruction::Div:
          if (0 && isa<ConstantInt>(LHSI->getOperand(1))) {
            std::cerr << "COULD FOLD: " << *LHSI;
            std::cerr << "COULD FOLD: " << I << "\n";
          }
          break;
        case Instruction::Select:
          // If either operand of the select is a constant, we can fold the
          // comparison into the select arms, which will cause one to be
          // constant folded and the select turned into a bitwise or.
          Value *Op1 = 0, *Op2 = 0;
          if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(1))) {
            // Fold the known value into the constant operand.
            Op1 = ConstantExpr::get(I.getOpcode(), C, CI);
            // Insert a new SetCC of the other select operand.
            Op2 = InsertNewInstBefore(new SetCondInst(I.getOpcode(),
                                                      LHSI->getOperand(2), CI,
                                                      I.getName()), I);
          } else if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(2))) {
            // Fold the known value into the constant operand.
            Op2 = ConstantExpr::get(I.getOpcode(), C, CI);
            // Insert a new SetCC of the other select operand.
            Op1 = InsertNewInstBefore(new SetCondInst(I.getOpcode(),
                                                      LHSI->getOperand(1), CI,
                                                      I.getName()), I);
          }

          if (Op1)
            return new SelectInst(LHSI->getOperand(0), Op1, Op2);
          break;
        }

    // Simplify seteq and setne instructions...
    if (I.getOpcode() == Instruction::SetEQ ||
        I.getOpcode() == Instruction::SetNE) {
      bool isSetNE = I.getOpcode() == Instruction::SetNE;

      // If the first operand is (and|or|xor) with a constant, and the second
      // operand is a constant, simplify a bit.
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0)) {
        switch (BO->getOpcode()) {
        case Instruction::Rem:
          // If we have a signed (X % (2^c)) == 0, turn it into an unsigned one.
          if (CI->isNullValue() && isa<ConstantSInt>(BO->getOperand(1)) &&
              BO->hasOneUse() &&
              cast<ConstantSInt>(BO->getOperand(1))->getValue() > 1)
            if (unsigned L2 =
                Log2(cast<ConstantSInt>(BO->getOperand(1))->getValue())) {
              const Type *UTy = BO->getType()->getUnsignedVersion();
              Value *NewX = InsertNewInstBefore(new CastInst(BO->getOperand(0),
                                                             UTy, "tmp"), I);
              Constant *RHSCst = ConstantUInt::get(UTy, 1ULL << L2);
              Value *NewRem =InsertNewInstBefore(BinaryOperator::createRem(NewX,
                                                    RHSCst, BO->getName()), I);
              return BinaryOperator::create(I.getOpcode(), NewRem,
                                            Constant::getNullValue(UTy));
            }
          break;          

        case Instruction::Add:
          // Replace ((add A, B) != C) with (A != C-B) if B & C are constants.
          if (ConstantInt *BOp1C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
            return new SetCondInst(I.getOpcode(), BO->getOperand(0),
                                   ConstantExpr::getSub(CI, BOp1C));
          } else if (CI->isNullValue()) {
            // Replace ((add A, B) != 0) with (A != -B) if A or B is
            // efficiently invertible, or if the add has just this one use.
            Value *BOp0 = BO->getOperand(0), *BOp1 = BO->getOperand(1);
            
            if (Value *NegVal = dyn_castNegVal(BOp1))
              return new SetCondInst(I.getOpcode(), BOp0, NegVal);
            else if (Value *NegVal = dyn_castNegVal(BOp0))
              return new SetCondInst(I.getOpcode(), NegVal, BOp1);
            else if (BO->hasOneUse()) {
              Instruction *Neg = BinaryOperator::createNeg(BOp1, BO->getName());
              BO->setName("");
              InsertNewInstBefore(Neg, I);
              return new SetCondInst(I.getOpcode(), BOp0, Neg);
            }
          }
          break;
        case Instruction::Xor:
          // For the xor case, we can xor two constants together, eliminating
          // the explicit xor.
          if (Constant *BOC = dyn_cast<Constant>(BO->getOperand(1)))
            return BinaryOperator::create(I.getOpcode(), BO->getOperand(0),
                                  ConstantExpr::getXor(CI, BOC));

          // FALLTHROUGH
        case Instruction::Sub:
          // Replace (([sub|xor] A, B) != 0) with (A != B)
          if (CI->isNullValue())
            return new SetCondInst(I.getOpcode(), BO->getOperand(0),
                                   BO->getOperand(1));
          break;

        case Instruction::Or:
          // If bits are being or'd in that are not present in the constant we
          // are comparing against, then the comparison could never succeed!
          if (Constant *BOC = dyn_cast<Constant>(BO->getOperand(1))) {
            Constant *NotCI = ConstantExpr::getNot(CI);
            if (!ConstantExpr::getAnd(BOC, NotCI)->isNullValue())
              return ReplaceInstUsesWith(I, ConstantBool::get(isSetNE));
          }
          break;

        case Instruction::And:
          if (ConstantInt *BOC = dyn_cast<ConstantInt>(BO->getOperand(1))) {
            // If bits are being compared against that are and'd out, then the
            // comparison can never succeed!
            if (!ConstantExpr::getAnd(CI,
                                      ConstantExpr::getNot(BOC))->isNullValue())
              return ReplaceInstUsesWith(I, ConstantBool::get(isSetNE));

            // If we have ((X & C) == C), turn it into ((X & C) != 0).
            if (CI == BOC && isOneBitSet(CI))
              return new SetCondInst(isSetNE ? Instruction::SetEQ :
                                     Instruction::SetNE, Op0,
                                     Constant::getNullValue(CI->getType()));

            // Replace (and X, (1 << size(X)-1) != 0) with x < 0, converting X
            // to be a signed value as appropriate.
            if (isSignBit(BOC)) {
              Value *X = BO->getOperand(0);
              // If 'X' is not signed, insert a cast now...
              if (!BOC->getType()->isSigned()) {
                const Type *DestTy = BOC->getType()->getSignedVersion();
                CastInst *NewCI = new CastInst(X,DestTy,X->getName()+".signed");
                InsertNewInstBefore(NewCI, I);
                X = NewCI;
              }
              return new SetCondInst(isSetNE ? Instruction::SetLT :
                                         Instruction::SetGE, X,
                                     Constant::getNullValue(X->getType()));
            }
          }
        default: break;
        }
      }
    } else {  // Not a SetEQ/SetNE
      // If the LHS is a cast from an integral value of the same size, 
      if (CastInst *Cast = dyn_cast<CastInst>(Op0)) {
        Value *CastOp = Cast->getOperand(0);
        const Type *SrcTy = CastOp->getType();
        unsigned SrcTySize = SrcTy->getPrimitiveSize();
        if (SrcTy != Cast->getType() && SrcTy->isInteger() &&
            SrcTySize == Cast->getType()->getPrimitiveSize()) {
          assert((SrcTy->isSigned() ^ Cast->getType()->isSigned()) && 
                 "Source and destination signednesses should differ!");
          if (Cast->getType()->isSigned()) {
            // If this is a signed comparison, check for comparisons in the
            // vicinity of zero.
            if (I.getOpcode() == Instruction::SetLT && CI->isNullValue())
              // X < 0  => x > 127
              return BinaryOperator::createSetGT(CastOp,
                         ConstantUInt::get(SrcTy, (1ULL << (SrcTySize*8-1))-1));
            else if (I.getOpcode() == Instruction::SetGT &&
                     cast<ConstantSInt>(CI)->getValue() == -1)
              // X > -1  => x < 128
              return BinaryOperator::createSetLT(CastOp,
                         ConstantUInt::get(SrcTy, 1ULL << (SrcTySize*8-1)));
          } else {
            ConstantUInt *CUI = cast<ConstantUInt>(CI);
            if (I.getOpcode() == Instruction::SetLT &&
                CUI->getValue() == 1ULL << (SrcTySize*8-1))
              // X < 128 => X > -1
              return BinaryOperator::createSetGT(CastOp,
                                                 ConstantSInt::get(SrcTy, -1));
            else if (I.getOpcode() == Instruction::SetGT &&
                     CUI->getValue() == (1ULL << (SrcTySize*8-1))-1)
              // X > 127 => X < 0
              return BinaryOperator::createSetLT(CastOp,
                                                 Constant::getNullValue(SrcTy));
          }
        }
      }
    }

    // Check to see if we are comparing against the minimum or maximum value...
    if (CI->isMinValue()) {
      if (I.getOpcode() == Instruction::SetLT)       // A < MIN -> FALSE
        return ReplaceInstUsesWith(I, ConstantBool::False);
      if (I.getOpcode() == Instruction::SetGE)       // A >= MIN -> TRUE
        return ReplaceInstUsesWith(I, ConstantBool::True);
      if (I.getOpcode() == Instruction::SetLE)       // A <= MIN -> A == MIN
        return BinaryOperator::createSetEQ(Op0, Op1);
      if (I.getOpcode() == Instruction::SetGT)       // A > MIN -> A != MIN
        return BinaryOperator::createSetNE(Op0, Op1);

    } else if (CI->isMaxValue()) {
      if (I.getOpcode() == Instruction::SetGT)       // A > MAX -> FALSE
        return ReplaceInstUsesWith(I, ConstantBool::False);
      if (I.getOpcode() == Instruction::SetLE)       // A <= MAX -> TRUE
        return ReplaceInstUsesWith(I, ConstantBool::True);
      if (I.getOpcode() == Instruction::SetGE)       // A >= MAX -> A == MAX
        return BinaryOperator::createSetEQ(Op0, Op1);
      if (I.getOpcode() == Instruction::SetLT)       // A < MAX -> A != MAX
        return BinaryOperator::createSetNE(Op0, Op1);

      // Comparing against a value really close to min or max?
    } else if (isMinValuePlusOne(CI)) {
      if (I.getOpcode() == Instruction::SetLT)       // A < MIN+1 -> A == MIN
        return BinaryOperator::createSetEQ(Op0, SubOne(CI));
      if (I.getOpcode() == Instruction::SetGE)       // A >= MIN-1 -> A != MIN
        return BinaryOperator::createSetNE(Op0, SubOne(CI));

    } else if (isMaxValueMinusOne(CI)) {
      if (I.getOpcode() == Instruction::SetGT)       // A > MAX-1 -> A == MAX
        return BinaryOperator::createSetEQ(Op0, AddOne(CI));
      if (I.getOpcode() == Instruction::SetLE)       // A <= MAX-1 -> A != MAX
        return BinaryOperator::createSetNE(Op0, AddOne(CI));
    }

    // If we still have a setle or setge instruction, turn it into the
    // appropriate setlt or setgt instruction.  Since the border cases have
    // already been handled above, this requires little checking.
    //
    if (I.getOpcode() == Instruction::SetLE)
      return BinaryOperator::createSetLT(Op0, AddOne(CI));
    if (I.getOpcode() == Instruction::SetGE)
      return BinaryOperator::createSetGT(Op0, SubOne(CI));
  }

  // Test to see if the operands of the setcc are casted versions of other
  // values.  If the cast can be stripped off both arguments, we do so now.
  if (CastInst *CI = dyn_cast<CastInst>(Op0)) {
    Value *CastOp0 = CI->getOperand(0);
    if (CastOp0->getType()->isLosslesslyConvertibleTo(CI->getType()) &&
        (isa<Constant>(Op1) || isa<CastInst>(Op1)) &&
        (I.getOpcode() == Instruction::SetEQ ||
         I.getOpcode() == Instruction::SetNE)) {
      // We keep moving the cast from the left operand over to the right
      // operand, where it can often be eliminated completely.
      Op0 = CastOp0;
      
      // If operand #1 is a cast instruction, see if we can eliminate it as
      // well.
      if (CastInst *CI2 = dyn_cast<CastInst>(Op1))
        if (CI2->getOperand(0)->getType()->isLosslesslyConvertibleTo(
                                                               Op0->getType()))
          Op1 = CI2->getOperand(0);
      
      // If Op1 is a constant, we can fold the cast into the constant.
      if (Op1->getType() != Op0->getType())
        if (Constant *Op1C = dyn_cast<Constant>(Op1)) {
          Op1 = ConstantExpr::getCast(Op1C, Op0->getType());
        } else {
          // Otherwise, cast the RHS right before the setcc
          Op1 = new CastInst(Op1, Op0->getType(), Op1->getName());
          InsertNewInstBefore(cast<Instruction>(Op1), I);
        }
      return BinaryOperator::create(I.getOpcode(), Op0, Op1);
    }

    // Handle the special case of: setcc (cast bool to X), <cst>
    // This comes up when you have code like
    //   int X = A < B;
    //   if (X) ...
    // For generality, we handle any zero-extension of any operand comparison
    // with a constant.
    if (ConstantInt *ConstantRHS = dyn_cast<ConstantInt>(Op1)) {
      const Type *SrcTy = CastOp0->getType();
      const Type *DestTy = Op0->getType();
      if (SrcTy->getPrimitiveSize() < DestTy->getPrimitiveSize() &&
          (SrcTy->isUnsigned() || SrcTy == Type::BoolTy)) {
        // Ok, we have an expansion of operand 0 into a new type.  Get the
        // constant value, masink off bits which are not set in the RHS.  These
        // could be set if the destination value is signed.
        uint64_t ConstVal = ConstantRHS->getRawValue();
        ConstVal &= (1ULL << DestTy->getPrimitiveSize()*8)-1;

        // If the constant we are comparing it with has high bits set, which
        // don't exist in the original value, the values could never be equal,
        // because the source would be zero extended.
        unsigned SrcBits =
          SrcTy == Type::BoolTy ? 1 : SrcTy->getPrimitiveSize()*8;
        bool HasSignBit = ConstVal & (1ULL << (DestTy->getPrimitiveSize()*8-1));
        if (ConstVal & ~((1ULL << SrcBits)-1)) {
          switch (I.getOpcode()) {
          default: assert(0 && "Unknown comparison type!");
          case Instruction::SetEQ:
            return ReplaceInstUsesWith(I, ConstantBool::False);
          case Instruction::SetNE:
            return ReplaceInstUsesWith(I, ConstantBool::True);
          case Instruction::SetLT:
          case Instruction::SetLE:
            if (DestTy->isSigned() && HasSignBit)
              return ReplaceInstUsesWith(I, ConstantBool::False);
            return ReplaceInstUsesWith(I, ConstantBool::True);
          case Instruction::SetGT:
          case Instruction::SetGE:
            if (DestTy->isSigned() && HasSignBit)
              return ReplaceInstUsesWith(I, ConstantBool::True);
            return ReplaceInstUsesWith(I, ConstantBool::False);
          }
        }
        
        // Otherwise, we can replace the setcc with a setcc of the smaller
        // operand value.
        Op1 = ConstantExpr::getCast(cast<Constant>(Op1), SrcTy);
        return BinaryOperator::create(I.getOpcode(), CastOp0, Op1);
      }
    }
  }
  return Changed ? &I : 0;
}



Instruction *InstCombiner::visitShiftInst(ShiftInst &I) {
  assert(I.getOperand(1)->getType() == Type::UByteTy);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  bool isLeftShift = I.getOpcode() == Instruction::Shl;

  // shl X, 0 == X and shr X, 0 == X
  // shl 0, X == 0 and shr 0, X == 0
  if (Op1 == Constant::getNullValue(Type::UByteTy) ||
      Op0 == Constant::getNullValue(Op0->getType()))
    return ReplaceInstUsesWith(I, Op0);

  // shr int -1, X = -1   (for any arithmetic shift rights of ~0)
  if (!isLeftShift)
    if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(Op0))
      if (CSI->isAllOnesValue())
        return ReplaceInstUsesWith(I, CSI);

  // Try to fold constant and into select arguments.
  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldBinOpIntoSelect(I, SI, this))
        return R;

  if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(Op1)) {
    // shl uint X, 32 = 0 and shr ubyte Y, 9 = 0, ... just don't eliminate shr
    // of a signed value.
    //
    unsigned TypeBits = Op0->getType()->getPrimitiveSize()*8;
    if (CUI->getValue() >= TypeBits) {
      if (!Op0->getType()->isSigned() || isLeftShift)
        return ReplaceInstUsesWith(I, Constant::getNullValue(Op0->getType()));
      else {
        I.setOperand(1, ConstantUInt::get(Type::UByteTy, TypeBits-1));
        return &I;
      }
    }

    // ((X*C1) << C2) == (X * (C1 << C2))
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0))
      if (BO->getOpcode() == Instruction::Mul && isLeftShift)
        if (Constant *BOOp = dyn_cast<Constant>(BO->getOperand(1)))
          return BinaryOperator::createMul(BO->getOperand(0),
                                           ConstantExpr::getShl(BOOp, CUI));
    
    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldBinOpIntoSelect(I, SI, this))
        return R;

    // If the operand is an bitwise operator with a constant RHS, and the
    // shift is the only use, we can pull it out of the shift.
    if (Op0->hasOneUse())
      if (BinaryOperator *Op0BO = dyn_cast<BinaryOperator>(Op0))
        if (ConstantInt *Op0C = dyn_cast<ConstantInt>(Op0BO->getOperand(1))) {
          bool isValid = true;     // Valid only for And, Or, Xor
          bool highBitSet = false; // Transform if high bit of constant set?

          switch (Op0BO->getOpcode()) {
          default: isValid = false; break;   // Do not perform transform!
          case Instruction::Or:
          case Instruction::Xor:
            highBitSet = false;
            break;
          case Instruction::And:
            highBitSet = true;
            break;
          }

          // If this is a signed shift right, and the high bit is modified
          // by the logical operation, do not perform the transformation.
          // The highBitSet boolean indicates the value of the high bit of
          // the constant which would cause it to be modified for this
          // operation.
          //
          if (isValid && !isLeftShift && !I.getType()->isUnsigned()) {
            uint64_t Val = Op0C->getRawValue();
            isValid = ((Val & (1 << (TypeBits-1))) != 0) == highBitSet;
          }

          if (isValid) {
            Constant *NewRHS = ConstantExpr::get(I.getOpcode(), Op0C, CUI);

            Instruction *NewShift =
              new ShiftInst(I.getOpcode(), Op0BO->getOperand(0), CUI,
                            Op0BO->getName());
            Op0BO->setName("");
            InsertNewInstBefore(NewShift, I);

            return BinaryOperator::create(Op0BO->getOpcode(), NewShift,
                                          NewRHS);
          }
        }

    // If this is a shift of a shift, see if we can fold the two together...
    if (ShiftInst *Op0SI = dyn_cast<ShiftInst>(Op0))
      if (ConstantUInt *ShiftAmt1C =
                                 dyn_cast<ConstantUInt>(Op0SI->getOperand(1))) {
        unsigned ShiftAmt1 = ShiftAmt1C->getValue();
        unsigned ShiftAmt2 = CUI->getValue();
        
        // Check for (A << c1) << c2   and   (A >> c1) >> c2
        if (I.getOpcode() == Op0SI->getOpcode()) {
          unsigned Amt = ShiftAmt1+ShiftAmt2;   // Fold into one big shift...
          if (Op0->getType()->getPrimitiveSize()*8 < Amt)
            Amt = Op0->getType()->getPrimitiveSize()*8;
          return new ShiftInst(I.getOpcode(), Op0SI->getOperand(0),
                               ConstantUInt::get(Type::UByteTy, Amt));
        }
        
        // Check for (A << c1) >> c2 or visaversa.  If we are dealing with
        // signed types, we can only support the (A >> c1) << c2 configuration,
        // because it can not turn an arbitrary bit of A into a sign bit.
        if (I.getType()->isUnsigned() || isLeftShift) {
          // Calculate bitmask for what gets shifted off the edge...
          Constant *C = ConstantIntegral::getAllOnesValue(I.getType());
          if (isLeftShift)
            C = ConstantExpr::getShl(C, ShiftAmt1C);
          else
            C = ConstantExpr::getShr(C, ShiftAmt1C);
          
          Instruction *Mask =
            BinaryOperator::createAnd(Op0SI->getOperand(0), C,
                                      Op0SI->getOperand(0)->getName()+".mask");
          InsertNewInstBefore(Mask, I);
          
          // Figure out what flavor of shift we should use...
          if (ShiftAmt1 == ShiftAmt2)
            return ReplaceInstUsesWith(I, Mask);  // (A << c) >> c  === A & c2
          else if (ShiftAmt1 < ShiftAmt2) {
            return new ShiftInst(I.getOpcode(), Mask,
                         ConstantUInt::get(Type::UByteTy, ShiftAmt2-ShiftAmt1));
          } else {
            return new ShiftInst(Op0SI->getOpcode(), Mask,
                         ConstantUInt::get(Type::UByteTy, ShiftAmt1-ShiftAmt2));
          }
        }
      }
  }

  return 0;
}

enum CastType {
  Noop     = 0,
  Truncate = 1,
  Signext  = 2,
  Zeroext  = 3
};

/// getCastType - In the future, we will split the cast instruction into these
/// various types.  Until then, we have to do the analysis here.
static CastType getCastType(const Type *Src, const Type *Dest) {
  assert(Src->isIntegral() && Dest->isIntegral() &&
         "Only works on integral types!");
  unsigned SrcSize = Src->getPrimitiveSize()*8;
  if (Src == Type::BoolTy) SrcSize = 1;
  unsigned DestSize = Dest->getPrimitiveSize()*8;
  if (Dest == Type::BoolTy) DestSize = 1;

  if (SrcSize == DestSize) return Noop;
  if (SrcSize > DestSize)  return Truncate;
  if (Src->isSigned()) return Signext;
  return Zeroext;
}


// isEliminableCastOfCast - Return true if it is valid to eliminate the CI
// instruction.
//
static inline bool isEliminableCastOfCast(const Type *SrcTy, const Type *MidTy,
                                          const Type *DstTy, TargetData *TD) {

  // It is legal to eliminate the instruction if casting A->B->A if the sizes
  // are identical and the bits don't get reinterpreted (for example 
  // int->float->int would not be allowed).
  if (SrcTy == DstTy && SrcTy->isLosslesslyConvertibleTo(MidTy))
    return true;

  // If we are casting between pointer and integer types, treat pointers as
  // integers of the appropriate size for the code below.
  if (isa<PointerType>(SrcTy)) SrcTy = TD->getIntPtrType();
  if (isa<PointerType>(MidTy)) MidTy = TD->getIntPtrType();
  if (isa<PointerType>(DstTy)) DstTy = TD->getIntPtrType();

  // Allow free casting and conversion of sizes as long as the sign doesn't
  // change...
  if (SrcTy->isIntegral() && MidTy->isIntegral() && DstTy->isIntegral()) {
    CastType FirstCast = getCastType(SrcTy, MidTy);
    CastType SecondCast = getCastType(MidTy, DstTy);

    // Capture the effect of these two casts.  If the result is a legal cast,
    // the CastType is stored here, otherwise a special code is used.
    static const unsigned CastResult[] = {
      // First cast is noop
      0, 1, 2, 3,
      // First cast is a truncate
      1, 1, 4, 4,         // trunc->extend is not safe to eliminate
      // First cast is a sign ext
      2, 5, 2, 4,         // signext->zeroext never ok
      // First cast is a zero ext
      3, 5, 3, 3,
    };

    unsigned Result = CastResult[FirstCast*4+SecondCast];
    switch (Result) {
    default: assert(0 && "Illegal table value!");
    case 0:
    case 1:
    case 2:
    case 3:
      // FIXME: in the future, when LLVM has explicit sign/zeroextends and
      // truncates, we could eliminate more casts.
      return (unsigned)getCastType(SrcTy, DstTy) == Result;
    case 4:
      return false;  // Not possible to eliminate this here.
    case 5:
      // Sign or zero extend followed by truncate is always ok if the result
      // is a truncate or noop.
      CastType ResultCast = getCastType(SrcTy, DstTy);
      if (ResultCast == Noop || ResultCast == Truncate)
        return true;
      // Otherwise we are still growing the value, we are only safe if the 
      // result will match the sign/zeroextendness of the result.
      return ResultCast == FirstCast;
    }
  }
  return false;
}

static bool ValueRequiresCast(const Value *V, const Type *Ty, TargetData *TD) {
  if (V->getType() == Ty || isa<Constant>(V)) return false;
  if (const CastInst *CI = dyn_cast<CastInst>(V))
    if (isEliminableCastOfCast(CI->getOperand(0)->getType(), CI->getType(), Ty,
                               TD))
      return false;
  return true;
}

/// InsertOperandCastBefore - This inserts a cast of V to DestTy before the
/// InsertBefore instruction.  This is specialized a bit to avoid inserting
/// casts that are known to not do anything...
///
Value *InstCombiner::InsertOperandCastBefore(Value *V, const Type *DestTy,
                                             Instruction *InsertBefore) {
  if (V->getType() == DestTy) return V;
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getCast(C, DestTy);

  CastInst *CI = new CastInst(V, DestTy, V->getName());
  InsertNewInstBefore(CI, *InsertBefore);
  return CI;
}

// CastInst simplification
//
Instruction *InstCombiner::visitCastInst(CastInst &CI) {
  Value *Src = CI.getOperand(0);

  // If the user is casting a value to the same type, eliminate this cast
  // instruction...
  if (CI.getType() == Src->getType())
    return ReplaceInstUsesWith(CI, Src);

  // If casting the result of another cast instruction, try to eliminate this
  // one!
  //
  if (CastInst *CSrc = dyn_cast<CastInst>(Src)) {
    if (isEliminableCastOfCast(CSrc->getOperand(0)->getType(),
                               CSrc->getType(), CI.getType(), TD)) {
      // This instruction now refers directly to the cast's src operand.  This
      // has a good chance of making CSrc dead.
      CI.setOperand(0, CSrc->getOperand(0));
      return &CI;
    }

    // If this is an A->B->A cast, and we are dealing with integral types, try
    // to convert this into a logical 'and' instruction.
    //
    if (CSrc->getOperand(0)->getType() == CI.getType() &&
        CI.getType()->isInteger() && CSrc->getType()->isInteger() &&
        CI.getType()->isUnsigned() && CSrc->getType()->isUnsigned() &&
        CSrc->getType()->getPrimitiveSize() < CI.getType()->getPrimitiveSize()){
      assert(CSrc->getType() != Type::ULongTy &&
             "Cannot have type bigger than ulong!");
      uint64_t AndValue = (1ULL << CSrc->getType()->getPrimitiveSize()*8)-1;
      Constant *AndOp = ConstantUInt::get(CI.getType(), AndValue);
      return BinaryOperator::createAnd(CSrc->getOperand(0), AndOp);
    }
  }

  // If this is a cast to bool, turn it into the appropriate setne instruction.
  if (CI.getType() == Type::BoolTy)
    return BinaryOperator::createSetNE(CI.getOperand(0),
                       Constant::getNullValue(CI.getOperand(0)->getType()));

  // If casting the result of a getelementptr instruction with no offset, turn
  // this into a cast of the original pointer!
  //
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Src)) {
    bool AllZeroOperands = true;
    for (unsigned i = 1, e = GEP->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEP->getOperand(i)) ||
          !cast<Constant>(GEP->getOperand(i))->isNullValue()) {
        AllZeroOperands = false;
        break;
      }
    if (AllZeroOperands) {
      CI.setOperand(0, GEP->getOperand(0));
      return &CI;
    }
  }

  // If we are casting a malloc or alloca to a pointer to a type of the same
  // size, rewrite the allocation instruction to allocate the "right" type.
  //
  if (AllocationInst *AI = dyn_cast<AllocationInst>(Src))
    if (AI->hasOneUse() && !AI->isArrayAllocation())
      if (const PointerType *PTy = dyn_cast<PointerType>(CI.getType())) {
        // Get the type really allocated and the type casted to...
        const Type *AllocElTy = AI->getAllocatedType();
        const Type *CastElTy = PTy->getElementType();
        if (AllocElTy->isSized() && CastElTy->isSized()) {
          unsigned AllocElTySize = TD->getTypeSize(AllocElTy);
          unsigned CastElTySize = TD->getTypeSize(CastElTy);

          // If the allocation is for an even multiple of the cast type size
          if (CastElTySize && (AllocElTySize % CastElTySize == 0)) {
            Value *Amt = ConstantUInt::get(Type::UIntTy, 
                                         AllocElTySize/CastElTySize);
            std::string Name = AI->getName(); AI->setName("");
            AllocationInst *New;
            if (isa<MallocInst>(AI))
              New = new MallocInst(CastElTy, Amt, Name);
            else
              New = new AllocaInst(CastElTy, Amt, Name);
            InsertNewInstBefore(New, *AI);
            return ReplaceInstUsesWith(CI, New);
          }
        }
      }

  // If the source value is an instruction with only this use, we can attempt to
  // propagate the cast into the instruction.  Also, only handle integral types
  // for now.
  if (Instruction *SrcI = dyn_cast<Instruction>(Src))
    if (SrcI->hasOneUse() && Src->getType()->isIntegral() &&
        CI.getType()->isInteger()) {  // Don't mess with casts to bool here
      const Type *DestTy = CI.getType();
      unsigned SrcBitSize = getTypeSizeInBits(Src->getType());
      unsigned DestBitSize = getTypeSizeInBits(DestTy);

      Value *Op0 = SrcI->getNumOperands() > 0 ? SrcI->getOperand(0) : 0;
      Value *Op1 = SrcI->getNumOperands() > 1 ? SrcI->getOperand(1) : 0;

      switch (SrcI->getOpcode()) {
      case Instruction::Add:
      case Instruction::Mul:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:
        // If we are discarding information, or just changing the sign, rewrite.
        if (DestBitSize <= SrcBitSize && DestBitSize != 1) {
          // Don't insert two casts if they cannot be eliminated.  We allow two
          // casts to be inserted if the sizes are the same.  This could only be
          // converting signedness, which is a noop.
          if (DestBitSize == SrcBitSize || !ValueRequiresCast(Op1, DestTy,TD) ||
              !ValueRequiresCast(Op0, DestTy, TD)) {
            Value *Op0c = InsertOperandCastBefore(Op0, DestTy, SrcI);
            Value *Op1c = InsertOperandCastBefore(Op1, DestTy, SrcI);
            return BinaryOperator::create(cast<BinaryOperator>(SrcI)
                             ->getOpcode(), Op0c, Op1c);
          }
        }
        break;
      case Instruction::Shl:
        // Allow changing the sign of the source operand.  Do not allow changing
        // the size of the shift, UNLESS the shift amount is a constant.  We
        // mush not change variable sized shifts to a smaller size, because it
        // is undefined to shift more bits out than exist in the value.
        if (DestBitSize == SrcBitSize ||
            (DestBitSize < SrcBitSize && isa<Constant>(Op1))) {
          Value *Op0c = InsertOperandCastBefore(Op0, DestTy, SrcI);
          return new ShiftInst(Instruction::Shl, Op0c, Op1);
        }
        break;
      }
    }
  
  return 0;
}

/// GetSelectFoldableOperands - We want to turn code that looks like this:
///   %C = or %A, %B
///   %D = select %cond, %C, %A
/// into:
///   %C = select %cond, %B, 0
///   %D = or %A, %C
///
/// Assuming that the specified instruction is an operand to the select, return
/// a bitmask indicating which operands of this instruction are foldable if they
/// equal the other incoming value of the select.
///
static unsigned GetSelectFoldableOperands(Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Add:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return 3;              // Can fold through either operand.
  case Instruction::Sub:   // Can only fold on the amount subtracted.
  case Instruction::Shl:   // Can only fold on the shift amount.
  case Instruction::Shr:
    return 1;           
  default:
    return 0;              // Cannot fold
  }
}

/// GetSelectFoldableConstant - For the same transformation as the previous
/// function, return the identity constant that goes into the select.
static Constant *GetSelectFoldableConstant(Instruction *I) {
  switch (I->getOpcode()) {
  default: assert(0 && "This cannot happen!"); abort();
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Or:
  case Instruction::Xor:
    return Constant::getNullValue(I->getType());
  case Instruction::Shl:
  case Instruction::Shr:
    return Constant::getNullValue(Type::UByteTy);
  case Instruction::And:
    return ConstantInt::getAllOnesValue(I->getType());
  case Instruction::Mul:
    return ConstantInt::get(I->getType(), 1);
  }
}

Instruction *InstCombiner::visitSelectInst(SelectInst &SI) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  // select true, X, Y  -> X
  // select false, X, Y -> Y
  if (ConstantBool *C = dyn_cast<ConstantBool>(CondVal))
    if (C == ConstantBool::True)
      return ReplaceInstUsesWith(SI, TrueVal);
    else {
      assert(C == ConstantBool::False);
      return ReplaceInstUsesWith(SI, FalseVal);
    }

  // select C, X, X -> X
  if (TrueVal == FalseVal)
    return ReplaceInstUsesWith(SI, TrueVal);

  if (SI.getType() == Type::BoolTy)
    if (ConstantBool *C = dyn_cast<ConstantBool>(TrueVal)) {
      if (C == ConstantBool::True) {
        // Change: A = select B, true, C --> A = or B, C
        return BinaryOperator::createOr(CondVal, FalseVal);
      } else {
        // Change: A = select B, false, C --> A = and !B, C
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::createNot(CondVal,
                                             "not."+CondVal->getName()), SI);
        return BinaryOperator::createAnd(NotCond, FalseVal);
      }
    } else if (ConstantBool *C = dyn_cast<ConstantBool>(FalseVal)) {
      if (C == ConstantBool::False) {
        // Change: A = select B, C, false --> A = and B, C
        return BinaryOperator::createAnd(CondVal, TrueVal);
      } else {
        // Change: A = select B, C, true --> A = or !B, C
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::createNot(CondVal,
                                             "not."+CondVal->getName()), SI);
        return BinaryOperator::createOr(NotCond, TrueVal);
      }
    }

  // Selecting between two integer constants?
  if (ConstantInt *TrueValC = dyn_cast<ConstantInt>(TrueVal))
    if (ConstantInt *FalseValC = dyn_cast<ConstantInt>(FalseVal)) {
      // select C, 1, 0 -> cast C to int
      if (FalseValC->isNullValue() && TrueValC->getRawValue() == 1) {
        return new CastInst(CondVal, SI.getType());
      } else if (TrueValC->isNullValue() && FalseValC->getRawValue() == 1) {
        // select C, 0, 1 -> cast !C to int
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::createNot(CondVal,
                                               "not."+CondVal->getName()), SI);
        return new CastInst(NotCond, SI.getType());
      }

      // If one of the constants is zero (we know they can't both be) and we
      // have a setcc instruction with zero, and we have an 'and' with the
      // non-constant value, eliminate this whole mess.  This corresponds to
      // cases like this: ((X & 27) ? 27 : 0)
      if (TrueValC->isNullValue() || FalseValC->isNullValue())
        if (Instruction *IC = dyn_cast<Instruction>(SI.getCondition()))
          if ((IC->getOpcode() == Instruction::SetEQ ||
               IC->getOpcode() == Instruction::SetNE) &&
              isa<ConstantInt>(IC->getOperand(1)) &&
              cast<Constant>(IC->getOperand(1))->isNullValue())
            if (Instruction *ICA = dyn_cast<Instruction>(IC->getOperand(0)))
              if (ICA->getOpcode() == Instruction::And &&
                  isa<ConstantInt>(ICA->getOperand(1)) && 
                  (ICA->getOperand(1) == TrueValC || 
                   ICA->getOperand(1) == FalseValC) && 
                  isOneBitSet(cast<ConstantInt>(ICA->getOperand(1)))) {
                // Okay, now we know that everything is set up, we just don't
                // know whether we have a setne or seteq and whether the true or
                // false val is the zero.
                bool ShouldNotVal = !TrueValC->isNullValue();
                ShouldNotVal ^= IC->getOpcode() == Instruction::SetNE;
                Value *V = ICA;
                if (ShouldNotVal)
                  V = InsertNewInstBefore(BinaryOperator::create(
                                  Instruction::Xor, V, ICA->getOperand(1)), SI);
                return ReplaceInstUsesWith(SI, V);
              }
    }

  // See if we are selecting two values based on a comparison of the two values.
  if (SetCondInst *SCI = dyn_cast<SetCondInst>(CondVal)) {
    if (SCI->getOperand(0) == TrueVal && SCI->getOperand(1) == FalseVal) {
      // Transform (X == Y) ? X : Y  -> Y
      if (SCI->getOpcode() == Instruction::SetEQ)
        return ReplaceInstUsesWith(SI, FalseVal);
      // Transform (X != Y) ? X : Y  -> X
      if (SCI->getOpcode() == Instruction::SetNE)
        return ReplaceInstUsesWith(SI, TrueVal);
      // NOTE: if we wanted to, this is where to detect MIN/MAX/ABS/etc.

    } else if (SCI->getOperand(0) == FalseVal && SCI->getOperand(1) == TrueVal){
      // Transform (X == Y) ? Y : X  -> X
      if (SCI->getOpcode() == Instruction::SetEQ)
        return ReplaceInstUsesWith(SI, FalseVal);
      // Transform (X != Y) ? Y : X  -> Y
      if (SCI->getOpcode() == Instruction::SetNE)
        return ReplaceInstUsesWith(SI, TrueVal);
      // NOTE: if we wanted to, this is where to detect MIN/MAX/ABS/etc.
    }
  }
  
  // See if we can fold the select into one of our operands.
  if (SI.getType()->isInteger()) {
    // See the comment above GetSelectFoldableOperands for a description of the
    // transformation we are doing here.
    if (Instruction *TVI = dyn_cast<Instruction>(TrueVal))
      if (TVI->hasOneUse() && TVI->getNumOperands() == 2 &&
          !isa<Constant>(FalseVal))
        if (unsigned SFO = GetSelectFoldableOperands(TVI)) {
          unsigned OpToFold = 0;
          if ((SFO & 1) && FalseVal == TVI->getOperand(0)) {
            OpToFold = 1;
          } else  if ((SFO & 2) && FalseVal == TVI->getOperand(1)) {
            OpToFold = 2;
          }

          if (OpToFold) {
            Constant *C = GetSelectFoldableConstant(TVI);
            std::string Name = TVI->getName(); TVI->setName("");
            Instruction *NewSel =
              new SelectInst(SI.getCondition(), TVI->getOperand(2-OpToFold), C,
                             Name);
            InsertNewInstBefore(NewSel, SI);
            if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TVI))
              return BinaryOperator::create(BO->getOpcode(), FalseVal, NewSel);
            else if (ShiftInst *SI = dyn_cast<ShiftInst>(TVI))
              return new ShiftInst(SI->getOpcode(), FalseVal, NewSel);
            else {
              assert(0 && "Unknown instruction!!");
            }
          }
        }
    
    if (Instruction *FVI = dyn_cast<Instruction>(FalseVal))
      if (FVI->hasOneUse() && FVI->getNumOperands() == 2 &&
          !isa<Constant>(TrueVal))
        if (unsigned SFO = GetSelectFoldableOperands(FVI)) {
          unsigned OpToFold = 0;
          if ((SFO & 1) && TrueVal == FVI->getOperand(0)) {
            OpToFold = 1;
          } else  if ((SFO & 2) && TrueVal == FVI->getOperand(1)) {
            OpToFold = 2;
          }

          if (OpToFold) {
            Constant *C = GetSelectFoldableConstant(FVI);
            std::string Name = FVI->getName(); FVI->setName("");
            Instruction *NewSel =
              new SelectInst(SI.getCondition(), C, FVI->getOperand(2-OpToFold),
                             Name);
            InsertNewInstBefore(NewSel, SI);
            if (BinaryOperator *BO = dyn_cast<BinaryOperator>(FVI))
              return BinaryOperator::create(BO->getOpcode(), TrueVal, NewSel);
            else if (ShiftInst *SI = dyn_cast<ShiftInst>(FVI))
              return new ShiftInst(SI->getOpcode(), TrueVal, NewSel);
            else {
              assert(0 && "Unknown instruction!!");
            }
          }
        }
  }
  return 0;
}


// CallInst simplification
//
Instruction *InstCombiner::visitCallInst(CallInst &CI) {
  // Intrinsics cannot occur in an invoke, so handle them here instead of in
  // visitCallSite.
  if (Function *F = CI.getCalledFunction())
    switch (F->getIntrinsicID()) {
    case Intrinsic::memmove:
    case Intrinsic::memcpy:
    case Intrinsic::memset:
      // memmove/cpy/set of zero bytes is a noop.
      if (Constant *NumBytes = dyn_cast<Constant>(CI.getOperand(3))) {
        if (NumBytes->isNullValue())
          return EraseInstFromFunction(CI);
      }
      break;
    default:
      break;
    }

  return visitCallSite(&CI);
}

// InvokeInst simplification
//
Instruction *InstCombiner::visitInvokeInst(InvokeInst &II) {
  return visitCallSite(&II);
}

// visitCallSite - Improvements for call and invoke instructions.
//
Instruction *InstCombiner::visitCallSite(CallSite CS) {
  bool Changed = false;

  // If the callee is a constexpr cast of a function, attempt to move the cast
  // to the arguments of the call/invoke.
  if (transformConstExprCastCall(CS)) return 0;

  Value *Callee = CS.getCalledValue();
  const PointerType *PTy = cast<PointerType>(Callee->getType());
  const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  if (FTy->isVarArg()) {
    // See if we can optimize any arguments passed through the varargs area of
    // the call.
    for (CallSite::arg_iterator I = CS.arg_begin()+FTy->getNumParams(),
           E = CS.arg_end(); I != E; ++I)
      if (CastInst *CI = dyn_cast<CastInst>(*I)) {
        // If this cast does not effect the value passed through the varargs
        // area, we can eliminate the use of the cast.
        Value *Op = CI->getOperand(0);
        if (CI->getType()->isLosslesslyConvertibleTo(Op->getType())) {
          *I = Op;
          Changed = true;
        }
      }
  }
  
  return Changed ? CS.getInstruction() : 0;
}

// transformConstExprCastCall - If the callee is a constexpr cast of a function,
// attempt to move the cast to the arguments of the call/invoke.
//
bool InstCombiner::transformConstExprCastCall(CallSite CS) {
  if (!isa<ConstantExpr>(CS.getCalledValue())) return false;
  ConstantExpr *CE = cast<ConstantExpr>(CS.getCalledValue());
  if (CE->getOpcode() != Instruction::Cast || !isa<Function>(CE->getOperand(0)))
    return false;
  Function *Callee = cast<Function>(CE->getOperand(0));
  Instruction *Caller = CS.getInstruction();

  // Okay, this is a cast from a function to a different type.  Unless doing so
  // would cause a type conversion of one of our arguments, change this call to
  // be a direct call with arguments casted to the appropriate types.
  //
  const FunctionType *FT = Callee->getFunctionType();
  const Type *OldRetTy = Caller->getType();

  // Check to see if we are changing the return type...
  if (OldRetTy != FT->getReturnType()) {
    if (Callee->isExternal() &&
        !OldRetTy->isLosslesslyConvertibleTo(FT->getReturnType()) &&
        !Caller->use_empty())
      return false;   // Cannot transform this return value...

    // If the callsite is an invoke instruction, and the return value is used by
    // a PHI node in a successor, we cannot change the return type of the call
    // because there is no place to put the cast instruction (without breaking
    // the critical edge).  Bail out in this case.
    if (!Caller->use_empty())
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller))
        for (Value::use_iterator UI = II->use_begin(), E = II->use_end();
             UI != E; ++UI)
          if (PHINode *PN = dyn_cast<PHINode>(*UI))
            if (PN->getParent() == II->getNormalDest() ||
                PN->getParent() == II->getUnwindDest())
              return false;
  }

  unsigned NumActualArgs = unsigned(CS.arg_end()-CS.arg_begin());
  unsigned NumCommonArgs = std::min(FT->getNumParams(), NumActualArgs);
                                    
  CallSite::arg_iterator AI = CS.arg_begin();
  for (unsigned i = 0, e = NumCommonArgs; i != e; ++i, ++AI) {
    const Type *ParamTy = FT->getParamType(i);
    bool isConvertible = (*AI)->getType()->isLosslesslyConvertibleTo(ParamTy);
    if (Callee->isExternal() && !isConvertible) return false;    
  }

  if (FT->getNumParams() < NumActualArgs && !FT->isVarArg() &&
      Callee->isExternal())
    return false;   // Do not delete arguments unless we have a function body...

  // Okay, we decided that this is a safe thing to do: go ahead and start
  // inserting cast instructions as necessary...
  std::vector<Value*> Args;
  Args.reserve(NumActualArgs);

  AI = CS.arg_begin();
  for (unsigned i = 0; i != NumCommonArgs; ++i, ++AI) {
    const Type *ParamTy = FT->getParamType(i);
    if ((*AI)->getType() == ParamTy) {
      Args.push_back(*AI);
    } else {
      Args.push_back(InsertNewInstBefore(new CastInst(*AI, ParamTy, "tmp"),
                                         *Caller));
    }
  }

  // If the function takes more arguments than the call was taking, add them
  // now...
  for (unsigned i = NumCommonArgs; i != FT->getNumParams(); ++i)
    Args.push_back(Constant::getNullValue(FT->getParamType(i)));

  // If we are removing arguments to the function, emit an obnoxious warning...
  if (FT->getNumParams() < NumActualArgs)
    if (!FT->isVarArg()) {
      std::cerr << "WARNING: While resolving call to function '"
                << Callee->getName() << "' arguments were dropped!\n";
    } else {
      // Add all of the arguments in their promoted form to the arg list...
      for (unsigned i = FT->getNumParams(); i != NumActualArgs; ++i, ++AI) {
        const Type *PTy = getPromotedType((*AI)->getType());
        if (PTy != (*AI)->getType()) {
          // Must promote to pass through va_arg area!
          Instruction *Cast = new CastInst(*AI, PTy, "tmp");
          InsertNewInstBefore(Cast, *Caller);
          Args.push_back(Cast);
        } else {
          Args.push_back(*AI);
        }
      }
    }

  if (FT->getReturnType() == Type::VoidTy)
    Caller->setName("");   // Void type should not have a name...

  Instruction *NC;
  if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
    NC = new InvokeInst(Callee, II->getNormalDest(), II->getUnwindDest(),
                        Args, Caller->getName(), Caller);
  } else {
    NC = new CallInst(Callee, Args, Caller->getName(), Caller);
  }

  // Insert a cast of the return type as necessary...
  Value *NV = NC;
  if (Caller->getType() != NV->getType() && !Caller->use_empty()) {
    if (NV->getType() != Type::VoidTy) {
      NV = NC = new CastInst(NC, Caller->getType(), "tmp");

      // If this is an invoke instruction, we should insert it after the first
      // non-phi, instruction in the normal successor block.
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        BasicBlock::iterator I = II->getNormalDest()->begin();
        while (isa<PHINode>(I)) ++I;
        InsertNewInstBefore(NC, *I);
      } else {
        // Otherwise, it's a call, just insert cast right after the call instr
        InsertNewInstBefore(NC, *Caller);
      }
      AddUsersToWorkList(*Caller);
    } else {
      NV = Constant::getNullValue(Caller->getType());
    }
  }

  if (Caller->getType() != Type::VoidTy && !Caller->use_empty())
    Caller->replaceAllUsesWith(NV);
  Caller->getParent()->getInstList().erase(Caller);
  removeFromWorkList(Caller);
  return true;
}



// PHINode simplification
//
Instruction *InstCombiner::visitPHINode(PHINode &PN) {
  if (Value *V = hasConstantValue(&PN))
    return ReplaceInstUsesWith(PN, V);

  // If the only user of this instruction is a cast instruction, and all of the
  // incoming values are constants, change this PHI to merge together the casted
  // constants.
  if (PN.hasOneUse())
    if (CastInst *CI = dyn_cast<CastInst>(PN.use_back()))
      if (CI->getType() != PN.getType()) {  // noop casts will be folded
        bool AllConstant = true;
        for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
          if (!isa<Constant>(PN.getIncomingValue(i))) {
            AllConstant = false;
            break;
          }
        if (AllConstant) {
          // Make a new PHI with all casted values.
          PHINode *New = new PHINode(CI->getType(), PN.getName(), &PN);
          for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
            Constant *OldArg = cast<Constant>(PN.getIncomingValue(i));
            New->addIncoming(ConstantExpr::getCast(OldArg, New->getType()),
                             PN.getIncomingBlock(i));
          }

          // Update the cast instruction.
          CI->setOperand(0, New);
          WorkList.push_back(CI);    // revisit the cast instruction to fold.
          WorkList.push_back(New);   // Make sure to revisit the new Phi
          return &PN;                // PN is now dead!
        }
      }
  return 0;
}

static Value *InsertSignExtendToPtrTy(Value *V, const Type *DTy,
                                      Instruction *InsertPoint,
                                      InstCombiner *IC) {
  unsigned PS = IC->getTargetData().getPointerSize();
  const Type *VTy = V->getType();
  Instruction *Cast;
  if (!VTy->isSigned() && VTy->getPrimitiveSize() < PS)
    // We must insert a cast to ensure we sign-extend.
    V = IC->InsertNewInstBefore(new CastInst(V, VTy->getSignedVersion(),
                                             V->getName()), *InsertPoint);
  return IC->InsertNewInstBefore(new CastInst(V, DTy, V->getName()),
                                 *InsertPoint);
}


Instruction *InstCombiner::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  Value *PtrOp = GEP.getOperand(0);
  // Is it 'getelementptr %P, long 0'  or 'getelementptr %P'
  // If so, eliminate the noop.
  if (GEP.getNumOperands() == 1)
    return ReplaceInstUsesWith(GEP, PtrOp);

  bool HasZeroPointerIndex = false;
  if (Constant *C = dyn_cast<Constant>(GEP.getOperand(1)))
    HasZeroPointerIndex = C->isNullValue();

  if (GEP.getNumOperands() == 2 && HasZeroPointerIndex)
    return ReplaceInstUsesWith(GEP, PtrOp);

  // Eliminate unneeded casts for indices.
  bool MadeChange = false;
  gep_type_iterator GTI = gep_type_begin(GEP);
  for (unsigned i = 1, e = GEP.getNumOperands(); i != e; ++i, ++GTI)
    if (isa<SequentialType>(*GTI)) {
      if (CastInst *CI = dyn_cast<CastInst>(GEP.getOperand(i))) {
        Value *Src = CI->getOperand(0);
        const Type *SrcTy = Src->getType();
        const Type *DestTy = CI->getType();
        if (Src->getType()->isInteger()) {
          if (SrcTy->getPrimitiveSize() == DestTy->getPrimitiveSize()) {
            // We can always eliminate a cast from ulong or long to the other.
            // We can always eliminate a cast from uint to int or the other on
            // 32-bit pointer platforms.
            if (DestTy->getPrimitiveSize() >= TD->getPointerSize()) {
              MadeChange = true;
              GEP.setOperand(i, Src);
            }
          } else if (SrcTy->getPrimitiveSize() < DestTy->getPrimitiveSize() &&
                     SrcTy->getPrimitiveSize() == 4) {
            // We can always eliminate a cast from int to [u]long.  We can
            // eliminate a cast from uint to [u]long iff the target is a 32-bit
            // pointer target.
            if (SrcTy->isSigned() || 
                SrcTy->getPrimitiveSize() >= TD->getPointerSize()) {
              MadeChange = true;
              GEP.setOperand(i, Src);
            }
          }
        }
      }
      // If we are using a wider index than needed for this platform, shrink it
      // to what we need.  If the incoming value needs a cast instruction,
      // insert it.  This explicit cast can make subsequent optimizations more
      // obvious.
      Value *Op = GEP.getOperand(i);
      if (Op->getType()->getPrimitiveSize() > TD->getPointerSize())
        if (Constant *C = dyn_cast<Constant>(Op)) {
          GEP.setOperand(i, ConstantExpr::getCast(C,
                                     TD->getIntPtrType()->getSignedVersion()));
          MadeChange = true;
        } else {
          Op = InsertNewInstBefore(new CastInst(Op, TD->getIntPtrType(),
                                                Op->getName()), GEP);
          GEP.setOperand(i, Op);
          MadeChange = true;
        }

      // If this is a constant idx, make sure to canonicalize it to be a signed
      // operand, otherwise CSE and other optimizations are pessimized.
      if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(Op)) {
        GEP.setOperand(i, ConstantExpr::getCast(CUI,
                                          CUI->getType()->getSignedVersion()));
        MadeChange = true;
      }
    }
  if (MadeChange) return &GEP;

  // Combine Indices - If the source pointer to this getelementptr instruction
  // is a getelementptr instruction, combine the indices of the two
  // getelementptr instructions into a single instruction.
  //
  std::vector<Value*> SrcGEPOperands;
  if (GetElementPtrInst *Src = dyn_cast<GetElementPtrInst>(PtrOp)) {
    SrcGEPOperands.assign(Src->op_begin(), Src->op_end());
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(PtrOp)) {
    if (CE->getOpcode() == Instruction::GetElementPtr)
      SrcGEPOperands.assign(CE->op_begin(), CE->op_end());
  }

  if (!SrcGEPOperands.empty()) {
    // Note that if our source is a gep chain itself that we wait for that
    // chain to be resolved before we perform this transformation.  This
    // avoids us creating a TON of code in some cases.
    //
    if (isa<GetElementPtrInst>(SrcGEPOperands[0]) &&
        cast<Instruction>(SrcGEPOperands[0])->getNumOperands() == 2)
      return 0;   // Wait until our source is folded to completion.

    std::vector<Value *> Indices;

    // Find out whether the last index in the source GEP is a sequential idx.
    bool EndsWithSequential = false;
    for (gep_type_iterator I = gep_type_begin(*cast<User>(PtrOp)),
           E = gep_type_end(*cast<User>(PtrOp)); I != E; ++I)
      EndsWithSequential = !isa<StructType>(*I);
  
    // Can we combine the two pointer arithmetics offsets?
    if (EndsWithSequential) {
      // Replace: gep (gep %P, long B), long A, ...
      // With:    T = long A+B; gep %P, T, ...
      //
      Value *Sum, *SO1 = SrcGEPOperands.back(), *GO1 = GEP.getOperand(1);
      if (SO1 == Constant::getNullValue(SO1->getType())) {
        Sum = GO1;
      } else if (GO1 == Constant::getNullValue(GO1->getType())) {
        Sum = SO1;
      } else {
        // If they aren't the same type, convert both to an integer of the
        // target's pointer size.
        if (SO1->getType() != GO1->getType()) {
          if (Constant *SO1C = dyn_cast<Constant>(SO1)) {
            SO1 = ConstantExpr::getCast(SO1C, GO1->getType());
          } else if (Constant *GO1C = dyn_cast<Constant>(GO1)) {
            GO1 = ConstantExpr::getCast(GO1C, SO1->getType());
          } else {
            unsigned PS = TD->getPointerSize();
            Instruction *Cast;
            if (SO1->getType()->getPrimitiveSize() == PS) {
              // Convert GO1 to SO1's type.
              GO1 = InsertSignExtendToPtrTy(GO1, SO1->getType(), &GEP, this);

            } else if (GO1->getType()->getPrimitiveSize() == PS) {
              // Convert SO1 to GO1's type.
              SO1 = InsertSignExtendToPtrTy(SO1, GO1->getType(), &GEP, this);
            } else {
              const Type *PT = TD->getIntPtrType();
              SO1 = InsertSignExtendToPtrTy(SO1, PT, &GEP, this);
              GO1 = InsertSignExtendToPtrTy(GO1, PT, &GEP, this);
            }
          }
        }
        if (isa<Constant>(SO1) && isa<Constant>(GO1))
          Sum = ConstantExpr::getAdd(cast<Constant>(SO1), cast<Constant>(GO1));
        else {
          Sum = BinaryOperator::createAdd(SO1, GO1, PtrOp->getName()+".sum");
          InsertNewInstBefore(cast<Instruction>(Sum), GEP);
        }
      }

      // Recycle the GEP we already have if possible.
      if (SrcGEPOperands.size() == 2) {
        GEP.setOperand(0, SrcGEPOperands[0]);
        GEP.setOperand(1, Sum);
        return &GEP;
      } else {
        Indices.insert(Indices.end(), SrcGEPOperands.begin()+1,
                       SrcGEPOperands.end()-1);
        Indices.push_back(Sum);
        Indices.insert(Indices.end(), GEP.op_begin()+2, GEP.op_end());
      }
    } else if (isa<Constant>(*GEP.idx_begin()) && 
               cast<Constant>(*GEP.idx_begin())->isNullValue() &&
               SrcGEPOperands.size() != 1) { 
      // Otherwise we can do the fold if the first index of the GEP is a zero
      Indices.insert(Indices.end(), SrcGEPOperands.begin()+1,
                     SrcGEPOperands.end());
      Indices.insert(Indices.end(), GEP.idx_begin()+1, GEP.idx_end());
    }

    if (!Indices.empty())
      return new GetElementPtrInst(SrcGEPOperands[0], Indices, GEP.getName());

  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(PtrOp)) {
    // GEP of global variable.  If all of the indices for this GEP are
    // constants, we can promote this to a constexpr instead of an instruction.

    // Scan for nonconstants...
    std::vector<Constant*> Indices;
    User::op_iterator I = GEP.idx_begin(), E = GEP.idx_end();
    for (; I != E && isa<Constant>(*I); ++I)
      Indices.push_back(cast<Constant>(*I));

    if (I == E) {  // If they are all constants...
      Constant *CE = ConstantExpr::getGetElementPtr(GV, Indices);

      // Replace all uses of the GEP with the new constexpr...
      return ReplaceInstUsesWith(GEP, CE);
    }
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(PtrOp)) {
    if (CE->getOpcode() == Instruction::Cast) {
      if (HasZeroPointerIndex) {
        // transform: GEP (cast [10 x ubyte]* X to [0 x ubyte]*), long 0, ...
        // into     : GEP [10 x ubyte]* X, long 0, ...
        //
        // This occurs when the program declares an array extern like "int X[];"
        //
        Constant *X = CE->getOperand(0);
        const PointerType *CPTy = cast<PointerType>(CE->getType());
        if (const PointerType *XTy = dyn_cast<PointerType>(X->getType()))
          if (const ArrayType *XATy =
              dyn_cast<ArrayType>(XTy->getElementType()))
            if (const ArrayType *CATy =
                dyn_cast<ArrayType>(CPTy->getElementType()))
              if (CATy->getElementType() == XATy->getElementType()) {
                // At this point, we know that the cast source type is a pointer
                // to an array of the same type as the destination pointer
                // array.  Because the array type is never stepped over (there
                // is a leading zero) we can fold the cast into this GEP.
                GEP.setOperand(0, X);
                return &GEP;
              }
      }
    }
  }

  return 0;
}

Instruction *InstCombiner::visitAllocationInst(AllocationInst &AI) {
  // Convert: malloc Ty, C - where C is a constant != 1 into: malloc [C x Ty], 1
  if (AI.isArrayAllocation())    // Check C != 1
    if (const ConstantUInt *C = dyn_cast<ConstantUInt>(AI.getArraySize())) {
      const Type *NewTy = ArrayType::get(AI.getAllocatedType(), C->getValue());
      AllocationInst *New = 0;

      // Create and insert the replacement instruction...
      if (isa<MallocInst>(AI))
        New = new MallocInst(NewTy, 0, AI.getName());
      else {
        assert(isa<AllocaInst>(AI) && "Unknown type of allocation inst!");
        New = new AllocaInst(NewTy, 0, AI.getName());
      }

      InsertNewInstBefore(New, AI);
      
      // Scan to the end of the allocation instructions, to skip over a block of
      // allocas if possible...
      //
      BasicBlock::iterator It = New;
      while (isa<AllocationInst>(*It)) ++It;

      // Now that I is pointing to the first non-allocation-inst in the block,
      // insert our getelementptr instruction...
      //
      std::vector<Value*> Idx(2, Constant::getNullValue(Type::IntTy));
      Value *V = new GetElementPtrInst(New, Idx, New->getName()+".sub", It);

      // Now make everything use the getelementptr instead of the original
      // allocation.
      return ReplaceInstUsesWith(AI, V);
    }

  // If alloca'ing a zero byte object, replace the alloca with a null pointer.
  // Note that we only do this for alloca's, because malloc should allocate and
  // return a unique pointer, even for a zero byte allocation.
  if (isa<AllocaInst>(AI) && AI.getAllocatedType()->isSized() && 
      TD->getTypeSize(AI.getAllocatedType()) == 0)
    return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));

  return 0;
}

Instruction *InstCombiner::visitFreeInst(FreeInst &FI) {
  Value *Op = FI.getOperand(0);

  // Change free <ty>* (cast <ty2>* X to <ty>*) into free <ty2>* X
  if (CastInst *CI = dyn_cast<CastInst>(Op))
    if (isa<PointerType>(CI->getOperand(0)->getType())) {
      FI.setOperand(0, CI->getOperand(0));
      return &FI;
    }

  // If we have 'free null' delete the instruction.  This can happen in stl code
  // when lots of inlining happens.
  if (isa<ConstantPointerNull>(Op))
    return EraseInstFromFunction(FI);

  return 0;
}


/// GetGEPGlobalInitializer - Given a constant, and a getelementptr
/// constantexpr, return the constant value being addressed by the constant
/// expression, or null if something is funny.
///
static Constant *GetGEPGlobalInitializer(Constant *C, ConstantExpr *CE) {
  if (CE->getOperand(1) != Constant::getNullValue(CE->getOperand(1)->getType()))
    return 0;  // Do not allow stepping over the value!

  // Loop over all of the operands, tracking down which value we are
  // addressing...
  gep_type_iterator I = gep_type_begin(CE), E = gep_type_end(CE);
  for (++I; I != E; ++I)
    if (const StructType *STy = dyn_cast<StructType>(*I)) {
      ConstantUInt *CU = cast<ConstantUInt>(I.getOperand());
      assert(CU->getValue() < STy->getNumElements() &&
             "Struct index out of range!");
      if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
        C = CS->getOperand(CU->getValue());
      } else if (isa<ConstantAggregateZero>(C)) {
	C = Constant::getNullValue(STy->getElementType(CU->getValue()));
      } else {
        return 0;
      }
    } else if (ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand())) {
      const ArrayType *ATy = cast<ArrayType>(*I);
      if ((uint64_t)CI->getRawValue() >= ATy->getNumElements()) return 0;
      if (ConstantArray *CA = dyn_cast<ConstantArray>(C))
        C = CA->getOperand(CI->getRawValue());
      else if (isa<ConstantAggregateZero>(C))
        C = Constant::getNullValue(ATy->getElementType());
      else
        return 0;
    } else {
      return 0;
    }
  return C;
}

static Instruction *InstCombineLoadCast(InstCombiner &IC, LoadInst &LI) {
  User *CI = cast<User>(LI.getOperand(0));

  const Type *DestPTy = cast<PointerType>(CI->getType())->getElementType();
  if (const PointerType *SrcTy =
      dyn_cast<PointerType>(CI->getOperand(0)->getType())) {
    const Type *SrcPTy = SrcTy->getElementType();
    if (SrcPTy->isSized() && DestPTy->isSized() &&
        IC.getTargetData().getTypeSize(SrcPTy) == 
            IC.getTargetData().getTypeSize(DestPTy) &&
        (SrcPTy->isInteger() || isa<PointerType>(SrcPTy)) &&
        (DestPTy->isInteger() || isa<PointerType>(DestPTy))) {
      // Okay, we are casting from one integer or pointer type to another of
      // the same size.  Instead of casting the pointer before the load, cast
      // the result of the loaded value.
      Value *NewLoad = IC.InsertNewInstBefore(new LoadInst(CI->getOperand(0),
                                                           CI->getName()), LI);
      // Now cast the result of the load.
      return new CastInst(NewLoad, LI.getType());
    }
  }
  return 0;
}

Instruction *InstCombiner::visitLoadInst(LoadInst &LI) {
  Value *Op = LI.getOperand(0);
  if (LI.isVolatile()) return 0;

  if (Constant *C = dyn_cast<Constant>(Op))
    if (C->isNullValue())  // load null -> 0
      return ReplaceInstUsesWith(LI, Constant::getNullValue(LI.getType()));

  // Instcombine load (constant global) into the value loaded...
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Op))
    if (GV->isConstant() && !GV->isExternal())
      return ReplaceInstUsesWith(LI, GV->getInitializer());

  // Instcombine load (constantexpr_GEP global, 0, ...) into the value loaded...
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Op))
    if (CE->getOpcode() == Instruction::GetElementPtr) {
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(CE->getOperand(0)))
        if (GV->isConstant() && !GV->isExternal())
          if (Constant *V = GetGEPGlobalInitializer(GV->getInitializer(), CE))
            return ReplaceInstUsesWith(LI, V);
    } else if (CE->getOpcode() == Instruction::Cast) {
      if (Instruction *Res = InstCombineLoadCast(*this, LI))
        return Res;
    }

  // load (cast X) --> cast (load X) iff safe
  if (CastInst *CI = dyn_cast<CastInst>(Op))
    if (Instruction *Res = InstCombineLoadCast(*this, LI))
      return Res;

  return 0;
}


Instruction *InstCombiner::visitBranchInst(BranchInst &BI) {
  // Change br (not X), label True, label False to: br X, label False, True
  Value *X;
  BasicBlock *TrueDest;
  BasicBlock *FalseDest;
  if (match(&BI, m_Br(m_Not(m_Value(X)), TrueDest, FalseDest)) &&
      !isa<Constant>(X)) {
    // Swap Destinations and condition...
    BI.setCondition(X);
    BI.setSuccessor(0, FalseDest);
    BI.setSuccessor(1, TrueDest);
    return &BI;
  }

  // Cannonicalize setne -> seteq
  Instruction::BinaryOps Op; Value *Y;
  if (match(&BI, m_Br(m_SetCond(Op, m_Value(X), m_Value(Y)),
                      TrueDest, FalseDest)))
    if ((Op == Instruction::SetNE || Op == Instruction::SetLE ||
         Op == Instruction::SetGE) && BI.getCondition()->hasOneUse()) {
      SetCondInst *I = cast<SetCondInst>(BI.getCondition());
      std::string Name = I->getName(); I->setName("");
      Instruction::BinaryOps NewOpcode = SetCondInst::getInverseCondition(Op);
      Value *NewSCC =  BinaryOperator::create(NewOpcode, X, Y, Name, I);
      // Swap Destinations and condition...
      BI.setCondition(NewSCC);
      BI.setSuccessor(0, FalseDest);
      BI.setSuccessor(1, TrueDest);
      removeFromWorkList(I);
      I->getParent()->getInstList().erase(I);
      WorkList.push_back(cast<Instruction>(NewSCC));
      return &BI;
    }
  
  return 0;
}

Instruction *InstCombiner::visitSwitchInst(SwitchInst &SI) {
  Value *Cond = SI.getCondition();
  if (Instruction *I = dyn_cast<Instruction>(Cond)) {
    if (I->getOpcode() == Instruction::Add)
      if (ConstantInt *AddRHS = dyn_cast<ConstantInt>(I->getOperand(1))) {
        // change 'switch (X+4) case 1:' into 'switch (X) case -3'
        for (unsigned i = 2, e = SI.getNumOperands(); i != e; i += 2)
          SI.setOperand(i, ConstantExpr::getSub(cast<Constant>(SI.getOperand(i)),
                                                AddRHS));
        SI.setOperand(0, I->getOperand(0));
        WorkList.push_back(I);
        return &SI;
      }
  }
  return 0;
}


void InstCombiner::removeFromWorkList(Instruction *I) {
  WorkList.erase(std::remove(WorkList.begin(), WorkList.end(), I),
                 WorkList.end());
}

bool InstCombiner::runOnFunction(Function &F) {
  bool Changed = false;
  TD = &getAnalysis<TargetData>();

  for (inst_iterator i = inst_begin(F), e = inst_end(F); i != e; ++i)
    WorkList.push_back(&*i);


  while (!WorkList.empty()) {
    Instruction *I = WorkList.back();  // Get an instruction from the worklist
    WorkList.pop_back();

    // Check to see if we can DCE or ConstantPropagate the instruction...
    // Check to see if we can DIE the instruction...
    if (isInstructionTriviallyDead(I)) {
      // Add operands to the worklist...
      if (I->getNumOperands() < 4)
        AddUsesToWorkList(*I);
      ++NumDeadInst;

      I->getParent()->getInstList().erase(I);
      removeFromWorkList(I);
      continue;
    }

    // Instruction isn't dead, see if we can constant propagate it...
    if (Constant *C = ConstantFoldInstruction(I)) {
      // Add operands to the worklist...
      AddUsesToWorkList(*I);
      ReplaceInstUsesWith(*I, C);

      ++NumConstProp;
      I->getParent()->getInstList().erase(I);
      removeFromWorkList(I);
      continue;
    }

    // Now that we have an instruction, try combining it to simplify it...
    if (Instruction *Result = visit(*I)) {
      ++NumCombined;
      // Should we replace the old instruction with a new one?
      if (Result != I) {
        DEBUG(std::cerr << "IC: Old = " << *I
                        << "    New = " << *Result);

        // Everything uses the new instruction now.
        I->replaceAllUsesWith(Result);

        // Push the new instruction and any users onto the worklist.
        WorkList.push_back(Result);
        AddUsersToWorkList(*Result);

        // Move the name to the new instruction first...
        std::string OldName = I->getName(); I->setName("");
        Result->setName(OldName);

        // Insert the new instruction into the basic block...
        BasicBlock *InstParent = I->getParent();
        InstParent->getInstList().insert(I, Result);

        // Make sure that we reprocess all operands now that we reduced their
        // use counts.
        for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
          if (Instruction *OpI = dyn_cast<Instruction>(I->getOperand(i)))
            WorkList.push_back(OpI);

        // Instructions can end up on the worklist more than once.  Make sure
        // we do not process an instruction that has been deleted.
        removeFromWorkList(I);

        // Erase the old instruction.
        InstParent->getInstList().erase(I);
      } else {
        DEBUG(std::cerr << "IC: MOD = " << *I);

        // If the instruction was modified, it's possible that it is now dead.
        // if so, remove it.
        if (isInstructionTriviallyDead(I)) {
          // Make sure we process all operands now that we are reducing their
          // use counts.
          for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
            if (Instruction *OpI = dyn_cast<Instruction>(I->getOperand(i)))
              WorkList.push_back(OpI);
          
          // Instructions may end up in the worklist more than once.  Erase all
          // occurrances of this instruction.
          removeFromWorkList(I);
          I->getParent()->getInstList().erase(I);
        } else {
          WorkList.push_back(Result);
          AddUsersToWorkList(*Result);
        }
      }
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createInstructionCombiningPass() {
  return new InstCombiner();
}

