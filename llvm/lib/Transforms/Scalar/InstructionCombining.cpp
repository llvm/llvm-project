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
//   ... etc.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instcombine"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <iostream>
using namespace llvm;
using namespace llvm::PatternMatch;

namespace {
  Statistic<> NumCombined ("instcombine", "Number of insts combined");
  Statistic<> NumConstProp("instcombine", "Number of constant folds");
  Statistic<> NumDeadInst ("instcombine", "Number of dead inst eliminated");
  Statistic<> NumDeadStore("instcombine", "Number of dead stores eliminated");
  Statistic<> NumSunkInst ("instcombine", "Number of instructions sunk");

  class VISIBILITY_HIDDEN InstCombiner
    : public FunctionPass,
      public InstVisitor<InstCombiner, Instruction*> {
    // Worklist of all of the instructions that need to be simplified.
    std::vector<Instruction*> WorkList;
    TargetData *TD;

    /// AddUsersToWorkList - When an instruction is simplified, add all users of
    /// the instruction to the work lists because they might get more simplified
    /// now.
    ///
    void AddUsersToWorkList(Value &I) {
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
    
    /// AddSoonDeadInstToWorklist - The specified instruction is about to become
    /// dead.  Add all of its operands to the worklist, turning them into
    /// undef's to reduce the number of uses of those instructions.
    ///
    /// Return the specified operand before it is turned into an undef.
    ///
    Value *AddSoonDeadInstToWorklist(Instruction &I, unsigned op) {
      Value *R = I.getOperand(op);
      
      for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
        if (Instruction *Op = dyn_cast<Instruction>(I.getOperand(i))) {
          WorkList.push_back(Op);
          // Set the operand to undef to drop the use.
          I.setOperand(i, UndefValue::get(Op->getType()));
        }
      
      return R;
    }

    // removeFromWorkList - remove all instances of I from the worklist.
    void removeFromWorkList(Instruction *I);
  public:
    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
      AU.addPreservedID(LCSSAID);
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
    Instruction *visitURem(BinaryOperator &I);
    Instruction *visitSRem(BinaryOperator &I);
    Instruction *visitFRem(BinaryOperator &I);
    Instruction *commonRemTransforms(BinaryOperator &I);
    Instruction *commonIRemTransforms(BinaryOperator &I);
    Instruction *commonDivTransforms(BinaryOperator &I);
    Instruction *commonIDivTransforms(BinaryOperator &I);
    Instruction *visitUDiv(BinaryOperator &I);
    Instruction *visitSDiv(BinaryOperator &I);
    Instruction *visitFDiv(BinaryOperator &I);
    Instruction *visitAnd(BinaryOperator &I);
    Instruction *visitOr (BinaryOperator &I);
    Instruction *visitXor(BinaryOperator &I);
    Instruction *visitSetCondInst(SetCondInst &I);
    Instruction *visitSetCondInstWithCastAndCast(SetCondInst &SCI);

    Instruction *FoldGEPSetCC(User *GEPLHS, Value *RHS,
                              Instruction::BinaryOps Cond, Instruction &I);
    Instruction *visitShiftInst(ShiftInst &I);
    Instruction *FoldShiftByConstant(Value *Op0, ConstantInt *Op1,
                                     ShiftInst &I);
    Instruction *visitCastInst(CastInst &CI);
    Instruction *FoldSelectOpOp(SelectInst &SI, Instruction *TI,
                                Instruction *FI);
    Instruction *visitSelectInst(SelectInst &CI);
    Instruction *visitCallInst(CallInst &CI);
    Instruction *visitInvokeInst(InvokeInst &II);
    Instruction *visitPHINode(PHINode &PN);
    Instruction *visitGetElementPtrInst(GetElementPtrInst &GEP);
    Instruction *visitAllocationInst(AllocationInst &AI);
    Instruction *visitFreeInst(FreeInst &FI);
    Instruction *visitLoadInst(LoadInst &LI);
    Instruction *visitStoreInst(StoreInst &SI);
    Instruction *visitBranchInst(BranchInst &BI);
    Instruction *visitSwitchInst(SwitchInst &SI);
    Instruction *visitInsertElementInst(InsertElementInst &IE);
    Instruction *visitExtractElementInst(ExtractElementInst &EI);
    Instruction *visitShuffleVectorInst(ShuffleVectorInst &SVI);

    // visitInstruction - Specify what to return for unhandled instructions...
    Instruction *visitInstruction(Instruction &I) { return 0; }

  private:
    Instruction *visitCallSite(CallSite CS);
    bool transformConstExprCastCall(CallSite CS);

  public:
    // InsertNewInstBefore - insert an instruction New before instruction Old
    // in the program.  Add the new instruction to the worklist.
    //
    Instruction *InsertNewInstBefore(Instruction *New, Instruction &Old) {
      assert(New && New->getParent() == 0 &&
             "New instruction already inserted into a basic block!");
      BasicBlock *BB = Old.getParent();
      BB->getInstList().insert(&Old, New);  // Insert inst
      WorkList.push_back(New);              // Add to worklist
      return New;
    }

    /// InsertCastBefore - Insert a cast of V to TY before the instruction POS.
    /// This also adds the cast to the worklist.  Finally, this returns the
    /// cast.
    Value *InsertCastBefore(Value *V, const Type *Ty, Instruction &Pos) {
      if (V->getType() == Ty) return V;

      if (Constant *CV = dyn_cast<Constant>(V))
        return ConstantExpr::getCast(CV, Ty);
      
      Instruction *C = new CastInst(V, Ty, V->getName(), &Pos);
      WorkList.push_back(C);
      return C;
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
        I.replaceAllUsesWith(UndefValue::get(I.getType()));
        return &I;
      }
    }

    // UpdateValueUsesWith - This method is to be used when an value is
    // found to be replacable with another preexisting expression or was
    // updated.  Here we add all uses of I to the worklist, replace all uses of
    // I with the new value (unless the instruction was just updated), then
    // return true, so that the inst combiner will know that I was modified.
    //
    bool UpdateValueUsesWith(Value *Old, Value *New) {
      AddUsersToWorkList(*Old);         // Add all modified instrs to worklist
      if (Old != New)
        Old->replaceAllUsesWith(New);
      if (Instruction *I = dyn_cast<Instruction>(Old))
        WorkList.push_back(I);
      if (Instruction *I = dyn_cast<Instruction>(New))
        WorkList.push_back(I);
      return true;
    }
    
    // EraseInstFromFunction - When dealing with an instruction that has side
    // effects or produces a void value, we can't rely on DCE to delete the
    // instruction.  Instead, visit methods should return the value returned by
    // this function.
    Instruction *EraseInstFromFunction(Instruction &I) {
      assert(I.use_empty() && "Cannot erase instruction that is used!");
      AddUsesToWorkList(I);
      removeFromWorkList(&I);
      I.eraseFromParent();
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
    // operators.
    bool SimplifyCommutative(BinaryOperator &I);

    bool SimplifyDemandedBits(Value *V, uint64_t Mask, 
                              uint64_t &KnownZero, uint64_t &KnownOne,
                              unsigned Depth = 0);

    Value *SimplifyDemandedVectorElts(Value *V, uint64_t DemandedElts,
                                      uint64_t &UndefElts, unsigned Depth = 0);
      
    // FoldOpIntoPhi - Given a binary operator or cast instruction which has a
    // PHI node as operand #0, see if we can fold the instruction into the PHI
    // (which is only possible if all operands to the PHI are constants).
    Instruction *FoldOpIntoPhi(Instruction &I);

    // FoldPHIArgOpIntoPHI - If all operands to a PHI node are the same "unary"
    // operator and they all are only used by the PHI, PHI together their
    // inputs, and do the operation once, to the result of the PHI.
    Instruction *FoldPHIArgOpIntoPHI(PHINode &PN);
    Instruction *FoldPHIArgBinOpIntoPHI(PHINode &PN);
    
    
    Instruction *OptAndOp(Instruction *Op, ConstantIntegral *OpRHS,
                          ConstantIntegral *AndRHS, BinaryOperator &TheAnd);
    
    Value *FoldLogicalPlusAnd(Value *LHS, Value *RHS, ConstantIntegral *Mask,
                              bool isSub, Instruction &I);
    Instruction *InsertRangeTest(Value *V, Constant *Lo, Constant *Hi,
                                 bool Inside, Instruction &IB);
    Instruction *PromoteCastOfAllocation(CastInst &CI, AllocationInst &AI);
    Instruction *MatchBSwap(BinaryOperator &I);

    Value *EvaluateInDifferentType(Value *V, const Type *Ty);
  };

  RegisterPass<InstCombiner> X("instcombine", "Combine redundant instructions");
}

// getComplexity:  Assign a complexity or rank value to LLVM Values...
//   0 -> undef, 1 -> Const, 2 -> Other, 3 -> Arg, 3 -> Unary, 4 -> OtherInst
static unsigned getComplexity(Value *V) {
  if (isa<Instruction>(V)) {
    if (BinaryOperator::isNeg(V) || BinaryOperator::isNot(V))
      return 3;
    return 4;
  }
  if (isa<Argument>(V)) return 3;
  return isa<Constant>(V) ? (isa<UndefValue>(V) ? 0 : 1) : 2;
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

/// isCast - If the specified operand is a CastInst or a constant expr cast,
/// return the operand value, otherwise return null.
static Value *isCast(Value *V) {
  if (CastInst *I = dyn_cast<CastInst>(V))
    return I->getOperand(0);
  else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    if (CE->getOpcode() == Instruction::Cast)
      return CE->getOperand(0);
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
  unsigned SrcSize = Src->getPrimitiveSizeInBits();
  unsigned DestSize = Dest->getPrimitiveSizeInBits();
  
  if (SrcSize == DestSize) return Noop;
  if (SrcSize > DestSize)  return Truncate;
  if (Src->isSigned()) return Signext;
  return Zeroext;
}


// isEliminableCastOfCast - Return true if it is valid to eliminate the CI
// instruction.
//
static bool isEliminableCastOfCast(const Type *SrcTy, const Type *MidTy,
                                   const Type *DstTy, TargetData *TD) {
  
  // It is legal to eliminate the instruction if casting A->B->A if the sizes
  // are identical and the bits don't get reinterpreted (for example
  // int->float->int would not be allowed).
  if (SrcTy == DstTy && SrcTy->isLosslesslyConvertibleTo(MidTy))
    return true;
  
  // If we are casting between pointer and integer types, treat pointers as
  // integers of the appropriate size for the code below.
  if (isa<PointerType>(SrcTy)) SrcTy = TD->getIntPtrType()->getSignedVersion();
  if (isa<PointerType>(MidTy)) MidTy = TD->getIntPtrType()->getSignedVersion();
  if (isa<PointerType>(DstTy)) DstTy = TD->getIntPtrType()->getSignedVersion();
  
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
  
  // If this is a cast from 'float -> double -> integer', cast from
  // 'float -> integer' directly, as the value isn't changed by the 
  // float->double conversion.
  if (SrcTy->isFloatingPoint() && MidTy->isFloatingPoint() &&
      DstTy->isIntegral() && 
      SrcTy->getPrimitiveSize() < MidTy->getPrimitiveSize())
    return true;
  
  // Packed type conversions don't modify bits.
  if (isa<PackedType>(SrcTy) && isa<PackedType>(MidTy) &&isa<PackedType>(DstTy))
    return true;
  
  return false;
}

/// ValueRequiresCast - Return true if the cast from "V to Ty" actually results
/// in any code being generated.  It does not require codegen if V is simple
/// enough or if the cast can be folded into other casts.
static bool ValueRequiresCast(const Value *V, const Type *Ty, TargetData *TD) {
  if (V->getType() == Ty || isa<Constant>(V)) return false;
  
  // If this is a noop cast, it isn't real codegen.
  if (V->getType()->isLosslesslyConvertibleTo(Ty))
    return false;

  // If this is another cast that can be eliminated, it isn't codegen either.
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
  
  return InsertCastBefore(V, DestTy, *InsertBefore);
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
    return BinaryOperator::getNegArgument(V);

  // Constants can be considered to be negated values if they can be folded.
  if (ConstantInt *C = dyn_cast<ConstantInt>(V))
    return ConstantExpr::getNeg(C);
  return 0;
}

static inline Value *dyn_castNotVal(Value *V) {
  if (BinaryOperator::isNot(V))
    return BinaryOperator::getNotArgument(V);

  // Constants can be considered to be not'ed values...
  if (ConstantIntegral *C = dyn_cast<ConstantIntegral>(V))
    return ConstantExpr::getNot(C);
  return 0;
}

// dyn_castFoldableMul - If this value is a multiply that can be folded into
// other computations (because it has a constant operand), return the
// non-constant operand of the multiply, and set CST to point to the multiplier.
// Otherwise, return null.
//
static inline Value *dyn_castFoldableMul(Value *V, ConstantInt *&CST) {
  if (V->hasOneUse() && V->getType()->isInteger())
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      if (I->getOpcode() == Instruction::Mul)
        if ((CST = dyn_cast<ConstantInt>(I->getOperand(1))))
          return I->getOperand(0);
      if (I->getOpcode() == Instruction::Shl)
        if ((CST = dyn_cast<ConstantInt>(I->getOperand(1)))) {
          // The multiplier is really 1 << CST.
          Constant *One = ConstantInt::get(V->getType(), 1);
          CST = cast<ConstantInt>(ConstantExpr::getShl(One, CST));
          return I->getOperand(0);
        }
    }
  return 0;
}

/// dyn_castGetElementPtr - If this is a getelementptr instruction or constant
/// expression, return it.
static User *dyn_castGetElementPtr(Value *V) {
  if (isa<GetElementPtrInst>(V)) return cast<User>(V);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    if (CE->getOpcode() == Instruction::GetElementPtr)
      return cast<User>(V);
  return false;
}

// AddOne, SubOne - Add or subtract a constant one from an integer constant...
static ConstantInt *AddOne(ConstantInt *C) {
  return cast<ConstantInt>(ConstantExpr::getAdd(C,
                                         ConstantInt::get(C->getType(), 1)));
}
static ConstantInt *SubOne(ConstantInt *C) {
  return cast<ConstantInt>(ConstantExpr::getSub(C,
                                         ConstantInt::get(C->getType(), 1)));
}

/// GetConstantInType - Return a ConstantInt with the specified type and value.
///
static ConstantIntegral *GetConstantInType(const Type *Ty, uint64_t Val) {
  if (Ty->isUnsigned()) 
    return ConstantInt::get(Ty, Val);
  else if (Ty->getTypeID() == Type::BoolTyID)
    return ConstantBool::get(Val);
  int64_t SVal = Val;
  SVal <<= 64-Ty->getPrimitiveSizeInBits();
  SVal >>= 64-Ty->getPrimitiveSizeInBits();
  return ConstantInt::get(Ty, SVal);
}


/// ComputeMaskedBits - Determine which of the bits specified in Mask are
/// known to be either zero or one and return them in the KnownZero/KnownOne
/// bitsets.  This code only analyzes bits in Mask, in order to short-circuit
/// processing.
static void ComputeMaskedBits(Value *V, uint64_t Mask, uint64_t &KnownZero,
                              uint64_t &KnownOne, unsigned Depth = 0) {
  // Note, we cannot consider 'undef' to be "IsZero" here.  The problem is that
  // we cannot optimize based on the assumption that it is zero without changing
  // it to be an explicit zero.  If we don't change it to zero, other code could
  // optimized based on the contradictory assumption that it is non-zero.
  // Because instcombine aggressively folds operations with undef args anyway,
  // this won't lose us code quality.
  if (ConstantIntegral *CI = dyn_cast<ConstantIntegral>(V)) {
    // We know all of the bits for a constant!
    KnownOne = CI->getZExtValue() & Mask;
    KnownZero = ~KnownOne & Mask;
    return;
  }

  KnownZero = KnownOne = 0;   // Don't know anything.
  if (Depth == 6 || Mask == 0)
    return;  // Limit search depth.

  uint64_t KnownZero2, KnownOne2;
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return;

  Mask &= V->getType()->getIntegralTypeMask();
  
  switch (I->getOpcode()) {
  case Instruction::And:
    // If either the LHS or the RHS are Zero, the result is zero.
    ComputeMaskedBits(I->getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    Mask &= ~KnownZero;
    ComputeMaskedBits(I->getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    return;
  case Instruction::Or:
    ComputeMaskedBits(I->getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    Mask &= ~KnownOne;
    ComputeMaskedBits(I->getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    return;
  case Instruction::Xor: {
    ComputeMaskedBits(I->getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(I->getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    uint64_t KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOne = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    KnownZero = KnownZeroOut;
    return;
  }
  case Instruction::Select:
    ComputeMaskedBits(I->getOperand(2), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(I->getOperand(1), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    return;
  case Instruction::Cast: {
    const Type *SrcTy = I->getOperand(0)->getType();
    if (!SrcTy->isIntegral()) return;
    
    // If this is an integer truncate or noop, just look in the input.
    if (SrcTy->getPrimitiveSizeInBits() >= 
           I->getType()->getPrimitiveSizeInBits()) {
      ComputeMaskedBits(I->getOperand(0), Mask, KnownZero, KnownOne, Depth+1);
      return;
    }

    // Sign or Zero extension.  Compute the bits in the result that are not
    // present in the input.
    uint64_t NotIn = ~SrcTy->getIntegralTypeMask();
    uint64_t NewBits = I->getType()->getIntegralTypeMask() & NotIn;
      
    // Handle zero extension.
    if (!SrcTy->isSigned()) {
      Mask &= SrcTy->getIntegralTypeMask();
      ComputeMaskedBits(I->getOperand(0), Mask, KnownZero, KnownOne, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      // The top bits are known to be zero.
      KnownZero |= NewBits;
    } else {
      // Sign extension.
      Mask &= SrcTy->getIntegralTypeMask();
      ComputeMaskedBits(I->getOperand(0), Mask, KnownZero, KnownOne, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 

      // If the sign bit of the input is known set or clear, then we know the
      // top bits of the result.
      uint64_t InSignBit = 1ULL << (SrcTy->getPrimitiveSizeInBits()-1);
      if (KnownZero & InSignBit) {          // Input sign bit known zero
        KnownZero |= NewBits;
        KnownOne &= ~NewBits;
      } else if (KnownOne & InSignBit) {    // Input sign bit known set
        KnownOne |= NewBits;
        KnownZero &= ~NewBits;
      } else {                              // Input sign bit unknown
        KnownZero &= ~NewBits;
        KnownOne &= ~NewBits;
      }
    }
    return;
  }
  case Instruction::Shl:
    // (shl X, C1) & C2 == 0   iff   (X & C2 >>u C1) == 0
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      uint64_t ShiftAmt = SA->getZExtValue();
      Mask >>= ShiftAmt;
      ComputeMaskedBits(I->getOperand(0), Mask, KnownZero, KnownOne, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero <<= ShiftAmt;
      KnownOne  <<= ShiftAmt;
      KnownZero |= (1ULL << ShiftAmt)-1;  // low bits known zero.
      return;
    }
    break;
  case Instruction::Shr:
    // (ushr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      // Compute the new bits that are at the top now.
      uint64_t ShiftAmt = SA->getZExtValue();
      uint64_t HighBits = (1ULL << ShiftAmt)-1;
      HighBits <<= I->getType()->getPrimitiveSizeInBits()-ShiftAmt;
      
      if (I->getType()->isUnsigned()) {   // Unsigned shift right.
        Mask <<= ShiftAmt;
        ComputeMaskedBits(I->getOperand(0), Mask, KnownZero,KnownOne,Depth+1);
        assert((KnownZero & KnownOne) == 0&&"Bits known to be one AND zero?"); 
        KnownZero >>= ShiftAmt;
        KnownOne  >>= ShiftAmt;
        KnownZero |= HighBits;  // high bits known zero.
      } else {
        Mask <<= ShiftAmt;
        ComputeMaskedBits(I->getOperand(0), Mask, KnownZero,KnownOne,Depth+1);
        assert((KnownZero & KnownOne) == 0&&"Bits known to be one AND zero?"); 
        KnownZero >>= ShiftAmt;
        KnownOne  >>= ShiftAmt;
        
        // Handle the sign bits.
        uint64_t SignBit = 1ULL << (I->getType()->getPrimitiveSizeInBits()-1);
        SignBit >>= ShiftAmt;  // Adjust to where it is now in the mask.
        
        if (KnownZero & SignBit) {       // New bits are known zero.
          KnownZero |= HighBits;
        } else if (KnownOne & SignBit) { // New bits are known one.
          KnownOne |= HighBits;
        }
      }
      return;
    }
    break;
  }
}

/// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  Mask is known to be zero
/// for bits that V cannot have.
static bool MaskedValueIsZero(Value *V, uint64_t Mask, unsigned Depth = 0) {
  uint64_t KnownZero, KnownOne;
  ComputeMaskedBits(V, Mask, KnownZero, KnownOne, Depth);
  assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
  return (KnownZero & Mask) == Mask;
}

/// ShrinkDemandedConstant - Check to see if the specified operand of the 
/// specified instruction is a constant integer.  If so, check to see if there
/// are any bits set in the constant that are not demanded.  If so, shrink the
/// constant and return true.
static bool ShrinkDemandedConstant(Instruction *I, unsigned OpNo, 
                                   uint64_t Demanded) {
  ConstantInt *OpC = dyn_cast<ConstantInt>(I->getOperand(OpNo));
  if (!OpC) return false;

  // If there are no bits set that aren't demanded, nothing to do.
  if ((~Demanded & OpC->getZExtValue()) == 0)
    return false;

  // This is producing any bits that are not needed, shrink the RHS.
  uint64_t Val = Demanded & OpC->getZExtValue();
  I->setOperand(OpNo, GetConstantInType(OpC->getType(), Val));
  return true;
}

// ComputeSignedMinMaxValuesFromKnownBits - Given a signed integer type and a 
// set of known zero and one bits, compute the maximum and minimum values that
// could have the specified known zero and known one bits, returning them in
// min/max.
static void ComputeSignedMinMaxValuesFromKnownBits(const Type *Ty,
                                                   uint64_t KnownZero,
                                                   uint64_t KnownOne,
                                                   int64_t &Min, int64_t &Max) {
  uint64_t TypeBits = Ty->getIntegralTypeMask();
  uint64_t UnknownBits = ~(KnownZero|KnownOne) & TypeBits;

  uint64_t SignBit = 1ULL << (Ty->getPrimitiveSizeInBits()-1);
  
  // The minimum value is when all unknown bits are zeros, EXCEPT for the sign
  // bit if it is unknown.
  Min = KnownOne;
  Max = KnownOne|UnknownBits;
  
  if (SignBit & UnknownBits) { // Sign bit is unknown
    Min |= SignBit;
    Max &= ~SignBit;
  }
  
  // Sign extend the min/max values.
  int ShAmt = 64-Ty->getPrimitiveSizeInBits();
  Min = (Min << ShAmt) >> ShAmt;
  Max = (Max << ShAmt) >> ShAmt;
}

// ComputeUnsignedMinMaxValuesFromKnownBits - Given an unsigned integer type and
// a set of known zero and one bits, compute the maximum and minimum values that
// could have the specified known zero and known one bits, returning them in
// min/max.
static void ComputeUnsignedMinMaxValuesFromKnownBits(const Type *Ty,
                                                     uint64_t KnownZero,
                                                     uint64_t KnownOne,
                                                     uint64_t &Min,
                                                     uint64_t &Max) {
  uint64_t TypeBits = Ty->getIntegralTypeMask();
  uint64_t UnknownBits = ~(KnownZero|KnownOne) & TypeBits;
  
  // The minimum value is when the unknown bits are all zeros.
  Min = KnownOne;
  // The maximum value is when the unknown bits are all ones.
  Max = KnownOne|UnknownBits;
}


/// SimplifyDemandedBits - Look at V.  At this point, we know that only the
/// DemandedMask bits of the result of V are ever used downstream.  If we can
/// use this information to simplify V, do so and return true.  Otherwise,
/// analyze the expression and return a mask of KnownOne and KnownZero bits for
/// the expression (used to simplify the caller).  The KnownZero/One bits may
/// only be accurate for those bits in the DemandedMask.
bool InstCombiner::SimplifyDemandedBits(Value *V, uint64_t DemandedMask,
                                        uint64_t &KnownZero, uint64_t &KnownOne,
                                        unsigned Depth) {
  if (ConstantIntegral *CI = dyn_cast<ConstantIntegral>(V)) {
    // We know all of the bits for a constant!
    KnownOne = CI->getZExtValue() & DemandedMask;
    KnownZero = ~KnownOne & DemandedMask;
    return false;
  }
  
  KnownZero = KnownOne = 0;
  if (!V->hasOneUse()) {    // Other users may use these bits.
    if (Depth != 0) {       // Not at the root.
      // Just compute the KnownZero/KnownOne bits to simplify things downstream.
      ComputeMaskedBits(V, DemandedMask, KnownZero, KnownOne, Depth);
      return false;
    }
    // If this is the root being simplified, allow it to have multiple uses,
    // just set the DemandedMask to all bits.
    DemandedMask = V->getType()->getIntegralTypeMask();
  } else if (DemandedMask == 0) {   // Not demanding any bits from V.
    if (V != UndefValue::get(V->getType()))
      return UpdateValueUsesWith(V, UndefValue::get(V->getType()));
    return false;
  } else if (Depth == 6) {        // Limit search depth.
    return false;
  }
  
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;        // Only analyze instructions.

  DemandedMask &= V->getType()->getIntegralTypeMask();
  
  uint64_t KnownZero2, KnownOne2;
  switch (I->getOpcode()) {
  default: break;
  case Instruction::And:
    // If either the LHS or the RHS are Zero, the result is zero.
    if (SimplifyDemandedBits(I->getOperand(1), DemandedMask,
                             KnownZero, KnownOne, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 

    // If something is known zero on the RHS, the bits aren't demanded on the
    // LHS.
    if (SimplifyDemandedBits(I->getOperand(0), DemandedMask & ~KnownZero,
                             KnownZero2, KnownOne2, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 

    // If all of the demanded bits are known one on one side, return the other.
    // These bits cannot contribute to the result of the 'and'.
    if ((DemandedMask & ~KnownZero2 & KnownOne) == (DemandedMask & ~KnownZero2))
      return UpdateValueUsesWith(I, I->getOperand(0));
    if ((DemandedMask & ~KnownZero & KnownOne2) == (DemandedMask & ~KnownZero))
      return UpdateValueUsesWith(I, I->getOperand(1));
    
    // If all of the demanded bits in the inputs are known zeros, return zero.
    if ((DemandedMask & (KnownZero|KnownZero2)) == DemandedMask)
      return UpdateValueUsesWith(I, Constant::getNullValue(I->getType()));
      
    // If the RHS is a constant, see if we can simplify it.
    if (ShrinkDemandedConstant(I, 1, DemandedMask & ~KnownZero2))
      return UpdateValueUsesWith(I, I);
      
    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    break;
  case Instruction::Or:
    if (SimplifyDemandedBits(I->getOperand(1), DemandedMask, 
                             KnownZero, KnownOne, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    if (SimplifyDemandedBits(I->getOperand(0), DemandedMask & ~KnownOne, 
                             KnownZero2, KnownOne2, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'or'.
    if ((DemandedMask & ~KnownOne2 & KnownZero) == (DemandedMask & ~KnownOne2))
      return UpdateValueUsesWith(I, I->getOperand(0));
    if ((DemandedMask & ~KnownOne & KnownZero2) == (DemandedMask & ~KnownOne))
      return UpdateValueUsesWith(I, I->getOperand(1));

    // If all of the potentially set bits on one side are known to be set on
    // the other side, just use the 'other' side.
    if ((DemandedMask & (~KnownZero) & KnownOne2) == 
        (DemandedMask & (~KnownZero)))
      return UpdateValueUsesWith(I, I->getOperand(0));
    if ((DemandedMask & (~KnownZero2) & KnownOne) == 
        (DemandedMask & (~KnownZero2)))
      return UpdateValueUsesWith(I, I->getOperand(1));
        
    // If the RHS is a constant, see if we can simplify it.
    if (ShrinkDemandedConstant(I, 1, DemandedMask))
      return UpdateValueUsesWith(I, I);
          
    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    break;
  case Instruction::Xor: {
    if (SimplifyDemandedBits(I->getOperand(1), DemandedMask,
                             KnownZero, KnownOne, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    if (SimplifyDemandedBits(I->getOperand(0), DemandedMask, 
                             KnownZero2, KnownOne2, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'xor'.
    if ((DemandedMask & KnownZero) == DemandedMask)
      return UpdateValueUsesWith(I, I->getOperand(0));
    if ((DemandedMask & KnownZero2) == DemandedMask)
      return UpdateValueUsesWith(I, I->getOperand(1));
    
    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    uint64_t KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    uint64_t KnownOneOut = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    
    // If all of the unknown bits are known to be zero on one side or the other
    // (but not both) turn this into an *inclusive* or.
    //    e.g. (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
    if (uint64_t UnknownBits = DemandedMask & ~(KnownZeroOut|KnownOneOut)) {
      if ((UnknownBits & (KnownZero|KnownZero2)) == UnknownBits) {
        Instruction *Or =
          BinaryOperator::createOr(I->getOperand(0), I->getOperand(1),
                                   I->getName());
        InsertNewInstBefore(Or, *I);
        return UpdateValueUsesWith(I, Or);
      }
    }
    
    // If all of the demanded bits on one side are known, and all of the set
    // bits on that side are also known to be set on the other side, turn this
    // into an AND, as we know the bits will be cleared.
    //    e.g. (X | C1) ^ C2 --> (X | C1) & ~C2 iff (C1&C2) == C2
    if ((DemandedMask & (KnownZero|KnownOne)) == DemandedMask) { // all known
      if ((KnownOne & KnownOne2) == KnownOne) {
        Constant *AndC = GetConstantInType(I->getType(), 
                                           ~KnownOne & DemandedMask);
        Instruction *And = 
          BinaryOperator::createAnd(I->getOperand(0), AndC, "tmp");
        InsertNewInstBefore(And, *I);
        return UpdateValueUsesWith(I, And);
      }
    }
    
    // If the RHS is a constant, see if we can simplify it.
    // FIXME: for XOR, we prefer to force bits to 1 if they will make a -1.
    if (ShrinkDemandedConstant(I, 1, DemandedMask))
      return UpdateValueUsesWith(I, I);
    
    KnownZero = KnownZeroOut;
    KnownOne  = KnownOneOut;
    break;
  }
  case Instruction::Select:
    if (SimplifyDemandedBits(I->getOperand(2), DemandedMask,
                             KnownZero, KnownOne, Depth+1))
      return true;
    if (SimplifyDemandedBits(I->getOperand(1), DemandedMask, 
                             KnownZero2, KnownOne2, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If the operands are constants, see if we can simplify them.
    if (ShrinkDemandedConstant(I, 1, DemandedMask))
      return UpdateValueUsesWith(I, I);
    if (ShrinkDemandedConstant(I, 2, DemandedMask))
      return UpdateValueUsesWith(I, I);
    
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case Instruction::Cast: {
    const Type *SrcTy = I->getOperand(0)->getType();
    if (!SrcTy->isIntegral()) return false;
    
    // If this is an integer truncate or noop, just look in the input.
    if (SrcTy->getPrimitiveSizeInBits() >= 
        I->getType()->getPrimitiveSizeInBits()) {
      // Cast to bool is a comparison against 0, which demands all bits.  We
      // can't propagate anything useful up.
      if (I->getType() == Type::BoolTy)
        break;
      
      if (SimplifyDemandedBits(I->getOperand(0), DemandedMask,
                               KnownZero, KnownOne, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      break;
    }
    
    // Sign or Zero extension.  Compute the bits in the result that are not
    // present in the input.
    uint64_t NotIn = ~SrcTy->getIntegralTypeMask();
    uint64_t NewBits = I->getType()->getIntegralTypeMask() & NotIn;
    
    // Handle zero extension.
    if (!SrcTy->isSigned()) {
      DemandedMask &= SrcTy->getIntegralTypeMask();
      if (SimplifyDemandedBits(I->getOperand(0), DemandedMask,
                               KnownZero, KnownOne, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      // The top bits are known to be zero.
      KnownZero |= NewBits;
    } else {
      // Sign extension.
      uint64_t InSignBit = 1ULL << (SrcTy->getPrimitiveSizeInBits()-1);
      int64_t InputDemandedBits = DemandedMask & SrcTy->getIntegralTypeMask();

      // If any of the sign extended bits are demanded, we know that the sign
      // bit is demanded.
      if (NewBits & DemandedMask)
        InputDemandedBits |= InSignBit;
      
      if (SimplifyDemandedBits(I->getOperand(0), InputDemandedBits,
                               KnownZero, KnownOne, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      
      // If the sign bit of the input is known set or clear, then we know the
      // top bits of the result.

      // If the input sign bit is known zero, or if the NewBits are not demanded
      // convert this into a zero extension.
      if ((KnownZero & InSignBit) || (NewBits & ~DemandedMask) == NewBits) {
        // Convert to unsigned first.
        Value *NewVal = 
          InsertCastBefore(I->getOperand(0), SrcTy->getUnsignedVersion(), *I);
        // Then cast that to the destination type.
        NewVal = new CastInst(NewVal, I->getType(), I->getName());
        InsertNewInstBefore(cast<Instruction>(NewVal), *I);
        return UpdateValueUsesWith(I, NewVal);
      } else if (KnownOne & InSignBit) {    // Input sign bit known set
        KnownOne |= NewBits;
        KnownZero &= ~NewBits;
      } else {                              // Input sign bit unknown
        KnownZero &= ~NewBits;
        KnownOne &= ~NewBits;
      }
    }
    break;
  }
  case Instruction::Shl:
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      uint64_t ShiftAmt = SA->getZExtValue();
      if (SimplifyDemandedBits(I->getOperand(0), DemandedMask >> ShiftAmt, 
                               KnownZero, KnownOne, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero <<= ShiftAmt;
      KnownOne  <<= ShiftAmt;
      KnownZero |= (1ULL << ShiftAmt) - 1;  // low bits known zero.
    }
    break;
  case Instruction::Shr:
    // If this is an arithmetic shift right and only the low-bit is set, we can
    // always convert this into a logical shr, even if the shift amount is
    // variable.  The low bit of the shift cannot be an input sign bit unless
    // the shift amount is >= the size of the datatype, which is undefined.
    if (DemandedMask == 1 && I->getType()->isSigned()) {
      // Convert the input to unsigned.
      Value *NewVal = InsertCastBefore(I->getOperand(0), 
                                       I->getType()->getUnsignedVersion(), *I);
      // Perform the unsigned shift right.
      NewVal = new ShiftInst(Instruction::Shr, NewVal, I->getOperand(1),
                             I->getName());
      InsertNewInstBefore(cast<Instruction>(NewVal), *I);
      // Then cast that to the destination type.
      NewVal = new CastInst(NewVal, I->getType(), I->getName());
      InsertNewInstBefore(cast<Instruction>(NewVal), *I);
      return UpdateValueUsesWith(I, NewVal);
    }    
    
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      unsigned ShiftAmt = SA->getZExtValue();
      
      // Compute the new bits that are at the top now.
      uint64_t HighBits = (1ULL << ShiftAmt)-1;
      HighBits <<= I->getType()->getPrimitiveSizeInBits() - ShiftAmt;
      uint64_t TypeMask = I->getType()->getIntegralTypeMask();
      if (I->getType()->isUnsigned()) {   // Unsigned shift right.
        if (SimplifyDemandedBits(I->getOperand(0),
                                 (DemandedMask << ShiftAmt) & TypeMask,
                                 KnownZero, KnownOne, Depth+1))
          return true;
        assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
        KnownZero &= TypeMask;
        KnownOne  &= TypeMask;
        KnownZero >>= ShiftAmt;
        KnownOne  >>= ShiftAmt;
        KnownZero |= HighBits;  // high bits known zero.
      } else {                            // Signed shift right.
        if (SimplifyDemandedBits(I->getOperand(0),
                                 (DemandedMask << ShiftAmt) & TypeMask,
                                 KnownZero, KnownOne, Depth+1))
          return true;
        assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
        KnownZero &= TypeMask;
        KnownOne  &= TypeMask;
        KnownZero >>= ShiftAmt;
        KnownOne  >>= ShiftAmt;
        
        // Handle the sign bits.
        uint64_t SignBit = 1ULL << (I->getType()->getPrimitiveSizeInBits()-1);
        SignBit >>= ShiftAmt;  // Adjust to where it is now in the mask.
        
        // If the input sign bit is known to be zero, or if none of the top bits
        // are demanded, turn this into an unsigned shift right.
        if ((KnownZero & SignBit) || (HighBits & ~DemandedMask) == HighBits) {
          // Convert the input to unsigned.
          Value *NewVal = InsertCastBefore(I->getOperand(0), 
                             I->getType()->getUnsignedVersion(), *I);
          // Perform the unsigned shift right.
          NewVal = new ShiftInst(Instruction::Shr, NewVal, SA, I->getName());
          InsertNewInstBefore(cast<Instruction>(NewVal), *I);
          // Then cast that to the destination type.
          NewVal = new CastInst(NewVal, I->getType(), I->getName());
          InsertNewInstBefore(cast<Instruction>(NewVal), *I);
          return UpdateValueUsesWith(I, NewVal);
        } else if (KnownOne & SignBit) { // New bits are known one.
          KnownOne |= HighBits;
        }
      }
    }
    break;
  }
  
  // If the client is only demanding bits that we know, return the known
  // constant.
  if ((DemandedMask & (KnownZero|KnownOne)) == DemandedMask)
    return UpdateValueUsesWith(I, GetConstantInType(I->getType(), KnownOne));
  return false;
}  


/// SimplifyDemandedVectorElts - The specified value producecs a vector with
/// 64 or fewer elements.  DemandedElts contains the set of elements that are
/// actually used by the caller.  This method analyzes which elements of the
/// operand are undef and returns that information in UndefElts.
///
/// If the information about demanded elements can be used to simplify the
/// operation, the operation is simplified, then the resultant value is
/// returned.  This returns null if no change was made.
Value *InstCombiner::SimplifyDemandedVectorElts(Value *V, uint64_t DemandedElts,
                                                uint64_t &UndefElts,
                                                unsigned Depth) {
  unsigned VWidth = cast<PackedType>(V->getType())->getNumElements();
  assert(VWidth <= 64 && "Vector too wide to analyze!");
  uint64_t EltMask = ~0ULL >> (64-VWidth);
  assert(DemandedElts != EltMask && (DemandedElts & ~EltMask) == 0 &&
         "Invalid DemandedElts!");

  if (isa<UndefValue>(V)) {
    // If the entire vector is undefined, just return this info.
    UndefElts = EltMask;
    return 0;
  } else if (DemandedElts == 0) { // If nothing is demanded, provide undef.
    UndefElts = EltMask;
    return UndefValue::get(V->getType());
  }
  
  UndefElts = 0;
  if (ConstantPacked *CP = dyn_cast<ConstantPacked>(V)) {
    const Type *EltTy = cast<PackedType>(V->getType())->getElementType();
    Constant *Undef = UndefValue::get(EltTy);

    std::vector<Constant*> Elts;
    for (unsigned i = 0; i != VWidth; ++i)
      if (!(DemandedElts & (1ULL << i))) {   // If not demanded, set to undef.
        Elts.push_back(Undef);
        UndefElts |= (1ULL << i);
      } else if (isa<UndefValue>(CP->getOperand(i))) {   // Already undef.
        Elts.push_back(Undef);
        UndefElts |= (1ULL << i);
      } else {                               // Otherwise, defined.
        Elts.push_back(CP->getOperand(i));
      }
        
    // If we changed the constant, return it.
    Constant *NewCP = ConstantPacked::get(Elts);
    return NewCP != CP ? NewCP : 0;
  } else if (isa<ConstantAggregateZero>(V)) {
    // Simplify the CAZ to a ConstantPacked where the non-demanded elements are
    // set to undef.
    const Type *EltTy = cast<PackedType>(V->getType())->getElementType();
    Constant *Zero = Constant::getNullValue(EltTy);
    Constant *Undef = UndefValue::get(EltTy);
    std::vector<Constant*> Elts;
    for (unsigned i = 0; i != VWidth; ++i)
      Elts.push_back((DemandedElts & (1ULL << i)) ? Zero : Undef);
    UndefElts = DemandedElts ^ EltMask;
    return ConstantPacked::get(Elts);
  }
  
  if (!V->hasOneUse()) {    // Other users may use these bits.
    if (Depth != 0) {       // Not at the root.
      // TODO: Just compute the UndefElts information recursively.
      return false;
    }
    return false;
  } else if (Depth == 10) {        // Limit search depth.
    return false;
  }
  
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;        // Only analyze instructions.
  
  bool MadeChange = false;
  uint64_t UndefElts2;
  Value *TmpV;
  switch (I->getOpcode()) {
  default: break;
    
  case Instruction::InsertElement: {
    // If this is a variable index, we don't know which element it overwrites.
    // demand exactly the same input as we produce.
    ConstantInt *Idx = dyn_cast<ConstantInt>(I->getOperand(2));
    if (Idx == 0) {
      // Note that we can't propagate undef elt info, because we don't know
      // which elt is getting updated.
      TmpV = SimplifyDemandedVectorElts(I->getOperand(0), DemandedElts,
                                        UndefElts2, Depth+1);
      if (TmpV) { I->setOperand(0, TmpV); MadeChange = true; }
      break;
    }
    
    // If this is inserting an element that isn't demanded, remove this
    // insertelement.
    unsigned IdxNo = Idx->getZExtValue();
    if (IdxNo >= VWidth || (DemandedElts & (1ULL << IdxNo)) == 0)
      return AddSoonDeadInstToWorklist(*I, 0);
    
    // Otherwise, the element inserted overwrites whatever was there, so the
    // input demanded set is simpler than the output set.
    TmpV = SimplifyDemandedVectorElts(I->getOperand(0),
                                      DemandedElts & ~(1ULL << IdxNo),
                                      UndefElts, Depth+1);
    if (TmpV) { I->setOperand(0, TmpV); MadeChange = true; }

    // The inserted element is defined.
    UndefElts |= 1ULL << IdxNo;
    break;
  }
    
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    // div/rem demand all inputs, because they don't want divide by zero.
    TmpV = SimplifyDemandedVectorElts(I->getOperand(0), DemandedElts,
                                      UndefElts, Depth+1);
    if (TmpV) { I->setOperand(0, TmpV); MadeChange = true; }
    TmpV = SimplifyDemandedVectorElts(I->getOperand(1), DemandedElts,
                                      UndefElts2, Depth+1);
    if (TmpV) { I->setOperand(1, TmpV); MadeChange = true; }
      
    // Output elements are undefined if both are undefined.  Consider things
    // like undef&0.  The result is known zero, not undef.
    UndefElts &= UndefElts2;
    break;
    
  case Instruction::Call: {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
    if (!II) break;
    switch (II->getIntrinsicID()) {
    default: break;
      
    // Binary vector operations that work column-wise.  A dest element is a
    // function of the corresponding input elements from the two inputs.
    case Intrinsic::x86_sse_sub_ss:
    case Intrinsic::x86_sse_mul_ss:
    case Intrinsic::x86_sse_min_ss:
    case Intrinsic::x86_sse_max_ss:
    case Intrinsic::x86_sse2_sub_sd:
    case Intrinsic::x86_sse2_mul_sd:
    case Intrinsic::x86_sse2_min_sd:
    case Intrinsic::x86_sse2_max_sd:
      TmpV = SimplifyDemandedVectorElts(II->getOperand(1), DemandedElts,
                                        UndefElts, Depth+1);
      if (TmpV) { II->setOperand(1, TmpV); MadeChange = true; }
      TmpV = SimplifyDemandedVectorElts(II->getOperand(2), DemandedElts,
                                        UndefElts2, Depth+1);
      if (TmpV) { II->setOperand(2, TmpV); MadeChange = true; }

      // If only the low elt is demanded and this is a scalarizable intrinsic,
      // scalarize it now.
      if (DemandedElts == 1) {
        switch (II->getIntrinsicID()) {
        default: break;
        case Intrinsic::x86_sse_sub_ss:
        case Intrinsic::x86_sse_mul_ss:
        case Intrinsic::x86_sse2_sub_sd:
        case Intrinsic::x86_sse2_mul_sd:
          // TODO: Lower MIN/MAX/ABS/etc
          Value *LHS = II->getOperand(1);
          Value *RHS = II->getOperand(2);
          // Extract the element as scalars.
          LHS = InsertNewInstBefore(new ExtractElementInst(LHS, 0U,"tmp"), *II);
          RHS = InsertNewInstBefore(new ExtractElementInst(RHS, 0U,"tmp"), *II);
          
          switch (II->getIntrinsicID()) {
          default: assert(0 && "Case stmts out of sync!");
          case Intrinsic::x86_sse_sub_ss:
          case Intrinsic::x86_sse2_sub_sd:
            TmpV = InsertNewInstBefore(BinaryOperator::createSub(LHS, RHS,
                                                        II->getName()), *II);
            break;
          case Intrinsic::x86_sse_mul_ss:
          case Intrinsic::x86_sse2_mul_sd:
            TmpV = InsertNewInstBefore(BinaryOperator::createMul(LHS, RHS,
                                                         II->getName()), *II);
            break;
          }
          
          Instruction *New =
            new InsertElementInst(UndefValue::get(II->getType()), TmpV, 0U,
                                  II->getName());
          InsertNewInstBefore(New, *II);
          AddSoonDeadInstToWorklist(*II, 0);
          return New;
        }            
      }
        
      // Output elements are undefined if both are undefined.  Consider things
      // like undef&0.  The result is known zero, not undef.
      UndefElts &= UndefElts2;
      break;
    }
    break;
  }
  }
  return MadeChange ? I : 0;
}

// isTrueWhenEqual - Return true if the specified setcondinst instruction is
// true when both operands are equal...
//
static bool isTrueWhenEqual(Instruction &I) {
  return I.getOpcode() == Instruction::SetEQ ||
         I.getOpcode() == Instruction::SetGE ||
         I.getOpcode() == Instruction::SetLE;
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

static Value *FoldOperationIntoSelectOperand(Instruction &I, Value *SO,
                                             InstCombiner *IC) {
  if (isa<CastInst>(I)) {
    if (Constant *SOC = dyn_cast<Constant>(SO))
      return ConstantExpr::getCast(SOC, I.getType());

    return IC->InsertNewInstBefore(new CastInst(SO, I.getType(),
                                                SO->getName() + ".cast"), I);
  }

  // Figure out if the constant is the left or the right argument.
  bool ConstIsRHS = isa<Constant>(I.getOperand(1));
  Constant *ConstOperand = cast<Constant>(I.getOperand(ConstIsRHS));

  if (Constant *SOC = dyn_cast<Constant>(SO)) {
    if (ConstIsRHS)
      return ConstantExpr::get(I.getOpcode(), SOC, ConstOperand);
    return ConstantExpr::get(I.getOpcode(), ConstOperand, SOC);
  }

  Value *Op0 = SO, *Op1 = ConstOperand;
  if (!ConstIsRHS)
    std::swap(Op0, Op1);
  Instruction *New;
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(&I))
    New = BinaryOperator::create(BO->getOpcode(), Op0, Op1,SO->getName()+".op");
  else if (ShiftInst *SI = dyn_cast<ShiftInst>(&I))
    New = new ShiftInst(SI->getOpcode(), Op0, Op1, SO->getName()+".sh");
  else {
    assert(0 && "Unknown binary instruction type!");
    abort();
  }
  return IC->InsertNewInstBefore(New, I);
}

// FoldOpIntoSelect - Given an instruction with a select as one operand and a
// constant as the other operand, try to fold the binary operator into the
// select arguments.  This also works for Cast instructions, which obviously do
// not have a second operand.
static Instruction *FoldOpIntoSelect(Instruction &Op, SelectInst *SI,
                                     InstCombiner *IC) {
  // Don't modify shared select instructions
  if (!SI->hasOneUse()) return 0;
  Value *TV = SI->getOperand(1);
  Value *FV = SI->getOperand(2);

  if (isa<Constant>(TV) || isa<Constant>(FV)) {
    // Bool selects with constant operands can be folded to logical ops.
    if (SI->getType() == Type::BoolTy) return 0;

    Value *SelectTrueVal = FoldOperationIntoSelectOperand(Op, TV, IC);
    Value *SelectFalseVal = FoldOperationIntoSelectOperand(Op, FV, IC);

    return new SelectInst(SI->getCondition(), SelectTrueVal,
                          SelectFalseVal);
  }
  return 0;
}


/// FoldOpIntoPhi - Given a binary operator or cast instruction which has a PHI
/// node as operand #0, see if we can fold the instruction into the PHI (which
/// is only possible if all operands to the PHI are constants).
Instruction *InstCombiner::FoldOpIntoPhi(Instruction &I) {
  PHINode *PN = cast<PHINode>(I.getOperand(0));
  unsigned NumPHIValues = PN->getNumIncomingValues();
  if (!PN->hasOneUse() || NumPHIValues == 0) return 0;

  // Check to see if all of the operands of the PHI are constants.  If there is
  // one non-constant value, remember the BB it is.  If there is more than one
  // bail out.
  BasicBlock *NonConstBB = 0;
  for (unsigned i = 0; i != NumPHIValues; ++i)
    if (!isa<Constant>(PN->getIncomingValue(i))) {
      if (NonConstBB) return 0;  // More than one non-const value.
      NonConstBB = PN->getIncomingBlock(i);
      
      // If the incoming non-constant value is in I's block, we have an infinite
      // loop.
      if (NonConstBB == I.getParent())
        return 0;
    }
  
  // If there is exactly one non-constant value, we can insert a copy of the
  // operation in that block.  However, if this is a critical edge, we would be
  // inserting the computation one some other paths (e.g. inside a loop).  Only
  // do this if the pred block is unconditionally branching into the phi block.
  if (NonConstBB) {
    BranchInst *BI = dyn_cast<BranchInst>(NonConstBB->getTerminator());
    if (!BI || !BI->isUnconditional()) return 0;
  }

  // Okay, we can do the transformation: create the new PHI node.
  PHINode *NewPN = new PHINode(I.getType(), I.getName());
  I.setName("");
  NewPN->reserveOperandSpace(PN->getNumOperands()/2);
  InsertNewInstBefore(NewPN, *PN);

  // Next, add all of the operands to the PHI.
  if (I.getNumOperands() == 2) {
    Constant *C = cast<Constant>(I.getOperand(1));
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i))) {
        InV = ConstantExpr::get(I.getOpcode(), InC, C);
      } else {
        assert(PN->getIncomingBlock(i) == NonConstBB);
        if (BinaryOperator *BO = dyn_cast<BinaryOperator>(&I)) 
          InV = BinaryOperator::create(BO->getOpcode(),
                                       PN->getIncomingValue(i), C, "phitmp",
                                       NonConstBB->getTerminator());
        else if (ShiftInst *SI = dyn_cast<ShiftInst>(&I))
          InV = new ShiftInst(SI->getOpcode(),
                              PN->getIncomingValue(i), C, "phitmp",
                              NonConstBB->getTerminator());
        else
          assert(0 && "Unknown binop!");
        
        WorkList.push_back(cast<Instruction>(InV));
      }
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  } else {
    assert(isa<CastInst>(I) && "Unary op should be a cast!");
    const Type *RetTy = I.getType();
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i))) {
        InV = ConstantExpr::getCast(InC, RetTy);
      } else {
        assert(PN->getIncomingBlock(i) == NonConstBB);
        InV = new CastInst(PN->getIncomingValue(i), I.getType(), "phitmp",
                           NonConstBB->getTerminator());
        WorkList.push_back(cast<Instruction>(InV));
      }
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  }
  return ReplaceInstUsesWith(I, NewPN);
}

Instruction *InstCombiner::visitAdd(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
    // X + undef -> undef
    if (isa<UndefValue>(RHS))
      return ReplaceInstUsesWith(I, RHS);

    // X + 0 --> X
    if (!I.getType()->isFloatingPoint()) { // NOTE: -0 + +0 = +0.
      if (RHSC->isNullValue())
        return ReplaceInstUsesWith(I, LHS);
    } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHSC)) {
      if (CFP->isExactlyValue(-0.0))
        return ReplaceInstUsesWith(I, LHS);
    }

    // X + (signbit) --> X ^ signbit
    if (ConstantInt *CI = dyn_cast<ConstantInt>(RHSC)) {
      uint64_t Val = CI->getZExtValue();
      if (Val == (1ULL << (CI->getType()->getPrimitiveSizeInBits()-1)))
        return BinaryOperator::createXor(LHS, RHS);
    }

    if (isa<PHINode>(LHS))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
    
    ConstantInt *XorRHS = 0;
    Value *XorLHS = 0;
    if (match(LHS, m_Xor(m_Value(XorLHS), m_ConstantInt(XorRHS)))) {
      unsigned TySizeBits = I.getType()->getPrimitiveSizeInBits();
      int64_t  RHSSExt = cast<ConstantInt>(RHSC)->getSExtValue();
      uint64_t RHSZExt = cast<ConstantInt>(RHSC)->getZExtValue();
      
      uint64_t C0080Val = 1ULL << 31;
      int64_t CFF80Val = -C0080Val;
      unsigned Size = 32;
      do {
        if (TySizeBits > Size) {
          bool Found = false;
          // If we have ADD(XOR(AND(X, 0xFF), 0x80), 0xF..F80), it's a sext.
          // If we have ADD(XOR(AND(X, 0xFF), 0xF..F80), 0x80), it's a sext.
          if (RHSSExt == CFF80Val) {
            if (XorRHS->getZExtValue() == C0080Val)
              Found = true;
          } else if (RHSZExt == C0080Val) {
            if (XorRHS->getSExtValue() == CFF80Val)
              Found = true;
          }
          if (Found) {
            // This is a sign extend if the top bits are known zero.
            uint64_t Mask = ~0ULL;
            Mask <<= 64-(TySizeBits-Size);
            Mask &= XorLHS->getType()->getIntegralTypeMask();
            if (!MaskedValueIsZero(XorLHS, Mask))
              Size = 0;  // Not a sign ext, but can't be any others either.
            goto FoundSExt;
          }
        }
        Size >>= 1;
        C0080Val >>= Size;
        CFF80Val >>= Size;
      } while (Size >= 8);
      
FoundSExt:
      const Type *MiddleType = 0;
      switch (Size) {
      default: break;
      case 32: MiddleType = Type::IntTy; break;
      case 16: MiddleType = Type::ShortTy; break;
      case 8:  MiddleType = Type::SByteTy; break;
      }
      if (MiddleType) {
        Instruction *NewTrunc = new CastInst(XorLHS, MiddleType, "sext");
        InsertNewInstBefore(NewTrunc, I);
        return new CastInst(NewTrunc, I.getType());
      }
    }
  }

  // X + X --> X << 1
  if (I.getType()->isInteger()) {
    if (Instruction *Result = AssociativeOpt(I, AddRHS(RHS))) return Result;

    if (Instruction *RHSI = dyn_cast<Instruction>(RHS)) {
      if (RHSI->getOpcode() == Instruction::Sub)
        if (LHS == RHSI->getOperand(1))                   // A + (B - A) --> B
          return ReplaceInstUsesWith(I, RHSI->getOperand(0));
    }
    if (Instruction *LHSI = dyn_cast<Instruction>(LHS)) {
      if (LHSI->getOpcode() == Instruction::Sub)
        if (RHS == LHSI->getOperand(1))                   // (B - A) + A --> B
          return ReplaceInstUsesWith(I, LHSI->getOperand(0));
    }
  }

  // -A + B  -->  B - A
  if (Value *V = dyn_castNegVal(LHS))
    return BinaryOperator::createSub(RHS, V);

  // A + -B  -->  A - B
  if (!isa<Constant>(RHS))
    if (Value *V = dyn_castNegVal(RHS))
      return BinaryOperator::createSub(LHS, V);


  ConstantInt *C2;
  if (Value *X = dyn_castFoldableMul(LHS, C2)) {
    if (X == RHS)   // X*C + X --> X * (C+1)
      return BinaryOperator::createMul(RHS, AddOne(C2));

    // X*C1 + X*C2 --> X * (C1+C2)
    ConstantInt *C1;
    if (X == dyn_castFoldableMul(RHS, C1))
      return BinaryOperator::createMul(X, ConstantExpr::getAdd(C1, C2));
  }

  // X + X*C --> X * (C+1)
  if (dyn_castFoldableMul(RHS, C2) == LHS)
    return BinaryOperator::createMul(LHS, AddOne(C2));


  // (A & C1)+(B & C2) --> (A & C1)|(B & C2) iff C1&C2 == 0
  if (match(RHS, m_And(m_Value(), m_ConstantInt(C2))))
    if (Instruction *R = AssociativeOpt(I, AddMaskingAnd(C2))) return R;

  if (ConstantInt *CRHS = dyn_cast<ConstantInt>(RHS)) {
    Value *X = 0;
    if (match(LHS, m_Not(m_Value(X)))) {   // ~X + C --> (C-1) - X
      Constant *C= ConstantExpr::getSub(CRHS, ConstantInt::get(I.getType(), 1));
      return BinaryOperator::createSub(C, X);
    }

    // (X & FF00) + xx00  -> (X+xx00) & FF00
    if (LHS->hasOneUse() && match(LHS, m_And(m_Value(X), m_ConstantInt(C2)))) {
      Constant *Anded = ConstantExpr::getAnd(CRHS, C2);
      if (Anded == CRHS) {
        // See if all bits from the first bit set in the Add RHS up are included
        // in the mask.  First, get the rightmost bit.
        uint64_t AddRHSV = CRHS->getZExtValue();

        // Form a mask of all bits from the lowest bit added through the top.
        uint64_t AddRHSHighBits = ~((AddRHSV & -AddRHSV)-1);
        AddRHSHighBits &= C2->getType()->getIntegralTypeMask();

        // See if the and mask includes all of these bits.
        uint64_t AddRHSHighBitsAnd = AddRHSHighBits & C2->getZExtValue();

        if (AddRHSHighBits == AddRHSHighBitsAnd) {
          // Okay, the xform is safe.  Insert the new add pronto.
          Value *NewAdd = InsertNewInstBefore(BinaryOperator::createAdd(X, CRHS,
                                                            LHS->getName()), I);
          return BinaryOperator::createAnd(NewAdd, C2);
        }
      }
    }

    // Try to fold constant add into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(LHS))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;
  }

  // add (cast *A to intptrtype) B -> 
  //   cast (GEP (cast *A to sbyte*) B) -> 
  //     intptrtype
  {
    CastInst* CI = dyn_cast<CastInst>(LHS);
    Value* Other = RHS;
    if (!CI) {
      CI = dyn_cast<CastInst>(RHS);
      Other = LHS;
    }
    if (CI && CI->getType()->isSized() && 
        (CI->getType()->getPrimitiveSize() == 
         TD->getIntPtrType()->getPrimitiveSize()) 
        && isa<PointerType>(CI->getOperand(0)->getType())) {
      Value* I2 = InsertCastBefore(CI->getOperand(0),
                                   PointerType::get(Type::SByteTy), I);
      I2 = InsertNewInstBefore(new GetElementPtrInst(I2, Other, "ctg2"), I);
      return new CastInst(I2, CI->getType());
    }
  }

  return Changed ? &I : 0;
}

// isSignBit - Return true if the value represented by the constant only has the
// highest order bit set.
static bool isSignBit(ConstantInt *CI) {
  unsigned NumBits = CI->getType()->getPrimitiveSizeInBits();
  return (CI->getZExtValue() & (~0ULL >> (64-NumBits))) == (1ULL << (NumBits-1));
}

/// RemoveNoopCast - Strip off nonconverting casts from the value.
///
static Value *RemoveNoopCast(Value *V) {
  if (CastInst *CI = dyn_cast<CastInst>(V)) {
    const Type *CTy = CI->getType();
    const Type *OpTy = CI->getOperand(0)->getType();
    if (CTy->isInteger() && OpTy->isInteger()) {
      if (CTy->getPrimitiveSizeInBits() == OpTy->getPrimitiveSizeInBits())
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

  if (isa<UndefValue>(Op0))
    return ReplaceInstUsesWith(I, Op0);    // undef - X -> undef
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);    // X - undef -> undef

  if (ConstantInt *C = dyn_cast<ConstantInt>(Op0)) {
    // Replace (-1 - A) with (~A)...
    if (C->isAllOnesValue())
      return BinaryOperator::createNot(Op1);

    // C - ~X == X + (1+C)
    Value *X = 0;
    if (match(Op1, m_Not(m_Value(X))))
      return BinaryOperator::createAdd(X,
                    ConstantExpr::getAdd(C, ConstantInt::get(I.getType(), 1)));
    // -((uint)X >> 31) -> ((int)X >> 31)
    // -((int)X >> 31) -> ((uint)X >> 31)
    if (C->isNullValue()) {
      Value *NoopCastedRHS = RemoveNoopCast(Op1);
      if (ShiftInst *SI = dyn_cast<ShiftInst>(NoopCastedRHS))
        if (SI->getOpcode() == Instruction::Shr)
          if (ConstantInt *CU = dyn_cast<ConstantInt>(SI->getOperand(1))) {
            const Type *NewTy;
            if (SI->getType()->isSigned())
              NewTy = SI->getType()->getUnsignedVersion();
            else
              NewTy = SI->getType()->getSignedVersion();
            // Check to see if we are shifting out everything but the sign bit.
            if (CU->getZExtValue() == 
                SI->getType()->getPrimitiveSizeInBits()-1) {
              // Ok, the transformation is safe.  Insert a cast of the incoming
              // value, then the new shift, then the new cast.
              Value *InV = InsertCastBefore(SI->getOperand(0), NewTy, I);
              Instruction *NewShift = new ShiftInst(Instruction::Shr, InV,
                                                    CU, SI->getName());
              if (NewShift->getType() == I.getType())
                return NewShift;
              else {
                InsertNewInstBefore(NewShift, I);
                return new CastInst(NewShift, I.getType());
              }
            }
          }
    }

    // Try to fold constant sub into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  if (BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1)) {
    if (Op1I->getOpcode() == Instruction::Add &&
        !Op0->getType()->isFloatingPoint()) {
      if (Op1I->getOperand(0) == Op0)              // X-(X+Y) == -Y
        return BinaryOperator::createNeg(Op1I->getOperand(1), I.getName());
      else if (Op1I->getOperand(1) == Op0)         // X-(Y+X) == -Y
        return BinaryOperator::createNeg(Op1I->getOperand(0), I.getName());
      else if (ConstantInt *CI1 = dyn_cast<ConstantInt>(I.getOperand(0))) {
        if (ConstantInt *CI2 = dyn_cast<ConstantInt>(Op1I->getOperand(1)))
          // C1-(X+C2) --> (C1-C2)-X
          return BinaryOperator::createSub(ConstantExpr::getSub(CI1, CI2),
                                           Op1I->getOperand(0));
      }
    }

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

      // 0 - (X sdiv C)  -> (X sdiv -C)
      if (Op1I->getOpcode() == Instruction::SDiv)
        if (ConstantInt *CSI = dyn_cast<ConstantInt>(Op0))
          if (CSI->isNullValue())
            if (Constant *DivRHS = dyn_cast<Constant>(Op1I->getOperand(1)))
              return BinaryOperator::createSDiv(Op1I->getOperand(0),
                                               ConstantExpr::getNeg(DivRHS));

      // X - X*C --> X * (1-C)
      ConstantInt *C2 = 0;
      if (dyn_castFoldableMul(Op1I, C2) == Op0) {
        Constant *CP1 =
          ConstantExpr::getSub(ConstantInt::get(I.getType(), 1), C2);
        return BinaryOperator::createMul(Op0, CP1);
      }
    }
  }

  if (!Op0->getType()->isFloatingPoint())
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0))
      if (Op0I->getOpcode() == Instruction::Add) {
        if (Op0I->getOperand(0) == Op1)             // (Y+X)-Y == X
          return ReplaceInstUsesWith(I, Op0I->getOperand(1));
        else if (Op0I->getOperand(1) == Op1)        // (X+Y)-Y == X
          return ReplaceInstUsesWith(I, Op0I->getOperand(0));
      } else if (Op0I->getOpcode() == Instruction::Sub) {
        if (Op0I->getOperand(0) == Op1)             // (X-Y)-X == -Y
          return BinaryOperator::createNeg(Op0I->getOperand(1), I.getName());
      }

  ConstantInt *C1;
  if (Value *X = dyn_castFoldableMul(Op0, C1)) {
    if (X == Op1) { // X*C - X --> X * (C-1)
      Constant *CP1 = ConstantExpr::getSub(C1, ConstantInt::get(I.getType(),1));
      return BinaryOperator::createMul(Op1, CP1);
    }

    ConstantInt *C2;   // X*C1 - X*C2 -> X * (C1-C2)
    if (X == dyn_castFoldableMul(Op1, C2))
      return BinaryOperator::createMul(Op1, ConstantExpr::getSub(C1, C2));
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
    ConstantInt *RHSC = cast<ConstantInt>(RHS);
    // True if source is LHS > 127 or LHS >= 128, where the constants depend on
    // the size of the integer type.
    if (Opcode == Instruction::SetGE)
      return RHSC->getZExtValue() ==
        1ULL << (RHS->getType()->getPrimitiveSizeInBits()-1);
    if (Opcode == Instruction::SetGT)
      return RHSC->getZExtValue() ==
        (1ULL << (RHS->getType()->getPrimitiveSizeInBits()-1))-1;
  }
  return false;
}

Instruction *InstCombiner::visitMul(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0);

  if (isa<UndefValue>(I.getOperand(1)))              // undef * X -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

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

      int64_t Val = (int64_t)cast<ConstantInt>(CI)->getZExtValue();
      if (isPowerOf2_64(Val)) {          // Replace X*(2^C) with X << C
        uint64_t C = Log2_64(Val);
        return new ShiftInst(Instruction::Shl, Op0,
                             ConstantInt::get(Type::UByteTy, C));
      }
    } else if (ConstantFP *Op1F = dyn_cast<ConstantFP>(Op1)) {
      if (Op1F->isNullValue())
        return ReplaceInstUsesWith(I, Op1);

      // "In IEEE floating point, x*1 is not equivalent to x for nans.  However,
      // ANSI says we can drop signals, so we can do this anyway." (from GCC)
      if (Op1F->getValue() == 1.0)
        return ReplaceInstUsesWith(I, Op0);  // Eliminate 'mul double %X, 1.0'
    }
    
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0))
      if (Op0I->getOpcode() == Instruction::Add && Op0I->hasOneUse() &&
          isa<ConstantInt>(Op0I->getOperand(1))) {
        // Canonicalize (X+C1)*C2 -> X*C2+C1*C2.
        Instruction *Add = BinaryOperator::createMul(Op0I->getOperand(0),
                                                     Op1, "tmp");
        InsertNewInstBefore(Add, I);
        Value *C1C2 = ConstantExpr::getMul(Op1, 
                                           cast<Constant>(Op0I->getOperand(1)));
        return BinaryOperator::createAdd(Add, C1C2);
        
      }

    // Try to fold constant mul into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
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
        Constant *Amt = ConstantInt::get(Type::UByteTy,
                                          SCOpTy->getPrimitiveSizeInBits()-1);
        if (SCIOp0->getType()->isUnsigned()) {
          const Type *NewTy = SCIOp0->getType()->getSignedVersion();
          SCIOp0 = InsertCastBefore(SCIOp0, NewTy, I);
        }

        Value *V =
          InsertNewInstBefore(new ShiftInst(Instruction::Shr, SCIOp0, Amt,
                                            BoolCast->getOperand(0)->getName()+
                                            ".mask"), I);

        // If the multiply type is not the same as the source type, sign extend
        // or truncate to the multiply type.
        if (I.getType() != V->getType())
          V = InsertCastBefore(V, I.getType(), I);

        Value *OtherOp = Op0 == BoolCast ? I.getOperand(1) : Op0;
        return BinaryOperator::createAnd(V, OtherOp);
      }
    }
  }

  return Changed ? &I : 0;
}

/// This function implements the transforms on div instructions that work
/// regardless of the kind of div instruction it is (udiv, sdiv, or fdiv). It is
/// used by the visitors to those instructions.
/// @brief Transforms common to all three div instructions
Instruction* InstCombiner::commonDivTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // undef / X -> 0
  if (isa<UndefValue>(Op0))
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // X / undef -> undef
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);

  // Handle cases involving: div X, (select Cond, Y, Z)
  if (SelectInst *SI = dyn_cast<SelectInst>(Op1)) {
    // div X, (Cond ? 0 : Y) -> div X, Y.  If the div and the select are in the
    // same basic block, then we replace the select with Y, and the condition 
    // of the select with false (if the cond value is in the same BB).  If the
    // select has uses other than the div, this allows them to be simplified
    // also. Note that div X, Y is just as good as div X, 0 (undef)
    if (Constant *ST = dyn_cast<Constant>(SI->getOperand(1)))
      if (ST->isNullValue()) {
        Instruction *CondI = dyn_cast<Instruction>(SI->getOperand(0));
        if (CondI && CondI->getParent() == I.getParent())
          UpdateValueUsesWith(CondI, ConstantBool::getFalse());
        else if (I.getParent() != SI->getParent() || SI->hasOneUse())
          I.setOperand(1, SI->getOperand(2));
        else
          UpdateValueUsesWith(SI, SI->getOperand(2));
        return &I;
      }

    // Likewise for: div X, (Cond ? Y : 0) -> div X, Y
    if (Constant *ST = dyn_cast<Constant>(SI->getOperand(2)))
      if (ST->isNullValue()) {
        Instruction *CondI = dyn_cast<Instruction>(SI->getOperand(0));
        if (CondI && CondI->getParent() == I.getParent())
          UpdateValueUsesWith(CondI, ConstantBool::getTrue());
        else if (I.getParent() != SI->getParent() || SI->hasOneUse())
          I.setOperand(1, SI->getOperand(1));
        else
          UpdateValueUsesWith(SI, SI->getOperand(1));
        return &I;
      }
  }

  return 0;
}

/// This function implements the transforms common to both integer division
/// instructions (udiv and sdiv). It is called by the visitors to those integer
/// division instructions.
/// @brief Common integer divide transforms
Instruction* InstCombiner::commonIDivTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Instruction *Common = commonDivTransforms(I))
    return Common;

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // div X, 1 == X
    if (RHS->equalsInt(1))
      return ReplaceInstUsesWith(I, Op0);

    // (X / C1) / C2  -> X / (C1*C2)
    if (Instruction *LHS = dyn_cast<Instruction>(Op0))
      if (Instruction::BinaryOps(LHS->getOpcode()) == I.getOpcode())
        if (ConstantInt *LHSRHS = dyn_cast<ConstantInt>(LHS->getOperand(1))) {
          return BinaryOperator::create(I.getOpcode(), LHS->getOperand(0),
                                        ConstantExpr::getMul(RHS, LHSRHS));
        }

    if (!RHS->isNullValue()) { // avoid X udiv 0
      if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
        if (Instruction *R = FoldOpIntoSelect(I, SI, this))
          return R;
      if (isa<PHINode>(Op0))
        if (Instruction *NV = FoldOpIntoPhi(I))
          return NV;
    }
  }

  // 0 / X == 0, we don't need to preserve faults!
  if (ConstantInt *LHS = dyn_cast<ConstantInt>(Op0))
    if (LHS->equalsInt(0))
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  return 0;
}

Instruction *InstCombiner::visitUDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  // X udiv C^2 -> X >> C
  // Check to see if this is an unsigned division with an exact power of 2,
  // if so, convert to a right shift.
  if (ConstantInt *C = dyn_cast<ConstantInt>(Op1)) {
    if (uint64_t Val = C->getZExtValue())    // Don't break X / 0
      if (isPowerOf2_64(Val)) {
        uint64_t ShiftAmt = Log2_64(Val);
        Value* X = Op0;
        const Type* XTy = X->getType();
        bool isSigned = XTy->isSigned();
        if (isSigned)
          X = InsertCastBefore(X, XTy->getUnsignedVersion(), I);
        Instruction* Result = 
          new ShiftInst(Instruction::Shr, X, 
                        ConstantInt::get(Type::UByteTy, ShiftAmt));
        if (!isSigned)
          return Result;
        InsertNewInstBefore(Result, I);
        return new CastInst(Result, XTy->getSignedVersion(), I.getName());
      }
  }

  // X udiv (C1 << N), where C1 is "1<<C2"  -->  X >> (N+C2)
  if (ShiftInst *RHSI = dyn_cast<ShiftInst>(I.getOperand(1))) {
    if (RHSI->getOpcode() == Instruction::Shl &&
        isa<ConstantInt>(RHSI->getOperand(0))) {
      uint64_t C1 = cast<ConstantInt>(RHSI->getOperand(0))->getZExtValue();
      if (isPowerOf2_64(C1)) {
        Value *N = RHSI->getOperand(1);
        const Type* NTy = N->getType();
        bool isSigned = NTy->isSigned();
        if (uint64_t C2 = Log2_64(C1)) {
          if (isSigned) {
            NTy = NTy->getUnsignedVersion();
            N = InsertCastBefore(N, NTy, I);
          }
          Constant *C2V = ConstantInt::get(NTy, C2);
          N = InsertNewInstBefore(BinaryOperator::createAdd(N, C2V, "tmp"), I);
        }
        Instruction* Result = new ShiftInst(Instruction::Shr, Op0, N);
        if (!isSigned)
          return Result;
        InsertNewInstBefore(Result, I);
        return new CastInst(Result, NTy->getSignedVersion(), I.getName());
      }
    }
  }
  
  // udiv X, (Select Cond, C1, C2) --> Select Cond, (shr X, C1), (shr X, C2)
  // where C1&C2 are powers of two.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op1)) {
    if (ConstantInt *STO = dyn_cast<ConstantInt>(SI->getOperand(1)))
      if (ConstantInt *SFO = dyn_cast<ConstantInt>(SI->getOperand(2))) 
        if (!STO->isNullValue() && !STO->isNullValue()) {
          uint64_t TVA = STO->getZExtValue(), FVA = SFO->getZExtValue();
          if (isPowerOf2_64(TVA) && isPowerOf2_64(FVA)) {
            // Compute the shift amounts
            unsigned TSA = Log2_64(TVA), FSA = Log2_64(FVA);
            // Make sure we get the unsigned version of X
            Value* X = Op0;
            const Type* origXTy = X->getType();
            bool isSigned = origXTy->isSigned();
            if (isSigned)
              X = InsertCastBefore(X, X->getType()->getUnsignedVersion(), I);
            // Construct the "on true" case of the select
            Constant *TC = ConstantInt::get(Type::UByteTy, TSA);
            Instruction *TSI = 
              new ShiftInst(Instruction::Shr, X, TC, SI->getName()+".t");
            TSI = InsertNewInstBefore(TSI, I);
    
            // Construct the "on false" case of the select
            Constant *FC = ConstantInt::get(Type::UByteTy, FSA); 
            Instruction *FSI = 
              new ShiftInst(Instruction::Shr, X, FC, SI->getName()+".f");
            FSI = InsertNewInstBefore(FSI, I);

            // construct the select instruction and return it.
            SelectInst* NewSI = 
              new SelectInst(SI->getOperand(0), TSI, FSI, SI->getName());
            if (!isSigned)
              return NewSI;
            InsertNewInstBefore(NewSI, I);
            return new CastInst(NewSI, origXTy, NewSI->getName());
          }
        }
  }
  return 0;
}

Instruction *InstCombiner::visitSDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // sdiv X, -1 == -X
    if (RHS->isAllOnesValue())
      return BinaryOperator::createNeg(Op0);

    // -X/C -> X/-C
    if (Value *LHSNeg = dyn_castNegVal(Op0))
      return BinaryOperator::createSDiv(LHSNeg, ConstantExpr::getNeg(RHS));
  }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a udiv.
  if (I.getType()->isInteger()) {
    uint64_t Mask = 1ULL << (I.getType()->getPrimitiveSizeInBits()-1);
    if (MaskedValueIsZero(Op1, Mask) && MaskedValueIsZero(Op0, Mask)) {
      return BinaryOperator::createUDiv(Op0, Op1, I.getName());
    }
  }      
  
  return 0;
}

Instruction *InstCombiner::visitFDiv(BinaryOperator &I) {
  return commonDivTransforms(I);
}

/// GetFactor - If we can prove that the specified value is at least a multiple
/// of some factor, return that factor.
static Constant *GetFactor(Value *V) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
    return CI;
  
  // Unless we can be tricky, we know this is a multiple of 1.
  Constant *Result = ConstantInt::get(V->getType(), 1);
  
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return Result;
  
  if (I->getOpcode() == Instruction::Mul) {
    // Handle multiplies by a constant, etc.
    return ConstantExpr::getMul(GetFactor(I->getOperand(0)),
                                GetFactor(I->getOperand(1)));
  } else if (I->getOpcode() == Instruction::Shl) {
    // (X<<C) -> X * (1 << C)
    if (Constant *ShRHS = dyn_cast<Constant>(I->getOperand(1))) {
      ShRHS = ConstantExpr::getShl(Result, ShRHS);
      return ConstantExpr::getMul(GetFactor(I->getOperand(0)), ShRHS);
    }
  } else if (I->getOpcode() == Instruction::And) {
    if (ConstantInt *RHS = dyn_cast<ConstantInt>(I->getOperand(1))) {
      // X & 0xFFF0 is known to be a multiple of 16.
      unsigned Zeros = CountTrailingZeros_64(RHS->getZExtValue());
      if (Zeros != V->getType()->getPrimitiveSizeInBits())
        return ConstantExpr::getShl(Result, 
                                    ConstantInt::get(Type::UByteTy, Zeros));
    }
  } else if (I->getOpcode() == Instruction::Cast) {
    Value *Op = I->getOperand(0);
    // Only handle int->int casts.
    if (!Op->getType()->isInteger()) return Result;
    return ConstantExpr::getCast(GetFactor(Op), V->getType());
  }    
  return Result;
}

/// This function implements the transforms on rem instructions that work
/// regardless of the kind of rem instruction it is (urem, srem, or frem). It 
/// is used by the visitors to those instructions.
/// @brief Transforms common to all three rem instructions
Instruction *InstCombiner::commonRemTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // 0 % X == 0, we don't need to preserve faults!
  if (Constant *LHS = dyn_cast<Constant>(Op0))
    if (LHS->isNullValue())
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  if (isa<UndefValue>(Op0))              // undef % X -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);  // X % undef -> undef

  // Handle cases involving: rem X, (select Cond, Y, Z)
  if (SelectInst *SI = dyn_cast<SelectInst>(Op1)) {
    // rem X, (Cond ? 0 : Y) -> rem X, Y.  If the rem and the select are in
    // the same basic block, then we replace the select with Y, and the
    // condition of the select with false (if the cond value is in the same
    // BB).  If the select has uses other than the div, this allows them to be
    // simplified also.
    if (Constant *ST = dyn_cast<Constant>(SI->getOperand(1)))
      if (ST->isNullValue()) {
        Instruction *CondI = dyn_cast<Instruction>(SI->getOperand(0));
        if (CondI && CondI->getParent() == I.getParent())
          UpdateValueUsesWith(CondI, ConstantBool::getFalse());
        else if (I.getParent() != SI->getParent() || SI->hasOneUse())
          I.setOperand(1, SI->getOperand(2));
        else
          UpdateValueUsesWith(SI, SI->getOperand(2));
        return &I;
      }
    // Likewise for: rem X, (Cond ? Y : 0) -> rem X, Y
    if (Constant *ST = dyn_cast<Constant>(SI->getOperand(2)))
      if (ST->isNullValue()) {
        Instruction *CondI = dyn_cast<Instruction>(SI->getOperand(0));
        if (CondI && CondI->getParent() == I.getParent())
          UpdateValueUsesWith(CondI, ConstantBool::getTrue());
        else if (I.getParent() != SI->getParent() || SI->hasOneUse())
          I.setOperand(1, SI->getOperand(1));
        else
          UpdateValueUsesWith(SI, SI->getOperand(1));
        return &I;
      }
  }

  return 0;
}

/// This function implements the transforms common to both integer remainder
/// instructions (urem and srem). It is called by the visitors to those integer
/// remainder instructions.
/// @brief Common integer remainder transforms
Instruction *InstCombiner::commonIRemTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Instruction *common = commonRemTransforms(I))
    return common;

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // X % 0 == undef, we don't need to preserve faults!
    if (RHS->equalsInt(0))
      return ReplaceInstUsesWith(I, UndefValue::get(I.getType()));
    
    if (RHS->equalsInt(1))  // X % 1 == 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

    if (Instruction *Op0I = dyn_cast<Instruction>(Op0)) {
      if (SelectInst *SI = dyn_cast<SelectInst>(Op0I)) {
        if (Instruction *R = FoldOpIntoSelect(I, SI, this))
          return R;
      } else if (isa<PHINode>(Op0I)) {
        if (Instruction *NV = FoldOpIntoPhi(I))
          return NV;
      }
      // (X * C1) % C2 --> 0  iff  C1 % C2 == 0
      if (ConstantExpr::getSRem(GetFactor(Op0I), RHS)->isNullValue())
        return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
    }
  }

  return 0;
}

Instruction *InstCombiner::visitURem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Instruction *common = commonIRemTransforms(I))
    return common;
  
  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // X urem C^2 -> X and C
    // Check to see if this is an unsigned remainder with an exact power of 2,
    // if so, convert to a bitwise and.
    if (ConstantInt *C = dyn_cast<ConstantInt>(RHS))
      if (isPowerOf2_64(C->getZExtValue()))
        return BinaryOperator::createAnd(Op0, SubOne(C));
  }

  if (Instruction *RHSI = dyn_cast<Instruction>(I.getOperand(1))) {
    // Turn A % (C << N), where C is 2^k, into A & ((C << N)-1)  
    if (RHSI->getOpcode() == Instruction::Shl &&
        isa<ConstantInt>(RHSI->getOperand(0))) {
      unsigned C1 = cast<ConstantInt>(RHSI->getOperand(0))->getZExtValue();
      if (isPowerOf2_64(C1)) {
        Constant *N1 = ConstantInt::getAllOnesValue(I.getType());
        Value *Add = InsertNewInstBefore(BinaryOperator::createAdd(RHSI, N1,
                                                                   "tmp"), I);
        return BinaryOperator::createAnd(Op0, Add);
      }
    }
  }

  // urem X, (select Cond, 2^C1, 2^C2) --> select Cond, (and X, C1), (and X, C2)
  // where C1&C2 are powers of two.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op1)) {
    if (ConstantInt *STO = dyn_cast<ConstantInt>(SI->getOperand(1)))
      if (ConstantInt *SFO = dyn_cast<ConstantInt>(SI->getOperand(2))) {
        // STO == 0 and SFO == 0 handled above.
        if (isPowerOf2_64(STO->getZExtValue()) && 
            isPowerOf2_64(SFO->getZExtValue())) {
          Value *TrueAnd = InsertNewInstBefore(
            BinaryOperator::createAnd(Op0, SubOne(STO), SI->getName()+".t"), I);
          Value *FalseAnd = InsertNewInstBefore(
            BinaryOperator::createAnd(Op0, SubOne(SFO), SI->getName()+".f"), I);
          return new SelectInst(SI->getOperand(0), TrueAnd, FalseAnd);
        }
      }
  }
  
  return 0;
}

Instruction *InstCombiner::visitSRem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Instruction *common = commonIRemTransforms(I))
    return common;
  
  if (Value *RHSNeg = dyn_castNegVal(Op1))
    if (!isa<ConstantInt>(RHSNeg) || 
        cast<ConstantInt>(RHSNeg)->getSExtValue() > 0) {
      // X % -Y -> X % Y
      AddUsesToWorkList(I);
      I.setOperand(1, RHSNeg);
      return &I;
    }
 
  // If the top bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a urem.
  uint64_t Mask = 1ULL << (I.getType()->getPrimitiveSizeInBits()-1);
  if (MaskedValueIsZero(Op1, Mask) && MaskedValueIsZero(Op0, Mask)) {
    // X srem Y -> X urem Y, iff X and Y don't have sign bit set
    return BinaryOperator::createURem(Op0, Op1, I.getName());
  }

  return 0;
}

Instruction *InstCombiner::visitFRem(BinaryOperator &I) {
  return commonRemTransforms(I);
}

// isMaxValueMinusOne - return true if this is Max-1
static bool isMaxValueMinusOne(const ConstantInt *C) {
  if (C->getType()->isUnsigned()) 
    return C->getZExtValue() == C->getType()->getIntegralTypeMask()-1;

  // Calculate 0111111111..11111
  unsigned TypeBits = C->getType()->getPrimitiveSizeInBits();
  int64_t Val = INT64_MAX;             // All ones
  Val >>= 64-TypeBits;                 // Shift out unwanted 1 bits...
  return C->getSExtValue() == Val-1;
}

// isMinValuePlusOne - return true if this is Min+1
static bool isMinValuePlusOne(const ConstantInt *C) {
  if (C->getType()->isUnsigned())
    return C->getZExtValue() == 1;

  // Calculate 1111111111000000000000
  unsigned TypeBits = C->getType()->getPrimitiveSizeInBits();
  int64_t Val = -1;                    // All ones
  Val <<= TypeBits-1;                  // Shift over to the right spot
  return C->getSExtValue() == Val+1;
}

// isOneBitSet - Return true if there is exactly one bit set in the specified
// constant.
static bool isOneBitSet(const ConstantInt *CI) {
  uint64_t V = CI->getZExtValue();
  return V && (V & (V-1)) == 0;
}

#if 0   // Currently unused
// isLowOnes - Return true if the constant is of the form 0+1+.
static bool isLowOnes(const ConstantInt *CI) {
  uint64_t V = CI->getZExtValue();

  // There won't be bits set in parts that the type doesn't contain.
  V &= ConstantInt::getAllOnesValue(CI->getType())->getZExtValue();

  uint64_t U = V+1;  // If it is low ones, this should be a power of two.
  return U && V && (U & V) == 0;
}
#endif

// isHighOnes - Return true if the constant is of the form 1+0+.
// This is the same as lowones(~X).
static bool isHighOnes(const ConstantInt *CI) {
  uint64_t V = ~CI->getZExtValue();
  if (~V == 0) return false;  // 0's does not match "1+"

  // There won't be bits set in parts that the type doesn't contain.
  V &= ConstantInt::getAllOnesValue(CI->getType())->getZExtValue();

  uint64_t U = V+1;  // If it is low ones, this should be a power of two.
  return U && V && (U & V) == 0;
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
  case 0: return ConstantBool::getFalse();
  case 1: return new SetCondInst(Instruction::SetGT, LHS, RHS);
  case 2: return new SetCondInst(Instruction::SetEQ, LHS, RHS);
  case 3: return new SetCondInst(Instruction::SetGE, LHS, RHS);
  case 4: return new SetCondInst(Instruction::SetLT, LHS, RHS);
  case 5: return new SetCondInst(Instruction::SetNE, LHS, RHS);
  case 6: return new SetCondInst(Instruction::SetLE, LHS, RHS);
  case 7: return ConstantBool::getTrue();
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
    if (Op->hasOneUse()) {
      // (X ^ C1) & C2 --> (X & C2) ^ (C1&C2)
      std::string OpName = Op->getName(); Op->setName("");
      Instruction *And = BinaryOperator::createAnd(X, AndRHS, OpName);
      InsertNewInstBefore(And, TheAnd);
      return BinaryOperator::createXor(And, Together);
    }
    break;
  case Instruction::Or:
    if (Together == AndRHS) // (X | C) & C --> C
      return ReplaceInstUsesWith(TheAnd, AndRHS);

    if (Op->hasOneUse() && Together != OpRHS) {
      // (X | C1) & C2 --> (X | (C1&C2)) & C2
      std::string Op0Name = Op->getName(); Op->setName("");
      Instruction *Or = BinaryOperator::createOr(X, Together, Op0Name);
      InsertNewInstBefore(Or, TheAnd);
      return BinaryOperator::createAnd(Or, AndRHS);
    }
    break;
  case Instruction::Add:
    if (Op->hasOneUse()) {
      // Adding a one to a single bit bit-field should be turned into an XOR
      // of the bit.  First thing to check is to see if this AND is with a
      // single bit constant.
      uint64_t AndRHSV = cast<ConstantInt>(AndRHS)->getZExtValue();

      // Clear bits that are not part of the constant.
      AndRHSV &= AndRHS->getType()->getIntegralTypeMask();

      // If there is only one bit set...
      if (isOneBitSet(cast<ConstantInt>(AndRHS))) {
        // Ok, at this point, we know that we are masking the result of the
        // ADD down to exactly one bit.  If the constant we are adding has
        // no bits set below this bit, then we can eliminate the ADD.
        uint64_t AddRHS = cast<ConstantInt>(OpRHS)->getZExtValue();

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
    Constant *ShlMask = ConstantExpr::getShl(AllOne, OpRHS);
    Constant *CI = ConstantExpr::getAnd(AndRHS, ShlMask);

    if (CI == ShlMask) {   // Masking out bits that the shift already masks
      return ReplaceInstUsesWith(TheAnd, Op);   // No need for the and.
    } else if (CI != AndRHS) {                  // Reducing bits set in and.
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
      Constant *ShrMask = ConstantExpr::getShr(AllOne, OpRHS);
      Constant *CI = ConstantExpr::getAnd(AndRHS, ShrMask);

      if (CI == ShrMask) {   // Masking out bits that the shift already masks.
        return ReplaceInstUsesWith(TheAnd, Op);
      } else if (CI != AndRHS) {
        TheAnd.setOperand(1, CI);  // Reduce bits set in and cst.
        return &TheAnd;
      }
    } else {   // Signed shr.
      // See if this is shifting in some sign extension, then masking it out
      // with an and.
      if (Op->hasOneUse()) {
        Constant *AllOne = ConstantIntegral::getAllOnesValue(AndRHS->getType());
        Constant *ShrMask = ConstantExpr::getUShr(AllOne, OpRHS);
        Constant *CI = ConstantExpr::getAnd(AndRHS, ShrMask);
        if (CI == AndRHS) {          // Masking out bits shifted in.
          // Make the argument unsigned.
          Value *ShVal = Op->getOperand(0);
          ShVal = InsertCastBefore(ShVal,
                                   ShVal->getType()->getUnsignedVersion(),
                                   TheAnd);
          ShVal = InsertNewInstBefore(new ShiftInst(Instruction::Shr, ShVal,
                                                    OpRHS, Op->getName()),
                                      TheAnd);
          Value *AndRHS2 = ConstantExpr::getCast(AndRHS, ShVal->getType());
          ShVal = InsertNewInstBefore(BinaryOperator::createAnd(ShVal, AndRHS2,
                                                             TheAnd.getName()),
                                      TheAnd);
          return new CastInst(ShVal, Op->getType());
        }
      }
    }
    break;
  }
  return 0;
}


/// InsertRangeTest - Emit a computation of: (V >= Lo && V < Hi) if Inside is
/// true, otherwise (V < Lo || V >= Hi).  In pratice, we emit the more efficient
/// (V-Lo) <u Hi-Lo.  This method expects that Lo <= Hi.  IB is the location to
/// insert new instructions.
Instruction *InstCombiner::InsertRangeTest(Value *V, Constant *Lo, Constant *Hi,
                                           bool Inside, Instruction &IB) {
  assert(cast<ConstantBool>(ConstantExpr::getSetLE(Lo, Hi))->getValue() &&
         "Lo is not <= Hi in range emission code!");
  if (Inside) {
    if (Lo == Hi)  // Trivially false.
      return new SetCondInst(Instruction::SetNE, V, V);
    if (cast<ConstantIntegral>(Lo)->isMinValue())
      return new SetCondInst(Instruction::SetLT, V, Hi);

    Constant *AddCST = ConstantExpr::getNeg(Lo);
    Instruction *Add = BinaryOperator::createAdd(V, AddCST,V->getName()+".off");
    InsertNewInstBefore(Add, IB);
    // Convert to unsigned for the comparison.
    const Type *UnsType = Add->getType()->getUnsignedVersion();
    Value *OffsetVal = InsertCastBefore(Add, UnsType, IB);
    AddCST = ConstantExpr::getAdd(AddCST, Hi);
    AddCST = ConstantExpr::getCast(AddCST, UnsType);
    return new SetCondInst(Instruction::SetLT, OffsetVal, AddCST);
  }

  if (Lo == Hi)  // Trivially true.
    return new SetCondInst(Instruction::SetEQ, V, V);

  Hi = SubOne(cast<ConstantInt>(Hi));

  // V < 0 || V >= Hi ->'V > Hi-1'
  if (cast<ConstantIntegral>(Lo)->isMinValue())
    return new SetCondInst(Instruction::SetGT, V, Hi);

  // Emit X-Lo > Hi-Lo-1
  Constant *AddCST = ConstantExpr::getNeg(Lo);
  Instruction *Add = BinaryOperator::createAdd(V, AddCST, V->getName()+".off");
  InsertNewInstBefore(Add, IB);
  // Convert to unsigned for the comparison.
  const Type *UnsType = Add->getType()->getUnsignedVersion();
  Value *OffsetVal = InsertCastBefore(Add, UnsType, IB);
  AddCST = ConstantExpr::getAdd(AddCST, Hi);
  AddCST = ConstantExpr::getCast(AddCST, UnsType);
  return new SetCondInst(Instruction::SetGT, OffsetVal, AddCST);
}

// isRunOfOnes - Returns true iff Val consists of one contiguous run of 1s with
// any number of 0s on either side.  The 1s are allowed to wrap from LSB to
// MSB, so 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.  0x0F0F0000 is
// not, since all 1s are not contiguous.
static bool isRunOfOnes(ConstantIntegral *Val, unsigned &MB, unsigned &ME) {
  uint64_t V = Val->getZExtValue();
  if (!isShiftedMask_64(V)) return false;

  // look for the first zero bit after the run of ones
  MB = 64-CountLeadingZeros_64((V - 1) ^ V);
  // look for the first non-zero bit
  ME = 64-CountLeadingZeros_64(V);
  return true;
}



/// FoldLogicalPlusAnd - This is part of an expression (LHS +/- RHS) & Mask,
/// where isSub determines whether the operator is a sub.  If we can fold one of
/// the following xforms:
/// 
/// ((A & N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == Mask
/// ((A | N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == 0
/// ((A ^ N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == 0
///
/// return (A +/- B).
///
Value *InstCombiner::FoldLogicalPlusAnd(Value *LHS, Value *RHS,
                                        ConstantIntegral *Mask, bool isSub,
                                        Instruction &I) {
  Instruction *LHSI = dyn_cast<Instruction>(LHS);
  if (!LHSI || LHSI->getNumOperands() != 2 ||
      !isa<ConstantInt>(LHSI->getOperand(1))) return 0;

  ConstantInt *N = cast<ConstantInt>(LHSI->getOperand(1));

  switch (LHSI->getOpcode()) {
  default: return 0;
  case Instruction::And:
    if (ConstantExpr::getAnd(N, Mask) == Mask) {
      // If the AndRHS is a power of two minus one (0+1+), this is simple.
      if ((Mask->getZExtValue() & Mask->getZExtValue()+1) == 0)
        break;

      // Otherwise, if Mask is 0+1+0+, and if B is known to have the low 0+
      // part, we don't need any explicit masks to take them out of A.  If that
      // is all N is, ignore it.
      unsigned MB, ME;
      if (isRunOfOnes(Mask, MB, ME)) {  // begin/end bit of run, inclusive
        uint64_t Mask = RHS->getType()->getIntegralTypeMask();
        Mask >>= 64-MB+1;
        if (MaskedValueIsZero(RHS, Mask))
          break;
      }
    }
    return 0;
  case Instruction::Or:
  case Instruction::Xor:
    // If the AndRHS is a power of two minus one (0+1+), and N&Mask == 0
    if ((Mask->getZExtValue() & Mask->getZExtValue()+1) == 0 &&
        ConstantExpr::getAnd(N, Mask)->isNullValue())
      break;
    return 0;
  }
  
  Instruction *New;
  if (isSub)
    New = BinaryOperator::createSub(LHSI->getOperand(0), RHS, "fold");
  else
    New = BinaryOperator::createAdd(LHSI->getOperand(0), RHS, "fold");
  return InsertNewInstBefore(New, I);
}

Instruction *InstCombiner::visitAnd(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op1))                         // X & undef -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // and X, X = X
  if (Op0 == Op1)
    return ReplaceInstUsesWith(I, Op1);

  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  uint64_t KnownZero, KnownOne;
  if (!isa<PackedType>(I.getType()) &&
      SimplifyDemandedBits(&I, I.getType()->getIntegralTypeMask(),
                           KnownZero, KnownOne))
    return &I;
  
  if (ConstantIntegral *AndRHS = dyn_cast<ConstantIntegral>(Op1)) {
    uint64_t AndRHSMask = AndRHS->getZExtValue();
    uint64_t TypeMask = Op0->getType()->getIntegralTypeMask();
    uint64_t NotAndRHS = AndRHSMask^TypeMask;

    // Optimize a variety of ((val OP C1) & C2) combinations...
    if (isa<BinaryOperator>(Op0) || isa<ShiftInst>(Op0)) {
      Instruction *Op0I = cast<Instruction>(Op0);
      Value *Op0LHS = Op0I->getOperand(0);
      Value *Op0RHS = Op0I->getOperand(1);
      switch (Op0I->getOpcode()) {
      case Instruction::Xor:
      case Instruction::Or:
        // If the mask is only needed on one incoming arm, push it up.
        if (Op0I->hasOneUse()) {
          if (MaskedValueIsZero(Op0LHS, NotAndRHS)) {
            // Not masking anything out for the LHS, move to RHS.
            Instruction *NewRHS = BinaryOperator::createAnd(Op0RHS, AndRHS,
                                                   Op0RHS->getName()+".masked");
            InsertNewInstBefore(NewRHS, I);
            return BinaryOperator::create(
                       cast<BinaryOperator>(Op0I)->getOpcode(), Op0LHS, NewRHS);
          }
          if (!isa<Constant>(Op0RHS) &&
              MaskedValueIsZero(Op0RHS, NotAndRHS)) {
            // Not masking anything out for the RHS, move to LHS.
            Instruction *NewLHS = BinaryOperator::createAnd(Op0LHS, AndRHS,
                                                   Op0LHS->getName()+".masked");
            InsertNewInstBefore(NewLHS, I);
            return BinaryOperator::create(
                       cast<BinaryOperator>(Op0I)->getOpcode(), NewLHS, Op0RHS);
          }
        }

        break;
      case Instruction::Add:
        // ((A & N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == AndRHS.
        // ((A | N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == 0
        // ((A ^ N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == 0
        if (Value *V = FoldLogicalPlusAnd(Op0LHS, Op0RHS, AndRHS, false, I))
          return BinaryOperator::createAnd(V, AndRHS);
        if (Value *V = FoldLogicalPlusAnd(Op0RHS, Op0LHS, AndRHS, false, I))
          return BinaryOperator::createAnd(V, AndRHS);  // Add commutes
        break;

      case Instruction::Sub:
        // ((A & N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == AndRHS.
        // ((A | N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == 0
        // ((A ^ N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == 0
        if (Value *V = FoldLogicalPlusAnd(Op0LHS, Op0RHS, AndRHS, true, I))
          return BinaryOperator::createAnd(V, AndRHS);
        break;
      }

      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1)))
        if (Instruction *Res = OptAndOp(Op0I, Op0CI, AndRHS, I))
          return Res;
    } else if (CastInst *CI = dyn_cast<CastInst>(Op0)) {
      const Type *SrcTy = CI->getOperand(0)->getType();

      // If this is an integer truncation or change from signed-to-unsigned, and
      // if the source is an and/or with immediate, transform it.  This
      // frequently occurs for bitfield accesses.
      if (Instruction *CastOp = dyn_cast<Instruction>(CI->getOperand(0))) {
        if (SrcTy->getPrimitiveSizeInBits() >= 
              I.getType()->getPrimitiveSizeInBits() &&
            CastOp->getNumOperands() == 2)
          if (ConstantInt *AndCI = dyn_cast<ConstantInt>(CastOp->getOperand(1)))
            if (CastOp->getOpcode() == Instruction::And) {
              // Change: and (cast (and X, C1) to T), C2
              // into  : and (cast X to T), trunc(C1)&C2
              // This will folds the two ands together, which may allow other
              // simplifications.
              Instruction *NewCast =
                new CastInst(CastOp->getOperand(0), I.getType(),
                             CastOp->getName()+".shrunk");
              NewCast = InsertNewInstBefore(NewCast, I);
              
              Constant *C3=ConstantExpr::getCast(AndCI, I.getType());//trunc(C1)
              C3 = ConstantExpr::getAnd(C3, AndRHS);            // trunc(C1)&C2
              return BinaryOperator::createAnd(NewCast, C3);
            } else if (CastOp->getOpcode() == Instruction::Or) {
              // Change: and (cast (or X, C1) to T), C2
              // into  : trunc(C1)&C2 iff trunc(C1)&C2 == C2
              Constant *C3=ConstantExpr::getCast(AndCI, I.getType());//trunc(C1)
              if (ConstantExpr::getAnd(C3, AndRHS) == AndRHS)   // trunc(C1)&C2
                return ReplaceInstUsesWith(I, AndRHS);
            }
      }
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
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
  
  {
    Value *A = 0, *B = 0;
    if (match(Op0, m_Or(m_Value(A), m_Value(B))))
      if (A == Op1 || B == Op1)    // (A | ?) & A  --> A
        return ReplaceInstUsesWith(I, Op1);
    if (match(Op1, m_Or(m_Value(A), m_Value(B))))
      if (A == Op0 || B == Op0)    // A & (A | ?)  --> A
        return ReplaceInstUsesWith(I, Op0);
    
    if (Op0->hasOneUse() &&
        match(Op0, m_Xor(m_Value(A), m_Value(B)))) {
      if (A == Op1) {                                // (A^B)&A -> A&(A^B)
        I.swapOperands();     // Simplify below
        std::swap(Op0, Op1);
      } else if (B == Op1) {                         // (A^B)&B -> B&(B^A)
        cast<BinaryOperator>(Op0)->swapOperands();
        I.swapOperands();     // Simplify below
        std::swap(Op0, Op1);
      }
    }
    if (Op1->hasOneUse() &&
        match(Op1, m_Xor(m_Value(A), m_Value(B)))) {
      if (B == Op0) {                                // B&(A^B) -> B&(B^A)
        cast<BinaryOperator>(Op1)->swapOperands();
        std::swap(A, B);
      }
      if (A == Op0) {                                // A&(A^B) -> A & ~B
        Instruction *NotB = BinaryOperator::createNot(B, "tmp");
        InsertNewInstBefore(NotB, I);
        return BinaryOperator::createAnd(A, NotB);
      }
    }
  }
  

  if (SetCondInst *RHS = dyn_cast<SetCondInst>(Op1)) {
    // (setcc1 A, B) & (setcc2 A, B) --> (setcc3 A, B)
    if (Instruction *R = AssociativeOpt(I, FoldSetCCLogical(*this, RHS)))
      return R;

    Value *LHSVal, *RHSVal;
    ConstantInt *LHSCst, *RHSCst;
    Instruction::BinaryOps LHSCC, RHSCC;
    if (match(Op0, m_SetCond(LHSCC, m_Value(LHSVal), m_ConstantInt(LHSCst))))
      if (match(RHS, m_SetCond(RHSCC, m_Value(RHSVal), m_ConstantInt(RHSCst))))
        if (LHSVal == RHSVal &&    // Found (X setcc C1) & (X setcc C2)
            // Set[GL]E X, CST is folded to Set[GL]T elsewhere.
            LHSCC != Instruction::SetGE && LHSCC != Instruction::SetLE &&
            RHSCC != Instruction::SetGE && RHSCC != Instruction::SetLE) {
          // Ensure that the larger constant is on the RHS.
          Constant *Cmp = ConstantExpr::getSetGT(LHSCst, RHSCst);
          SetCondInst *LHS = cast<SetCondInst>(Op0);
          if (cast<ConstantBool>(Cmp)->getValue()) {
            std::swap(LHS, RHS);
            std::swap(LHSCst, RHSCst);
            std::swap(LHSCC, RHSCC);
          }

          // At this point, we know we have have two setcc instructions
          // comparing a value against two constants and and'ing the result
          // together.  Because of the above check, we know that we only have
          // SetEQ, SetNE, SetLT, and SetGT here.  We also know (from the
          // FoldSetCCLogical check above), that the two constants are not
          // equal.
          assert(LHSCst != RHSCst && "Compares not folded above?");

          switch (LHSCC) {
          default: assert(0 && "Unknown integer condition code!");
          case Instruction::SetEQ:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:  // (X == 13 & X == 15) -> false
            case Instruction::SetGT:  // (X == 13 & X > 15)  -> false
              return ReplaceInstUsesWith(I, ConstantBool::getFalse());
            case Instruction::SetNE:  // (X == 13 & X != 15) -> X == 13
            case Instruction::SetLT:  // (X == 13 & X < 15)  -> X == 13
              return ReplaceInstUsesWith(I, LHS);
            }
          case Instruction::SetNE:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetLT:
              if (LHSCst == SubOne(RHSCst)) // (X != 13 & X < 14) -> X < 13
                return new SetCondInst(Instruction::SetLT, LHSVal, LHSCst);
              break;                        // (X != 13 & X < 15) -> no change
            case Instruction::SetEQ:        // (X != 13 & X == 15) -> X == 15
            case Instruction::SetGT:        // (X != 13 & X > 15)  -> X > 15
              return ReplaceInstUsesWith(I, RHS);
            case Instruction::SetNE:
              if (LHSCst == SubOne(RHSCst)) {// (X != 13 & X != 14) -> X-13 >u 1
                Constant *AddCST = ConstantExpr::getNeg(LHSCst);
                Instruction *Add = BinaryOperator::createAdd(LHSVal, AddCST,
                                                      LHSVal->getName()+".off");
                InsertNewInstBefore(Add, I);
                const Type *UnsType = Add->getType()->getUnsignedVersion();
                Value *OffsetVal = InsertCastBefore(Add, UnsType, I);
                AddCST = ConstantExpr::getSub(RHSCst, LHSCst);
                AddCST = ConstantExpr::getCast(AddCST, UnsType);
                return new SetCondInst(Instruction::SetGT, OffsetVal, AddCST);
              }
              break;                        // (X != 13 & X != 15) -> no change
            }
            break;
          case Instruction::SetLT:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:  // (X < 13 & X == 15) -> false
            case Instruction::SetGT:  // (X < 13 & X > 15)  -> false
              return ReplaceInstUsesWith(I, ConstantBool::getFalse());
            case Instruction::SetNE:  // (X < 13 & X != 15) -> X < 13
            case Instruction::SetLT:  // (X < 13 & X < 15) -> X < 13
              return ReplaceInstUsesWith(I, LHS);
            }
          case Instruction::SetGT:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:  // (X > 13 & X == 15) -> X > 13
              return ReplaceInstUsesWith(I, LHS);
            case Instruction::SetGT:  // (X > 13 & X > 15)  -> X > 15
              return ReplaceInstUsesWith(I, RHS);
            case Instruction::SetNE:
              if (RHSCst == AddOne(LHSCst)) // (X > 13 & X != 14) -> X > 14
                return new SetCondInst(Instruction::SetGT, LHSVal, RHSCst);
              break;                        // (X > 13 & X != 15) -> no change
            case Instruction::SetLT:   // (X > 13 & X < 15) -> (X-14) <u 1
              return InsertRangeTest(LHSVal, AddOne(LHSCst), RHSCst, true, I);
            }
          }
        }
  }

  // fold (and (cast A), (cast B)) -> (cast (and A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
    const Type *SrcTy = Op0C->getOperand(0)->getType();
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1))
      if (SrcTy == Op1C->getOperand(0)->getType() && SrcTy->isIntegral() &&
          // Only do this if the casts both really cause code to be generated.
          ValueRequiresCast(Op0C->getOperand(0), I.getType(), TD) &&
          ValueRequiresCast(Op1C->getOperand(0), I.getType(), TD)) {
        Instruction *NewOp = BinaryOperator::createAnd(Op0C->getOperand(0),
                                                       Op1C->getOperand(0),
                                                       I.getName());
        InsertNewInstBefore(NewOp, I);
        return new CastInst(NewOp, I.getType());
      }
  }

  return Changed ? &I : 0;
}

/// CollectBSwapParts - Look to see if the specified value defines a single byte
/// in the result.  If it does, and if the specified byte hasn't been filled in
/// yet, fill it in and return false.
static bool CollectBSwapParts(Value *V, std::vector<Value*> &ByteValues) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) return true;

  // If this is an or instruction, it is an inner node of the bswap.
  if (I->getOpcode() == Instruction::Or)
    return CollectBSwapParts(I->getOperand(0), ByteValues) ||
           CollectBSwapParts(I->getOperand(1), ByteValues);
  
  // If this is a shift by a constant int, and it is "24", then its operand
  // defines a byte.  We only handle unsigned types here.
  if (isa<ShiftInst>(I) && isa<ConstantInt>(I->getOperand(1))) {
    // Not shifting the entire input by N-1 bytes?
    if (cast<ConstantInt>(I->getOperand(1))->getZExtValue() !=
        8*(ByteValues.size()-1))
      return true;
    
    unsigned DestNo;
    if (I->getOpcode() == Instruction::Shl) {
      // X << 24 defines the top byte with the lowest of the input bytes.
      DestNo = ByteValues.size()-1;
    } else {
      // X >>u 24 defines the low byte with the highest of the input bytes.
      DestNo = 0;
    }
    
    // If the destination byte value is already defined, the values are or'd
    // together, which isn't a bswap (unless it's an or of the same bits).
    if (ByteValues[DestNo] && ByteValues[DestNo] != I->getOperand(0))
      return true;
    ByteValues[DestNo] = I->getOperand(0);
    return false;
  }
  
  // Otherwise, we can only handle and(shift X, imm), imm).  Bail out of if we
  // don't have this.
  Value *Shift = 0, *ShiftLHS = 0;
  ConstantInt *AndAmt = 0, *ShiftAmt = 0;
  if (!match(I, m_And(m_Value(Shift), m_ConstantInt(AndAmt))) ||
      !match(Shift, m_Shift(m_Value(ShiftLHS), m_ConstantInt(ShiftAmt))))
    return true;
  Instruction *SI = cast<Instruction>(Shift);

  // Make sure that the shift amount is by a multiple of 8 and isn't too big.
  if (ShiftAmt->getZExtValue() & 7 ||
      ShiftAmt->getZExtValue() > 8*ByteValues.size())
    return true;
  
  // Turn 0xFF -> 0, 0xFF00 -> 1, 0xFF0000 -> 2, etc.
  unsigned DestByte;
  for (DestByte = 0; DestByte != ByteValues.size(); ++DestByte)
    if (AndAmt->getZExtValue() == uint64_t(0xFF) << 8*DestByte)
      break;
  // Unknown mask for bswap.
  if (DestByte == ByteValues.size()) return true;
  
  unsigned ShiftBytes = ShiftAmt->getZExtValue()/8;
  unsigned SrcByte;
  if (SI->getOpcode() == Instruction::Shl)
    SrcByte = DestByte - ShiftBytes;
  else
    SrcByte = DestByte + ShiftBytes;
  
  // If the SrcByte isn't a bswapped value from the DestByte, reject it.
  if (SrcByte != ByteValues.size()-DestByte-1)
    return true;
  
  // If the destination byte value is already defined, the values are or'd
  // together, which isn't a bswap (unless it's an or of the same bits).
  if (ByteValues[DestByte] && ByteValues[DestByte] != SI->getOperand(0))
    return true;
  ByteValues[DestByte] = SI->getOperand(0);
  return false;
}

/// MatchBSwap - Given an OR instruction, check to see if this is a bswap idiom.
/// If so, insert the new bswap intrinsic and return it.
Instruction *InstCombiner::MatchBSwap(BinaryOperator &I) {
  // We can only handle bswap of unsigned integers, and cannot bswap one byte.
  if (!I.getType()->isUnsigned() || I.getType() == Type::UByteTy)
    return 0;
  
  /// ByteValues - For each byte of the result, we keep track of which value
  /// defines each byte.
  std::vector<Value*> ByteValues;
  ByteValues.resize(I.getType()->getPrimitiveSize());
    
  // Try to find all the pieces corresponding to the bswap.
  if (CollectBSwapParts(I.getOperand(0), ByteValues) ||
      CollectBSwapParts(I.getOperand(1), ByteValues))
    return 0;
  
  // Check to see if all of the bytes come from the same value.
  Value *V = ByteValues[0];
  if (V == 0) return 0;  // Didn't find a byte?  Must be zero.
  
  // Check to make sure that all of the bytes come from the same value.
  for (unsigned i = 1, e = ByteValues.size(); i != e; ++i)
    if (ByteValues[i] != V)
      return 0;
    
  // If they do then *success* we can turn this into a bswap.  Figure out what
  // bswap to make it into.
  Module *M = I.getParent()->getParent()->getParent();
  const char *FnName = 0;
  if (I.getType() == Type::UShortTy)
    FnName = "llvm.bswap.i16";
  else if (I.getType() == Type::UIntTy)
    FnName = "llvm.bswap.i32";
  else if (I.getType() == Type::ULongTy)
    FnName = "llvm.bswap.i64";
  else
    assert(0 && "Unknown integer type!");
  Function *F = M->getOrInsertFunction(FnName, I.getType(), I.getType(), NULL);
  
  return new CallInst(F, V);
}


Instruction *InstCombiner::visitOr(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I,                         // X | undef -> -1
                               ConstantIntegral::getAllOnesValue(I.getType()));

  // or X, X = X
  if (Op0 == Op1)
    return ReplaceInstUsesWith(I, Op0);

  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  uint64_t KnownZero, KnownOne;
  if (!isa<PackedType>(I.getType()) &&
      SimplifyDemandedBits(&I, I.getType()->getIntegralTypeMask(),
                           KnownZero, KnownOne))
    return &I;
  
  // or X, -1 == -1
  if (ConstantIntegral *RHS = dyn_cast<ConstantIntegral>(Op1)) {
    ConstantInt *C1 = 0; Value *X = 0;
    // (X & C1) | C2 --> (X | C2) & (C1|C2)
    if (match(Op0, m_And(m_Value(X), m_ConstantInt(C1))) && isOnlyUse(Op0)) {
      Instruction *Or = BinaryOperator::createOr(X, RHS, Op0->getName());
      Op0->setName("");
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
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  Value *A = 0, *B = 0;
  ConstantInt *C1 = 0, *C2 = 0;

  if (match(Op0, m_And(m_Value(A), m_Value(B))))
    if (A == Op1 || B == Op1)    // (A & ?) | A  --> A
      return ReplaceInstUsesWith(I, Op1);
  if (match(Op1, m_And(m_Value(A), m_Value(B))))
    if (A == Op0 || B == Op0)    // A | (A & ?)  --> A
      return ReplaceInstUsesWith(I, Op0);

  // (A | B) | C  and  A | (B | C)                  -> bswap if possible.
  // (A >> B) | (C << D)  and  (A << B) | (B >> C)  -> bswap if possible.
  if (match(Op0, m_Or(m_Value(), m_Value())) ||
      match(Op1, m_Or(m_Value(), m_Value())) ||
      (match(Op0, m_Shift(m_Value(), m_Value())) &&
       match(Op1, m_Shift(m_Value(), m_Value())))) {
    if (Instruction *BSwap = MatchBSwap(I))
      return BSwap;
  }
  
  // (X^C)|Y -> (X|Y)^C iff Y&C == 0
  if (Op0->hasOneUse() && match(Op0, m_Xor(m_Value(A), m_ConstantInt(C1))) &&
      MaskedValueIsZero(Op1, C1->getZExtValue())) {
    Instruction *NOr = BinaryOperator::createOr(A, Op1, Op0->getName());
    Op0->setName("");
    return BinaryOperator::createXor(InsertNewInstBefore(NOr, I), C1);
  }

  // Y|(X^C) -> (X|Y)^C iff Y&C == 0
  if (Op1->hasOneUse() && match(Op1, m_Xor(m_Value(A), m_ConstantInt(C1))) &&
      MaskedValueIsZero(Op0, C1->getZExtValue())) {
    Instruction *NOr = BinaryOperator::createOr(A, Op0, Op1->getName());
    Op0->setName("");
    return BinaryOperator::createXor(InsertNewInstBefore(NOr, I), C1);
  }

  // (A & C1)|(B & C2)
  if (match(Op0, m_And(m_Value(A), m_ConstantInt(C1))) &&
      match(Op1, m_And(m_Value(B), m_ConstantInt(C2)))) {

    if (A == B)  // (A & C1)|(A & C2) == A & (C1|C2)
      return BinaryOperator::createAnd(A, ConstantExpr::getOr(C1, C2));


    // If we have: ((V + N) & C1) | (V & C2)
    // .. and C2 = ~C1 and C2 is 0+1+ and (N & C2) == 0
    // replace with V+N.
    if (C1 == ConstantExpr::getNot(C2)) {
      Value *V1 = 0, *V2 = 0;
      if ((C2->getZExtValue() & (C2->getZExtValue()+1)) == 0 && // C2 == 0+1+
          match(A, m_Add(m_Value(V1), m_Value(V2)))) {
        // Add commutes, try both ways.
        if (V1 == B && MaskedValueIsZero(V2, C2->getZExtValue()))
          return ReplaceInstUsesWith(I, A);
        if (V2 == B && MaskedValueIsZero(V1, C2->getZExtValue()))
          return ReplaceInstUsesWith(I, A);
      }
      // Or commutes, try both ways.
      if ((C1->getZExtValue() & (C1->getZExtValue()+1)) == 0 &&
          match(B, m_Add(m_Value(V1), m_Value(V2)))) {
        // Add commutes, try both ways.
        if (V1 == A && MaskedValueIsZero(V2, C1->getZExtValue()))
          return ReplaceInstUsesWith(I, B);
        if (V2 == A && MaskedValueIsZero(V1, C1->getZExtValue()))
          return ReplaceInstUsesWith(I, B);
      }
    }
  }

  if (match(Op0, m_Not(m_Value(A)))) {   // ~A | Op1
    if (A == Op1)   // ~A | A == -1
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));
  } else {
    A = 0;
  }
  // Note, A is still live here!
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
  if (SetCondInst *RHS = dyn_cast<SetCondInst>(I.getOperand(1))) {
    if (Instruction *R = AssociativeOpt(I, FoldSetCCLogical(*this, RHS)))
      return R;

    Value *LHSVal, *RHSVal;
    ConstantInt *LHSCst, *RHSCst;
    Instruction::BinaryOps LHSCC, RHSCC;
    if (match(Op0, m_SetCond(LHSCC, m_Value(LHSVal), m_ConstantInt(LHSCst))))
      if (match(RHS, m_SetCond(RHSCC, m_Value(RHSVal), m_ConstantInt(RHSCst))))
        if (LHSVal == RHSVal &&    // Found (X setcc C1) | (X setcc C2)
            // Set[GL]E X, CST is folded to Set[GL]T elsewhere.
            LHSCC != Instruction::SetGE && LHSCC != Instruction::SetLE &&
            RHSCC != Instruction::SetGE && RHSCC != Instruction::SetLE) {
          // Ensure that the larger constant is on the RHS.
          Constant *Cmp = ConstantExpr::getSetGT(LHSCst, RHSCst);
          SetCondInst *LHS = cast<SetCondInst>(Op0);
          if (cast<ConstantBool>(Cmp)->getValue()) {
            std::swap(LHS, RHS);
            std::swap(LHSCst, RHSCst);
            std::swap(LHSCC, RHSCC);
          }

          // At this point, we know we have have two setcc instructions
          // comparing a value against two constants and or'ing the result
          // together.  Because of the above check, we know that we only have
          // SetEQ, SetNE, SetLT, and SetGT here.  We also know (from the
          // FoldSetCCLogical check above), that the two constants are not
          // equal.
          assert(LHSCst != RHSCst && "Compares not folded above?");

          switch (LHSCC) {
          default: assert(0 && "Unknown integer condition code!");
          case Instruction::SetEQ:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:
              if (LHSCst == SubOne(RHSCst)) {// (X == 13 | X == 14) -> X-13 <u 2
                Constant *AddCST = ConstantExpr::getNeg(LHSCst);
                Instruction *Add = BinaryOperator::createAdd(LHSVal, AddCST,
                                                      LHSVal->getName()+".off");
                InsertNewInstBefore(Add, I);
                const Type *UnsType = Add->getType()->getUnsignedVersion();
                Value *OffsetVal = InsertCastBefore(Add, UnsType, I);
                AddCST = ConstantExpr::getSub(AddOne(RHSCst), LHSCst);
                AddCST = ConstantExpr::getCast(AddCST, UnsType);
                return new SetCondInst(Instruction::SetLT, OffsetVal, AddCST);
              }
              break;                  // (X == 13 | X == 15) -> no change

            case Instruction::SetGT:  // (X == 13 | X > 14) -> no change
              break;
            case Instruction::SetNE:  // (X == 13 | X != 15) -> X != 15
            case Instruction::SetLT:  // (X == 13 | X < 15)  -> X < 15
              return ReplaceInstUsesWith(I, RHS);
            }
            break;
          case Instruction::SetNE:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:        // (X != 13 | X == 15) -> X != 13
            case Instruction::SetGT:        // (X != 13 | X > 15)  -> X != 13
              return ReplaceInstUsesWith(I, LHS);
            case Instruction::SetNE:        // (X != 13 | X != 15) -> true
            case Instruction::SetLT:        // (X != 13 | X < 15)  -> true
              return ReplaceInstUsesWith(I, ConstantBool::getTrue());
            }
            break;
          case Instruction::SetLT:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:  // (X < 13 | X == 14) -> no change
              break;
            case Instruction::SetGT:  // (X < 13 | X > 15)  -> (X-13) > 2
              return InsertRangeTest(LHSVal, LHSCst, AddOne(RHSCst), false, I);
            case Instruction::SetNE:  // (X < 13 | X != 15) -> X != 15
            case Instruction::SetLT:  // (X < 13 | X < 15) -> X < 15
              return ReplaceInstUsesWith(I, RHS);
            }
            break;
          case Instruction::SetGT:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:  // (X > 13 | X == 15) -> X > 13
            case Instruction::SetGT:  // (X > 13 | X > 15)  -> X > 13
              return ReplaceInstUsesWith(I, LHS);
            case Instruction::SetNE:  // (X > 13 | X != 15)  -> true
            case Instruction::SetLT:  // (X > 13 | X < 15) -> true
              return ReplaceInstUsesWith(I, ConstantBool::getTrue());
            }
          }
        }
  }
    
  // fold (or (cast A), (cast B)) -> (cast (or A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
    const Type *SrcTy = Op0C->getOperand(0)->getType();
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1))
      if (SrcTy == Op1C->getOperand(0)->getType() && SrcTy->isIntegral() &&
          // Only do this if the casts both really cause code to be generated.
          ValueRequiresCast(Op0C->getOperand(0), I.getType(), TD) &&
          ValueRequiresCast(Op1C->getOperand(0), I.getType(), TD)) {
        Instruction *NewOp = BinaryOperator::createOr(Op0C->getOperand(0),
                                                      Op1C->getOperand(0),
                                                      I.getName());
        InsertNewInstBefore(NewOp, I);
        return new CastInst(NewOp, I.getType());
      }
  }
      

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

  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);  // X ^ undef -> undef

  // xor X, X = 0, even if X is nested in a sequence of Xor's.
  if (Instruction *Result = AssociativeOpt(I, XorSelf(Op1))) {
    assert(Result == &I && "AssociativeOpt didn't work?");
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }
  
  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  uint64_t KnownZero, KnownOne;
  if (!isa<PackedType>(I.getType()) &&
      SimplifyDemandedBits(&I, I.getType()->getIntegralTypeMask(),
                           KnownZero, KnownOne))
    return &I;

  if (ConstantIntegral *RHS = dyn_cast<ConstantIntegral>(Op1)) {
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      // xor (setcc A, B), true = not (setcc A, B) = setncc A, B
      if (SetCondInst *SCI = dyn_cast<SetCondInst>(Op0I))
        if (RHS == ConstantBool::getTrue() && SCI->hasOneUse())
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
        if (Op0I->getOpcode() == Instruction::Add) {
          // ~(X-c) --> (-c-1)-X
          if (RHS->isAllOnesValue()) {
            Constant *NegOp0CI = ConstantExpr::getNeg(Op0CI);
            return BinaryOperator::createSub(
                           ConstantExpr::getSub(NegOp0CI,
                                             ConstantInt::get(I.getType(), 1)),
                                          Op0I->getOperand(0));
          }
        } else if (Op0I->getOpcode() == Instruction::Or) {
          // (X|C1)^C2 -> X^(C1|C2) iff X&~C1 == 0
          if (MaskedValueIsZero(Op0I->getOperand(0), Op0CI->getZExtValue())) {
            Constant *NewRHS = ConstantExpr::getOr(Op0CI, RHS);
            // Anything in both C1 and C2 is known to be zero, remove it from
            // NewRHS.
            Constant *CommonBits = ConstantExpr::getAnd(Op0CI, RHS);
            NewRHS = ConstantExpr::getAnd(NewRHS, 
                                          ConstantExpr::getNot(CommonBits));
            WorkList.push_back(Op0I);
            I.setOperand(0, Op0I->getOperand(0));
            I.setOperand(1, NewRHS);
            return &I;
          }
        }
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  if (Value *X = dyn_castNotVal(Op0))   // ~A ^ A == -1
    if (X == Op1)
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));

  if (Value *X = dyn_castNotVal(Op1))   // A ^ ~A == -1
    if (X == Op0)
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));

  if (BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1))
    if (Op1I->getOpcode() == Instruction::Or) {
      if (Op1I->getOperand(0) == Op0) {              // B^(B|A) == (A|B)^B
        Op1I->swapOperands();
        I.swapOperands();
        std::swap(Op0, Op1);
      } else if (Op1I->getOperand(1) == Op0) {       // B^(A|B) == (A|B)^B
        I.swapOperands();     // Simplified below.
        std::swap(Op0, Op1);
      }
    } else if (Op1I->getOpcode() == Instruction::Xor) {
      if (Op0 == Op1I->getOperand(0))                        // A^(A^B) == B
        return ReplaceInstUsesWith(I, Op1I->getOperand(1));
      else if (Op0 == Op1I->getOperand(1))                   // A^(B^A) == B
        return ReplaceInstUsesWith(I, Op1I->getOperand(0));
    } else if (Op1I->getOpcode() == Instruction::And && Op1I->hasOneUse()) {
      if (Op1I->getOperand(0) == Op0)                      // A^(A&B) -> A^(B&A)
        Op1I->swapOperands();
      if (Op0 == Op1I->getOperand(1)) {                    // A^(B&A) -> (B&A)^A
        I.swapOperands();     // Simplified below.
        std::swap(Op0, Op1);
      }
    }

  if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0))
    if (Op0I->getOpcode() == Instruction::Or && Op0I->hasOneUse()) {
      if (Op0I->getOperand(0) == Op1)                // (B|A)^B == (A|B)^B
        Op0I->swapOperands();
      if (Op0I->getOperand(1) == Op1) {              // (A|B)^B == A & ~B
        Instruction *NotB = BinaryOperator::createNot(Op1, "tmp");
        InsertNewInstBefore(NotB, I);
        return BinaryOperator::createAnd(Op0I->getOperand(0), NotB);
      }
    } else if (Op0I->getOpcode() == Instruction::Xor) {
      if (Op1 == Op0I->getOperand(0))                        // (A^B)^A == B
        return ReplaceInstUsesWith(I, Op0I->getOperand(1));
      else if (Op1 == Op0I->getOperand(1))                   // (B^A)^A == B
        return ReplaceInstUsesWith(I, Op0I->getOperand(0));
    } else if (Op0I->getOpcode() == Instruction::And && Op0I->hasOneUse()) {
      if (Op0I->getOperand(0) == Op1)                      // (A&B)^A -> (B&A)^A
        Op0I->swapOperands();
      if (Op0I->getOperand(1) == Op1 &&                    // (B&A)^A == ~B & A
          !isa<ConstantInt>(Op1)) {  // Canonical form is (B&C)^C
        Instruction *N = BinaryOperator::createNot(Op0I->getOperand(0), "tmp");
        InsertNewInstBefore(N, I);
        return BinaryOperator::createAnd(N, Op1);
      }
    }

  // (setcc1 A, B) ^ (setcc2 A, B) --> (setcc3 A, B)
  if (SetCondInst *RHS = dyn_cast<SetCondInst>(I.getOperand(1)))
    if (Instruction *R = AssociativeOpt(I, FoldSetCCLogical(*this, RHS)))
      return R;

  // fold (xor (cast A), (cast B)) -> (cast (xor A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
    const Type *SrcTy = Op0C->getOperand(0)->getType();
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1))
      if (SrcTy == Op1C->getOperand(0)->getType() && SrcTy->isIntegral() &&
          // Only do this if the casts both really cause code to be generated.
          ValueRequiresCast(Op0C->getOperand(0), I.getType(), TD) &&
          ValueRequiresCast(Op1C->getOperand(0), I.getType(), TD)) {
        Instruction *NewOp = BinaryOperator::createXor(Op0C->getOperand(0),
                                                       Op1C->getOperand(0),
                                                       I.getName());
        InsertNewInstBefore(NewOp, I);
        return new CastInst(NewOp, I.getType());
      }
  }
    
  return Changed ? &I : 0;
}

static bool isPositive(ConstantInt *C) {
  return C->getSExtValue() >= 0;
}

/// AddWithOverflow - Compute Result = In1+In2, returning true if the result
/// overflowed for this type.
static bool AddWithOverflow(ConstantInt *&Result, ConstantInt *In1,
                            ConstantInt *In2) {
  Result = cast<ConstantInt>(ConstantExpr::getAdd(In1, In2));

  if (In1->getType()->isUnsigned())
    return cast<ConstantInt>(Result)->getZExtValue() <
           cast<ConstantInt>(In1)->getZExtValue();
  if (isPositive(In1) != isPositive(In2))
    return false;
  if (isPositive(In1))
    return cast<ConstantInt>(Result)->getSExtValue() <
           cast<ConstantInt>(In1)->getSExtValue();
  return cast<ConstantInt>(Result)->getSExtValue() >
         cast<ConstantInt>(In1)->getSExtValue();
}

/// EmitGEPOffset - Given a getelementptr instruction/constantexpr, emit the
/// code necessary to compute the offset from the base pointer (without adding
/// in the base pointer).  Return the result as a signed integer of intptr size.
static Value *EmitGEPOffset(User *GEP, Instruction &I, InstCombiner &IC) {
  TargetData &TD = IC.getTargetData();
  gep_type_iterator GTI = gep_type_begin(GEP);
  const Type *UIntPtrTy = TD.getIntPtrType();
  const Type *SIntPtrTy = UIntPtrTy->getSignedVersion();
  Value *Result = Constant::getNullValue(SIntPtrTy);

  // Build a mask for high order bits.
  uint64_t PtrSizeMask = ~0ULL >> (64-TD.getPointerSize()*8);

  for (unsigned i = 1, e = GEP->getNumOperands(); i != e; ++i, ++GTI) {
    Value *Op = GEP->getOperand(i);
    uint64_t Size = TD.getTypeSize(GTI.getIndexedType()) & PtrSizeMask;
    Constant *Scale = ConstantExpr::getCast(ConstantInt::get(UIntPtrTy, Size),
                                            SIntPtrTy);
    if (Constant *OpC = dyn_cast<Constant>(Op)) {
      if (!OpC->isNullValue()) {
        OpC = ConstantExpr::getCast(OpC, SIntPtrTy);
        Scale = ConstantExpr::getMul(OpC, Scale);
        if (Constant *RC = dyn_cast<Constant>(Result))
          Result = ConstantExpr::getAdd(RC, Scale);
        else {
          // Emit an add instruction.
          Result = IC.InsertNewInstBefore(
             BinaryOperator::createAdd(Result, Scale,
                                       GEP->getName()+".offs"), I);
        }
      }
    } else {
      // Convert to correct type.
      Op = IC.InsertNewInstBefore(new CastInst(Op, SIntPtrTy,
                                               Op->getName()+".c"), I);
      if (Size != 1)
        // We'll let instcombine(mul) convert this to a shl if possible.
        Op = IC.InsertNewInstBefore(BinaryOperator::createMul(Op, Scale,
                                                    GEP->getName()+".idx"), I);

      // Emit an add instruction.
      Result = IC.InsertNewInstBefore(BinaryOperator::createAdd(Op, Result,
                                                    GEP->getName()+".offs"), I);
    }
  }
  return Result;
}

/// FoldGEPSetCC - Fold comparisons between a GEP instruction and something
/// else.  At this point we know that the GEP is on the LHS of the comparison.
Instruction *InstCombiner::FoldGEPSetCC(User *GEPLHS, Value *RHS,
                                        Instruction::BinaryOps Cond,
                                        Instruction &I) {
  assert(dyn_castGetElementPtr(GEPLHS) && "LHS is not a getelementptr!");

  if (CastInst *CI = dyn_cast<CastInst>(RHS))
    if (isa<PointerType>(CI->getOperand(0)->getType()))
      RHS = CI->getOperand(0);

  Value *PtrBase = GEPLHS->getOperand(0);
  if (PtrBase == RHS) {
    // As an optimization, we don't actually have to compute the actual value of
    // OFFSET if this is a seteq or setne comparison, just return whether each
    // index is zero or not.
    if (Cond == Instruction::SetEQ || Cond == Instruction::SetNE) {
      Instruction *InVal = 0;
      gep_type_iterator GTI = gep_type_begin(GEPLHS);
      for (unsigned i = 1, e = GEPLHS->getNumOperands(); i != e; ++i, ++GTI) {
        bool EmitIt = true;
        if (Constant *C = dyn_cast<Constant>(GEPLHS->getOperand(i))) {
          if (isa<UndefValue>(C))  // undef index -> undef.
            return ReplaceInstUsesWith(I, UndefValue::get(I.getType()));
          if (C->isNullValue())
            EmitIt = false;
          else if (TD->getTypeSize(GTI.getIndexedType()) == 0) {
            EmitIt = false;  // This is indexing into a zero sized array?
          } else if (isa<ConstantInt>(C))
            return ReplaceInstUsesWith(I, // No comparison is needed here.
                                 ConstantBool::get(Cond == Instruction::SetNE));
        }

        if (EmitIt) {
          Instruction *Comp =
            new SetCondInst(Cond, GEPLHS->getOperand(i),
                    Constant::getNullValue(GEPLHS->getOperand(i)->getType()));
          if (InVal == 0)
            InVal = Comp;
          else {
            InVal = InsertNewInstBefore(InVal, I);
            InsertNewInstBefore(Comp, I);
            if (Cond == Instruction::SetNE)   // True if any are unequal
              InVal = BinaryOperator::createOr(InVal, Comp);
            else                              // True if all are equal
              InVal = BinaryOperator::createAnd(InVal, Comp);
          }
        }
      }

      if (InVal)
        return InVal;
      else
        ReplaceInstUsesWith(I, // No comparison is needed here, all indexes = 0
                            ConstantBool::get(Cond == Instruction::SetEQ));
    }

    // Only lower this if the setcc is the only user of the GEP or if we expect
    // the result to fold to a constant!
    if (isa<ConstantExpr>(GEPLHS) || GEPLHS->hasOneUse()) {
      // ((gep Ptr, OFFSET) cmp Ptr)   ---> (OFFSET cmp 0).
      Value *Offset = EmitGEPOffset(GEPLHS, I, *this);
      return new SetCondInst(Cond, Offset,
                             Constant::getNullValue(Offset->getType()));
    }
  } else if (User *GEPRHS = dyn_castGetElementPtr(RHS)) {
    // If the base pointers are different, but the indices are the same, just
    // compare the base pointer.
    if (PtrBase != GEPRHS->getOperand(0)) {
      bool IndicesTheSame = GEPLHS->getNumOperands()==GEPRHS->getNumOperands();
      IndicesTheSame &= GEPLHS->getOperand(0)->getType() ==
                        GEPRHS->getOperand(0)->getType();
      if (IndicesTheSame)
        for (unsigned i = 1, e = GEPLHS->getNumOperands(); i != e; ++i)
          if (GEPLHS->getOperand(i) != GEPRHS->getOperand(i)) {
            IndicesTheSame = false;
            break;
          }

      // If all indices are the same, just compare the base pointers.
      if (IndicesTheSame)
        return new SetCondInst(Cond, GEPLHS->getOperand(0),
                               GEPRHS->getOperand(0));

      // Otherwise, the base pointers are different and the indices are
      // different, bail out.
      return 0;
    }

    // If one of the GEPs has all zero indices, recurse.
    bool AllZeros = true;
    for (unsigned i = 1, e = GEPLHS->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEPLHS->getOperand(i)) ||
          !cast<Constant>(GEPLHS->getOperand(i))->isNullValue()) {
        AllZeros = false;
        break;
      }
    if (AllZeros)
      return FoldGEPSetCC(GEPRHS, GEPLHS->getOperand(0),
                          SetCondInst::getSwappedCondition(Cond), I);

    // If the other GEP has all zero indices, recurse.
    AllZeros = true;
    for (unsigned i = 1, e = GEPRHS->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEPRHS->getOperand(i)) ||
          !cast<Constant>(GEPRHS->getOperand(i))->isNullValue()) {
        AllZeros = false;
        break;
      }
    if (AllZeros)
      return FoldGEPSetCC(GEPLHS, GEPRHS->getOperand(0), Cond, I);

    if (GEPLHS->getNumOperands() == GEPRHS->getNumOperands()) {
      // If the GEPs only differ by one index, compare it.
      unsigned NumDifferences = 0;  // Keep track of # differences.
      unsigned DiffOperand = 0;     // The operand that differs.
      for (unsigned i = 1, e = GEPRHS->getNumOperands(); i != e; ++i)
        if (GEPLHS->getOperand(i) != GEPRHS->getOperand(i)) {
          if (GEPLHS->getOperand(i)->getType()->getPrimitiveSizeInBits() !=
                   GEPRHS->getOperand(i)->getType()->getPrimitiveSizeInBits()) {
            // Irreconcilable differences.
            NumDifferences = 2;
            break;
          } else {
            if (NumDifferences++) break;
            DiffOperand = i;
          }
        }

      if (NumDifferences == 0)   // SAME GEP?
        return ReplaceInstUsesWith(I, // No comparison is needed here.
                                 ConstantBool::get(Cond == Instruction::SetEQ));
      else if (NumDifferences == 1) {
        Value *LHSV = GEPLHS->getOperand(DiffOperand);
        Value *RHSV = GEPRHS->getOperand(DiffOperand);

        // Convert the operands to signed values to make sure to perform a
        // signed comparison.
        const Type *NewTy = LHSV->getType()->getSignedVersion();
        if (LHSV->getType() != NewTy)
          LHSV = InsertCastBefore(LHSV, NewTy, I);
        if (RHSV->getType() != NewTy)
          RHSV = InsertCastBefore(RHSV, NewTy, I);
        return new SetCondInst(Cond, LHSV, RHSV);
      }
    }

    // Only lower this if the setcc is the only user of the GEP or if we expect
    // the result to fold to a constant!
    if ((isa<ConstantExpr>(GEPLHS) || GEPLHS->hasOneUse()) &&
        (isa<ConstantExpr>(GEPRHS) || GEPRHS->hasOneUse())) {
      // ((gep Ptr, OFFSET1) cmp (gep Ptr, OFFSET2)  --->  (OFFSET1 cmp OFFSET2)
      Value *L = EmitGEPOffset(GEPLHS, I, *this);
      Value *R = EmitGEPOffset(GEPRHS, I, *this);
      return new SetCondInst(Cond, L, R);
    }
  }
  return 0;
}


Instruction *InstCombiner::visitSetCondInst(SetCondInst &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  const Type *Ty = Op0->getType();

  // setcc X, X
  if (Op0 == Op1)
    return ReplaceInstUsesWith(I, ConstantBool::get(isTrueWhenEqual(I)));

  if (isa<UndefValue>(Op1))                  // X setcc undef -> undef
    return ReplaceInstUsesWith(I, UndefValue::get(Type::BoolTy));

  // setcc <global/alloca*/null>, <global/alloca*/null> - Global/Stack value
  // addresses never equal each other!  We already know that Op0 != Op1.
  if ((isa<GlobalValue>(Op0) || isa<AllocaInst>(Op0) ||
       isa<ConstantPointerNull>(Op0)) &&
      (isa<GlobalValue>(Op1) || isa<AllocaInst>(Op1) ||
       isa<ConstantPointerNull>(Op1)))
    return ReplaceInstUsesWith(I, ConstantBool::get(!isTrueWhenEqual(I)));

  // setcc's with boolean values can always be turned into bitwise operations
  if (Ty == Type::BoolTy) {
    switch (I.getOpcode()) {
    default: assert(0 && "Invalid setcc instruction!");
    case Instruction::SetEQ: {     //  seteq bool %A, %B -> ~(A^B)
      Instruction *Xor = BinaryOperator::createXor(Op0, Op1, I.getName()+"tmp");
      InsertNewInstBefore(Xor, I);
      return BinaryOperator::createNot(Xor);
    }
    case Instruction::SetNE:
      return BinaryOperator::createXor(Op0, Op1);

    case Instruction::SetGT:
      std::swap(Op0, Op1);                   // Change setgt -> setlt
      // FALL THROUGH
    case Instruction::SetLT: {               // setlt bool A, B -> ~X & Y
      Instruction *Not = BinaryOperator::createNot(Op0, I.getName()+"tmp");
      InsertNewInstBefore(Not, I);
      return BinaryOperator::createAnd(Not, Op1);
    }
    case Instruction::SetGE:
      std::swap(Op0, Op1);                   // Change setge -> setle
      // FALL THROUGH
    case Instruction::SetLE: {     //  setle bool %A, %B -> ~A | B
      Instruction *Not = BinaryOperator::createNot(Op0, I.getName()+"tmp");
      InsertNewInstBefore(Not, I);
      return BinaryOperator::createOr(Not, Op1);
    }
    }
  }

  // See if we are doing a comparison between a constant and an instruction that
  // can be folded into the comparison.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    // Check to see if we are comparing against the minimum or maximum value...
    if (CI->isMinValue()) {
      if (I.getOpcode() == Instruction::SetLT)       // A < MIN -> FALSE
        return ReplaceInstUsesWith(I, ConstantBool::getFalse());
      if (I.getOpcode() == Instruction::SetGE)       // A >= MIN -> TRUE
        return ReplaceInstUsesWith(I, ConstantBool::getTrue());
      if (I.getOpcode() == Instruction::SetLE)       // A <= MIN -> A == MIN
        return BinaryOperator::createSetEQ(Op0, Op1);
      if (I.getOpcode() == Instruction::SetGT)       // A > MIN -> A != MIN
        return BinaryOperator::createSetNE(Op0, Op1);

    } else if (CI->isMaxValue()) {
      if (I.getOpcode() == Instruction::SetGT)       // A > MAX -> FALSE
        return ReplaceInstUsesWith(I, ConstantBool::getFalse());
      if (I.getOpcode() == Instruction::SetLE)       // A <= MAX -> TRUE
        return ReplaceInstUsesWith(I, ConstantBool::getTrue());
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

    
    // See if we can fold the comparison based on bits known to be zero or one
    // in the input.
    uint64_t KnownZero, KnownOne;
    if (SimplifyDemandedBits(Op0, Ty->getIntegralTypeMask(),
                             KnownZero, KnownOne, 0))
      return &I;
        
    // Given the known and unknown bits, compute a range that the LHS could be
    // in.
    if (KnownOne | KnownZero) {
      if (Ty->isUnsigned()) {   // Unsigned comparison.
        uint64_t Min, Max;
        uint64_t RHSVal = CI->getZExtValue();
        ComputeUnsignedMinMaxValuesFromKnownBits(Ty, KnownZero, KnownOne,
                                                 Min, Max);
        switch (I.getOpcode()) {  // LE/GE have been folded already.
        default: assert(0 && "Unknown setcc opcode!");
        case Instruction::SetEQ:
          if (Max < RHSVal || Min > RHSVal)
            return ReplaceInstUsesWith(I, ConstantBool::getFalse());
          break;
        case Instruction::SetNE:
          if (Max < RHSVal || Min > RHSVal)
            return ReplaceInstUsesWith(I, ConstantBool::getTrue());
          break;
        case Instruction::SetLT:
          if (Max < RHSVal)
            return ReplaceInstUsesWith(I, ConstantBool::getTrue());
          if (Min > RHSVal)
            return ReplaceInstUsesWith(I, ConstantBool::getFalse());
          break;
        case Instruction::SetGT:
          if (Min > RHSVal)
            return ReplaceInstUsesWith(I, ConstantBool::getTrue());
          if (Max < RHSVal)
            return ReplaceInstUsesWith(I, ConstantBool::getFalse());
          break;
        }
      } else {              // Signed comparison.
        int64_t Min, Max;
        int64_t RHSVal = CI->getSExtValue();
        ComputeSignedMinMaxValuesFromKnownBits(Ty, KnownZero, KnownOne,
                                               Min, Max);
        switch (I.getOpcode()) {  // LE/GE have been folded already.
        default: assert(0 && "Unknown setcc opcode!");
        case Instruction::SetEQ:
          if (Max < RHSVal || Min > RHSVal)
            return ReplaceInstUsesWith(I, ConstantBool::getFalse());
          break;
        case Instruction::SetNE:
          if (Max < RHSVal || Min > RHSVal)
            return ReplaceInstUsesWith(I, ConstantBool::getTrue());
          break;
        case Instruction::SetLT:
          if (Max < RHSVal)
            return ReplaceInstUsesWith(I, ConstantBool::getTrue());
          if (Min > RHSVal)
            return ReplaceInstUsesWith(I, ConstantBool::getFalse());
          break;
        case Instruction::SetGT:
          if (Min > RHSVal)
            return ReplaceInstUsesWith(I, ConstantBool::getTrue());
          if (Max < RHSVal)
            return ReplaceInstUsesWith(I, ConstantBool::getFalse());
          break;
        }
      }
    }
          
    // Since the RHS is a constantInt (CI), if the left hand side is an 
    // instruction, see if that instruction also has constants so that the 
    // instruction can be folded into the setcc
    if (Instruction *LHSI = dyn_cast<Instruction>(Op0))
      switch (LHSI->getOpcode()) {
      case Instruction::And:
        if (LHSI->hasOneUse() && isa<ConstantInt>(LHSI->getOperand(1)) &&
            LHSI->getOperand(0)->hasOneUse()) {
          ConstantInt *AndCST = cast<ConstantInt>(LHSI->getOperand(1));

          // If an operand is an AND of a truncating cast, we can widen the
          // and/compare to be the input width without changing the value
          // produced, eliminating a cast.
          if (CastInst *Cast = dyn_cast<CastInst>(LHSI->getOperand(0))) {
            // We can do this transformation if either the AND constant does not
            // have its sign bit set or if it is an equality comparison. 
            // Extending a relational comparison when we're checking the sign
            // bit would not work.
            if (Cast->hasOneUse() && Cast->isTruncIntCast() && 
                (I.isEquality() ||
                 (AndCST->getZExtValue() == (uint64_t)AndCST->getSExtValue()) &&
                 (CI->getZExtValue() == (uint64_t)CI->getSExtValue()))) {
              ConstantInt *NewCST;
              ConstantInt *NewCI;
              if (Cast->getOperand(0)->getType()->isSigned()) {
                NewCST = ConstantInt::get(Cast->getOperand(0)->getType(),
                                           AndCST->getZExtValue());
                NewCI = ConstantInt::get(Cast->getOperand(0)->getType(),
                                          CI->getZExtValue());
              } else {
                NewCST = ConstantInt::get(Cast->getOperand(0)->getType(),
                                           AndCST->getZExtValue());
                NewCI = ConstantInt::get(Cast->getOperand(0)->getType(),
                                          CI->getZExtValue());
              }
              Instruction *NewAnd = 
                BinaryOperator::createAnd(Cast->getOperand(0), NewCST, 
                                          LHSI->getName());
              InsertNewInstBefore(NewAnd, I);
              return new SetCondInst(I.getOpcode(), NewAnd, NewCI);
            }
          }
          
          // If this is: (X >> C1) & C2 != C3 (where any shift and any compare
          // could exist), turn it into (X & (C2 << C1)) != (C3 << C1).  This
          // happens a LOT in code produced by the C front-end, for bitfield
          // access.
          ShiftInst *Shift = dyn_cast<ShiftInst>(LHSI->getOperand(0));

          // Check to see if there is a noop-cast between the shift and the and.
          if (!Shift) {
            if (CastInst *CI = dyn_cast<CastInst>(LHSI->getOperand(0)))
              if (CI->getOperand(0)->getType()->isIntegral() &&
                  CI->getOperand(0)->getType()->getPrimitiveSizeInBits() ==
                     CI->getType()->getPrimitiveSizeInBits())
                Shift = dyn_cast<ShiftInst>(CI->getOperand(0));
          }

          ConstantInt *ShAmt;
          ShAmt = Shift ? dyn_cast<ConstantInt>(Shift->getOperand(1)) : 0;
          const Type *Ty = Shift ? Shift->getType() : 0;  // Type of the shift.
          const Type *AndTy = AndCST->getType();          // Type of the and.

          // We can fold this as long as we can't shift unknown bits
          // into the mask.  This can only happen with signed shift
          // rights, as they sign-extend.
          if (ShAmt) {
            bool CanFold = Shift->isLogicalShift();
            if (!CanFold) {
              // To test for the bad case of the signed shr, see if any
              // of the bits shifted in could be tested after the mask.
              int ShAmtVal = Ty->getPrimitiveSizeInBits()-ShAmt->getZExtValue();
              if (ShAmtVal < 0) ShAmtVal = 0; // Out of range shift.

              Constant *OShAmt = ConstantInt::get(Type::UByteTy, ShAmtVal);
              Constant *ShVal =
                ConstantExpr::getShl(ConstantInt::getAllOnesValue(AndTy), 
                                     OShAmt);
              if (ConstantExpr::getAnd(ShVal, AndCST)->isNullValue())
                CanFold = true;
            }

            if (CanFold) {
              Constant *NewCst;
              if (Shift->getOpcode() == Instruction::Shl)
                NewCst = ConstantExpr::getUShr(CI, ShAmt);
              else
                NewCst = ConstantExpr::getShl(CI, ShAmt);

              // Check to see if we are shifting out any of the bits being
              // compared.
              if (ConstantExpr::get(Shift->getOpcode(), NewCst, ShAmt) != CI){
                // If we shifted bits out, the fold is not going to work out.
                // As a special case, check to see if this means that the
                // result is always true or false now.
                if (I.getOpcode() == Instruction::SetEQ)
                  return ReplaceInstUsesWith(I, ConstantBool::getFalse());
                if (I.getOpcode() == Instruction::SetNE)
                  return ReplaceInstUsesWith(I, ConstantBool::getTrue());
              } else {
                I.setOperand(1, NewCst);
                Constant *NewAndCST;
                if (Shift->getOpcode() == Instruction::Shl)
                  NewAndCST = ConstantExpr::getUShr(AndCST, ShAmt);
                else
                  NewAndCST = ConstantExpr::getShl(AndCST, ShAmt);
                LHSI->setOperand(1, NewAndCST);
                if (AndTy == Ty) 
                  LHSI->setOperand(0, Shift->getOperand(0));
                else {
                  Value *NewCast = InsertCastBefore(Shift->getOperand(0), AndTy,
                                                    *Shift);
                  LHSI->setOperand(0, NewCast);
                }
                WorkList.push_back(Shift); // Shift is dead.
                AddUsesToWorkList(I);
                return &I;
              }
            }
          }
          
          // Turn ((X >> Y) & C) == 0  into  (X & (C << Y)) == 0.  The later is
          // preferable because it allows the C<<Y expression to be hoisted out
          // of a loop if Y is invariant and X is not.
          if (Shift && Shift->hasOneUse() && CI->isNullValue() &&
              I.isEquality() && !Shift->isArithmeticShift() &&
              isa<Instruction>(Shift->getOperand(0))) {
            // Compute C << Y.
            Value *NS;
            if (Shift->getOpcode() == Instruction::Shr) {
              NS = new ShiftInst(Instruction::Shl, AndCST, Shift->getOperand(1),
                                 "tmp");
            } else {
              // Make sure we insert a logical shift.
              Constant *NewAndCST = AndCST;
              if (AndCST->getType()->isSigned())
                NewAndCST = ConstantExpr::getCast(AndCST,
                                      AndCST->getType()->getUnsignedVersion());
              NS = new ShiftInst(Instruction::Shr, NewAndCST,
                                 Shift->getOperand(1), "tmp");
            }
            InsertNewInstBefore(cast<Instruction>(NS), I);

            // If C's sign doesn't agree with the and, insert a cast now.
            if (NS->getType() != LHSI->getType())
              NS = InsertCastBefore(NS, LHSI->getType(), I);

            Value *ShiftOp = Shift->getOperand(0);
            if (ShiftOp->getType() != LHSI->getType())
              ShiftOp = InsertCastBefore(ShiftOp, LHSI->getType(), I);
              
            // Compute X & (C << Y).
            Instruction *NewAnd =
              BinaryOperator::createAnd(ShiftOp, NS, LHSI->getName());
            InsertNewInstBefore(NewAnd, I);
            
            I.setOperand(0, NewAnd);
            return &I;
          }
        }
        break;

      case Instruction::Shl:         // (setcc (shl X, ShAmt), CI)
        if (ConstantInt *ShAmt = dyn_cast<ConstantInt>(LHSI->getOperand(1))) {
          if (I.isEquality()) {
            unsigned TypeBits = CI->getType()->getPrimitiveSizeInBits();

            // Check that the shift amount is in range.  If not, don't perform
            // undefined shifts.  When the shift is visited it will be
            // simplified.
            if (ShAmt->getZExtValue() >= TypeBits)
              break;

            // If we are comparing against bits always shifted out, the
            // comparison cannot succeed.
            Constant *Comp =
              ConstantExpr::getShl(ConstantExpr::getShr(CI, ShAmt), ShAmt);
            if (Comp != CI) {// Comparing against a bit that we know is zero.
              bool IsSetNE = I.getOpcode() == Instruction::SetNE;
              Constant *Cst = ConstantBool::get(IsSetNE);
              return ReplaceInstUsesWith(I, Cst);
            }

            if (LHSI->hasOneUse()) {
              // Otherwise strength reduce the shift into an and.
              unsigned ShAmtVal = (unsigned)ShAmt->getZExtValue();
              uint64_t Val = (1ULL << (TypeBits-ShAmtVal))-1;

              Constant *Mask;
              if (CI->getType()->isUnsigned()) {
                Mask = ConstantInt::get(CI->getType(), Val);
              } else if (ShAmtVal != 0) {
                Mask = ConstantInt::get(CI->getType(), Val);
              } else {
                Mask = ConstantInt::getAllOnesValue(CI->getType());
              }

              Instruction *AndI =
                BinaryOperator::createAnd(LHSI->getOperand(0),
                                          Mask, LHSI->getName()+".mask");
              Value *And = InsertNewInstBefore(AndI, I);
              return new SetCondInst(I.getOpcode(), And,
                                     ConstantExpr::getUShr(CI, ShAmt));
            }
          }
        }
        break;

      case Instruction::Shr:         // (setcc (shr X, ShAmt), CI)
        if (ConstantInt *ShAmt = dyn_cast<ConstantInt>(LHSI->getOperand(1))) {
          if (I.isEquality()) {
            // Check that the shift amount is in range.  If not, don't perform
            // undefined shifts.  When the shift is visited it will be
            // simplified.
            unsigned TypeBits = CI->getType()->getPrimitiveSizeInBits();
            if (ShAmt->getZExtValue() >= TypeBits)
              break;

            // If we are comparing against bits always shifted out, the
            // comparison cannot succeed.
            Constant *Comp =
              ConstantExpr::getShr(ConstantExpr::getShl(CI, ShAmt), ShAmt);

            if (Comp != CI) {// Comparing against a bit that we know is zero.
              bool IsSetNE = I.getOpcode() == Instruction::SetNE;
              Constant *Cst = ConstantBool::get(IsSetNE);
              return ReplaceInstUsesWith(I, Cst);
            }

            if (LHSI->hasOneUse() || CI->isNullValue()) {
              unsigned ShAmtVal = (unsigned)ShAmt->getZExtValue();

              // Otherwise strength reduce the shift into an and.
              uint64_t Val = ~0ULL;          // All ones.
              Val <<= ShAmtVal;              // Shift over to the right spot.

              Constant *Mask;
              if (CI->getType()->isUnsigned()) {
                Val &= ~0ULL >> (64-TypeBits);
                Mask = ConstantInt::get(CI->getType(), Val);
              } else {
                Mask = ConstantInt::get(CI->getType(), Val);
              }

              Instruction *AndI =
                BinaryOperator::createAnd(LHSI->getOperand(0),
                                          Mask, LHSI->getName()+".mask");
              Value *And = InsertNewInstBefore(AndI, I);
              return new SetCondInst(I.getOpcode(), And,
                                     ConstantExpr::getShl(CI, ShAmt));
            }
          }
        }
        break;

      case Instruction::SDiv:
      case Instruction::UDiv:
        // Fold: setcc ([us]div X, C1), C2 -> range test
        // Fold this div into the comparison, producing a range check. 
        // Determine, based on the divide type, what the range is being 
        // checked.  If there is an overflow on the low or high side, remember 
        // it, otherwise compute the range [low, hi) bounding the new value.
        // See: InsertRangeTest above for the kinds of replacements possible.
        if (ConstantInt *DivRHS = dyn_cast<ConstantInt>(LHSI->getOperand(1))) {
          // FIXME: If the operand types don't match the type of the divide 
          // then don't attempt this transform. The code below doesn't have the
          // logic to deal with a signed divide and an unsigned compare (and
          // vice versa). This is because (x /s C1) <s C2  produces different 
          // results than (x /s C1) <u C2 or (x /u C1) <s C2 or even
          // (x /u C1) <u C2.  Simply casting the operands and result won't 
          // work. :(  The if statement below tests that condition and bails 
          // if it finds it. 
          const Type* DivRHSTy = DivRHS->getType();
          unsigned DivOpCode = LHSI->getOpcode();
          if (I.isEquality() &&
              ((DivOpCode == Instruction::SDiv && DivRHSTy->isUnsigned()) ||
               (DivOpCode == Instruction::UDiv && DivRHSTy->isSigned())))
            break;

          // Initialize the variables that will indicate the nature of the
          // range check.
          bool LoOverflow = false, HiOverflow = false;
          ConstantInt *LoBound = 0, *HiBound = 0;

          // Compute Prod = CI * DivRHS. We are essentially solving an equation
          // of form X/C1=C2. We solve for X by multiplying C1 (DivRHS) and 
          // C2 (CI). By solving for X we can turn this into a range check 
          // instead of computing a divide. 
          ConstantInt *Prod = 
            cast<ConstantInt>(ConstantExpr::getMul(CI, DivRHS));

          // Determine if the product overflows by seeing if the product is
          // not equal to the divide. Make sure we do the same kind of divide
          // as in the LHS instruction that we're folding. 
          bool ProdOV = !DivRHS->isNullValue() && 
            (DivOpCode == Instruction::SDiv ?  
             ConstantExpr::getSDiv(Prod, DivRHS) :
              ConstantExpr::getUDiv(Prod, DivRHS)) != CI;

          // Get the SetCC opcode
          Instruction::BinaryOps Opcode = I.getOpcode();

          if (DivRHS->isNullValue()) {  
            // Don't hack on divide by zeros!
          } else if (DivOpCode == Instruction::UDiv) {  // udiv
            LoBound = Prod;
            LoOverflow = ProdOV;
            HiOverflow = ProdOV || AddWithOverflow(HiBound, LoBound, DivRHS);
          } else if (isPositive(DivRHS)) { // Divisor is > 0.
            if (CI->isNullValue()) {       // (X / pos) op 0
              // Can't overflow.
              LoBound = cast<ConstantInt>(ConstantExpr::getNeg(SubOne(DivRHS)));
              HiBound = DivRHS;
            } else if (isPositive(CI)) {   // (X / pos) op pos
              LoBound = Prod;
              LoOverflow = ProdOV;
              HiOverflow = ProdOV || AddWithOverflow(HiBound, Prod, DivRHS);
            } else {                       // (X / pos) op neg
              Constant *DivRHSH = ConstantExpr::getNeg(SubOne(DivRHS));
              LoOverflow = AddWithOverflow(LoBound, Prod,
                                           cast<ConstantInt>(DivRHSH));
              HiBound = Prod;
              HiOverflow = ProdOV;
            }
          } else {                         // Divisor is < 0.
            if (CI->isNullValue()) {       // (X / neg) op 0
              LoBound = AddOne(DivRHS);
              HiBound = cast<ConstantInt>(ConstantExpr::getNeg(DivRHS));
              if (HiBound == DivRHS)
                LoBound = 0;               // - INTMIN = INTMIN
            } else if (isPositive(CI)) {   // (X / neg) op pos
              HiOverflow = LoOverflow = ProdOV;
              if (!LoOverflow)
                LoOverflow = AddWithOverflow(LoBound, Prod, AddOne(DivRHS));
              HiBound = AddOne(Prod);
            } else {                       // (X / neg) op neg
              LoBound = Prod;
              LoOverflow = HiOverflow = ProdOV;
              HiBound = cast<ConstantInt>(ConstantExpr::getSub(Prod, DivRHS));
            }

            // Dividing by a negate swaps the condition.
            Opcode = SetCondInst::getSwappedCondition(Opcode);
          }

          if (LoBound) {
            Value *X = LHSI->getOperand(0);
            switch (Opcode) {
            default: assert(0 && "Unhandled setcc opcode!");
            case Instruction::SetEQ:
              if (LoOverflow && HiOverflow)
                return ReplaceInstUsesWith(I, ConstantBool::getFalse());
              else if (HiOverflow)
                return new SetCondInst(Instruction::SetGE, X, LoBound);
              else if (LoOverflow)
                return new SetCondInst(Instruction::SetLT, X, HiBound);
              else
                return InsertRangeTest(X, LoBound, HiBound, true, I);
            case Instruction::SetNE:
              if (LoOverflow && HiOverflow)
                return ReplaceInstUsesWith(I, ConstantBool::getTrue());
              else if (HiOverflow)
                return new SetCondInst(Instruction::SetLT, X, LoBound);
              else if (LoOverflow)
                return new SetCondInst(Instruction::SetGE, X, HiBound);
              else
                return InsertRangeTest(X, LoBound, HiBound, false, I);
            case Instruction::SetLT:
              if (LoOverflow)
                return ReplaceInstUsesWith(I, ConstantBool::getFalse());
              return new SetCondInst(Instruction::SetLT, X, LoBound);
            case Instruction::SetGT:
              if (HiOverflow)
                return ReplaceInstUsesWith(I, ConstantBool::getFalse());
              return new SetCondInst(Instruction::SetGE, X, HiBound);
            }
          }
        }
        break;
      }

    // Simplify seteq and setne instructions...
    if (I.isEquality()) {
      bool isSetNE = I.getOpcode() == Instruction::SetNE;

      // If the first operand is (add|sub|and|or|xor|rem) with a constant, and 
      // the second operand is a constant, simplify a bit.
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0)) {
        switch (BO->getOpcode()) {
        case Instruction::SRem:
          // If we have a signed (X % (2^c)) == 0, turn it into an unsigned one.
          if (CI->isNullValue() && isa<ConstantInt>(BO->getOperand(1)) &&
              BO->hasOneUse()) {
            int64_t V = cast<ConstantInt>(BO->getOperand(1))->getSExtValue();
            if (V > 1 && isPowerOf2_64(V)) {
              Value *NewRem = InsertNewInstBefore(BinaryOperator::createURem(
                  BO->getOperand(0), BO->getOperand(1), BO->getName()), I);
              return BinaryOperator::create(I.getOpcode(), NewRem,
                Constant::getNullValue(BO->getType()));
            }
          }
          break;
        case Instruction::Add:
          // Replace ((add A, B) != C) with (A != C-B) if B & C are constants.
          if (ConstantInt *BOp1C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
            if (BO->hasOneUse())
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
                X = InsertCastBefore(X, DestTy, I);
              }
              return new SetCondInst(isSetNE ? Instruction::SetLT :
                                         Instruction::SetGE, X,
                                     Constant::getNullValue(X->getType()));
            }

            // ((X & ~7) == 0) --> X < 8
            if (CI->isNullValue() && isHighOnes(BOC)) {
              Value *X = BO->getOperand(0);
              Constant *NegX = ConstantExpr::getNeg(BOC);

              // If 'X' is signed, insert a cast now.
              if (NegX->getType()->isSigned()) {
                const Type *DestTy = NegX->getType()->getUnsignedVersion();
                X = InsertCastBefore(X, DestTy, I);
                NegX = ConstantExpr::getCast(NegX, DestTy);
              }

              return new SetCondInst(isSetNE ? Instruction::SetGE :
                                     Instruction::SetLT, X, NegX);
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
        unsigned SrcTySize = SrcTy->getPrimitiveSizeInBits();
        if (SrcTy != Cast->getType() && SrcTy->isInteger() &&
            SrcTySize == Cast->getType()->getPrimitiveSizeInBits()) {
          assert((SrcTy->isSigned() ^ Cast->getType()->isSigned()) &&
                 "Source and destination signednesses should differ!");
          if (Cast->getType()->isSigned()) {
            // If this is a signed comparison, check for comparisons in the
            // vicinity of zero.
            if (I.getOpcode() == Instruction::SetLT && CI->isNullValue())
              // X < 0  => x > 127
              return BinaryOperator::createSetGT(CastOp,
                         ConstantInt::get(SrcTy, (1ULL << (SrcTySize-1))-1));
            else if (I.getOpcode() == Instruction::SetGT &&
                     cast<ConstantInt>(CI)->getSExtValue() == -1)
              // X > -1  => x < 128
              return BinaryOperator::createSetLT(CastOp,
                         ConstantInt::get(SrcTy, 1ULL << (SrcTySize-1)));
          } else {
            ConstantInt *CUI = cast<ConstantInt>(CI);
            if (I.getOpcode() == Instruction::SetLT &&
                CUI->getZExtValue() == 1ULL << (SrcTySize-1))
              // X < 128 => X > -1
              return BinaryOperator::createSetGT(CastOp,
                                                 ConstantInt::get(SrcTy, -1));
            else if (I.getOpcode() == Instruction::SetGT &&
                     CUI->getZExtValue() == (1ULL << (SrcTySize-1))-1)
              // X > 127 => X < 0
              return BinaryOperator::createSetLT(CastOp,
                                                 Constant::getNullValue(SrcTy));
          }
        }
      }
    }
  }

  // Handle setcc with constant RHS's that can be integer, FP or pointer.
  if (Constant *RHSC = dyn_cast<Constant>(Op1)) {
    if (Instruction *LHSI = dyn_cast<Instruction>(Op0))
      switch (LHSI->getOpcode()) {
      case Instruction::GetElementPtr:
        if (RHSC->isNullValue()) {
          // Transform setcc GEP P, int 0, int 0, int 0, null -> setcc P, null
          bool isAllZeros = true;
          for (unsigned i = 1, e = LHSI->getNumOperands(); i != e; ++i)
            if (!isa<Constant>(LHSI->getOperand(i)) ||
                !cast<Constant>(LHSI->getOperand(i))->isNullValue()) {
              isAllZeros = false;
              break;
            }
          if (isAllZeros)
            return new SetCondInst(I.getOpcode(), LHSI->getOperand(0),
                    Constant::getNullValue(LHSI->getOperand(0)->getType()));
        }
        break;

      case Instruction::PHI:
        if (Instruction *NV = FoldOpIntoPhi(I))
          return NV;
        break;
      case Instruction::Select:
        // If either operand of the select is a constant, we can fold the
        // comparison into the select arms, which will cause one to be
        // constant folded and the select turned into a bitwise or.
        Value *Op1 = 0, *Op2 = 0;
        if (LHSI->hasOneUse()) {
          if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(1))) {
            // Fold the known value into the constant operand.
            Op1 = ConstantExpr::get(I.getOpcode(), C, RHSC);
            // Insert a new SetCC of the other select operand.
            Op2 = InsertNewInstBefore(new SetCondInst(I.getOpcode(),
                                                      LHSI->getOperand(2), RHSC,
                                                      I.getName()), I);
          } else if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(2))) {
            // Fold the known value into the constant operand.
            Op2 = ConstantExpr::get(I.getOpcode(), C, RHSC);
            // Insert a new SetCC of the other select operand.
            Op1 = InsertNewInstBefore(new SetCondInst(I.getOpcode(),
                                                      LHSI->getOperand(1), RHSC,
                                                      I.getName()), I);
          }
        }

        if (Op1)
          return new SelectInst(LHSI->getOperand(0), Op1, Op2);
        break;
      }
  }

  // If we can optimize a 'setcc GEP, P' or 'setcc P, GEP', do so now.
  if (User *GEP = dyn_castGetElementPtr(Op0))
    if (Instruction *NI = FoldGEPSetCC(GEP, Op1, I.getOpcode(), I))
      return NI;
  if (User *GEP = dyn_castGetElementPtr(Op1))
    if (Instruction *NI = FoldGEPSetCC(GEP, Op0,
                           SetCondInst::getSwappedCondition(I.getOpcode()), I))
      return NI;

  // Test to see if the operands of the setcc are casted versions of other
  // values.  If the cast can be stripped off both arguments, we do so now.
  if (CastInst *CI = dyn_cast<CastInst>(Op0)) {
    Value *CastOp0 = CI->getOperand(0);
    if (CastOp0->getType()->isLosslesslyConvertibleTo(CI->getType()) &&
        (isa<Constant>(Op1) || isa<CastInst>(Op1)) && I.isEquality()) {
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
          Op1 = InsertCastBefore(Op1, Op0->getType(), I);
        }
      return BinaryOperator::create(I.getOpcode(), Op0, Op1);
    }

    // Handle the special case of: setcc (cast bool to X), <cst>
    // This comes up when you have code like
    //   int X = A < B;
    //   if (X) ...
    // For generality, we handle any zero-extension of any operand comparison
    // with a constant or another cast from the same type.
    if (isa<ConstantInt>(Op1) || isa<CastInst>(Op1))
      if (Instruction *R = visitSetCondInstWithCastAndCast(I))
        return R;
  }
  
  if (I.isEquality()) {
    Value *A, *B;
    if (match(Op0, m_Xor(m_Value(A), m_Value(B))) &&
        (A == Op1 || B == Op1)) {
      // (A^B) == A  ->  B == 0
      Value *OtherVal = A == Op1 ? B : A;
      return BinaryOperator::create(I.getOpcode(), OtherVal,
                                    Constant::getNullValue(A->getType()));
    } else if (match(Op1, m_Xor(m_Value(A), m_Value(B))) &&
               (A == Op0 || B == Op0)) {
      // A == (A^B)  ->  B == 0
      Value *OtherVal = A == Op0 ? B : A;
      return BinaryOperator::create(I.getOpcode(), OtherVal,
                                    Constant::getNullValue(A->getType()));
    } else if (match(Op0, m_Sub(m_Value(A), m_Value(B))) && A == Op1) {
      // (A-B) == A  ->  B == 0
      return BinaryOperator::create(I.getOpcode(), B,
                                    Constant::getNullValue(B->getType()));
    } else if (match(Op1, m_Sub(m_Value(A), m_Value(B))) && A == Op0) {
      // A == (A-B)  ->  B == 0
      return BinaryOperator::create(I.getOpcode(), B,
                                    Constant::getNullValue(B->getType()));
    }
  }
  return Changed ? &I : 0;
}

// visitSetCondInstWithCastAndCast - Handle setcond (cast x to y), (cast/cst).
// We only handle extending casts so far.
//
Instruction *InstCombiner::visitSetCondInstWithCastAndCast(SetCondInst &SCI) {
  Value *LHSCIOp = cast<CastInst>(SCI.getOperand(0))->getOperand(0);
  const Type *SrcTy = LHSCIOp->getType();
  const Type *DestTy = SCI.getOperand(0)->getType();
  Value *RHSCIOp;

  if (!DestTy->isIntegral() || !SrcTy->isIntegral())
    return 0;

  unsigned SrcBits  = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBits = DestTy->getPrimitiveSizeInBits();
  if (SrcBits >= DestBits) return 0;  // Only handle extending cast.

  // Is this a sign or zero extension?
  bool isSignSrc  = SrcTy->isSigned();
  bool isSignDest = DestTy->isSigned();

  if (CastInst *CI = dyn_cast<CastInst>(SCI.getOperand(1))) {
    // Not an extension from the same type?
    RHSCIOp = CI->getOperand(0);
    if (RHSCIOp->getType() != LHSCIOp->getType()) return 0;
  } else if (ConstantInt *CI = dyn_cast<ConstantInt>(SCI.getOperand(1))) {
    // Compute the constant that would happen if we truncated to SrcTy then
    // reextended to DestTy.
    Constant *Res = ConstantExpr::getCast(CI, SrcTy);

    if (ConstantExpr::getCast(Res, DestTy) == CI) {
      // Make sure that src sign and dest sign match. For example,
      //
      // %A = cast short %X to uint
      // %B = setgt uint %A, 1330
      //
      // It is incorrect to transform this into 
      //
      // %B = setgt short %X, 1330 
      // 
      // because %A may have negative value. 
      // However, it is OK if SrcTy is bool (See cast-set.ll testcase)
      // OR operation is EQ/NE.
      if (isSignSrc == isSignDest || SrcTy == Type::BoolTy || SCI.isEquality())
        RHSCIOp = Res;
      else
        return 0;
    } else {
      // If the value cannot be represented in the shorter type, we cannot emit
      // a simple comparison.
      if (SCI.getOpcode() == Instruction::SetEQ)
        return ReplaceInstUsesWith(SCI, ConstantBool::getFalse());
      if (SCI.getOpcode() == Instruction::SetNE)
        return ReplaceInstUsesWith(SCI, ConstantBool::getTrue());

      // Evaluate the comparison for LT.
      Value *Result;
      if (DestTy->isSigned()) {
        // We're performing a signed comparison.
        if (isSignSrc) {
          // Signed extend and signed comparison.
          if (cast<ConstantInt>(CI)->getSExtValue() < 0)// X < (small) --> false
            Result = ConstantBool::getFalse();
          else
            Result = ConstantBool::getTrue();           // X < (large) --> true
        } else {
          // Unsigned extend and signed comparison.
          if (cast<ConstantInt>(CI)->getSExtValue() < 0)
            Result = ConstantBool::getFalse();
          else
            Result = ConstantBool::getTrue();
        }
      } else {
        // We're performing an unsigned comparison.
        if (!isSignSrc) {
          // Unsigned extend & compare -> always true.
          Result = ConstantBool::getTrue();
        } else {
          // We're performing an unsigned comp with a sign extended value.
          // This is true if the input is >= 0. [aka >s -1]
          Constant *NegOne = ConstantIntegral::getAllOnesValue(SrcTy);
          Result = InsertNewInstBefore(BinaryOperator::createSetGT(LHSCIOp,
                                                  NegOne, SCI.getName()), SCI);
        }
      }

      // Finally, return the value computed.
      if (SCI.getOpcode() == Instruction::SetLT) {
        return ReplaceInstUsesWith(SCI, Result);
      } else {
        assert(SCI.getOpcode()==Instruction::SetGT &&"SetCC should be folded!");
        if (Constant *CI = dyn_cast<Constant>(Result))
          return ReplaceInstUsesWith(SCI, ConstantExpr::getNot(CI));
        else
          return BinaryOperator::createNot(Result);
      }
    }
  } else {
    return 0;
  }

  // Okay, just insert a compare of the reduced operands now!
  return BinaryOperator::create(SCI.getOpcode(), LHSCIOp, RHSCIOp);
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
  
  if (isa<UndefValue>(Op0)) {            // undef >>s X -> undef
    if (!isLeftShift && I.getType()->isSigned())
      return ReplaceInstUsesWith(I, Op0);
    else                         // undef << X -> 0   AND  undef >>u X -> 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }
  if (isa<UndefValue>(Op1)) {
    if (isLeftShift || I.getType()->isUnsigned())// X << undef, X >>u undef -> 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
    else
      return ReplaceInstUsesWith(I, Op0);          // X >>s undef -> X
  }

  // shr int -1, X = -1   (for any arithmetic shift rights of ~0)
  if (!isLeftShift)
    if (ConstantInt *CSI = dyn_cast<ConstantInt>(Op0))
      if (CSI->isAllOnesValue() && Op0->getType()->isSigned())
        return ReplaceInstUsesWith(I, CSI);

  // Try to fold constant and into select arguments.
  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;

  // See if we can turn a signed shr into an unsigned shr.
  if (I.isArithmeticShift()) {
    if (MaskedValueIsZero(Op0,
                          1ULL << (I.getType()->getPrimitiveSizeInBits()-1))) {
      Value *V = InsertCastBefore(Op0, I.getType()->getUnsignedVersion(), I);
      V = InsertNewInstBefore(new ShiftInst(Instruction::Shr, V, Op1,
                                            I.getName()), I);
      return new CastInst(V, I.getType());
    }
  }

  if (ConstantInt *CUI = dyn_cast<ConstantInt>(Op1))
    if (CUI->getType()->isUnsigned())
      if (Instruction *Res = FoldShiftByConstant(Op0, CUI, I))
        return Res;
  return 0;
}

Instruction *InstCombiner::FoldShiftByConstant(Value *Op0, ConstantInt *Op1,
                                               ShiftInst &I) {
  bool isLeftShift = I.getOpcode() == Instruction::Shl;
  bool isSignedShift = Op0->getType()->isSigned();
  bool isUnsignedShift = !isSignedShift;

  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  uint64_t KnownZero, KnownOne;
  if (SimplifyDemandedBits(&I, I.getType()->getIntegralTypeMask(),
                           KnownZero, KnownOne))
    return &I;
  
  // shl uint X, 32 = 0 and shr ubyte Y, 9 = 0, ... just don't eliminate shr
  // of a signed value.
  //
  unsigned TypeBits = Op0->getType()->getPrimitiveSizeInBits();
  if (Op1->getZExtValue() >= TypeBits) {
    if (isUnsignedShift || isLeftShift)
      return ReplaceInstUsesWith(I, Constant::getNullValue(Op0->getType()));
    else {
      I.setOperand(1, ConstantInt::get(Type::UByteTy, TypeBits-1));
      return &I;
    }
  }
  
  // ((X*C1) << C2) == (X * (C1 << C2))
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0))
    if (BO->getOpcode() == Instruction::Mul && isLeftShift)
      if (Constant *BOOp = dyn_cast<Constant>(BO->getOperand(1)))
        return BinaryOperator::createMul(BO->getOperand(0),
                                         ConstantExpr::getShl(BOOp, Op1));
  
  // Try to fold constant and into select arguments.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
    if (Instruction *R = FoldOpIntoSelect(I, SI, this))
      return R;
  if (isa<PHINode>(Op0))
    if (Instruction *NV = FoldOpIntoPhi(I))
      return NV;
  
  if (Op0->hasOneUse()) {
    if (BinaryOperator *Op0BO = dyn_cast<BinaryOperator>(Op0)) {
      // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
      Value *V1, *V2;
      ConstantInt *CC;
      switch (Op0BO->getOpcode()) {
        default: break;
        case Instruction::Add:
        case Instruction::And:
        case Instruction::Or:
        case Instruction::Xor:
          // These operators commute.
          // Turn (Y + (X >> C)) << C  ->  (X + (Y << C)) & (~0 << C)
          if (isLeftShift && Op0BO->getOperand(1)->hasOneUse() &&
              match(Op0BO->getOperand(1),
                    m_Shr(m_Value(V1), m_ConstantInt(CC))) && CC == Op1) {
            Instruction *YS = new ShiftInst(Instruction::Shl, 
                                            Op0BO->getOperand(0), Op1,
                                            Op0BO->getName());
            InsertNewInstBefore(YS, I); // (Y << C)
            Instruction *X = 
              BinaryOperator::create(Op0BO->getOpcode(), YS, V1,
                                     Op0BO->getOperand(1)->getName());
            InsertNewInstBefore(X, I);  // (X + (Y << C))
            Constant *C2 = ConstantInt::getAllOnesValue(X->getType());
            C2 = ConstantExpr::getShl(C2, Op1);
            return BinaryOperator::createAnd(X, C2);
          }
          
          // Turn (Y + ((X >> C) & CC)) << C  ->  ((X & (CC << C)) + (Y << C))
          if (isLeftShift && Op0BO->getOperand(1)->hasOneUse() &&
              match(Op0BO->getOperand(1),
                    m_And(m_Shr(m_Value(V1), m_Value(V2)),
                          m_ConstantInt(CC))) && V2 == Op1 &&
      cast<BinaryOperator>(Op0BO->getOperand(1))->getOperand(0)->hasOneUse()) {
            Instruction *YS = new ShiftInst(Instruction::Shl, 
                                            Op0BO->getOperand(0), Op1,
                                            Op0BO->getName());
            InsertNewInstBefore(YS, I); // (Y << C)
            Instruction *XM =
              BinaryOperator::createAnd(V1, ConstantExpr::getShl(CC, Op1),
                                        V1->getName()+".mask");
            InsertNewInstBefore(XM, I); // X & (CC << C)
            
            return BinaryOperator::create(Op0BO->getOpcode(), YS, XM);
          }
          
          // FALL THROUGH.
        case Instruction::Sub:
          // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
          if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
              match(Op0BO->getOperand(0),
                    m_Shr(m_Value(V1), m_ConstantInt(CC))) && CC == Op1) {
            Instruction *YS = new ShiftInst(Instruction::Shl, 
                                            Op0BO->getOperand(1), Op1,
                                            Op0BO->getName());
            InsertNewInstBefore(YS, I); // (Y << C)
            Instruction *X =
              BinaryOperator::create(Op0BO->getOpcode(), V1, YS,
                                     Op0BO->getOperand(0)->getName());
            InsertNewInstBefore(X, I);  // (X + (Y << C))
            Constant *C2 = ConstantInt::getAllOnesValue(X->getType());
            C2 = ConstantExpr::getShl(C2, Op1);
            return BinaryOperator::createAnd(X, C2);
          }
          
          // Turn (((X >> C)&CC) + Y) << C  ->  (X + (Y << C)) & (CC << C)
          if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
              match(Op0BO->getOperand(0),
                    m_And(m_Shr(m_Value(V1), m_Value(V2)),
                          m_ConstantInt(CC))) && V2 == Op1 &&
              cast<BinaryOperator>(Op0BO->getOperand(0))
                  ->getOperand(0)->hasOneUse()) {
            Instruction *YS = new ShiftInst(Instruction::Shl, 
                                            Op0BO->getOperand(1), Op1,
                                            Op0BO->getName());
            InsertNewInstBefore(YS, I); // (Y << C)
            Instruction *XM =
              BinaryOperator::createAnd(V1, ConstantExpr::getShl(CC, Op1),
                                        V1->getName()+".mask");
            InsertNewInstBefore(XM, I); // X & (CC << C)
            
            return BinaryOperator::create(Op0BO->getOpcode(), XM, YS);
          }
          
          break;
      }
      
      
      // If the operand is an bitwise operator with a constant RHS, and the
      // shift is the only use, we can pull it out of the shift.
      if (ConstantInt *Op0C = dyn_cast<ConstantInt>(Op0BO->getOperand(1))) {
        bool isValid = true;     // Valid only for And, Or, Xor
        bool highBitSet = false; // Transform if high bit of constant set?
        
        switch (Op0BO->getOpcode()) {
          default: isValid = false; break;   // Do not perform transform!
          case Instruction::Add:
            isValid = isLeftShift;
            break;
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
        if (isValid && !isLeftShift && isSignedShift) {
          uint64_t Val = Op0C->getZExtValue();
          isValid = ((Val & (1 << (TypeBits-1))) != 0) == highBitSet;
        }
        
        if (isValid) {
          Constant *NewRHS = ConstantExpr::get(I.getOpcode(), Op0C, Op1);
          
          Instruction *NewShift =
            new ShiftInst(I.getOpcode(), Op0BO->getOperand(0), Op1,
                          Op0BO->getName());
          Op0BO->setName("");
          InsertNewInstBefore(NewShift, I);
          
          return BinaryOperator::create(Op0BO->getOpcode(), NewShift,
                                        NewRHS);
        }
      }
    }
  }
  
  // Find out if this is a shift of a shift by a constant.
  ShiftInst *ShiftOp = 0;
  if (ShiftInst *Op0SI = dyn_cast<ShiftInst>(Op0))
    ShiftOp = Op0SI;
  else if (CastInst *CI = dyn_cast<CastInst>(Op0)) {
    // If this is a noop-integer case of a shift instruction, use the shift.
    if (CI->getOperand(0)->getType()->isInteger() &&
        CI->getOperand(0)->getType()->getPrimitiveSizeInBits() ==
        CI->getType()->getPrimitiveSizeInBits() &&
        isa<ShiftInst>(CI->getOperand(0))) {
      ShiftOp = cast<ShiftInst>(CI->getOperand(0));
    }
  }
  
  if (ShiftOp && isa<ConstantInt>(ShiftOp->getOperand(1))) {
    // Find the operands and properties of the input shift.  Note that the
    // signedness of the input shift may differ from the current shift if there
    // is a noop cast between the two.
    bool isShiftOfLeftShift = ShiftOp->getOpcode() == Instruction::Shl;
    bool isShiftOfSignedShift = ShiftOp->getType()->isSigned();
    bool isShiftOfUnsignedShift = !isShiftOfSignedShift;
    
    ConstantInt *ShiftAmt1C = cast<ConstantInt>(ShiftOp->getOperand(1));

    unsigned ShiftAmt1 = (unsigned)ShiftAmt1C->getZExtValue();
    unsigned ShiftAmt2 = (unsigned)Op1->getZExtValue();
    
    // Check for (A << c1) << c2   and   (A >> c1) >> c2.
    if (isLeftShift == isShiftOfLeftShift) {
      // Do not fold these shifts if the first one is signed and the second one
      // is unsigned and this is a right shift.  Further, don't do any folding
      // on them.
      if (isShiftOfSignedShift && isUnsignedShift && !isLeftShift)
        return 0;
      
      unsigned Amt = ShiftAmt1+ShiftAmt2;   // Fold into one big shift.
      if (Amt > Op0->getType()->getPrimitiveSizeInBits())
        Amt = Op0->getType()->getPrimitiveSizeInBits();
      
      Value *Op = ShiftOp->getOperand(0);
      if (isShiftOfSignedShift != isSignedShift)
        Op = InsertNewInstBefore(new CastInst(Op, I.getType(), "tmp"), I);
      return new ShiftInst(I.getOpcode(), Op,
                           ConstantInt::get(Type::UByteTy, Amt));
    }
    
    // Check for (A << c1) >> c2 or (A >> c1) << c2.  If we are dealing with
    // signed types, we can only support the (A >> c1) << c2 configuration,
    // because it can not turn an arbitrary bit of A into a sign bit.
    if (isUnsignedShift || isLeftShift) {
      // Calculate bitmask for what gets shifted off the edge.
      Constant *C = ConstantIntegral::getAllOnesValue(I.getType());
      if (isLeftShift)
        C = ConstantExpr::getShl(C, ShiftAmt1C);
      else
        C = ConstantExpr::getUShr(C, ShiftAmt1C);
      
      Value *Op = ShiftOp->getOperand(0);
      if (isShiftOfSignedShift != isSignedShift)
        Op = InsertCastBefore(Op, I.getType(), I);
      
      Instruction *Mask =
        BinaryOperator::createAnd(Op, C, Op->getName()+".mask");
      InsertNewInstBefore(Mask, I);
      
      // Figure out what flavor of shift we should use...
      if (ShiftAmt1 == ShiftAmt2) {
        return ReplaceInstUsesWith(I, Mask);  // (A << c) >> c  === A & c2
      } else if (ShiftAmt1 < ShiftAmt2) {
        return new ShiftInst(I.getOpcode(), Mask,
                         ConstantInt::get(Type::UByteTy, ShiftAmt2-ShiftAmt1));
      } else if (isShiftOfUnsignedShift || isShiftOfLeftShift) {
        if (isShiftOfUnsignedShift && !isShiftOfLeftShift && isSignedShift) {
          // Make sure to emit an unsigned shift right, not a signed one.
          Mask = InsertNewInstBefore(new CastInst(Mask, 
                                        Mask->getType()->getUnsignedVersion(),
                                                  Op->getName()), I);
          Mask = new ShiftInst(Instruction::Shr, Mask,
                         ConstantInt::get(Type::UByteTy, ShiftAmt1-ShiftAmt2));
          InsertNewInstBefore(Mask, I);
          return new CastInst(Mask, I.getType());
        } else {
          return new ShiftInst(ShiftOp->getOpcode(), Mask,
                    ConstantInt::get(Type::UByteTy, ShiftAmt1-ShiftAmt2));
        }
      } else {
        // (X >>s C1) << C2  where C1 > C2  === (X >>s (C1-C2)) & mask
        Op = InsertCastBefore(Mask, I.getType()->getSignedVersion(), I);
        Instruction *Shift =
          new ShiftInst(ShiftOp->getOpcode(), Op,
                        ConstantInt::get(Type::UByteTy, ShiftAmt1-ShiftAmt2));
        InsertNewInstBefore(Shift, I);
        
        C = ConstantIntegral::getAllOnesValue(Shift->getType());
        C = ConstantExpr::getShl(C, Op1);
        Mask = BinaryOperator::createAnd(Shift, C, Op->getName()+".mask");
        InsertNewInstBefore(Mask, I);
        return new CastInst(Mask, I.getType());
      }
    } else {
      // We can handle signed (X << C1) >>s C2 if it's a sign extend.  In
      // this case, C1 == C2 and C1 is 8, 16, or 32.
      if (ShiftAmt1 == ShiftAmt2) {
        const Type *SExtType = 0;
        switch (Op0->getType()->getPrimitiveSizeInBits() - ShiftAmt1) {
        case 8 : SExtType = Type::SByteTy; break;
        case 16: SExtType = Type::ShortTy; break;
        case 32: SExtType = Type::IntTy; break;
        }
        
        if (SExtType) {
          Instruction *NewTrunc = new CastInst(ShiftOp->getOperand(0),
                                               SExtType, "sext");
          InsertNewInstBefore(NewTrunc, I);
          return new CastInst(NewTrunc, I.getType());
        }
      }
    }
  }
  return 0;
}


/// DecomposeSimpleLinearExpr - Analyze 'Val', seeing if it is a simple linear
/// expression.  If so, decompose it, returning some value X, such that Val is
/// X*Scale+Offset.
///
static Value *DecomposeSimpleLinearExpr(Value *Val, unsigned &Scale,
                                        unsigned &Offset) {
  assert(Val->getType() == Type::UIntTy && "Unexpected allocation size type!");
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Val)) {
    if (CI->getType()->isUnsigned()) {
      Offset = CI->getZExtValue();
      Scale  = 1;
      return ConstantInt::get(Type::UIntTy, 0);
    }
  } else if (Instruction *I = dyn_cast<Instruction>(Val)) {
    if (I->getNumOperands() == 2) {
      if (ConstantInt *CUI = dyn_cast<ConstantInt>(I->getOperand(1))) {
        if (CUI->getType()->isUnsigned()) {
          if (I->getOpcode() == Instruction::Shl) {
            // This is a value scaled by '1 << the shift amt'.
            Scale = 1U << CUI->getZExtValue();
            Offset = 0;
            return I->getOperand(0);
          } else if (I->getOpcode() == Instruction::Mul) {
            // This value is scaled by 'CUI'.
            Scale = CUI->getZExtValue();
            Offset = 0;
            return I->getOperand(0);
          } else if (I->getOpcode() == Instruction::Add) {
            // We have X+C.  Check to see if we really have (X*C2)+C1, 
            // where C1 is divisible by C2.
            unsigned SubScale;
            Value *SubVal = 
              DecomposeSimpleLinearExpr(I->getOperand(0), SubScale, Offset);
            Offset += CUI->getZExtValue();
            if (SubScale > 1 && (Offset % SubScale == 0)) {
              Scale = SubScale;
              return SubVal;
            }
          }
        }
      }
    }
  }

  // Otherwise, we can't look past this.
  Scale = 1;
  Offset = 0;
  return Val;
}


/// PromoteCastOfAllocation - If we find a cast of an allocation instruction,
/// try to eliminate the cast by moving the type information into the alloc.
Instruction *InstCombiner::PromoteCastOfAllocation(CastInst &CI,
                                                   AllocationInst &AI) {
  const PointerType *PTy = dyn_cast<PointerType>(CI.getType());
  if (!PTy) return 0;   // Not casting the allocation to a pointer type.
  
  // Remove any uses of AI that are dead.
  assert(!CI.use_empty() && "Dead instructions should be removed earlier!");
  std::vector<Instruction*> DeadUsers;
  for (Value::use_iterator UI = AI.use_begin(), E = AI.use_end(); UI != E; ) {
    Instruction *User = cast<Instruction>(*UI++);
    if (isInstructionTriviallyDead(User)) {
      while (UI != E && *UI == User)
        ++UI; // If this instruction uses AI more than once, don't break UI.
      
      // Add operands to the worklist.
      AddUsesToWorkList(*User);
      ++NumDeadInst;
      DEBUG(std::cerr << "IC: DCE: " << *User);
      
      User->eraseFromParent();
      removeFromWorkList(User);
    }
  }
  
  // Get the type really allocated and the type casted to.
  const Type *AllocElTy = AI.getAllocatedType();
  const Type *CastElTy = PTy->getElementType();
  if (!AllocElTy->isSized() || !CastElTy->isSized()) return 0;

  unsigned AllocElTyAlign = TD->getTypeAlignment(AllocElTy);
  unsigned CastElTyAlign = TD->getTypeAlignment(CastElTy);
  if (CastElTyAlign < AllocElTyAlign) return 0;

  // If the allocation has multiple uses, only promote it if we are strictly
  // increasing the alignment of the resultant allocation.  If we keep it the
  // same, we open the door to infinite loops of various kinds.
  if (!AI.hasOneUse() && CastElTyAlign == AllocElTyAlign) return 0;

  uint64_t AllocElTySize = TD->getTypeSize(AllocElTy);
  uint64_t CastElTySize = TD->getTypeSize(CastElTy);
  if (CastElTySize == 0 || AllocElTySize == 0) return 0;

  // See if we can satisfy the modulus by pulling a scale out of the array
  // size argument.
  unsigned ArraySizeScale, ArrayOffset;
  Value *NumElements = // See if the array size is a decomposable linear expr.
    DecomposeSimpleLinearExpr(AI.getOperand(0), ArraySizeScale, ArrayOffset);
 
  // If we can now satisfy the modulus, by using a non-1 scale, we really can
  // do the xform.
  if ((AllocElTySize*ArraySizeScale) % CastElTySize != 0 ||
      (AllocElTySize*ArrayOffset   ) % CastElTySize != 0) return 0;

  unsigned Scale = (AllocElTySize*ArraySizeScale)/CastElTySize;
  Value *Amt = 0;
  if (Scale == 1) {
    Amt = NumElements;
  } else {
    // If the allocation size is constant, form a constant mul expression
    Amt = ConstantInt::get(Type::UIntTy, Scale);
    if (isa<ConstantInt>(NumElements) && NumElements->getType()->isUnsigned())
      Amt = ConstantExpr::getMul(
              cast<ConstantInt>(NumElements), cast<ConstantInt>(Amt));
    // otherwise multiply the amount and the number of elements
    else if (Scale != 1) {
      Instruction *Tmp = BinaryOperator::createMul(Amt, NumElements, "tmp");
      Amt = InsertNewInstBefore(Tmp, AI);
    }
  }
  
  if (unsigned Offset = (AllocElTySize*ArrayOffset)/CastElTySize) {
    Value *Off = ConstantInt::get(Type::UIntTy, Offset);
    Instruction *Tmp = BinaryOperator::createAdd(Amt, Off, "tmp");
    Amt = InsertNewInstBefore(Tmp, AI);
  }
  
  std::string Name = AI.getName(); AI.setName("");
  AllocationInst *New;
  if (isa<MallocInst>(AI))
    New = new MallocInst(CastElTy, Amt, AI.getAlignment(), Name);
  else
    New = new AllocaInst(CastElTy, Amt, AI.getAlignment(), Name);
  InsertNewInstBefore(New, AI);
  
  // If the allocation has multiple uses, insert a cast and change all things
  // that used it to use the new cast.  This will also hack on CI, but it will
  // die soon.
  if (!AI.hasOneUse()) {
    AddUsesToWorkList(AI);
    CastInst *NewCast = new CastInst(New, AI.getType(), "tmpcast");
    InsertNewInstBefore(NewCast, AI);
    AI.replaceAllUsesWith(NewCast);
  }
  return ReplaceInstUsesWith(CI, New);
}

/// CanEvaluateInDifferentType - Return true if we can take the specified value
/// and return it without inserting any new casts.  This is used by code that
/// tries to decide whether promoting or shrinking integer operations to wider
/// or smaller types will allow us to eliminate a truncate or extend.
static bool CanEvaluateInDifferentType(Value *V, const Type *Ty,
                                       int &NumCastsRemoved) {
  if (isa<Constant>(V)) return true;
  
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I || !I->hasOneUse()) return false;
  
  switch (I->getOpcode()) {
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    // These operators can all arbitrarily be extended or truncated.
    return CanEvaluateInDifferentType(I->getOperand(0), Ty, NumCastsRemoved) &&
           CanEvaluateInDifferentType(I->getOperand(1), Ty, NumCastsRemoved);
  case Instruction::Cast:
    // If this is a cast from the destination type, we can trivially eliminate
    // it, and this will remove a cast overall.
    if (I->getOperand(0)->getType() == Ty) {
      // If the first operand is itself a cast, and is eliminable, do not count
      // this as an eliminable cast.  We would prefer to eliminate those two
      // casts first.
      if (isa<CastInst>(I->getOperand(0)))
        return true;
      
      ++NumCastsRemoved;
      return true;
    }
    // TODO: Can handle more cases here.
    break;
  }
  
  return false;
}

/// EvaluateInDifferentType - Given an expression that 
/// CanEvaluateInDifferentType returns true for, actually insert the code to
/// evaluate the expression.
Value *InstCombiner::EvaluateInDifferentType(Value *V, const Type *Ty) {
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getCast(C, Ty);

  // Otherwise, it must be an instruction.
  Instruction *I = cast<Instruction>(V);
  Instruction *Res = 0;
  switch (I->getOpcode()) {
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    Value *LHS = EvaluateInDifferentType(I->getOperand(0), Ty);
    Value *RHS = EvaluateInDifferentType(I->getOperand(1), Ty);
    Res = BinaryOperator::create((Instruction::BinaryOps)I->getOpcode(),
                                 LHS, RHS, I->getName());
    break;
  }
  case Instruction::Cast:
    // If this is a cast from the destination type, return the input.
    if (I->getOperand(0)->getType() == Ty)
      return I->getOperand(0);
    
    // TODO: Can handle more cases here.
    assert(0 && "Unreachable!");
    break;
  }
  
  return InsertNewInstBefore(Res, *I);
}


// CastInst simplification
//
Instruction *InstCombiner::visitCastInst(CastInst &CI) {
  Value *Src = CI.getOperand(0);

  // If the user is casting a value to the same type, eliminate this cast
  // instruction...
  if (CI.getType() == Src->getType())
    return ReplaceInstUsesWith(CI, Src);

  if (isa<UndefValue>(Src))   // cast undef -> undef
    return ReplaceInstUsesWith(CI, UndefValue::get(CI.getType()));

  // If casting the result of another cast instruction, try to eliminate this
  // one!
  //
  if (CastInst *CSrc = dyn_cast<CastInst>(Src)) {   // A->B->C cast
    Value *A = CSrc->getOperand(0);
    if (isEliminableCastOfCast(A->getType(), CSrc->getType(),
                               CI.getType(), TD)) {
      // This instruction now refers directly to the cast's src operand.  This
      // has a good chance of making CSrc dead.
      CI.setOperand(0, CSrc->getOperand(0));
      return &CI;
    }

    // If this is an A->B->A cast, and we are dealing with integral types, try
    // to convert this into a logical 'and' instruction.
    //
    if (A->getType()->isInteger() &&
        CI.getType()->isInteger() && CSrc->getType()->isInteger() &&
        CSrc->getType()->isUnsigned() &&   // B->A cast must zero extend
        CSrc->getType()->getPrimitiveSizeInBits() <
                    CI.getType()->getPrimitiveSizeInBits()&&
        A->getType()->getPrimitiveSizeInBits() ==
              CI.getType()->getPrimitiveSizeInBits()) {
      assert(CSrc->getType() != Type::ULongTy &&
             "Cannot have type bigger than ulong!");
      uint64_t AndValue = CSrc->getType()->getIntegralTypeMask();
      Constant *AndOp = ConstantInt::get(A->getType()->getUnsignedVersion(),
                                          AndValue);
      AndOp = ConstantExpr::getCast(AndOp, A->getType());
      Instruction *And = BinaryOperator::createAnd(CSrc->getOperand(0), AndOp);
      if (And->getType() != CI.getType()) {
        And->setName(CSrc->getName()+".mask");
        InsertNewInstBefore(And, CI);
        And = new CastInst(And, CI.getType());
      }
      return And;
    }
  }
  
  // If this is a cast to bool, turn it into the appropriate setne instruction.
  if (CI.getType() == Type::BoolTy)
    return BinaryOperator::createSetNE(CI.getOperand(0),
                       Constant::getNullValue(CI.getOperand(0)->getType()));

  // See if we can simplify any instructions used by the LHS whose sole 
  // purpose is to compute bits we don't care about.
  if (CI.getType()->isInteger() && CI.getOperand(0)->getType()->isIntegral()) {
    uint64_t KnownZero, KnownOne;
    if (SimplifyDemandedBits(&CI, CI.getType()->getIntegralTypeMask(),
                             KnownZero, KnownOne))
      return &CI;
  }
  
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
    if (Instruction *V = PromoteCastOfAllocation(CI, *AI))
      return V;

  if (SelectInst *SI = dyn_cast<SelectInst>(Src))
    if (Instruction *NV = FoldOpIntoSelect(CI, SI, this))
      return NV;
  if (isa<PHINode>(Src))
    if (Instruction *NV = FoldOpIntoPhi(CI))
      return NV;
  
  // If the source and destination are pointers, and this cast is equivalent to
  // a getelementptr X, 0, 0, 0...  turn it into the appropriate getelementptr.
  // This can enhance SROA and other transforms that want type-safe pointers.
  if (const PointerType *DstPTy = dyn_cast<PointerType>(CI.getType()))
    if (const PointerType *SrcPTy = dyn_cast<PointerType>(Src->getType())) {
      const Type *DstTy = DstPTy->getElementType();
      const Type *SrcTy = SrcPTy->getElementType();
      
      Constant *ZeroUInt = Constant::getNullValue(Type::UIntTy);
      unsigned NumZeros = 0;
      while (SrcTy != DstTy && 
             isa<CompositeType>(SrcTy) && !isa<PointerType>(SrcTy) &&
             SrcTy->getNumContainedTypes() /* not "{}" */) {
        SrcTy = cast<CompositeType>(SrcTy)->getTypeAtIndex(ZeroUInt);
        ++NumZeros;
      }

      // If we found a path from the src to dest, create the getelementptr now.
      if (SrcTy == DstTy) {
        std::vector<Value*> Idxs(NumZeros+1, ZeroUInt);
        return new GetElementPtrInst(Src, Idxs);
      }
    }
      
  // If the source value is an instruction with only this use, we can attempt to
  // propagate the cast into the instruction.  Also, only handle integral types
  // for now.
  if (Instruction *SrcI = dyn_cast<Instruction>(Src)) {
    if (SrcI->hasOneUse() && Src->getType()->isIntegral() &&
        CI.getType()->isInteger()) {  // Don't mess with casts to bool here
      
      int NumCastsRemoved = 0;
      if (CanEvaluateInDifferentType(SrcI, CI.getType(), NumCastsRemoved)) {
        // If this cast is a truncate, evaluting in a different type always
        // eliminates the cast, so it is always a win.  If this is a noop-cast
        // this just removes a noop cast which isn't pointful, but simplifies
        // the code.  If this is a zero-extension, we need to do an AND to
        // maintain the clear top-part of the computation, so we require that
        // the input have eliminated at least one cast.  If this is a sign
        // extension, we insert two new casts (to do the extension) so we
        // require that two casts have been eliminated.
        bool DoXForm;
        switch (getCastType(Src->getType(), CI.getType())) {
        default: assert(0 && "Unknown cast type!");
        case Noop:
        case Truncate:
          DoXForm = true;
          break;
        case Zeroext:
          DoXForm = NumCastsRemoved >= 1;
          break;
        case Signext:
          DoXForm = NumCastsRemoved >= 2;
          break;
        }
        
        if (DoXForm) {
          Value *Res = EvaluateInDifferentType(SrcI, CI.getType());
          assert(Res->getType() == CI.getType());
          switch (getCastType(Src->getType(), CI.getType())) {
          default: assert(0 && "Unknown cast type!");
          case Noop:
          case Truncate:
            // Just replace this cast with the result.
            return ReplaceInstUsesWith(CI, Res);
          case Zeroext: {
            // We need to emit an AND to clear the high bits.
            unsigned SrcBitSize = Src->getType()->getPrimitiveSizeInBits();
            unsigned DestBitSize = CI.getType()->getPrimitiveSizeInBits();
            assert(SrcBitSize < DestBitSize && "Not a zext?");
            Constant *C = 
              ConstantInt::get(Type::ULongTy, (1ULL << SrcBitSize)-1);
            C = ConstantExpr::getCast(C, CI.getType());
            return BinaryOperator::createAnd(Res, C);
          }
          case Signext:
            // We need to emit a cast to truncate, then a cast to sext.
            return new CastInst(InsertCastBefore(Res, Src->getType(), CI),
                                CI.getType());
          }
        }
      }
      
      const Type *DestTy = CI.getType();
      unsigned SrcBitSize = Src->getType()->getPrimitiveSizeInBits();
      unsigned DestBitSize = DestTy->getPrimitiveSizeInBits();

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

        // cast (xor bool X, true) to int  --> xor (cast bool X to int), 1
        if (SrcBitSize == 1 && SrcI->getOpcode() == Instruction::Xor &&
            Op1 == ConstantBool::getTrue() &&
            (!Op0->hasOneUse() || !isa<SetCondInst>(Op0))) {
          Value *New = InsertOperandCastBefore(Op0, DestTy, &CI);
          return BinaryOperator::createXor(New,
                                           ConstantInt::get(CI.getType(), 1));
        }
        break;
      case Instruction::SDiv:
      case Instruction::UDiv:
      case Instruction::SRem:
      case Instruction::URem:
        // If we are just changing the sign, rewrite.
        if (DestBitSize == SrcBitSize) {
          // Don't insert two casts if they cannot be eliminated.  We allow two
          // casts to be inserted if the sizes are the same.  This could only be
          // converting signedness, which is a noop.
          if (!ValueRequiresCast(Op1, DestTy,TD) || 
              !ValueRequiresCast(Op0, DestTy, TD)) {
            Value *Op0c = InsertOperandCastBefore(Op0, DestTy, SrcI);
            Value *Op1c = InsertOperandCastBefore(Op1, DestTy, SrcI);
            return BinaryOperator::create(
              cast<BinaryOperator>(SrcI)->getOpcode(), Op0c, Op1c);
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
      case Instruction::Shr:
        // If this is a signed shr, and if all bits shifted in are about to be
        // truncated off, turn it into an unsigned shr to allow greater
        // simplifications.
        if (DestBitSize < SrcBitSize && Src->getType()->isSigned() &&
            isa<ConstantInt>(Op1)) {
          unsigned ShiftAmt = cast<ConstantInt>(Op1)->getZExtValue();
          if (SrcBitSize > ShiftAmt && SrcBitSize-ShiftAmt >= DestBitSize) {
            // Convert to unsigned.
            Value *N1 = InsertOperandCastBefore(Op0,
                                     Op0->getType()->getUnsignedVersion(), &CI);
            // Insert the new shift, which is now unsigned.
            N1 = InsertNewInstBefore(new ShiftInst(Instruction::Shr, N1,
                                                   Op1, Src->getName()), CI);
            return new CastInst(N1, CI.getType());
          }
        }
        break;

      case Instruction::SetEQ:
      case Instruction::SetNE:
        // We if we are just checking for a seteq of a single bit and casting it
        // to an integer.  If so, shift the bit to the appropriate place then
        // cast to integer to avoid the comparison.
        if (ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
          uint64_t Op1CV = Op1C->getZExtValue();
          // cast (X == 0) to int --> X^1        iff X has only the low bit set.
          // cast (X == 0) to int --> (X>>1)^1   iff X has only the 2nd bit set.
          // cast (X == 1) to int --> X          iff X has only the low bit set.
          // cast (X == 2) to int --> X>>1       iff X has only the 2nd bit set.
          // cast (X != 0) to int --> X          iff X has only the low bit set.
          // cast (X != 0) to int --> X>>1       iff X has only the 2nd bit set.
          // cast (X != 1) to int --> X^1        iff X has only the low bit set.
          // cast (X != 2) to int --> (X>>1)^1   iff X has only the 2nd bit set.
          if (Op1CV == 0 || isPowerOf2_64(Op1CV)) {
            // If Op1C some other power of two, convert:
            uint64_t KnownZero, KnownOne;
            uint64_t TypeMask = Op1->getType()->getIntegralTypeMask();
            ComputeMaskedBits(Op0, TypeMask, KnownZero, KnownOne);
            
            if (isPowerOf2_64(KnownZero^TypeMask)) { // Exactly one possible 1?
              bool isSetNE = SrcI->getOpcode() == Instruction::SetNE;
              if (Op1CV && (Op1CV != (KnownZero^TypeMask))) {
                // (X&4) == 2 --> false
                // (X&4) != 2 --> true
                Constant *Res = ConstantBool::get(isSetNE);
                Res = ConstantExpr::getCast(Res, CI.getType());
                return ReplaceInstUsesWith(CI, Res);
              }
              
              unsigned ShiftAmt = Log2_64(KnownZero^TypeMask);
              Value *In = Op0;
              if (ShiftAmt) {
                // Perform an unsigned shr by shiftamt.  Convert input to
                // unsigned if it is signed.
                if (In->getType()->isSigned())
                  In = InsertCastBefore(
                         In, In->getType()->getUnsignedVersion(), CI);
                // Insert the shift to put the result in the low bit.
                In = InsertNewInstBefore(new ShiftInst(Instruction::Shr, In,
                                     ConstantInt::get(Type::UByteTy, ShiftAmt),
                                     In->getName()+".lobit"), CI);
              }
              
              if ((Op1CV != 0) == isSetNE) { // Toggle the low bit.
                Constant *One = ConstantInt::get(In->getType(), 1);
                In = BinaryOperator::createXor(In, One, "tmp");
                InsertNewInstBefore(cast<Instruction>(In), CI);
              }
              
              if (CI.getType() == In->getType())
                return ReplaceInstUsesWith(CI, In);
              else
                return new CastInst(In, CI.getType());
            }
          }
        }
        break;
      }
    }
    
    if (SrcI->hasOneUse()) {
      if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(SrcI)) {
        // Okay, we have (cast (shuffle ..)).  We know this cast is a bitconvert
        // because the inputs are known to be a vector.  Check to see if this is
        // a cast to a vector with the same # elts.
        if (isa<PackedType>(CI.getType()) && 
            cast<PackedType>(CI.getType())->getNumElements() == 
                  SVI->getType()->getNumElements()) {
          CastInst *Tmp;
          // If either of the operands is a cast from CI.getType(), then
          // evaluating the shuffle in the casted destination's type will allow
          // us to eliminate at least one cast.
          if (((Tmp = dyn_cast<CastInst>(SVI->getOperand(0))) && 
               Tmp->getOperand(0)->getType() == CI.getType()) ||
              ((Tmp = dyn_cast<CastInst>(SVI->getOperand(1))) && 
               Tmp->getOperand(0)->getType() == CI.getType())) {
            Value *LHS = InsertOperandCastBefore(SVI->getOperand(0),
                                                 CI.getType(), &CI);
            Value *RHS = InsertOperandCastBefore(SVI->getOperand(1),
                                                 CI.getType(), &CI);
            // Return a new shuffle vector.  Use the same element ID's, as we
            // know the vector types match #elts.
            return new ShuffleVectorInst(LHS, RHS, SVI->getOperand(2));
          }
        }
      }
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

/// FoldSelectOpOp - Here we have (select c, TI, FI), and we know that TI and FI
/// have the same opcode and only one use each.  Try to simplify this.
Instruction *InstCombiner::FoldSelectOpOp(SelectInst &SI, Instruction *TI,
                                          Instruction *FI) {
  if (TI->getNumOperands() == 1) {
    // If this is a non-volatile load or a cast from the same type,
    // merge.
    if (TI->getOpcode() == Instruction::Cast) {
      if (TI->getOperand(0)->getType() != FI->getOperand(0)->getType())
        return 0;
    } else {
      return 0;  // unknown unary op.
    }

    // Fold this by inserting a select from the input values.
    SelectInst *NewSI = new SelectInst(SI.getCondition(), TI->getOperand(0),
                                       FI->getOperand(0), SI.getName()+".v");
    InsertNewInstBefore(NewSI, SI);
    return new CastInst(NewSI, TI->getType());
  }

  // Only handle binary operators here.
  if (!isa<ShiftInst>(TI) && !isa<BinaryOperator>(TI))
    return 0;

  // Figure out if the operations have any operands in common.
  Value *MatchOp, *OtherOpT, *OtherOpF;
  bool MatchIsOpZero;
  if (TI->getOperand(0) == FI->getOperand(0)) {
    MatchOp  = TI->getOperand(0);
    OtherOpT = TI->getOperand(1);
    OtherOpF = FI->getOperand(1);
    MatchIsOpZero = true;
  } else if (TI->getOperand(1) == FI->getOperand(1)) {
    MatchOp  = TI->getOperand(1);
    OtherOpT = TI->getOperand(0);
    OtherOpF = FI->getOperand(0);
    MatchIsOpZero = false;
  } else if (!TI->isCommutative()) {
    return 0;
  } else if (TI->getOperand(0) == FI->getOperand(1)) {
    MatchOp  = TI->getOperand(0);
    OtherOpT = TI->getOperand(1);
    OtherOpF = FI->getOperand(0);
    MatchIsOpZero = true;
  } else if (TI->getOperand(1) == FI->getOperand(0)) {
    MatchOp  = TI->getOperand(1);
    OtherOpT = TI->getOperand(0);
    OtherOpF = FI->getOperand(1);
    MatchIsOpZero = true;
  } else {
    return 0;
  }

  // If we reach here, they do have operations in common.
  SelectInst *NewSI = new SelectInst(SI.getCondition(), OtherOpT,
                                     OtherOpF, SI.getName()+".v");
  InsertNewInstBefore(NewSI, SI);

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TI)) {
    if (MatchIsOpZero)
      return BinaryOperator::create(BO->getOpcode(), MatchOp, NewSI);
    else
      return BinaryOperator::create(BO->getOpcode(), NewSI, MatchOp);
  } else {
    if (MatchIsOpZero)
      return new ShiftInst(cast<ShiftInst>(TI)->getOpcode(), MatchOp, NewSI);
    else
      return new ShiftInst(cast<ShiftInst>(TI)->getOpcode(), NewSI, MatchOp);
  }
}

Instruction *InstCombiner::visitSelectInst(SelectInst &SI) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  // select true, X, Y  -> X
  // select false, X, Y -> Y
  if (ConstantBool *C = dyn_cast<ConstantBool>(CondVal))
    return ReplaceInstUsesWith(SI, C->getValue() ? TrueVal : FalseVal);

  // select C, X, X -> X
  if (TrueVal == FalseVal)
    return ReplaceInstUsesWith(SI, TrueVal);

  if (isa<UndefValue>(TrueVal))   // select C, undef, X -> X
    return ReplaceInstUsesWith(SI, FalseVal);
  if (isa<UndefValue>(FalseVal))   // select C, X, undef -> X
    return ReplaceInstUsesWith(SI, TrueVal);
  if (isa<UndefValue>(CondVal)) {  // select undef, X, Y -> X or Y
    if (isa<Constant>(TrueVal))
      return ReplaceInstUsesWith(SI, TrueVal);
    else
      return ReplaceInstUsesWith(SI, FalseVal);
  }

  if (SI.getType() == Type::BoolTy)
    if (ConstantBool *C = dyn_cast<ConstantBool>(TrueVal)) {
      if (C->getValue()) {
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
      if (C->getValue() == false) {
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
      if (FalseValC->isNullValue() && TrueValC->getZExtValue() == 1) {
        return new CastInst(CondVal, SI.getType());
      } else if (TrueValC->isNullValue() && FalseValC->getZExtValue() == 1) {
        // select C, 0, 1 -> cast !C to int
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::createNot(CondVal,
                                               "not."+CondVal->getName()), SI);
        return new CastInst(NotCond, SI.getType());
      }

      if (SetCondInst *IC = dyn_cast<SetCondInst>(SI.getCondition())) {

        // (x <s 0) ? -1 : 0 -> sra x, 31
        // (x >u 2147483647) ? -1 : 0 -> sra x, 31
        if (TrueValC->isAllOnesValue() && FalseValC->isNullValue())
          if (ConstantInt *CmpCst = dyn_cast<ConstantInt>(IC->getOperand(1))) {
            bool CanXForm = false;
            if (CmpCst->getType()->isSigned())
              CanXForm = CmpCst->isNullValue() && 
                         IC->getOpcode() == Instruction::SetLT;
            else {
              unsigned Bits = CmpCst->getType()->getPrimitiveSizeInBits();
              CanXForm = (CmpCst->getZExtValue() == ~0ULL >> (64-Bits+1)) &&
                         IC->getOpcode() == Instruction::SetGT;
            }
            
            if (CanXForm) {
              // The comparison constant and the result are not neccessarily the
              // same width.  In any case, the first step to do is make sure
              // that X is signed.
              Value *X = IC->getOperand(0);
              if (!X->getType()->isSigned())
                X = InsertCastBefore(X, X->getType()->getSignedVersion(), SI);
              
              // Now that X is signed, we have to make the all ones value.  Do
              // this by inserting a new SRA.
              unsigned Bits = X->getType()->getPrimitiveSizeInBits();
              Constant *ShAmt = ConstantInt::get(Type::UByteTy, Bits-1);
              Instruction *SRA = new ShiftInst(Instruction::Shr, X,
                                               ShAmt, "ones");
              InsertNewInstBefore(SRA, SI);
              
              // Finally, convert to the type of the select RHS.  If this is
              // smaller than the compare value, it will truncate the ones to
              // fit. If it is larger, it will sext the ones to fit.
              return new CastInst(SRA, SI.getType());
            }
          }


        // If one of the constants is zero (we know they can't both be) and we
        // have a setcc instruction with zero, and we have an 'and' with the
        // non-constant value, eliminate this whole mess.  This corresponds to
        // cases like this: ((X & 27) ? 27 : 0)
        if (TrueValC->isNullValue() || FalseValC->isNullValue())
          if (IC->isEquality() && isa<ConstantInt>(IC->getOperand(1)) &&
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

  if (Instruction *TI = dyn_cast<Instruction>(TrueVal))
    if (Instruction *FI = dyn_cast<Instruction>(FalseVal))
      if (TI->hasOneUse() && FI->hasOneUse()) {
        Instruction *AddOp = 0, *SubOp = 0;

        // Turn (select C, (op X, Y), (op X, Z)) -> (op X, (select C, Y, Z))
        if (TI->getOpcode() == FI->getOpcode())
          if (Instruction *IV = FoldSelectOpOp(SI, TI, FI))
            return IV;

        // Turn select C, (X+Y), (X-Y) --> (X+(select C, Y, (-Y))).  This is
        // even legal for FP.
        if (TI->getOpcode() == Instruction::Sub &&
            FI->getOpcode() == Instruction::Add) {
          AddOp = FI; SubOp = TI;
        } else if (FI->getOpcode() == Instruction::Sub &&
                   TI->getOpcode() == Instruction::Add) {
          AddOp = TI; SubOp = FI;
        }

        if (AddOp) {
          Value *OtherAddOp = 0;
          if (SubOp->getOperand(0) == AddOp->getOperand(0)) {
            OtherAddOp = AddOp->getOperand(1);
          } else if (SubOp->getOperand(0) == AddOp->getOperand(1)) {
            OtherAddOp = AddOp->getOperand(0);
          }

          if (OtherAddOp) {
            // So at this point we know we have (Y -> OtherAddOp):
            //        select C, (add X, Y), (sub X, Z)
            Value *NegVal;  // Compute -Z
            if (Constant *C = dyn_cast<Constant>(SubOp->getOperand(1))) {
              NegVal = ConstantExpr::getNeg(C);
            } else {
              NegVal = InsertNewInstBefore(
                    BinaryOperator::createNeg(SubOp->getOperand(1), "tmp"), SI);
            }

            Value *NewTrueOp = OtherAddOp;
            Value *NewFalseOp = NegVal;
            if (AddOp != TI)
              std::swap(NewTrueOp, NewFalseOp);
            Instruction *NewSel =
              new SelectInst(CondVal, NewTrueOp,NewFalseOp,SI.getName()+".p");

            NewSel = InsertNewInstBefore(NewSel, SI);
            return BinaryOperator::createAdd(SubOp->getOperand(0), NewSel);
          }
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

  if (BinaryOperator::isNot(CondVal)) {
    SI.setOperand(0, BinaryOperator::getNotArgument(CondVal));
    SI.setOperand(1, FalseVal);
    SI.setOperand(2, TrueVal);
    return &SI;
  }

  return 0;
}

/// GetKnownAlignment - If the specified pointer has an alignment that we can
/// determine, return it, otherwise return 0.
static unsigned GetKnownAlignment(Value *V, TargetData *TD) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    unsigned Align = GV->getAlignment();
    if (Align == 0 && TD) 
      Align = TD->getTypeAlignment(GV->getType()->getElementType());
    return Align;
  } else if (AllocationInst *AI = dyn_cast<AllocationInst>(V)) {
    unsigned Align = AI->getAlignment();
    if (Align == 0 && TD) {
      if (isa<AllocaInst>(AI))
        Align = TD->getTypeAlignment(AI->getType()->getElementType());
      else if (isa<MallocInst>(AI)) {
        // Malloc returns maximally aligned memory.
        Align = TD->getTypeAlignment(AI->getType()->getElementType());
        Align = std::max(Align, (unsigned)TD->getTypeAlignment(Type::DoubleTy));
        Align = std::max(Align, (unsigned)TD->getTypeAlignment(Type::LongTy));
      }
    }
    return Align;
  } else if (isa<CastInst>(V) ||
             (isa<ConstantExpr>(V) && 
              cast<ConstantExpr>(V)->getOpcode() == Instruction::Cast)) {
    User *CI = cast<User>(V);
    if (isa<PointerType>(CI->getOperand(0)->getType()))
      return GetKnownAlignment(CI->getOperand(0), TD);
    return 0;
  } else if (isa<GetElementPtrInst>(V) ||
             (isa<ConstantExpr>(V) && 
              cast<ConstantExpr>(V)->getOpcode()==Instruction::GetElementPtr)) {
    User *GEPI = cast<User>(V);
    unsigned BaseAlignment = GetKnownAlignment(GEPI->getOperand(0), TD);
    if (BaseAlignment == 0) return 0;
    
    // If all indexes are zero, it is just the alignment of the base pointer.
    bool AllZeroOperands = true;
    for (unsigned i = 1, e = GEPI->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEPI->getOperand(i)) ||
          !cast<Constant>(GEPI->getOperand(i))->isNullValue()) {
        AllZeroOperands = false;
        break;
      }
    if (AllZeroOperands)
      return BaseAlignment;
    
    // Otherwise, if the base alignment is >= the alignment we expect for the
    // base pointer type, then we know that the resultant pointer is aligned at
    // least as much as its type requires.
    if (!TD) return 0;

    const Type *BasePtrTy = GEPI->getOperand(0)->getType();
    if (TD->getTypeAlignment(cast<PointerType>(BasePtrTy)->getElementType())
        <= BaseAlignment) {
      const Type *GEPTy = GEPI->getType();
      return TD->getTypeAlignment(cast<PointerType>(GEPTy)->getElementType());
    }
    return 0;
  }
  return 0;
}


/// visitCallInst - CallInst simplification.  This mostly only handles folding 
/// of intrinsic instructions.  For normal calls, it allows visitCallSite to do
/// the heavy lifting.
///
Instruction *InstCombiner::visitCallInst(CallInst &CI) {
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(&CI);
  if (!II) return visitCallSite(&CI);
  
  // Intrinsics cannot occur in an invoke, so handle them here instead of in
  // visitCallSite.
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(II)) {
    bool Changed = false;

    // memmove/cpy/set of zero bytes is a noop.
    if (Constant *NumBytes = dyn_cast<Constant>(MI->getLength())) {
      if (NumBytes->isNullValue()) return EraseInstFromFunction(CI);

      if (ConstantInt *CI = dyn_cast<ConstantInt>(NumBytes))
        if (CI->getZExtValue() == 1) {
          // Replace the instruction with just byte operations.  We would
          // transform other cases to loads/stores, but we don't know if
          // alignment is sufficient.
        }
    }

    // If we have a memmove and the source operation is a constant global,
    // then the source and dest pointers can't alias, so we can change this
    // into a call to memcpy.
    if (MemMoveInst *MMI = dyn_cast<MemMoveInst>(II)) {
      if (GlobalVariable *GVSrc = dyn_cast<GlobalVariable>(MMI->getSource()))
        if (GVSrc->isConstant()) {
          Module *M = CI.getParent()->getParent()->getParent();
          const char *Name;
          if (CI.getCalledFunction()->getFunctionType()->getParamType(2) == 
              Type::UIntTy)
            Name = "llvm.memcpy.i32";
          else
            Name = "llvm.memcpy.i64";
          Function *MemCpy = M->getOrInsertFunction(Name,
                                     CI.getCalledFunction()->getFunctionType());
          CI.setOperand(0, MemCpy);
          Changed = true;
        }
    }

    // If we can determine a pointer alignment that is bigger than currently
    // set, update the alignment.
    if (isa<MemCpyInst>(MI) || isa<MemMoveInst>(MI)) {
      unsigned Alignment1 = GetKnownAlignment(MI->getOperand(1), TD);
      unsigned Alignment2 = GetKnownAlignment(MI->getOperand(2), TD);
      unsigned Align = std::min(Alignment1, Alignment2);
      if (MI->getAlignment()->getZExtValue() < Align) {
        MI->setAlignment(ConstantInt::get(Type::UIntTy, Align));
        Changed = true;
      }
    } else if (isa<MemSetInst>(MI)) {
      unsigned Alignment = GetKnownAlignment(MI->getDest(), TD);
      if (MI->getAlignment()->getZExtValue() < Alignment) {
        MI->setAlignment(ConstantInt::get(Type::UIntTy, Alignment));
        Changed = true;
      }
    }
          
    if (Changed) return II;
  } else {
    switch (II->getIntrinsicID()) {
    default: break;
    case Intrinsic::ppc_altivec_lvx:
    case Intrinsic::ppc_altivec_lvxl:
    case Intrinsic::x86_sse_loadu_ps:
    case Intrinsic::x86_sse2_loadu_pd:
    case Intrinsic::x86_sse2_loadu_dq:
      // Turn PPC lvx     -> load if the pointer is known aligned.
      // Turn X86 loadups -> load if the pointer is known aligned.
      if (GetKnownAlignment(II->getOperand(1), TD) >= 16) {
        Value *Ptr = InsertCastBefore(II->getOperand(1),
                                      PointerType::get(II->getType()), CI);
        return new LoadInst(Ptr);
      }
      break;
    case Intrinsic::ppc_altivec_stvx:
    case Intrinsic::ppc_altivec_stvxl:
      // Turn stvx -> store if the pointer is known aligned.
      if (GetKnownAlignment(II->getOperand(2), TD) >= 16) {
        const Type *OpPtrTy = PointerType::get(II->getOperand(1)->getType());
        Value *Ptr = InsertCastBefore(II->getOperand(2), OpPtrTy, CI);
        return new StoreInst(II->getOperand(1), Ptr);
      }
      break;
    case Intrinsic::x86_sse_storeu_ps:
    case Intrinsic::x86_sse2_storeu_pd:
    case Intrinsic::x86_sse2_storeu_dq:
    case Intrinsic::x86_sse2_storel_dq:
      // Turn X86 storeu -> store if the pointer is known aligned.
      if (GetKnownAlignment(II->getOperand(1), TD) >= 16) {
        const Type *OpPtrTy = PointerType::get(II->getOperand(2)->getType());
        Value *Ptr = InsertCastBefore(II->getOperand(1), OpPtrTy, CI);
        return new StoreInst(II->getOperand(2), Ptr);
      }
      break;
      
    case Intrinsic::x86_sse_cvttss2si: {
      // These intrinsics only demands the 0th element of its input vector.  If
      // we can simplify the input based on that, do so now.
      uint64_t UndefElts;
      if (Value *V = SimplifyDemandedVectorElts(II->getOperand(1), 1, 
                                                UndefElts)) {
        II->setOperand(1, V);
        return II;
      }
      break;
    }
      
    case Intrinsic::ppc_altivec_vperm:
      // Turn vperm(V1,V2,mask) -> shuffle(V1,V2,mask) if mask is a constant.
      if (ConstantPacked *Mask = dyn_cast<ConstantPacked>(II->getOperand(3))) {
        assert(Mask->getNumOperands() == 16 && "Bad type for intrinsic!");
        
        // Check that all of the elements are integer constants or undefs.
        bool AllEltsOk = true;
        for (unsigned i = 0; i != 16; ++i) {
          if (!isa<ConstantInt>(Mask->getOperand(i)) && 
              !isa<UndefValue>(Mask->getOperand(i))) {
            AllEltsOk = false;
            break;
          }
        }
        
        if (AllEltsOk) {
          // Cast the input vectors to byte vectors.
          Value *Op0 = InsertCastBefore(II->getOperand(1), Mask->getType(), CI);
          Value *Op1 = InsertCastBefore(II->getOperand(2), Mask->getType(), CI);
          Value *Result = UndefValue::get(Op0->getType());
          
          // Only extract each element once.
          Value *ExtractedElts[32];
          memset(ExtractedElts, 0, sizeof(ExtractedElts));
          
          for (unsigned i = 0; i != 16; ++i) {
            if (isa<UndefValue>(Mask->getOperand(i)))
              continue;
            unsigned Idx =cast<ConstantInt>(Mask->getOperand(i))->getZExtValue();
            Idx &= 31;  // Match the hardware behavior.
            
            if (ExtractedElts[Idx] == 0) {
              Instruction *Elt = 
                new ExtractElementInst(Idx < 16 ? Op0 : Op1, Idx&15, "tmp");
              InsertNewInstBefore(Elt, CI);
              ExtractedElts[Idx] = Elt;
            }
          
            // Insert this value into the result vector.
            Result = new InsertElementInst(Result, ExtractedElts[Idx], i,"tmp");
            InsertNewInstBefore(cast<Instruction>(Result), CI);
          }
          return new CastInst(Result, CI.getType());
        }
      }
      break;

    case Intrinsic::stackrestore: {
      // If the save is right next to the restore, remove the restore.  This can
      // happen when variable allocas are DCE'd.
      if (IntrinsicInst *SS = dyn_cast<IntrinsicInst>(II->getOperand(1))) {
        if (SS->getIntrinsicID() == Intrinsic::stacksave) {
          BasicBlock::iterator BI = SS;
          if (&*++BI == II)
            return EraseInstFromFunction(CI);
        }
      }
      
      // If the stack restore is in a return/unwind block and if there are no
      // allocas or calls between the restore and the return, nuke the restore.
      TerminatorInst *TI = II->getParent()->getTerminator();
      if (isa<ReturnInst>(TI) || isa<UnwindInst>(TI)) {
        BasicBlock::iterator BI = II;
        bool CannotRemove = false;
        for (++BI; &*BI != TI; ++BI) {
          if (isa<AllocaInst>(BI) ||
              (isa<CallInst>(BI) && !isa<IntrinsicInst>(BI))) {
            CannotRemove = true;
            break;
          }
        }
        if (!CannotRemove)
          return EraseInstFromFunction(CI);
      }
      break;
    }
    }
  }

  return visitCallSite(II);
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

  if (Function *CalleeF = dyn_cast<Function>(Callee))
    if (CalleeF->getCallingConv() != CS.getCallingConv()) {
      Instruction *OldCall = CS.getInstruction();
      // If the call and callee calling conventions don't match, this call must
      // be unreachable, as the call is undefined.
      new StoreInst(ConstantBool::getTrue(),
                    UndefValue::get(PointerType::get(Type::BoolTy)), OldCall);
      if (!OldCall->use_empty())
        OldCall->replaceAllUsesWith(UndefValue::get(OldCall->getType()));
      if (isa<CallInst>(OldCall))   // Not worth removing an invoke here.
        return EraseInstFromFunction(*OldCall);
      return 0;
    }

  if (isa<ConstantPointerNull>(Callee) || isa<UndefValue>(Callee)) {
    // This instruction is not reachable, just remove it.  We insert a store to
    // undef so that we know that this code is not reachable, despite the fact
    // that we can't modify the CFG here.
    new StoreInst(ConstantBool::getTrue(),
                  UndefValue::get(PointerType::get(Type::BoolTy)),
                  CS.getInstruction());

    if (!CS.getInstruction()->use_empty())
      CS.getInstruction()->
        replaceAllUsesWith(UndefValue::get(CS.getInstruction()->getType()));

    if (InvokeInst *II = dyn_cast<InvokeInst>(CS.getInstruction())) {
      // Don't break the CFG, insert a dummy cond branch.
      new BranchInst(II->getNormalDest(), II->getUnwindDest(),
                     ConstantBool::getTrue(), II);
    }
    return EraseInstFromFunction(*CS.getInstruction());
  }

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
        !(OldRetTy->isLosslesslyConvertibleTo(FT->getReturnType()) ||
          (isa<PointerType>(FT->getReturnType()) && 
           TD->getIntPtrType()->isLosslesslyConvertibleTo(OldRetTy)))
        && !Caller->use_empty())
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
    const Type *ActTy = (*AI)->getType();
    ConstantInt* c = dyn_cast<ConstantInt>(*AI);
    //Either we can cast directly, or we can upconvert the argument
    bool isConvertible = ActTy->isLosslesslyConvertibleTo(ParamTy) ||
      (ParamTy->isIntegral() && ActTy->isIntegral() &&
       ParamTy->isSigned() == ActTy->isSigned() &&
       ParamTy->getPrimitiveSize() >= ActTy->getPrimitiveSize()) ||
      (c && ParamTy->getPrimitiveSize() >= ActTy->getPrimitiveSize() &&
       c->getSExtValue() > 0);
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
    cast<InvokeInst>(II)->setCallingConv(II->getCallingConv());
  } else {
    NC = new CallInst(Callee, Args, Caller->getName(), Caller);
    if (cast<CallInst>(Caller)->isTailCall())
      cast<CallInst>(NC)->setTailCall();
   cast<CallInst>(NC)->setCallingConv(cast<CallInst>(Caller)->getCallingConv());
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
      NV = UndefValue::get(Caller->getType());
    }
  }

  if (Caller->getType() != Type::VoidTy && !Caller->use_empty())
    Caller->replaceAllUsesWith(NV);
  Caller->getParent()->getInstList().erase(Caller);
  removeFromWorkList(Caller);
  return true;
}

/// FoldPHIArgBinOpIntoPHI - If we have something like phi [add (a,b), add(c,d)]
/// and if a/b/c/d and the add's all have a single use, turn this into two phi's
/// and a single binop.
Instruction *InstCombiner::FoldPHIArgBinOpIntoPHI(PHINode &PN) {
  Instruction *FirstInst = cast<Instruction>(PN.getIncomingValue(0));
  assert(isa<BinaryOperator>(FirstInst) || isa<ShiftInst>(FirstInst) ||
         isa<GetElementPtrInst>(FirstInst));
  unsigned Opc = FirstInst->getOpcode();
  const Type *LHSType = FirstInst->getOperand(0)->getType();
  const Type *RHSType = FirstInst->getOperand(1)->getType();
  
  // Scan to see if all operands are the same opcode, all have one use, and all
  // kill their operands (i.e. the operands have one use).
  for (unsigned i = 0; i != PN.getNumIncomingValues(); ++i) {
    Instruction *I = dyn_cast<Instruction>(PN.getIncomingValue(i));
    if (!I || I->getOpcode() != Opc || !I->hasOneUse() ||
        // Verify type of the LHS matches so we don't fold setcc's of different
        // types or GEP's with different index types.
        I->getOperand(0)->getType() != LHSType ||
        I->getOperand(1)->getType() != RHSType)
      return 0;
  }
  
  // Otherwise, this is safe and profitable to transform.  Create two phi nodes.
  PHINode *NewLHS = new PHINode(FirstInst->getOperand(0)->getType(),
                                FirstInst->getOperand(0)->getName()+".pn");
  NewLHS->reserveOperandSpace(PN.getNumOperands()/2);
  PHINode *NewRHS = new PHINode(FirstInst->getOperand(1)->getType(),
                                FirstInst->getOperand(1)->getName()+".pn");
  NewRHS->reserveOperandSpace(PN.getNumOperands()/2);
  
  Value *InLHS = FirstInst->getOperand(0);
  NewLHS->addIncoming(InLHS, PN.getIncomingBlock(0));
  Value *InRHS = FirstInst->getOperand(1);
  NewRHS->addIncoming(InRHS, PN.getIncomingBlock(0));
  
  // Add all operands to the new PHsI.
  for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
    Value *NewInLHS = cast<Instruction>(PN.getIncomingValue(i))->getOperand(0);
    Value *NewInRHS = cast<Instruction>(PN.getIncomingValue(i))->getOperand(1);
    if (NewInLHS != InLHS) InLHS = 0;
    if (NewInRHS != InRHS) InRHS = 0;
    NewLHS->addIncoming(NewInLHS, PN.getIncomingBlock(i));
    NewRHS->addIncoming(NewInRHS, PN.getIncomingBlock(i));
  }
  
  Value *LHSVal;
  if (InLHS) {
    // The new PHI unions all of the same values together.  This is really
    // common, so we handle it intelligently here for compile-time speed.
    LHSVal = InLHS;
    delete NewLHS;
  } else {
    InsertNewInstBefore(NewLHS, PN);
    LHSVal = NewLHS;
  }
  Value *RHSVal;
  if (InRHS) {
    // The new PHI unions all of the same values together.  This is really
    // common, so we handle it intelligently here for compile-time speed.
    RHSVal = InRHS;
    delete NewRHS;
  } else {
    InsertNewInstBefore(NewRHS, PN);
    RHSVal = NewRHS;
  }
  
  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(FirstInst))
    return BinaryOperator::create(BinOp->getOpcode(), LHSVal, RHSVal);
  else if (ShiftInst *SI = dyn_cast<ShiftInst>(FirstInst))
    return new ShiftInst(SI->getOpcode(), LHSVal, RHSVal);
  else {
    assert(isa<GetElementPtrInst>(FirstInst));
    return new GetElementPtrInst(LHSVal, RHSVal);
  }
}

/// isSafeToSinkLoad - Return true if we know that it is safe sink the load out
/// of the block that defines it.  This means that it must be obvious the value
/// of the load is not changed from the point of the load to the end of the
/// block it is in.
static bool isSafeToSinkLoad(LoadInst *L) {
  BasicBlock::iterator BBI = L, E = L->getParent()->end();
  
  for (++BBI; BBI != E; ++BBI)
    if (BBI->mayWriteToMemory())
      return false;
  return true;
}


// FoldPHIArgOpIntoPHI - If all operands to a PHI node are the same "unary"
// operator and they all are only used by the PHI, PHI together their
// inputs, and do the operation once, to the result of the PHI.
Instruction *InstCombiner::FoldPHIArgOpIntoPHI(PHINode &PN) {
  Instruction *FirstInst = cast<Instruction>(PN.getIncomingValue(0));

  // Scan the instruction, looking for input operations that can be folded away.
  // If all input operands to the phi are the same instruction (e.g. a cast from
  // the same type or "+42") we can pull the operation through the PHI, reducing
  // code size and simplifying code.
  Constant *ConstantOp = 0;
  const Type *CastSrcTy = 0;
  bool isVolatile = false;
  if (isa<CastInst>(FirstInst)) {
    CastSrcTy = FirstInst->getOperand(0)->getType();
  } else if (isa<BinaryOperator>(FirstInst) || isa<ShiftInst>(FirstInst)) {
    // Can fold binop or shift here if the RHS is a constant, otherwise call
    // FoldPHIArgBinOpIntoPHI.
    ConstantOp = dyn_cast<Constant>(FirstInst->getOperand(1));
    if (ConstantOp == 0)
      return FoldPHIArgBinOpIntoPHI(PN);
  } else if (LoadInst *LI = dyn_cast<LoadInst>(FirstInst)) {
    isVolatile = LI->isVolatile();
    // We can't sink the load if the loaded value could be modified between the
    // load and the PHI.
    if (LI->getParent() != PN.getIncomingBlock(0) ||
        !isSafeToSinkLoad(LI))
      return 0;
  } else if (isa<GetElementPtrInst>(FirstInst)) {
    if (FirstInst->getNumOperands() == 2)
      return FoldPHIArgBinOpIntoPHI(PN);
    // Can't handle general GEPs yet.
    return 0;
  } else {
    return 0;  // Cannot fold this operation.
  }

  // Check to see if all arguments are the same operation.
  for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
    if (!isa<Instruction>(PN.getIncomingValue(i))) return 0;
    Instruction *I = cast<Instruction>(PN.getIncomingValue(i));
    if (!I->hasOneUse() || I->getOpcode() != FirstInst->getOpcode())
      return 0;
    if (CastSrcTy) {
      if (I->getOperand(0)->getType() != CastSrcTy)
        return 0;  // Cast operation must match.
    } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      // We can't sink the load if the loaded value could be modified between the
      // load and the PHI.
      if (LI->isVolatile() != isVolatile ||
          LI->getParent() != PN.getIncomingBlock(i) ||
          !isSafeToSinkLoad(LI))
        return 0;
    } else if (I->getOperand(1) != ConstantOp) {
      return 0;
    }
  }

  // Okay, they are all the same operation.  Create a new PHI node of the
  // correct type, and PHI together all of the LHS's of the instructions.
  PHINode *NewPN = new PHINode(FirstInst->getOperand(0)->getType(),
                               PN.getName()+".in");
  NewPN->reserveOperandSpace(PN.getNumOperands()/2);

  Value *InVal = FirstInst->getOperand(0);
  NewPN->addIncoming(InVal, PN.getIncomingBlock(0));

  // Add all operands to the new PHI.
  for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
    Value *NewInVal = cast<Instruction>(PN.getIncomingValue(i))->getOperand(0);
    if (NewInVal != InVal)
      InVal = 0;
    NewPN->addIncoming(NewInVal, PN.getIncomingBlock(i));
  }

  Value *PhiVal;
  if (InVal) {
    // The new PHI unions all of the same values together.  This is really
    // common, so we handle it intelligently here for compile-time speed.
    PhiVal = InVal;
    delete NewPN;
  } else {
    InsertNewInstBefore(NewPN, PN);
    PhiVal = NewPN;
  }

  // Insert and return the new operation.
  if (isa<CastInst>(FirstInst))
    return new CastInst(PhiVal, PN.getType());
  else if (isa<LoadInst>(FirstInst))
    return new LoadInst(PhiVal, "", isVolatile);
  else if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(FirstInst))
    return BinaryOperator::create(BinOp->getOpcode(), PhiVal, ConstantOp);
  else
    return new ShiftInst(cast<ShiftInst>(FirstInst)->getOpcode(),
                         PhiVal, ConstantOp);
}

/// DeadPHICycle - Return true if this PHI node is only used by a PHI node cycle
/// that is dead.
static bool DeadPHICycle(PHINode *PN, std::set<PHINode*> &PotentiallyDeadPHIs) {
  if (PN->use_empty()) return true;
  if (!PN->hasOneUse()) return false;

  // Remember this node, and if we find the cycle, return.
  if (!PotentiallyDeadPHIs.insert(PN).second)
    return true;

  if (PHINode *PU = dyn_cast<PHINode>(PN->use_back()))
    return DeadPHICycle(PU, PotentiallyDeadPHIs);

  return false;
}

// PHINode simplification
//
Instruction *InstCombiner::visitPHINode(PHINode &PN) {
  // If LCSSA is around, don't mess with Phi nodes
  if (mustPreserveAnalysisID(LCSSAID)) return 0;
  
  if (Value *V = PN.hasConstantValue())
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

  // If all PHI operands are the same operation, pull them through the PHI,
  // reducing code size.
  if (isa<Instruction>(PN.getIncomingValue(0)) &&
      PN.getIncomingValue(0)->hasOneUse())
    if (Instruction *Result = FoldPHIArgOpIntoPHI(PN))
      return Result;

  // If this is a trivial cycle in the PHI node graph, remove it.  Basically, if
  // this PHI only has a single use (a PHI), and if that PHI only has one use (a
  // PHI)... break the cycle.
  if (PN.hasOneUse())
    if (PHINode *PU = dyn_cast<PHINode>(PN.use_back())) {
      std::set<PHINode*> PotentiallyDeadPHIs;
      PotentiallyDeadPHIs.insert(&PN);
      if (DeadPHICycle(PU, PotentiallyDeadPHIs))
        return ReplaceInstUsesWith(PN, UndefValue::get(PN.getType()));
    }

  return 0;
}

static Value *InsertSignExtendToPtrTy(Value *V, const Type *DTy,
                                      Instruction *InsertPoint,
                                      InstCombiner *IC) {
  unsigned PS = IC->getTargetData().getPointerSize();
  const Type *VTy = V->getType();
  if (!VTy->isSigned() && VTy->getPrimitiveSize() < PS)
    // We must insert a cast to ensure we sign-extend.
    V = IC->InsertCastBefore(V, VTy->getSignedVersion(), *InsertPoint);
  return IC->InsertCastBefore(V, DTy, *InsertPoint);
}


Instruction *InstCombiner::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  Value *PtrOp = GEP.getOperand(0);
  // Is it 'getelementptr %P, long 0'  or 'getelementptr %P'
  // If so, eliminate the noop.
  if (GEP.getNumOperands() == 1)
    return ReplaceInstUsesWith(GEP, PtrOp);

  if (isa<UndefValue>(GEP.getOperand(0)))
    return ReplaceInstUsesWith(GEP, UndefValue::get(GEP.getType()));

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
          if (SrcTy->getPrimitiveSizeInBits() ==
                       DestTy->getPrimitiveSizeInBits()) {
            // We can always eliminate a cast from ulong or long to the other.
            // We can always eliminate a cast from uint to int or the other on
            // 32-bit pointer platforms.
            if (DestTy->getPrimitiveSizeInBits() >= TD->getPointerSizeInBits()){
              MadeChange = true;
              GEP.setOperand(i, Src);
            }
          } else if (SrcTy->getPrimitiveSize() < DestTy->getPrimitiveSize() &&
                     SrcTy->getPrimitiveSize() == 4) {
            // We can always eliminate a cast from int to [u]long.  We can
            // eliminate a cast from uint to [u]long iff the target is a 32-bit
            // pointer target.
            if (SrcTy->isSigned() ||
                SrcTy->getPrimitiveSizeInBits() >= TD->getPointerSizeInBits()) {
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
          Op = InsertCastBefore(Op, TD->getIntPtrType(), GEP);
          GEP.setOperand(i, Op);
          MadeChange = true;
        }

      // If this is a constant idx, make sure to canonicalize it to be a signed
      // operand, otherwise CSE and other optimizations are pessimized.
      if (ConstantInt *CUI = dyn_cast<ConstantInt>(Op))
        if (CUI->getType()->isUnsigned()) {
          GEP.setOperand(i, 
            ConstantExpr::getCast(CUI, CUI->getType()->getSignedVersion()));
          MadeChange = true;
        }
    }
  if (MadeChange) return &GEP;

  // Combine Indices - If the source pointer to this getelementptr instruction
  // is a getelementptr instruction, combine the indices of the two
  // getelementptr instructions into a single instruction.
  //
  std::vector<Value*> SrcGEPOperands;
  if (User *Src = dyn_castGetElementPtr(PtrOp))
    SrcGEPOperands.assign(Src->op_begin(), Src->op_end());

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
  } else if (Value *X = isCast(PtrOp)) {  // Is the operand a cast?
    if (!isa<PointerType>(X->getType())) {
      // Not interesting.  Source pointer must be a cast from pointer.
    } else if (HasZeroPointerIndex) {
      // transform: GEP (cast [10 x ubyte]* X to [0 x ubyte]*), long 0, ...
      // into     : GEP [10 x ubyte]* X, long 0, ...
      //
      // This occurs when the program declares an array extern like "int X[];"
      //
      const PointerType *CPTy = cast<PointerType>(PtrOp->getType());
      const PointerType *XTy = cast<PointerType>(X->getType());
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
    } else if (GEP.getNumOperands() == 2) {
      // Transform things like:
      // %t = getelementptr ubyte* cast ([2 x int]* %str to uint*), uint %V
      // into:  %t1 = getelementptr [2 x int*]* %str, int 0, uint %V; cast
      const Type *SrcElTy = cast<PointerType>(X->getType())->getElementType();
      const Type *ResElTy=cast<PointerType>(PtrOp->getType())->getElementType();
      if (isa<ArrayType>(SrcElTy) &&
          TD->getTypeSize(cast<ArrayType>(SrcElTy)->getElementType()) ==
          TD->getTypeSize(ResElTy)) {
        Value *V = InsertNewInstBefore(
               new GetElementPtrInst(X, Constant::getNullValue(Type::IntTy),
                                     GEP.getOperand(1), GEP.getName()), GEP);
        return new CastInst(V, GEP.getType());
      }
      
      // Transform things like:
      // getelementptr sbyte* cast ([100 x double]* X to sbyte*), int %tmp
      //   (where tmp = 8*tmp2) into:
      // getelementptr [100 x double]* %arr, int 0, int %tmp.2
      
      if (isa<ArrayType>(SrcElTy) &&
          (ResElTy == Type::SByteTy || ResElTy == Type::UByteTy)) {
        uint64_t ArrayEltSize =
            TD->getTypeSize(cast<ArrayType>(SrcElTy)->getElementType());
        
        // Check to see if "tmp" is a scale by a multiple of ArrayEltSize.  We
        // allow either a mul, shift, or constant here.
        Value *NewIdx = 0;
        ConstantInt *Scale = 0;
        if (ArrayEltSize == 1) {
          NewIdx = GEP.getOperand(1);
          Scale = ConstantInt::get(NewIdx->getType(), 1);
        } else if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP.getOperand(1))) {
          NewIdx = ConstantInt::get(CI->getType(), 1);
          Scale = CI;
        } else if (Instruction *Inst =dyn_cast<Instruction>(GEP.getOperand(1))){
          if (Inst->getOpcode() == Instruction::Shl &&
              isa<ConstantInt>(Inst->getOperand(1))) {
            unsigned ShAmt =
              cast<ConstantInt>(Inst->getOperand(1))->getZExtValue();
            if (Inst->getType()->isSigned())
              Scale = ConstantInt::get(Inst->getType(), 1ULL << ShAmt);
            else
              Scale = ConstantInt::get(Inst->getType(), 1ULL << ShAmt);
            NewIdx = Inst->getOperand(0);
          } else if (Inst->getOpcode() == Instruction::Mul &&
                     isa<ConstantInt>(Inst->getOperand(1))) {
            Scale = cast<ConstantInt>(Inst->getOperand(1));
            NewIdx = Inst->getOperand(0);
          }
        }

        // If the index will be to exactly the right offset with the scale taken
        // out, perform the transformation.
        if (Scale && Scale->getZExtValue() % ArrayEltSize == 0) {
          if (isa<ConstantInt>(Scale))
            Scale = ConstantInt::get(Scale->getType(),
                                      Scale->getZExtValue() / ArrayEltSize);
          if (Scale->getZExtValue() != 1) {
            Constant *C = ConstantExpr::getCast(Scale, NewIdx->getType());
            Instruction *Sc = BinaryOperator::createMul(NewIdx, C, "idxscale");
            NewIdx = InsertNewInstBefore(Sc, GEP);
          }

          // Insert the new GEP instruction.
          Instruction *Idx =
            new GetElementPtrInst(X, Constant::getNullValue(Type::IntTy),
                                  NewIdx, GEP.getName());
          Idx = InsertNewInstBefore(Idx, GEP);
          return new CastInst(Idx, GEP.getType());
        }
      }
    }
  }

  return 0;
}

Instruction *InstCombiner::visitAllocationInst(AllocationInst &AI) {
  // Convert: malloc Ty, C - where C is a constant != 1 into: malloc [C x Ty], 1
  if (AI.isArrayAllocation())    // Check C != 1
    if (const ConstantInt *C = dyn_cast<ConstantInt>(AI.getArraySize())) {
      const Type *NewTy = 
        ArrayType::get(AI.getAllocatedType(), C->getZExtValue());
      AllocationInst *New = 0;

      // Create and insert the replacement instruction...
      if (isa<MallocInst>(AI))
        New = new MallocInst(NewTy, 0, AI.getAlignment(), AI.getName());
      else {
        assert(isa<AllocaInst>(AI) && "Unknown type of allocation inst!");
        New = new AllocaInst(NewTy, 0, AI.getAlignment(), AI.getName());
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
      Value *NullIdx = Constant::getNullValue(Type::IntTy);
      Value *V = new GetElementPtrInst(New, NullIdx, NullIdx,
                                       New->getName()+".sub", It);

      // Now make everything use the getelementptr instead of the original
      // allocation.
      return ReplaceInstUsesWith(AI, V);
    } else if (isa<UndefValue>(AI.getArraySize())) {
      return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));
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

  // free undef -> unreachable.
  if (isa<UndefValue>(Op)) {
    // Insert a new store to null because we cannot modify the CFG here.
    new StoreInst(ConstantBool::getTrue(),
                  UndefValue::get(PointerType::get(Type::BoolTy)), &FI);
    return EraseInstFromFunction(FI);
  }

  // If we have 'free null' delete the instruction.  This can happen in stl code
  // when lots of inlining happens.
  if (isa<ConstantPointerNull>(Op))
    return EraseInstFromFunction(FI);

  return 0;
}


/// InstCombineLoadCast - Fold 'load (cast P)' -> cast (load P)' when possible.
static Instruction *InstCombineLoadCast(InstCombiner &IC, LoadInst &LI) {
  User *CI = cast<User>(LI.getOperand(0));
  Value *CastOp = CI->getOperand(0);

  const Type *DestPTy = cast<PointerType>(CI->getType())->getElementType();
  if (const PointerType *SrcTy = dyn_cast<PointerType>(CastOp->getType())) {
    const Type *SrcPTy = SrcTy->getElementType();

    if (DestPTy->isInteger() || isa<PointerType>(DestPTy) || 
        isa<PackedType>(DestPTy)) {
      // If the source is an array, the code below will not succeed.  Check to
      // see if a trivial 'gep P, 0, 0' will help matters.  Only do this for
      // constants.
      if (const ArrayType *ASrcTy = dyn_cast<ArrayType>(SrcPTy))
        if (Constant *CSrc = dyn_cast<Constant>(CastOp))
          if (ASrcTy->getNumElements() != 0) {
            std::vector<Value*> Idxs(2, Constant::getNullValue(Type::IntTy));
            CastOp = ConstantExpr::getGetElementPtr(CSrc, Idxs);
            SrcTy = cast<PointerType>(CastOp->getType());
            SrcPTy = SrcTy->getElementType();
          }

      if ((SrcPTy->isInteger() || isa<PointerType>(SrcPTy) || 
           isa<PackedType>(SrcPTy)) &&
          // Do not allow turning this into a load of an integer, which is then
          // casted to a pointer, this pessimizes pointer analysis a lot.
          (isa<PointerType>(SrcPTy) == isa<PointerType>(LI.getType())) &&
          IC.getTargetData().getTypeSize(SrcPTy) ==
               IC.getTargetData().getTypeSize(DestPTy)) {

        // Okay, we are casting from one integer or pointer type to another of
        // the same size.  Instead of casting the pointer before the load, cast
        // the result of the loaded value.
        Value *NewLoad = IC.InsertNewInstBefore(new LoadInst(CastOp,
                                                             CI->getName(),
                                                         LI.isVolatile()),LI);
        // Now cast the result of the load.
        return new CastInst(NewLoad, LI.getType());
      }
    }
  }
  return 0;
}

/// isSafeToLoadUnconditionally - Return true if we know that executing a load
/// from this value cannot trap.  If it is not obviously safe to load from the
/// specified pointer, we do a quick local scan of the basic block containing
/// ScanFrom, to determine if the address is already accessed.
static bool isSafeToLoadUnconditionally(Value *V, Instruction *ScanFrom) {
  // If it is an alloca or global variable, it is always safe to load from.
  if (isa<AllocaInst>(V) || isa<GlobalVariable>(V)) return true;

  // Otherwise, be a little bit agressive by scanning the local block where we
  // want to check to see if the pointer is already being loaded or stored
  // from/to.  If so, the previous load or store would have already trapped,
  // so there is no harm doing an extra load (also, CSE will later eliminate
  // the load entirely).
  BasicBlock::iterator BBI = ScanFrom, E = ScanFrom->getParent()->begin();

  while (BBI != E) {
    --BBI;

    if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
      if (LI->getOperand(0) == V) return true;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(BBI))
      if (SI->getOperand(1) == V) return true;

  }
  return false;
}

Instruction *InstCombiner::visitLoadInst(LoadInst &LI) {
  Value *Op = LI.getOperand(0);

  // load (cast X) --> cast (load X) iff safe
  if (isa<CastInst>(Op))
    if (Instruction *Res = InstCombineLoadCast(*this, LI))
      return Res;

  // None of the following transforms are legal for volatile loads.
  if (LI.isVolatile()) return 0;
  
  if (&LI.getParent()->front() != &LI) {
    BasicBlock::iterator BBI = &LI; --BBI;
    // If the instruction immediately before this is a store to the same
    // address, do a simple form of store->load forwarding.
    if (StoreInst *SI = dyn_cast<StoreInst>(BBI))
      if (SI->getOperand(1) == LI.getOperand(0))
        return ReplaceInstUsesWith(LI, SI->getOperand(0));
    if (LoadInst *LIB = dyn_cast<LoadInst>(BBI))
      if (LIB->getOperand(0) == LI.getOperand(0))
        return ReplaceInstUsesWith(LI, LIB);
  }

  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(Op))
    if (isa<ConstantPointerNull>(GEPI->getOperand(0)) ||
        isa<UndefValue>(GEPI->getOperand(0))) {
      // Insert a new store to null instruction before the load to indicate
      // that this code is not reachable.  We do this instead of inserting
      // an unreachable instruction directly because we cannot modify the
      // CFG.
      new StoreInst(UndefValue::get(LI.getType()),
                    Constant::getNullValue(Op->getType()), &LI);
      return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
    }

  if (Constant *C = dyn_cast<Constant>(Op)) {
    // load null/undef -> undef
    if ((C->isNullValue() || isa<UndefValue>(C))) {
      // Insert a new store to null instruction before the load to indicate that
      // this code is not reachable.  We do this instead of inserting an
      // unreachable instruction directly because we cannot modify the CFG.
      new StoreInst(UndefValue::get(LI.getType()),
                    Constant::getNullValue(Op->getType()), &LI);
      return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
    }

    // Instcombine load (constant global) into the value loaded.
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Op))
      if (GV->isConstant() && !GV->isExternal())
        return ReplaceInstUsesWith(LI, GV->getInitializer());

    // Instcombine load (constantexpr_GEP global, 0, ...) into the value loaded.
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Op))
      if (CE->getOpcode() == Instruction::GetElementPtr) {
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(CE->getOperand(0)))
          if (GV->isConstant() && !GV->isExternal())
            if (Constant *V = 
               ConstantFoldLoadThroughGEPConstantExpr(GV->getInitializer(), CE))
              return ReplaceInstUsesWith(LI, V);
        if (CE->getOperand(0)->isNullValue()) {
          // Insert a new store to null instruction before the load to indicate
          // that this code is not reachable.  We do this instead of inserting
          // an unreachable instruction directly because we cannot modify the
          // CFG.
          new StoreInst(UndefValue::get(LI.getType()),
                        Constant::getNullValue(Op->getType()), &LI);
          return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
        }

      } else if (CE->getOpcode() == Instruction::Cast) {
        if (Instruction *Res = InstCombineLoadCast(*this, LI))
          return Res;
      }
  }

  if (Op->hasOneUse()) {
    // Change select and PHI nodes to select values instead of addresses: this
    // helps alias analysis out a lot, allows many others simplifications, and
    // exposes redundancy in the code.
    //
    // Note that we cannot do the transformation unless we know that the
    // introduced loads cannot trap!  Something like this is valid as long as
    // the condition is always false: load (select bool %C, int* null, int* %G),
    // but it would not be valid if we transformed it to load from null
    // unconditionally.
    //
    if (SelectInst *SI = dyn_cast<SelectInst>(Op)) {
      // load (select (Cond, &V1, &V2))  --> select(Cond, load &V1, load &V2).
      if (isSafeToLoadUnconditionally(SI->getOperand(1), SI) &&
          isSafeToLoadUnconditionally(SI->getOperand(2), SI)) {
        Value *V1 = InsertNewInstBefore(new LoadInst(SI->getOperand(1),
                                     SI->getOperand(1)->getName()+".val"), LI);
        Value *V2 = InsertNewInstBefore(new LoadInst(SI->getOperand(2),
                                     SI->getOperand(2)->getName()+".val"), LI);
        return new SelectInst(SI->getCondition(), V1, V2);
      }

      // load (select (cond, null, P)) -> load P
      if (Constant *C = dyn_cast<Constant>(SI->getOperand(1)))
        if (C->isNullValue()) {
          LI.setOperand(0, SI->getOperand(2));
          return &LI;
        }

      // load (select (cond, P, null)) -> load P
      if (Constant *C = dyn_cast<Constant>(SI->getOperand(2)))
        if (C->isNullValue()) {
          LI.setOperand(0, SI->getOperand(1));
          return &LI;
        }
    }
  }
  return 0;
}

/// InstCombineStoreToCast - Fold 'store V, (cast P)' -> store (cast V), P'
/// when possible.
static Instruction *InstCombineStoreToCast(InstCombiner &IC, StoreInst &SI) {
  User *CI = cast<User>(SI.getOperand(1));
  Value *CastOp = CI->getOperand(0);

  const Type *DestPTy = cast<PointerType>(CI->getType())->getElementType();
  if (const PointerType *SrcTy = dyn_cast<PointerType>(CastOp->getType())) {
    const Type *SrcPTy = SrcTy->getElementType();

    if (DestPTy->isInteger() || isa<PointerType>(DestPTy)) {
      // If the source is an array, the code below will not succeed.  Check to
      // see if a trivial 'gep P, 0, 0' will help matters.  Only do this for
      // constants.
      if (const ArrayType *ASrcTy = dyn_cast<ArrayType>(SrcPTy))
        if (Constant *CSrc = dyn_cast<Constant>(CastOp))
          if (ASrcTy->getNumElements() != 0) {
            std::vector<Value*> Idxs(2, Constant::getNullValue(Type::IntTy));
            CastOp = ConstantExpr::getGetElementPtr(CSrc, Idxs);
            SrcTy = cast<PointerType>(CastOp->getType());
            SrcPTy = SrcTy->getElementType();
          }

      if ((SrcPTy->isInteger() || isa<PointerType>(SrcPTy)) &&
          IC.getTargetData().getTypeSize(SrcPTy) ==
               IC.getTargetData().getTypeSize(DestPTy)) {

        // Okay, we are casting from one integer or pointer type to another of
        // the same size.  Instead of casting the pointer before the store, cast
        // the value to be stored.
        Value *NewCast;
        if (Constant *C = dyn_cast<Constant>(SI.getOperand(0)))
          NewCast = ConstantExpr::getCast(C, SrcPTy);
        else
          NewCast = IC.InsertNewInstBefore(new CastInst(SI.getOperand(0),
                                                        SrcPTy,
                                         SI.getOperand(0)->getName()+".c"), SI);

        return new StoreInst(NewCast, CastOp);
      }
    }
  }
  return 0;
}

Instruction *InstCombiner::visitStoreInst(StoreInst &SI) {
  Value *Val = SI.getOperand(0);
  Value *Ptr = SI.getOperand(1);

  if (isa<UndefValue>(Ptr)) {     // store X, undef -> noop (even if volatile)
    EraseInstFromFunction(SI);
    ++NumCombined;
    return 0;
  }

  // Do really simple DSE, to catch cases where there are several consequtive
  // stores to the same location, separated by a few arithmetic operations. This
  // situation often occurs with bitfield accesses.
  BasicBlock::iterator BBI = &SI;
  for (unsigned ScanInsts = 6; BBI != SI.getParent()->begin() && ScanInsts;
       --ScanInsts) {
    --BBI;
    
    if (StoreInst *PrevSI = dyn_cast<StoreInst>(BBI)) {
      // Prev store isn't volatile, and stores to the same location?
      if (!PrevSI->isVolatile() && PrevSI->getOperand(1) == SI.getOperand(1)) {
        ++NumDeadStore;
        ++BBI;
        EraseInstFromFunction(*PrevSI);
        continue;
      }
      break;
    }
    
    // If this is a load, we have to stop.  However, if the loaded value is from
    // the pointer we're loading and is producing the pointer we're storing,
    // then *this* store is dead (X = load P; store X -> P).
    if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
      if (LI == Val && LI->getOperand(0) == Ptr) {
        EraseInstFromFunction(SI);
        ++NumCombined;
        return 0;
      }
      // Otherwise, this is a load from some other location.  Stores before it
      // may not be dead.
      break;
    }
    
    // Don't skip over loads or things that can modify memory.
    if (BBI->mayWriteToMemory())
      break;
  }
  
  
  if (SI.isVolatile()) return 0;  // Don't hack volatile stores.

  // store X, null    -> turns into 'unreachable' in SimplifyCFG
  if (isa<ConstantPointerNull>(Ptr)) {
    if (!isa<UndefValue>(Val)) {
      SI.setOperand(0, UndefValue::get(Val->getType()));
      if (Instruction *U = dyn_cast<Instruction>(Val))
        WorkList.push_back(U);  // Dropped a use.
      ++NumCombined;
    }
    return 0;  // Do not modify these!
  }

  // store undef, Ptr -> noop
  if (isa<UndefValue>(Val)) {
    EraseInstFromFunction(SI);
    ++NumCombined;
    return 0;
  }

  // If the pointer destination is a cast, see if we can fold the cast into the
  // source instead.
  if (isa<CastInst>(Ptr))
    if (Instruction *Res = InstCombineStoreToCast(*this, SI))
      return Res;
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr))
    if (CE->getOpcode() == Instruction::Cast)
      if (Instruction *Res = InstCombineStoreToCast(*this, SI))
        return Res;

  
  // If this store is the last instruction in the basic block, and if the block
  // ends with an unconditional branch, try to move it to the successor block.
  BBI = &SI; ++BBI;
  if (BranchInst *BI = dyn_cast<BranchInst>(BBI))
    if (BI->isUnconditional()) {
      // Check to see if the successor block has exactly two incoming edges.  If
      // so, see if the other predecessor contains a store to the same location.
      // if so, insert a PHI node (if needed) and move the stores down.
      BasicBlock *Dest = BI->getSuccessor(0);

      pred_iterator PI = pred_begin(Dest);
      BasicBlock *Other = 0;
      if (*PI != BI->getParent())
        Other = *PI;
      ++PI;
      if (PI != pred_end(Dest)) {
        if (*PI != BI->getParent())
          if (Other)
            Other = 0;
          else
            Other = *PI;
        if (++PI != pred_end(Dest))
          Other = 0;
      }
      if (Other) {  // If only one other pred...
        BBI = Other->getTerminator();
        // Make sure this other block ends in an unconditional branch and that
        // there is an instruction before the branch.
        if (isa<BranchInst>(BBI) && cast<BranchInst>(BBI)->isUnconditional() &&
            BBI != Other->begin()) {
          --BBI;
          StoreInst *OtherStore = dyn_cast<StoreInst>(BBI);
          
          // If this instruction is a store to the same location.
          if (OtherStore && OtherStore->getOperand(1) == SI.getOperand(1)) {
            // Okay, we know we can perform this transformation.  Insert a PHI
            // node now if we need it.
            Value *MergedVal = OtherStore->getOperand(0);
            if (MergedVal != SI.getOperand(0)) {
              PHINode *PN = new PHINode(MergedVal->getType(), "storemerge");
              PN->reserveOperandSpace(2);
              PN->addIncoming(SI.getOperand(0), SI.getParent());
              PN->addIncoming(OtherStore->getOperand(0), Other);
              MergedVal = InsertNewInstBefore(PN, Dest->front());
            }
            
            // Advance to a place where it is safe to insert the new store and
            // insert it.
            BBI = Dest->begin();
            while (isa<PHINode>(BBI)) ++BBI;
            InsertNewInstBefore(new StoreInst(MergedVal, SI.getOperand(1),
                                              OtherStore->isVolatile()), *BBI);

            // Nuke the old stores.
            EraseInstFromFunction(SI);
            EraseInstFromFunction(*OtherStore);
            ++NumCombined;
            return 0;
          }
        }
      }
    }
  
  return 0;
}


Instruction *InstCombiner::visitBranchInst(BranchInst &BI) {
  // Change br (not X), label True, label False to: br X, label False, True
  Value *X = 0;
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
          SI.setOperand(i,ConstantExpr::getSub(cast<Constant>(SI.getOperand(i)),
                                                AddRHS));
        SI.setOperand(0, I->getOperand(0));
        WorkList.push_back(I);
        return &SI;
      }
  }
  return 0;
}

/// CheapToScalarize - Return true if the value is cheaper to scalarize than it
/// is to leave as a vector operation.
static bool CheapToScalarize(Value *V, bool isConstant) {
  if (isa<ConstantAggregateZero>(V)) 
    return true;
  if (ConstantPacked *C = dyn_cast<ConstantPacked>(V)) {
    if (isConstant) return true;
    // If all elts are the same, we can extract.
    Constant *Op0 = C->getOperand(0);
    for (unsigned i = 1; i < C->getNumOperands(); ++i)
      if (C->getOperand(i) != Op0)
        return false;
    return true;
  }
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;
  
  // Insert element gets simplified to the inserted element or is deleted if
  // this is constant idx extract element and its a constant idx insertelt.
  if (I->getOpcode() == Instruction::InsertElement && isConstant &&
      isa<ConstantInt>(I->getOperand(2)))
    return true;
  if (I->getOpcode() == Instruction::Load && I->hasOneUse())
    return true;
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I))
    if (BO->hasOneUse() &&
        (CheapToScalarize(BO->getOperand(0), isConstant) ||
         CheapToScalarize(BO->getOperand(1), isConstant)))
      return true;
  
  return false;
}

/// getShuffleMask - Read and decode a shufflevector mask.  It turns undef
/// elements into values that are larger than the #elts in the input.
static std::vector<unsigned> getShuffleMask(const ShuffleVectorInst *SVI) {
  unsigned NElts = SVI->getType()->getNumElements();
  if (isa<ConstantAggregateZero>(SVI->getOperand(2)))
    return std::vector<unsigned>(NElts, 0);
  if (isa<UndefValue>(SVI->getOperand(2)))
    return std::vector<unsigned>(NElts, 2*NElts);

  std::vector<unsigned> Result;
  const ConstantPacked *CP = cast<ConstantPacked>(SVI->getOperand(2));
  for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
    if (isa<UndefValue>(CP->getOperand(i)))
      Result.push_back(NElts*2);  // undef -> 8
    else
      Result.push_back(cast<ConstantInt>(CP->getOperand(i))->getZExtValue());
  return Result;
}

/// FindScalarElement - Given a vector and an element number, see if the scalar
/// value is already around as a register, for example if it were inserted then
/// extracted from the vector.
static Value *FindScalarElement(Value *V, unsigned EltNo) {
  assert(isa<PackedType>(V->getType()) && "Not looking at a vector?");
  const PackedType *PTy = cast<PackedType>(V->getType());
  unsigned Width = PTy->getNumElements();
  if (EltNo >= Width)  // Out of range access.
    return UndefValue::get(PTy->getElementType());
  
  if (isa<UndefValue>(V))
    return UndefValue::get(PTy->getElementType());
  else if (isa<ConstantAggregateZero>(V))
    return Constant::getNullValue(PTy->getElementType());
  else if (ConstantPacked *CP = dyn_cast<ConstantPacked>(V))
    return CP->getOperand(EltNo);
  else if (InsertElementInst *III = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert to a variable element, we don't know what it is.
    if (!isa<ConstantInt>(III->getOperand(2))) 
      return 0;
    unsigned IIElt = cast<ConstantInt>(III->getOperand(2))->getZExtValue();
    
    // If this is an insert to the element we are looking for, return the
    // inserted value.
    if (EltNo == IIElt) 
      return III->getOperand(1);
    
    // Otherwise, the insertelement doesn't modify the value, recurse on its
    // vector input.
    return FindScalarElement(III->getOperand(0), EltNo);
  } else if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(V)) {
    unsigned InEl = getShuffleMask(SVI)[EltNo];
    if (InEl < Width)
      return FindScalarElement(SVI->getOperand(0), InEl);
    else if (InEl < Width*2)
      return FindScalarElement(SVI->getOperand(1), InEl - Width);
    else
      return UndefValue::get(PTy->getElementType());
  }
  
  // Otherwise, we don't know.
  return 0;
}

Instruction *InstCombiner::visitExtractElementInst(ExtractElementInst &EI) {

  // If packed val is undef, replace extract with scalar undef.
  if (isa<UndefValue>(EI.getOperand(0)))
    return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));

  // If packed val is constant 0, replace extract with scalar 0.
  if (isa<ConstantAggregateZero>(EI.getOperand(0)))
    return ReplaceInstUsesWith(EI, Constant::getNullValue(EI.getType()));
  
  if (ConstantPacked *C = dyn_cast<ConstantPacked>(EI.getOperand(0))) {
    // If packed val is constant with uniform operands, replace EI
    // with that operand
    Constant *op0 = C->getOperand(0);
    for (unsigned i = 1; i < C->getNumOperands(); ++i)
      if (C->getOperand(i) != op0) {
        op0 = 0; 
        break;
      }
    if (op0)
      return ReplaceInstUsesWith(EI, op0);
  }
  
  // If extracting a specified index from the vector, see if we can recursively
  // find a previously computed scalar that was inserted into the vector.
  if (ConstantInt *IdxC = dyn_cast<ConstantInt>(EI.getOperand(1))) {
    // This instruction only demands the single element from the input vector.
    // If the input vector has a single use, simplify it based on this use
    // property.
    uint64_t IndexVal = IdxC->getZExtValue();
    if (EI.getOperand(0)->hasOneUse()) {
      uint64_t UndefElts;
      if (Value *V = SimplifyDemandedVectorElts(EI.getOperand(0),
                                                1 << IndexVal,
                                                UndefElts)) {
        EI.setOperand(0, V);
        return &EI;
      }
    }
    
    if (Value *Elt = FindScalarElement(EI.getOperand(0), IndexVal))
      return ReplaceInstUsesWith(EI, Elt);
  }
  
  if (Instruction *I = dyn_cast<Instruction>(EI.getOperand(0))) {
    if (I->hasOneUse()) {
      // Push extractelement into predecessor operation if legal and
      // profitable to do so
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
        bool isConstantElt = isa<ConstantInt>(EI.getOperand(1));
        if (CheapToScalarize(BO, isConstantElt)) {
          ExtractElementInst *newEI0 = 
            new ExtractElementInst(BO->getOperand(0), EI.getOperand(1),
                                   EI.getName()+".lhs");
          ExtractElementInst *newEI1 =
            new ExtractElementInst(BO->getOperand(1), EI.getOperand(1),
                                   EI.getName()+".rhs");
          InsertNewInstBefore(newEI0, EI);
          InsertNewInstBefore(newEI1, EI);
          return BinaryOperator::create(BO->getOpcode(), newEI0, newEI1);
        }
      } else if (isa<LoadInst>(I)) {
        Value *Ptr = InsertCastBefore(I->getOperand(0),
                                      PointerType::get(EI.getType()), EI);
        GetElementPtrInst *GEP = 
          new GetElementPtrInst(Ptr, EI.getOperand(1),
                                I->getName() + ".gep");
        InsertNewInstBefore(GEP, EI);
        return new LoadInst(GEP);
      }
    }
    if (InsertElementInst *IE = dyn_cast<InsertElementInst>(I)) {
      // Extracting the inserted element?
      if (IE->getOperand(2) == EI.getOperand(1))
        return ReplaceInstUsesWith(EI, IE->getOperand(1));
      // If the inserted and extracted elements are constants, they must not
      // be the same value, extract from the pre-inserted value instead.
      if (isa<Constant>(IE->getOperand(2)) &&
          isa<Constant>(EI.getOperand(1))) {
        AddUsesToWorkList(EI);
        EI.setOperand(0, IE->getOperand(0));
        return &EI;
      }
    } else if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(I)) {
      // If this is extracting an element from a shufflevector, figure out where
      // it came from and extract from the appropriate input element instead.
      if (ConstantInt *Elt = dyn_cast<ConstantInt>(EI.getOperand(1))) {
        unsigned SrcIdx = getShuffleMask(SVI)[Elt->getZExtValue()];
        Value *Src;
        if (SrcIdx < SVI->getType()->getNumElements())
          Src = SVI->getOperand(0);
        else if (SrcIdx < SVI->getType()->getNumElements()*2) {
          SrcIdx -= SVI->getType()->getNumElements();
          Src = SVI->getOperand(1);
        } else {
          return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));
        }
        return new ExtractElementInst(Src, SrcIdx);
      }
    }
  }
  return 0;
}

/// CollectSingleShuffleElements - If V is a shuffle of values that ONLY returns
/// elements from either LHS or RHS, return the shuffle mask and true. 
/// Otherwise, return false.
static bool CollectSingleShuffleElements(Value *V, Value *LHS, Value *RHS,
                                         std::vector<Constant*> &Mask) {
  assert(V->getType() == LHS->getType() && V->getType() == RHS->getType() &&
         "Invalid CollectSingleShuffleElements");
  unsigned NumElts = cast<PackedType>(V->getType())->getNumElements();

  if (isa<UndefValue>(V)) {
    Mask.assign(NumElts, UndefValue::get(Type::UIntTy));
    return true;
  } else if (V == LHS) {
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(ConstantInt::get(Type::UIntTy, i));
    return true;
  } else if (V == RHS) {
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(ConstantInt::get(Type::UIntTy, i+NumElts));
    return true;
  } else if (InsertElementInst *IEI = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert of an extract from some other vector, include it.
    Value *VecOp    = IEI->getOperand(0);
    Value *ScalarOp = IEI->getOperand(1);
    Value *IdxOp    = IEI->getOperand(2);
    
    if (!isa<ConstantInt>(IdxOp))
      return false;
    unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();
    
    if (isa<UndefValue>(ScalarOp)) {  // inserting undef into vector.
      // Okay, we can handle this if the vector we are insertinting into is
      // transitively ok.
      if (CollectSingleShuffleElements(VecOp, LHS, RHS, Mask)) {
        // If so, update the mask to reflect the inserted undef.
        Mask[InsertedIdx] = UndefValue::get(Type::UIntTy);
        return true;
      }      
    } else if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)){
      if (isa<ConstantInt>(EI->getOperand(1)) &&
          EI->getOperand(0)->getType() == V->getType()) {
        unsigned ExtractedIdx =
          cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
        
        // This must be extracting from either LHS or RHS.
        if (EI->getOperand(0) == LHS || EI->getOperand(0) == RHS) {
          // Okay, we can handle this if the vector we are insertinting into is
          // transitively ok.
          if (CollectSingleShuffleElements(VecOp, LHS, RHS, Mask)) {
            // If so, update the mask to reflect the inserted value.
            if (EI->getOperand(0) == LHS) {
              Mask[InsertedIdx & (NumElts-1)] = 
                 ConstantInt::get(Type::UIntTy, ExtractedIdx);
            } else {
              assert(EI->getOperand(0) == RHS);
              Mask[InsertedIdx & (NumElts-1)] = 
                ConstantInt::get(Type::UIntTy, ExtractedIdx+NumElts);
              
            }
            return true;
          }
        }
      }
    }
  }
  // TODO: Handle shufflevector here!
  
  return false;
}

/// CollectShuffleElements - We are building a shuffle of V, using RHS as the
/// RHS of the shuffle instruction, if it is not null.  Return a shuffle mask
/// that computes V and the LHS value of the shuffle.
static Value *CollectShuffleElements(Value *V, std::vector<Constant*> &Mask,
                                     Value *&RHS) {
  assert(isa<PackedType>(V->getType()) && 
         (RHS == 0 || V->getType() == RHS->getType()) &&
         "Invalid shuffle!");
  unsigned NumElts = cast<PackedType>(V->getType())->getNumElements();

  if (isa<UndefValue>(V)) {
    Mask.assign(NumElts, UndefValue::get(Type::UIntTy));
    return V;
  } else if (isa<ConstantAggregateZero>(V)) {
    Mask.assign(NumElts, ConstantInt::get(Type::UIntTy, 0));
    return V;
  } else if (InsertElementInst *IEI = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert of an extract from some other vector, include it.
    Value *VecOp    = IEI->getOperand(0);
    Value *ScalarOp = IEI->getOperand(1);
    Value *IdxOp    = IEI->getOperand(2);
    
    if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)) {
      if (isa<ConstantInt>(EI->getOperand(1)) && isa<ConstantInt>(IdxOp) &&
          EI->getOperand(0)->getType() == V->getType()) {
        unsigned ExtractedIdx =
          cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
        unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();
        
        // Either the extracted from or inserted into vector must be RHSVec,
        // otherwise we'd end up with a shuffle of three inputs.
        if (EI->getOperand(0) == RHS || RHS == 0) {
          RHS = EI->getOperand(0);
          Value *V = CollectShuffleElements(VecOp, Mask, RHS);
          Mask[InsertedIdx & (NumElts-1)] = 
            ConstantInt::get(Type::UIntTy, NumElts+ExtractedIdx);
          return V;
        }
        
        if (VecOp == RHS) {
          Value *V = CollectShuffleElements(EI->getOperand(0), Mask, RHS);
          // Everything but the extracted element is replaced with the RHS.
          for (unsigned i = 0; i != NumElts; ++i) {
            if (i != InsertedIdx)
              Mask[i] = ConstantInt::get(Type::UIntTy, NumElts+i);
          }
          return V;
        }
        
        // If this insertelement is a chain that comes from exactly these two
        // vectors, return the vector and the effective shuffle.
        if (CollectSingleShuffleElements(IEI, EI->getOperand(0), RHS, Mask))
          return EI->getOperand(0);
        
      }
    }
  }
  // TODO: Handle shufflevector here!
  
  // Otherwise, can't do anything fancy.  Return an identity vector.
  for (unsigned i = 0; i != NumElts; ++i)
    Mask.push_back(ConstantInt::get(Type::UIntTy, i));
  return V;
}

Instruction *InstCombiner::visitInsertElementInst(InsertElementInst &IE) {
  Value *VecOp    = IE.getOperand(0);
  Value *ScalarOp = IE.getOperand(1);
  Value *IdxOp    = IE.getOperand(2);
  
  // If the inserted element was extracted from some other vector, and if the 
  // indexes are constant, try to turn this into a shufflevector operation.
  if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)) {
    if (isa<ConstantInt>(EI->getOperand(1)) && isa<ConstantInt>(IdxOp) &&
        EI->getOperand(0)->getType() == IE.getType()) {
      unsigned NumVectorElts = IE.getType()->getNumElements();
      unsigned ExtractedIdx=cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
      unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();
      
      if (ExtractedIdx >= NumVectorElts) // Out of range extract.
        return ReplaceInstUsesWith(IE, VecOp);
      
      if (InsertedIdx >= NumVectorElts)  // Out of range insert.
        return ReplaceInstUsesWith(IE, UndefValue::get(IE.getType()));
      
      // If we are extracting a value from a vector, then inserting it right
      // back into the same place, just use the input vector.
      if (EI->getOperand(0) == VecOp && ExtractedIdx == InsertedIdx)
        return ReplaceInstUsesWith(IE, VecOp);      
      
      // We could theoretically do this for ANY input.  However, doing so could
      // turn chains of insertelement instructions into a chain of shufflevector
      // instructions, and right now we do not merge shufflevectors.  As such,
      // only do this in a situation where it is clear that there is benefit.
      if (isa<UndefValue>(VecOp) || isa<ConstantAggregateZero>(VecOp)) {
        // Turn this into shuffle(EIOp0, VecOp, Mask).  The result has all of
        // the values of VecOp, except then one read from EIOp0.
        // Build a new shuffle mask.
        std::vector<Constant*> Mask;
        if (isa<UndefValue>(VecOp))
          Mask.assign(NumVectorElts, UndefValue::get(Type::UIntTy));
        else {
          assert(isa<ConstantAggregateZero>(VecOp) && "Unknown thing");
          Mask.assign(NumVectorElts, ConstantInt::get(Type::UIntTy,
                                                       NumVectorElts));
        } 
        Mask[InsertedIdx] = ConstantInt::get(Type::UIntTy, ExtractedIdx);
        return new ShuffleVectorInst(EI->getOperand(0), VecOp,
                                     ConstantPacked::get(Mask));
      }
      
      // If this insertelement isn't used by some other insertelement, turn it
      // (and any insertelements it points to), into one big shuffle.
      if (!IE.hasOneUse() || !isa<InsertElementInst>(IE.use_back())) {
        std::vector<Constant*> Mask;
        Value *RHS = 0;
        Value *LHS = CollectShuffleElements(&IE, Mask, RHS);
        if (RHS == 0) RHS = UndefValue::get(LHS->getType());
        // We now have a shuffle of LHS, RHS, Mask.
        return new ShuffleVectorInst(LHS, RHS, ConstantPacked::get(Mask));
      }
    }
  }

  return 0;
}


Instruction *InstCombiner::visitShuffleVectorInst(ShuffleVectorInst &SVI) {
  Value *LHS = SVI.getOperand(0);
  Value *RHS = SVI.getOperand(1);
  std::vector<unsigned> Mask = getShuffleMask(&SVI);

  bool MadeChange = false;
  
  // Undefined shuffle mask -> undefined value.
  if (isa<UndefValue>(SVI.getOperand(2)))
    return ReplaceInstUsesWith(SVI, UndefValue::get(SVI.getType()));
  
  // TODO: If we have shuffle(x, undef, mask) and any elements of mask refer to
  // the undef, change them to undefs.
  
  // Canonicalize shuffle(x    ,x,mask) -> shuffle(x, undef,mask')
  // Canonicalize shuffle(undef,x,mask) -> shuffle(x, undef,mask').
  if (LHS == RHS || isa<UndefValue>(LHS)) {
    if (isa<UndefValue>(LHS) && LHS == RHS) {
      // shuffle(undef,undef,mask) -> undef.
      return ReplaceInstUsesWith(SVI, LHS);
    }
    
    // Remap any references to RHS to use LHS.
    std::vector<Constant*> Elts;
    for (unsigned i = 0, e = Mask.size(); i != e; ++i) {
      if (Mask[i] >= 2*e)
        Elts.push_back(UndefValue::get(Type::UIntTy));
      else {
        if ((Mask[i] >= e && isa<UndefValue>(RHS)) ||
            (Mask[i] <  e && isa<UndefValue>(LHS)))
          Mask[i] = 2*e;     // Turn into undef.
        else
          Mask[i] &= (e-1);  // Force to LHS.
        Elts.push_back(ConstantInt::get(Type::UIntTy, Mask[i]));
      }
    }
    SVI.setOperand(0, SVI.getOperand(1));
    SVI.setOperand(1, UndefValue::get(RHS->getType()));
    SVI.setOperand(2, ConstantPacked::get(Elts));
    LHS = SVI.getOperand(0);
    RHS = SVI.getOperand(1);
    MadeChange = true;
  }
  
  // Analyze the shuffle, are the LHS or RHS and identity shuffles?
  bool isLHSID = true, isRHSID = true;
    
  for (unsigned i = 0, e = Mask.size(); i != e; ++i) {
    if (Mask[i] >= e*2) continue;  // Ignore undef values.
    // Is this an identity shuffle of the LHS value?
    isLHSID &= (Mask[i] == i);
      
    // Is this an identity shuffle of the RHS value?
    isRHSID &= (Mask[i]-e == i);
  }

  // Eliminate identity shuffles.
  if (isLHSID) return ReplaceInstUsesWith(SVI, LHS);
  if (isRHSID) return ReplaceInstUsesWith(SVI, RHS);
  
  // If the LHS is a shufflevector itself, see if we can combine it with this
  // one without producing an unusual shuffle.  Here we are really conservative:
  // we are absolutely afraid of producing a shuffle mask not in the input
  // program, because the code gen may not be smart enough to turn a merged
  // shuffle into two specific shuffles: it may produce worse code.  As such,
  // we only merge two shuffles if the result is one of the two input shuffle
  // masks.  In this case, merging the shuffles just removes one instruction,
  // which we know is safe.  This is good for things like turning:
  // (splat(splat)) -> splat.
  if (ShuffleVectorInst *LHSSVI = dyn_cast<ShuffleVectorInst>(LHS)) {
    if (isa<UndefValue>(RHS)) {
      std::vector<unsigned> LHSMask = getShuffleMask(LHSSVI);

      std::vector<unsigned> NewMask;
      for (unsigned i = 0, e = Mask.size(); i != e; ++i)
        if (Mask[i] >= 2*e)
          NewMask.push_back(2*e);
        else
          NewMask.push_back(LHSMask[Mask[i]]);
      
      // If the result mask is equal to the src shuffle or this shuffle mask, do
      // the replacement.
      if (NewMask == LHSMask || NewMask == Mask) {
        std::vector<Constant*> Elts;
        for (unsigned i = 0, e = NewMask.size(); i != e; ++i) {
          if (NewMask[i] >= e*2) {
            Elts.push_back(UndefValue::get(Type::UIntTy));
          } else {
            Elts.push_back(ConstantInt::get(Type::UIntTy, NewMask[i]));
          }
        }
        return new ShuffleVectorInst(LHSSVI->getOperand(0),
                                     LHSSVI->getOperand(1),
                                     ConstantPacked::get(Elts));
      }
    }
  }
  
  return MadeChange ? &SVI : 0;
}



void InstCombiner::removeFromWorkList(Instruction *I) {
  WorkList.erase(std::remove(WorkList.begin(), WorkList.end(), I),
                 WorkList.end());
}


/// TryToSinkInstruction - Try to move the specified instruction from its
/// current block into the beginning of DestBlock, which can only happen if it's
/// safe to move the instruction past all of the instructions between it and the
/// end of its block.
static bool TryToSinkInstruction(Instruction *I, BasicBlock *DestBlock) {
  assert(I->hasOneUse() && "Invariants didn't hold!");

  // Cannot move control-flow-involving, volatile loads, vaarg, etc.
  if (isa<PHINode>(I) || I->mayWriteToMemory()) return false;

  // Do not sink alloca instructions out of the entry block.
  if (isa<AllocaInst>(I) && I->getParent() == &DestBlock->getParent()->front())
    return false;

  // We can only sink load instructions if there is nothing between the load and
  // the end of block that could change the value.
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    for (BasicBlock::iterator Scan = LI, E = LI->getParent()->end();
         Scan != E; ++Scan)
      if (Scan->mayWriteToMemory())
        return false;
  }

  BasicBlock::iterator InsertPos = DestBlock->begin();
  while (isa<PHINode>(InsertPos)) ++InsertPos;

  I->moveBefore(InsertPos);
  ++NumSunkInst;
  return true;
}

/// OptimizeConstantExpr - Given a constant expression and target data layout
/// information, symbolically evaluation the constant expr to something simpler
/// if possible.
static Constant *OptimizeConstantExpr(ConstantExpr *CE, const TargetData *TD) {
  if (!TD) return CE;
  
  Constant *Ptr = CE->getOperand(0);
  if (CE->getOpcode() == Instruction::GetElementPtr && Ptr->isNullValue() &&
      cast<PointerType>(Ptr->getType())->getElementType()->isSized()) {
    // If this is a constant expr gep that is effectively computing an
    // "offsetof", fold it into 'cast int Size to T*' instead of 'gep 0, 0, 12'
    bool isFoldableGEP = true;
    for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
      if (!isa<ConstantInt>(CE->getOperand(i)))
        isFoldableGEP = false;
    if (isFoldableGEP) {
      std::vector<Value*> Ops(CE->op_begin()+1, CE->op_end());
      uint64_t Offset = TD->getIndexedOffset(Ptr->getType(), Ops);
      Constant *C = ConstantInt::get(Type::ULongTy, Offset);
      C = ConstantExpr::getCast(C, TD->getIntPtrType());
      return ConstantExpr::getCast(C, CE->getType());
    }
  }
  
  return CE;
}


/// AddReachableCodeToWorklist - Walk the function in depth-first order, adding
/// all reachable code to the worklist.
///
/// This has a couple of tricks to make the code faster and more powerful.  In
/// particular, we constant fold and DCE instructions as we go, to avoid adding
/// them to the worklist (this significantly speeds up instcombine on code where
/// many instructions are dead or constant).  Additionally, if we find a branch
/// whose condition is a known constant, we only visit the reachable successors.
///
static void AddReachableCodeToWorklist(BasicBlock *BB, 
                                       std::set<BasicBlock*> &Visited,
                                       std::vector<Instruction*> &WorkList,
                                       const TargetData *TD) {
  // We have now visited this block!  If we've already been here, bail out.
  if (!Visited.insert(BB).second) return;
    
  for (BasicBlock::iterator BBI = BB->begin(), E = BB->end(); BBI != E; ) {
    Instruction *Inst = BBI++;
    
    // DCE instruction if trivially dead.
    if (isInstructionTriviallyDead(Inst)) {
      ++NumDeadInst;
      DEBUG(std::cerr << "IC: DCE: " << *Inst);
      Inst->eraseFromParent();
      continue;
    }
    
    // ConstantProp instruction if trivially constant.
    if (Constant *C = ConstantFoldInstruction(Inst)) {
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
        C = OptimizeConstantExpr(CE, TD);
      DEBUG(std::cerr << "IC: ConstFold to: " << *C << " from: " << *Inst);
      Inst->replaceAllUsesWith(C);
      ++NumConstProp;
      Inst->eraseFromParent();
      continue;
    }
    
    WorkList.push_back(Inst);
  }

  // Recursively visit successors.  If this is a branch or switch on a constant,
  // only visit the reachable successor.
  TerminatorInst *TI = BB->getTerminator();
  if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    if (BI->isConditional() && isa<ConstantBool>(BI->getCondition())) {
      bool CondVal = cast<ConstantBool>(BI->getCondition())->getValue();
      AddReachableCodeToWorklist(BI->getSuccessor(!CondVal), Visited, WorkList,
                                 TD);
      return;
    }
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    if (ConstantInt *Cond = dyn_cast<ConstantInt>(SI->getCondition())) {
      // See if this is an explicit destination.
      for (unsigned i = 1, e = SI->getNumSuccessors(); i != e; ++i)
        if (SI->getCaseValue(i) == Cond) {
          AddReachableCodeToWorklist(SI->getSuccessor(i), Visited, WorkList,TD);
          return;
        }
      
      // Otherwise it is the default destination.
      AddReachableCodeToWorklist(SI->getSuccessor(0), Visited, WorkList, TD);
      return;
    }
  }
  
  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
    AddReachableCodeToWorklist(TI->getSuccessor(i), Visited, WorkList, TD);
}

bool InstCombiner::runOnFunction(Function &F) {
  bool Changed = false;
  TD = &getAnalysis<TargetData>();

  {
    // Do a depth-first traversal of the function, populate the worklist with
    // the reachable instructions.  Ignore blocks that are not reachable.  Keep
    // track of which blocks we visit.
    std::set<BasicBlock*> Visited;
    AddReachableCodeToWorklist(F.begin(), Visited, WorkList, TD);

    // Do a quick scan over the function.  If we find any blocks that are
    // unreachable, remove any instructions inside of them.  This prevents
    // the instcombine code from having to deal with some bad special cases.
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
      if (!Visited.count(BB)) {
        Instruction *Term = BB->getTerminator();
        while (Term != BB->begin()) {   // Remove instrs bottom-up
          BasicBlock::iterator I = Term; --I;

          DEBUG(std::cerr << "IC: DCE: " << *I);
          ++NumDeadInst;

          if (!I->use_empty())
            I->replaceAllUsesWith(UndefValue::get(I->getType()));
          I->eraseFromParent();
        }
      }
  }

  while (!WorkList.empty()) {
    Instruction *I = WorkList.back();  // Get an instruction from the worklist
    WorkList.pop_back();

    // Check to see if we can DCE the instruction.
    if (isInstructionTriviallyDead(I)) {
      // Add operands to the worklist.
      if (I->getNumOperands() < 4)
        AddUsesToWorkList(*I);
      ++NumDeadInst;

      DEBUG(std::cerr << "IC: DCE: " << *I);

      I->eraseFromParent();
      removeFromWorkList(I);
      continue;
    }

    // Instruction isn't dead, see if we can constant propagate it.
    if (Constant *C = ConstantFoldInstruction(I)) {
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
        C = OptimizeConstantExpr(CE, TD);
      DEBUG(std::cerr << "IC: ConstFold to: " << *C << " from: " << *I);

      // Add operands to the worklist.
      AddUsesToWorkList(*I);
      ReplaceInstUsesWith(*I, C);

      ++NumConstProp;
      I->eraseFromParent();
      removeFromWorkList(I);
      continue;
    }

    // See if we can trivially sink this instruction to a successor basic block.
    if (I->hasOneUse()) {
      BasicBlock *BB = I->getParent();
      BasicBlock *UserParent = cast<Instruction>(I->use_back())->getParent();
      if (UserParent != BB) {
        bool UserIsSuccessor = false;
        // See if the user is one of our successors.
        for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI)
          if (*SI == UserParent) {
            UserIsSuccessor = true;
            break;
          }

        // If the user is one of our immediate successors, and if that successor
        // only has us as a predecessors (we'd have to split the critical edge
        // otherwise), we can keep going.
        if (UserIsSuccessor && !isa<PHINode>(I->use_back()) &&
            next(pred_begin(UserParent)) == pred_end(UserParent))
          // Okay, the CFG is simple enough, try to sink this instruction.
          Changed |= TryToSinkInstruction(I, UserParent);
      }
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
        BasicBlock::iterator InsertPos = I;

        if (!isa<PHINode>(Result))        // If combining a PHI, don't insert
          while (isa<PHINode>(InsertPos)) // middle of a block of PHIs.
            ++InsertPos;

        InstParent->getInstList().insert(InsertPos, Result);

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
          // occurrences of this instruction.
          removeFromWorkList(I);
          I->eraseFromParent();
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

