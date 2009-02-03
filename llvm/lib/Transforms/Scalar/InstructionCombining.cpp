//===- InstructionCombining.cpp - Combine multiple instructions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// InstructionCombining - Combine instructions to form fewer, simple
// instructions.  This pass does not modify the CFG.  This pass is where
// algebraic simplification happens.
//
// This pass combines things like:
//    %Y = add i32 %X, 1
//    %Z = add i32 %Y, 1
// into:
//    %Z = add i32 %X, 2
//
// This is a simple worklist driven algorithm.
//
// This pass guarantees that the following canonicalizations are performed on
// the program:
//    1. If a binary operator has a constant operand, it is moved to the RHS
//    2. Bitwise operators with constant operands are always grouped so that
//       shifts are performed first, then or's, then and's, then xor's.
//    3. Compare instructions are converted from <,>,<=,>= to ==,!= if possible
//    4. All cmp instructions on boolean values are replaced with logical ops
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
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <climits>
#include <sstream>
using namespace llvm;
using namespace llvm::PatternMatch;

STATISTIC(NumCombined , "Number of insts combined");
STATISTIC(NumConstProp, "Number of constant folds");
STATISTIC(NumDeadInst , "Number of dead inst eliminated");
STATISTIC(NumDeadStore, "Number of dead stores eliminated");
STATISTIC(NumSunkInst , "Number of instructions sunk");

namespace {
  class VISIBILITY_HIDDEN InstCombiner
    : public FunctionPass,
      public InstVisitor<InstCombiner, Instruction*> {
    // Worklist of all of the instructions that need to be simplified.
    SmallVector<Instruction*, 256> Worklist;
    DenseMap<Instruction*, unsigned> WorklistMap;
    TargetData *TD;
    bool MustPreserveLCSSA;
  public:
    static char ID; // Pass identification, replacement for typeid
    InstCombiner() : FunctionPass(&ID) {}

    /// AddToWorkList - Add the specified instruction to the worklist if it
    /// isn't already in it.
    void AddToWorkList(Instruction *I) {
      if (WorklistMap.insert(std::make_pair(I, Worklist.size())).second)
        Worklist.push_back(I);
    }
    
    // RemoveFromWorkList - remove I from the worklist if it exists.
    void RemoveFromWorkList(Instruction *I) {
      DenseMap<Instruction*, unsigned>::iterator It = WorklistMap.find(I);
      if (It == WorklistMap.end()) return; // Not in worklist.
      
      // Don't bother moving everything down, just null out the slot.
      Worklist[It->second] = 0;
      
      WorklistMap.erase(It);
    }
    
    Instruction *RemoveOneFromWorkList() {
      Instruction *I = Worklist.back();
      Worklist.pop_back();
      WorklistMap.erase(I);
      return I;
    }

    
    /// AddUsersToWorkList - When an instruction is simplified, add all users of
    /// the instruction to the work lists because they might get more simplified
    /// now.
    ///
    void AddUsersToWorkList(Value &I) {
      for (Value::use_iterator UI = I.use_begin(), UE = I.use_end();
           UI != UE; ++UI)
        AddToWorkList(cast<Instruction>(*UI));
    }

    /// AddUsesToWorkList - When an instruction is simplified, add operands to
    /// the work lists because they might get more simplified now.
    ///
    void AddUsesToWorkList(Instruction &I) {
      for (User::op_iterator i = I.op_begin(), e = I.op_end(); i != e; ++i)
        if (Instruction *Op = dyn_cast<Instruction>(*i))
          AddToWorkList(Op);
    }
    
    /// AddSoonDeadInstToWorklist - The specified instruction is about to become
    /// dead.  Add all of its operands to the worklist, turning them into
    /// undef's to reduce the number of uses of those instructions.
    ///
    /// Return the specified operand before it is turned into an undef.
    ///
    Value *AddSoonDeadInstToWorklist(Instruction &I, unsigned op) {
      Value *R = I.getOperand(op);
      
      for (User::op_iterator i = I.op_begin(), e = I.op_end(); i != e; ++i)
        if (Instruction *Op = dyn_cast<Instruction>(*i)) {
          AddToWorkList(Op);
          // Set the operand to undef to drop the use.
          *i = UndefValue::get(Op->getType());
        }
      
      return R;
    }

  public:
    virtual bool runOnFunction(Function &F);
    
    bool DoOneIteration(Function &F, unsigned ItNum);

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
    bool SimplifyDivRemOfSelect(BinaryOperator &I);
    Instruction *commonRemTransforms(BinaryOperator &I);
    Instruction *commonIRemTransforms(BinaryOperator &I);
    Instruction *commonDivTransforms(BinaryOperator &I);
    Instruction *commonIDivTransforms(BinaryOperator &I);
    Instruction *visitUDiv(BinaryOperator &I);
    Instruction *visitSDiv(BinaryOperator &I);
    Instruction *visitFDiv(BinaryOperator &I);
    Instruction *FoldAndOfICmps(Instruction &I, ICmpInst *LHS, ICmpInst *RHS);
    Instruction *visitAnd(BinaryOperator &I);
    Instruction *FoldOrOfICmps(Instruction &I, ICmpInst *LHS, ICmpInst *RHS);
    Instruction *FoldOrWithConstants(BinaryOperator &I, Value *Op,
                                     Value *A, Value *B, Value *C);
    Instruction *visitOr (BinaryOperator &I);
    Instruction *visitXor(BinaryOperator &I);
    Instruction *visitShl(BinaryOperator &I);
    Instruction *visitAShr(BinaryOperator &I);
    Instruction *visitLShr(BinaryOperator &I);
    Instruction *commonShiftTransforms(BinaryOperator &I);
    Instruction *FoldFCmp_IntToFP_Cst(FCmpInst &I, Instruction *LHSI,
                                      Constant *RHSC);
    Instruction *visitFCmpInst(FCmpInst &I);
    Instruction *visitICmpInst(ICmpInst &I);
    Instruction *visitICmpInstWithCastAndCast(ICmpInst &ICI);
    Instruction *visitICmpInstWithInstAndIntCst(ICmpInst &ICI,
                                                Instruction *LHS,
                                                ConstantInt *RHS);
    Instruction *FoldICmpDivCst(ICmpInst &ICI, BinaryOperator *DivI,
                                ConstantInt *DivRHS);

    Instruction *FoldGEPICmp(User *GEPLHS, Value *RHS,
                             ICmpInst::Predicate Cond, Instruction &I);
    Instruction *FoldShiftByConstant(Value *Op0, ConstantInt *Op1,
                                     BinaryOperator &I);
    Instruction *commonCastTransforms(CastInst &CI);
    Instruction *commonIntCastTransforms(CastInst &CI);
    Instruction *commonPointerCastTransforms(CastInst &CI);
    Instruction *visitTrunc(TruncInst &CI);
    Instruction *visitZExt(ZExtInst &CI);
    Instruction *visitSExt(SExtInst &CI);
    Instruction *visitFPTrunc(FPTruncInst &CI);
    Instruction *visitFPExt(CastInst &CI);
    Instruction *visitFPToUI(FPToUIInst &FI);
    Instruction *visitFPToSI(FPToSIInst &FI);
    Instruction *visitUIToFP(CastInst &CI);
    Instruction *visitSIToFP(CastInst &CI);
    Instruction *visitPtrToInt(CastInst &CI);
    Instruction *visitIntToPtr(IntToPtrInst &CI);
    Instruction *visitBitCast(BitCastInst &CI);
    Instruction *FoldSelectOpOp(SelectInst &SI, Instruction *TI,
                                Instruction *FI);
    Instruction *visitSelectInst(SelectInst &SI);
    Instruction *visitSelectInstWithICmp(SelectInst &SI, ICmpInst *ICI);
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
    Instruction *visitExtractValueInst(ExtractValueInst &EV);

    // visitInstruction - Specify what to return for unhandled instructions...
    Instruction *visitInstruction(Instruction &I) { return 0; }

  private:
    Instruction *visitCallSite(CallSite CS);
    bool transformConstExprCastCall(CallSite CS);
    Instruction *transformCallThroughTrampoline(CallSite CS);
    Instruction *transformZExtICmp(ICmpInst *ICI, Instruction &CI,
                                   bool DoXform = true);
    bool WillNotOverflowSignedAdd(Value *LHS, Value *RHS);

  public:
    // InsertNewInstBefore - insert an instruction New before instruction Old
    // in the program.  Add the new instruction to the worklist.
    //
    Instruction *InsertNewInstBefore(Instruction *New, Instruction &Old) {
      assert(New && New->getParent() == 0 &&
             "New instruction already inserted into a basic block!");
      BasicBlock *BB = Old.getParent();
      BB->getInstList().insert(&Old, New);  // Insert inst
      AddToWorkList(New);
      return New;
    }

    /// InsertCastBefore - Insert a cast of V to TY before the instruction POS.
    /// This also adds the cast to the worklist.  Finally, this returns the
    /// cast.
    Value *InsertCastBefore(Instruction::CastOps opc, Value *V, const Type *Ty,
                            Instruction &Pos) {
      if (V->getType() == Ty) return V;

      if (Constant *CV = dyn_cast<Constant>(V))
        return ConstantExpr::getCast(opc, CV, Ty);
      
      Instruction *C = CastInst::Create(opc, V, Ty, V->getName(), &Pos);
      AddToWorkList(C);
      return C;
    }
        
    Value *InsertBitCastBefore(Value *V, const Type *Ty, Instruction &Pos) {
      return InsertCastBefore(Instruction::BitCast, V, Ty, Pos);
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

    // EraseInstFromFunction - When dealing with an instruction that has side
    // effects or produces a void value, we can't rely on DCE to delete the
    // instruction.  Instead, visit methods should return the value returned by
    // this function.
    Instruction *EraseInstFromFunction(Instruction &I) {
      assert(I.use_empty() && "Cannot erase instruction that is used!");
      AddUsesToWorkList(I);
      RemoveFromWorkList(&I);
      I.eraseFromParent();
      return 0;  // Don't do anything with FI
    }
        
    void ComputeMaskedBits(Value *V, const APInt &Mask, APInt &KnownZero,
                           APInt &KnownOne, unsigned Depth = 0) const {
      return llvm::ComputeMaskedBits(V, Mask, KnownZero, KnownOne, TD, Depth);
    }
    
    bool MaskedValueIsZero(Value *V, const APInt &Mask, 
                           unsigned Depth = 0) const {
      return llvm::MaskedValueIsZero(V, Mask, TD, Depth);
    }
    unsigned ComputeNumSignBits(Value *Op, unsigned Depth = 0) const {
      return llvm::ComputeNumSignBits(Op, TD, Depth);
    }

  private:

    /// SimplifyCommutative - This performs a few simplifications for 
    /// commutative operators.
    bool SimplifyCommutative(BinaryOperator &I);

    /// SimplifyCompare - This reorders the operands of a CmpInst to get them in
    /// most-complex to least-complex order.
    bool SimplifyCompare(CmpInst &I);

    /// SimplifyDemandedUseBits - Attempts to replace V with a simpler value
    /// based on the demanded bits.
    Value *SimplifyDemandedUseBits(Value *V, APInt DemandedMask, 
                                   APInt& KnownZero, APInt& KnownOne,
                                   unsigned Depth);
    bool SimplifyDemandedBits(Use &U, APInt DemandedMask, 
                              APInt& KnownZero, APInt& KnownOne,
                              unsigned Depth=0);
        
    /// SimplifyDemandedInstructionBits - Inst is an integer instruction that
    /// SimplifyDemandedBits knows about.  See if the instruction has any
    /// properties that allow us to simplify its operands.
    bool SimplifyDemandedInstructionBits(Instruction &Inst);
        
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
    Instruction *FoldPHIArgGEPIntoPHI(PHINode &PN);

    
    Instruction *OptAndOp(Instruction *Op, ConstantInt *OpRHS,
                          ConstantInt *AndRHS, BinaryOperator &TheAnd);
    
    Value *FoldLogicalPlusAnd(Value *LHS, Value *RHS, ConstantInt *Mask,
                              bool isSub, Instruction &I);
    Instruction *InsertRangeTest(Value *V, Constant *Lo, Constant *Hi,
                                 bool isSigned, bool Inside, Instruction &IB);
    Instruction *PromoteCastOfAllocation(BitCastInst &CI, AllocationInst &AI);
    Instruction *MatchBSwap(BinaryOperator &I);
    bool SimplifyStoreAtEndOfBlock(StoreInst &SI);
    Instruction *SimplifyMemTransfer(MemIntrinsic *MI);
    Instruction *SimplifyMemSet(MemSetInst *MI);


    Value *EvaluateInDifferentType(Value *V, const Type *Ty, bool isSigned);

    bool CanEvaluateInDifferentType(Value *V, const IntegerType *Ty,
                                    unsigned CastOpc, int &NumCastsRemoved);
    unsigned GetOrEnforceKnownAlignment(Value *V,
                                        unsigned PrefAlign = 0);

  };
}

char InstCombiner::ID = 0;
static RegisterPass<InstCombiner>
X("instcombine", "Combine redundant instructions");

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
  if (const IntegerType* ITy = dyn_cast<IntegerType>(Ty)) {
    if (ITy->getBitWidth() < 32)
      return Type::Int32Ty;
  }
  return Ty;
}

/// getBitCastOperand - If the specified operand is a CastInst, a constant
/// expression bitcast, or a GetElementPtrInst with all zero indices, return the
/// operand value, otherwise return null.
static Value *getBitCastOperand(Value *V) {
  if (BitCastInst *I = dyn_cast<BitCastInst>(V))
    // BitCastInst?
    return I->getOperand(0);
  else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V)) {
    // GetElementPtrInst?
    if (GEP->hasAllZeroIndices())
      return GEP->getOperand(0);
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::BitCast)
      // BitCast ConstantExp?
      return CE->getOperand(0);
    else if (CE->getOpcode() == Instruction::GetElementPtr) {
      // GetElementPtr ConstantExp?
      for (User::op_iterator I = CE->op_begin() + 1, E = CE->op_end();
           I != E; ++I) {
        ConstantInt *CI = dyn_cast<ConstantInt>(I);
        if (!CI || !CI->isZero())
          // Any non-zero indices? Not cast-like.
          return 0;
      }
      // All-zero indices? This is just like casting.
      return CE->getOperand(0);
    }
  }
  return 0;
}

/// This function is a wrapper around CastInst::isEliminableCastPair. It
/// simply extracts arguments and returns what that function returns.
static Instruction::CastOps 
isEliminableCastPair(
  const CastInst *CI, ///< The first cast instruction
  unsigned opcode,       ///< The opcode of the second cast instruction
  const Type *DstTy,     ///< The target type for the second cast instruction
  TargetData *TD         ///< The target data for pointer size
) {
  
  const Type *SrcTy = CI->getOperand(0)->getType();   // A from above
  const Type *MidTy = CI->getType();                  // B from above

  // Get the opcodes of the two Cast instructions
  Instruction::CastOps firstOp = Instruction::CastOps(CI->getOpcode());
  Instruction::CastOps secondOp = Instruction::CastOps(opcode);

  return Instruction::CastOps(
      CastInst::isEliminableCastPair(firstOp, secondOp, SrcTy, MidTy,
                                     DstTy, TD->getIntPtrType()));
}

/// ValueRequiresCast - Return true if the cast from "V to Ty" actually results
/// in any code being generated.  It does not require codegen if V is simple
/// enough or if the cast can be folded into other casts.
static bool ValueRequiresCast(Instruction::CastOps opcode, const Value *V, 
                              const Type *Ty, TargetData *TD) {
  if (V->getType() == Ty || isa<Constant>(V)) return false;
  
  // If this is another cast that can be eliminated, it isn't codegen either.
  if (const CastInst *CI = dyn_cast<CastInst>(V))
    if (isEliminableCastPair(CI, opcode, Ty, TD)) 
      return false;
  return true;
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
          Instruction *New = BinaryOperator::Create(Opcode, Op->getOperand(0),
                                                    Op1->getOperand(0),
                                                    Op1->getName(), &I);
          AddToWorkList(New);
          I.setOperand(0, New);
          I.setOperand(1, Folded);
          return true;
        }
    }
  return Changed;
}

/// SimplifyCompare - For a CmpInst this function just orders the operands
/// so that theyare listed from right (least complex) to left (most complex).
/// This puts constants before unary operators before binary operators.
bool InstCombiner::SimplifyCompare(CmpInst &I) {
  if (getComplexity(I.getOperand(0)) >= getComplexity(I.getOperand(1)))
    return false;
  I.swapOperands();
  // Compare instructions are not associative so there's nothing else we can do.
  return true;
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

  if (ConstantVector *C = dyn_cast<ConstantVector>(V))
    if (C->getType()->getElementType()->isInteger())
      return ConstantExpr::getNeg(C);

  return 0;
}

static inline Value *dyn_castNotVal(Value *V) {
  if (BinaryOperator::isNot(V))
    return BinaryOperator::getNotArgument(V);

  // Constants can be considered to be not'ed values...
  if (ConstantInt *C = dyn_cast<ConstantInt>(V))
    return ConstantInt::get(~C->getValue());
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
          uint32_t BitWidth = cast<IntegerType>(V->getType())->getBitWidth();
          uint32_t CSTVal = CST->getLimitedValue(BitWidth);
          CST = ConstantInt::get(APInt(BitWidth, 1).shl(CSTVal));
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

/// getOpcode - If this is an Instruction or a ConstantExpr, return the
/// opcode value. Otherwise return UserOp1.
static unsigned getOpcode(const Value *V) {
  if (const Instruction *I = dyn_cast<Instruction>(V))
    return I->getOpcode();
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    return CE->getOpcode();
  // Use UserOp1 to mean there's no opcode.
  return Instruction::UserOp1;
}

/// AddOne - Add one to a ConstantInt
static ConstantInt *AddOne(ConstantInt *C) {
  APInt Val(C->getValue());
  return ConstantInt::get(++Val);
}
/// SubOne - Subtract one from a ConstantInt
static ConstantInt *SubOne(ConstantInt *C) {
  APInt Val(C->getValue());
  return ConstantInt::get(--Val);
}
/// Add - Add two ConstantInts together
static ConstantInt *Add(ConstantInt *C1, ConstantInt *C2) {
  return ConstantInt::get(C1->getValue() + C2->getValue());
}
/// And - Bitwise AND two ConstantInts together
static ConstantInt *And(ConstantInt *C1, ConstantInt *C2) {
  return ConstantInt::get(C1->getValue() & C2->getValue());
}
/// Subtract - Subtract one ConstantInt from another
static ConstantInt *Subtract(ConstantInt *C1, ConstantInt *C2) {
  return ConstantInt::get(C1->getValue() - C2->getValue());
}
/// Multiply - Multiply two ConstantInts together
static ConstantInt *Multiply(ConstantInt *C1, ConstantInt *C2) {
  return ConstantInt::get(C1->getValue() * C2->getValue());
}
/// MultiplyOverflows - True if the multiply can not be expressed in an int
/// this size.
static bool MultiplyOverflows(ConstantInt *C1, ConstantInt *C2, bool sign) {
  uint32_t W = C1->getBitWidth();
  APInt LHSExt = C1->getValue(), RHSExt = C2->getValue();
  if (sign) {
    LHSExt.sext(W * 2);
    RHSExt.sext(W * 2);
  } else {
    LHSExt.zext(W * 2);
    RHSExt.zext(W * 2);
  }

  APInt MulExt = LHSExt * RHSExt;

  if (sign) {
    APInt Min = APInt::getSignedMinValue(W).sext(W * 2);
    APInt Max = APInt::getSignedMaxValue(W).sext(W * 2);
    return MulExt.slt(Min) || MulExt.sgt(Max);
  } else 
    return MulExt.ugt(APInt::getLowBitsSet(W * 2, W));
}


/// ShrinkDemandedConstant - Check to see if the specified operand of the 
/// specified instruction is a constant integer.  If so, check to see if there
/// are any bits set in the constant that are not demanded.  If so, shrink the
/// constant and return true.
static bool ShrinkDemandedConstant(Instruction *I, unsigned OpNo, 
                                   APInt Demanded) {
  assert(I && "No instruction?");
  assert(OpNo < I->getNumOperands() && "Operand index too large");

  // If the operand is not a constant integer, nothing to do.
  ConstantInt *OpC = dyn_cast<ConstantInt>(I->getOperand(OpNo));
  if (!OpC) return false;

  // If there are no bits set that aren't demanded, nothing to do.
  Demanded.zextOrTrunc(OpC->getValue().getBitWidth());
  if ((~Demanded & OpC->getValue()) == 0)
    return false;

  // This instruction is producing bits that are not demanded. Shrink the RHS.
  Demanded &= OpC->getValue();
  I->setOperand(OpNo, ConstantInt::get(Demanded));
  return true;
}

// ComputeSignedMinMaxValuesFromKnownBits - Given a signed integer type and a 
// set of known zero and one bits, compute the maximum and minimum values that
// could have the specified known zero and known one bits, returning them in
// min/max.
static void ComputeSignedMinMaxValuesFromKnownBits(const Type *Ty,
                                                   const APInt& KnownZero,
                                                   const APInt& KnownOne,
                                                   APInt& Min, APInt& Max) {
  uint32_t BitWidth = cast<IntegerType>(Ty)->getBitWidth();
  assert(KnownZero.getBitWidth() == BitWidth && 
         KnownOne.getBitWidth() == BitWidth &&
         Min.getBitWidth() == BitWidth && Max.getBitWidth() == BitWidth &&
         "Ty, KnownZero, KnownOne and Min, Max must have equal bitwidth.");
  APInt UnknownBits = ~(KnownZero|KnownOne);

  // The minimum value is when all unknown bits are zeros, EXCEPT for the sign
  // bit if it is unknown.
  Min = KnownOne;
  Max = KnownOne|UnknownBits;
  
  if (UnknownBits[BitWidth-1]) { // Sign bit is unknown
    Min.set(BitWidth-1);
    Max.clear(BitWidth-1);
  }
}

// ComputeUnsignedMinMaxValuesFromKnownBits - Given an unsigned integer type and
// a set of known zero and one bits, compute the maximum and minimum values that
// could have the specified known zero and known one bits, returning them in
// min/max.
static void ComputeUnsignedMinMaxValuesFromKnownBits(const Type *Ty,
                                                     const APInt &KnownZero,
                                                     const APInt &KnownOne,
                                                     APInt &Min, APInt &Max) {
  uint32_t BitWidth = cast<IntegerType>(Ty)->getBitWidth(); BitWidth = BitWidth;
  assert(KnownZero.getBitWidth() == BitWidth && 
         KnownOne.getBitWidth() == BitWidth &&
         Min.getBitWidth() == BitWidth && Max.getBitWidth() &&
         "Ty, KnownZero, KnownOne and Min, Max must have equal bitwidth.");
  APInt UnknownBits = ~(KnownZero|KnownOne);
  
  // The minimum value is when the unknown bits are all zeros.
  Min = KnownOne;
  // The maximum value is when the unknown bits are all ones.
  Max = KnownOne|UnknownBits;
}

/// SimplifyDemandedInstructionBits - Inst is an integer instruction that
/// SimplifyDemandedBits knows about.  See if the instruction has any
/// properties that allow us to simplify its operands.
bool InstCombiner::SimplifyDemandedInstructionBits(Instruction &Inst) {
  unsigned BitWidth = cast<IntegerType>(Inst.getType())->getBitWidth();
  APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
  APInt DemandedMask(APInt::getAllOnesValue(BitWidth));
  
  Value *V = SimplifyDemandedUseBits(&Inst, DemandedMask, 
                                     KnownZero, KnownOne, 0);
  if (V == 0) return false;
  if (V == &Inst) return true;
  ReplaceInstUsesWith(Inst, V);
  return true;
}

/// SimplifyDemandedBits - This form of SimplifyDemandedBits simplifies the
/// specified instruction operand if possible, updating it in place.  It returns
/// true if it made any change and false otherwise.
bool InstCombiner::SimplifyDemandedBits(Use &U, APInt DemandedMask, 
                                        APInt &KnownZero, APInt &KnownOne,
                                        unsigned Depth) {
  Value *NewVal = SimplifyDemandedUseBits(U.get(), DemandedMask,
                                          KnownZero, KnownOne, Depth);
  if (NewVal == 0) return false;
  U.set(NewVal);
  return true;
}


/// SimplifyDemandedUseBits - This function attempts to replace V with a simpler
/// value based on the demanded bits.  When this function is called, it is known
/// that only the bits set in DemandedMask of the result of V are ever used
/// downstream. Consequently, depending on the mask and V, it may be possible
/// to replace V with a constant or one of its operands. In such cases, this
/// function does the replacement and returns true. In all other cases, it
/// returns false after analyzing the expression and setting KnownOne and known
/// to be one in the expression.  KnownZero contains all the bits that are known
/// to be zero in the expression. These are provided to potentially allow the
/// caller (which might recursively be SimplifyDemandedBits itself) to simplify
/// the expression. KnownOne and KnownZero always follow the invariant that 
/// KnownOne & KnownZero == 0. That is, a bit can't be both 1 and 0. Note that
/// the bits in KnownOne and KnownZero may only be accurate for those bits set
/// in DemandedMask. Note also that the bitwidth of V, DemandedMask, KnownZero
/// and KnownOne must all be the same.
///
/// This returns null if it did not change anything and it permits no
/// simplification.  This returns V itself if it did some simplification of V's
/// operands based on the information about what bits are demanded. This returns
/// some other non-null value if it found out that V is equal to another value
/// in the context where the specified bits are demanded, but not for all users.
Value *InstCombiner::SimplifyDemandedUseBits(Value *V, APInt DemandedMask,
                                             APInt &KnownZero, APInt &KnownOne,
                                             unsigned Depth) {
  assert(V != 0 && "Null pointer of Value???");
  assert(Depth <= 6 && "Limit Search Depth");
  uint32_t BitWidth = DemandedMask.getBitWidth();
  const IntegerType *VTy = cast<IntegerType>(V->getType());
  assert(VTy->getBitWidth() == BitWidth && 
         KnownZero.getBitWidth() == BitWidth && 
         KnownOne.getBitWidth() == BitWidth &&
         "Value *V, DemandedMask, KnownZero and KnownOne \
          must have same BitWidth");
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    // We know all of the bits for a constant!
    KnownOne = CI->getValue() & DemandedMask;
    KnownZero = ~KnownOne & DemandedMask;
    return 0;
  }
  
  KnownZero.clear();
  KnownOne.clear();
  if (DemandedMask == 0) {   // Not demanding any bits from V.
    if (isa<UndefValue>(V))
      return 0;
    return UndefValue::get(VTy);
  }
  
  if (Depth == 6)        // Limit search depth.
    return 0;
  
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return 0;        // Only analyze instructions.
  
  APInt LHSKnownZero(BitWidth, 0), LHSKnownOne(BitWidth, 0);
  APInt &RHSKnownZero = KnownZero, &RHSKnownOne = KnownOne;

  // If there are multiple uses of this value and we aren't at the root, then
  // we can't do any simplifications of the operands, because DemandedMask
  // only reflects the bits demanded by *one* of the users.
  if (Depth != 0 && !I->hasOneUse()) {
    // Despite the fact that we can't simplify this instruction in all User's
    // context, we can at least compute the knownzero/knownone bits, and we can
    // do simplifications that apply to *just* the one user if we know that
    // this instruction has a simpler value in that context.
    if (I->getOpcode() == Instruction::And) {
      // If either the LHS or the RHS are Zero, the result is zero.
      ComputeMaskedBits(I->getOperand(1), DemandedMask,
                        RHSKnownZero, RHSKnownOne, Depth+1);
      ComputeMaskedBits(I->getOperand(0), DemandedMask & ~RHSKnownZero,
                        LHSKnownZero, LHSKnownOne, Depth+1);
      
      // If all of the demanded bits are known 1 on one side, return the other.
      // These bits cannot contribute to the result of the 'and' in this
      // context.
      if ((DemandedMask & ~LHSKnownZero & RHSKnownOne) == 
          (DemandedMask & ~LHSKnownZero))
        return I->getOperand(0);
      if ((DemandedMask & ~RHSKnownZero & LHSKnownOne) == 
          (DemandedMask & ~RHSKnownZero))
        return I->getOperand(1);
      
      // If all of the demanded bits in the inputs are known zeros, return zero.
      if ((DemandedMask & (RHSKnownZero|LHSKnownZero)) == DemandedMask)
        return Constant::getNullValue(VTy);
      
    } else if (I->getOpcode() == Instruction::Or) {
      // We can simplify (X|Y) -> X or Y in the user's context if we know that
      // only bits from X or Y are demanded.
      
      // If either the LHS or the RHS are One, the result is One.
      ComputeMaskedBits(I->getOperand(1), DemandedMask, 
                        RHSKnownZero, RHSKnownOne, Depth+1);
      ComputeMaskedBits(I->getOperand(0), DemandedMask & ~RHSKnownOne, 
                        LHSKnownZero, LHSKnownOne, Depth+1);
      
      // If all of the demanded bits are known zero on one side, return the
      // other.  These bits cannot contribute to the result of the 'or' in this
      // context.
      if ((DemandedMask & ~LHSKnownOne & RHSKnownZero) == 
          (DemandedMask & ~LHSKnownOne))
        return I->getOperand(0);
      if ((DemandedMask & ~RHSKnownOne & LHSKnownZero) == 
          (DemandedMask & ~RHSKnownOne))
        return I->getOperand(1);
      
      // If all of the potentially set bits on one side are known to be set on
      // the other side, just use the 'other' side.
      if ((DemandedMask & (~RHSKnownZero) & LHSKnownOne) == 
          (DemandedMask & (~RHSKnownZero)))
        return I->getOperand(0);
      if ((DemandedMask & (~LHSKnownZero) & RHSKnownOne) == 
          (DemandedMask & (~LHSKnownZero)))
        return I->getOperand(1);
    }
    
    // Compute the KnownZero/KnownOne bits to simplify things downstream.
    ComputeMaskedBits(I, DemandedMask, KnownZero, KnownOne, Depth);
    return 0;
  }
  
  // If this is the root being simplified, allow it to have multiple uses,
  // just set the DemandedMask to all bits so that we can try to simplify the
  // operands.  This allows visitTruncInst (for example) to simplify the
  // operand of a trunc without duplicating all the logic below.
  if (Depth == 0 && !V->hasOneUse())
    DemandedMask = APInt::getAllOnesValue(BitWidth);
  
  switch (I->getOpcode()) {
  default:
    ComputeMaskedBits(I, DemandedMask, RHSKnownZero, RHSKnownOne, Depth);
    break;
  case Instruction::And:
    // If either the LHS or the RHS are Zero, the result is zero.
    if (SimplifyDemandedBits(I->getOperandUse(1), DemandedMask,
                             RHSKnownZero, RHSKnownOne, Depth+1) ||
        SimplifyDemandedBits(I->getOperandUse(0), DemandedMask & ~RHSKnownZero,
                             LHSKnownZero, LHSKnownOne, Depth+1))
      return I;
    assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?"); 
    assert(!(LHSKnownZero & LHSKnownOne) && "Bits known to be one AND zero?"); 

    // If all of the demanded bits are known 1 on one side, return the other.
    // These bits cannot contribute to the result of the 'and'.
    if ((DemandedMask & ~LHSKnownZero & RHSKnownOne) == 
        (DemandedMask & ~LHSKnownZero))
      return I->getOperand(0);
    if ((DemandedMask & ~RHSKnownZero & LHSKnownOne) == 
        (DemandedMask & ~RHSKnownZero))
      return I->getOperand(1);
    
    // If all of the demanded bits in the inputs are known zeros, return zero.
    if ((DemandedMask & (RHSKnownZero|LHSKnownZero)) == DemandedMask)
      return Constant::getNullValue(VTy);
      
    // If the RHS is a constant, see if we can simplify it.
    if (ShrinkDemandedConstant(I, 1, DemandedMask & ~LHSKnownZero))
      return I;
      
    // Output known-1 bits are only known if set in both the LHS & RHS.
    RHSKnownOne &= LHSKnownOne;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    RHSKnownZero |= LHSKnownZero;
    break;
  case Instruction::Or:
    // If either the LHS or the RHS are One, the result is One.
    if (SimplifyDemandedBits(I->getOperandUse(1), DemandedMask, 
                             RHSKnownZero, RHSKnownOne, Depth+1) ||
        SimplifyDemandedBits(I->getOperandUse(0), DemandedMask & ~RHSKnownOne, 
                             LHSKnownZero, LHSKnownOne, Depth+1))
      return I;
    assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?"); 
    assert(!(LHSKnownZero & LHSKnownOne) && "Bits known to be one AND zero?"); 
    
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'or'.
    if ((DemandedMask & ~LHSKnownOne & RHSKnownZero) == 
        (DemandedMask & ~LHSKnownOne))
      return I->getOperand(0);
    if ((DemandedMask & ~RHSKnownOne & LHSKnownZero) == 
        (DemandedMask & ~RHSKnownOne))
      return I->getOperand(1);

    // If all of the potentially set bits on one side are known to be set on
    // the other side, just use the 'other' side.
    if ((DemandedMask & (~RHSKnownZero) & LHSKnownOne) == 
        (DemandedMask & (~RHSKnownZero)))
      return I->getOperand(0);
    if ((DemandedMask & (~LHSKnownZero) & RHSKnownOne) == 
        (DemandedMask & (~LHSKnownZero)))
      return I->getOperand(1);
        
    // If the RHS is a constant, see if we can simplify it.
    if (ShrinkDemandedConstant(I, 1, DemandedMask))
      return I;
          
    // Output known-0 bits are only known if clear in both the LHS & RHS.
    RHSKnownZero &= LHSKnownZero;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    RHSKnownOne |= LHSKnownOne;
    break;
  case Instruction::Xor: {
    if (SimplifyDemandedBits(I->getOperandUse(1), DemandedMask,
                             RHSKnownZero, RHSKnownOne, Depth+1) ||
        SimplifyDemandedBits(I->getOperandUse(0), DemandedMask, 
                             LHSKnownZero, LHSKnownOne, Depth+1))
      return I;
    assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?"); 
    assert(!(LHSKnownZero & LHSKnownOne) && "Bits known to be one AND zero?"); 
    
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'xor'.
    if ((DemandedMask & RHSKnownZero) == DemandedMask)
      return I->getOperand(0);
    if ((DemandedMask & LHSKnownZero) == DemandedMask)
      return I->getOperand(1);
    
    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    APInt KnownZeroOut = (RHSKnownZero & LHSKnownZero) | 
                         (RHSKnownOne & LHSKnownOne);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    APInt KnownOneOut = (RHSKnownZero & LHSKnownOne) | 
                        (RHSKnownOne & LHSKnownZero);
    
    // If all of the demanded bits are known to be zero on one side or the
    // other, turn this into an *inclusive* or.
    //    e.g. (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
    if ((DemandedMask & ~RHSKnownZero & ~LHSKnownZero) == 0) {
      Instruction *Or =
        BinaryOperator::CreateOr(I->getOperand(0), I->getOperand(1),
                                 I->getName());
      return InsertNewInstBefore(Or, *I);
    }
    
    // If all of the demanded bits on one side are known, and all of the set
    // bits on that side are also known to be set on the other side, turn this
    // into an AND, as we know the bits will be cleared.
    //    e.g. (X | C1) ^ C2 --> (X | C1) & ~C2 iff (C1&C2) == C2
    if ((DemandedMask & (RHSKnownZero|RHSKnownOne)) == DemandedMask) { 
      // all known
      if ((RHSKnownOne & LHSKnownOne) == RHSKnownOne) {
        Constant *AndC = ConstantInt::get(~RHSKnownOne & DemandedMask);
        Instruction *And = 
          BinaryOperator::CreateAnd(I->getOperand(0), AndC, "tmp");
        return InsertNewInstBefore(And, *I);
      }
    }
    
    // If the RHS is a constant, see if we can simplify it.
    // FIXME: for XOR, we prefer to force bits to 1 if they will make a -1.
    if (ShrinkDemandedConstant(I, 1, DemandedMask))
      return I;
    
    RHSKnownZero = KnownZeroOut;
    RHSKnownOne  = KnownOneOut;
    break;
  }
  case Instruction::Select:
    if (SimplifyDemandedBits(I->getOperandUse(2), DemandedMask,
                             RHSKnownZero, RHSKnownOne, Depth+1) ||
        SimplifyDemandedBits(I->getOperandUse(1), DemandedMask, 
                             LHSKnownZero, LHSKnownOne, Depth+1))
      return I;
    assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?"); 
    assert(!(LHSKnownZero & LHSKnownOne) && "Bits known to be one AND zero?"); 
    
    // If the operands are constants, see if we can simplify them.
    if (ShrinkDemandedConstant(I, 1, DemandedMask) ||
        ShrinkDemandedConstant(I, 2, DemandedMask))
      return I;
    
    // Only known if known in both the LHS and RHS.
    RHSKnownOne &= LHSKnownOne;
    RHSKnownZero &= LHSKnownZero;
    break;
  case Instruction::Trunc: {
    unsigned truncBf = I->getOperand(0)->getType()->getPrimitiveSizeInBits();
    DemandedMask.zext(truncBf);
    RHSKnownZero.zext(truncBf);
    RHSKnownOne.zext(truncBf);
    if (SimplifyDemandedBits(I->getOperandUse(0), DemandedMask, 
                             RHSKnownZero, RHSKnownOne, Depth+1))
      return I;
    DemandedMask.trunc(BitWidth);
    RHSKnownZero.trunc(BitWidth);
    RHSKnownOne.trunc(BitWidth);
    assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?"); 
    break;
  }
  case Instruction::BitCast:
    if (!I->getOperand(0)->getType()->isInteger())
      return false;  // vector->int or fp->int?
    if (SimplifyDemandedBits(I->getOperandUse(0), DemandedMask,
                             RHSKnownZero, RHSKnownOne, Depth+1))
      return I;
    assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?"); 
    break;
  case Instruction::ZExt: {
    // Compute the bits in the result that are not present in the input.
    unsigned SrcBitWidth =I->getOperand(0)->getType()->getPrimitiveSizeInBits();
    
    DemandedMask.trunc(SrcBitWidth);
    RHSKnownZero.trunc(SrcBitWidth);
    RHSKnownOne.trunc(SrcBitWidth);
    if (SimplifyDemandedBits(I->getOperandUse(0), DemandedMask,
                             RHSKnownZero, RHSKnownOne, Depth+1))
      return I;
    DemandedMask.zext(BitWidth);
    RHSKnownZero.zext(BitWidth);
    RHSKnownOne.zext(BitWidth);
    assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?"); 
    // The top bits are known to be zero.
    RHSKnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth);
    break;
  }
  case Instruction::SExt: {
    // Compute the bits in the result that are not present in the input.
    unsigned SrcBitWidth =I->getOperand(0)->getType()->getPrimitiveSizeInBits();
    
    APInt InputDemandedBits = DemandedMask & 
                              APInt::getLowBitsSet(BitWidth, SrcBitWidth);

    APInt NewBits(APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth));
    // If any of the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    if ((NewBits & DemandedMask) != 0)
      InputDemandedBits.set(SrcBitWidth-1);
      
    InputDemandedBits.trunc(SrcBitWidth);
    RHSKnownZero.trunc(SrcBitWidth);
    RHSKnownOne.trunc(SrcBitWidth);
    if (SimplifyDemandedBits(I->getOperandUse(0), InputDemandedBits,
                             RHSKnownZero, RHSKnownOne, Depth+1))
      return I;
    InputDemandedBits.zext(BitWidth);
    RHSKnownZero.zext(BitWidth);
    RHSKnownOne.zext(BitWidth);
    assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?"); 
      
    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.

    // If the input sign bit is known zero, or if the NewBits are not demanded
    // convert this into a zero extension.
    if (RHSKnownZero[SrcBitWidth-1] || (NewBits & ~DemandedMask) == NewBits) {
      // Convert to ZExt cast
      CastInst *NewCast = new ZExtInst(I->getOperand(0), VTy, I->getName());
      return InsertNewInstBefore(NewCast, *I);
    } else if (RHSKnownOne[SrcBitWidth-1]) {    // Input sign bit known set
      RHSKnownOne |= NewBits;
    }
    break;
  }
  case Instruction::Add: {
    // Figure out what the input bits are.  If the top bits of the and result
    // are not demanded, then the add doesn't demand them from its input
    // either.
    unsigned NLZ = DemandedMask.countLeadingZeros();
      
    // If there is a constant on the RHS, there are a variety of xformations
    // we can do.
    if (ConstantInt *RHS = dyn_cast<ConstantInt>(I->getOperand(1))) {
      // If null, this should be simplified elsewhere.  Some of the xforms here
      // won't work if the RHS is zero.
      if (RHS->isZero())
        break;
      
      // If the top bit of the output is demanded, demand everything from the
      // input.  Otherwise, we demand all the input bits except NLZ top bits.
      APInt InDemandedBits(APInt::getLowBitsSet(BitWidth, BitWidth - NLZ));

      // Find information about known zero/one bits in the input.
      if (SimplifyDemandedBits(I->getOperandUse(0), InDemandedBits, 
                               LHSKnownZero, LHSKnownOne, Depth+1))
        return I;

      // If the RHS of the add has bits set that can't affect the input, reduce
      // the constant.
      if (ShrinkDemandedConstant(I, 1, InDemandedBits))
        return I;
      
      // Avoid excess work.
      if (LHSKnownZero == 0 && LHSKnownOne == 0)
        break;
      
      // Turn it into OR if input bits are zero.
      if ((LHSKnownZero & RHS->getValue()) == RHS->getValue()) {
        Instruction *Or =
          BinaryOperator::CreateOr(I->getOperand(0), I->getOperand(1),
                                   I->getName());
        return InsertNewInstBefore(Or, *I);
      }
      
      // We can say something about the output known-zero and known-one bits,
      // depending on potential carries from the input constant and the
      // unknowns.  For example if the LHS is known to have at most the 0x0F0F0
      // bits set and the RHS constant is 0x01001, then we know we have a known
      // one mask of 0x00001 and a known zero mask of 0xE0F0E.
      
      // To compute this, we first compute the potential carry bits.  These are
      // the bits which may be modified.  I'm not aware of a better way to do
      // this scan.
      const APInt &RHSVal = RHS->getValue();
      APInt CarryBits((~LHSKnownZero + RHSVal) ^ (~LHSKnownZero ^ RHSVal));
      
      // Now that we know which bits have carries, compute the known-1/0 sets.
      
      // Bits are known one if they are known zero in one operand and one in the
      // other, and there is no input carry.
      RHSKnownOne = ((LHSKnownZero & RHSVal) | 
                     (LHSKnownOne & ~RHSVal)) & ~CarryBits;
      
      // Bits are known zero if they are known zero in both operands and there
      // is no input carry.
      RHSKnownZero = LHSKnownZero & ~RHSVal & ~CarryBits;
    } else {
      // If the high-bits of this ADD are not demanded, then it does not demand
      // the high bits of its LHS or RHS.
      if (DemandedMask[BitWidth-1] == 0) {
        // Right fill the mask of bits for this ADD to demand the most
        // significant bit and all those below it.
        APInt DemandedFromOps(APInt::getLowBitsSet(BitWidth, BitWidth-NLZ));
        if (SimplifyDemandedBits(I->getOperandUse(0), DemandedFromOps,
                                 LHSKnownZero, LHSKnownOne, Depth+1) ||
            SimplifyDemandedBits(I->getOperandUse(1), DemandedFromOps,
                                 LHSKnownZero, LHSKnownOne, Depth+1))
          return I;
      }
    }
    break;
  }
  case Instruction::Sub:
    // If the high-bits of this SUB are not demanded, then it does not demand
    // the high bits of its LHS or RHS.
    if (DemandedMask[BitWidth-1] == 0) {
      // Right fill the mask of bits for this SUB to demand the most
      // significant bit and all those below it.
      uint32_t NLZ = DemandedMask.countLeadingZeros();
      APInt DemandedFromOps(APInt::getLowBitsSet(BitWidth, BitWidth-NLZ));
      if (SimplifyDemandedBits(I->getOperandUse(0), DemandedFromOps,
                               LHSKnownZero, LHSKnownOne, Depth+1) ||
          SimplifyDemandedBits(I->getOperandUse(1), DemandedFromOps,
                               LHSKnownZero, LHSKnownOne, Depth+1))
        return I;
    }
    // Otherwise just hand the sub off to ComputeMaskedBits to fill in
    // the known zeros and ones.
    ComputeMaskedBits(V, DemandedMask, RHSKnownZero, RHSKnownOne, Depth);
    break;
  case Instruction::Shl:
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      uint64_t ShiftAmt = SA->getLimitedValue(BitWidth);
      APInt DemandedMaskIn(DemandedMask.lshr(ShiftAmt));
      if (SimplifyDemandedBits(I->getOperandUse(0), DemandedMaskIn, 
                               RHSKnownZero, RHSKnownOne, Depth+1))
        return I;
      assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?");
      RHSKnownZero <<= ShiftAmt;
      RHSKnownOne  <<= ShiftAmt;
      // low bits known zero.
      if (ShiftAmt)
        RHSKnownZero |= APInt::getLowBitsSet(BitWidth, ShiftAmt);
    }
    break;
  case Instruction::LShr:
    // For a logical shift right
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      uint64_t ShiftAmt = SA->getLimitedValue(BitWidth);
      
      // Unsigned shift right.
      APInt DemandedMaskIn(DemandedMask.shl(ShiftAmt));
      if (SimplifyDemandedBits(I->getOperandUse(0), DemandedMaskIn,
                               RHSKnownZero, RHSKnownOne, Depth+1))
        return I;
      assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?");
      RHSKnownZero = APIntOps::lshr(RHSKnownZero, ShiftAmt);
      RHSKnownOne  = APIntOps::lshr(RHSKnownOne, ShiftAmt);
      if (ShiftAmt) {
        // Compute the new bits that are at the top now.
        APInt HighBits(APInt::getHighBitsSet(BitWidth, ShiftAmt));
        RHSKnownZero |= HighBits;  // high bits known zero.
      }
    }
    break;
  case Instruction::AShr:
    // If this is an arithmetic shift right and only the low-bit is set, we can
    // always convert this into a logical shr, even if the shift amount is
    // variable.  The low bit of the shift cannot be an input sign bit unless
    // the shift amount is >= the size of the datatype, which is undefined.
    if (DemandedMask == 1) {
      // Perform the logical shift right.
      Instruction *NewVal = BinaryOperator::CreateLShr(
                        I->getOperand(0), I->getOperand(1), I->getName());
      return InsertNewInstBefore(NewVal, *I);
    }    

    // If the sign bit is the only bit demanded by this ashr, then there is no
    // need to do it, the shift doesn't change the high bit.
    if (DemandedMask.isSignBit())
      return I->getOperand(0);
    
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      uint32_t ShiftAmt = SA->getLimitedValue(BitWidth);
      
      // Signed shift right.
      APInt DemandedMaskIn(DemandedMask.shl(ShiftAmt));
      // If any of the "high bits" are demanded, we should set the sign bit as
      // demanded.
      if (DemandedMask.countLeadingZeros() <= ShiftAmt)
        DemandedMaskIn.set(BitWidth-1);
      if (SimplifyDemandedBits(I->getOperandUse(0), DemandedMaskIn,
                               RHSKnownZero, RHSKnownOne, Depth+1))
        return I;
      assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?");
      // Compute the new bits that are at the top now.
      APInt HighBits(APInt::getHighBitsSet(BitWidth, ShiftAmt));
      RHSKnownZero = APIntOps::lshr(RHSKnownZero, ShiftAmt);
      RHSKnownOne  = APIntOps::lshr(RHSKnownOne, ShiftAmt);
        
      // Handle the sign bits.
      APInt SignBit(APInt::getSignBit(BitWidth));
      // Adjust to where it is now in the mask.
      SignBit = APIntOps::lshr(SignBit, ShiftAmt);  
        
      // If the input sign bit is known to be zero, or if none of the top bits
      // are demanded, turn this into an unsigned shift right.
      if (BitWidth <= ShiftAmt || RHSKnownZero[BitWidth-ShiftAmt-1] || 
          (HighBits & ~DemandedMask) == HighBits) {
        // Perform the logical shift right.
        Instruction *NewVal = BinaryOperator::CreateLShr(
                          I->getOperand(0), SA, I->getName());
        return InsertNewInstBefore(NewVal, *I);
      } else if ((RHSKnownOne & SignBit) != 0) { // New bits are known one.
        RHSKnownOne |= HighBits;
      }
    }
    break;
  case Instruction::SRem:
    if (ConstantInt *Rem = dyn_cast<ConstantInt>(I->getOperand(1))) {
      APInt RA = Rem->getValue().abs();
      if (RA.isPowerOf2()) {
        if (DemandedMask.ule(RA))    // srem won't affect demanded bits
          return I->getOperand(0);

        APInt LowBits = RA - 1;
        APInt Mask2 = LowBits | APInt::getSignBit(BitWidth);
        if (SimplifyDemandedBits(I->getOperandUse(0), Mask2,
                                 LHSKnownZero, LHSKnownOne, Depth+1))
          return I;

        if (LHSKnownZero[BitWidth-1] || ((LHSKnownZero & LowBits) == LowBits))
          LHSKnownZero |= ~LowBits;

        KnownZero |= LHSKnownZero & DemandedMask;

        assert(!(KnownZero & KnownOne) && "Bits known to be one AND zero?"); 
      }
    }
    break;
  case Instruction::URem: {
    APInt KnownZero2(BitWidth, 0), KnownOne2(BitWidth, 0);
    APInt AllOnes = APInt::getAllOnesValue(BitWidth);
    if (SimplifyDemandedBits(I->getOperandUse(0), AllOnes,
                             KnownZero2, KnownOne2, Depth+1) ||
        SimplifyDemandedBits(I->getOperandUse(1), AllOnes,
                             KnownZero2, KnownOne2, Depth+1))
      return I;

    unsigned Leaders = KnownZero2.countLeadingOnes();
    Leaders = std::max(Leaders,
                       KnownZero2.countLeadingOnes());
    KnownZero = APInt::getHighBitsSet(BitWidth, Leaders) & DemandedMask;
    break;
  }
  case Instruction::Call:
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::bswap: {
        // If the only bits demanded come from one byte of the bswap result,
        // just shift the input byte into position to eliminate the bswap.
        unsigned NLZ = DemandedMask.countLeadingZeros();
        unsigned NTZ = DemandedMask.countTrailingZeros();
          
        // Round NTZ down to the next byte.  If we have 11 trailing zeros, then
        // we need all the bits down to bit 8.  Likewise, round NLZ.  If we
        // have 14 leading zeros, round to 8.
        NLZ &= ~7;
        NTZ &= ~7;
        // If we need exactly one byte, we can do this transformation.
        if (BitWidth-NLZ-NTZ == 8) {
          unsigned ResultBit = NTZ;
          unsigned InputBit = BitWidth-NTZ-8;
          
          // Replace this with either a left or right shift to get the byte into
          // the right place.
          Instruction *NewVal;
          if (InputBit > ResultBit)
            NewVal = BinaryOperator::CreateLShr(I->getOperand(1),
                    ConstantInt::get(I->getType(), InputBit-ResultBit));
          else
            NewVal = BinaryOperator::CreateShl(I->getOperand(1),
                    ConstantInt::get(I->getType(), ResultBit-InputBit));
          NewVal->takeName(I);
          return InsertNewInstBefore(NewVal, *I);
        }
          
        // TODO: Could compute known zero/one bits based on the input.
        break;
      }
      }
    }
    ComputeMaskedBits(V, DemandedMask, RHSKnownZero, RHSKnownOne, Depth);
    break;
  }
  
  // If the client is only demanding bits that we know, return the known
  // constant.
  if ((DemandedMask & (RHSKnownZero|RHSKnownOne)) == DemandedMask)
    return ConstantInt::get(RHSKnownOne);
  return false;
}


/// SimplifyDemandedVectorElts - The specified value produces a vector with
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
  unsigned VWidth = cast<VectorType>(V->getType())->getNumElements();
  assert(VWidth <= 64 && "Vector too wide to analyze!");
  uint64_t EltMask = ~0ULL >> (64-VWidth);
  assert((DemandedElts & ~EltMask) == 0 && "Invalid DemandedElts!");

  if (isa<UndefValue>(V)) {
    // If the entire vector is undefined, just return this info.
    UndefElts = EltMask;
    return 0;
  } else if (DemandedElts == 0) { // If nothing is demanded, provide undef.
    UndefElts = EltMask;
    return UndefValue::get(V->getType());
  }

  UndefElts = 0;
  if (ConstantVector *CP = dyn_cast<ConstantVector>(V)) {
    const Type *EltTy = cast<VectorType>(V->getType())->getElementType();
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
    Constant *NewCP = ConstantVector::get(Elts);
    return NewCP != CP ? NewCP : 0;
  } else if (isa<ConstantAggregateZero>(V)) {
    // Simplify the CAZ to a ConstantVector where the non-demanded elements are
    // set to undef.
    
    // Check if this is identity. If so, return 0 since we are not simplifying
    // anything.
    if (DemandedElts == ((1ULL << VWidth) -1))
      return 0;
    
    const Type *EltTy = cast<VectorType>(V->getType())->getElementType();
    Constant *Zero = Constant::getNullValue(EltTy);
    Constant *Undef = UndefValue::get(EltTy);
    std::vector<Constant*> Elts;
    for (unsigned i = 0; i != VWidth; ++i)
      Elts.push_back((DemandedElts & (1ULL << i)) ? Zero : Undef);
    UndefElts = DemandedElts ^ EltMask;
    return ConstantVector::get(Elts);
  }
  
  // Limit search depth.
  if (Depth == 10)
    return false;

  // If multiple users are using the root value, procede with
  // simplification conservatively assuming that all elements
  // are needed.
  if (!V->hasOneUse()) {
    // Quit if we find multiple users of a non-root value though.
    // They'll be handled when it's their turn to be visited by
    // the main instcombine process.
    if (Depth != 0)
      // TODO: Just compute the UndefElts information recursively.
      return false;

    // Conservatively assume that all elements are needed.
    DemandedElts = EltMask;
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
    UndefElts &= ~(1ULL << IdxNo);
    break;
  }
  case Instruction::ShuffleVector: {
    ShuffleVectorInst *Shuffle = cast<ShuffleVectorInst>(I);
    uint64_t LHSVWidth =
      cast<VectorType>(Shuffle->getOperand(0)->getType())->getNumElements();
    uint64_t LeftDemanded = 0, RightDemanded = 0;
    for (unsigned i = 0; i < VWidth; i++) {
      if (DemandedElts & (1ULL << i)) {
        unsigned MaskVal = Shuffle->getMaskValue(i);
        if (MaskVal != -1u) {
          assert(MaskVal < LHSVWidth * 2 &&
                 "shufflevector mask index out of range!");
          if (MaskVal < LHSVWidth)
            LeftDemanded |= 1ULL << MaskVal;
          else
            RightDemanded |= 1ULL << (MaskVal - LHSVWidth);
        }
      }
    }

    TmpV = SimplifyDemandedVectorElts(I->getOperand(0), LeftDemanded,
                                      UndefElts2, Depth+1);
    if (TmpV) { I->setOperand(0, TmpV); MadeChange = true; }

    uint64_t UndefElts3;
    TmpV = SimplifyDemandedVectorElts(I->getOperand(1), RightDemanded,
                                      UndefElts3, Depth+1);
    if (TmpV) { I->setOperand(1, TmpV); MadeChange = true; }

    bool NewUndefElts = false;
    for (unsigned i = 0; i < VWidth; i++) {
      unsigned MaskVal = Shuffle->getMaskValue(i);
      if (MaskVal == -1u) {
        uint64_t NewBit = 1ULL << i;
        UndefElts |= NewBit;
      } else if (MaskVal < LHSVWidth) {
        uint64_t NewBit = ((UndefElts2 >> MaskVal) & 1) << i;
        NewUndefElts |= NewBit;
        UndefElts |= NewBit;
      } else {
        uint64_t NewBit = ((UndefElts3 >> (MaskVal - LHSVWidth)) & 1) << i;
        NewUndefElts |= NewBit;
        UndefElts |= NewBit;
      }
    }

    if (NewUndefElts) {
      // Add additional discovered undefs.
      std::vector<Constant*> Elts;
      for (unsigned i = 0; i < VWidth; ++i) {
        if (UndefElts & (1ULL << i))
          Elts.push_back(UndefValue::get(Type::Int32Ty));
        else
          Elts.push_back(ConstantInt::get(Type::Int32Ty,
                                          Shuffle->getMaskValue(i)));
      }
      I->setOperand(2, ConstantVector::get(Elts));
      MadeChange = true;
    }
    break;
  }
  case Instruction::BitCast: {
    // Vector->vector casts only.
    const VectorType *VTy = dyn_cast<VectorType>(I->getOperand(0)->getType());
    if (!VTy) break;
    unsigned InVWidth = VTy->getNumElements();
    uint64_t InputDemandedElts = 0;
    unsigned Ratio;

    if (VWidth == InVWidth) {
      // If we are converting from <4 x i32> -> <4 x f32>, we demand the same
      // elements as are demanded of us.
      Ratio = 1;
      InputDemandedElts = DemandedElts;
    } else if (VWidth > InVWidth) {
      // Untested so far.
      break;
      
      // If there are more elements in the result than there are in the source,
      // then an input element is live if any of the corresponding output
      // elements are live.
      Ratio = VWidth/InVWidth;
      for (unsigned OutIdx = 0; OutIdx != VWidth; ++OutIdx) {
        if (DemandedElts & (1ULL << OutIdx))
          InputDemandedElts |= 1ULL << (OutIdx/Ratio);
      }
    } else {
      // Untested so far.
      break;
      
      // If there are more elements in the source than there are in the result,
      // then an input element is live if the corresponding output element is
      // live.
      Ratio = InVWidth/VWidth;
      for (unsigned InIdx = 0; InIdx != InVWidth; ++InIdx)
        if (DemandedElts & (1ULL << InIdx/Ratio))
          InputDemandedElts |= 1ULL << InIdx;
    }
    
    // div/rem demand all inputs, because they don't want divide by zero.
    TmpV = SimplifyDemandedVectorElts(I->getOperand(0), InputDemandedElts,
                                      UndefElts2, Depth+1);
    if (TmpV) {
      I->setOperand(0, TmpV);
      MadeChange = true;
    }
    
    UndefElts = UndefElts2;
    if (VWidth > InVWidth) {
      assert(0 && "Unimp");
      // If there are more elements in the result than there are in the source,
      // then an output element is undef if the corresponding input element is
      // undef.
      for (unsigned OutIdx = 0; OutIdx != VWidth; ++OutIdx)
        if (UndefElts2 & (1ULL << (OutIdx/Ratio)))
          UndefElts |= 1ULL << OutIdx;
    } else if (VWidth < InVWidth) {
      assert(0 && "Unimp");
      // If there are more elements in the source than there are in the result,
      // then a result element is undef if all of the corresponding input
      // elements are undef.
      UndefElts = ~0ULL >> (64-VWidth);  // Start out all undef.
      for (unsigned InIdx = 0; InIdx != InVWidth; ++InIdx)
        if ((UndefElts2 & (1ULL << InIdx)) == 0)    // Not undef?
          UndefElts &= ~(1ULL << (InIdx/Ratio));    // Clear undef bit.
    }
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
            TmpV = InsertNewInstBefore(BinaryOperator::CreateSub(LHS, RHS,
                                                        II->getName()), *II);
            break;
          case Intrinsic::x86_sse_mul_ss:
          case Intrinsic::x86_sse2_mul_sd:
            TmpV = InsertNewInstBefore(BinaryOperator::CreateMul(LHS, RHS,
                                                         II->getName()), *II);
            break;
          }
          
          Instruction *New =
            InsertElementInst::Create(UndefValue::get(II->getType()), TmpV, 0U,
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


/// AssociativeOpt - Perform an optimization on an associative operator.  This
/// function is designed to check a chain of associative operators for a
/// potential to apply a certain optimization.  Since the optimization may be
/// applicable if the expression was reassociated, this checks the chain, then
/// reassociates the expression as necessary to expose the optimization
/// opportunity.  This makes use of a special Functor, which must define
/// 'shouldApply' and 'apply' methods.
///
template<typename Functor>
static Instruction *AssociativeOpt(BinaryOperator &Root, const Functor &F) {
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
      BasicBlock::iterator ARI = &Root; ++ARI;
      TmpLHSI->moveBefore(ARI);                  // Move TmpLHSI to after Root
      ARI = Root;

      // Now propagate the ExtraOperand down the chain of instructions until we
      // get to LHSI.
      while (TmpLHSI != LHSI) {
        Instruction *NextLHSI = cast<Instruction>(TmpLHSI->getOperand(0));
        // Move the instruction to immediately before the chain we are
        // constructing to avoid breaking dominance properties.
        NextLHSI->moveBefore(ARI);
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

namespace {

// AddRHS - Implements: X + X --> X << 1
struct AddRHS {
  Value *RHS;
  AddRHS(Value *rhs) : RHS(rhs) {}
  bool shouldApply(Value *LHS) const { return LHS == RHS; }
  Instruction *apply(BinaryOperator &Add) const {
    return BinaryOperator::CreateShl(Add.getOperand(0),
                                     ConstantInt::get(Add.getType(), 1));
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
    return BinaryOperator::CreateOr(Add.getOperand(0), Add.getOperand(1));
  }
};

}

static Value *FoldOperationIntoSelectOperand(Instruction &I, Value *SO,
                                             InstCombiner *IC) {
  if (CastInst *CI = dyn_cast<CastInst>(&I)) {
    return IC->InsertCastBefore(CI->getOpcode(), SO, I.getType(), I);
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
    New = BinaryOperator::Create(BO->getOpcode(), Op0, Op1,SO->getName()+".op");
  else if (CmpInst *CI = dyn_cast<CmpInst>(&I))
    New = CmpInst::Create(CI->getOpcode(), CI->getPredicate(), Op0, Op1, 
                          SO->getName()+".cmp");
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
    if (SI->getType() == Type::Int1Ty) return 0;

    Value *SelectTrueVal = FoldOperationIntoSelectOperand(Op, TV, IC);
    Value *SelectFalseVal = FoldOperationIntoSelectOperand(Op, FV, IC);

    return SelectInst::Create(SI->getCondition(), SelectTrueVal,
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
  // or if *it* is a PHI, bail out.
  BasicBlock *NonConstBB = 0;
  for (unsigned i = 0; i != NumPHIValues; ++i)
    if (!isa<Constant>(PN->getIncomingValue(i))) {
      if (NonConstBB) return 0;  // More than one non-const value.
      if (isa<PHINode>(PN->getIncomingValue(i))) return 0;  // Itself a phi.
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
  PHINode *NewPN = PHINode::Create(I.getType(), "");
  NewPN->reserveOperandSpace(PN->getNumOperands()/2);
  InsertNewInstBefore(NewPN, *PN);
  NewPN->takeName(PN);

  // Next, add all of the operands to the PHI.
  if (I.getNumOperands() == 2) {
    Constant *C = cast<Constant>(I.getOperand(1));
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV = 0;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i))) {
        if (CmpInst *CI = dyn_cast<CmpInst>(&I))
          InV = ConstantExpr::getCompare(CI->getPredicate(), InC, C);
        else
          InV = ConstantExpr::get(I.getOpcode(), InC, C);
      } else {
        assert(PN->getIncomingBlock(i) == NonConstBB);
        if (BinaryOperator *BO = dyn_cast<BinaryOperator>(&I)) 
          InV = BinaryOperator::Create(BO->getOpcode(),
                                       PN->getIncomingValue(i), C, "phitmp",
                                       NonConstBB->getTerminator());
        else if (CmpInst *CI = dyn_cast<CmpInst>(&I))
          InV = CmpInst::Create(CI->getOpcode(), 
                                CI->getPredicate(),
                                PN->getIncomingValue(i), C, "phitmp",
                                NonConstBB->getTerminator());
        else
          assert(0 && "Unknown binop!");
        
        AddToWorkList(cast<Instruction>(InV));
      }
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  } else { 
    CastInst *CI = cast<CastInst>(&I);
    const Type *RetTy = CI->getType();
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i))) {
        InV = ConstantExpr::getCast(CI->getOpcode(), InC, RetTy);
      } else {
        assert(PN->getIncomingBlock(i) == NonConstBB);
        InV = CastInst::Create(CI->getOpcode(), PN->getIncomingValue(i), 
                               I.getType(), "phitmp", 
                               NonConstBB->getTerminator());
        AddToWorkList(cast<Instruction>(InV));
      }
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  }
  return ReplaceInstUsesWith(I, NewPN);
}


/// WillNotOverflowSignedAdd - Return true if we can prove that:
///    (sext (add LHS, RHS))  === (add (sext LHS), (sext RHS))
/// This basically requires proving that the add in the original type would not
/// overflow to change the sign bit or have a carry out.
bool InstCombiner::WillNotOverflowSignedAdd(Value *LHS, Value *RHS) {
  // There are different heuristics we can use for this.  Here are some simple
  // ones.
  
  // Add has the property that adding any two 2's complement numbers can only 
  // have one carry bit which can change a sign.  As such, if LHS and RHS each
  // have at least two sign bits, we know that the addition of the two values will
  // sign extend fine.
  if (ComputeNumSignBits(LHS) > 1 && ComputeNumSignBits(RHS) > 1)
    return true;
  
  
  // If one of the operands only has one non-zero bit, and if the other operand
  // has a known-zero bit in a more significant place than it (not including the
  // sign bit) the ripple may go up to and fill the zero, but won't change the
  // sign.  For example, (X & ~4) + 1.
  
  // TODO: Implement.
  
  return false;
}


Instruction *InstCombiner::visitAdd(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
    // X + undef -> undef
    if (isa<UndefValue>(RHS))
      return ReplaceInstUsesWith(I, RHS);

    // X + 0 --> X
    if (!I.getType()->isFPOrFPVector()) { // NOTE: -0 + +0 = +0.
      if (RHSC->isNullValue())
        return ReplaceInstUsesWith(I, LHS);
    } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHSC)) {
      if (CFP->isExactlyValue(ConstantFP::getNegativeZero
                              (I.getType())->getValueAPF()))
        return ReplaceInstUsesWith(I, LHS);
    }

    if (ConstantInt *CI = dyn_cast<ConstantInt>(RHSC)) {
      // X + (signbit) --> X ^ signbit
      const APInt& Val = CI->getValue();
      uint32_t BitWidth = Val.getBitWidth();
      if (Val == APInt::getSignBit(BitWidth))
        return BinaryOperator::CreateXor(LHS, RHS);
      
      // See if SimplifyDemandedBits can simplify this.  This handles stuff like
      // (X & 254)+1 -> (X&254)|1
      if (!isa<VectorType>(I.getType()) && SimplifyDemandedInstructionBits(I))
        return &I;

      // zext(i1) - 1  ->  select i1, 0, -1
      if (ZExtInst *ZI = dyn_cast<ZExtInst>(LHS))
        if (CI->isAllOnesValue() &&
            ZI->getOperand(0)->getType() == Type::Int1Ty)
          return SelectInst::Create(ZI->getOperand(0),
                                    Constant::getNullValue(I.getType()),
                                    ConstantInt::getAllOnesValue(I.getType()));
    }

    if (isa<PHINode>(LHS))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
    
    ConstantInt *XorRHS = 0;
    Value *XorLHS = 0;
    if (isa<ConstantInt>(RHSC) &&
        match(LHS, m_Xor(m_Value(XorLHS), m_ConstantInt(XorRHS)))) {
      uint32_t TySizeBits = I.getType()->getPrimitiveSizeInBits();
      const APInt& RHSVal = cast<ConstantInt>(RHSC)->getValue();
      
      uint32_t Size = TySizeBits / 2;
      APInt C0080Val(APInt(TySizeBits, 1ULL).shl(Size - 1));
      APInt CFF80Val(-C0080Val);
      do {
        if (TySizeBits > Size) {
          // If we have ADD(XOR(AND(X, 0xFF), 0x80), 0xF..F80), it's a sext.
          // If we have ADD(XOR(AND(X, 0xFF), 0xF..F80), 0x80), it's a sext.
          if ((RHSVal == CFF80Val && XorRHS->getValue() == C0080Val) ||
              (RHSVal == C0080Val && XorRHS->getValue() == CFF80Val)) {
            // This is a sign extend if the top bits are known zero.
            if (!MaskedValueIsZero(XorLHS, 
                   APInt::getHighBitsSet(TySizeBits, TySizeBits - Size)))
              Size = 0;  // Not a sign ext, but can't be any others either.
            break;
          }
        }
        Size >>= 1;
        C0080Val = APIntOps::lshr(C0080Val, Size);
        CFF80Val = APIntOps::ashr(CFF80Val, Size);
      } while (Size >= 1);
      
      // FIXME: This shouldn't be necessary. When the backends can handle types
      // with funny bit widths then this switch statement should be removed. It
      // is just here to get the size of the "middle" type back up to something
      // that the back ends can handle.
      const Type *MiddleType = 0;
      switch (Size) {
        default: break;
        case 32: MiddleType = Type::Int32Ty; break;
        case 16: MiddleType = Type::Int16Ty; break;
        case  8: MiddleType = Type::Int8Ty; break;
      }
      if (MiddleType) {
        Instruction *NewTrunc = new TruncInst(XorLHS, MiddleType, "sext");
        InsertNewInstBefore(NewTrunc, I);
        return new SExtInst(NewTrunc, I.getType(), I.getName());
      }
    }
  }

  if (I.getType() == Type::Int1Ty)
    return BinaryOperator::CreateXor(LHS, RHS);

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
  // -A + -B  -->  -(A + B)
  if (Value *LHSV = dyn_castNegVal(LHS)) {
    if (LHS->getType()->isIntOrIntVector()) {
      if (Value *RHSV = dyn_castNegVal(RHS)) {
        Instruction *NewAdd = BinaryOperator::CreateAdd(LHSV, RHSV, "sum");
        InsertNewInstBefore(NewAdd, I);
        return BinaryOperator::CreateNeg(NewAdd);
      }
    }
    
    return BinaryOperator::CreateSub(RHS, LHSV);
  }

  // A + -B  -->  A - B
  if (!isa<Constant>(RHS))
    if (Value *V = dyn_castNegVal(RHS))
      return BinaryOperator::CreateSub(LHS, V);


  ConstantInt *C2;
  if (Value *X = dyn_castFoldableMul(LHS, C2)) {
    if (X == RHS)   // X*C + X --> X * (C+1)
      return BinaryOperator::CreateMul(RHS, AddOne(C2));

    // X*C1 + X*C2 --> X * (C1+C2)
    ConstantInt *C1;
    if (X == dyn_castFoldableMul(RHS, C1))
      return BinaryOperator::CreateMul(X, Add(C1, C2));
  }

  // X + X*C --> X * (C+1)
  if (dyn_castFoldableMul(RHS, C2) == LHS)
    return BinaryOperator::CreateMul(LHS, AddOne(C2));

  // X + ~X --> -1   since   ~X = -X-1
  if (dyn_castNotVal(LHS) == RHS || dyn_castNotVal(RHS) == LHS)
    return ReplaceInstUsesWith(I, Constant::getAllOnesValue(I.getType()));
  

  // (A & C1)+(B & C2) --> (A & C1)|(B & C2) iff C1&C2 == 0
  if (match(RHS, m_And(m_Value(), m_ConstantInt(C2))))
    if (Instruction *R = AssociativeOpt(I, AddMaskingAnd(C2)))
      return R;
  
  // A+B --> A|B iff A and B have no bits set in common.
  if (const IntegerType *IT = dyn_cast<IntegerType>(I.getType())) {
    APInt Mask = APInt::getAllOnesValue(IT->getBitWidth());
    APInt LHSKnownOne(IT->getBitWidth(), 0);
    APInt LHSKnownZero(IT->getBitWidth(), 0);
    ComputeMaskedBits(LHS, Mask, LHSKnownZero, LHSKnownOne);
    if (LHSKnownZero != 0) {
      APInt RHSKnownOne(IT->getBitWidth(), 0);
      APInt RHSKnownZero(IT->getBitWidth(), 0);
      ComputeMaskedBits(RHS, Mask, RHSKnownZero, RHSKnownOne);
      
      // No bits in common -> bitwise or.
      if ((LHSKnownZero|RHSKnownZero).isAllOnesValue())
        return BinaryOperator::CreateOr(LHS, RHS);
    }
  }

  // W*X + Y*Z --> W * (X+Z)  iff W == Y
  if (I.getType()->isIntOrIntVector()) {
    Value *W, *X, *Y, *Z;
    if (match(LHS, m_Mul(m_Value(W), m_Value(X))) &&
        match(RHS, m_Mul(m_Value(Y), m_Value(Z)))) {
      if (W != Y) {
        if (W == Z) {
          std::swap(Y, Z);
        } else if (Y == X) {
          std::swap(W, X);
        } else if (X == Z) {
          std::swap(Y, Z);
          std::swap(W, X);
        }
      }

      if (W == Y) {
        Value *NewAdd = InsertNewInstBefore(BinaryOperator::CreateAdd(X, Z,
                                                            LHS->getName()), I);
        return BinaryOperator::CreateMul(W, NewAdd);
      }
    }
  }

  if (ConstantInt *CRHS = dyn_cast<ConstantInt>(RHS)) {
    Value *X = 0;
    if (match(LHS, m_Not(m_Value(X))))    // ~X + C --> (C-1) - X
      return BinaryOperator::CreateSub(SubOne(CRHS), X);

    // (X & FF00) + xx00  -> (X+xx00) & FF00
    if (LHS->hasOneUse() && match(LHS, m_And(m_Value(X), m_ConstantInt(C2)))) {
      Constant *Anded = And(CRHS, C2);
      if (Anded == CRHS) {
        // See if all bits from the first bit set in the Add RHS up are included
        // in the mask.  First, get the rightmost bit.
        const APInt& AddRHSV = CRHS->getValue();

        // Form a mask of all bits from the lowest bit added through the top.
        APInt AddRHSHighBits(~((AddRHSV & -AddRHSV)-1));

        // See if the and mask includes all of these bits.
        APInt AddRHSHighBitsAnd(AddRHSHighBits & C2->getValue());

        if (AddRHSHighBits == AddRHSHighBitsAnd) {
          // Okay, the xform is safe.  Insert the new add pronto.
          Value *NewAdd = InsertNewInstBefore(BinaryOperator::CreateAdd(X, CRHS,
                                                            LHS->getName()), I);
          return BinaryOperator::CreateAnd(NewAdd, C2);
        }
      }
    }

    // Try to fold constant add into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(LHS))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;
  }

  // add (cast *A to intptrtype) B -> 
  //   cast (GEP (cast *A to sbyte*) B)  -->  intptrtype
  {
    CastInst *CI = dyn_cast<CastInst>(LHS);
    Value *Other = RHS;
    if (!CI) {
      CI = dyn_cast<CastInst>(RHS);
      Other = LHS;
    }
    if (CI && CI->getType()->isSized() && 
        (CI->getType()->getPrimitiveSizeInBits() == 
         TD->getIntPtrType()->getPrimitiveSizeInBits()) 
        && isa<PointerType>(CI->getOperand(0)->getType())) {
      unsigned AS =
        cast<PointerType>(CI->getOperand(0)->getType())->getAddressSpace();
      Value *I2 = InsertBitCastBefore(CI->getOperand(0),
                                      PointerType::get(Type::Int8Ty, AS), I);
      I2 = InsertNewInstBefore(GetElementPtrInst::Create(I2, Other, "ctg2"), I);
      return new PtrToIntInst(I2, CI->getType());
    }
  }
  
  // add (select X 0 (sub n A)) A  -->  select X A n
  {
    SelectInst *SI = dyn_cast<SelectInst>(LHS);
    Value *A = RHS;
    if (!SI) {
      SI = dyn_cast<SelectInst>(RHS);
      A = LHS;
    }
    if (SI && SI->hasOneUse()) {
      Value *TV = SI->getTrueValue();
      Value *FV = SI->getFalseValue();
      Value *N;

      // Can we fold the add into the argument of the select?
      // We check both true and false select arguments for a matching subtract.
      if (match(FV, m_Zero()) && match(TV, m_Sub(m_Value(N), m_Specific(A))))
        // Fold the add into the true select value.
        return SelectInst::Create(SI->getCondition(), N, A);
      if (match(TV, m_Zero()) && match(FV, m_Sub(m_Value(N), m_Specific(A))))
        // Fold the add into the false select value.
        return SelectInst::Create(SI->getCondition(), A, N);
    }
  }
  
  // Check for X+0.0.  Simplify it to X if we know X is not -0.0.
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHS))
    if (CFP->getValueAPF().isPosZero() && CannotBeNegativeZero(LHS))
      return ReplaceInstUsesWith(I, LHS);

  // Check for (add (sext x), y), see if we can merge this into an
  // integer add followed by a sext.
  if (SExtInst *LHSConv = dyn_cast<SExtInst>(LHS)) {
    // (add (sext x), cst) --> (sext (add x, cst'))
    if (ConstantInt *RHSC = dyn_cast<ConstantInt>(RHS)) {
      Constant *CI = 
        ConstantExpr::getTrunc(RHSC, LHSConv->getOperand(0)->getType());
      if (LHSConv->hasOneUse() &&
          ConstantExpr::getSExt(CI, I.getType()) == RHSC &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0), CI)) {
        // Insert the new, smaller add.
        Instruction *NewAdd = BinaryOperator::CreateAdd(LHSConv->getOperand(0), 
                                                        CI, "addconv");
        InsertNewInstBefore(NewAdd, I);
        return new SExtInst(NewAdd, I.getType());
      }
    }
    
    // (add (sext x), (sext y)) --> (sext (add int x, y))
    if (SExtInst *RHSConv = dyn_cast<SExtInst>(RHS)) {
      // Only do this if x/y have the same type, if at last one of them has a
      // single use (so we don't increase the number of sexts), and if the
      // integer add will not overflow.
      if (LHSConv->getOperand(0)->getType()==RHSConv->getOperand(0)->getType()&&
          (LHSConv->hasOneUse() || RHSConv->hasOneUse()) &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0),
                                   RHSConv->getOperand(0))) {
        // Insert the new integer add.
        Instruction *NewAdd = BinaryOperator::CreateAdd(LHSConv->getOperand(0), 
                                                        RHSConv->getOperand(0),
                                                        "addconv");
        InsertNewInstBefore(NewAdd, I);
        return new SExtInst(NewAdd, I.getType());
      }
    }
  }
  
  // Check for (add double (sitofp x), y), see if we can merge this into an
  // integer add followed by a promotion.
  if (SIToFPInst *LHSConv = dyn_cast<SIToFPInst>(LHS)) {
    // (add double (sitofp x), fpcst) --> (sitofp (add int x, intcst))
    // ... if the constant fits in the integer value.  This is useful for things
    // like (double)(x & 1234) + 4.0 -> (double)((X & 1234)+4) which no longer
    // requires a constant pool load, and generally allows the add to be better
    // instcombined.
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHS)) {
      Constant *CI = 
      ConstantExpr::getFPToSI(CFP, LHSConv->getOperand(0)->getType());
      if (LHSConv->hasOneUse() &&
          ConstantExpr::getSIToFP(CI, I.getType()) == CFP &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0), CI)) {
        // Insert the new integer add.
        Instruction *NewAdd = BinaryOperator::CreateAdd(LHSConv->getOperand(0), 
                                                        CI, "addconv");
        InsertNewInstBefore(NewAdd, I);
        return new SIToFPInst(NewAdd, I.getType());
      }
    }
    
    // (add double (sitofp x), (sitofp y)) --> (sitofp (add int x, y))
    if (SIToFPInst *RHSConv = dyn_cast<SIToFPInst>(RHS)) {
      // Only do this if x/y have the same type, if at last one of them has a
      // single use (so we don't increase the number of int->fp conversions),
      // and if the integer add will not overflow.
      if (LHSConv->getOperand(0)->getType()==RHSConv->getOperand(0)->getType()&&
          (LHSConv->hasOneUse() || RHSConv->hasOneUse()) &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0),
                                   RHSConv->getOperand(0))) {
        // Insert the new integer add.
        Instruction *NewAdd = BinaryOperator::CreateAdd(LHSConv->getOperand(0), 
                                                        RHSConv->getOperand(0),
                                                        "addconv");
        InsertNewInstBefore(NewAdd, I);
        return new SIToFPInst(NewAdd, I.getType());
      }
    }
  }
  
  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitSub(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Op0 == Op1 &&                        // sub X, X  -> 0
      !I.getType()->isFPOrFPVector())
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // If this is a 'B = x-(-A)', change to B = x+A...
  if (Value *V = dyn_castNegVal(Op1))
    return BinaryOperator::CreateAdd(Op0, V);

  if (isa<UndefValue>(Op0))
    return ReplaceInstUsesWith(I, Op0);    // undef - X -> undef
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);    // X - undef -> undef

  if (ConstantInt *C = dyn_cast<ConstantInt>(Op0)) {
    // Replace (-1 - A) with (~A)...
    if (C->isAllOnesValue())
      return BinaryOperator::CreateNot(Op1);

    // C - ~X == X + (1+C)
    Value *X = 0;
    if (match(Op1, m_Not(m_Value(X))))
      return BinaryOperator::CreateAdd(X, AddOne(C));

    // -(X >>u 31) -> (X >>s 31)
    // -(X >>s 31) -> (X >>u 31)
    if (C->isZero()) {
      if (BinaryOperator *SI = dyn_cast<BinaryOperator>(Op1)) {
        if (SI->getOpcode() == Instruction::LShr) {
          if (ConstantInt *CU = dyn_cast<ConstantInt>(SI->getOperand(1))) {
            // Check to see if we are shifting out everything but the sign bit.
            if (CU->getLimitedValue(SI->getType()->getPrimitiveSizeInBits()) ==
                SI->getType()->getPrimitiveSizeInBits()-1) {
              // Ok, the transformation is safe.  Insert AShr.
              return BinaryOperator::Create(Instruction::AShr, 
                                          SI->getOperand(0), CU, SI->getName());
            }
          }
        }
        else if (SI->getOpcode() == Instruction::AShr) {
          if (ConstantInt *CU = dyn_cast<ConstantInt>(SI->getOperand(1))) {
            // Check to see if we are shifting out everything but the sign bit.
            if (CU->getLimitedValue(SI->getType()->getPrimitiveSizeInBits()) ==
                SI->getType()->getPrimitiveSizeInBits()-1) {
              // Ok, the transformation is safe.  Insert LShr. 
              return BinaryOperator::CreateLShr(
                                          SI->getOperand(0), CU, SI->getName());
            }
          }
        }
      }
    }

    // Try to fold constant sub into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;
  }

  if (I.getType() == Type::Int1Ty)
    return BinaryOperator::CreateXor(Op0, Op1);

  if (BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1)) {
    if (Op1I->getOpcode() == Instruction::Add &&
        !Op0->getType()->isFPOrFPVector()) {
      if (Op1I->getOperand(0) == Op0)              // X-(X+Y) == -Y
        return BinaryOperator::CreateNeg(Op1I->getOperand(1), I.getName());
      else if (Op1I->getOperand(1) == Op0)         // X-(Y+X) == -Y
        return BinaryOperator::CreateNeg(Op1I->getOperand(0), I.getName());
      else if (ConstantInt *CI1 = dyn_cast<ConstantInt>(I.getOperand(0))) {
        if (ConstantInt *CI2 = dyn_cast<ConstantInt>(Op1I->getOperand(1)))
          // C1-(X+C2) --> (C1-C2)-X
          return BinaryOperator::CreateSub(Subtract(CI1, CI2), 
                                           Op1I->getOperand(0));
      }
    }

    if (Op1I->hasOneUse()) {
      // Replace (x - (y - z)) with (x + (z - y)) if the (y - z) subexpression
      // is not used by anyone else...
      //
      if (Op1I->getOpcode() == Instruction::Sub &&
          !Op1I->getType()->isFPOrFPVector()) {
        // Swap the two operands of the subexpr...
        Value *IIOp0 = Op1I->getOperand(0), *IIOp1 = Op1I->getOperand(1);
        Op1I->setOperand(0, IIOp1);
        Op1I->setOperand(1, IIOp0);

        // Create the new top level add instruction...
        return BinaryOperator::CreateAdd(Op0, Op1);
      }

      // Replace (A - (A & B)) with (A & ~B) if this is the only use of (A&B)...
      //
      if (Op1I->getOpcode() == Instruction::And &&
          (Op1I->getOperand(0) == Op0 || Op1I->getOperand(1) == Op0)) {
        Value *OtherOp = Op1I->getOperand(Op1I->getOperand(0) == Op0);

        Value *NewNot =
          InsertNewInstBefore(BinaryOperator::CreateNot(OtherOp, "B.not"), I);
        return BinaryOperator::CreateAnd(Op0, NewNot);
      }

      // 0 - (X sdiv C)  -> (X sdiv -C)
      if (Op1I->getOpcode() == Instruction::SDiv)
        if (ConstantInt *CSI = dyn_cast<ConstantInt>(Op0))
          if (CSI->isZero())
            if (Constant *DivRHS = dyn_cast<Constant>(Op1I->getOperand(1)))
              return BinaryOperator::CreateSDiv(Op1I->getOperand(0),
                                               ConstantExpr::getNeg(DivRHS));

      // X - X*C --> X * (1-C)
      ConstantInt *C2 = 0;
      if (dyn_castFoldableMul(Op1I, C2) == Op0) {
        Constant *CP1 = Subtract(ConstantInt::get(I.getType(), 1), C2);
        return BinaryOperator::CreateMul(Op0, CP1);
      }
    }
  }

  if (!Op0->getType()->isFPOrFPVector())
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      if (Op0I->getOpcode() == Instruction::Add) {
        if (Op0I->getOperand(0) == Op1)             // (Y+X)-Y == X
          return ReplaceInstUsesWith(I, Op0I->getOperand(1));
        else if (Op0I->getOperand(1) == Op1)        // (X+Y)-Y == X
          return ReplaceInstUsesWith(I, Op0I->getOperand(0));
      } else if (Op0I->getOpcode() == Instruction::Sub) {
        if (Op0I->getOperand(0) == Op1)             // (X-Y)-X == -Y
          return BinaryOperator::CreateNeg(Op0I->getOperand(1), I.getName());
      }
    }

  ConstantInt *C1;
  if (Value *X = dyn_castFoldableMul(Op0, C1)) {
    if (X == Op1)  // X*C - X --> X * (C-1)
      return BinaryOperator::CreateMul(Op1, SubOne(C1));

    ConstantInt *C2;   // X*C1 - X*C2 -> X * (C1-C2)
    if (X == dyn_castFoldableMul(Op1, C2))
      return BinaryOperator::CreateMul(X, Subtract(C1, C2));
  }
  return 0;
}

/// isSignBitCheck - Given an exploded icmp instruction, return true if the
/// comparison only checks the sign bit.  If it only checks the sign bit, set
/// TrueIfSigned if the result of the comparison is true when the input value is
/// signed.
static bool isSignBitCheck(ICmpInst::Predicate pred, ConstantInt *RHS,
                           bool &TrueIfSigned) {
  switch (pred) {
  case ICmpInst::ICMP_SLT:   // True if LHS s< 0
    TrueIfSigned = true;
    return RHS->isZero();
  case ICmpInst::ICMP_SLE:   // True if LHS s<= RHS and RHS == -1
    TrueIfSigned = true;
    return RHS->isAllOnesValue();
  case ICmpInst::ICMP_SGT:   // True if LHS s> -1
    TrueIfSigned = false;
    return RHS->isAllOnesValue();
  case ICmpInst::ICMP_UGT:
    // True if LHS u> RHS and RHS == high-bit-mask - 1
    TrueIfSigned = true;
    return RHS->getValue() ==
      APInt::getSignedMaxValue(RHS->getType()->getPrimitiveSizeInBits());
  case ICmpInst::ICMP_UGE: 
    // True if LHS u>= RHS and RHS == high-bit-mask (2^7, 2^15, 2^31, etc)
    TrueIfSigned = true;
    return RHS->getValue().isSignBit();
  default:
    return false;
  }
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
      if (BinaryOperator *SI = dyn_cast<BinaryOperator>(Op0))
        if (SI->getOpcode() == Instruction::Shl)
          if (Constant *ShOp = dyn_cast<Constant>(SI->getOperand(1)))
            return BinaryOperator::CreateMul(SI->getOperand(0),
                                             ConstantExpr::getShl(CI, ShOp));

      if (CI->isZero())
        return ReplaceInstUsesWith(I, Op1);  // X * 0  == 0
      if (CI->equalsInt(1))                  // X * 1  == X
        return ReplaceInstUsesWith(I, Op0);
      if (CI->isAllOnesValue())              // X * -1 == 0 - X
        return BinaryOperator::CreateNeg(Op0, I.getName());

      const APInt& Val = cast<ConstantInt>(CI)->getValue();
      if (Val.isPowerOf2()) {          // Replace X*(2^C) with X << C
        return BinaryOperator::CreateShl(Op0,
                 ConstantInt::get(Op0->getType(), Val.logBase2()));
      }
    } else if (ConstantFP *Op1F = dyn_cast<ConstantFP>(Op1)) {
      if (Op1F->isNullValue())
        return ReplaceInstUsesWith(I, Op1);

      // "In IEEE floating point, x*1 is not equivalent to x for nans.  However,
      // ANSI says we can drop signals, so we can do this anyway." (from GCC)
      if (Op1F->isExactlyValue(1.0))
        return ReplaceInstUsesWith(I, Op0);  // Eliminate 'mul double %X, 1.0'
    } else if (isa<VectorType>(Op1->getType())) {
      if (isa<ConstantAggregateZero>(Op1))
        return ReplaceInstUsesWith(I, Op1);

      if (ConstantVector *Op1V = dyn_cast<ConstantVector>(Op1)) {
        if (Op1V->isAllOnesValue())              // X * -1 == 0 - X
          return BinaryOperator::CreateNeg(Op0, I.getName());

        // As above, vector X*splat(1.0) -> X in all defined cases.
        if (Constant *Splat = Op1V->getSplatValue()) {
          if (ConstantFP *F = dyn_cast<ConstantFP>(Splat))
            if (F->isExactlyValue(1.0))
              return ReplaceInstUsesWith(I, Op0);
          if (ConstantInt *CI = dyn_cast<ConstantInt>(Splat))
            if (CI->equalsInt(1))
              return ReplaceInstUsesWith(I, Op0);
        }
      }
    }
    
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0))
      if (Op0I->getOpcode() == Instruction::Add && Op0I->hasOneUse() &&
          isa<ConstantInt>(Op0I->getOperand(1)) && isa<ConstantInt>(Op1)) {
        // Canonicalize (X+C1)*C2 -> X*C2+C1*C2.
        Instruction *Add = BinaryOperator::CreateMul(Op0I->getOperand(0),
                                                     Op1, "tmp");
        InsertNewInstBefore(Add, I);
        Value *C1C2 = ConstantExpr::getMul(Op1, 
                                           cast<Constant>(Op0I->getOperand(1)));
        return BinaryOperator::CreateAdd(Add, C1C2);
        
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
      return BinaryOperator::CreateMul(Op0v, Op1v);

  // (X / Y) *  Y = X - (X % Y)
  // (X / Y) * -Y = (X % Y) - X
  {
    Value *Op1 = I.getOperand(1);
    BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0);
    if (!BO ||
        (BO->getOpcode() != Instruction::UDiv && 
         BO->getOpcode() != Instruction::SDiv)) {
      Op1 = Op0;
      BO = dyn_cast<BinaryOperator>(I.getOperand(1));
    }
    Value *Neg = dyn_castNegVal(Op1);
    if (BO && BO->hasOneUse() &&
        (BO->getOperand(1) == Op1 || BO->getOperand(1) == Neg) &&
        (BO->getOpcode() == Instruction::UDiv ||
         BO->getOpcode() == Instruction::SDiv)) {
      Value *Op0BO = BO->getOperand(0), *Op1BO = BO->getOperand(1);

      Instruction *Rem;
      if (BO->getOpcode() == Instruction::UDiv)
        Rem = BinaryOperator::CreateURem(Op0BO, Op1BO);
      else
        Rem = BinaryOperator::CreateSRem(Op0BO, Op1BO);

      InsertNewInstBefore(Rem, I);
      Rem->takeName(BO);

      if (Op1BO == Op1)
        return BinaryOperator::CreateSub(Op0BO, Rem);
      else
        return BinaryOperator::CreateSub(Rem, Op0BO);
    }
  }

  if (I.getType() == Type::Int1Ty)
    return BinaryOperator::CreateAnd(Op0, I.getOperand(1));

  // If one of the operands of the multiply is a cast from a boolean value, then
  // we know the bool is either zero or one, so this is a 'masking' multiply.
  // See if we can simplify things based on how the boolean was originally
  // formed.
  CastInst *BoolCast = 0;
  if (ZExtInst *CI = dyn_cast<ZExtInst>(Op0))
    if (CI->getOperand(0)->getType() == Type::Int1Ty)
      BoolCast = CI;
  if (!BoolCast)
    if (ZExtInst *CI = dyn_cast<ZExtInst>(I.getOperand(1)))
      if (CI->getOperand(0)->getType() == Type::Int1Ty)
        BoolCast = CI;
  if (BoolCast) {
    if (ICmpInst *SCI = dyn_cast<ICmpInst>(BoolCast->getOperand(0))) {
      Value *SCIOp0 = SCI->getOperand(0), *SCIOp1 = SCI->getOperand(1);
      const Type *SCOpTy = SCIOp0->getType();
      bool TIS = false;
      
      // If the icmp is true iff the sign bit of X is set, then convert this
      // multiply into a shift/and combination.
      if (isa<ConstantInt>(SCIOp1) &&
          isSignBitCheck(SCI->getPredicate(), cast<ConstantInt>(SCIOp1), TIS) &&
          TIS) {
        // Shift the X value right to turn it into "all signbits".
        Constant *Amt = ConstantInt::get(SCIOp0->getType(),
                                          SCOpTy->getPrimitiveSizeInBits()-1);
        Value *V =
          InsertNewInstBefore(
            BinaryOperator::Create(Instruction::AShr, SCIOp0, Amt,
                                            BoolCast->getOperand(0)->getName()+
                                            ".mask"), I);

        // If the multiply type is not the same as the source type, sign extend
        // or truncate to the multiply type.
        if (I.getType() != V->getType()) {
          uint32_t SrcBits = V->getType()->getPrimitiveSizeInBits();
          uint32_t DstBits = I.getType()->getPrimitiveSizeInBits();
          Instruction::CastOps opcode = 
            (SrcBits == DstBits ? Instruction::BitCast : 
             (SrcBits < DstBits ? Instruction::SExt : Instruction::Trunc));
          V = InsertCastBefore(opcode, V, I.getType(), I);
        }

        Value *OtherOp = Op0 == BoolCast ? I.getOperand(1) : Op0;
        return BinaryOperator::CreateAnd(V, OtherOp);
      }
    }
  }

  return Changed ? &I : 0;
}

/// SimplifyDivRemOfSelect - Try to fold a divide or remainder of a select
/// instruction.
bool InstCombiner::SimplifyDivRemOfSelect(BinaryOperator &I) {
  SelectInst *SI = cast<SelectInst>(I.getOperand(1));
  
  // div/rem X, (Cond ? 0 : Y) -> div/rem X, Y
  int NonNullOperand = -1;
  if (Constant *ST = dyn_cast<Constant>(SI->getOperand(1)))
    if (ST->isNullValue())
      NonNullOperand = 2;
  // div/rem X, (Cond ? Y : 0) -> div/rem X, Y
  if (Constant *ST = dyn_cast<Constant>(SI->getOperand(2)))
    if (ST->isNullValue())
      NonNullOperand = 1;
  
  if (NonNullOperand == -1)
    return false;
  
  Value *SelectCond = SI->getOperand(0);
  
  // Change the div/rem to use 'Y' instead of the select.
  I.setOperand(1, SI->getOperand(NonNullOperand));
  
  // Okay, we know we replace the operand of the div/rem with 'Y' with no
  // problem.  However, the select, or the condition of the select may have
  // multiple uses.  Based on our knowledge that the operand must be non-zero,
  // propagate the known value for the select into other uses of it, and
  // propagate a known value of the condition into its other users.
  
  // If the select and condition only have a single use, don't bother with this,
  // early exit.
  if (SI->use_empty() && SelectCond->hasOneUse())
    return true;
  
  // Scan the current block backward, looking for other uses of SI.
  BasicBlock::iterator BBI = &I, BBFront = I.getParent()->begin();
  
  while (BBI != BBFront) {
    --BBI;
    // If we found a call to a function, we can't assume it will return, so
    // information from below it cannot be propagated above it.
    if (isa<CallInst>(BBI) && !isa<IntrinsicInst>(BBI))
      break;
    
    // Replace uses of the select or its condition with the known values.
    for (Instruction::op_iterator I = BBI->op_begin(), E = BBI->op_end();
         I != E; ++I) {
      if (*I == SI) {
        *I = SI->getOperand(NonNullOperand);
        AddToWorkList(BBI);
      } else if (*I == SelectCond) {
        *I = NonNullOperand == 1 ? ConstantInt::getTrue() :
                                   ConstantInt::getFalse();
        AddToWorkList(BBI);
      }
    }
    
    // If we past the instruction, quit looking for it.
    if (&*BBI == SI)
      SI = 0;
    if (&*BBI == SelectCond)
      SelectCond = 0;
    
    // If we ran out of things to eliminate, break out of the loop.
    if (SelectCond == 0 && SI == 0)
      break;
    
  }
  return true;
}


/// This function implements the transforms on div instructions that work
/// regardless of the kind of div instruction it is (udiv, sdiv, or fdiv). It is
/// used by the visitors to those instructions.
/// @brief Transforms common to all three div instructions
Instruction *InstCombiner::commonDivTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // undef / X -> 0        for integer.
  // undef / X -> undef    for FP (the undef could be a snan).
  if (isa<UndefValue>(Op0)) {
    if (Op0->getType()->isFPOrFPVector())
      return ReplaceInstUsesWith(I, Op0);
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }

  // X / undef -> undef
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);

  return 0;
}

/// This function implements the transforms common to both integer division
/// instructions (udiv and sdiv). It is called by the visitors to those integer
/// division instructions.
/// @brief Common integer divide transforms
Instruction *InstCombiner::commonIDivTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // (sdiv X, X) --> 1     (udiv X, X) --> 1
  if (Op0 == Op1) {
    if (const VectorType *Ty = dyn_cast<VectorType>(I.getType())) {
      ConstantInt *CI = ConstantInt::get(Ty->getElementType(), 1);
      std::vector<Constant*> Elts(Ty->getNumElements(), CI);
      return ReplaceInstUsesWith(I, ConstantVector::get(Elts));
    }

    ConstantInt *CI = ConstantInt::get(I.getType(), 1);
    return ReplaceInstUsesWith(I, CI);
  }
  
  if (Instruction *Common = commonDivTransforms(I))
    return Common;
  
  // Handle cases involving: [su]div X, (select Cond, Y, Z)
  // This does not apply for fdiv.
  if (isa<SelectInst>(Op1) && SimplifyDivRemOfSelect(I))
    return &I;

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // div X, 1 == X
    if (RHS->equalsInt(1))
      return ReplaceInstUsesWith(I, Op0);

    // (X / C1) / C2  -> X / (C1*C2)
    if (Instruction *LHS = dyn_cast<Instruction>(Op0))
      if (Instruction::BinaryOps(LHS->getOpcode()) == I.getOpcode())
        if (ConstantInt *LHSRHS = dyn_cast<ConstantInt>(LHS->getOperand(1))) {
          if (MultiplyOverflows(RHS, LHSRHS, I.getOpcode()==Instruction::SDiv))
            return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
          else 
            return BinaryOperator::Create(I.getOpcode(), LHS->getOperand(0),
                                          Multiply(RHS, LHSRHS));
        }

    if (!RHS->isZero()) { // avoid X udiv 0
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

  // It can't be division by zero, hence it must be division by one.
  if (I.getType() == Type::Int1Ty)
    return ReplaceInstUsesWith(I, Op0);

  if (ConstantVector *Op1V = dyn_cast<ConstantVector>(Op1)) {
    if (ConstantInt *X = cast_or_null<ConstantInt>(Op1V->getSplatValue()))
      // div X, 1 == X
      if (X->isOne())
        return ReplaceInstUsesWith(I, Op0);
  }

  return 0;
}

Instruction *InstCombiner::visitUDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  if (ConstantInt *C = dyn_cast<ConstantInt>(Op1)) {
    // X udiv C^2 -> X >> C
    // Check to see if this is an unsigned division with an exact power of 2,
    // if so, convert to a right shift.
    if (C->getValue().isPowerOf2())  // 0 not included in isPowerOf2
      return BinaryOperator::CreateLShr(Op0, 
               ConstantInt::get(Op0->getType(), C->getValue().logBase2()));

    // X udiv C, where C >= signbit
    if (C->getValue().isNegative()) {
      Value *IC = InsertNewInstBefore(new ICmpInst(ICmpInst::ICMP_ULT, Op0, C),
                                      I);
      return SelectInst::Create(IC, Constant::getNullValue(I.getType()),
                                ConstantInt::get(I.getType(), 1));
    }
  }

  // X udiv (C1 << N), where C1 is "1<<C2"  -->  X >> (N+C2)
  if (BinaryOperator *RHSI = dyn_cast<BinaryOperator>(I.getOperand(1))) {
    if (RHSI->getOpcode() == Instruction::Shl &&
        isa<ConstantInt>(RHSI->getOperand(0))) {
      const APInt& C1 = cast<ConstantInt>(RHSI->getOperand(0))->getValue();
      if (C1.isPowerOf2()) {
        Value *N = RHSI->getOperand(1);
        const Type *NTy = N->getType();
        if (uint32_t C2 = C1.logBase2()) {
          Constant *C2V = ConstantInt::get(NTy, C2);
          N = InsertNewInstBefore(BinaryOperator::CreateAdd(N, C2V, "tmp"), I);
        }
        return BinaryOperator::CreateLShr(Op0, N);
      }
    }
  }
  
  // udiv X, (Select Cond, C1, C2) --> Select Cond, (shr X, C1), (shr X, C2)
  // where C1&C2 are powers of two.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op1)) 
    if (ConstantInt *STO = dyn_cast<ConstantInt>(SI->getOperand(1)))
      if (ConstantInt *SFO = dyn_cast<ConstantInt>(SI->getOperand(2)))  {
        const APInt &TVA = STO->getValue(), &FVA = SFO->getValue();
        if (TVA.isPowerOf2() && FVA.isPowerOf2()) {
          // Compute the shift amounts
          uint32_t TSA = TVA.logBase2(), FSA = FVA.logBase2();
          // Construct the "on true" case of the select
          Constant *TC = ConstantInt::get(Op0->getType(), TSA);
          Instruction *TSI = BinaryOperator::CreateLShr(
                                                 Op0, TC, SI->getName()+".t");
          TSI = InsertNewInstBefore(TSI, I);
  
          // Construct the "on false" case of the select
          Constant *FC = ConstantInt::get(Op0->getType(), FSA); 
          Instruction *FSI = BinaryOperator::CreateLShr(
                                                 Op0, FC, SI->getName()+".f");
          FSI = InsertNewInstBefore(FSI, I);

          // construct the select instruction and return it.
          return SelectInst::Create(SI->getOperand(0), TSI, FSI, SI->getName());
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
      return BinaryOperator::CreateNeg(Op0);
  }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a udiv.
  if (I.getType()->isInteger()) {
    APInt Mask(APInt::getSignBit(I.getType()->getPrimitiveSizeInBits()));
    if (MaskedValueIsZero(Op1, Mask) && MaskedValueIsZero(Op0, Mask)) {
      // X sdiv Y -> X udiv Y, iff X and Y don't have sign bit set
      return BinaryOperator::CreateUDiv(Op0, Op1, I.getName());
    }
  }      
  
  return 0;
}

Instruction *InstCombiner::visitFDiv(BinaryOperator &I) {
  return commonDivTransforms(I);
}

/// This function implements the transforms on rem instructions that work
/// regardless of the kind of rem instruction it is (urem, srem, or frem). It 
/// is used by the visitors to those instructions.
/// @brief Transforms common to all three rem instructions
Instruction *InstCombiner::commonRemTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op0)) {             // undef % X -> 0
    if (I.getType()->isFPOrFPVector())
      return ReplaceInstUsesWith(I, Op0);  // X % undef -> undef (could be SNaN)
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);  // X % undef -> undef

  // Handle cases involving: rem X, (select Cond, Y, Z)
  if (isa<SelectInst>(Op1) && SimplifyDivRemOfSelect(I))
    return &I;

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

  // 0 % X == 0 for integer, we don't need to preserve faults!
  if (Constant *LHS = dyn_cast<Constant>(Op0))
    if (LHS->isNullValue())
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

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

      // See if we can fold away this rem instruction.
      if (SimplifyDemandedInstructionBits(I))
        return &I;
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
      if (C->getValue().isPowerOf2())
        return BinaryOperator::CreateAnd(Op0, SubOne(C));
  }

  if (Instruction *RHSI = dyn_cast<Instruction>(I.getOperand(1))) {
    // Turn A % (C << N), where C is 2^k, into A & ((C << N)-1)  
    if (RHSI->getOpcode() == Instruction::Shl &&
        isa<ConstantInt>(RHSI->getOperand(0))) {
      if (cast<ConstantInt>(RHSI->getOperand(0))->getValue().isPowerOf2()) {
        Constant *N1 = ConstantInt::getAllOnesValue(I.getType());
        Value *Add = InsertNewInstBefore(BinaryOperator::CreateAdd(RHSI, N1,
                                                                   "tmp"), I);
        return BinaryOperator::CreateAnd(Op0, Add);
      }
    }
  }

  // urem X, (select Cond, 2^C1, 2^C2) --> select Cond, (and X, C1), (and X, C2)
  // where C1&C2 are powers of two.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op1)) {
    if (ConstantInt *STO = dyn_cast<ConstantInt>(SI->getOperand(1)))
      if (ConstantInt *SFO = dyn_cast<ConstantInt>(SI->getOperand(2))) {
        // STO == 0 and SFO == 0 handled above.
        if ((STO->getValue().isPowerOf2()) && 
            (SFO->getValue().isPowerOf2())) {
          Value *TrueAnd = InsertNewInstBefore(
            BinaryOperator::CreateAnd(Op0, SubOne(STO), SI->getName()+".t"), I);
          Value *FalseAnd = InsertNewInstBefore(
            BinaryOperator::CreateAnd(Op0, SubOne(SFO), SI->getName()+".f"), I);
          return SelectInst::Create(SI->getOperand(0), TrueAnd, FalseAnd);
        }
      }
  }
  
  return 0;
}

Instruction *InstCombiner::visitSRem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Handle the integer rem common cases
  if (Instruction *common = commonIRemTransforms(I))
    return common;
  
  if (Value *RHSNeg = dyn_castNegVal(Op1))
    if (!isa<Constant>(RHSNeg) ||
        (isa<ConstantInt>(RHSNeg) &&
         cast<ConstantInt>(RHSNeg)->getValue().isStrictlyPositive())) {
      // X % -Y -> X % Y
      AddUsesToWorkList(I);
      I.setOperand(1, RHSNeg);
      return &I;
    }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a urem.
  if (I.getType()->isInteger()) {
    APInt Mask(APInt::getSignBit(I.getType()->getPrimitiveSizeInBits()));
    if (MaskedValueIsZero(Op1, Mask) && MaskedValueIsZero(Op0, Mask)) {
      // X srem Y -> X urem Y, iff X and Y don't have sign bit set
      return BinaryOperator::CreateURem(Op0, Op1, I.getName());
    }
  }

  // If it's a constant vector, flip any negative values positive.
  if (ConstantVector *RHSV = dyn_cast<ConstantVector>(Op1)) {
    unsigned VWidth = RHSV->getNumOperands();

    bool hasNegative = false;
    for (unsigned i = 0; !hasNegative && i != VWidth; ++i)
      if (ConstantInt *RHS = dyn_cast<ConstantInt>(RHSV->getOperand(i)))
        if (RHS->getValue().isNegative())
          hasNegative = true;

    if (hasNegative) {
      std::vector<Constant *> Elts(VWidth);
      for (unsigned i = 0; i != VWidth; ++i) {
        if (ConstantInt *RHS = dyn_cast<ConstantInt>(RHSV->getOperand(i))) {
          if (RHS->getValue().isNegative())
            Elts[i] = cast<ConstantInt>(ConstantExpr::getNeg(RHS));
          else
            Elts[i] = RHS;
        }
      }

      Constant *NewRHSV = ConstantVector::get(Elts);
      if (NewRHSV != RHSV) {
        AddUsesToWorkList(I);
        I.setOperand(1, NewRHSV);
        return &I;
      }
    }
  }

  return 0;
}

Instruction *InstCombiner::visitFRem(BinaryOperator &I) {
  return commonRemTransforms(I);
}

// isOneBitSet - Return true if there is exactly one bit set in the specified
// constant.
static bool isOneBitSet(const ConstantInt *CI) {
  return CI->getValue().isPowerOf2();
}

// isHighOnes - Return true if the constant is of the form 1+0+.
// This is the same as lowones(~X).
static bool isHighOnes(const ConstantInt *CI) {
  return (~CI->getValue() + 1).isPowerOf2();
}

/// getICmpCode - Encode a icmp predicate into a three bit mask.  These bits
/// are carefully arranged to allow folding of expressions such as:
///
///      (A < B) | (A > B) --> (A != B)
///
/// Note that this is only valid if the first and second predicates have the
/// same sign. Is illegal to do: (A u< B) | (A s> B) 
///
/// Three bits are used to represent the condition, as follows:
///   0  A > B
///   1  A == B
///   2  A < B
///
/// <=>  Value  Definition
/// 000     0   Always false
/// 001     1   A >  B
/// 010     2   A == B
/// 011     3   A >= B
/// 100     4   A <  B
/// 101     5   A != B
/// 110     6   A <= B
/// 111     7   Always true
///  
static unsigned getICmpCode(const ICmpInst *ICI) {
  switch (ICI->getPredicate()) {
    // False -> 0
  case ICmpInst::ICMP_UGT: return 1;  // 001
  case ICmpInst::ICMP_SGT: return 1;  // 001
  case ICmpInst::ICMP_EQ:  return 2;  // 010
  case ICmpInst::ICMP_UGE: return 3;  // 011
  case ICmpInst::ICMP_SGE: return 3;  // 011
  case ICmpInst::ICMP_ULT: return 4;  // 100
  case ICmpInst::ICMP_SLT: return 4;  // 100
  case ICmpInst::ICMP_NE:  return 5;  // 101
  case ICmpInst::ICMP_ULE: return 6;  // 110
  case ICmpInst::ICMP_SLE: return 6;  // 110
    // True -> 7
  default:
    assert(0 && "Invalid ICmp predicate!");
    return 0;
  }
}

/// getFCmpCode - Similar to getICmpCode but for FCmpInst. This encodes a fcmp
/// predicate into a three bit mask. It also returns whether it is an ordered
/// predicate by reference.
static unsigned getFCmpCode(FCmpInst::Predicate CC, bool &isOrdered) {
  isOrdered = false;
  switch (CC) {
  case FCmpInst::FCMP_ORD: isOrdered = true; return 0;  // 000
  case FCmpInst::FCMP_UNO:                   return 0;  // 000
  case FCmpInst::FCMP_OGT: isOrdered = true; return 1;  // 001
  case FCmpInst::FCMP_UGT:                   return 1;  // 001
  case FCmpInst::FCMP_OEQ: isOrdered = true; return 2;  // 010
  case FCmpInst::FCMP_UEQ:                   return 2;  // 010
  case FCmpInst::FCMP_OGE: isOrdered = true; return 3;  // 011
  case FCmpInst::FCMP_UGE:                   return 3;  // 011
  case FCmpInst::FCMP_OLT: isOrdered = true; return 4;  // 100
  case FCmpInst::FCMP_ULT:                   return 4;  // 100
  case FCmpInst::FCMP_ONE: isOrdered = true; return 5;  // 101
  case FCmpInst::FCMP_UNE:                   return 5;  // 101
  case FCmpInst::FCMP_OLE: isOrdered = true; return 6;  // 110
  case FCmpInst::FCMP_ULE:                   return 6;  // 110
    // True -> 7
  default:
    // Not expecting FCMP_FALSE and FCMP_TRUE;
    assert(0 && "Unexpected FCmp predicate!");
    return 0;
  }
}

/// getICmpValue - This is the complement of getICmpCode, which turns an
/// opcode and two operands into either a constant true or false, or a brand 
/// new ICmp instruction. The sign is passed in to determine which kind
/// of predicate to use in the new icmp instruction.
static Value *getICmpValue(bool sign, unsigned code, Value *LHS, Value *RHS) {
  switch (code) {
  default: assert(0 && "Illegal ICmp code!");
  case  0: return ConstantInt::getFalse();
  case  1: 
    if (sign)
      return new ICmpInst(ICmpInst::ICMP_SGT, LHS, RHS);
    else
      return new ICmpInst(ICmpInst::ICMP_UGT, LHS, RHS);
  case  2: return new ICmpInst(ICmpInst::ICMP_EQ,  LHS, RHS);
  case  3: 
    if (sign)
      return new ICmpInst(ICmpInst::ICMP_SGE, LHS, RHS);
    else
      return new ICmpInst(ICmpInst::ICMP_UGE, LHS, RHS);
  case  4: 
    if (sign)
      return new ICmpInst(ICmpInst::ICMP_SLT, LHS, RHS);
    else
      return new ICmpInst(ICmpInst::ICMP_ULT, LHS, RHS);
  case  5: return new ICmpInst(ICmpInst::ICMP_NE,  LHS, RHS);
  case  6: 
    if (sign)
      return new ICmpInst(ICmpInst::ICMP_SLE, LHS, RHS);
    else
      return new ICmpInst(ICmpInst::ICMP_ULE, LHS, RHS);
  case  7: return ConstantInt::getTrue();
  }
}

/// getFCmpValue - This is the complement of getFCmpCode, which turns an
/// opcode and two operands into either a FCmp instruction. isordered is passed
/// in to determine which kind of predicate to use in the new fcmp instruction.
static Value *getFCmpValue(bool isordered, unsigned code,
                           Value *LHS, Value *RHS) {
  switch (code) {
  default: assert(0 && "Illegal FCmp code!");
  case  0:
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_ORD, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_UNO, LHS, RHS);
  case  1: 
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_OGT, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_UGT, LHS, RHS);
  case  2: 
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_OEQ, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_UEQ, LHS, RHS);
  case  3: 
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_OGE, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_UGE, LHS, RHS);
  case  4: 
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_OLT, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_ULT, LHS, RHS);
  case  5: 
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_ONE, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_UNE, LHS, RHS);
  case  6: 
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_OLE, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_ULE, LHS, RHS);
  case  7: return ConstantInt::getTrue();
  }
}

/// PredicatesFoldable - Return true if both predicates match sign or if at
/// least one of them is an equality comparison (which is signless).
static bool PredicatesFoldable(ICmpInst::Predicate p1, ICmpInst::Predicate p2) {
  return (ICmpInst::isSignedPredicate(p1) == ICmpInst::isSignedPredicate(p2)) ||
         (ICmpInst::isSignedPredicate(p1) && ICmpInst::isEquality(p2)) ||
         (ICmpInst::isSignedPredicate(p2) && ICmpInst::isEquality(p1));
}

namespace { 
// FoldICmpLogical - Implements (icmp1 A, B) & (icmp2 A, B) --> (icmp3 A, B)
struct FoldICmpLogical {
  InstCombiner &IC;
  Value *LHS, *RHS;
  ICmpInst::Predicate pred;
  FoldICmpLogical(InstCombiner &ic, ICmpInst *ICI)
    : IC(ic), LHS(ICI->getOperand(0)), RHS(ICI->getOperand(1)),
      pred(ICI->getPredicate()) {}
  bool shouldApply(Value *V) const {
    if (ICmpInst *ICI = dyn_cast<ICmpInst>(V))
      if (PredicatesFoldable(pred, ICI->getPredicate()))
        return ((ICI->getOperand(0) == LHS && ICI->getOperand(1) == RHS) ||
                (ICI->getOperand(0) == RHS && ICI->getOperand(1) == LHS));
    return false;
  }
  Instruction *apply(Instruction &Log) const {
    ICmpInst *ICI = cast<ICmpInst>(Log.getOperand(0));
    if (ICI->getOperand(0) != LHS) {
      assert(ICI->getOperand(1) == LHS);
      ICI->swapOperands();  // Swap the LHS and RHS of the ICmp
    }

    ICmpInst *RHSICI = cast<ICmpInst>(Log.getOperand(1));
    unsigned LHSCode = getICmpCode(ICI);
    unsigned RHSCode = getICmpCode(RHSICI);
    unsigned Code;
    switch (Log.getOpcode()) {
    case Instruction::And: Code = LHSCode & RHSCode; break;
    case Instruction::Or:  Code = LHSCode | RHSCode; break;
    case Instruction::Xor: Code = LHSCode ^ RHSCode; break;
    default: assert(0 && "Illegal logical opcode!"); return 0;
    }

    bool isSigned = ICmpInst::isSignedPredicate(RHSICI->getPredicate()) || 
                    ICmpInst::isSignedPredicate(ICI->getPredicate());
      
    Value *RV = getICmpValue(isSigned, Code, LHS, RHS);
    if (Instruction *I = dyn_cast<Instruction>(RV))
      return I;
    // Otherwise, it's a constant boolean value...
    return IC.ReplaceInstUsesWith(Log, RV);
  }
};
} // end anonymous namespace

// OptAndOp - This handles expressions of the form ((val OP C1) & C2).  Where
// the Op parameter is 'OP', OpRHS is 'C1', and AndRHS is 'C2'.  Op is
// guaranteed to be a binary operator.
Instruction *InstCombiner::OptAndOp(Instruction *Op,
                                    ConstantInt *OpRHS,
                                    ConstantInt *AndRHS,
                                    BinaryOperator &TheAnd) {
  Value *X = Op->getOperand(0);
  Constant *Together = 0;
  if (!Op->isShift())
    Together = And(AndRHS, OpRHS);

  switch (Op->getOpcode()) {
  case Instruction::Xor:
    if (Op->hasOneUse()) {
      // (X ^ C1) & C2 --> (X & C2) ^ (C1&C2)
      Instruction *And = BinaryOperator::CreateAnd(X, AndRHS);
      InsertNewInstBefore(And, TheAnd);
      And->takeName(Op);
      return BinaryOperator::CreateXor(And, Together);
    }
    break;
  case Instruction::Or:
    if (Together == AndRHS) // (X | C) & C --> C
      return ReplaceInstUsesWith(TheAnd, AndRHS);

    if (Op->hasOneUse() && Together != OpRHS) {
      // (X | C1) & C2 --> (X | (C1&C2)) & C2
      Instruction *Or = BinaryOperator::CreateOr(X, Together);
      InsertNewInstBefore(Or, TheAnd);
      Or->takeName(Op);
      return BinaryOperator::CreateAnd(Or, AndRHS);
    }
    break;
  case Instruction::Add:
    if (Op->hasOneUse()) {
      // Adding a one to a single bit bit-field should be turned into an XOR
      // of the bit.  First thing to check is to see if this AND is with a
      // single bit constant.
      const APInt& AndRHSV = cast<ConstantInt>(AndRHS)->getValue();

      // If there is only one bit set...
      if (isOneBitSet(cast<ConstantInt>(AndRHS))) {
        // Ok, at this point, we know that we are masking the result of the
        // ADD down to exactly one bit.  If the constant we are adding has
        // no bits set below this bit, then we can eliminate the ADD.
        const APInt& AddRHS = cast<ConstantInt>(OpRHS)->getValue();

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
            // Pull the XOR out of the AND.
            Instruction *NewAnd = BinaryOperator::CreateAnd(X, AndRHS);
            InsertNewInstBefore(NewAnd, TheAnd);
            NewAnd->takeName(Op);
            return BinaryOperator::CreateXor(NewAnd, AndRHS);
          }
        }
      }
    }
    break;

  case Instruction::Shl: {
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!
    //
    uint32_t BitWidth = AndRHS->getType()->getBitWidth();
    uint32_t OpRHSVal = OpRHS->getLimitedValue(BitWidth);
    APInt ShlMask(APInt::getHighBitsSet(BitWidth, BitWidth-OpRHSVal));
    ConstantInt *CI = ConstantInt::get(AndRHS->getValue() & ShlMask);

    if (CI->getValue() == ShlMask) { 
    // Masking out bits that the shift already masks
      return ReplaceInstUsesWith(TheAnd, Op);   // No need for the and.
    } else if (CI != AndRHS) {                  // Reducing bits set in and.
      TheAnd.setOperand(1, CI);
      return &TheAnd;
    }
    break;
  }
  case Instruction::LShr:
  {
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!  This only applies to
    // unsigned shifts, because a signed shr may bring in set bits!
    //
    uint32_t BitWidth = AndRHS->getType()->getBitWidth();
    uint32_t OpRHSVal = OpRHS->getLimitedValue(BitWidth);
    APInt ShrMask(APInt::getLowBitsSet(BitWidth, BitWidth - OpRHSVal));
    ConstantInt *CI = ConstantInt::get(AndRHS->getValue() & ShrMask);

    if (CI->getValue() == ShrMask) {   
    // Masking out bits that the shift already masks.
      return ReplaceInstUsesWith(TheAnd, Op);
    } else if (CI != AndRHS) {
      TheAnd.setOperand(1, CI);  // Reduce bits set in and cst.
      return &TheAnd;
    }
    break;
  }
  case Instruction::AShr:
    // Signed shr.
    // See if this is shifting in some sign extension, then masking it out
    // with an and.
    if (Op->hasOneUse()) {
      uint32_t BitWidth = AndRHS->getType()->getBitWidth();
      uint32_t OpRHSVal = OpRHS->getLimitedValue(BitWidth);
      APInt ShrMask(APInt::getLowBitsSet(BitWidth, BitWidth - OpRHSVal));
      Constant *C = ConstantInt::get(AndRHS->getValue() & ShrMask);
      if (C == AndRHS) {          // Masking out bits shifted in.
        // (Val ashr C1) & C2 -> (Val lshr C1) & C2
        // Make the argument unsigned.
        Value *ShVal = Op->getOperand(0);
        ShVal = InsertNewInstBefore(
            BinaryOperator::CreateLShr(ShVal, OpRHS, 
                                   Op->getName()), TheAnd);
        return BinaryOperator::CreateAnd(ShVal, AndRHS, TheAnd.getName());
      }
    }
    break;
  }
  return 0;
}


/// InsertRangeTest - Emit a computation of: (V >= Lo && V < Hi) if Inside is
/// true, otherwise (V < Lo || V >= Hi).  In pratice, we emit the more efficient
/// (V-Lo) <u Hi-Lo.  This method expects that Lo <= Hi. isSigned indicates
/// whether to treat the V, Lo and HI as signed or not. IB is the location to
/// insert new instructions.
Instruction *InstCombiner::InsertRangeTest(Value *V, Constant *Lo, Constant *Hi,
                                           bool isSigned, bool Inside, 
                                           Instruction &IB) {
  assert(cast<ConstantInt>(ConstantExpr::getICmp((isSigned ? 
            ICmpInst::ICMP_SLE:ICmpInst::ICMP_ULE), Lo, Hi))->getZExtValue() &&
         "Lo is not <= Hi in range emission code!");
    
  if (Inside) {
    if (Lo == Hi)  // Trivially false.
      return new ICmpInst(ICmpInst::ICMP_NE, V, V);

    // V >= Min && V < Hi --> V < Hi
    if (cast<ConstantInt>(Lo)->isMinValue(isSigned)) {
      ICmpInst::Predicate pred = (isSigned ? 
        ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT);
      return new ICmpInst(pred, V, Hi);
    }

    // Emit V-Lo <u Hi-Lo
    Constant *NegLo = ConstantExpr::getNeg(Lo);
    Instruction *Add = BinaryOperator::CreateAdd(V, NegLo, V->getName()+".off");
    InsertNewInstBefore(Add, IB);
    Constant *UpperBound = ConstantExpr::getAdd(NegLo, Hi);
    return new ICmpInst(ICmpInst::ICMP_ULT, Add, UpperBound);
  }

  if (Lo == Hi)  // Trivially true.
    return new ICmpInst(ICmpInst::ICMP_EQ, V, V);

  // V < Min || V >= Hi -> V > Hi-1
  Hi = SubOne(cast<ConstantInt>(Hi));
  if (cast<ConstantInt>(Lo)->isMinValue(isSigned)) {
    ICmpInst::Predicate pred = (isSigned ? 
        ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT);
    return new ICmpInst(pred, V, Hi);
  }

  // Emit V-Lo >u Hi-1-Lo
  // Note that Hi has already had one subtracted from it, above.
  ConstantInt *NegLo = cast<ConstantInt>(ConstantExpr::getNeg(Lo));
  Instruction *Add = BinaryOperator::CreateAdd(V, NegLo, V->getName()+".off");
  InsertNewInstBefore(Add, IB);
  Constant *LowerBound = ConstantExpr::getAdd(NegLo, Hi);
  return new ICmpInst(ICmpInst::ICMP_UGT, Add, LowerBound);
}

// isRunOfOnes - Returns true iff Val consists of one contiguous run of 1s with
// any number of 0s on either side.  The 1s are allowed to wrap from LSB to
// MSB, so 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.  0x0F0F0000 is
// not, since all 1s are not contiguous.
static bool isRunOfOnes(ConstantInt *Val, uint32_t &MB, uint32_t &ME) {
  const APInt& V = Val->getValue();
  uint32_t BitWidth = Val->getType()->getBitWidth();
  if (!APIntOps::isShiftedMask(BitWidth, V)) return false;

  // look for the first zero bit after the run of ones
  MB = BitWidth - ((V - 1) ^ V).countLeadingZeros();
  // look for the first non-zero bit
  ME = V.getActiveBits(); 
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
                                        ConstantInt *Mask, bool isSub,
                                        Instruction &I) {
  Instruction *LHSI = dyn_cast<Instruction>(LHS);
  if (!LHSI || LHSI->getNumOperands() != 2 ||
      !isa<ConstantInt>(LHSI->getOperand(1))) return 0;

  ConstantInt *N = cast<ConstantInt>(LHSI->getOperand(1));

  switch (LHSI->getOpcode()) {
  default: return 0;
  case Instruction::And:
    if (And(N, Mask) == Mask) {
      // If the AndRHS is a power of two minus one (0+1+), this is simple.
      if ((Mask->getValue().countLeadingZeros() + 
           Mask->getValue().countPopulation()) == 
          Mask->getValue().getBitWidth())
        break;

      // Otherwise, if Mask is 0+1+0+, and if B is known to have the low 0+
      // part, we don't need any explicit masks to take them out of A.  If that
      // is all N is, ignore it.
      uint32_t MB = 0, ME = 0;
      if (isRunOfOnes(Mask, MB, ME)) {  // begin/end bit of run, inclusive
        uint32_t BitWidth = cast<IntegerType>(RHS->getType())->getBitWidth();
        APInt Mask(APInt::getLowBitsSet(BitWidth, MB-1));
        if (MaskedValueIsZero(RHS, Mask))
          break;
      }
    }
    return 0;
  case Instruction::Or:
  case Instruction::Xor:
    // If the AndRHS is a power of two minus one (0+1+), and N&Mask == 0
    if ((Mask->getValue().countLeadingZeros() + 
         Mask->getValue().countPopulation()) == Mask->getValue().getBitWidth()
        && And(N, Mask)->isZero())
      break;
    return 0;
  }
  
  Instruction *New;
  if (isSub)
    New = BinaryOperator::CreateSub(LHSI->getOperand(0), RHS, "fold");
  else
    New = BinaryOperator::CreateAdd(LHSI->getOperand(0), RHS, "fold");
  return InsertNewInstBefore(New, I);
}

/// FoldAndOfICmps - Fold (icmp)&(icmp) if possible.
Instruction *InstCombiner::FoldAndOfICmps(Instruction &I,
                                          ICmpInst *LHS, ICmpInst *RHS) {
  Value *Val, *Val2;
  ConstantInt *LHSCst, *RHSCst;
  ICmpInst::Predicate LHSCC, RHSCC;
  
  // This only handles icmp of constants: (icmp1 A, C1) & (icmp2 B, C2).
  if (!match(LHS, m_ICmp(LHSCC, m_Value(Val), m_ConstantInt(LHSCst))) ||
      !match(RHS, m_ICmp(RHSCC, m_Value(Val2), m_ConstantInt(RHSCst))))
    return 0;
  
  // (icmp ult A, C) & (icmp ult B, C) --> (icmp ult (A|B), C)
  // where C is a power of 2
  if (LHSCst == RHSCst && LHSCC == RHSCC && LHSCC == ICmpInst::ICMP_ULT &&
      LHSCst->getValue().isPowerOf2()) {
    Instruction *NewOr = BinaryOperator::CreateOr(Val, Val2);
    InsertNewInstBefore(NewOr, I);
    return new ICmpInst(LHSCC, NewOr, LHSCst);
  }
  
  // From here on, we only handle:
  //    (icmp1 A, C1) & (icmp2 A, C2) --> something simpler.
  if (Val != Val2) return 0;
  
  // ICMP_[US][GL]E X, CST is folded to ICMP_[US][GL]T elsewhere.
  if (LHSCC == ICmpInst::ICMP_UGE || LHSCC == ICmpInst::ICMP_ULE ||
      RHSCC == ICmpInst::ICMP_UGE || RHSCC == ICmpInst::ICMP_ULE ||
      LHSCC == ICmpInst::ICMP_SGE || LHSCC == ICmpInst::ICMP_SLE ||
      RHSCC == ICmpInst::ICMP_SGE || RHSCC == ICmpInst::ICMP_SLE)
    return 0;
  
  // We can't fold (ugt x, C) & (sgt x, C2).
  if (!PredicatesFoldable(LHSCC, RHSCC))
    return 0;
    
  // Ensure that the larger constant is on the RHS.
  bool ShouldSwap;
  if (ICmpInst::isSignedPredicate(LHSCC) ||
      (ICmpInst::isEquality(LHSCC) && 
       ICmpInst::isSignedPredicate(RHSCC)))
    ShouldSwap = LHSCst->getValue().sgt(RHSCst->getValue());
  else
    ShouldSwap = LHSCst->getValue().ugt(RHSCst->getValue());
    
  if (ShouldSwap) {
    std::swap(LHS, RHS);
    std::swap(LHSCst, RHSCst);
    std::swap(LHSCC, RHSCC);
  }

  // At this point, we know we have have two icmp instructions
  // comparing a value against two constants and and'ing the result
  // together.  Because of the above check, we know that we only have
  // icmp eq, icmp ne, icmp [su]lt, and icmp [SU]gt here. We also know 
  // (from the FoldICmpLogical check above), that the two constants 
  // are not equal and that the larger constant is on the RHS
  assert(LHSCst != RHSCst && "Compares not folded above?");

  switch (LHSCC) {
  default: assert(0 && "Unknown integer condition code!");
  case ICmpInst::ICMP_EQ:
    switch (RHSCC) {
    default: assert(0 && "Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X == 13 & X == 15) -> false
    case ICmpInst::ICMP_UGT:        // (X == 13 & X >  15) -> false
    case ICmpInst::ICMP_SGT:        // (X == 13 & X >  15) -> false
      return ReplaceInstUsesWith(I, ConstantInt::getFalse());
    case ICmpInst::ICMP_NE:         // (X == 13 & X != 15) -> X == 13
    case ICmpInst::ICMP_ULT:        // (X == 13 & X <  15) -> X == 13
    case ICmpInst::ICMP_SLT:        // (X == 13 & X <  15) -> X == 13
      return ReplaceInstUsesWith(I, LHS);
    }
  case ICmpInst::ICMP_NE:
    switch (RHSCC) {
    default: assert(0 && "Unknown integer condition code!");
    case ICmpInst::ICMP_ULT:
      if (LHSCst == SubOne(RHSCst)) // (X != 13 & X u< 14) -> X < 13
        return new ICmpInst(ICmpInst::ICMP_ULT, Val, LHSCst);
      break;                        // (X != 13 & X u< 15) -> no change
    case ICmpInst::ICMP_SLT:
      if (LHSCst == SubOne(RHSCst)) // (X != 13 & X s< 14) -> X < 13
        return new ICmpInst(ICmpInst::ICMP_SLT, Val, LHSCst);
      break;                        // (X != 13 & X s< 15) -> no change
    case ICmpInst::ICMP_EQ:         // (X != 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_UGT:        // (X != 13 & X u> 15) -> X u> 15
    case ICmpInst::ICMP_SGT:        // (X != 13 & X s> 15) -> X s> 15
      return ReplaceInstUsesWith(I, RHS);
    case ICmpInst::ICMP_NE:
      if (LHSCst == SubOne(RHSCst)){// (X != 13 & X != 14) -> X-13 >u 1
        Constant *AddCST = ConstantExpr::getNeg(LHSCst);
        Instruction *Add = BinaryOperator::CreateAdd(Val, AddCST,
                                                     Val->getName()+".off");
        InsertNewInstBefore(Add, I);
        return new ICmpInst(ICmpInst::ICMP_UGT, Add,
                            ConstantInt::get(Add->getType(), 1));
      }
      break;                        // (X != 13 & X != 15) -> no change
    }
    break;
  case ICmpInst::ICMP_ULT:
    switch (RHSCC) {
    default: assert(0 && "Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u< 13 & X == 15) -> false
    case ICmpInst::ICMP_UGT:        // (X u< 13 & X u> 15) -> false
      return ReplaceInstUsesWith(I, ConstantInt::getFalse());
    case ICmpInst::ICMP_SGT:        // (X u< 13 & X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X u< 13 & X != 15) -> X u< 13
    case ICmpInst::ICMP_ULT:        // (X u< 13 & X u< 15) -> X u< 13
      return ReplaceInstUsesWith(I, LHS);
    case ICmpInst::ICMP_SLT:        // (X u< 13 & X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SLT:
    switch (RHSCC) {
    default: assert(0 && "Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s< 13 & X == 15) -> false
    case ICmpInst::ICMP_SGT:        // (X s< 13 & X s> 15) -> false
      return ReplaceInstUsesWith(I, ConstantInt::getFalse());
    case ICmpInst::ICMP_UGT:        // (X s< 13 & X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X s< 13 & X != 15) -> X < 13
    case ICmpInst::ICMP_SLT:        // (X s< 13 & X s< 15) -> X < 13
      return ReplaceInstUsesWith(I, LHS);
    case ICmpInst::ICMP_ULT:        // (X s< 13 & X u< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_UGT:
    switch (RHSCC) {
    default: assert(0 && "Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u> 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_UGT:        // (X u> 13 & X u> 15) -> X u> 15
      return ReplaceInstUsesWith(I, RHS);
    case ICmpInst::ICMP_SGT:        // (X u> 13 & X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:
      if (RHSCst == AddOne(LHSCst)) // (X u> 13 & X != 14) -> X u> 14
        return new ICmpInst(LHSCC, Val, RHSCst);
      break;                        // (X u> 13 & X != 15) -> no change
    case ICmpInst::ICMP_ULT:        // (X u> 13 & X u< 15) -> (X-14) <u 1
      return InsertRangeTest(Val, AddOne(LHSCst), RHSCst, false, true, I);
    case ICmpInst::ICMP_SLT:        // (X u> 13 & X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SGT:
    switch (RHSCC) {
    default: assert(0 && "Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s> 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_SGT:        // (X s> 13 & X s> 15) -> X s> 15
      return ReplaceInstUsesWith(I, RHS);
    case ICmpInst::ICMP_UGT:        // (X s> 13 & X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:
      if (RHSCst == AddOne(LHSCst)) // (X s> 13 & X != 14) -> X s> 14
        return new ICmpInst(LHSCC, Val, RHSCst);
      break;                        // (X s> 13 & X != 15) -> no change
    case ICmpInst::ICMP_SLT:        // (X s> 13 & X s< 15) -> (X-14) s< 1
      return InsertRangeTest(Val, AddOne(LHSCst), RHSCst, true, true, I);
    case ICmpInst::ICMP_ULT:        // (X s> 13 & X u< 15) -> no change
      break;
    }
    break;
  }
 
  return 0;
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
  if (!isa<VectorType>(I.getType())) {
    if (SimplifyDemandedInstructionBits(I))
      return &I;
  } else {
    if (ConstantVector *CP = dyn_cast<ConstantVector>(Op1)) {
      if (CP->isAllOnesValue())            // X & <-1,-1> -> X
        return ReplaceInstUsesWith(I, I.getOperand(0));
    } else if (isa<ConstantAggregateZero>(Op1)) {
      return ReplaceInstUsesWith(I, Op1);  // X & <0,0> -> <0,0>
    }
  }
  
  if (ConstantInt *AndRHS = dyn_cast<ConstantInt>(Op1)) {
    const APInt& AndRHSMask = AndRHS->getValue();
    APInt NotAndRHS(~AndRHSMask);

    // Optimize a variety of ((val OP C1) & C2) combinations...
    if (isa<BinaryOperator>(Op0)) {
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
            Instruction *NewRHS = BinaryOperator::CreateAnd(Op0RHS, AndRHS,
                                                   Op0RHS->getName()+".masked");
            InsertNewInstBefore(NewRHS, I);
            return BinaryOperator::Create(
                       cast<BinaryOperator>(Op0I)->getOpcode(), Op0LHS, NewRHS);
          }
          if (!isa<Constant>(Op0RHS) &&
              MaskedValueIsZero(Op0RHS, NotAndRHS)) {
            // Not masking anything out for the RHS, move to LHS.
            Instruction *NewLHS = BinaryOperator::CreateAnd(Op0LHS, AndRHS,
                                                   Op0LHS->getName()+".masked");
            InsertNewInstBefore(NewLHS, I);
            return BinaryOperator::Create(
                       cast<BinaryOperator>(Op0I)->getOpcode(), NewLHS, Op0RHS);
          }
        }

        break;
      case Instruction::Add:
        // ((A & N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == AndRHS.
        // ((A | N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == 0
        // ((A ^ N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == 0
        if (Value *V = FoldLogicalPlusAnd(Op0LHS, Op0RHS, AndRHS, false, I))
          return BinaryOperator::CreateAnd(V, AndRHS);
        if (Value *V = FoldLogicalPlusAnd(Op0RHS, Op0LHS, AndRHS, false, I))
          return BinaryOperator::CreateAnd(V, AndRHS);  // Add commutes
        break;

      case Instruction::Sub:
        // ((A & N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == AndRHS.
        // ((A | N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == 0
        // ((A ^ N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == 0
        if (Value *V = FoldLogicalPlusAnd(Op0LHS, Op0RHS, AndRHS, true, I))
          return BinaryOperator::CreateAnd(V, AndRHS);

        // (A - N) & AndRHS -> -N & AndRHS iff A&AndRHS==0 and AndRHS
        // has 1's for all bits that the subtraction with A might affect.
        if (Op0I->hasOneUse()) {
          uint32_t BitWidth = AndRHSMask.getBitWidth();
          uint32_t Zeros = AndRHSMask.countLeadingZeros();
          APInt Mask = APInt::getLowBitsSet(BitWidth, BitWidth - Zeros);

          ConstantInt *A = dyn_cast<ConstantInt>(Op0LHS);
          if (!(A && A->isZero()) &&               // avoid infinite recursion.
              MaskedValueIsZero(Op0LHS, Mask)) {
            Instruction *NewNeg = BinaryOperator::CreateNeg(Op0RHS);
            InsertNewInstBefore(NewNeg, I);
            return BinaryOperator::CreateAnd(NewNeg, AndRHS);
          }
        }
        break;

      case Instruction::Shl:
      case Instruction::LShr:
        // (1 << x) & 1 --> zext(x == 0)
        // (1 >> x) & 1 --> zext(x == 0)
        if (AndRHSMask == 1 && Op0LHS == AndRHS) {
          Instruction *NewICmp = new ICmpInst(ICmpInst::ICMP_EQ, Op0RHS,
                                           Constant::getNullValue(I.getType()));
          InsertNewInstBefore(NewICmp, I);
          return new ZExtInst(NewICmp, I.getType());
        }
        break;
      }

      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1)))
        if (Instruction *Res = OptAndOp(Op0I, Op0CI, AndRHS, I))
          return Res;
    } else if (CastInst *CI = dyn_cast<CastInst>(Op0)) {
      // If this is an integer truncation or change from signed-to-unsigned, and
      // if the source is an and/or with immediate, transform it.  This
      // frequently occurs for bitfield accesses.
      if (Instruction *CastOp = dyn_cast<Instruction>(CI->getOperand(0))) {
        if ((isa<TruncInst>(CI) || isa<BitCastInst>(CI)) &&
            CastOp->getNumOperands() == 2)
          if (ConstantInt *AndCI = dyn_cast<ConstantInt>(CastOp->getOperand(1))) {
            if (CastOp->getOpcode() == Instruction::And) {
              // Change: and (cast (and X, C1) to T), C2
              // into  : and (cast X to T), trunc_or_bitcast(C1)&C2
              // This will fold the two constants together, which may allow 
              // other simplifications.
              Instruction *NewCast = CastInst::CreateTruncOrBitCast(
                CastOp->getOperand(0), I.getType(), 
                CastOp->getName()+".shrunk");
              NewCast = InsertNewInstBefore(NewCast, I);
              // trunc_or_bitcast(C1)&C2
              Constant *C3 = ConstantExpr::getTruncOrBitCast(AndCI,I.getType());
              C3 = ConstantExpr::getAnd(C3, AndRHS);
              return BinaryOperator::CreateAnd(NewCast, C3);
            } else if (CastOp->getOpcode() == Instruction::Or) {
              // Change: and (cast (or X, C1) to T), C2
              // into  : trunc(C1)&C2 iff trunc(C1)&C2 == C2
              Constant *C3 = ConstantExpr::getTruncOrBitCast(AndCI,I.getType());
              if (ConstantExpr::getAnd(C3, AndRHS) == AndRHS)   // trunc(C1)&C2
                return ReplaceInstUsesWith(I, AndRHS);
            }
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
    Instruction *Or = BinaryOperator::CreateOr(Op0NotVal, Op1NotVal,
                                               I.getName()+".demorgan");
    InsertNewInstBefore(Or, I);
    return BinaryOperator::CreateNot(Or);
  }
  
  {
    Value *A = 0, *B = 0, *C = 0, *D = 0;
    if (match(Op0, m_Or(m_Value(A), m_Value(B)))) {
      if (A == Op1 || B == Op1)    // (A | ?) & A  --> A
        return ReplaceInstUsesWith(I, Op1);
    
      // (A|B) & ~(A&B) -> A^B
      if (match(Op1, m_Not(m_And(m_Value(C), m_Value(D))))) {
        if ((A == C && B == D) || (A == D && B == C))
          return BinaryOperator::CreateXor(A, B);
      }
    }
    
    if (match(Op1, m_Or(m_Value(A), m_Value(B)))) {
      if (A == Op0 || B == Op0)    // A & (A | ?)  --> A
        return ReplaceInstUsesWith(I, Op0);

      // ~(A&B) & (A|B) -> A^B
      if (match(Op0, m_Not(m_And(m_Value(C), m_Value(D))))) {
        if ((A == C && B == D) || (A == D && B == C))
          return BinaryOperator::CreateXor(A, B);
      }
    }
    
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
        Instruction *NotB = BinaryOperator::CreateNot(B, "tmp");
        InsertNewInstBefore(NotB, I);
        return BinaryOperator::CreateAnd(A, NotB);
      }
    }

    // (A&((~A)|B)) -> A&B
    if (match(Op0, m_Or(m_Not(m_Specific(Op1)), m_Value(A))) ||
        match(Op0, m_Or(m_Value(A), m_Not(m_Specific(Op1)))))
      return BinaryOperator::CreateAnd(A, Op1);
    if (match(Op1, m_Or(m_Not(m_Specific(Op0)), m_Value(A))) ||
        match(Op1, m_Or(m_Value(A), m_Not(m_Specific(Op0)))))
      return BinaryOperator::CreateAnd(A, Op0);
  }
  
  if (ICmpInst *RHS = dyn_cast<ICmpInst>(Op1)) {
    // (icmp1 A, B) & (icmp2 A, B) --> (icmp3 A, B)
    if (Instruction *R = AssociativeOpt(I, FoldICmpLogical(*this, RHS)))
      return R;

    if (ICmpInst *LHS = dyn_cast<ICmpInst>(Op0))
      if (Instruction *Res = FoldAndOfICmps(I, LHS, RHS))
        return Res;
  }

  // fold (and (cast A), (cast B)) -> (cast (and A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0))
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1))
      if (Op0C->getOpcode() == Op1C->getOpcode()) { // same cast kind ?
        const Type *SrcTy = Op0C->getOperand(0)->getType();
        if (SrcTy == Op1C->getOperand(0)->getType() && SrcTy->isInteger() &&
            // Only do this if the casts both really cause code to be generated.
            ValueRequiresCast(Op0C->getOpcode(), Op0C->getOperand(0), 
                              I.getType(), TD) &&
            ValueRequiresCast(Op1C->getOpcode(), Op1C->getOperand(0), 
                              I.getType(), TD)) {
          Instruction *NewOp = BinaryOperator::CreateAnd(Op0C->getOperand(0),
                                                         Op1C->getOperand(0),
                                                         I.getName());
          InsertNewInstBefore(NewOp, I);
          return CastInst::Create(Op0C->getOpcode(), NewOp, I.getType());
        }
      }
    
  // (X >> Z) & (Y >> Z)  -> (X&Y) >> Z  for all shifts.
  if (BinaryOperator *SI1 = dyn_cast<BinaryOperator>(Op1)) {
    if (BinaryOperator *SI0 = dyn_cast<BinaryOperator>(Op0))
      if (SI0->isShift() && SI0->getOpcode() == SI1->getOpcode() && 
          SI0->getOperand(1) == SI1->getOperand(1) &&
          (SI0->hasOneUse() || SI1->hasOneUse())) {
        Instruction *NewOp =
          InsertNewInstBefore(BinaryOperator::CreateAnd(SI0->getOperand(0),
                                                        SI1->getOperand(0),
                                                        SI0->getName()), I);
        return BinaryOperator::Create(SI1->getOpcode(), NewOp, 
                                      SI1->getOperand(1));
      }
  }

  // If and'ing two fcmp, try combine them into one.
  if (FCmpInst *LHS = dyn_cast<FCmpInst>(I.getOperand(0))) {
    if (FCmpInst *RHS = dyn_cast<FCmpInst>(I.getOperand(1))) {
      if (LHS->getPredicate() == FCmpInst::FCMP_ORD &&
          RHS->getPredicate() == FCmpInst::FCMP_ORD) {
        // (fcmp ord x, c) & (fcmp ord y, c)  -> (fcmp ord x, y)
        if (ConstantFP *LHSC = dyn_cast<ConstantFP>(LHS->getOperand(1)))
          if (ConstantFP *RHSC = dyn_cast<ConstantFP>(RHS->getOperand(1))) {
            // If either of the constants are nans, then the whole thing returns
            // false.
            if (LHSC->getValueAPF().isNaN() || RHSC->getValueAPF().isNaN())
              return ReplaceInstUsesWith(I, ConstantInt::getFalse());
            return new FCmpInst(FCmpInst::FCMP_ORD, LHS->getOperand(0),
                                RHS->getOperand(0));
          }
      } else {
        Value *Op0LHS, *Op0RHS, *Op1LHS, *Op1RHS;
        FCmpInst::Predicate Op0CC, Op1CC;
        if (match(Op0, m_FCmp(Op0CC, m_Value(Op0LHS), m_Value(Op0RHS))) &&
            match(Op1, m_FCmp(Op1CC, m_Value(Op1LHS), m_Value(Op1RHS)))) {
          if (Op0LHS == Op1RHS && Op0RHS == Op1LHS) {
            // Swap RHS operands to match LHS.
            Op1CC = FCmpInst::getSwappedPredicate(Op1CC);
            std::swap(Op1LHS, Op1RHS);
          }
          if (Op0LHS == Op1LHS && Op0RHS == Op1RHS) {
            // Simplify (fcmp cc0 x, y) & (fcmp cc1 x, y).
            if (Op0CC == Op1CC)
              return new FCmpInst((FCmpInst::Predicate)Op0CC, Op0LHS, Op0RHS);
            else if (Op0CC == FCmpInst::FCMP_FALSE ||
                     Op1CC == FCmpInst::FCMP_FALSE)
              return ReplaceInstUsesWith(I, ConstantInt::getFalse());
            else if (Op0CC == FCmpInst::FCMP_TRUE)
              return ReplaceInstUsesWith(I, Op1);
            else if (Op1CC == FCmpInst::FCMP_TRUE)
              return ReplaceInstUsesWith(I, Op0);
            bool Op0Ordered;
            bool Op1Ordered;
            unsigned Op0Pred = getFCmpCode(Op0CC, Op0Ordered);
            unsigned Op1Pred = getFCmpCode(Op1CC, Op1Ordered);
            if (Op1Pred == 0) {
              std::swap(Op0, Op1);
              std::swap(Op0Pred, Op1Pred);
              std::swap(Op0Ordered, Op1Ordered);
            }
            if (Op0Pred == 0) {
              // uno && ueq -> uno && (uno || eq) -> ueq
              // ord && olt -> ord && (ord && lt) -> olt
              if (Op0Ordered == Op1Ordered)
                return ReplaceInstUsesWith(I, Op1);
              // uno && oeq -> uno && (ord && eq) -> false
              // uno && ord -> false
              if (!Op0Ordered)
                return ReplaceInstUsesWith(I, ConstantInt::getFalse());
              // ord && ueq -> ord && (uno || eq) -> oeq
              return cast<Instruction>(getFCmpValue(true, Op1Pred,
                                                    Op0LHS, Op0RHS));
            }
          }
        }
      }
    }
  }

  return Changed ? &I : 0;
}

/// CollectBSwapParts - Analyze the specified subexpression and see if it is
/// capable of providing pieces of a bswap.  The subexpression provides pieces
/// of a bswap if it is proven that each of the non-zero bytes in the output of
/// the expression came from the corresponding "byte swapped" byte in some other
/// value.  For example, if the current subexpression is "(shl i32 %X, 24)" then
/// we know that the expression deposits the low byte of %X into the high byte
/// of the bswap result and that all other bytes are zero.  This expression is
/// accepted, the high byte of ByteValues is set to X to indicate a correct
/// match.
///
/// This function returns true if the match was unsuccessful and false if so.
/// On entry to the function the "OverallLeftShift" is a signed integer value
/// indicating the number of bytes that the subexpression is later shifted.  For
/// example, if the expression is later right shifted by 16 bits, the
/// OverallLeftShift value would be -2 on entry.  This is used to specify which
/// byte of ByteValues is actually being set.
///
/// Similarly, ByteMask is a bitmask where a bit is clear if its corresponding
/// byte is masked to zero by a user.  For example, in (X & 255), X will be
/// processed with a bytemask of 1.  Because bytemask is 32-bits, this limits
/// this function to working on up to 32-byte (256 bit) values.  ByteMask is
/// always in the local (OverallLeftShift) coordinate space.
///
static bool CollectBSwapParts(Value *V, int OverallLeftShift, uint32_t ByteMask,
                              SmallVector<Value*, 8> &ByteValues) {
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    // If this is an or instruction, it may be an inner node of the bswap.
    if (I->getOpcode() == Instruction::Or) {
      return CollectBSwapParts(I->getOperand(0), OverallLeftShift, ByteMask,
                               ByteValues) ||
             CollectBSwapParts(I->getOperand(1), OverallLeftShift, ByteMask,
                               ByteValues);
    }
  
    // If this is a logical shift by a constant multiple of 8, recurse with
    // OverallLeftShift and ByteMask adjusted.
    if (I->isLogicalShift() && isa<ConstantInt>(I->getOperand(1))) {
      unsigned ShAmt = 
        cast<ConstantInt>(I->getOperand(1))->getLimitedValue(~0U);
      // Ensure the shift amount is defined and of a byte value.
      if ((ShAmt & 7) || (ShAmt > 8*ByteValues.size()))
        return true;

      unsigned ByteShift = ShAmt >> 3;
      if (I->getOpcode() == Instruction::Shl) {
        // X << 2 -> collect(X, +2)
        OverallLeftShift += ByteShift;
        ByteMask >>= ByteShift;
      } else {
        // X >>u 2 -> collect(X, -2)
        OverallLeftShift -= ByteShift;
        ByteMask <<= ByteShift;
        ByteMask &= (~0U >> (32-ByteValues.size()));
      }

      if (OverallLeftShift >= (int)ByteValues.size()) return true;
      if (OverallLeftShift <= -(int)ByteValues.size()) return true;

      return CollectBSwapParts(I->getOperand(0), OverallLeftShift, ByteMask, 
                               ByteValues);
    }

    // If this is a logical 'and' with a mask that clears bytes, clear the
    // corresponding bytes in ByteMask.
    if (I->getOpcode() == Instruction::And &&
        isa<ConstantInt>(I->getOperand(1))) {
      // Scan every byte of the and mask, seeing if the byte is either 0 or 255.
      unsigned NumBytes = ByteValues.size();
      APInt Byte(I->getType()->getPrimitiveSizeInBits(), 255);
      const APInt &AndMask = cast<ConstantInt>(I->getOperand(1))->getValue();
      
      for (unsigned i = 0; i != NumBytes; ++i, Byte <<= 8) {
        // If this byte is masked out by a later operation, we don't care what
        // the and mask is.
        if ((ByteMask & (1 << i)) == 0)
          continue;
        
        // If the AndMask is all zeros for this byte, clear the bit.
        APInt MaskB = AndMask & Byte;
        if (MaskB == 0) {
          ByteMask &= ~(1U << i);
          continue;
        }
        
        // If the AndMask is not all ones for this byte, it's not a bytezap.
        if (MaskB != Byte)
          return true;

        // Otherwise, this byte is kept.
      }

      return CollectBSwapParts(I->getOperand(0), OverallLeftShift, ByteMask, 
                               ByteValues);
    }
  }
  
  // Okay, we got to something that isn't a shift, 'or' or 'and'.  This must be
  // the input value to the bswap.  Some observations: 1) if more than one byte
  // is demanded from this input, then it could not be successfully assembled
  // into a byteswap.  At least one of the two bytes would not be aligned with
  // their ultimate destination.
  if (!isPowerOf2_32(ByteMask)) return true;
  unsigned InputByteNo = CountTrailingZeros_32(ByteMask);
  
  // 2) The input and ultimate destinations must line up: if byte 3 of an i32
  // is demanded, it needs to go into byte 0 of the result.  This means that the
  // byte needs to be shifted until it lands in the right byte bucket.  The
  // shift amount depends on the position: if the byte is coming from the high
  // part of the value (e.g. byte 3) then it must be shifted right.  If from the
  // low part, it must be shifted left.
  unsigned DestByteNo = InputByteNo + OverallLeftShift;
  if (InputByteNo < ByteValues.size()/2) {
    if (ByteValues.size()-1-DestByteNo != InputByteNo)
      return true;
  } else {
    if (ByteValues.size()-1-DestByteNo != InputByteNo)
      return true;
  }
  
  // If the destination byte value is already defined, the values are or'd
  // together, which isn't a bswap (unless it's an or of the same bits).
  if (ByteValues[DestByteNo] && ByteValues[DestByteNo] != V)
    return true;
  ByteValues[DestByteNo] = V;
  return false;
}

/// MatchBSwap - Given an OR instruction, check to see if this is a bswap idiom.
/// If so, insert the new bswap intrinsic and return it.
Instruction *InstCombiner::MatchBSwap(BinaryOperator &I) {
  const IntegerType *ITy = dyn_cast<IntegerType>(I.getType());
  if (!ITy || ITy->getBitWidth() % 16 || 
      // ByteMask only allows up to 32-byte values.
      ITy->getBitWidth() > 32*8) 
    return 0;   // Can only bswap pairs of bytes.  Can't do vectors.
  
  /// ByteValues - For each byte of the result, we keep track of which value
  /// defines each byte.
  SmallVector<Value*, 8> ByteValues;
  ByteValues.resize(ITy->getBitWidth()/8);
    
  // Try to find all the pieces corresponding to the bswap.
  uint32_t ByteMask = ~0U >> (32-ByteValues.size());
  if (CollectBSwapParts(&I, 0, ByteMask, ByteValues))
    return 0;
  
  // Check to see if all of the bytes come from the same value.
  Value *V = ByteValues[0];
  if (V == 0) return 0;  // Didn't find a byte?  Must be zero.
  
  // Check to make sure that all of the bytes come from the same value.
  for (unsigned i = 1, e = ByteValues.size(); i != e; ++i)
    if (ByteValues[i] != V)
      return 0;
  const Type *Tys[] = { ITy };
  Module *M = I.getParent()->getParent()->getParent();
  Function *F = Intrinsic::getDeclaration(M, Intrinsic::bswap, Tys, 1);
  return CallInst::Create(F, V);
}

/// MatchSelectFromAndOr - We have an expression of the form (A&C)|(B&D).  Check
/// If A is (cond?-1:0) and either B or D is ~(cond?-1,0) or (cond?0,-1), then
/// we can simplify this expression to "cond ? C : D or B".
static Instruction *MatchSelectFromAndOr(Value *A, Value *B,
                                         Value *C, Value *D) {
  // If A is not a select of -1/0, this cannot match.
  Value *Cond = 0;
  if (!match(A, m_SelectCst<-1, 0>(m_Value(Cond))))
    return 0;

  // ((cond?-1:0)&C) | (B&(cond?0:-1)) -> cond ? C : B.
  if (match(D, m_SelectCst<0, -1>(m_Specific(Cond))))
    return SelectInst::Create(Cond, C, B);
  if (match(D, m_Not(m_SelectCst<-1, 0>(m_Specific(Cond)))))
    return SelectInst::Create(Cond, C, B);
  // ((cond?-1:0)&C) | ((cond?0:-1)&D) -> cond ? C : D.
  if (match(B, m_SelectCst<0, -1>(m_Specific(Cond))))
    return SelectInst::Create(Cond, C, D);
  if (match(B, m_Not(m_SelectCst<-1, 0>(m_Specific(Cond)))))
    return SelectInst::Create(Cond, C, D);
  return 0;
}

/// FoldOrOfICmps - Fold (icmp)|(icmp) if possible.
Instruction *InstCombiner::FoldOrOfICmps(Instruction &I,
                                         ICmpInst *LHS, ICmpInst *RHS) {
  Value *Val, *Val2;
  ConstantInt *LHSCst, *RHSCst;
  ICmpInst::Predicate LHSCC, RHSCC;
  
  // This only handles icmp of constants: (icmp1 A, C1) | (icmp2 B, C2).
  if (!match(LHS, m_ICmp(LHSCC, m_Value(Val), m_ConstantInt(LHSCst))) ||
      !match(RHS, m_ICmp(RHSCC, m_Value(Val2), m_ConstantInt(RHSCst))))
    return 0;
  
  // From here on, we only handle:
  //    (icmp1 A, C1) | (icmp2 A, C2) --> something simpler.
  if (Val != Val2) return 0;
  
  // ICMP_[US][GL]E X, CST is folded to ICMP_[US][GL]T elsewhere.
  if (LHSCC == ICmpInst::ICMP_UGE || LHSCC == ICmpInst::ICMP_ULE ||
      RHSCC == ICmpInst::ICMP_UGE || RHSCC == ICmpInst::ICMP_ULE ||
      LHSCC == ICmpInst::ICMP_SGE || LHSCC == ICmpInst::ICMP_SLE ||
      RHSCC == ICmpInst::ICMP_SGE || RHSCC == ICmpInst::ICMP_SLE)
    return 0;
  
  // We can't fold (ugt x, C) | (sgt x, C2).
  if (!PredicatesFoldable(LHSCC, RHSCC))
    return 0;
  
  // Ensure that the larger constant is on the RHS.
  bool ShouldSwap;
  if (ICmpInst::isSignedPredicate(LHSCC) ||
      (ICmpInst::isEquality(LHSCC) && 
       ICmpInst::isSignedPredicate(RHSCC)))
    ShouldSwap = LHSCst->getValue().sgt(RHSCst->getValue());
  else
    ShouldSwap = LHSCst->getValue().ugt(RHSCst->getValue());
  
  if (ShouldSwap) {
    std::swap(LHS, RHS);
    std::swap(LHSCst, RHSCst);
    std::swap(LHSCC, RHSCC);
  }
  
  // At this point, we know we have have two icmp instructions
  // comparing a value against two constants and or'ing the result
  // together.  Because of the above check, we know that we only have
  // ICMP_EQ, ICMP_NE, ICMP_LT, and ICMP_GT here. We also know (from the
  // FoldICmpLogical check above), that the two constants are not
  // equal.
  assert(LHSCst != RHSCst && "Compares not folded above?");

  switch (LHSCC) {
  default: assert(0 && "Unknown integer condition code!");
  case ICmpInst::ICMP_EQ:
    switch (RHSCC) {
    default: assert(0 && "Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:
      if (LHSCst == SubOne(RHSCst)) { // (X == 13 | X == 14) -> X-13 <u 2
        Constant *AddCST = ConstantExpr::getNeg(LHSCst);
        Instruction *Add = BinaryOperator::CreateAdd(Val, AddCST,
                                                     Val->getName()+".off");
        InsertNewInstBefore(Add, I);
        AddCST = Subtract(AddOne(RHSCst), LHSCst);
        return new ICmpInst(ICmpInst::ICMP_ULT, Add, AddCST);
      }
      break;                         // (X == 13 | X == 15) -> no change
    case ICmpInst::ICMP_UGT:         // (X == 13 | X u> 14) -> no change
    case ICmpInst::ICMP_SGT:         // (X == 13 | X s> 14) -> no change
      break;
    case ICmpInst::ICMP_NE:          // (X == 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_ULT:         // (X == 13 | X u< 15) -> X u< 15
    case ICmpInst::ICMP_SLT:         // (X == 13 | X s< 15) -> X s< 15
      return ReplaceInstUsesWith(I, RHS);
    }
    break;
  case ICmpInst::ICMP_NE:
    switch (RHSCC) {
    default: assert(0 && "Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:          // (X != 13 | X == 15) -> X != 13
    case ICmpInst::ICMP_UGT:         // (X != 13 | X u> 15) -> X != 13
    case ICmpInst::ICMP_SGT:         // (X != 13 | X s> 15) -> X != 13
      return ReplaceInstUsesWith(I, LHS);
    case ICmpInst::ICMP_NE:          // (X != 13 | X != 15) -> true
    case ICmpInst::ICMP_ULT:         // (X != 13 | X u< 15) -> true
    case ICmpInst::ICMP_SLT:         // (X != 13 | X s< 15) -> true
      return ReplaceInstUsesWith(I, ConstantInt::getTrue());
    }
    break;
  case ICmpInst::ICMP_ULT:
    switch (RHSCC) {
    default: assert(0 && "Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u< 13 | X == 14) -> no change
      break;
    case ICmpInst::ICMP_UGT:        // (X u< 13 | X u> 15) -> (X-13) u> 2
      // If RHSCst is [us]MAXINT, it is always false.  Not handling
      // this can cause overflow.
      if (RHSCst->isMaxValue(false))
        return ReplaceInstUsesWith(I, LHS);
      return InsertRangeTest(Val, LHSCst, AddOne(RHSCst), false, false, I);
    case ICmpInst::ICMP_SGT:        // (X u< 13 | X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X u< 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_ULT:        // (X u< 13 | X u< 15) -> X u< 15
      return ReplaceInstUsesWith(I, RHS);
    case ICmpInst::ICMP_SLT:        // (X u< 13 | X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SLT:
    switch (RHSCC) {
    default: assert(0 && "Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s< 13 | X == 14) -> no change
      break;
    case ICmpInst::ICMP_SGT:        // (X s< 13 | X s> 15) -> (X-13) s> 2
      // If RHSCst is [us]MAXINT, it is always false.  Not handling
      // this can cause overflow.
      if (RHSCst->isMaxValue(true))
        return ReplaceInstUsesWith(I, LHS);
      return InsertRangeTest(Val, LHSCst, AddOne(RHSCst), true, false, I);
    case ICmpInst::ICMP_UGT:        // (X s< 13 | X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X s< 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_SLT:        // (X s< 13 | X s< 15) -> X s< 15
      return ReplaceInstUsesWith(I, RHS);
    case ICmpInst::ICMP_ULT:        // (X s< 13 | X u< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_UGT:
    switch (RHSCC) {
    default: assert(0 && "Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u> 13 | X == 15) -> X u> 13
    case ICmpInst::ICMP_UGT:        // (X u> 13 | X u> 15) -> X u> 13
      return ReplaceInstUsesWith(I, LHS);
    case ICmpInst::ICMP_SGT:        // (X u> 13 | X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X u> 13 | X != 15) -> true
    case ICmpInst::ICMP_ULT:        // (X u> 13 | X u< 15) -> true
      return ReplaceInstUsesWith(I, ConstantInt::getTrue());
    case ICmpInst::ICMP_SLT:        // (X u> 13 | X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SGT:
    switch (RHSCC) {
    default: assert(0 && "Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s> 13 | X == 15) -> X > 13
    case ICmpInst::ICMP_SGT:        // (X s> 13 | X s> 15) -> X > 13
      return ReplaceInstUsesWith(I, LHS);
    case ICmpInst::ICMP_UGT:        // (X s> 13 | X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X s> 13 | X != 15) -> true
    case ICmpInst::ICMP_SLT:        // (X s> 13 | X s< 15) -> true
      return ReplaceInstUsesWith(I, ConstantInt::getTrue());
    case ICmpInst::ICMP_ULT:        // (X s> 13 | X u< 15) -> no change
      break;
    }
    break;
  }
  return 0;
}

/// FoldOrWithConstants - This helper function folds:
///
///     ((A | B) & C1) | (B & C2)
///
/// into:
/// 
///     (A & C1) | B
///
/// when the XOR of the two constants is "all ones" (-1).
Instruction *InstCombiner::FoldOrWithConstants(BinaryOperator &I, Value *Op,
                                               Value *A, Value *B, Value *C) {
  ConstantInt *CI1 = dyn_cast<ConstantInt>(C);
  if (!CI1) return 0;

  Value *V1 = 0;
  ConstantInt *CI2 = 0;
  if (!match(Op, m_And(m_Value(V1), m_ConstantInt(CI2)))) return 0;

  APInt Xor = CI1->getValue() ^ CI2->getValue();
  if (!Xor.isAllOnesValue()) return 0;

  if (V1 == A || V1 == B) {
    Instruction *NewOp =
      InsertNewInstBefore(BinaryOperator::CreateAnd((V1 == A) ? B : A, CI1), I);
    return BinaryOperator::CreateOr(NewOp, V1);
  }

  return 0;
}

Instruction *InstCombiner::visitOr(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op1))                       // X | undef -> -1
    return ReplaceInstUsesWith(I, Constant::getAllOnesValue(I.getType()));

  // or X, X = X
  if (Op0 == Op1)
    return ReplaceInstUsesWith(I, Op0);

  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  if (!isa<VectorType>(I.getType())) {
    if (SimplifyDemandedInstructionBits(I))
      return &I;
  } else if (isa<ConstantAggregateZero>(Op1)) {
    return ReplaceInstUsesWith(I, Op0);  // X | <0,0> -> X
  } else if (ConstantVector *CP = dyn_cast<ConstantVector>(Op1)) {
    if (CP->isAllOnesValue())            // X | <-1,-1> -> <-1,-1>
      return ReplaceInstUsesWith(I, I.getOperand(1));
  }
    

  
  // or X, -1 == -1
  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    ConstantInt *C1 = 0; Value *X = 0;
    // (X & C1) | C2 --> (X | C2) & (C1|C2)
    if (match(Op0, m_And(m_Value(X), m_ConstantInt(C1))) && isOnlyUse(Op0)) {
      Instruction *Or = BinaryOperator::CreateOr(X, RHS);
      InsertNewInstBefore(Or, I);
      Or->takeName(Op0);
      return BinaryOperator::CreateAnd(Or, 
               ConstantInt::get(RHS->getValue() | C1->getValue()));
    }

    // (X ^ C1) | C2 --> (X | C2) ^ (C1&~C2)
    if (match(Op0, m_Xor(m_Value(X), m_ConstantInt(C1))) && isOnlyUse(Op0)) {
      Instruction *Or = BinaryOperator::CreateOr(X, RHS);
      InsertNewInstBefore(Or, I);
      Or->takeName(Op0);
      return BinaryOperator::CreateXor(Or,
                 ConstantInt::get(C1->getValue() & ~RHS->getValue()));
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
      MaskedValueIsZero(Op1, C1->getValue())) {
    Instruction *NOr = BinaryOperator::CreateOr(A, Op1);
    InsertNewInstBefore(NOr, I);
    NOr->takeName(Op0);
    return BinaryOperator::CreateXor(NOr, C1);
  }

  // Y|(X^C) -> (X|Y)^C iff Y&C == 0
  if (Op1->hasOneUse() && match(Op1, m_Xor(m_Value(A), m_ConstantInt(C1))) &&
      MaskedValueIsZero(Op0, C1->getValue())) {
    Instruction *NOr = BinaryOperator::CreateOr(A, Op0);
    InsertNewInstBefore(NOr, I);
    NOr->takeName(Op0);
    return BinaryOperator::CreateXor(NOr, C1);
  }

  // (A & C)|(B & D)
  Value *C = 0, *D = 0;
  if (match(Op0, m_And(m_Value(A), m_Value(C))) &&
      match(Op1, m_And(m_Value(B), m_Value(D)))) {
    Value *V1 = 0, *V2 = 0, *V3 = 0;
    C1 = dyn_cast<ConstantInt>(C);
    C2 = dyn_cast<ConstantInt>(D);
    if (C1 && C2) {  // (A & C1)|(B & C2)
      // If we have: ((V + N) & C1) | (V & C2)
      // .. and C2 = ~C1 and C2 is 0+1+ and (N & C2) == 0
      // replace with V+N.
      if (C1->getValue() == ~C2->getValue()) {
        if ((C2->getValue() & (C2->getValue()+1)) == 0 && // C2 == 0+1+
            match(A, m_Add(m_Value(V1), m_Value(V2)))) {
          // Add commutes, try both ways.
          if (V1 == B && MaskedValueIsZero(V2, C2->getValue()))
            return ReplaceInstUsesWith(I, A);
          if (V2 == B && MaskedValueIsZero(V1, C2->getValue()))
            return ReplaceInstUsesWith(I, A);
        }
        // Or commutes, try both ways.
        if ((C1->getValue() & (C1->getValue()+1)) == 0 &&
            match(B, m_Add(m_Value(V1), m_Value(V2)))) {
          // Add commutes, try both ways.
          if (V1 == A && MaskedValueIsZero(V2, C1->getValue()))
            return ReplaceInstUsesWith(I, B);
          if (V2 == A && MaskedValueIsZero(V1, C1->getValue()))
            return ReplaceInstUsesWith(I, B);
        }
      }
      V1 = 0; V2 = 0; V3 = 0;
    }
    
    // Check to see if we have any common things being and'ed.  If so, find the
    // terms for V1 & (V2|V3).
    if (isOnlyUse(Op0) || isOnlyUse(Op1)) {
      if (A == B)      // (A & C)|(A & D) == A & (C|D)
        V1 = A, V2 = C, V3 = D;
      else if (A == D) // (A & C)|(B & A) == A & (B|C)
        V1 = A, V2 = B, V3 = C;
      else if (C == B) // (A & C)|(C & D) == C & (A|D)
        V1 = C, V2 = A, V3 = D;
      else if (C == D) // (A & C)|(B & C) == C & (A|B)
        V1 = C, V2 = A, V3 = B;
      
      if (V1) {
        Value *Or =
          InsertNewInstBefore(BinaryOperator::CreateOr(V2, V3, "tmp"), I);
        return BinaryOperator::CreateAnd(V1, Or);
      }
    }

    // (A & (C0?-1:0)) | (B & ~(C0?-1:0)) ->  C0 ? A : B, and commuted variants
    if (Instruction *Match = MatchSelectFromAndOr(A, B, C, D))
      return Match;
    if (Instruction *Match = MatchSelectFromAndOr(B, A, D, C))
      return Match;
    if (Instruction *Match = MatchSelectFromAndOr(C, B, A, D))
      return Match;
    if (Instruction *Match = MatchSelectFromAndOr(D, A, B, C))
      return Match;

    // ((A&~B)|(~A&B)) -> A^B
    if ((match(C, m_Not(m_Specific(D))) &&
         match(B, m_Not(m_Specific(A)))))
      return BinaryOperator::CreateXor(A, D);
    // ((~B&A)|(~A&B)) -> A^B
    if ((match(A, m_Not(m_Specific(D))) &&
         match(B, m_Not(m_Specific(C)))))
      return BinaryOperator::CreateXor(C, D);
    // ((A&~B)|(B&~A)) -> A^B
    if ((match(C, m_Not(m_Specific(B))) &&
         match(D, m_Not(m_Specific(A)))))
      return BinaryOperator::CreateXor(A, B);
    // ((~B&A)|(B&~A)) -> A^B
    if ((match(A, m_Not(m_Specific(B))) &&
         match(D, m_Not(m_Specific(C)))))
      return BinaryOperator::CreateXor(C, B);
  }
  
  // (X >> Z) | (Y >> Z)  -> (X|Y) >> Z  for all shifts.
  if (BinaryOperator *SI1 = dyn_cast<BinaryOperator>(Op1)) {
    if (BinaryOperator *SI0 = dyn_cast<BinaryOperator>(Op0))
      if (SI0->isShift() && SI0->getOpcode() == SI1->getOpcode() && 
          SI0->getOperand(1) == SI1->getOperand(1) &&
          (SI0->hasOneUse() || SI1->hasOneUse())) {
        Instruction *NewOp =
        InsertNewInstBefore(BinaryOperator::CreateOr(SI0->getOperand(0),
                                                     SI1->getOperand(0),
                                                     SI0->getName()), I);
        return BinaryOperator::Create(SI1->getOpcode(), NewOp, 
                                      SI1->getOperand(1));
      }
  }

  // ((A|B)&1)|(B&-2) -> (A&1) | B
  if (match(Op0, m_And(m_Or(m_Value(A), m_Value(B)), m_Value(C))) ||
      match(Op0, m_And(m_Value(C), m_Or(m_Value(A), m_Value(B))))) {
    Instruction *Ret = FoldOrWithConstants(I, Op1, A, B, C);
    if (Ret) return Ret;
  }
  // (B&-2)|((A|B)&1) -> (A&1) | B
  if (match(Op1, m_And(m_Or(m_Value(A), m_Value(B)), m_Value(C))) ||
      match(Op1, m_And(m_Value(C), m_Or(m_Value(A), m_Value(B))))) {
    Instruction *Ret = FoldOrWithConstants(I, Op0, A, B, C);
    if (Ret) return Ret;
  }

  if (match(Op0, m_Not(m_Value(A)))) {   // ~A | Op1
    if (A == Op1)   // ~A | A == -1
      return ReplaceInstUsesWith(I, Constant::getAllOnesValue(I.getType()));
  } else {
    A = 0;
  }
  // Note, A is still live here!
  if (match(Op1, m_Not(m_Value(B)))) {   // Op0 | ~B
    if (Op0 == B)
      return ReplaceInstUsesWith(I, Constant::getAllOnesValue(I.getType()));

    // (~A | ~B) == (~(A & B)) - De Morgan's Law
    if (A && isOnlyUse(Op0) && isOnlyUse(Op1)) {
      Value *And = InsertNewInstBefore(BinaryOperator::CreateAnd(A, B,
                                              I.getName()+".demorgan"), I);
      return BinaryOperator::CreateNot(And);
    }
  }

  // (icmp1 A, B) | (icmp2 A, B) --> (icmp3 A, B)
  if (ICmpInst *RHS = dyn_cast<ICmpInst>(I.getOperand(1))) {
    if (Instruction *R = AssociativeOpt(I, FoldICmpLogical(*this, RHS)))
      return R;

    if (ICmpInst *LHS = dyn_cast<ICmpInst>(I.getOperand(0)))
      if (Instruction *Res = FoldOrOfICmps(I, LHS, RHS))
        return Res;
  }
    
  // fold (or (cast A), (cast B)) -> (cast (or A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1))
      if (Op0C->getOpcode() == Op1C->getOpcode()) {// same cast kind ?
        if (!isa<ICmpInst>(Op0C->getOperand(0)) ||
            !isa<ICmpInst>(Op1C->getOperand(0))) {
          const Type *SrcTy = Op0C->getOperand(0)->getType();
          if (SrcTy == Op1C->getOperand(0)->getType() && SrcTy->isInteger() &&
              // Only do this if the casts both really cause code to be
              // generated.
              ValueRequiresCast(Op0C->getOpcode(), Op0C->getOperand(0), 
                                I.getType(), TD) &&
              ValueRequiresCast(Op1C->getOpcode(), Op1C->getOperand(0), 
                                I.getType(), TD)) {
            Instruction *NewOp = BinaryOperator::CreateOr(Op0C->getOperand(0),
                                                          Op1C->getOperand(0),
                                                          I.getName());
            InsertNewInstBefore(NewOp, I);
            return CastInst::Create(Op0C->getOpcode(), NewOp, I.getType());
          }
        }
      }
  }
  
    
  // (fcmp uno x, c) | (fcmp uno y, c)  -> (fcmp uno x, y)
  if (FCmpInst *LHS = dyn_cast<FCmpInst>(I.getOperand(0))) {
    if (FCmpInst *RHS = dyn_cast<FCmpInst>(I.getOperand(1))) {
      if (LHS->getPredicate() == FCmpInst::FCMP_UNO &&
          RHS->getPredicate() == FCmpInst::FCMP_UNO && 
          LHS->getOperand(0)->getType() == RHS->getOperand(0)->getType()) {
        if (ConstantFP *LHSC = dyn_cast<ConstantFP>(LHS->getOperand(1)))
          if (ConstantFP *RHSC = dyn_cast<ConstantFP>(RHS->getOperand(1))) {
            // If either of the constants are nans, then the whole thing returns
            // true.
            if (LHSC->getValueAPF().isNaN() || RHSC->getValueAPF().isNaN())
              return ReplaceInstUsesWith(I, ConstantInt::getTrue());
            
            // Otherwise, no need to compare the two constants, compare the
            // rest.
            return new FCmpInst(FCmpInst::FCMP_UNO, LHS->getOperand(0),
                                RHS->getOperand(0));
          }
      } else {
        Value *Op0LHS, *Op0RHS, *Op1LHS, *Op1RHS;
        FCmpInst::Predicate Op0CC, Op1CC;
        if (match(Op0, m_FCmp(Op0CC, m_Value(Op0LHS), m_Value(Op0RHS))) &&
            match(Op1, m_FCmp(Op1CC, m_Value(Op1LHS), m_Value(Op1RHS)))) {
          if (Op0LHS == Op1RHS && Op0RHS == Op1LHS) {
            // Swap RHS operands to match LHS.
            Op1CC = FCmpInst::getSwappedPredicate(Op1CC);
            std::swap(Op1LHS, Op1RHS);
          }
          if (Op0LHS == Op1LHS && Op0RHS == Op1RHS) {
            // Simplify (fcmp cc0 x, y) | (fcmp cc1 x, y).
            if (Op0CC == Op1CC)
              return new FCmpInst((FCmpInst::Predicate)Op0CC, Op0LHS, Op0RHS);
            else if (Op0CC == FCmpInst::FCMP_TRUE ||
                     Op1CC == FCmpInst::FCMP_TRUE)
              return ReplaceInstUsesWith(I, ConstantInt::getTrue());
            else if (Op0CC == FCmpInst::FCMP_FALSE)
              return ReplaceInstUsesWith(I, Op1);
            else if (Op1CC == FCmpInst::FCMP_FALSE)
              return ReplaceInstUsesWith(I, Op0);
            bool Op0Ordered;
            bool Op1Ordered;
            unsigned Op0Pred = getFCmpCode(Op0CC, Op0Ordered);
            unsigned Op1Pred = getFCmpCode(Op1CC, Op1Ordered);
            if (Op0Ordered == Op1Ordered) {
              // If both are ordered or unordered, return a new fcmp with
              // or'ed predicates.
              Value *RV = getFCmpValue(Op0Ordered, Op0Pred|Op1Pred,
                                       Op0LHS, Op0RHS);
              if (Instruction *I = dyn_cast<Instruction>(RV))
                return I;
              // Otherwise, it's a constant boolean value...
              return ReplaceInstUsesWith(I, RV);
            }
          }
        }
      }
    }
  }

  return Changed ? &I : 0;
}

namespace {

// XorSelf - Implements: X ^ X --> 0
struct XorSelf {
  Value *RHS;
  XorSelf(Value *rhs) : RHS(rhs) {}
  bool shouldApply(Value *LHS) const { return LHS == RHS; }
  Instruction *apply(BinaryOperator &Xor) const {
    return &Xor;
  }
};

}

Instruction *InstCombiner::visitXor(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op1)) {
    if (isa<UndefValue>(Op0))
      // Handle undef ^ undef -> 0 special case. This is a common
      // idiom (misuse).
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
    return ReplaceInstUsesWith(I, Op1);  // X ^ undef -> undef
  }

  // xor X, X = 0, even if X is nested in a sequence of Xor's.
  if (Instruction *Result = AssociativeOpt(I, XorSelf(Op1))) {
    assert(Result == &I && "AssociativeOpt didn't work?"); Result=Result;
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }
  
  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  if (!isa<VectorType>(I.getType())) {
    if (SimplifyDemandedInstructionBits(I))
      return &I;
  } else if (isa<ConstantAggregateZero>(Op1)) {
    return ReplaceInstUsesWith(I, Op0);  // X ^ <0,0> -> X
  }

  // Is this a ~ operation?
  if (Value *NotOp = dyn_castNotVal(&I)) {
    // ~(~X & Y) --> (X | ~Y) - De Morgan's Law
    // ~(~X | Y) === (X & ~Y) - De Morgan's Law
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(NotOp)) {
      if (Op0I->getOpcode() == Instruction::And || 
          Op0I->getOpcode() == Instruction::Or) {
        if (dyn_castNotVal(Op0I->getOperand(1))) Op0I->swapOperands();
        if (Value *Op0NotVal = dyn_castNotVal(Op0I->getOperand(0))) {
          Instruction *NotY =
            BinaryOperator::CreateNot(Op0I->getOperand(1),
                                      Op0I->getOperand(1)->getName()+".not");
          InsertNewInstBefore(NotY, I);
          if (Op0I->getOpcode() == Instruction::And)
            return BinaryOperator::CreateOr(Op0NotVal, NotY);
          else
            return BinaryOperator::CreateAnd(Op0NotVal, NotY);
        }
      }
    }
  }
  
  
  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    if (RHS == ConstantInt::getTrue() && Op0->hasOneUse()) {
      // xor (cmp A, B), true = not (cmp A, B) = !cmp A, B
      if (ICmpInst *ICI = dyn_cast<ICmpInst>(Op0))
        return new ICmpInst(ICI->getInversePredicate(),
                            ICI->getOperand(0), ICI->getOperand(1));

      if (FCmpInst *FCI = dyn_cast<FCmpInst>(Op0))
        return new FCmpInst(FCI->getInversePredicate(),
                            FCI->getOperand(0), FCI->getOperand(1));
    }

    // fold (xor(zext(cmp)), 1) and (xor(sext(cmp)), -1) to ext(!cmp).
    if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
      if (CmpInst *CI = dyn_cast<CmpInst>(Op0C->getOperand(0))) {
        if (CI->hasOneUse() && Op0C->hasOneUse()) {
          Instruction::CastOps Opcode = Op0C->getOpcode();
          if (Opcode == Instruction::ZExt || Opcode == Instruction::SExt) {
            if (RHS == ConstantExpr::getCast(Opcode, ConstantInt::getTrue(),
                                             Op0C->getDestTy())) {
              Instruction *NewCI = InsertNewInstBefore(CmpInst::Create(
                                     CI->getOpcode(), CI->getInversePredicate(),
                                     CI->getOperand(0), CI->getOperand(1)), I);
              NewCI->takeName(CI);
              return CastInst::Create(Opcode, NewCI, Op0C->getType());
            }
          }
        }
      }
    }

    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      // ~(c-X) == X-c-1 == X+(-c-1)
      if (Op0I->getOpcode() == Instruction::Sub && RHS->isAllOnesValue())
        if (Constant *Op0I0C = dyn_cast<Constant>(Op0I->getOperand(0))) {
          Constant *NegOp0I0C = ConstantExpr::getNeg(Op0I0C);
          Constant *ConstantRHS = ConstantExpr::getSub(NegOp0I0C,
                                              ConstantInt::get(I.getType(), 1));
          return BinaryOperator::CreateAdd(Op0I->getOperand(1), ConstantRHS);
        }
          
      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1))) {
        if (Op0I->getOpcode() == Instruction::Add) {
          // ~(X-c) --> (-c-1)-X
          if (RHS->isAllOnesValue()) {
            Constant *NegOp0CI = ConstantExpr::getNeg(Op0CI);
            return BinaryOperator::CreateSub(
                           ConstantExpr::getSub(NegOp0CI,
                                             ConstantInt::get(I.getType(), 1)),
                                          Op0I->getOperand(0));
          } else if (RHS->getValue().isSignBit()) {
            // (X + C) ^ signbit -> (X + C + signbit)
            Constant *C = ConstantInt::get(RHS->getValue() + Op0CI->getValue());
            return BinaryOperator::CreateAdd(Op0I->getOperand(0), C);

          }
        } else if (Op0I->getOpcode() == Instruction::Or) {
          // (X|C1)^C2 -> X^(C1|C2) iff X&~C1 == 0
          if (MaskedValueIsZero(Op0I->getOperand(0), Op0CI->getValue())) {
            Constant *NewRHS = ConstantExpr::getOr(Op0CI, RHS);
            // Anything in both C1 and C2 is known to be zero, remove it from
            // NewRHS.
            Constant *CommonBits = And(Op0CI, RHS);
            NewRHS = ConstantExpr::getAnd(NewRHS, 
                                          ConstantExpr::getNot(CommonBits));
            AddToWorkList(Op0I);
            I.setOperand(0, Op0I->getOperand(0));
            I.setOperand(1, NewRHS);
            return &I;
          }
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
      return ReplaceInstUsesWith(I, Constant::getAllOnesValue(I.getType()));

  if (Value *X = dyn_castNotVal(Op1))   // A ^ ~A == -1
    if (X == Op0)
      return ReplaceInstUsesWith(I, Constant::getAllOnesValue(I.getType()));

  
  BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1);
  if (Op1I) {
    Value *A, *B;
    if (match(Op1I, m_Or(m_Value(A), m_Value(B)))) {
      if (A == Op0) {              // B^(B|A) == (A|B)^B
        Op1I->swapOperands();
        I.swapOperands();
        std::swap(Op0, Op1);
      } else if (B == Op0) {       // B^(A|B) == (A|B)^B
        I.swapOperands();     // Simplified below.
        std::swap(Op0, Op1);
      }
    } else if (match(Op1I, m_Xor(m_Specific(Op0), m_Value(B)))) {
      return ReplaceInstUsesWith(I, B);                      // A^(A^B) == B
    } else if (match(Op1I, m_Xor(m_Value(A), m_Specific(Op0)))) {
      return ReplaceInstUsesWith(I, A);                      // A^(B^A) == B
    } else if (match(Op1I, m_And(m_Value(A), m_Value(B))) && Op1I->hasOneUse()){
      if (A == Op0) {                                      // A^(A&B) -> A^(B&A)
        Op1I->swapOperands();
        std::swap(A, B);
      }
      if (B == Op0) {                                      // A^(B&A) -> (B&A)^A
        I.swapOperands();     // Simplified below.
        std::swap(Op0, Op1);
      }
    }
  }
  
  BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0);
  if (Op0I) {
    Value *A, *B;
    if (match(Op0I, m_Or(m_Value(A), m_Value(B))) && Op0I->hasOneUse()) {
      if (A == Op1)                                  // (B|A)^B == (A|B)^B
        std::swap(A, B);
      if (B == Op1) {                                // (A|B)^B == A & ~B
        Instruction *NotB =
          InsertNewInstBefore(BinaryOperator::CreateNot(Op1, "tmp"), I);
        return BinaryOperator::CreateAnd(A, NotB);
      }
    } else if (match(Op0I, m_Xor(m_Specific(Op1), m_Value(B)))) {
      return ReplaceInstUsesWith(I, B);                      // (A^B)^A == B
    } else if (match(Op0I, m_Xor(m_Value(A), m_Specific(Op1)))) {
      return ReplaceInstUsesWith(I, A);                      // (B^A)^A == B
    } else if (match(Op0I, m_And(m_Value(A), m_Value(B))) && Op0I->hasOneUse()){
      if (A == Op1)                                        // (A&B)^A -> (B&A)^A
        std::swap(A, B);
      if (B == Op1 &&                                      // (B&A)^A == ~B & A
          !isa<ConstantInt>(Op1)) {  // Canonical form is (B&C)^C
        Instruction *N =
          InsertNewInstBefore(BinaryOperator::CreateNot(A, "tmp"), I);
        return BinaryOperator::CreateAnd(N, Op1);
      }
    }
  }
  
  // (X >> Z) ^ (Y >> Z)  -> (X^Y) >> Z  for all shifts.
  if (Op0I && Op1I && Op0I->isShift() && 
      Op0I->getOpcode() == Op1I->getOpcode() && 
      Op0I->getOperand(1) == Op1I->getOperand(1) &&
      (Op1I->hasOneUse() || Op1I->hasOneUse())) {
    Instruction *NewOp =
      InsertNewInstBefore(BinaryOperator::CreateXor(Op0I->getOperand(0),
                                                    Op1I->getOperand(0),
                                                    Op0I->getName()), I);
    return BinaryOperator::Create(Op1I->getOpcode(), NewOp, 
                                  Op1I->getOperand(1));
  }
    
  if (Op0I && Op1I) {
    Value *A, *B, *C, *D;
    // (A & B)^(A | B) -> A ^ B
    if (match(Op0I, m_And(m_Value(A), m_Value(B))) &&
        match(Op1I, m_Or(m_Value(C), m_Value(D)))) {
      if ((A == C && B == D) || (A == D && B == C)) 
        return BinaryOperator::CreateXor(A, B);
    }
    // (A | B)^(A & B) -> A ^ B
    if (match(Op0I, m_Or(m_Value(A), m_Value(B))) &&
        match(Op1I, m_And(m_Value(C), m_Value(D)))) {
      if ((A == C && B == D) || (A == D && B == C)) 
        return BinaryOperator::CreateXor(A, B);
    }
    
    // (A & B)^(C & D)
    if ((Op0I->hasOneUse() || Op1I->hasOneUse()) &&
        match(Op0I, m_And(m_Value(A), m_Value(B))) &&
        match(Op1I, m_And(m_Value(C), m_Value(D)))) {
      // (X & Y)^(X & Y) -> (Y^Z) & X
      Value *X = 0, *Y = 0, *Z = 0;
      if (A == C)
        X = A, Y = B, Z = D;
      else if (A == D)
        X = A, Y = B, Z = C;
      else if (B == C)
        X = B, Y = A, Z = D;
      else if (B == D)
        X = B, Y = A, Z = C;
      
      if (X) {
        Instruction *NewOp =
        InsertNewInstBefore(BinaryOperator::CreateXor(Y, Z, Op0->getName()), I);
        return BinaryOperator::CreateAnd(NewOp, X);
      }
    }
  }
    
  // (icmp1 A, B) ^ (icmp2 A, B) --> (icmp3 A, B)
  if (ICmpInst *RHS = dyn_cast<ICmpInst>(I.getOperand(1)))
    if (Instruction *R = AssociativeOpt(I, FoldICmpLogical(*this, RHS)))
      return R;

  // fold (xor (cast A), (cast B)) -> (cast (xor A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1))
      if (Op0C->getOpcode() == Op1C->getOpcode()) { // same cast kind?
        const Type *SrcTy = Op0C->getOperand(0)->getType();
        if (SrcTy == Op1C->getOperand(0)->getType() && SrcTy->isInteger() &&
            // Only do this if the casts both really cause code to be generated.
            ValueRequiresCast(Op0C->getOpcode(), Op0C->getOperand(0), 
                              I.getType(), TD) &&
            ValueRequiresCast(Op1C->getOpcode(), Op1C->getOperand(0), 
                              I.getType(), TD)) {
          Instruction *NewOp = BinaryOperator::CreateXor(Op0C->getOperand(0),
                                                         Op1C->getOperand(0),
                                                         I.getName());
          InsertNewInstBefore(NewOp, I);
          return CastInst::Create(Op0C->getOpcode(), NewOp, I.getType());
        }
      }
  }

  return Changed ? &I : 0;
}

/// AddWithOverflow - Compute Result = In1+In2, returning true if the result
/// overflowed for this type.
static bool AddWithOverflow(ConstantInt *&Result, ConstantInt *In1,
                            ConstantInt *In2, bool IsSigned = false) {
  Result = cast<ConstantInt>(Add(In1, In2));

  if (IsSigned)
    if (In2->getValue().isNegative())
      return Result->getValue().sgt(In1->getValue());
    else
      return Result->getValue().slt(In1->getValue());
  else
    return Result->getValue().ult(In1->getValue());
}

/// SubWithOverflow - Compute Result = In1-In2, returning true if the result
/// overflowed for this type.
static bool SubWithOverflow(ConstantInt *&Result, ConstantInt *In1,
                            ConstantInt *In2, bool IsSigned = false) {
  Result = cast<ConstantInt>(Subtract(In1, In2));

  if (IsSigned)
    if (In2->getValue().isNegative())
      return Result->getValue().slt(In1->getValue());
    else
      return Result->getValue().sgt(In1->getValue());
  else
    return Result->getValue().ugt(In1->getValue());
}

/// EmitGEPOffset - Given a getelementptr instruction/constantexpr, emit the
/// code necessary to compute the offset from the base pointer (without adding
/// in the base pointer).  Return the result as a signed integer of intptr size.
static Value *EmitGEPOffset(User *GEP, Instruction &I, InstCombiner &IC) {
  TargetData &TD = IC.getTargetData();
  gep_type_iterator GTI = gep_type_begin(GEP);
  const Type *IntPtrTy = TD.getIntPtrType();
  Value *Result = Constant::getNullValue(IntPtrTy);

  // Build a mask for high order bits.
  unsigned IntPtrWidth = TD.getPointerSizeInBits();
  uint64_t PtrSizeMask = ~0ULL >> (64-IntPtrWidth);

  for (User::op_iterator i = GEP->op_begin() + 1, e = GEP->op_end(); i != e;
       ++i, ++GTI) {
    Value *Op = *i;
    uint64_t Size = TD.getTypePaddedSize(GTI.getIndexedType()) & PtrSizeMask;
    if (ConstantInt *OpC = dyn_cast<ConstantInt>(Op)) {
      if (OpC->isZero()) continue;
      
      // Handle a struct index, which adds its field offset to the pointer.
      if (const StructType *STy = dyn_cast<StructType>(*GTI)) {
        Size = TD.getStructLayout(STy)->getElementOffset(OpC->getZExtValue());
        
        if (ConstantInt *RC = dyn_cast<ConstantInt>(Result))
          Result = ConstantInt::get(RC->getValue() + APInt(IntPtrWidth, Size));
        else
          Result = IC.InsertNewInstBefore(
                   BinaryOperator::CreateAdd(Result,
                                             ConstantInt::get(IntPtrTy, Size),
                                             GEP->getName()+".offs"), I);
        continue;
      }
      
      Constant *Scale = ConstantInt::get(IntPtrTy, Size);
      Constant *OC = ConstantExpr::getIntegerCast(OpC, IntPtrTy, true /*SExt*/);
      Scale = ConstantExpr::getMul(OC, Scale);
      if (Constant *RC = dyn_cast<Constant>(Result))
        Result = ConstantExpr::getAdd(RC, Scale);
      else {
        // Emit an add instruction.
        Result = IC.InsertNewInstBefore(
           BinaryOperator::CreateAdd(Result, Scale,
                                     GEP->getName()+".offs"), I);
      }
      continue;
    }
    // Convert to correct type.
    if (Op->getType() != IntPtrTy) {
      if (Constant *OpC = dyn_cast<Constant>(Op))
        Op = ConstantExpr::getSExt(OpC, IntPtrTy);
      else
        Op = IC.InsertNewInstBefore(new SExtInst(Op, IntPtrTy,
                                                 Op->getName()+".c"), I);
    }
    if (Size != 1) {
      Constant *Scale = ConstantInt::get(IntPtrTy, Size);
      if (Constant *OpC = dyn_cast<Constant>(Op))
        Op = ConstantExpr::getMul(OpC, Scale);
      else    // We'll let instcombine(mul) convert this to a shl if possible.
        Op = IC.InsertNewInstBefore(BinaryOperator::CreateMul(Op, Scale,
                                                  GEP->getName()+".idx"), I);
    }

    // Emit an add instruction.
    if (isa<Constant>(Op) && isa<Constant>(Result))
      Result = ConstantExpr::getAdd(cast<Constant>(Op),
                                    cast<Constant>(Result));
    else
      Result = IC.InsertNewInstBefore(BinaryOperator::CreateAdd(Op, Result,
                                                  GEP->getName()+".offs"), I);
  }
  return Result;
}


/// EvaluateGEPOffsetExpression - Return an value that can be used to compare of
/// the *offset* implied by GEP to zero.  For example, if we have &A[i], we want
/// to return 'i' for "icmp ne i, 0".  Note that, in general, indices can be
/// complex, and scales are involved.  The above expression would also be legal
/// to codegen as "icmp ne (i*4), 0" (assuming A is a pointer to i32).  This
/// later form is less amenable to optimization though, and we are allowed to
/// generate the first by knowing that pointer arithmetic doesn't overflow.
///
/// If we can't emit an optimized form for this expression, this returns null.
/// 
static Value *EvaluateGEPOffsetExpression(User *GEP, Instruction &I,
                                          InstCombiner &IC) {
  TargetData &TD = IC.getTargetData();
  gep_type_iterator GTI = gep_type_begin(GEP);

  // Check to see if this gep only has a single variable index.  If so, and if
  // any constant indices are a multiple of its scale, then we can compute this
  // in terms of the scale of the variable index.  For example, if the GEP
  // implies an offset of "12 + i*4", then we can codegen this as "3 + i",
  // because the expression will cross zero at the same point.
  unsigned i, e = GEP->getNumOperands();
  int64_t Offset = 0;
  for (i = 1; i != e; ++i, ++GTI) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(i))) {
      // Compute the aggregate offset of constant indices.
      if (CI->isZero()) continue;

      // Handle a struct index, which adds its field offset to the pointer.
      if (const StructType *STy = dyn_cast<StructType>(*GTI)) {
        Offset += TD.getStructLayout(STy)->getElementOffset(CI->getZExtValue());
      } else {
        uint64_t Size = TD.getTypePaddedSize(GTI.getIndexedType());
        Offset += Size*CI->getSExtValue();
      }
    } else {
      // Found our variable index.
      break;
    }
  }
  
  // If there are no variable indices, we must have a constant offset, just
  // evaluate it the general way.
  if (i == e) return 0;
  
  Value *VariableIdx = GEP->getOperand(i);
  // Determine the scale factor of the variable element.  For example, this is
  // 4 if the variable index is into an array of i32.
  uint64_t VariableScale = TD.getTypePaddedSize(GTI.getIndexedType());
  
  // Verify that there are no other variable indices.  If so, emit the hard way.
  for (++i, ++GTI; i != e; ++i, ++GTI) {
    ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(i));
    if (!CI) return 0;
   
    // Compute the aggregate offset of constant indices.
    if (CI->isZero()) continue;
    
    // Handle a struct index, which adds its field offset to the pointer.
    if (const StructType *STy = dyn_cast<StructType>(*GTI)) {
      Offset += TD.getStructLayout(STy)->getElementOffset(CI->getZExtValue());
    } else {
      uint64_t Size = TD.getTypePaddedSize(GTI.getIndexedType());
      Offset += Size*CI->getSExtValue();
    }
  }
  
  // Okay, we know we have a single variable index, which must be a
  // pointer/array/vector index.  If there is no offset, life is simple, return
  // the index.
  unsigned IntPtrWidth = TD.getPointerSizeInBits();
  if (Offset == 0) {
    // Cast to intptrty in case a truncation occurs.  If an extension is needed,
    // we don't need to bother extending: the extension won't affect where the
    // computation crosses zero.
    if (VariableIdx->getType()->getPrimitiveSizeInBits() > IntPtrWidth)
      VariableIdx = new TruncInst(VariableIdx, TD.getIntPtrType(),
                                  VariableIdx->getNameStart(), &I);
    return VariableIdx;
  }
  
  // Otherwise, there is an index.  The computation we will do will be modulo
  // the pointer size, so get it.
  uint64_t PtrSizeMask = ~0ULL >> (64-IntPtrWidth);
  
  Offset &= PtrSizeMask;
  VariableScale &= PtrSizeMask;

  // To do this transformation, any constant index must be a multiple of the
  // variable scale factor.  For example, we can evaluate "12 + 4*i" as "3 + i",
  // but we can't evaluate "10 + 3*i" in terms of i.  Check that the offset is a
  // multiple of the variable scale.
  int64_t NewOffs = Offset / (int64_t)VariableScale;
  if (Offset != NewOffs*(int64_t)VariableScale)
    return 0;

  // Okay, we can do this evaluation.  Start by converting the index to intptr.
  const Type *IntPtrTy = TD.getIntPtrType();
  if (VariableIdx->getType() != IntPtrTy)
    VariableIdx = CastInst::CreateIntegerCast(VariableIdx, IntPtrTy,
                                              true /*SExt*/, 
                                              VariableIdx->getNameStart(), &I);
  Constant *OffsetVal = ConstantInt::get(IntPtrTy, NewOffs);
  return BinaryOperator::CreateAdd(VariableIdx, OffsetVal, "offset", &I);
}


/// FoldGEPICmp - Fold comparisons between a GEP instruction and something
/// else.  At this point we know that the GEP is on the LHS of the comparison.
Instruction *InstCombiner::FoldGEPICmp(User *GEPLHS, Value *RHS,
                                       ICmpInst::Predicate Cond,
                                       Instruction &I) {
  assert(dyn_castGetElementPtr(GEPLHS) && "LHS is not a getelementptr!");

  // Look through bitcasts.
  if (BitCastInst *BCI = dyn_cast<BitCastInst>(RHS))
    RHS = BCI->getOperand(0);

  Value *PtrBase = GEPLHS->getOperand(0);
  if (PtrBase == RHS) {
    // ((gep Ptr, OFFSET) cmp Ptr)   ---> (OFFSET cmp 0).
    // This transformation (ignoring the base and scales) is valid because we
    // know pointers can't overflow.  See if we can output an optimized form.
    Value *Offset = EvaluateGEPOffsetExpression(GEPLHS, I, *this);
    
    // If not, synthesize the offset the hard way.
    if (Offset == 0)
      Offset = EmitGEPOffset(GEPLHS, I, *this);
    return new ICmpInst(ICmpInst::getSignedPredicate(Cond), Offset,
                        Constant::getNullValue(Offset->getType()));
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
        return new ICmpInst(ICmpInst::getSignedPredicate(Cond), 
                            GEPLHS->getOperand(0), GEPRHS->getOperand(0));

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
      return FoldGEPICmp(GEPRHS, GEPLHS->getOperand(0),
                          ICmpInst::getSwappedPredicate(Cond), I);

    // If the other GEP has all zero indices, recurse.
    AllZeros = true;
    for (unsigned i = 1, e = GEPRHS->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEPRHS->getOperand(i)) ||
          !cast<Constant>(GEPRHS->getOperand(i))->isNullValue()) {
        AllZeros = false;
        break;
      }
    if (AllZeros)
      return FoldGEPICmp(GEPLHS, GEPRHS->getOperand(0), Cond, I);

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
                                   ConstantInt::get(Type::Int1Ty,
                                             ICmpInst::isTrueWhenEqual(Cond)));

      else if (NumDifferences == 1) {
        Value *LHSV = GEPLHS->getOperand(DiffOperand);
        Value *RHSV = GEPRHS->getOperand(DiffOperand);
        // Make sure we do a signed comparison here.
        return new ICmpInst(ICmpInst::getSignedPredicate(Cond), LHSV, RHSV);
      }
    }

    // Only lower this if the icmp is the only user of the GEP or if we expect
    // the result to fold to a constant!
    if ((isa<ConstantExpr>(GEPLHS) || GEPLHS->hasOneUse()) &&
        (isa<ConstantExpr>(GEPRHS) || GEPRHS->hasOneUse())) {
      // ((gep Ptr, OFFSET1) cmp (gep Ptr, OFFSET2)  --->  (OFFSET1 cmp OFFSET2)
      Value *L = EmitGEPOffset(GEPLHS, I, *this);
      Value *R = EmitGEPOffset(GEPRHS, I, *this);
      return new ICmpInst(ICmpInst::getSignedPredicate(Cond), L, R);
    }
  }
  return 0;
}

/// FoldFCmp_IntToFP_Cst - Fold fcmp ([us]itofp x, cst) if possible.
///
Instruction *InstCombiner::FoldFCmp_IntToFP_Cst(FCmpInst &I,
                                                Instruction *LHSI,
                                                Constant *RHSC) {
  if (!isa<ConstantFP>(RHSC)) return 0;
  const APFloat &RHS = cast<ConstantFP>(RHSC)->getValueAPF();
  
  // Get the width of the mantissa.  We don't want to hack on conversions that
  // might lose information from the integer, e.g. "i64 -> float"
  int MantissaWidth = LHSI->getType()->getFPMantissaWidth();
  if (MantissaWidth == -1) return 0;  // Unknown.
  
  // Check to see that the input is converted from an integer type that is small
  // enough that preserves all bits.  TODO: check here for "known" sign bits.
  // This would allow us to handle (fptosi (x >>s 62) to float) if x is i64 f.e.
  unsigned InputSize = LHSI->getOperand(0)->getType()->getPrimitiveSizeInBits();
  
  // If this is a uitofp instruction, we need an extra bit to hold the sign.
  bool LHSUnsigned = isa<UIToFPInst>(LHSI);
  if (LHSUnsigned)
    ++InputSize;
  
  // If the conversion would lose info, don't hack on this.
  if ((int)InputSize > MantissaWidth)
    return 0;
  
  // Otherwise, we can potentially simplify the comparison.  We know that it
  // will always come through as an integer value and we know the constant is
  // not a NAN (it would have been previously simplified).
  assert(!RHS.isNaN() && "NaN comparison not already folded!");
  
  ICmpInst::Predicate Pred;
  switch (I.getPredicate()) {
  default: assert(0 && "Unexpected predicate!");
  case FCmpInst::FCMP_UEQ:
  case FCmpInst::FCMP_OEQ:
    Pred = ICmpInst::ICMP_EQ;
    break;
  case FCmpInst::FCMP_UGT:
  case FCmpInst::FCMP_OGT:
    Pred = LHSUnsigned ? ICmpInst::ICMP_UGT : ICmpInst::ICMP_SGT;
    break;
  case FCmpInst::FCMP_UGE:
  case FCmpInst::FCMP_OGE:
    Pred = LHSUnsigned ? ICmpInst::ICMP_UGE : ICmpInst::ICMP_SGE;
    break;
  case FCmpInst::FCMP_ULT:
  case FCmpInst::FCMP_OLT:
    Pred = LHSUnsigned ? ICmpInst::ICMP_ULT : ICmpInst::ICMP_SLT;
    break;
  case FCmpInst::FCMP_ULE:
  case FCmpInst::FCMP_OLE:
    Pred = LHSUnsigned ? ICmpInst::ICMP_ULE : ICmpInst::ICMP_SLE;
    break;
  case FCmpInst::FCMP_UNE:
  case FCmpInst::FCMP_ONE:
    Pred = ICmpInst::ICMP_NE;
    break;
  case FCmpInst::FCMP_ORD:
    return ReplaceInstUsesWith(I, ConstantInt::getTrue());
  case FCmpInst::FCMP_UNO:
    return ReplaceInstUsesWith(I, ConstantInt::getFalse());
  }
  
  const IntegerType *IntTy = cast<IntegerType>(LHSI->getOperand(0)->getType());
  
  // Now we know that the APFloat is a normal number, zero or inf.
  
  // See if the FP constant is too large for the integer.  For example,
  // comparing an i8 to 300.0.
  unsigned IntWidth = IntTy->getPrimitiveSizeInBits();
  
  if (!LHSUnsigned) {
    // If the RHS value is > SignedMax, fold the comparison.  This handles +INF
    // and large values.
    APFloat SMax(RHS.getSemantics(), APFloat::fcZero, false);
    SMax.convertFromAPInt(APInt::getSignedMaxValue(IntWidth), true,
                          APFloat::rmNearestTiesToEven);
    if (SMax.compare(RHS) == APFloat::cmpLessThan) {  // smax < 13123.0
      if (Pred == ICmpInst::ICMP_NE  || Pred == ICmpInst::ICMP_SLT ||
          Pred == ICmpInst::ICMP_SLE)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      return ReplaceInstUsesWith(I, ConstantInt::getFalse());
    }
  } else {
    // If the RHS value is > UnsignedMax, fold the comparison. This handles
    // +INF and large values.
    APFloat UMax(RHS.getSemantics(), APFloat::fcZero, false);
    UMax.convertFromAPInt(APInt::getMaxValue(IntWidth), false,
                          APFloat::rmNearestTiesToEven);
    if (UMax.compare(RHS) == APFloat::cmpLessThan) {  // umax < 13123.0
      if (Pred == ICmpInst::ICMP_NE  || Pred == ICmpInst::ICMP_ULT ||
          Pred == ICmpInst::ICMP_ULE)
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      return ReplaceInstUsesWith(I, ConstantInt::getFalse());
    }
  }
  
  if (!LHSUnsigned) {
    // See if the RHS value is < SignedMin.
    APFloat SMin(RHS.getSemantics(), APFloat::fcZero, false);
    SMin.convertFromAPInt(APInt::getSignedMinValue(IntWidth), true,
                          APFloat::rmNearestTiesToEven);
    if (SMin.compare(RHS) == APFloat::cmpGreaterThan) { // smin > 12312.0
      if (Pred == ICmpInst::ICMP_NE || Pred == ICmpInst::ICMP_SGT ||
          Pred == ICmpInst::ICMP_SGE)
        return ReplaceInstUsesWith(I,ConstantInt::getTrue());
      return ReplaceInstUsesWith(I, ConstantInt::getFalse());
    }
  }

  // Okay, now we know that the FP constant fits in the range [SMIN, SMAX] or
  // [0, UMAX], but it may still be fractional.  See if it is fractional by
  // casting the FP value to the integer value and back, checking for equality.
  // Don't do this for zero, because -0.0 is not fractional.
  Constant *RHSInt = ConstantExpr::getFPToSI(RHSC, IntTy);
  if (!RHS.isZero() &&
      ConstantExpr::getSIToFP(RHSInt, RHSC->getType()) != RHSC) {
    // If we had a comparison against a fractional value, we have to adjust the
    // compare predicate and sometimes the value.  RHSC is rounded towards zero
    // at this point.
    switch (Pred) {
    default: assert(0 && "Unexpected integer comparison!");
    case ICmpInst::ICMP_NE:  // (float)int != 4.4   --> true
      return ReplaceInstUsesWith(I, ConstantInt::getTrue());
    case ICmpInst::ICMP_EQ:  // (float)int == 4.4   --> false
      return ReplaceInstUsesWith(I, ConstantInt::getFalse());
    case ICmpInst::ICMP_ULE:
      // (float)int <= 4.4   --> int <= 4
      // (float)int <= -4.4  --> false
      if (RHS.isNegative())
        return ReplaceInstUsesWith(I, ConstantInt::getFalse());
      break;
    case ICmpInst::ICMP_SLE:
      // (float)int <= 4.4   --> int <= 4
      // (float)int <= -4.4  --> int < -4
      if (RHS.isNegative())
        Pred = ICmpInst::ICMP_SLT;
      break;
    case ICmpInst::ICMP_ULT:
      // (float)int < -4.4   --> false
      // (float)int < 4.4    --> int <= 4
      if (RHS.isNegative())
        return ReplaceInstUsesWith(I, ConstantInt::getFalse());
      Pred = ICmpInst::ICMP_ULE;
      break;
    case ICmpInst::ICMP_SLT:
      // (float)int < -4.4   --> int < -4
      // (float)int < 4.4    --> int <= 4
      if (!RHS.isNegative())
        Pred = ICmpInst::ICMP_SLE;
      break;
    case ICmpInst::ICMP_UGT:
      // (float)int > 4.4    --> int > 4
      // (float)int > -4.4   --> true
      if (RHS.isNegative())
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      break;
    case ICmpInst::ICMP_SGT:
      // (float)int > 4.4    --> int > 4
      // (float)int > -4.4   --> int >= -4
      if (RHS.isNegative())
        Pred = ICmpInst::ICMP_SGE;
      break;
    case ICmpInst::ICMP_UGE:
      // (float)int >= -4.4   --> true
      // (float)int >= 4.4    --> int > 4
      if (!RHS.isNegative())
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      Pred = ICmpInst::ICMP_UGT;
      break;
    case ICmpInst::ICMP_SGE:
      // (float)int >= -4.4   --> int >= -4
      // (float)int >= 4.4    --> int > 4
      if (!RHS.isNegative())
        Pred = ICmpInst::ICMP_SGT;
      break;
    }
  }

  // Lower this FP comparison into an appropriate integer version of the
  // comparison.
  return new ICmpInst(Pred, LHSI->getOperand(0), RHSInt);
}

Instruction *InstCombiner::visitFCmpInst(FCmpInst &I) {
  bool Changed = SimplifyCompare(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Fold trivial predicates.
  if (I.getPredicate() == FCmpInst::FCMP_FALSE)
    return ReplaceInstUsesWith(I, ConstantInt::getFalse());
  if (I.getPredicate() == FCmpInst::FCMP_TRUE)
    return ReplaceInstUsesWith(I, ConstantInt::getTrue());
  
  // Simplify 'fcmp pred X, X'
  if (Op0 == Op1) {
    switch (I.getPredicate()) {
    default: assert(0 && "Unknown predicate!");
    case FCmpInst::FCMP_UEQ:    // True if unordered or equal
    case FCmpInst::FCMP_UGE:    // True if unordered, greater than, or equal
    case FCmpInst::FCMP_ULE:    // True if unordered, less than, or equal
      return ReplaceInstUsesWith(I, ConstantInt::getTrue());
    case FCmpInst::FCMP_OGT:    // True if ordered and greater than
    case FCmpInst::FCMP_OLT:    // True if ordered and less than
    case FCmpInst::FCMP_ONE:    // True if ordered and operands are unequal
      return ReplaceInstUsesWith(I, ConstantInt::getFalse());
      
    case FCmpInst::FCMP_UNO:    // True if unordered: isnan(X) | isnan(Y)
    case FCmpInst::FCMP_ULT:    // True if unordered or less than
    case FCmpInst::FCMP_UGT:    // True if unordered or greater than
    case FCmpInst::FCMP_UNE:    // True if unordered or not equal
      // Canonicalize these to be 'fcmp uno %X, 0.0'.
      I.setPredicate(FCmpInst::FCMP_UNO);
      I.setOperand(1, Constant::getNullValue(Op0->getType()));
      return &I;
      
    case FCmpInst::FCMP_ORD:    // True if ordered (no nans)
    case FCmpInst::FCMP_OEQ:    // True if ordered and equal
    case FCmpInst::FCMP_OGE:    // True if ordered and greater than or equal
    case FCmpInst::FCMP_OLE:    // True if ordered and less than or equal
      // Canonicalize these to be 'fcmp ord %X, 0.0'.
      I.setPredicate(FCmpInst::FCMP_ORD);
      I.setOperand(1, Constant::getNullValue(Op0->getType()));
      return &I;
    }
  }
    
  if (isa<UndefValue>(Op1))                  // fcmp pred X, undef -> undef
    return ReplaceInstUsesWith(I, UndefValue::get(Type::Int1Ty));

  // Handle fcmp with constant RHS
  if (Constant *RHSC = dyn_cast<Constant>(Op1)) {
    // If the constant is a nan, see if we can fold the comparison based on it.
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHSC)) {
      if (CFP->getValueAPF().isNaN()) {
        if (FCmpInst::isOrdered(I.getPredicate()))   // True if ordered and...
          return ReplaceInstUsesWith(I, ConstantInt::getFalse());
        assert(FCmpInst::isUnordered(I.getPredicate()) &&
               "Comparison must be either ordered or unordered!");
        // True if unordered.
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      }
    }
    
    if (Instruction *LHSI = dyn_cast<Instruction>(Op0))
      switch (LHSI->getOpcode()) {
      case Instruction::PHI:
        // Only fold fcmp into the PHI if the phi and fcmp are in the same
        // block.  If in the same block, we're encouraging jump threading.  If
        // not, we are just pessimizing the code by making an i1 phi.
        if (LHSI->getParent() == I.getParent())
          if (Instruction *NV = FoldOpIntoPhi(I))
            return NV;
        break;
      case Instruction::SIToFP:
      case Instruction::UIToFP:
        if (Instruction *NV = FoldFCmp_IntToFP_Cst(I, LHSI, RHSC))
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
            Op1 = ConstantExpr::getCompare(I.getPredicate(), C, RHSC);
            // Insert a new FCmp of the other select operand.
            Op2 = InsertNewInstBefore(new FCmpInst(I.getPredicate(),
                                                      LHSI->getOperand(2), RHSC,
                                                      I.getName()), I);
          } else if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(2))) {
            // Fold the known value into the constant operand.
            Op2 = ConstantExpr::getCompare(I.getPredicate(), C, RHSC);
            // Insert a new FCmp of the other select operand.
            Op1 = InsertNewInstBefore(new FCmpInst(I.getPredicate(),
                                                      LHSI->getOperand(1), RHSC,
                                                      I.getName()), I);
          }
        }

        if (Op1)
          return SelectInst::Create(LHSI->getOperand(0), Op1, Op2);
        break;
      }
  }

  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitICmpInst(ICmpInst &I) {
  bool Changed = SimplifyCompare(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  const Type *Ty = Op0->getType();

  // icmp X, X
  if (Op0 == Op1)
    return ReplaceInstUsesWith(I, ConstantInt::get(Type::Int1Ty, 
                                                   I.isTrueWhenEqual()));

  if (isa<UndefValue>(Op1))                  // X icmp undef -> undef
    return ReplaceInstUsesWith(I, UndefValue::get(Type::Int1Ty));
  
  // icmp <global/alloca*/null>, <global/alloca*/null> - Global/Stack value
  // addresses never equal each other!  We already know that Op0 != Op1.
  if ((isa<GlobalValue>(Op0) || isa<AllocaInst>(Op0) ||
       isa<ConstantPointerNull>(Op0)) &&
      (isa<GlobalValue>(Op1) || isa<AllocaInst>(Op1) ||
       isa<ConstantPointerNull>(Op1)))
    return ReplaceInstUsesWith(I, ConstantInt::get(Type::Int1Ty, 
                                                   !I.isTrueWhenEqual()));

  // icmp's with boolean values can always be turned into bitwise operations
  if (Ty == Type::Int1Ty) {
    switch (I.getPredicate()) {
    default: assert(0 && "Invalid icmp instruction!");
    case ICmpInst::ICMP_EQ: {               // icmp eq i1 A, B -> ~(A^B)
      Instruction *Xor = BinaryOperator::CreateXor(Op0, Op1, I.getName()+"tmp");
      InsertNewInstBefore(Xor, I);
      return BinaryOperator::CreateNot(Xor);
    }
    case ICmpInst::ICMP_NE:                  // icmp eq i1 A, B -> A^B
      return BinaryOperator::CreateXor(Op0, Op1);

    case ICmpInst::ICMP_UGT:
      std::swap(Op0, Op1);                   // Change icmp ugt -> icmp ult
      // FALL THROUGH
    case ICmpInst::ICMP_ULT:{               // icmp ult i1 A, B -> ~A & B
      Instruction *Not = BinaryOperator::CreateNot(Op0, I.getName()+"tmp");
      InsertNewInstBefore(Not, I);
      return BinaryOperator::CreateAnd(Not, Op1);
    }
    case ICmpInst::ICMP_SGT:
      std::swap(Op0, Op1);                   // Change icmp sgt -> icmp slt
      // FALL THROUGH
    case ICmpInst::ICMP_SLT: {               // icmp slt i1 A, B -> A & ~B
      Instruction *Not = BinaryOperator::CreateNot(Op1, I.getName()+"tmp");
      InsertNewInstBefore(Not, I);
      return BinaryOperator::CreateAnd(Not, Op0);
    }
    case ICmpInst::ICMP_UGE:
      std::swap(Op0, Op1);                   // Change icmp uge -> icmp ule
      // FALL THROUGH
    case ICmpInst::ICMP_ULE: {               //  icmp ule i1 A, B -> ~A | B
      Instruction *Not = BinaryOperator::CreateNot(Op0, I.getName()+"tmp");
      InsertNewInstBefore(Not, I);
      return BinaryOperator::CreateOr(Not, Op1);
    }
    case ICmpInst::ICMP_SGE:
      std::swap(Op0, Op1);                   // Change icmp sge -> icmp sle
      // FALL THROUGH
    case ICmpInst::ICMP_SLE: {               //  icmp sle i1 A, B -> A | ~B
      Instruction *Not = BinaryOperator::CreateNot(Op1, I.getName()+"tmp");
      InsertNewInstBefore(Not, I);
      return BinaryOperator::CreateOr(Not, Op0);
    }
    }
  }

  // See if we are doing a comparison with a constant.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    Value *A, *B;
    
    // (icmp ne/eq (sub A B) 0) -> (icmp ne/eq A, B)
    if (I.isEquality() && CI->isNullValue() &&
        match(Op0, m_Sub(m_Value(A), m_Value(B)))) {
      // (icmp cond A B) if cond is equality
      return new ICmpInst(I.getPredicate(), A, B);
    }
    
    // If we have an icmp le or icmp ge instruction, turn it into the
    // appropriate icmp lt or icmp gt instruction.  This allows us to rely on
    // them being folded in the code below.
    switch (I.getPredicate()) {
    default: break;
    case ICmpInst::ICMP_ULE:
      if (CI->isMaxValue(false))                 // A <=u MAX -> TRUE
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      return new ICmpInst(ICmpInst::ICMP_ULT, Op0, AddOne(CI));
    case ICmpInst::ICMP_SLE:
      if (CI->isMaxValue(true))                  // A <=s MAX -> TRUE
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      return new ICmpInst(ICmpInst::ICMP_SLT, Op0, AddOne(CI));
    case ICmpInst::ICMP_UGE:
      if (CI->isMinValue(false))                 // A >=u MIN -> TRUE
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      return new ICmpInst( ICmpInst::ICMP_UGT, Op0, SubOne(CI));
    case ICmpInst::ICMP_SGE:
      if (CI->isMinValue(true))                  // A >=s MIN -> TRUE
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      return new ICmpInst(ICmpInst::ICMP_SGT, Op0, SubOne(CI));
    }
    
    // See if we can fold the comparison based on range information we can get
    // by checking whether bits are known to be zero or one in the input.
    uint32_t BitWidth = cast<IntegerType>(Ty)->getBitWidth();
    APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
    
    // If this comparison is a normal comparison, it demands all
    // bits, if it is a sign bit comparison, it only demands the sign bit.
    bool UnusedBit;
    bool isSignBit = isSignBitCheck(I.getPredicate(), CI, UnusedBit);
    
    if (SimplifyDemandedBits(I.getOperandUse(0), 
                             isSignBit ? APInt::getSignBit(BitWidth)
                                       : APInt::getAllOnesValue(BitWidth),
                             KnownZero, KnownOne, 0))
      return &I;
        
    // Given the known and unknown bits, compute a range that the LHS could be
    // in.  Compute the Min, Max and RHS values based on the known bits. For the
    // EQ and NE we use unsigned values.
    APInt Min(BitWidth, 0), Max(BitWidth, 0);
    if (ICmpInst::isSignedPredicate(I.getPredicate()))
      ComputeSignedMinMaxValuesFromKnownBits(Ty, KnownZero, KnownOne, Min, Max);
    else
      ComputeUnsignedMinMaxValuesFromKnownBits(Ty, KnownZero, KnownOne,Min,Max);
    
    // If Min and Max are known to be the same, then SimplifyDemandedBits
    // figured out that the LHS is a constant.  Just constant fold this now so
    // that code below can assume that Min != Max.
    if (Min == Max)
      return ReplaceInstUsesWith(I, ConstantExpr::getICmp(I.getPredicate(),
                                                          ConstantInt::get(Min),
                                                          CI));
    
    // Based on the range information we know about the LHS, see if we can
    // simplify this comparison.  For example, (x&4) < 8  is always true.
    const APInt &RHSVal = CI->getValue();
    switch (I.getPredicate()) {  // LE/GE have been folded already.
    default: assert(0 && "Unknown icmp opcode!");
    case ICmpInst::ICMP_EQ:
      if (Max.ult(RHSVal) || Min.ugt(RHSVal))
        return ReplaceInstUsesWith(I, ConstantInt::getFalse());
      break;
    case ICmpInst::ICMP_NE:
      if (Max.ult(RHSVal) || Min.ugt(RHSVal))
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      break;
    case ICmpInst::ICMP_ULT:
      if (Max.ult(RHSVal))                    // A <u C -> true iff max(A) < C
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      if (Min.uge(RHSVal))                    // A <u C -> false iff min(A) >= C
        return ReplaceInstUsesWith(I, ConstantInt::getFalse());
      if (RHSVal == Max)                      // A <u MAX -> A != MAX
        return new ICmpInst(ICmpInst::ICMP_NE, Op0, Op1);
      if (RHSVal == Min+1)                    // A <u MIN+1 -> A == MIN
        return new ICmpInst(ICmpInst::ICMP_EQ, Op0, SubOne(CI));
        
      // (x <u 2147483648) -> (x >s -1)  -> true if sign bit clear
      if (CI->isMinValue(true))
        return new ICmpInst(ICmpInst::ICMP_SGT, Op0,
                            ConstantInt::getAllOnesValue(Op0->getType()));
      break;
    case ICmpInst::ICMP_UGT:
      if (Min.ugt(RHSVal))                    // A >u C -> true iff min(A) > C
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      if (Max.ule(RHSVal))                    // A >u C -> false iff max(A) <= C
        return ReplaceInstUsesWith(I, ConstantInt::getFalse());
        
      if (RHSVal == Min)                      // A >u MIN -> A != MIN
        return new ICmpInst(ICmpInst::ICMP_NE, Op0, Op1);
      if (RHSVal == Max-1)                    // A >u MAX-1 -> A == MAX
        return new ICmpInst(ICmpInst::ICMP_EQ, Op0, AddOne(CI));
      
      // (x >u 2147483647) -> (x <s 0)  -> true if sign bit set
      if (CI->isMaxValue(true))
        return new ICmpInst(ICmpInst::ICMP_SLT, Op0,
                            ConstantInt::getNullValue(Op0->getType()));
      break;
    case ICmpInst::ICMP_SLT:
      if (Max.slt(RHSVal))                    // A <s C -> true iff max(A) < C
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      if (Min.sge(RHSVal))                    // A <s C -> false iff min(A) >= C
        return ReplaceInstUsesWith(I, ConstantInt::getFalse());
      if (RHSVal == Max)                      // A <s MAX -> A != MAX
        return new ICmpInst(ICmpInst::ICMP_NE, Op0, Op1);
      if (RHSVal == Min+1)                    // A <s MIN+1 -> A == MIN
        return new ICmpInst(ICmpInst::ICMP_EQ, Op0, SubOne(CI));
      break;
    case ICmpInst::ICMP_SGT: 
      if (Min.sgt(RHSVal))                    // A >s C -> true iff min(A) > C
        return ReplaceInstUsesWith(I, ConstantInt::getTrue());
      if (Max.sle(RHSVal))                    // A >s C -> false iff max(A) <= C
        return ReplaceInstUsesWith(I, ConstantInt::getFalse());
        
      if (RHSVal == Min)                      // A >s MIN -> A != MIN
        return new ICmpInst(ICmpInst::ICMP_NE, Op0, Op1);
      if (RHSVal == Max-1)                    // A >s MAX-1 -> A == MAX
        return new ICmpInst(ICmpInst::ICMP_EQ, Op0, AddOne(CI));
      break;
    }
  }

  // Test if the ICmpInst instruction is used exclusively by a select as
  // part of a minimum or maximum operation. If so, refrain from doing
  // any other folding. This helps out other analyses which understand
  // non-obfuscated minimum and maximum idioms, such as ScalarEvolution
  // and CodeGen. And in this case, at least one of the comparison
  // operands has at least one user besides the compare (the select),
  // which would often largely negate the benefit of folding anyway.
  if (I.hasOneUse())
    if (SelectInst *SI = dyn_cast<SelectInst>(*I.use_begin()))
      if ((SI->getOperand(1) == Op0 && SI->getOperand(2) == Op1) ||
          (SI->getOperand(2) == Op0 && SI->getOperand(1) == Op1))
        return 0;

  // See if we are doing a comparison between a constant and an instruction that
  // can be folded into the comparison.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    // Since the RHS is a ConstantInt (CI), if the left hand side is an 
    // instruction, see if that instruction also has constants so that the 
    // instruction can be folded into the icmp 
    if (Instruction *LHSI = dyn_cast<Instruction>(Op0))
      if (Instruction *Res = visitICmpInstWithInstAndIntCst(I, LHSI, CI))
        return Res;
  }

  // Handle icmp with constant (but not simple integer constant) RHS
  if (Constant *RHSC = dyn_cast<Constant>(Op1)) {
    if (Instruction *LHSI = dyn_cast<Instruction>(Op0))
      switch (LHSI->getOpcode()) {
      case Instruction::GetElementPtr:
        if (RHSC->isNullValue()) {
          // icmp pred GEP (P, int 0, int 0, int 0), null -> icmp pred P, null
          bool isAllZeros = true;
          for (unsigned i = 1, e = LHSI->getNumOperands(); i != e; ++i)
            if (!isa<Constant>(LHSI->getOperand(i)) ||
                !cast<Constant>(LHSI->getOperand(i))->isNullValue()) {
              isAllZeros = false;
              break;
            }
          if (isAllZeros)
            return new ICmpInst(I.getPredicate(), LHSI->getOperand(0),
                    Constant::getNullValue(LHSI->getOperand(0)->getType()));
        }
        break;

      case Instruction::PHI:
        // Only fold icmp into the PHI if the phi and fcmp are in the same
        // block.  If in the same block, we're encouraging jump threading.  If
        // not, we are just pessimizing the code by making an i1 phi.
        if (LHSI->getParent() == I.getParent())
          if (Instruction *NV = FoldOpIntoPhi(I))
            return NV;
        break;
      case Instruction::Select: {
        // If either operand of the select is a constant, we can fold the
        // comparison into the select arms, which will cause one to be
        // constant folded and the select turned into a bitwise or.
        Value *Op1 = 0, *Op2 = 0;
        if (LHSI->hasOneUse()) {
          if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(1))) {
            // Fold the known value into the constant operand.
            Op1 = ConstantExpr::getICmp(I.getPredicate(), C, RHSC);
            // Insert a new ICmp of the other select operand.
            Op2 = InsertNewInstBefore(new ICmpInst(I.getPredicate(),
                                                   LHSI->getOperand(2), RHSC,
                                                   I.getName()), I);
          } else if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(2))) {
            // Fold the known value into the constant operand.
            Op2 = ConstantExpr::getICmp(I.getPredicate(), C, RHSC);
            // Insert a new ICmp of the other select operand.
            Op1 = InsertNewInstBefore(new ICmpInst(I.getPredicate(),
                                                   LHSI->getOperand(1), RHSC,
                                                   I.getName()), I);
          }
        }

        if (Op1)
          return SelectInst::Create(LHSI->getOperand(0), Op1, Op2);
        break;
      }
      case Instruction::Malloc:
        // If we have (malloc != null), and if the malloc has a single use, we
        // can assume it is successful and remove the malloc.
        if (LHSI->hasOneUse() && isa<ConstantPointerNull>(RHSC)) {
          AddToWorkList(LHSI);
          return ReplaceInstUsesWith(I, ConstantInt::get(Type::Int1Ty,
                                                         !I.isTrueWhenEqual()));
        }
        break;
      }
  }

  // If we can optimize a 'icmp GEP, P' or 'icmp P, GEP', do so now.
  if (User *GEP = dyn_castGetElementPtr(Op0))
    if (Instruction *NI = FoldGEPICmp(GEP, Op1, I.getPredicate(), I))
      return NI;
  if (User *GEP = dyn_castGetElementPtr(Op1))
    if (Instruction *NI = FoldGEPICmp(GEP, Op0,
                           ICmpInst::getSwappedPredicate(I.getPredicate()), I))
      return NI;

  // Test to see if the operands of the icmp are casted versions of other
  // values.  If the ptr->ptr cast can be stripped off both arguments, we do so
  // now.
  if (BitCastInst *CI = dyn_cast<BitCastInst>(Op0)) {
    if (isa<PointerType>(Op0->getType()) && 
        (isa<Constant>(Op1) || isa<BitCastInst>(Op1))) { 
      // We keep moving the cast from the left operand over to the right
      // operand, where it can often be eliminated completely.
      Op0 = CI->getOperand(0);

      // If operand #1 is a bitcast instruction, it must also be a ptr->ptr cast
      // so eliminate it as well.
      if (BitCastInst *CI2 = dyn_cast<BitCastInst>(Op1))
        Op1 = CI2->getOperand(0);

      // If Op1 is a constant, we can fold the cast into the constant.
      if (Op0->getType() != Op1->getType()) {
        if (Constant *Op1C = dyn_cast<Constant>(Op1)) {
          Op1 = ConstantExpr::getBitCast(Op1C, Op0->getType());
        } else {
          // Otherwise, cast the RHS right before the icmp
          Op1 = InsertBitCastBefore(Op1, Op0->getType(), I);
        }
      }
      return new ICmpInst(I.getPredicate(), Op0, Op1);
    }
  }
  
  if (isa<CastInst>(Op0)) {
    // Handle the special case of: icmp (cast bool to X), <cst>
    // This comes up when you have code like
    //   int X = A < B;
    //   if (X) ...
    // For generality, we handle any zero-extension of any operand comparison
    // with a constant or another cast from the same type.
    if (isa<ConstantInt>(Op1) || isa<CastInst>(Op1))
      if (Instruction *R = visitICmpInstWithCastAndCast(I))
        return R;
  }
  
  // See if it's the same type of instruction on the left and right.
  if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
    if (BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1)) {
      if (Op0I->getOpcode() == Op1I->getOpcode() && Op0I->hasOneUse() &&
          Op1I->hasOneUse() && Op0I->getOperand(1) == Op1I->getOperand(1)) {
        switch (Op0I->getOpcode()) {
        default: break;
        case Instruction::Add:
        case Instruction::Sub:
        case Instruction::Xor:
          if (I.isEquality())    // a+x icmp eq/ne b+x --> a icmp b
            return new ICmpInst(I.getPredicate(), Op0I->getOperand(0),
                                Op1I->getOperand(0));
          // icmp u/s (a ^ signbit), (b ^ signbit) --> icmp s/u a, b
          if (ConstantInt *CI = dyn_cast<ConstantInt>(Op0I->getOperand(1))) {
            if (CI->getValue().isSignBit()) {
              ICmpInst::Predicate Pred = I.isSignedPredicate()
                                             ? I.getUnsignedPredicate()
                                             : I.getSignedPredicate();
              return new ICmpInst(Pred, Op0I->getOperand(0),
                                  Op1I->getOperand(0));
            }
            
            if (CI->getValue().isMaxSignedValue()) {
              ICmpInst::Predicate Pred = I.isSignedPredicate()
                                             ? I.getUnsignedPredicate()
                                             : I.getSignedPredicate();
              Pred = I.getSwappedPredicate(Pred);
              return new ICmpInst(Pred, Op0I->getOperand(0),
                                  Op1I->getOperand(0));
            }
          }
          break;
        case Instruction::Mul:
          if (!I.isEquality())
            break;

          if (ConstantInt *CI = dyn_cast<ConstantInt>(Op0I->getOperand(1))) {
            // a * Cst icmp eq/ne b * Cst --> a & Mask icmp b & Mask
            // Mask = -1 >> count-trailing-zeros(Cst).
            if (!CI->isZero() && !CI->isOne()) {
              const APInt &AP = CI->getValue();
              ConstantInt *Mask = ConstantInt::get(
                                      APInt::getLowBitsSet(AP.getBitWidth(),
                                                           AP.getBitWidth() -
                                                      AP.countTrailingZeros()));
              Instruction *And1 = BinaryOperator::CreateAnd(Op0I->getOperand(0),
                                                            Mask);
              Instruction *And2 = BinaryOperator::CreateAnd(Op1I->getOperand(0),
                                                            Mask);
              InsertNewInstBefore(And1, I);
              InsertNewInstBefore(And2, I);
              return new ICmpInst(I.getPredicate(), And1, And2);
            }
          }
          break;
        }
      }
    }
  }
  
  // ~x < ~y --> y < x
  { Value *A, *B;
    if (match(Op0, m_Not(m_Value(A))) &&
        match(Op1, m_Not(m_Value(B))))
      return new ICmpInst(I.getPredicate(), B, A);
  }
  
  if (I.isEquality()) {
    Value *A, *B, *C, *D;
    
    // -x == -y --> x == y
    if (match(Op0, m_Neg(m_Value(A))) &&
        match(Op1, m_Neg(m_Value(B))))
      return new ICmpInst(I.getPredicate(), A, B);
    
    if (match(Op0, m_Xor(m_Value(A), m_Value(B)))) {
      if (A == Op1 || B == Op1) {    // (A^B) == A  ->  B == 0
        Value *OtherVal = A == Op1 ? B : A;
        return new ICmpInst(I.getPredicate(), OtherVal,
                            Constant::getNullValue(A->getType()));
      }

      if (match(Op1, m_Xor(m_Value(C), m_Value(D)))) {
        // A^c1 == C^c2 --> A == C^(c1^c2)
        ConstantInt *C1, *C2;
        if (match(B, m_ConstantInt(C1)) &&
            match(D, m_ConstantInt(C2)) && Op1->hasOneUse()) {
          Constant *NC = ConstantInt::get(C1->getValue() ^ C2->getValue());
          Instruction *Xor = BinaryOperator::CreateXor(C, NC, "tmp");
          return new ICmpInst(I.getPredicate(), A,
                              InsertNewInstBefore(Xor, I));
        }
        
        // A^B == A^D -> B == D
        if (A == C) return new ICmpInst(I.getPredicate(), B, D);
        if (A == D) return new ICmpInst(I.getPredicate(), B, C);
        if (B == C) return new ICmpInst(I.getPredicate(), A, D);
        if (B == D) return new ICmpInst(I.getPredicate(), A, C);
      }
    }
    
    if (match(Op1, m_Xor(m_Value(A), m_Value(B))) &&
        (A == Op0 || B == Op0)) {
      // A == (A^B)  ->  B == 0
      Value *OtherVal = A == Op0 ? B : A;
      return new ICmpInst(I.getPredicate(), OtherVal,
                          Constant::getNullValue(A->getType()));
    }

    // (A-B) == A  ->  B == 0
    if (match(Op0, m_Sub(m_Specific(Op1), m_Value(B))))
      return new ICmpInst(I.getPredicate(), B, 
                          Constant::getNullValue(B->getType()));

    // A == (A-B)  ->  B == 0
    if (match(Op1, m_Sub(m_Specific(Op0), m_Value(B))))
      return new ICmpInst(I.getPredicate(), B,
                          Constant::getNullValue(B->getType()));
    
    // (X&Z) == (Y&Z) -> (X^Y) & Z == 0
    if (Op0->hasOneUse() && Op1->hasOneUse() &&
        match(Op0, m_And(m_Value(A), m_Value(B))) && 
        match(Op1, m_And(m_Value(C), m_Value(D)))) {
      Value *X = 0, *Y = 0, *Z = 0;
      
      if (A == C) {
        X = B; Y = D; Z = A;
      } else if (A == D) {
        X = B; Y = C; Z = A;
      } else if (B == C) {
        X = A; Y = D; Z = B;
      } else if (B == D) {
        X = A; Y = C; Z = B;
      }
      
      if (X) {   // Build (X^Y) & Z
        Op1 = InsertNewInstBefore(BinaryOperator::CreateXor(X, Y, "tmp"), I);
        Op1 = InsertNewInstBefore(BinaryOperator::CreateAnd(Op1, Z, "tmp"), I);
        I.setOperand(0, Op1);
        I.setOperand(1, Constant::getNullValue(Op1->getType()));
        return &I;
      }
    }
  }
  return Changed ? &I : 0;
}


/// FoldICmpDivCst - Fold "icmp pred, ([su]div X, DivRHS), CmpRHS" where DivRHS
/// and CmpRHS are both known to be integer constants.
Instruction *InstCombiner::FoldICmpDivCst(ICmpInst &ICI, BinaryOperator *DivI,
                                          ConstantInt *DivRHS) {
  ConstantInt *CmpRHS = cast<ConstantInt>(ICI.getOperand(1));
  const APInt &CmpRHSV = CmpRHS->getValue();
  
  // FIXME: If the operand types don't match the type of the divide 
  // then don't attempt this transform. The code below doesn't have the
  // logic to deal with a signed divide and an unsigned compare (and
  // vice versa). This is because (x /s C1) <s C2  produces different 
  // results than (x /s C1) <u C2 or (x /u C1) <s C2 or even
  // (x /u C1) <u C2.  Simply casting the operands and result won't 
  // work. :(  The if statement below tests that condition and bails 
  // if it finds it. 
  bool DivIsSigned = DivI->getOpcode() == Instruction::SDiv;
  if (!ICI.isEquality() && DivIsSigned != ICI.isSignedPredicate())
    return 0;
  if (DivRHS->isZero())
    return 0; // The ProdOV computation fails on divide by zero.
  if (DivIsSigned && DivRHS->isAllOnesValue())
    return 0; // The overflow computation also screws up here
  if (DivRHS->isOne())
    return 0; // Not worth bothering, and eliminates some funny cases
              // with INT_MIN.

  // Compute Prod = CI * DivRHS. We are essentially solving an equation
  // of form X/C1=C2. We solve for X by multiplying C1 (DivRHS) and 
  // C2 (CI). By solving for X we can turn this into a range check 
  // instead of computing a divide. 
  ConstantInt *Prod = Multiply(CmpRHS, DivRHS);

  // Determine if the product overflows by seeing if the product is
  // not equal to the divide. Make sure we do the same kind of divide
  // as in the LHS instruction that we're folding. 
  bool ProdOV = (DivIsSigned ? ConstantExpr::getSDiv(Prod, DivRHS) :
                 ConstantExpr::getUDiv(Prod, DivRHS)) != CmpRHS;

  // Get the ICmp opcode
  ICmpInst::Predicate Pred = ICI.getPredicate();

  // Figure out the interval that is being checked.  For example, a comparison
  // like "X /u 5 == 0" is really checking that X is in the interval [0, 5). 
  // Compute this interval based on the constants involved and the signedness of
  // the compare/divide.  This computes a half-open interval, keeping track of
  // whether either value in the interval overflows.  After analysis each
  // overflow variable is set to 0 if it's corresponding bound variable is valid
  // -1 if overflowed off the bottom end, or +1 if overflowed off the top end.
  int LoOverflow = 0, HiOverflow = 0;
  ConstantInt *LoBound = 0, *HiBound = 0;
  
  if (!DivIsSigned) {  // udiv
    // e.g. X/5 op 3  --> [15, 20)
    LoBound = Prod;
    HiOverflow = LoOverflow = ProdOV;
    if (!HiOverflow)
      HiOverflow = AddWithOverflow(HiBound, LoBound, DivRHS, false);
  } else if (DivRHS->getValue().isStrictlyPositive()) { // Divisor is > 0.
    if (CmpRHSV == 0) {       // (X / pos) op 0
      // Can't overflow.  e.g.  X/2 op 0 --> [-1, 2)
      LoBound = cast<ConstantInt>(ConstantExpr::getNeg(SubOne(DivRHS)));
      HiBound = DivRHS;
    } else if (CmpRHSV.isStrictlyPositive()) {   // (X / pos) op pos
      LoBound = Prod;     // e.g.   X/5 op 3 --> [15, 20)
      HiOverflow = LoOverflow = ProdOV;
      if (!HiOverflow)
        HiOverflow = AddWithOverflow(HiBound, Prod, DivRHS, true);
    } else {                       // (X / pos) op neg
      // e.g. X/5 op -3  --> [-15-4, -15+1) --> [-19, -14)
      HiBound = AddOne(Prod);
      LoOverflow = HiOverflow = ProdOV ? -1 : 0;
      if (!LoOverflow) {
        ConstantInt* DivNeg = cast<ConstantInt>(ConstantExpr::getNeg(DivRHS));
        LoOverflow = AddWithOverflow(LoBound, HiBound, DivNeg,
                                     true) ? -1 : 0;
       }
    }
  } else if (DivRHS->getValue().isNegative()) { // Divisor is < 0.
    if (CmpRHSV == 0) {       // (X / neg) op 0
      // e.g. X/-5 op 0  --> [-4, 5)
      LoBound = AddOne(DivRHS);
      HiBound = cast<ConstantInt>(ConstantExpr::getNeg(DivRHS));
      if (HiBound == DivRHS) {     // -INTMIN = INTMIN
        HiOverflow = 1;            // [INTMIN+1, overflow)
        HiBound = 0;               // e.g. X/INTMIN = 0 --> X > INTMIN
      }
    } else if (CmpRHSV.isStrictlyPositive()) {   // (X / neg) op pos
      // e.g. X/-5 op 3  --> [-19, -14)
      HiBound = AddOne(Prod);
      HiOverflow = LoOverflow = ProdOV ? -1 : 0;
      if (!LoOverflow)
        LoOverflow = AddWithOverflow(LoBound, HiBound, DivRHS, true) ? -1 : 0;
    } else {                       // (X / neg) op neg
      LoBound = Prod;       // e.g. X/-5 op -3  --> [15, 20)
      LoOverflow = HiOverflow = ProdOV;
      if (!HiOverflow)
        HiOverflow = SubWithOverflow(HiBound, Prod, DivRHS, true);
    }
    
    // Dividing by a negative swaps the condition.  LT <-> GT
    Pred = ICmpInst::getSwappedPredicate(Pred);
  }

  Value *X = DivI->getOperand(0);
  switch (Pred) {
  default: assert(0 && "Unhandled icmp opcode!");
  case ICmpInst::ICMP_EQ:
    if (LoOverflow && HiOverflow)
      return ReplaceInstUsesWith(ICI, ConstantInt::getFalse());
    else if (HiOverflow)
      return new ICmpInst(DivIsSigned ? ICmpInst::ICMP_SGE : 
                          ICmpInst::ICMP_UGE, X, LoBound);
    else if (LoOverflow)
      return new ICmpInst(DivIsSigned ? ICmpInst::ICMP_SLT : 
                          ICmpInst::ICMP_ULT, X, HiBound);
    else
      return InsertRangeTest(X, LoBound, HiBound, DivIsSigned, true, ICI);
  case ICmpInst::ICMP_NE:
    if (LoOverflow && HiOverflow)
      return ReplaceInstUsesWith(ICI, ConstantInt::getTrue());
    else if (HiOverflow)
      return new ICmpInst(DivIsSigned ? ICmpInst::ICMP_SLT : 
                          ICmpInst::ICMP_ULT, X, LoBound);
    else if (LoOverflow)
      return new ICmpInst(DivIsSigned ? ICmpInst::ICMP_SGE : 
                          ICmpInst::ICMP_UGE, X, HiBound);
    else
      return InsertRangeTest(X, LoBound, HiBound, DivIsSigned, false, ICI);
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_SLT:
    if (LoOverflow == +1)   // Low bound is greater than input range.
      return ReplaceInstUsesWith(ICI, ConstantInt::getTrue());
    if (LoOverflow == -1)   // Low bound is less than input range.
      return ReplaceInstUsesWith(ICI, ConstantInt::getFalse());
    return new ICmpInst(Pred, X, LoBound);
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_SGT:
    if (HiOverflow == +1)       // High bound greater than input range.
      return ReplaceInstUsesWith(ICI, ConstantInt::getFalse());
    else if (HiOverflow == -1)  // High bound less than input range.
      return ReplaceInstUsesWith(ICI, ConstantInt::getTrue());
    if (Pred == ICmpInst::ICMP_UGT)
      return new ICmpInst(ICmpInst::ICMP_UGE, X, HiBound);
    else
      return new ICmpInst(ICmpInst::ICMP_SGE, X, HiBound);
  }
}


/// visitICmpInstWithInstAndIntCst - Handle "icmp (instr, intcst)".
///
Instruction *InstCombiner::visitICmpInstWithInstAndIntCst(ICmpInst &ICI,
                                                          Instruction *LHSI,
                                                          ConstantInt *RHS) {
  const APInt &RHSV = RHS->getValue();
  
  switch (LHSI->getOpcode()) {
  case Instruction::Trunc:
    if (ICI.isEquality() && LHSI->hasOneUse()) {
      // Simplify icmp eq (trunc x to i8), 42 -> icmp eq x, 42|highbits if all
      // of the high bits truncated out of x are known.
      unsigned DstBits = LHSI->getType()->getPrimitiveSizeInBits(),
             SrcBits = LHSI->getOperand(0)->getType()->getPrimitiveSizeInBits();
      APInt Mask(APInt::getHighBitsSet(SrcBits, SrcBits-DstBits));
      APInt KnownZero(SrcBits, 0), KnownOne(SrcBits, 0);
      ComputeMaskedBits(LHSI->getOperand(0), Mask, KnownZero, KnownOne);
      
      // If all the high bits are known, we can do this xform.
      if ((KnownZero|KnownOne).countLeadingOnes() >= SrcBits-DstBits) {
        // Pull in the high bits from known-ones set.
        APInt NewRHS(RHS->getValue());
        NewRHS.zext(SrcBits);
        NewRHS |= KnownOne;
        return new ICmpInst(ICI.getPredicate(), LHSI->getOperand(0),
                            ConstantInt::get(NewRHS));
      }
    }
    break;
      
  case Instruction::Xor:         // (icmp pred (xor X, XorCST), CI)
    if (ConstantInt *XorCST = dyn_cast<ConstantInt>(LHSI->getOperand(1))) {
      // If this is a comparison that tests the signbit (X < 0) or (x > -1),
      // fold the xor.
      if ((ICI.getPredicate() == ICmpInst::ICMP_SLT && RHSV == 0) ||
          (ICI.getPredicate() == ICmpInst::ICMP_SGT && RHSV.isAllOnesValue())) {
        Value *CompareVal = LHSI->getOperand(0);
        
        // If the sign bit of the XorCST is not set, there is no change to
        // the operation, just stop using the Xor.
        if (!XorCST->getValue().isNegative()) {
          ICI.setOperand(0, CompareVal);
          AddToWorkList(LHSI);
          return &ICI;
        }
        
        // Was the old condition true if the operand is positive?
        bool isTrueIfPositive = ICI.getPredicate() == ICmpInst::ICMP_SGT;
        
        // If so, the new one isn't.
        isTrueIfPositive ^= true;
        
        if (isTrueIfPositive)
          return new ICmpInst(ICmpInst::ICMP_SGT, CompareVal, SubOne(RHS));
        else
          return new ICmpInst(ICmpInst::ICMP_SLT, CompareVal, AddOne(RHS));
      }

      if (LHSI->hasOneUse()) {
        // (icmp u/s (xor A SignBit), C) -> (icmp s/u A, (xor C SignBit))
        if (!ICI.isEquality() && XorCST->getValue().isSignBit()) {
          const APInt &SignBit = XorCST->getValue();
          ICmpInst::Predicate Pred = ICI.isSignedPredicate()
                                         ? ICI.getUnsignedPredicate()
                                         : ICI.getSignedPredicate();
          return new ICmpInst(Pred, LHSI->getOperand(0),
                              ConstantInt::get(RHSV ^ SignBit));
        }

        // (icmp u/s (xor A ~SignBit), C) -> (icmp s/u (xor C ~SignBit), A)
        if (!ICI.isEquality() && XorCST->getValue().isMaxSignedValue()) {
          const APInt &NotSignBit = XorCST->getValue();
          ICmpInst::Predicate Pred = ICI.isSignedPredicate()
                                         ? ICI.getUnsignedPredicate()
                                         : ICI.getSignedPredicate();
          Pred = ICI.getSwappedPredicate(Pred);
          return new ICmpInst(Pred, LHSI->getOperand(0),
                              ConstantInt::get(RHSV ^ NotSignBit));
        }
      }
    }
    break;
  case Instruction::And:         // (icmp pred (and X, AndCST), RHS)
    if (LHSI->hasOneUse() && isa<ConstantInt>(LHSI->getOperand(1)) &&
        LHSI->getOperand(0)->hasOneUse()) {
      ConstantInt *AndCST = cast<ConstantInt>(LHSI->getOperand(1));
      
      // If the LHS is an AND of a truncating cast, we can widen the
      // and/compare to be the input width without changing the value
      // produced, eliminating a cast.
      if (TruncInst *Cast = dyn_cast<TruncInst>(LHSI->getOperand(0))) {
        // We can do this transformation if either the AND constant does not
        // have its sign bit set or if it is an equality comparison. 
        // Extending a relational comparison when we're checking the sign
        // bit would not work.
        if (Cast->hasOneUse() &&
            (ICI.isEquality() ||
             (AndCST->getValue().isNonNegative() && RHSV.isNonNegative()))) {
          uint32_t BitWidth = 
            cast<IntegerType>(Cast->getOperand(0)->getType())->getBitWidth();
          APInt NewCST = AndCST->getValue();
          NewCST.zext(BitWidth);
          APInt NewCI = RHSV;
          NewCI.zext(BitWidth);
          Instruction *NewAnd = 
            BinaryOperator::CreateAnd(Cast->getOperand(0),
                                      ConstantInt::get(NewCST),LHSI->getName());
          InsertNewInstBefore(NewAnd, ICI);
          return new ICmpInst(ICI.getPredicate(), NewAnd,
                              ConstantInt::get(NewCI));
        }
      }
      
      // If this is: (X >> C1) & C2 != C3 (where any shift and any compare
      // could exist), turn it into (X & (C2 << C1)) != (C3 << C1).  This
      // happens a LOT in code produced by the C front-end, for bitfield
      // access.
      BinaryOperator *Shift = dyn_cast<BinaryOperator>(LHSI->getOperand(0));
      if (Shift && !Shift->isShift())
        Shift = 0;
      
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
          uint32_t TyBits = Ty->getPrimitiveSizeInBits();
          int ShAmtVal = TyBits - ShAmt->getLimitedValue(TyBits);
          
          uint32_t BitWidth = AndTy->getPrimitiveSizeInBits();
          if ((APInt::getHighBitsSet(BitWidth, BitWidth-ShAmtVal) & 
               AndCST->getValue()) == 0)
            CanFold = true;
        }
        
        if (CanFold) {
          Constant *NewCst;
          if (Shift->getOpcode() == Instruction::Shl)
            NewCst = ConstantExpr::getLShr(RHS, ShAmt);
          else
            NewCst = ConstantExpr::getShl(RHS, ShAmt);
          
          // Check to see if we are shifting out any of the bits being
          // compared.
          if (ConstantExpr::get(Shift->getOpcode(), NewCst, ShAmt) != RHS) {
            // If we shifted bits out, the fold is not going to work out.
            // As a special case, check to see if this means that the
            // result is always true or false now.
            if (ICI.getPredicate() == ICmpInst::ICMP_EQ)
              return ReplaceInstUsesWith(ICI, ConstantInt::getFalse());
            if (ICI.getPredicate() == ICmpInst::ICMP_NE)
              return ReplaceInstUsesWith(ICI, ConstantInt::getTrue());
          } else {
            ICI.setOperand(1, NewCst);
            Constant *NewAndCST;
            if (Shift->getOpcode() == Instruction::Shl)
              NewAndCST = ConstantExpr::getLShr(AndCST, ShAmt);
            else
              NewAndCST = ConstantExpr::getShl(AndCST, ShAmt);
            LHSI->setOperand(1, NewAndCST);
            LHSI->setOperand(0, Shift->getOperand(0));
            AddToWorkList(Shift); // Shift is dead.
            AddUsesToWorkList(ICI);
            return &ICI;
          }
        }
      }
      
      // Turn ((X >> Y) & C) == 0  into  (X & (C << Y)) == 0.  The later is
      // preferable because it allows the C<<Y expression to be hoisted out
      // of a loop if Y is invariant and X is not.
      if (Shift && Shift->hasOneUse() && RHSV == 0 &&
          ICI.isEquality() && !Shift->isArithmeticShift() &&
          isa<Instruction>(Shift->getOperand(0))) {
        // Compute C << Y.
        Value *NS;
        if (Shift->getOpcode() == Instruction::LShr) {
          NS = BinaryOperator::CreateShl(AndCST, 
                                         Shift->getOperand(1), "tmp");
        } else {
          // Insert a logical shift.
          NS = BinaryOperator::CreateLShr(AndCST,
                                          Shift->getOperand(1), "tmp");
        }
        InsertNewInstBefore(cast<Instruction>(NS), ICI);
        
        // Compute X & (C << Y).
        Instruction *NewAnd = 
          BinaryOperator::CreateAnd(Shift->getOperand(0), NS, LHSI->getName());
        InsertNewInstBefore(NewAnd, ICI);
        
        ICI.setOperand(0, NewAnd);
        return &ICI;
      }
    }
    break;
    
  case Instruction::Shl: {       // (icmp pred (shl X, ShAmt), CI)
    ConstantInt *ShAmt = dyn_cast<ConstantInt>(LHSI->getOperand(1));
    if (!ShAmt) break;
    
    uint32_t TypeBits = RHSV.getBitWidth();
    
    // Check that the shift amount is in range.  If not, don't perform
    // undefined shifts.  When the shift is visited it will be
    // simplified.
    if (ShAmt->uge(TypeBits))
      break;
    
    if (ICI.isEquality()) {
      // If we are comparing against bits always shifted out, the
      // comparison cannot succeed.
      Constant *Comp =
        ConstantExpr::getShl(ConstantExpr::getLShr(RHS, ShAmt), ShAmt);
      if (Comp != RHS) {// Comparing against a bit that we know is zero.
        bool IsICMP_NE = ICI.getPredicate() == ICmpInst::ICMP_NE;
        Constant *Cst = ConstantInt::get(Type::Int1Ty, IsICMP_NE);
        return ReplaceInstUsesWith(ICI, Cst);
      }
      
      if (LHSI->hasOneUse()) {
        // Otherwise strength reduce the shift into an and.
        uint32_t ShAmtVal = (uint32_t)ShAmt->getLimitedValue(TypeBits);
        Constant *Mask =
          ConstantInt::get(APInt::getLowBitsSet(TypeBits, TypeBits-ShAmtVal));
        
        Instruction *AndI =
          BinaryOperator::CreateAnd(LHSI->getOperand(0),
                                    Mask, LHSI->getName()+".mask");
        Value *And = InsertNewInstBefore(AndI, ICI);
        return new ICmpInst(ICI.getPredicate(), And,
                            ConstantInt::get(RHSV.lshr(ShAmtVal)));
      }
    }
    
    // Otherwise, if this is a comparison of the sign bit, simplify to and/test.
    bool TrueIfSigned = false;
    if (LHSI->hasOneUse() &&
        isSignBitCheck(ICI.getPredicate(), RHS, TrueIfSigned)) {
      // (X << 31) <s 0  --> (X&1) != 0
      Constant *Mask = ConstantInt::get(APInt(TypeBits, 1) <<
                                           (TypeBits-ShAmt->getZExtValue()-1));
      Instruction *AndI =
        BinaryOperator::CreateAnd(LHSI->getOperand(0),
                                  Mask, LHSI->getName()+".mask");
      Value *And = InsertNewInstBefore(AndI, ICI);
      
      return new ICmpInst(TrueIfSigned ? ICmpInst::ICMP_NE : ICmpInst::ICMP_EQ,
                          And, Constant::getNullValue(And->getType()));
    }
    break;
  }
    
  case Instruction::LShr:         // (icmp pred (shr X, ShAmt), CI)
  case Instruction::AShr: {
    // Only handle equality comparisons of shift-by-constant.
    ConstantInt *ShAmt = dyn_cast<ConstantInt>(LHSI->getOperand(1));
    if (!ShAmt || !ICI.isEquality()) break;

    // Check that the shift amount is in range.  If not, don't perform
    // undefined shifts.  When the shift is visited it will be
    // simplified.
    uint32_t TypeBits = RHSV.getBitWidth();
    if (ShAmt->uge(TypeBits))
      break;
    
    uint32_t ShAmtVal = (uint32_t)ShAmt->getLimitedValue(TypeBits);
      
    // If we are comparing against bits always shifted out, the
    // comparison cannot succeed.
    APInt Comp = RHSV << ShAmtVal;
    if (LHSI->getOpcode() == Instruction::LShr)
      Comp = Comp.lshr(ShAmtVal);
    else
      Comp = Comp.ashr(ShAmtVal);
    
    if (Comp != RHSV) { // Comparing against a bit that we know is zero.
      bool IsICMP_NE = ICI.getPredicate() == ICmpInst::ICMP_NE;
      Constant *Cst = ConstantInt::get(Type::Int1Ty, IsICMP_NE);
      return ReplaceInstUsesWith(ICI, Cst);
    }
    
    // Otherwise, check to see if the bits shifted out are known to be zero.
    // If so, we can compare against the unshifted value:
    //  (X & 4) >> 1 == 2  --> (X & 4) == 4.
    if (LHSI->hasOneUse() &&
        MaskedValueIsZero(LHSI->getOperand(0), 
                          APInt::getLowBitsSet(Comp.getBitWidth(), ShAmtVal))) {
      return new ICmpInst(ICI.getPredicate(), LHSI->getOperand(0),
                          ConstantExpr::getShl(RHS, ShAmt));
    }
      
    if (LHSI->hasOneUse()) {
      // Otherwise strength reduce the shift into an and.
      APInt Val(APInt::getHighBitsSet(TypeBits, TypeBits - ShAmtVal));
      Constant *Mask = ConstantInt::get(Val);
      
      Instruction *AndI =
        BinaryOperator::CreateAnd(LHSI->getOperand(0),
                                  Mask, LHSI->getName()+".mask");
      Value *And = InsertNewInstBefore(AndI, ICI);
      return new ICmpInst(ICI.getPredicate(), And,
                          ConstantExpr::getShl(RHS, ShAmt));
    }
    break;
  }
    
  case Instruction::SDiv:
  case Instruction::UDiv:
    // Fold: icmp pred ([us]div X, C1), C2 -> range test
    // Fold this div into the comparison, producing a range check. 
    // Determine, based on the divide type, what the range is being 
    // checked.  If there is an overflow on the low or high side, remember 
    // it, otherwise compute the range [low, hi) bounding the new value.
    // See: InsertRangeTest above for the kinds of replacements possible.
    if (ConstantInt *DivRHS = dyn_cast<ConstantInt>(LHSI->getOperand(1)))
      if (Instruction *R = FoldICmpDivCst(ICI, cast<BinaryOperator>(LHSI),
                                          DivRHS))
        return R;
    break;

  case Instruction::Add:
    // Fold: icmp pred (add, X, C1), C2

    if (!ICI.isEquality()) {
      ConstantInt *LHSC = dyn_cast<ConstantInt>(LHSI->getOperand(1));
      if (!LHSC) break;
      const APInt &LHSV = LHSC->getValue();

      ConstantRange CR = ICI.makeConstantRange(ICI.getPredicate(), RHSV)
                            .subtract(LHSV);

      if (ICI.isSignedPredicate()) {
        if (CR.getLower().isSignBit()) {
          return new ICmpInst(ICmpInst::ICMP_SLT, LHSI->getOperand(0),
                              ConstantInt::get(CR.getUpper()));
        } else if (CR.getUpper().isSignBit()) {
          return new ICmpInst(ICmpInst::ICMP_SGE, LHSI->getOperand(0),
                              ConstantInt::get(CR.getLower()));
        }
      } else {
        if (CR.getLower().isMinValue()) {
          return new ICmpInst(ICmpInst::ICMP_ULT, LHSI->getOperand(0),
                              ConstantInt::get(CR.getUpper()));
        } else if (CR.getUpper().isMinValue()) {
          return new ICmpInst(ICmpInst::ICMP_UGE, LHSI->getOperand(0),
                              ConstantInt::get(CR.getLower()));
        }
      }
    }
    break;
  }
  
  // Simplify icmp_eq and icmp_ne instructions with integer constant RHS.
  if (ICI.isEquality()) {
    bool isICMP_NE = ICI.getPredicate() == ICmpInst::ICMP_NE;
    
    // If the first operand is (add|sub|and|or|xor|rem) with a constant, and 
    // the second operand is a constant, simplify a bit.
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(LHSI)) {
      switch (BO->getOpcode()) {
      case Instruction::SRem:
        // If we have a signed (X % (2^c)) == 0, turn it into an unsigned one.
        if (RHSV == 0 && isa<ConstantInt>(BO->getOperand(1)) &&BO->hasOneUse()){
          const APInt &V = cast<ConstantInt>(BO->getOperand(1))->getValue();
          if (V.sgt(APInt(V.getBitWidth(), 1)) && V.isPowerOf2()) {
            Instruction *NewRem =
              BinaryOperator::CreateURem(BO->getOperand(0), BO->getOperand(1),
                                         BO->getName());
            InsertNewInstBefore(NewRem, ICI);
            return new ICmpInst(ICI.getPredicate(), NewRem, 
                                Constant::getNullValue(BO->getType()));
          }
        }
        break;
      case Instruction::Add:
        // Replace ((add A, B) != C) with (A != C-B) if B & C are constants.
        if (ConstantInt *BOp1C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
          if (BO->hasOneUse())
            return new ICmpInst(ICI.getPredicate(), BO->getOperand(0),
                                Subtract(RHS, BOp1C));
        } else if (RHSV == 0) {
          // Replace ((add A, B) != 0) with (A != -B) if A or B is
          // efficiently invertible, or if the add has just this one use.
          Value *BOp0 = BO->getOperand(0), *BOp1 = BO->getOperand(1);
          
          if (Value *NegVal = dyn_castNegVal(BOp1))
            return new ICmpInst(ICI.getPredicate(), BOp0, NegVal);
          else if (Value *NegVal = dyn_castNegVal(BOp0))
            return new ICmpInst(ICI.getPredicate(), NegVal, BOp1);
          else if (BO->hasOneUse()) {
            Instruction *Neg = BinaryOperator::CreateNeg(BOp1);
            InsertNewInstBefore(Neg, ICI);
            Neg->takeName(BO);
            return new ICmpInst(ICI.getPredicate(), BOp0, Neg);
          }
        }
        break;
      case Instruction::Xor:
        // For the xor case, we can xor two constants together, eliminating
        // the explicit xor.
        if (Constant *BOC = dyn_cast<Constant>(BO->getOperand(1)))
          return new ICmpInst(ICI.getPredicate(), BO->getOperand(0), 
                              ConstantExpr::getXor(RHS, BOC));
        
        // FALLTHROUGH
      case Instruction::Sub:
        // Replace (([sub|xor] A, B) != 0) with (A != B)
        if (RHSV == 0)
          return new ICmpInst(ICI.getPredicate(), BO->getOperand(0),
                              BO->getOperand(1));
        break;
        
      case Instruction::Or:
        // If bits are being or'd in that are not present in the constant we
        // are comparing against, then the comparison could never succeed!
        if (Constant *BOC = dyn_cast<Constant>(BO->getOperand(1))) {
          Constant *NotCI = ConstantExpr::getNot(RHS);
          if (!ConstantExpr::getAnd(BOC, NotCI)->isNullValue())
            return ReplaceInstUsesWith(ICI, ConstantInt::get(Type::Int1Ty, 
                                                             isICMP_NE));
        }
        break;
        
      case Instruction::And:
        if (ConstantInt *BOC = dyn_cast<ConstantInt>(BO->getOperand(1))) {
          // If bits are being compared against that are and'd out, then the
          // comparison can never succeed!
          if ((RHSV & ~BOC->getValue()) != 0)
            return ReplaceInstUsesWith(ICI, ConstantInt::get(Type::Int1Ty,
                                                             isICMP_NE));
          
          // If we have ((X & C) == C), turn it into ((X & C) != 0).
          if (RHS == BOC && RHSV.isPowerOf2())
            return new ICmpInst(isICMP_NE ? ICmpInst::ICMP_EQ :
                                ICmpInst::ICMP_NE, LHSI,
                                Constant::getNullValue(RHS->getType()));
          
          // Replace (and X, (1 << size(X)-1) != 0) with x s< 0
          if (BOC->getValue().isSignBit()) {
            Value *X = BO->getOperand(0);
            Constant *Zero = Constant::getNullValue(X->getType());
            ICmpInst::Predicate pred = isICMP_NE ? 
              ICmpInst::ICMP_SLT : ICmpInst::ICMP_SGE;
            return new ICmpInst(pred, X, Zero);
          }
          
          // ((X & ~7) == 0) --> X < 8
          if (RHSV == 0 && isHighOnes(BOC)) {
            Value *X = BO->getOperand(0);
            Constant *NegX = ConstantExpr::getNeg(BOC);
            ICmpInst::Predicate pred = isICMP_NE ? 
              ICmpInst::ICMP_UGE : ICmpInst::ICMP_ULT;
            return new ICmpInst(pred, X, NegX);
          }
        }
      default: break;
      }
    } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(LHSI)) {
      // Handle icmp {eq|ne} <intrinsic>, intcst.
      if (II->getIntrinsicID() == Intrinsic::bswap) {
        AddToWorkList(II);
        ICI.setOperand(0, II->getOperand(1));
        ICI.setOperand(1, ConstantInt::get(RHSV.byteSwap()));
        return &ICI;
      }
    }
  }
  return 0;
}

/// visitICmpInstWithCastAndCast - Handle icmp (cast x to y), (cast/cst).
/// We only handle extending casts so far.
///
Instruction *InstCombiner::visitICmpInstWithCastAndCast(ICmpInst &ICI) {
  const CastInst *LHSCI = cast<CastInst>(ICI.getOperand(0));
  Value *LHSCIOp        = LHSCI->getOperand(0);
  const Type *SrcTy     = LHSCIOp->getType();
  const Type *DestTy    = LHSCI->getType();
  Value *RHSCIOp;

  // Turn icmp (ptrtoint x), (ptrtoint/c) into a compare of the input if the 
  // integer type is the same size as the pointer type.
  if (LHSCI->getOpcode() == Instruction::PtrToInt &&
      getTargetData().getPointerSizeInBits() == 
         cast<IntegerType>(DestTy)->getBitWidth()) {
    Value *RHSOp = 0;
    if (Constant *RHSC = dyn_cast<Constant>(ICI.getOperand(1))) {
      RHSOp = ConstantExpr::getIntToPtr(RHSC, SrcTy);
    } else if (PtrToIntInst *RHSC = dyn_cast<PtrToIntInst>(ICI.getOperand(1))) {
      RHSOp = RHSC->getOperand(0);
      // If the pointer types don't match, insert a bitcast.
      if (LHSCIOp->getType() != RHSOp->getType())
        RHSOp = InsertBitCastBefore(RHSOp, LHSCIOp->getType(), ICI);
    }

    if (RHSOp)
      return new ICmpInst(ICI.getPredicate(), LHSCIOp, RHSOp);
  }
  
  // The code below only handles extension cast instructions, so far.
  // Enforce this.
  if (LHSCI->getOpcode() != Instruction::ZExt &&
      LHSCI->getOpcode() != Instruction::SExt)
    return 0;

  bool isSignedExt = LHSCI->getOpcode() == Instruction::SExt;
  bool isSignedCmp = ICI.isSignedPredicate();

  if (CastInst *CI = dyn_cast<CastInst>(ICI.getOperand(1))) {
    // Not an extension from the same type?
    RHSCIOp = CI->getOperand(0);
    if (RHSCIOp->getType() != LHSCIOp->getType()) 
      return 0;
    
    // If the signedness of the two casts doesn't agree (i.e. one is a sext
    // and the other is a zext), then we can't handle this.
    if (CI->getOpcode() != LHSCI->getOpcode())
      return 0;

    // Deal with equality cases early.
    if (ICI.isEquality())
      return new ICmpInst(ICI.getPredicate(), LHSCIOp, RHSCIOp);

    // A signed comparison of sign extended values simplifies into a
    // signed comparison.
    if (isSignedCmp && isSignedExt)
      return new ICmpInst(ICI.getPredicate(), LHSCIOp, RHSCIOp);

    // The other three cases all fold into an unsigned comparison.
    return new ICmpInst(ICI.getUnsignedPredicate(), LHSCIOp, RHSCIOp);
  }

  // If we aren't dealing with a constant on the RHS, exit early
  ConstantInt *CI = dyn_cast<ConstantInt>(ICI.getOperand(1));
  if (!CI)
    return 0;

  // Compute the constant that would happen if we truncated to SrcTy then
  // reextended to DestTy.
  Constant *Res1 = ConstantExpr::getTrunc(CI, SrcTy);
  Constant *Res2 = ConstantExpr::getCast(LHSCI->getOpcode(), Res1, DestTy);

  // If the re-extended constant didn't change...
  if (Res2 == CI) {
    // Make sure that sign of the Cmp and the sign of the Cast are the same.
    // For example, we might have:
    //    %A = sext short %X to uint
    //    %B = icmp ugt uint %A, 1330
    // It is incorrect to transform this into 
    //    %B = icmp ugt short %X, 1330 
    // because %A may have negative value. 
    //
    // However, we allow this when the compare is EQ/NE, because they are
    // signless.
    if (isSignedExt == isSignedCmp || ICI.isEquality())
      return new ICmpInst(ICI.getPredicate(), LHSCIOp, Res1);
    return 0;
  }

  // The re-extended constant changed so the constant cannot be represented 
  // in the shorter type. Consequently, we cannot emit a simple comparison.

  // First, handle some easy cases. We know the result cannot be equal at this
  // point so handle the ICI.isEquality() cases
  if (ICI.getPredicate() == ICmpInst::ICMP_EQ)
    return ReplaceInstUsesWith(ICI, ConstantInt::getFalse());
  if (ICI.getPredicate() == ICmpInst::ICMP_NE)
    return ReplaceInstUsesWith(ICI, ConstantInt::getTrue());

  // Evaluate the comparison for LT (we invert for GT below). LE and GE cases
  // should have been folded away previously and not enter in here.
  Value *Result;
  if (isSignedCmp) {
    // We're performing a signed comparison.
    if (cast<ConstantInt>(CI)->getValue().isNegative())
      Result = ConstantInt::getFalse();          // X < (small) --> false
    else
      Result = ConstantInt::getTrue();           // X < (large) --> true
  } else {
    // We're performing an unsigned comparison.
    if (isSignedExt) {
      // We're performing an unsigned comp with a sign extended value.
      // This is true if the input is >= 0. [aka >s -1]
      Constant *NegOne = ConstantInt::getAllOnesValue(SrcTy);
      Result = InsertNewInstBefore(new ICmpInst(ICmpInst::ICMP_SGT, LHSCIOp,
                                   NegOne, ICI.getName()), ICI);
    } else {
      // Unsigned extend & unsigned compare -> always true.
      Result = ConstantInt::getTrue();
    }
  }

  // Finally, return the value computed.
  if (ICI.getPredicate() == ICmpInst::ICMP_ULT ||
      ICI.getPredicate() == ICmpInst::ICMP_SLT)
    return ReplaceInstUsesWith(ICI, Result);

  assert((ICI.getPredicate()==ICmpInst::ICMP_UGT || 
          ICI.getPredicate()==ICmpInst::ICMP_SGT) &&
         "ICmp should be folded!");
  if (Constant *CI = dyn_cast<Constant>(Result))
    return ReplaceInstUsesWith(ICI, ConstantExpr::getNot(CI));
  return BinaryOperator::CreateNot(Result);
}

Instruction *InstCombiner::visitShl(BinaryOperator &I) {
  return commonShiftTransforms(I);
}

Instruction *InstCombiner::visitLShr(BinaryOperator &I) {
  return commonShiftTransforms(I);
}

Instruction *InstCombiner::visitAShr(BinaryOperator &I) {
  if (Instruction *R = commonShiftTransforms(I))
    return R;
  
  Value *Op0 = I.getOperand(0);
  
  // ashr int -1, X = -1   (for any arithmetic shift rights of ~0)
  if (ConstantInt *CSI = dyn_cast<ConstantInt>(Op0))
    if (CSI->isAllOnesValue())
      return ReplaceInstUsesWith(I, CSI);
  
  // See if we can turn a signed shr into an unsigned shr.
  if (!isa<VectorType>(I.getType()) &&
      MaskedValueIsZero(Op0,
                      APInt::getSignBit(I.getType()->getPrimitiveSizeInBits())))
    return BinaryOperator::CreateLShr(Op0, I.getOperand(1));
  
  return 0;
}

Instruction *InstCombiner::commonShiftTransforms(BinaryOperator &I) {
  assert(I.getOperand(1)->getType() == I.getOperand(0)->getType());
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // shl X, 0 == X and shr X, 0 == X
  // shl 0, X == 0 and shr 0, X == 0
  if (Op1 == Constant::getNullValue(Op1->getType()) ||
      Op0 == Constant::getNullValue(Op0->getType()))
    return ReplaceInstUsesWith(I, Op0);
  
  if (isa<UndefValue>(Op0)) {            
    if (I.getOpcode() == Instruction::AShr) // undef >>s X -> undef
      return ReplaceInstUsesWith(I, Op0);
    else                                    // undef << X -> 0, undef >>u X -> 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }
  if (isa<UndefValue>(Op1)) {
    if (I.getOpcode() == Instruction::AShr)  // X >>s undef -> X
      return ReplaceInstUsesWith(I, Op0);          
    else                                     // X << undef, X >>u undef -> 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }

  // Try to fold constant and into select arguments.
  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;

  if (ConstantInt *CUI = dyn_cast<ConstantInt>(Op1))
    if (Instruction *Res = FoldShiftByConstant(Op0, CUI, I))
      return Res;
  return 0;
}

Instruction *InstCombiner::FoldShiftByConstant(Value *Op0, ConstantInt *Op1,
                                               BinaryOperator &I) {
  bool isLeftShift = I.getOpcode() == Instruction::Shl;

  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  uint32_t TypeBits = Op0->getType()->getPrimitiveSizeInBits();
  if (SimplifyDemandedInstructionBits(I))
    return &I;
  
  // shl uint X, 32 = 0 and shr ubyte Y, 9 = 0, ... just don't eliminate shr
  // of a signed value.
  //
  if (Op1->uge(TypeBits)) {
    if (I.getOpcode() != Instruction::AShr)
      return ReplaceInstUsesWith(I, Constant::getNullValue(Op0->getType()));
    else {
      I.setOperand(1, ConstantInt::get(I.getType(), TypeBits-1));
      return &I;
    }
  }
  
  // ((X*C1) << C2) == (X * (C1 << C2))
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0))
    if (BO->getOpcode() == Instruction::Mul && isLeftShift)
      if (Constant *BOOp = dyn_cast<Constant>(BO->getOperand(1)))
        return BinaryOperator::CreateMul(BO->getOperand(0),
                                         ConstantExpr::getShl(BOOp, Op1));
  
  // Try to fold constant and into select arguments.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
    if (Instruction *R = FoldOpIntoSelect(I, SI, this))
      return R;
  if (isa<PHINode>(Op0))
    if (Instruction *NV = FoldOpIntoPhi(I))
      return NV;
  
  // Fold shift2(trunc(shift1(x,c1)), c2) -> trunc(shift2(shift1(x,c1),c2))
  if (TruncInst *TI = dyn_cast<TruncInst>(Op0)) {
    Instruction *TrOp = dyn_cast<Instruction>(TI->getOperand(0));
    // If 'shift2' is an ashr, we would have to get the sign bit into a funny
    // place.  Don't try to do this transformation in this case.  Also, we
    // require that the input operand is a shift-by-constant so that we have
    // confidence that the shifts will get folded together.  We could do this
    // xform in more cases, but it is unlikely to be profitable.
    if (TrOp && I.isLogicalShift() && TrOp->isShift() && 
        isa<ConstantInt>(TrOp->getOperand(1))) {
      // Okay, we'll do this xform.  Make the shift of shift.
      Constant *ShAmt = ConstantExpr::getZExt(Op1, TrOp->getType());
      Instruction *NSh = BinaryOperator::Create(I.getOpcode(), TrOp, ShAmt,
                                                I.getName());
      InsertNewInstBefore(NSh, I); // (shift2 (shift1 & 0x00FF), c2)

      // For logical shifts, the truncation has the effect of making the high
      // part of the register be zeros.  Emulate this by inserting an AND to
      // clear the top bits as needed.  This 'and' will usually be zapped by
      // other xforms later if dead.
      unsigned SrcSize = TrOp->getType()->getPrimitiveSizeInBits();
      unsigned DstSize = TI->getType()->getPrimitiveSizeInBits();
      APInt MaskV(APInt::getLowBitsSet(SrcSize, DstSize));
      
      // The mask we constructed says what the trunc would do if occurring
      // between the shifts.  We want to know the effect *after* the second
      // shift.  We know that it is a logical shift by a constant, so adjust the
      // mask as appropriate.
      if (I.getOpcode() == Instruction::Shl)
        MaskV <<= Op1->getZExtValue();
      else {
        assert(I.getOpcode() == Instruction::LShr && "Unknown logical shift");
        MaskV = MaskV.lshr(Op1->getZExtValue());
      }

      Instruction *And = BinaryOperator::CreateAnd(NSh, ConstantInt::get(MaskV),
                                                   TI->getName());
      InsertNewInstBefore(And, I); // shift1 & 0x00FF

      // Return the value truncated to the interesting size.
      return new TruncInst(And, I.getType());
    }
  }
  
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
        case Instruction::Xor: {
          // These operators commute.
          // Turn (Y + (X >> C)) << C  ->  (X + (Y << C)) & (~0 << C)
          if (isLeftShift && Op0BO->getOperand(1)->hasOneUse() &&
              match(Op0BO->getOperand(1), m_Shr(m_Value(V1), m_Specific(Op1)))){
            Instruction *YS = BinaryOperator::CreateShl(
                                            Op0BO->getOperand(0), Op1,
                                            Op0BO->getName());
            InsertNewInstBefore(YS, I); // (Y << C)
            Instruction *X = 
              BinaryOperator::Create(Op0BO->getOpcode(), YS, V1,
                                     Op0BO->getOperand(1)->getName());
            InsertNewInstBefore(X, I);  // (X + (Y << C))
            uint32_t Op1Val = Op1->getLimitedValue(TypeBits);
            return BinaryOperator::CreateAnd(X, ConstantInt::get(
                       APInt::getHighBitsSet(TypeBits, TypeBits-Op1Val)));
          }
          
          // Turn (Y + ((X >> C) & CC)) << C  ->  ((X & (CC << C)) + (Y << C))
          Value *Op0BOOp1 = Op0BO->getOperand(1);
          if (isLeftShift && Op0BOOp1->hasOneUse() &&
              match(Op0BOOp1, 
                    m_And(m_Shr(m_Value(V1), m_Specific(Op1)),
                          m_ConstantInt(CC))) &&
              cast<BinaryOperator>(Op0BOOp1)->getOperand(0)->hasOneUse()) {
            Instruction *YS = BinaryOperator::CreateShl(
                                                     Op0BO->getOperand(0), Op1,
                                                     Op0BO->getName());
            InsertNewInstBefore(YS, I); // (Y << C)
            Instruction *XM =
              BinaryOperator::CreateAnd(V1, ConstantExpr::getShl(CC, Op1),
                                        V1->getName()+".mask");
            InsertNewInstBefore(XM, I); // X & (CC << C)
            
            return BinaryOperator::Create(Op0BO->getOpcode(), YS, XM);
          }
        }
          
        // FALL THROUGH.
        case Instruction::Sub: {
          // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
          if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
              match(Op0BO->getOperand(0), m_Shr(m_Value(V1), m_Specific(Op1)))){
            Instruction *YS = BinaryOperator::CreateShl(
                                                     Op0BO->getOperand(1), Op1,
                                                     Op0BO->getName());
            InsertNewInstBefore(YS, I); // (Y << C)
            Instruction *X =
              BinaryOperator::Create(Op0BO->getOpcode(), V1, YS,
                                     Op0BO->getOperand(0)->getName());
            InsertNewInstBefore(X, I);  // (X + (Y << C))
            uint32_t Op1Val = Op1->getLimitedValue(TypeBits);
            return BinaryOperator::CreateAnd(X, ConstantInt::get(
                       APInt::getHighBitsSet(TypeBits, TypeBits-Op1Val)));
          }
          
          // Turn (((X >> C)&CC) + Y) << C  ->  (X + (Y << C)) & (CC << C)
          if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
              match(Op0BO->getOperand(0),
                    m_And(m_Shr(m_Value(V1), m_Value(V2)),
                          m_ConstantInt(CC))) && V2 == Op1 &&
              cast<BinaryOperator>(Op0BO->getOperand(0))
                  ->getOperand(0)->hasOneUse()) {
            Instruction *YS = BinaryOperator::CreateShl(
                                                     Op0BO->getOperand(1), Op1,
                                                     Op0BO->getName());
            InsertNewInstBefore(YS, I); // (Y << C)
            Instruction *XM =
              BinaryOperator::CreateAnd(V1, ConstantExpr::getShl(CC, Op1),
                                        V1->getName()+".mask");
            InsertNewInstBefore(XM, I); // X & (CC << C)
            
            return BinaryOperator::Create(Op0BO->getOpcode(), XM, YS);
          }
          
          break;
        }
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
        if (isValid && I.getOpcode() == Instruction::AShr)
          isValid = Op0C->getValue()[TypeBits-1] == highBitSet;
        
        if (isValid) {
          Constant *NewRHS = ConstantExpr::get(I.getOpcode(), Op0C, Op1);
          
          Instruction *NewShift =
            BinaryOperator::Create(I.getOpcode(), Op0BO->getOperand(0), Op1);
          InsertNewInstBefore(NewShift, I);
          NewShift->takeName(Op0BO);
          
          return BinaryOperator::Create(Op0BO->getOpcode(), NewShift,
                                        NewRHS);
        }
      }
    }
  }
  
  // Find out if this is a shift of a shift by a constant.
  BinaryOperator *ShiftOp = dyn_cast<BinaryOperator>(Op0);
  if (ShiftOp && !ShiftOp->isShift())
    ShiftOp = 0;
  
  if (ShiftOp && isa<ConstantInt>(ShiftOp->getOperand(1))) {
    ConstantInt *ShiftAmt1C = cast<ConstantInt>(ShiftOp->getOperand(1));
    uint32_t ShiftAmt1 = ShiftAmt1C->getLimitedValue(TypeBits);
    uint32_t ShiftAmt2 = Op1->getLimitedValue(TypeBits);
    assert(ShiftAmt2 != 0 && "Should have been simplified earlier");
    if (ShiftAmt1 == 0) return 0;  // Will be simplified in the future.
    Value *X = ShiftOp->getOperand(0);
    
    uint32_t AmtSum = ShiftAmt1+ShiftAmt2;   // Fold into one big shift.
    if (AmtSum > TypeBits)
      AmtSum = TypeBits;
    
    const IntegerType *Ty = cast<IntegerType>(I.getType());
    
    // Check for (X << c1) << c2  and  (X >> c1) >> c2
    if (I.getOpcode() == ShiftOp->getOpcode()) {
      return BinaryOperator::Create(I.getOpcode(), X,
                                    ConstantInt::get(Ty, AmtSum));
    } else if (ShiftOp->getOpcode() == Instruction::LShr &&
               I.getOpcode() == Instruction::AShr) {
      // ((X >>u C1) >>s C2) -> (X >>u (C1+C2))  since C1 != 0.
      return BinaryOperator::CreateLShr(X, ConstantInt::get(Ty, AmtSum));
    } else if (ShiftOp->getOpcode() == Instruction::AShr &&
               I.getOpcode() == Instruction::LShr) {
      // ((X >>s C1) >>u C2) -> ((X >>s (C1+C2)) & mask) since C1 != 0.
      Instruction *Shift =
        BinaryOperator::CreateAShr(X, ConstantInt::get(Ty, AmtSum));
      InsertNewInstBefore(Shift, I);

      APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt2));
      return BinaryOperator::CreateAnd(Shift, ConstantInt::get(Mask));
    }
    
    // Okay, if we get here, one shift must be left, and the other shift must be
    // right.  See if the amounts are equal.
    if (ShiftAmt1 == ShiftAmt2) {
      // If we have ((X >>? C) << C), turn this into X & (-1 << C).
      if (I.getOpcode() == Instruction::Shl) {
        APInt Mask(APInt::getHighBitsSet(TypeBits, TypeBits - ShiftAmt1));
        return BinaryOperator::CreateAnd(X, ConstantInt::get(Mask));
      }
      // If we have ((X << C) >>u C), turn this into X & (-1 >>u C).
      if (I.getOpcode() == Instruction::LShr) {
        APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt1));
        return BinaryOperator::CreateAnd(X, ConstantInt::get(Mask));
      }
      // We can simplify ((X << C) >>s C) into a trunc + sext.
      // NOTE: we could do this for any C, but that would make 'unusual' integer
      // types.  For now, just stick to ones well-supported by the code
      // generators.
      const Type *SExtType = 0;
      switch (Ty->getBitWidth() - ShiftAmt1) {
      case 1  :
      case 8  :
      case 16 :
      case 32 :
      case 64 :
      case 128:
        SExtType = IntegerType::get(Ty->getBitWidth() - ShiftAmt1);
        break;
      default: break;
      }
      if (SExtType) {
        Instruction *NewTrunc = new TruncInst(X, SExtType, "sext");
        InsertNewInstBefore(NewTrunc, I);
        return new SExtInst(NewTrunc, Ty);
      }
      // Otherwise, we can't handle it yet.
    } else if (ShiftAmt1 < ShiftAmt2) {
      uint32_t ShiftDiff = ShiftAmt2-ShiftAmt1;
      
      // (X >>? C1) << C2 --> X << (C2-C1) & (-1 << C2)
      if (I.getOpcode() == Instruction::Shl) {
        assert(ShiftOp->getOpcode() == Instruction::LShr ||
               ShiftOp->getOpcode() == Instruction::AShr);
        Instruction *Shift =
          BinaryOperator::CreateShl(X, ConstantInt::get(Ty, ShiftDiff));
        InsertNewInstBefore(Shift, I);
        
        APInt Mask(APInt::getHighBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift, ConstantInt::get(Mask));
      }
      
      // (X << C1) >>u C2  --> X >>u (C2-C1) & (-1 >> C2)
      if (I.getOpcode() == Instruction::LShr) {
        assert(ShiftOp->getOpcode() == Instruction::Shl);
        Instruction *Shift =
          BinaryOperator::CreateLShr(X, ConstantInt::get(Ty, ShiftDiff));
        InsertNewInstBefore(Shift, I);
        
        APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift, ConstantInt::get(Mask));
      }
      
      // We can't handle (X << C1) >>s C2, it shifts arbitrary bits in.
    } else {
      assert(ShiftAmt2 < ShiftAmt1);
      uint32_t ShiftDiff = ShiftAmt1-ShiftAmt2;

      // (X >>? C1) << C2 --> X >>? (C1-C2) & (-1 << C2)
      if (I.getOpcode() == Instruction::Shl) {
        assert(ShiftOp->getOpcode() == Instruction::LShr ||
               ShiftOp->getOpcode() == Instruction::AShr);
        Instruction *Shift =
          BinaryOperator::Create(ShiftOp->getOpcode(), X,
                                 ConstantInt::get(Ty, ShiftDiff));
        InsertNewInstBefore(Shift, I);
        
        APInt Mask(APInt::getHighBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift, ConstantInt::get(Mask));
      }
      
      // (X << C1) >>u C2  --> X << (C1-C2) & (-1 >> C2)
      if (I.getOpcode() == Instruction::LShr) {
        assert(ShiftOp->getOpcode() == Instruction::Shl);
        Instruction *Shift =
          BinaryOperator::CreateShl(X, ConstantInt::get(Ty, ShiftDiff));
        InsertNewInstBefore(Shift, I);
        
        APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift, ConstantInt::get(Mask));
      }
      
      // We can't handle (X << C1) >>a C2, it shifts arbitrary bits in.
    }
  }
  return 0;
}


/// DecomposeSimpleLinearExpr - Analyze 'Val', seeing if it is a simple linear
/// expression.  If so, decompose it, returning some value X, such that Val is
/// X*Scale+Offset.
///
static Value *DecomposeSimpleLinearExpr(Value *Val, unsigned &Scale,
                                        int &Offset) {
  assert(Val->getType() == Type::Int32Ty && "Unexpected allocation size type!");
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Val)) {
    Offset = CI->getZExtValue();
    Scale  = 0;
    return ConstantInt::get(Type::Int32Ty, 0);
  } else if (BinaryOperator *I = dyn_cast<BinaryOperator>(Val)) {
    if (ConstantInt *RHS = dyn_cast<ConstantInt>(I->getOperand(1))) {
      if (I->getOpcode() == Instruction::Shl) {
        // This is a value scaled by '1 << the shift amt'.
        Scale = 1U << RHS->getZExtValue();
        Offset = 0;
        return I->getOperand(0);
      } else if (I->getOpcode() == Instruction::Mul) {
        // This value is scaled by 'RHS'.
        Scale = RHS->getZExtValue();
        Offset = 0;
        return I->getOperand(0);
      } else if (I->getOpcode() == Instruction::Add) {
        // We have X+C.  Check to see if we really have (X*C2)+C1, 
        // where C1 is divisible by C2.
        unsigned SubScale;
        Value *SubVal = 
          DecomposeSimpleLinearExpr(I->getOperand(0), SubScale, Offset);
        Offset += RHS->getZExtValue();
        Scale = SubScale;
        return SubVal;
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
Instruction *InstCombiner::PromoteCastOfAllocation(BitCastInst &CI,
                                                   AllocationInst &AI) {
  const PointerType *PTy = cast<PointerType>(CI.getType());
  
  // Remove any uses of AI that are dead.
  assert(!CI.use_empty() && "Dead instructions should be removed earlier!");
  
  for (Value::use_iterator UI = AI.use_begin(), E = AI.use_end(); UI != E; ) {
    Instruction *User = cast<Instruction>(*UI++);
    if (isInstructionTriviallyDead(User)) {
      while (UI != E && *UI == User)
        ++UI; // If this instruction uses AI more than once, don't break UI.
      
      ++NumDeadInst;
      DOUT << "IC: DCE: " << *User;
      EraseInstFromFunction(*User);
    }
  }
  
  // Get the type really allocated and the type casted to.
  const Type *AllocElTy = AI.getAllocatedType();
  const Type *CastElTy = PTy->getElementType();
  if (!AllocElTy->isSized() || !CastElTy->isSized()) return 0;

  unsigned AllocElTyAlign = TD->getABITypeAlignment(AllocElTy);
  unsigned CastElTyAlign = TD->getABITypeAlignment(CastElTy);
  if (CastElTyAlign < AllocElTyAlign) return 0;

  // If the allocation has multiple uses, only promote it if we are strictly
  // increasing the alignment of the resultant allocation.  If we keep it the
  // same, we open the door to infinite loops of various kinds.
  if (!AI.hasOneUse() && CastElTyAlign == AllocElTyAlign) return 0;

  uint64_t AllocElTySize = TD->getTypePaddedSize(AllocElTy);
  uint64_t CastElTySize = TD->getTypePaddedSize(CastElTy);
  if (CastElTySize == 0 || AllocElTySize == 0) return 0;

  // See if we can satisfy the modulus by pulling a scale out of the array
  // size argument.
  unsigned ArraySizeScale;
  int ArrayOffset;
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
    Amt = ConstantInt::get(Type::Int32Ty, Scale);
    if (isa<ConstantInt>(NumElements))
      Amt = Multiply(cast<ConstantInt>(NumElements), cast<ConstantInt>(Amt));
    // otherwise multiply the amount and the number of elements
    else if (Scale != 1) {
      Instruction *Tmp = BinaryOperator::CreateMul(Amt, NumElements, "tmp");
      Amt = InsertNewInstBefore(Tmp, AI);
    }
  }
  
  if (int Offset = (AllocElTySize*ArrayOffset)/CastElTySize) {
    Value *Off = ConstantInt::get(Type::Int32Ty, Offset, true);
    Instruction *Tmp = BinaryOperator::CreateAdd(Amt, Off, "tmp");
    Amt = InsertNewInstBefore(Tmp, AI);
  }
  
  AllocationInst *New;
  if (isa<MallocInst>(AI))
    New = new MallocInst(CastElTy, Amt, AI.getAlignment());
  else
    New = new AllocaInst(CastElTy, Amt, AI.getAlignment());
  InsertNewInstBefore(New, AI);
  New->takeName(&AI);
  
  // If the allocation has multiple uses, insert a cast and change all things
  // that used it to use the new cast.  This will also hack on CI, but it will
  // die soon.
  if (!AI.hasOneUse()) {
    AddUsesToWorkList(AI);
    // New is the allocation instruction, pointer typed. AI is the original
    // allocation instruction, also pointer typed. Thus, cast to use is BitCast.
    CastInst *NewCast = new BitCastInst(New, AI.getType(), "tmpcast");
    InsertNewInstBefore(NewCast, AI);
    AI.replaceAllUsesWith(NewCast);
  }
  return ReplaceInstUsesWith(CI, New);
}

/// CanEvaluateInDifferentType - Return true if we can take the specified value
/// and return it as type Ty without inserting any new casts and without
/// changing the computed value.  This is used by code that tries to decide
/// whether promoting or shrinking integer operations to wider or smaller types
/// will allow us to eliminate a truncate or extend.
///
/// This is a truncation operation if Ty is smaller than V->getType(), or an
/// extension operation if Ty is larger.
///
/// If CastOpc is a truncation, then Ty will be a type smaller than V.  We
/// should return true if trunc(V) can be computed by computing V in the smaller
/// type.  If V is an instruction, then trunc(inst(x,y)) can be computed as
/// inst(trunc(x),trunc(y)), which only makes sense if x and y can be
/// efficiently truncated.
///
/// If CastOpc is a sext or zext, we are asking if the low bits of the value can
/// bit computed in a larger type, which is then and'd or sext_in_reg'd to get
/// the final result.
bool InstCombiner::CanEvaluateInDifferentType(Value *V, const IntegerType *Ty,
                                              unsigned CastOpc,
                                              int &NumCastsRemoved){
  // We can always evaluate constants in another type.
  if (isa<ConstantInt>(V))
    return true;
  
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;
  
  const IntegerType *OrigTy = cast<IntegerType>(V->getType());
  
  // If this is an extension or truncate, we can often eliminate it.
  if (isa<TruncInst>(I) || isa<ZExtInst>(I) || isa<SExtInst>(I)) {
    // If this is a cast from the destination type, we can trivially eliminate
    // it, and this will remove a cast overall.
    if (I->getOperand(0)->getType() == Ty) {
      // If the first operand is itself a cast, and is eliminable, do not count
      // this as an eliminable cast.  We would prefer to eliminate those two
      // casts first.
      if (!isa<CastInst>(I->getOperand(0)) && I->hasOneUse())
        ++NumCastsRemoved;
      return true;
    }
  }

  // We can't extend or shrink something that has multiple uses: doing so would
  // require duplicating the instruction in general, which isn't profitable.
  if (!I->hasOneUse()) return false;

  unsigned Opc = I->getOpcode();
  switch (Opc) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    // These operators can all arbitrarily be extended or truncated.
    return CanEvaluateInDifferentType(I->getOperand(0), Ty, CastOpc,
                                      NumCastsRemoved) &&
           CanEvaluateInDifferentType(I->getOperand(1), Ty, CastOpc,
                                      NumCastsRemoved);

  case Instruction::Shl:
    // If we are truncating the result of this SHL, and if it's a shift of a
    // constant amount, we can always perform a SHL in a smaller type.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1))) {
      uint32_t BitWidth = Ty->getBitWidth();
      if (BitWidth < OrigTy->getBitWidth() && 
          CI->getLimitedValue(BitWidth) < BitWidth)
        return CanEvaluateInDifferentType(I->getOperand(0), Ty, CastOpc,
                                          NumCastsRemoved);
    }
    break;
  case Instruction::LShr:
    // If this is a truncate of a logical shr, we can truncate it to a smaller
    // lshr iff we know that the bits we would otherwise be shifting in are
    // already zeros.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1))) {
      uint32_t OrigBitWidth = OrigTy->getBitWidth();
      uint32_t BitWidth = Ty->getBitWidth();
      if (BitWidth < OrigBitWidth &&
          MaskedValueIsZero(I->getOperand(0),
            APInt::getHighBitsSet(OrigBitWidth, OrigBitWidth-BitWidth)) &&
          CI->getLimitedValue(BitWidth) < BitWidth) {
        return CanEvaluateInDifferentType(I->getOperand(0), Ty, CastOpc,
                                          NumCastsRemoved);
      }
    }
    break;
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::Trunc:
    // If this is the same kind of case as our original (e.g. zext+zext), we
    // can safely replace it.  Note that replacing it does not reduce the number
    // of casts in the input.
    if (Opc == CastOpc)
      return true;

    // sext (zext ty1), ty2 -> zext ty2
    if (CastOpc == Instruction::SExt && Opc == Instruction::ZExt)
      return true;
    break;
  case Instruction::Select: {
    SelectInst *SI = cast<SelectInst>(I);
    return CanEvaluateInDifferentType(SI->getTrueValue(), Ty, CastOpc,
                                      NumCastsRemoved) &&
           CanEvaluateInDifferentType(SI->getFalseValue(), Ty, CastOpc,
                                      NumCastsRemoved);
  }
  case Instruction::PHI: {
    // We can change a phi if we can change all operands.
    PHINode *PN = cast<PHINode>(I);
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (!CanEvaluateInDifferentType(PN->getIncomingValue(i), Ty, CastOpc,
                                      NumCastsRemoved))
        return false;
    return true;
  }
  default:
    // TODO: Can handle more cases here.
    break;
  }
  
  return false;
}

/// EvaluateInDifferentType - Given an expression that 
/// CanEvaluateInDifferentType returns true for, actually insert the code to
/// evaluate the expression.
Value *InstCombiner::EvaluateInDifferentType(Value *V, const Type *Ty, 
                                             bool isSigned) {
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getIntegerCast(C, Ty, isSigned /*Sext or ZExt*/);

  // Otherwise, it must be an instruction.
  Instruction *I = cast<Instruction>(V);
  Instruction *Res = 0;
  unsigned Opc = I->getOpcode();
  switch (Opc) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::AShr:
  case Instruction::LShr:
  case Instruction::Shl: {
    Value *LHS = EvaluateInDifferentType(I->getOperand(0), Ty, isSigned);
    Value *RHS = EvaluateInDifferentType(I->getOperand(1), Ty, isSigned);
    Res = BinaryOperator::Create((Instruction::BinaryOps)Opc, LHS, RHS);
    break;
  }    
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
    // If the source type of the cast is the type we're trying for then we can
    // just return the source.  There's no need to insert it because it is not
    // new.
    if (I->getOperand(0)->getType() == Ty)
      return I->getOperand(0);
    
    // Otherwise, must be the same type of cast, so just reinsert a new one.
    Res = CastInst::Create(cast<CastInst>(I)->getOpcode(), I->getOperand(0),
                           Ty);
    break;
  case Instruction::Select: {
    Value *True = EvaluateInDifferentType(I->getOperand(1), Ty, isSigned);
    Value *False = EvaluateInDifferentType(I->getOperand(2), Ty, isSigned);
    Res = SelectInst::Create(I->getOperand(0), True, False);
    break;
  }
  case Instruction::PHI: {
    PHINode *OPN = cast<PHINode>(I);
    PHINode *NPN = PHINode::Create(Ty);
    for (unsigned i = 0, e = OPN->getNumIncomingValues(); i != e; ++i) {
      Value *V =EvaluateInDifferentType(OPN->getIncomingValue(i), Ty, isSigned);
      NPN->addIncoming(V, OPN->getIncomingBlock(i));
    }
    Res = NPN;
    break;
  }
  default: 
    // TODO: Can handle more cases here.
    assert(0 && "Unreachable!");
    break;
  }
  
  Res->takeName(I);
  return InsertNewInstBefore(Res, *I);
}

/// @brief Implement the transforms common to all CastInst visitors.
Instruction *InstCombiner::commonCastTransforms(CastInst &CI) {
  Value *Src = CI.getOperand(0);

  // Many cases of "cast of a cast" are eliminable. If it's eliminable we just
  // eliminate it now.
  if (CastInst *CSrc = dyn_cast<CastInst>(Src)) {   // A->B->C cast
    if (Instruction::CastOps opc = 
        isEliminableCastPair(CSrc, CI.getOpcode(), CI.getType(), TD)) {
      // The first cast (CSrc) is eliminable so we need to fix up or replace
      // the second cast (CI). CSrc will then have a good chance of being dead.
      return CastInst::Create(opc, CSrc->getOperand(0), CI.getType());
    }
  }

  // If we are casting a select then fold the cast into the select
  if (SelectInst *SI = dyn_cast<SelectInst>(Src))
    if (Instruction *NV = FoldOpIntoSelect(CI, SI, this))
      return NV;

  // If we are casting a PHI then fold the cast into the PHI
  if (isa<PHINode>(Src))
    if (Instruction *NV = FoldOpIntoPhi(CI))
      return NV;
  
  return 0;
}

/// FindElementAtOffset - Given a type and a constant offset, determine whether
/// or not there is a sequence of GEP indices into the type that will land us at
/// the specified offset.  If so, fill them into NewIndices and return the
/// resultant element type, otherwise return null.
static const Type *FindElementAtOffset(const Type *Ty, int64_t Offset, 
                                       SmallVectorImpl<Value*> &NewIndices,
                                       const TargetData *TD) {
  if (!Ty->isSized()) return 0;
  
  // Start with the index over the outer type.  Note that the type size
  // might be zero (even if the offset isn't zero) if the indexed type
  // is something like [0 x {int, int}]
  const Type *IntPtrTy = TD->getIntPtrType();
  int64_t FirstIdx = 0;
  if (int64_t TySize = TD->getTypePaddedSize(Ty)) {
    FirstIdx = Offset/TySize;
    Offset -= FirstIdx*TySize;
    
    // Handle hosts where % returns negative instead of values [0..TySize).
    if (Offset < 0) {
      --FirstIdx;
      Offset += TySize;
      assert(Offset >= 0);
    }
    assert((uint64_t)Offset < (uint64_t)TySize && "Out of range offset");
  }
  
  NewIndices.push_back(ConstantInt::get(IntPtrTy, FirstIdx));
    
  // Index into the types.  If we fail, set OrigBase to null.
  while (Offset) {
    // Indexing into tail padding between struct/array elements.
    if (uint64_t(Offset*8) >= TD->getTypeSizeInBits(Ty))
      return 0;
    
    if (const StructType *STy = dyn_cast<StructType>(Ty)) {
      const StructLayout *SL = TD->getStructLayout(STy);
      assert(Offset < (int64_t)SL->getSizeInBytes() &&
             "Offset must stay within the indexed type");
      
      unsigned Elt = SL->getElementContainingOffset(Offset);
      NewIndices.push_back(ConstantInt::get(Type::Int32Ty, Elt));
      
      Offset -= SL->getElementOffset(Elt);
      Ty = STy->getElementType(Elt);
    } else if (const ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
      uint64_t EltSize = TD->getTypePaddedSize(AT->getElementType());
      assert(EltSize && "Cannot index into a zero-sized array");
      NewIndices.push_back(ConstantInt::get(IntPtrTy,Offset/EltSize));
      Offset %= EltSize;
      Ty = AT->getElementType();
    } else {
      // Otherwise, we can't index into the middle of this atomic type, bail.
      return 0;
    }
  }
  
  return Ty;
}

/// @brief Implement the transforms for cast of pointer (bitcast/ptrtoint)
Instruction *InstCombiner::commonPointerCastTransforms(CastInst &CI) {
  Value *Src = CI.getOperand(0);
  
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Src)) {
    // If casting the result of a getelementptr instruction with no offset, turn
    // this into a cast of the original pointer!
    if (GEP->hasAllZeroIndices()) {
      // Changing the cast operand is usually not a good idea but it is safe
      // here because the pointer operand is being replaced with another 
      // pointer operand so the opcode doesn't need to change.
      AddToWorkList(GEP);
      CI.setOperand(0, GEP->getOperand(0));
      return &CI;
    }
    
    // If the GEP has a single use, and the base pointer is a bitcast, and the
    // GEP computes a constant offset, see if we can convert these three
    // instructions into fewer.  This typically happens with unions and other
    // non-type-safe code.
    if (GEP->hasOneUse() && isa<BitCastInst>(GEP->getOperand(0))) {
      if (GEP->hasAllConstantIndices()) {
        // We are guaranteed to get a constant from EmitGEPOffset.
        ConstantInt *OffsetV = cast<ConstantInt>(EmitGEPOffset(GEP, CI, *this));
        int64_t Offset = OffsetV->getSExtValue();
        
        // Get the base pointer input of the bitcast, and the type it points to.
        Value *OrigBase = cast<BitCastInst>(GEP->getOperand(0))->getOperand(0);
        const Type *GEPIdxTy =
          cast<PointerType>(OrigBase->getType())->getElementType();
        SmallVector<Value*, 8> NewIndices;
        if (FindElementAtOffset(GEPIdxTy, Offset, NewIndices, TD)) {
          // If we were able to index down into an element, create the GEP
          // and bitcast the result.  This eliminates one bitcast, potentially
          // two.
          Instruction *NGEP = GetElementPtrInst::Create(OrigBase, 
                                                        NewIndices.begin(),
                                                        NewIndices.end(), "");
          InsertNewInstBefore(NGEP, CI);
          NGEP->takeName(GEP);
          
          if (isa<BitCastInst>(CI))
            return new BitCastInst(NGEP, CI.getType());
          assert(isa<PtrToIntInst>(CI));
          return new PtrToIntInst(NGEP, CI.getType());
        }
      }      
    }
  }
    
  return commonCastTransforms(CI);
}


/// Only the TRUNC, ZEXT, SEXT, and BITCAST can both operand and result as
/// integer types. This function implements the common transforms for all those
/// cases.
/// @brief Implement the transforms common to CastInst with integer operands
Instruction *InstCombiner::commonIntCastTransforms(CastInst &CI) {
  if (Instruction *Result = commonCastTransforms(CI))
    return Result;

  Value *Src = CI.getOperand(0);
  const Type *SrcTy = Src->getType();
  const Type *DestTy = CI.getType();
  uint32_t SrcBitSize = SrcTy->getPrimitiveSizeInBits();
  uint32_t DestBitSize = DestTy->getPrimitiveSizeInBits();

  // See if we can simplify any instructions used by the LHS whose sole 
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(CI))
    return &CI;

  // If the source isn't an instruction or has more than one use then we
  // can't do anything more. 
  Instruction *SrcI = dyn_cast<Instruction>(Src);
  if (!SrcI || !Src->hasOneUse())
    return 0;

  // Attempt to propagate the cast into the instruction for int->int casts.
  int NumCastsRemoved = 0;
  if (!isa<BitCastInst>(CI) &&
      CanEvaluateInDifferentType(SrcI, cast<IntegerType>(DestTy),
                                 CI.getOpcode(), NumCastsRemoved)) {
    // If this cast is a truncate, evaluting in a different type always
    // eliminates the cast, so it is always a win.  If this is a zero-extension,
    // we need to do an AND to maintain the clear top-part of the computation,
    // so we require that the input have eliminated at least one cast.  If this
    // is a sign extension, we insert two new casts (to do the extension) so we
    // require that two casts have been eliminated.
    bool DoXForm = false;
    bool JustReplace = false;
    switch (CI.getOpcode()) {
    default:
      // All the others use floating point so we shouldn't actually 
      // get here because of the check above.
      assert(0 && "Unknown cast type");
    case Instruction::Trunc:
      DoXForm = true;
      break;
    case Instruction::ZExt: {
      DoXForm = NumCastsRemoved >= 1;
      if (!DoXForm && 0) {
        // If it's unnecessary to issue an AND to clear the high bits, it's
        // always profitable to do this xform.
        Value *TryRes = EvaluateInDifferentType(SrcI, DestTy, false);
        APInt Mask(APInt::getBitsSet(DestBitSize, SrcBitSize, DestBitSize));
        if (MaskedValueIsZero(TryRes, Mask))
          return ReplaceInstUsesWith(CI, TryRes);
        
        if (Instruction *TryI = dyn_cast<Instruction>(TryRes))
          if (TryI->use_empty())
            EraseInstFromFunction(*TryI);
      }
      break;
    }
    case Instruction::SExt: {
      DoXForm = NumCastsRemoved >= 2;
      if (!DoXForm && !isa<TruncInst>(SrcI) && 0) {
        // If we do not have to emit the truncate + sext pair, then it's always
        // profitable to do this xform.
        //
        // It's not safe to eliminate the trunc + sext pair if one of the
        // eliminated cast is a truncate. e.g.
        // t2 = trunc i32 t1 to i16
        // t3 = sext i16 t2 to i32
        // !=
        // i32 t1
        Value *TryRes = EvaluateInDifferentType(SrcI, DestTy, true);
        unsigned NumSignBits = ComputeNumSignBits(TryRes);
        if (NumSignBits > (DestBitSize - SrcBitSize))
          return ReplaceInstUsesWith(CI, TryRes);
        
        if (Instruction *TryI = dyn_cast<Instruction>(TryRes))
          if (TryI->use_empty())
            EraseInstFromFunction(*TryI);
      }
      break;
    }
    }
    
    if (DoXForm) {
      DOUT << "ICE: EvaluateInDifferentType converting expression type to avoid"
           << " cast: " << CI;
      Value *Res = EvaluateInDifferentType(SrcI, DestTy, 
                                           CI.getOpcode() == Instruction::SExt);
      if (JustReplace)
        // Just replace this cast with the result.
        return ReplaceInstUsesWith(CI, Res);

      assert(Res->getType() == DestTy);
      switch (CI.getOpcode()) {
      default: assert(0 && "Unknown cast type!");
      case Instruction::Trunc:
      case Instruction::BitCast:
        // Just replace this cast with the result.
        return ReplaceInstUsesWith(CI, Res);
      case Instruction::ZExt: {
        assert(SrcBitSize < DestBitSize && "Not a zext?");

        // If the high bits are already zero, just replace this cast with the
        // result.
        APInt Mask(APInt::getBitsSet(DestBitSize, SrcBitSize, DestBitSize));
        if (MaskedValueIsZero(Res, Mask))
          return ReplaceInstUsesWith(CI, Res);

        // We need to emit an AND to clear the high bits.
        Constant *C = ConstantInt::get(APInt::getLowBitsSet(DestBitSize,
                                                            SrcBitSize));
        return BinaryOperator::CreateAnd(Res, C);
      }
      case Instruction::SExt: {
        // If the high bits are already filled with sign bit, just replace this
        // cast with the result.
        unsigned NumSignBits = ComputeNumSignBits(Res);
        if (NumSignBits > (DestBitSize - SrcBitSize))
          return ReplaceInstUsesWith(CI, Res);

        // We need to emit a cast to truncate, then a cast to sext.
        return CastInst::Create(Instruction::SExt,
            InsertCastBefore(Instruction::Trunc, Res, Src->getType(), 
                             CI), DestTy);
      }
      }
    }
  }
  
  Value *Op0 = SrcI->getNumOperands() > 0 ? SrcI->getOperand(0) : 0;
  Value *Op1 = SrcI->getNumOperands() > 1 ? SrcI->getOperand(1) : 0;

  switch (SrcI->getOpcode()) {
  case Instruction::Add:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    // If we are discarding information, rewrite.
    if (DestBitSize <= SrcBitSize && DestBitSize != 1) {
      // Don't insert two casts if they cannot be eliminated.  We allow 
      // two casts to be inserted if the sizes are the same.  This could 
      // only be converting signedness, which is a noop.
      if (DestBitSize == SrcBitSize || 
          !ValueRequiresCast(CI.getOpcode(), Op1, DestTy,TD) ||
          !ValueRequiresCast(CI.getOpcode(), Op0, DestTy, TD)) {
        Instruction::CastOps opcode = CI.getOpcode();
        Value *Op0c = InsertCastBefore(opcode, Op0, DestTy, *SrcI);
        Value *Op1c = InsertCastBefore(opcode, Op1, DestTy, *SrcI);
        return BinaryOperator::Create(
            cast<BinaryOperator>(SrcI)->getOpcode(), Op0c, Op1c);
      }
    }

    // cast (xor bool X, true) to int  --> xor (cast bool X to int), 1
    if (isa<ZExtInst>(CI) && SrcBitSize == 1 && 
        SrcI->getOpcode() == Instruction::Xor &&
        Op1 == ConstantInt::getTrue() &&
        (!Op0->hasOneUse() || !isa<CmpInst>(Op0))) {
      Value *New = InsertCastBefore(Instruction::ZExt, Op0, DestTy, CI);
      return BinaryOperator::CreateXor(New, ConstantInt::get(CI.getType(), 1));
    }
    break;
  case Instruction::SDiv:
  case Instruction::UDiv:
  case Instruction::SRem:
  case Instruction::URem:
    // If we are just changing the sign, rewrite.
    if (DestBitSize == SrcBitSize) {
      // Don't insert two casts if they cannot be eliminated.  We allow 
      // two casts to be inserted if the sizes are the same.  This could 
      // only be converting signedness, which is a noop.
      if (!ValueRequiresCast(CI.getOpcode(), Op1, DestTy, TD) || 
          !ValueRequiresCast(CI.getOpcode(), Op0, DestTy, TD)) {
        Value *Op0c = InsertCastBefore(Instruction::BitCast, 
                                       Op0, DestTy, *SrcI);
        Value *Op1c = InsertCastBefore(Instruction::BitCast, 
                                       Op1, DestTy, *SrcI);
        return BinaryOperator::Create(
          cast<BinaryOperator>(SrcI)->getOpcode(), Op0c, Op1c);
      }
    }
    break;

  case Instruction::Shl:
    // Allow changing the sign of the source operand.  Do not allow 
    // changing the size of the shift, UNLESS the shift amount is a 
    // constant.  We must not change variable sized shifts to a smaller 
    // size, because it is undefined to shift more bits out than exist 
    // in the value.
    if (DestBitSize == SrcBitSize ||
        (DestBitSize < SrcBitSize && isa<Constant>(Op1))) {
      Instruction::CastOps opcode = (DestBitSize == SrcBitSize ?
          Instruction::BitCast : Instruction::Trunc);
      Value *Op0c = InsertCastBefore(opcode, Op0, DestTy, *SrcI);
      Value *Op1c = InsertCastBefore(opcode, Op1, DestTy, *SrcI);
      return BinaryOperator::CreateShl(Op0c, Op1c);
    }
    break;
  case Instruction::AShr:
    // If this is a signed shr, and if all bits shifted in are about to be
    // truncated off, turn it into an unsigned shr to allow greater
    // simplifications.
    if (DestBitSize < SrcBitSize &&
        isa<ConstantInt>(Op1)) {
      uint32_t ShiftAmt = cast<ConstantInt>(Op1)->getLimitedValue(SrcBitSize);
      if (SrcBitSize > ShiftAmt && SrcBitSize-ShiftAmt >= DestBitSize) {
        // Insert the new logical shift right.
        return BinaryOperator::CreateLShr(Op0, Op1);
      }
    }
    break;
  }
  return 0;
}

Instruction *InstCombiner::visitTrunc(TruncInst &CI) {
  if (Instruction *Result = commonIntCastTransforms(CI))
    return Result;
  
  Value *Src = CI.getOperand(0);
  const Type *Ty = CI.getType();
  uint32_t DestBitWidth = Ty->getPrimitiveSizeInBits();
  uint32_t SrcBitWidth = cast<IntegerType>(Src->getType())->getBitWidth();
  
  if (Instruction *SrcI = dyn_cast<Instruction>(Src)) {
    switch (SrcI->getOpcode()) {
    default: break;
    case Instruction::LShr:
      // We can shrink lshr to something smaller if we know the bits shifted in
      // are already zeros.
      if (ConstantInt *ShAmtV = dyn_cast<ConstantInt>(SrcI->getOperand(1))) {
        uint32_t ShAmt = ShAmtV->getLimitedValue(SrcBitWidth);
        
        // Get a mask for the bits shifting in.
        APInt Mask(APInt::getLowBitsSet(SrcBitWidth, ShAmt).shl(DestBitWidth));
        Value* SrcIOp0 = SrcI->getOperand(0);
        if (SrcI->hasOneUse() && MaskedValueIsZero(SrcIOp0, Mask)) {
          if (ShAmt >= DestBitWidth)        // All zeros.
            return ReplaceInstUsesWith(CI, Constant::getNullValue(Ty));

          // Okay, we can shrink this.  Truncate the input, then return a new
          // shift.
          Value *V1 = InsertCastBefore(Instruction::Trunc, SrcIOp0, Ty, CI);
          Value *V2 = InsertCastBefore(Instruction::Trunc, SrcI->getOperand(1),
                                       Ty, CI);
          return BinaryOperator::CreateLShr(V1, V2);
        }
      } else {     // This is a variable shr.
        
        // Turn 'trunc (lshr X, Y) to bool' into '(X & (1 << Y)) != 0'.  This is
        // more LLVM instructions, but allows '1 << Y' to be hoisted if
        // loop-invariant and CSE'd.
        if (CI.getType() == Type::Int1Ty && SrcI->hasOneUse()) {
          Value *One = ConstantInt::get(SrcI->getType(), 1);

          Value *V = InsertNewInstBefore(
              BinaryOperator::CreateShl(One, SrcI->getOperand(1),
                                     "tmp"), CI);
          V = InsertNewInstBefore(BinaryOperator::CreateAnd(V,
                                                            SrcI->getOperand(0),
                                                            "tmp"), CI);
          Value *Zero = Constant::getNullValue(V->getType());
          return new ICmpInst(ICmpInst::ICMP_NE, V, Zero);
        }
      }
      break;
    }
  }
  
  return 0;
}

/// transformZExtICmp - Transform (zext icmp) to bitwise / integer operations
/// in order to eliminate the icmp.
Instruction *InstCombiner::transformZExtICmp(ICmpInst *ICI, Instruction &CI,
                                             bool DoXform) {
  // If we are just checking for a icmp eq of a single bit and zext'ing it
  // to an integer, then shift the bit to the appropriate place and then
  // cast to integer to avoid the comparison.
  if (ConstantInt *Op1C = dyn_cast<ConstantInt>(ICI->getOperand(1))) {
    const APInt &Op1CV = Op1C->getValue();
      
    // zext (x <s  0) to i32 --> x>>u31      true if signbit set.
    // zext (x >s -1) to i32 --> (x>>u31)^1  true if signbit clear.
    if ((ICI->getPredicate() == ICmpInst::ICMP_SLT && Op1CV == 0) ||
        (ICI->getPredicate() == ICmpInst::ICMP_SGT &&Op1CV.isAllOnesValue())) {
      if (!DoXform) return ICI;

      Value *In = ICI->getOperand(0);
      Value *Sh = ConstantInt::get(In->getType(),
                                   In->getType()->getPrimitiveSizeInBits()-1);
      In = InsertNewInstBefore(BinaryOperator::CreateLShr(In, Sh,
                                                        In->getName()+".lobit"),
                               CI);
      if (In->getType() != CI.getType())
        In = CastInst::CreateIntegerCast(In, CI.getType(),
                                         false/*ZExt*/, "tmp", &CI);

      if (ICI->getPredicate() == ICmpInst::ICMP_SGT) {
        Constant *One = ConstantInt::get(In->getType(), 1);
        In = InsertNewInstBefore(BinaryOperator::CreateXor(In, One,
                                                         In->getName()+".not"),
                                 CI);
      }

      return ReplaceInstUsesWith(CI, In);
    }
      
      
      
    // zext (X == 0) to i32 --> X^1      iff X has only the low bit set.
    // zext (X == 0) to i32 --> (X>>1)^1 iff X has only the 2nd bit set.
    // zext (X == 1) to i32 --> X        iff X has only the low bit set.
    // zext (X == 2) to i32 --> X>>1     iff X has only the 2nd bit set.
    // zext (X != 0) to i32 --> X        iff X has only the low bit set.
    // zext (X != 0) to i32 --> X>>1     iff X has only the 2nd bit set.
    // zext (X != 1) to i32 --> X^1      iff X has only the low bit set.
    // zext (X != 2) to i32 --> (X>>1)^1 iff X has only the 2nd bit set.
    if ((Op1CV == 0 || Op1CV.isPowerOf2()) && 
        // This only works for EQ and NE
        ICI->isEquality()) {
      // If Op1C some other power of two, convert:
      uint32_t BitWidth = Op1C->getType()->getBitWidth();
      APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
      APInt TypeMask(APInt::getAllOnesValue(BitWidth));
      ComputeMaskedBits(ICI->getOperand(0), TypeMask, KnownZero, KnownOne);
        
      APInt KnownZeroMask(~KnownZero);
      if (KnownZeroMask.isPowerOf2()) { // Exactly 1 possible 1?
        if (!DoXform) return ICI;

        bool isNE = ICI->getPredicate() == ICmpInst::ICMP_NE;
        if (Op1CV != 0 && (Op1CV != KnownZeroMask)) {
          // (X&4) == 2 --> false
          // (X&4) != 2 --> true
          Constant *Res = ConstantInt::get(Type::Int1Ty, isNE);
          Res = ConstantExpr::getZExt(Res, CI.getType());
          return ReplaceInstUsesWith(CI, Res);
        }
          
        uint32_t ShiftAmt = KnownZeroMask.logBase2();
        Value *In = ICI->getOperand(0);
        if (ShiftAmt) {
          // Perform a logical shr by shiftamt.
          // Insert the shift to put the result in the low bit.
          In = InsertNewInstBefore(BinaryOperator::CreateLShr(In,
                                  ConstantInt::get(In->getType(), ShiftAmt),
                                                   In->getName()+".lobit"), CI);
        }
          
        if ((Op1CV != 0) == isNE) { // Toggle the low bit.
          Constant *One = ConstantInt::get(In->getType(), 1);
          In = BinaryOperator::CreateXor(In, One, "tmp");
          InsertNewInstBefore(cast<Instruction>(In), CI);
        }
          
        if (CI.getType() == In->getType())
          return ReplaceInstUsesWith(CI, In);
        else
          return CastInst::CreateIntegerCast(In, CI.getType(), false/*ZExt*/);
      }
    }
  }

  return 0;
}

Instruction *InstCombiner::visitZExt(ZExtInst &CI) {
  // If one of the common conversion will work ..
  if (Instruction *Result = commonIntCastTransforms(CI))
    return Result;

  Value *Src = CI.getOperand(0);

  // If this is a cast of a cast
  if (CastInst *CSrc = dyn_cast<CastInst>(Src)) {   // A->B->C cast
    // If this is a TRUNC followed by a ZEXT then we are dealing with integral
    // types and if the sizes are just right we can convert this into a logical
    // 'and' which will be much cheaper than the pair of casts.
    if (isa<TruncInst>(CSrc)) {
      // Get the sizes of the types involved
      Value *A = CSrc->getOperand(0);
      uint32_t SrcSize = A->getType()->getPrimitiveSizeInBits();
      uint32_t MidSize = CSrc->getType()->getPrimitiveSizeInBits();
      uint32_t DstSize = CI.getType()->getPrimitiveSizeInBits();
      // If we're actually extending zero bits and the trunc is a no-op
      if (MidSize < DstSize && SrcSize == DstSize) {
        // Replace both of the casts with an And of the type mask.
        APInt AndValue(APInt::getLowBitsSet(SrcSize, MidSize));
        Constant *AndConst = ConstantInt::get(AndValue);
        Instruction *And = 
          BinaryOperator::CreateAnd(CSrc->getOperand(0), AndConst);
        // Unfortunately, if the type changed, we need to cast it back.
        if (And->getType() != CI.getType()) {
          And->setName(CSrc->getName()+".mask");
          InsertNewInstBefore(And, CI);
          And = CastInst::CreateIntegerCast(And, CI.getType(), false/*ZExt*/);
        }
        return And;
      }
    }
  }

  if (ICmpInst *ICI = dyn_cast<ICmpInst>(Src))
    return transformZExtICmp(ICI, CI);

  BinaryOperator *SrcI = dyn_cast<BinaryOperator>(Src);
  if (SrcI && SrcI->getOpcode() == Instruction::Or) {
    // zext (or icmp, icmp) --> or (zext icmp), (zext icmp) if at least one
    // of the (zext icmp) will be transformed.
    ICmpInst *LHS = dyn_cast<ICmpInst>(SrcI->getOperand(0));
    ICmpInst *RHS = dyn_cast<ICmpInst>(SrcI->getOperand(1));
    if (LHS && RHS && LHS->hasOneUse() && RHS->hasOneUse() &&
        (transformZExtICmp(LHS, CI, false) ||
         transformZExtICmp(RHS, CI, false))) {
      Value *LCast = InsertCastBefore(Instruction::ZExt, LHS, CI.getType(), CI);
      Value *RCast = InsertCastBefore(Instruction::ZExt, RHS, CI.getType(), CI);
      return BinaryOperator::Create(Instruction::Or, LCast, RCast);
    }
  }

  return 0;
}

Instruction *InstCombiner::visitSExt(SExtInst &CI) {
  if (Instruction *I = commonIntCastTransforms(CI))
    return I;
  
  Value *Src = CI.getOperand(0);
  
  // Canonicalize sign-extend from i1 to a select.
  if (Src->getType() == Type::Int1Ty)
    return SelectInst::Create(Src,
                              ConstantInt::getAllOnesValue(CI.getType()),
                              Constant::getNullValue(CI.getType()));

  // See if the value being truncated is already sign extended.  If so, just
  // eliminate the trunc/sext pair.
  if (getOpcode(Src) == Instruction::Trunc) {
    Value *Op = cast<User>(Src)->getOperand(0);
    unsigned OpBits   = cast<IntegerType>(Op->getType())->getBitWidth();
    unsigned MidBits  = cast<IntegerType>(Src->getType())->getBitWidth();
    unsigned DestBits = cast<IntegerType>(CI.getType())->getBitWidth();
    unsigned NumSignBits = ComputeNumSignBits(Op);

    if (OpBits == DestBits) {
      // Op is i32, Mid is i8, and Dest is i32.  If Op has more than 24 sign
      // bits, it is already ready.
      if (NumSignBits > DestBits-MidBits)
        return ReplaceInstUsesWith(CI, Op);
    } else if (OpBits < DestBits) {
      // Op is i32, Mid is i8, and Dest is i64.  If Op has more than 24 sign
      // bits, just sext from i32.
      if (NumSignBits > OpBits-MidBits)
        return new SExtInst(Op, CI.getType(), "tmp");
    } else {
      // Op is i64, Mid is i8, and Dest is i32.  If Op has more than 56 sign
      // bits, just truncate to i32.
      if (NumSignBits > OpBits-MidBits)
        return new TruncInst(Op, CI.getType(), "tmp");
    }
  }

  // If the input is a shl/ashr pair of a same constant, then this is a sign
  // extension from a smaller value.  If we could trust arbitrary bitwidth
  // integers, we could turn this into a truncate to the smaller bit and then
  // use a sext for the whole extension.  Since we don't, look deeper and check
  // for a truncate.  If the source and dest are the same type, eliminate the
  // trunc and extend and just do shifts.  For example, turn:
  //   %a = trunc i32 %i to i8
  //   %b = shl i8 %a, 6
  //   %c = ashr i8 %b, 6
  //   %d = sext i8 %c to i32
  // into:
  //   %a = shl i32 %i, 30
  //   %d = ashr i32 %a, 30
  Value *A = 0;
  ConstantInt *BA = 0, *CA = 0;
  if (match(Src, m_AShr(m_Shl(m_Value(A), m_ConstantInt(BA)),
                        m_ConstantInt(CA))) &&
      BA == CA && isa<TruncInst>(A)) {
    Value *I = cast<TruncInst>(A)->getOperand(0);
    if (I->getType() == CI.getType()) {
      unsigned MidSize = Src->getType()->getPrimitiveSizeInBits();
      unsigned SrcDstSize = CI.getType()->getPrimitiveSizeInBits();
      unsigned ShAmt = CA->getZExtValue()+SrcDstSize-MidSize;
      Constant *ShAmtV = ConstantInt::get(CI.getType(), ShAmt);
      I = InsertNewInstBefore(BinaryOperator::CreateShl(I, ShAmtV,
                                                        CI.getName()), CI);
      return BinaryOperator::CreateAShr(I, ShAmtV);
    }
  }
  
  return 0;
}

/// FitsInFPType - Return a Constant* for the specified FP constant if it fits
/// in the specified FP type without changing its value.
static Constant *FitsInFPType(ConstantFP *CFP, const fltSemantics &Sem) {
  bool losesInfo;
  APFloat F = CFP->getValueAPF();
  (void)F.convert(Sem, APFloat::rmNearestTiesToEven, &losesInfo);
  if (!losesInfo)
    return ConstantFP::get(F);
  return 0;
}

/// LookThroughFPExtensions - If this is an fp extension instruction, look
/// through it until we get the source value.
static Value *LookThroughFPExtensions(Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (I->getOpcode() == Instruction::FPExt)
      return LookThroughFPExtensions(I->getOperand(0));
  
  // If this value is a constant, return the constant in the smallest FP type
  // that can accurately represent it.  This allows us to turn
  // (float)((double)X+2.0) into x+2.0f.
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(V)) {
    if (CFP->getType() == Type::PPC_FP128Ty)
      return V;  // No constant folding of this.
    // See if the value can be truncated to float and then reextended.
    if (Value *V = FitsInFPType(CFP, APFloat::IEEEsingle))
      return V;
    if (CFP->getType() == Type::DoubleTy)
      return V;  // Won't shrink.
    if (Value *V = FitsInFPType(CFP, APFloat::IEEEdouble))
      return V;
    // Don't try to shrink to various long double types.
  }
  
  return V;
}

Instruction *InstCombiner::visitFPTrunc(FPTruncInst &CI) {
  if (Instruction *I = commonCastTransforms(CI))
    return I;
  
  // If we have fptrunc(add (fpextend x), (fpextend y)), where x and y are
  // smaller than the destination type, we can eliminate the truncate by doing
  // the add as the smaller type.  This applies to add/sub/mul/div as well as
  // many builtins (sqrt, etc).
  BinaryOperator *OpI = dyn_cast<BinaryOperator>(CI.getOperand(0));
  if (OpI && OpI->hasOneUse()) {
    switch (OpI->getOpcode()) {
    default: break;
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::FDiv:
    case Instruction::FRem:
      const Type *SrcTy = OpI->getType();
      Value *LHSTrunc = LookThroughFPExtensions(OpI->getOperand(0));
      Value *RHSTrunc = LookThroughFPExtensions(OpI->getOperand(1));
      if (LHSTrunc->getType() != SrcTy && 
          RHSTrunc->getType() != SrcTy) {
        unsigned DstSize = CI.getType()->getPrimitiveSizeInBits();
        // If the source types were both smaller than the destination type of
        // the cast, do this xform.
        if (LHSTrunc->getType()->getPrimitiveSizeInBits() <= DstSize &&
            RHSTrunc->getType()->getPrimitiveSizeInBits() <= DstSize) {
          LHSTrunc = InsertCastBefore(Instruction::FPExt, LHSTrunc,
                                      CI.getType(), CI);
          RHSTrunc = InsertCastBefore(Instruction::FPExt, RHSTrunc,
                                      CI.getType(), CI);
          return BinaryOperator::Create(OpI->getOpcode(), LHSTrunc, RHSTrunc);
        }
      }
      break;  
    }
  }
  return 0;
}

Instruction *InstCombiner::visitFPExt(CastInst &CI) {
  return commonCastTransforms(CI);
}

Instruction *InstCombiner::visitFPToUI(FPToUIInst &FI) {
  Instruction *OpI = dyn_cast<Instruction>(FI.getOperand(0));
  if (OpI == 0)
    return commonCastTransforms(FI);

  // fptoui(uitofp(X)) --> X
  // fptoui(sitofp(X)) --> X
  // This is safe if the intermediate type has enough bits in its mantissa to
  // accurately represent all values of X.  For example, do not do this with
  // i64->float->i64.  This is also safe for sitofp case, because any negative
  // 'X' value would cause an undefined result for the fptoui. 
  if ((isa<UIToFPInst>(OpI) || isa<SIToFPInst>(OpI)) &&
      OpI->getOperand(0)->getType() == FI.getType() &&
      (int)FI.getType()->getPrimitiveSizeInBits() < /*extra bit for sign */
                    OpI->getType()->getFPMantissaWidth())
    return ReplaceInstUsesWith(FI, OpI->getOperand(0));

  return commonCastTransforms(FI);
}

Instruction *InstCombiner::visitFPToSI(FPToSIInst &FI) {
  Instruction *OpI = dyn_cast<Instruction>(FI.getOperand(0));
  if (OpI == 0)
    return commonCastTransforms(FI);
  
  // fptosi(sitofp(X)) --> X
  // fptosi(uitofp(X)) --> X
  // This is safe if the intermediate type has enough bits in its mantissa to
  // accurately represent all values of X.  For example, do not do this with
  // i64->float->i64.  This is also safe for sitofp case, because any negative
  // 'X' value would cause an undefined result for the fptoui. 
  if ((isa<UIToFPInst>(OpI) || isa<SIToFPInst>(OpI)) &&
      OpI->getOperand(0)->getType() == FI.getType() &&
      (int)FI.getType()->getPrimitiveSizeInBits() <= 
                    OpI->getType()->getFPMantissaWidth())
    return ReplaceInstUsesWith(FI, OpI->getOperand(0));
  
  return commonCastTransforms(FI);
}

Instruction *InstCombiner::visitUIToFP(CastInst &CI) {
  return commonCastTransforms(CI);
}

Instruction *InstCombiner::visitSIToFP(CastInst &CI) {
  return commonCastTransforms(CI);
}

Instruction *InstCombiner::visitPtrToInt(CastInst &CI) {
  return commonPointerCastTransforms(CI);
}

Instruction *InstCombiner::visitIntToPtr(IntToPtrInst &CI) {
  if (Instruction *I = commonCastTransforms(CI))
    return I;
  
  const Type *DestPointee = cast<PointerType>(CI.getType())->getElementType();
  if (!DestPointee->isSized()) return 0;

  // If this is inttoptr(add (ptrtoint x), cst), try to turn this into a GEP.
  ConstantInt *Cst;
  Value *X;
  if (match(CI.getOperand(0), m_Add(m_Cast<PtrToIntInst>(m_Value(X)),
                                    m_ConstantInt(Cst)))) {
    // If the source and destination operands have the same type, see if this
    // is a single-index GEP.
    if (X->getType() == CI.getType()) {
      // Get the size of the pointee type.
      uint64_t Size = TD->getTypePaddedSize(DestPointee);

      // Convert the constant to intptr type.
      APInt Offset = Cst->getValue();
      Offset.sextOrTrunc(TD->getPointerSizeInBits());

      // If Offset is evenly divisible by Size, we can do this xform.
      if (Size && !APIntOps::srem(Offset, APInt(Offset.getBitWidth(), Size))){
        Offset = APIntOps::sdiv(Offset, APInt(Offset.getBitWidth(), Size));
        return GetElementPtrInst::Create(X, ConstantInt::get(Offset));
      }
    }
    // TODO: Could handle other cases, e.g. where add is indexing into field of
    // struct etc.
  } else if (CI.getOperand(0)->hasOneUse() &&
             match(CI.getOperand(0), m_Add(m_Value(X), m_ConstantInt(Cst)))) {
    // Otherwise, if this is inttoptr(add x, cst), try to turn this into an
    // "inttoptr+GEP" instead of "add+intptr".
    
    // Get the size of the pointee type.
    uint64_t Size = TD->getTypePaddedSize(DestPointee);
    
    // Convert the constant to intptr type.
    APInt Offset = Cst->getValue();
    Offset.sextOrTrunc(TD->getPointerSizeInBits());
    
    // If Offset is evenly divisible by Size, we can do this xform.
    if (Size && !APIntOps::srem(Offset, APInt(Offset.getBitWidth(), Size))){
      Offset = APIntOps::sdiv(Offset, APInt(Offset.getBitWidth(), Size));
      
      Instruction *P = InsertNewInstBefore(new IntToPtrInst(X, CI.getType(),
                                                            "tmp"), CI);
      return GetElementPtrInst::Create(P, ConstantInt::get(Offset), "tmp");
    }
  }
  return 0;
}

Instruction *InstCombiner::visitBitCast(BitCastInst &CI) {
  // If the operands are integer typed then apply the integer transforms,
  // otherwise just apply the common ones.
  Value *Src = CI.getOperand(0);
  const Type *SrcTy = Src->getType();
  const Type *DestTy = CI.getType();

  if (SrcTy->isInteger() && DestTy->isInteger()) {
    if (Instruction *Result = commonIntCastTransforms(CI))
      return Result;
  } else if (isa<PointerType>(SrcTy)) {
    if (Instruction *I = commonPointerCastTransforms(CI))
      return I;
  } else {
    if (Instruction *Result = commonCastTransforms(CI))
      return Result;
  }


  // Get rid of casts from one type to the same type. These are useless and can
  // be replaced by the operand.
  if (DestTy == Src->getType())
    return ReplaceInstUsesWith(CI, Src);

  if (const PointerType *DstPTy = dyn_cast<PointerType>(DestTy)) {
    const PointerType *SrcPTy = cast<PointerType>(SrcTy);
    const Type *DstElTy = DstPTy->getElementType();
    const Type *SrcElTy = SrcPTy->getElementType();
    
    // If the address spaces don't match, don't eliminate the bitcast, which is
    // required for changing types.
    if (SrcPTy->getAddressSpace() != DstPTy->getAddressSpace())
      return 0;
    
    // If we are casting a malloc or alloca to a pointer to a type of the same
    // size, rewrite the allocation instruction to allocate the "right" type.
    if (AllocationInst *AI = dyn_cast<AllocationInst>(Src))
      if (Instruction *V = PromoteCastOfAllocation(CI, *AI))
        return V;
    
    // If the source and destination are pointers, and this cast is equivalent
    // to a getelementptr X, 0, 0, 0...  turn it into the appropriate gep.
    // This can enhance SROA and other transforms that want type-safe pointers.
    Constant *ZeroUInt = Constant::getNullValue(Type::Int32Ty);
    unsigned NumZeros = 0;
    while (SrcElTy != DstElTy && 
           isa<CompositeType>(SrcElTy) && !isa<PointerType>(SrcElTy) &&
           SrcElTy->getNumContainedTypes() /* not "{}" */) {
      SrcElTy = cast<CompositeType>(SrcElTy)->getTypeAtIndex(ZeroUInt);
      ++NumZeros;
    }

    // If we found a path from the src to dest, create the getelementptr now.
    if (SrcElTy == DstElTy) {
      SmallVector<Value*, 8> Idxs(NumZeros+1, ZeroUInt);
      return GetElementPtrInst::Create(Src, Idxs.begin(), Idxs.end(), "", 
                                       ((Instruction*) NULL));
    }
  }

  if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(Src)) {
    if (SVI->hasOneUse()) {
      // Okay, we have (bitconvert (shuffle ..)).  Check to see if this is
      // a bitconvert to a vector with the same # elts.
      if (isa<VectorType>(DestTy) && 
          cast<VectorType>(DestTy)->getNumElements() ==
                SVI->getType()->getNumElements() &&
          SVI->getType()->getNumElements() ==
            cast<VectorType>(SVI->getOperand(0)->getType())->getNumElements()) {
        CastInst *Tmp;
        // If either of the operands is a cast from CI.getType(), then
        // evaluating the shuffle in the casted destination's type will allow
        // us to eliminate at least one cast.
        if (((Tmp = dyn_cast<CastInst>(SVI->getOperand(0))) && 
             Tmp->getOperand(0)->getType() == DestTy) ||
            ((Tmp = dyn_cast<CastInst>(SVI->getOperand(1))) && 
             Tmp->getOperand(0)->getType() == DestTy)) {
          Value *LHS = InsertCastBefore(Instruction::BitCast,
                                        SVI->getOperand(0), DestTy, CI);
          Value *RHS = InsertCastBefore(Instruction::BitCast,
                                        SVI->getOperand(1), DestTy, CI);
          // Return a new shuffle vector.  Use the same element ID's, as we
          // know the vector types match #elts.
          return new ShuffleVectorInst(LHS, RHS, SVI->getOperand(2));
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
  case Instruction::LShr:
  case Instruction::AShr:
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
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    return Constant::getNullValue(I->getType());
  case Instruction::And:
    return Constant::getAllOnesValue(I->getType());
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
    if (TI->isCast()) {
      if (TI->getOperand(0)->getType() != FI->getOperand(0)->getType())
        return 0;
    } else {
      return 0;  // unknown unary op.
    }

    // Fold this by inserting a select from the input values.
    SelectInst *NewSI = SelectInst::Create(SI.getCondition(), TI->getOperand(0),
                                           FI->getOperand(0), SI.getName()+".v");
    InsertNewInstBefore(NewSI, SI);
    return CastInst::Create(Instruction::CastOps(TI->getOpcode()), NewSI, 
                            TI->getType());
  }

  // Only handle binary operators here.
  if (!isa<BinaryOperator>(TI))
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
  SelectInst *NewSI = SelectInst::Create(SI.getCondition(), OtherOpT,
                                         OtherOpF, SI.getName()+".v");
  InsertNewInstBefore(NewSI, SI);

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TI)) {
    if (MatchIsOpZero)
      return BinaryOperator::Create(BO->getOpcode(), MatchOp, NewSI);
    else
      return BinaryOperator::Create(BO->getOpcode(), NewSI, MatchOp);
  }
  assert(0 && "Shouldn't get here");
  return 0;
}

/// visitSelectInstWithICmp - Visit a SelectInst that has an
/// ICmpInst as its first operand.
///
Instruction *InstCombiner::visitSelectInstWithICmp(SelectInst &SI,
                                                   ICmpInst *ICI) {
  bool Changed = false;
  ICmpInst::Predicate Pred = ICI->getPredicate();
  Value *CmpLHS = ICI->getOperand(0);
  Value *CmpRHS = ICI->getOperand(1);
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  // Check cases where the comparison is with a constant that
  // can be adjusted to fit the min/max idiom. We may edit ICI in
  // place here, so make sure the select is the only user.
  if (ICI->hasOneUse())
    if (ConstantInt *CI = dyn_cast<ConstantInt>(CmpRHS)) {
      switch (Pred) {
      default: break;
      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_SLT: {
        // X < MIN ? T : F  -->  F
        if (CI->isMinValue(Pred == ICmpInst::ICMP_SLT))
          return ReplaceInstUsesWith(SI, FalseVal);
        // X < C ? X : C-1  -->  X > C-1 ? C-1 : X
        Constant *AdjustedRHS = SubOne(CI);
        if ((CmpLHS == TrueVal && AdjustedRHS == FalseVal) ||
            (CmpLHS == FalseVal && AdjustedRHS == TrueVal)) {
          Pred = ICmpInst::getSwappedPredicate(Pred);
          CmpRHS = AdjustedRHS;
          std::swap(FalseVal, TrueVal);
          ICI->setPredicate(Pred);
          ICI->setOperand(1, CmpRHS);
          SI.setOperand(1, TrueVal);
          SI.setOperand(2, FalseVal);
          Changed = true;
        }
        break;
      }
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_SGT: {
        // X > MAX ? T : F  -->  F
        if (CI->isMaxValue(Pred == ICmpInst::ICMP_SGT))
          return ReplaceInstUsesWith(SI, FalseVal);
        // X > C ? X : C+1  -->  X < C+1 ? C+1 : X
        Constant *AdjustedRHS = AddOne(CI);
        if ((CmpLHS == TrueVal && AdjustedRHS == FalseVal) ||
            (CmpLHS == FalseVal && AdjustedRHS == TrueVal)) {
          Pred = ICmpInst::getSwappedPredicate(Pred);
          CmpRHS = AdjustedRHS;
          std::swap(FalseVal, TrueVal);
          ICI->setPredicate(Pred);
          ICI->setOperand(1, CmpRHS);
          SI.setOperand(1, TrueVal);
          SI.setOperand(2, FalseVal);
          Changed = true;
        }
        break;
      }
      }

      // (x <s 0) ? -1 : 0 -> ashr x, 31   -> all ones if signed
      // (x >s -1) ? -1 : 0 -> ashr x, 31  -> all ones if not signed
      CmpInst::Predicate Pred = CmpInst::BAD_ICMP_PREDICATE;
      if (match(TrueVal, m_ConstantInt<-1>()) &&
          match(FalseVal, m_ConstantInt<0>()))
        Pred = ICI->getPredicate();
      else if (match(TrueVal, m_ConstantInt<0>()) &&
               match(FalseVal, m_ConstantInt<-1>()))
        Pred = CmpInst::getInversePredicate(ICI->getPredicate());
      
      if (Pred != CmpInst::BAD_ICMP_PREDICATE) {
        // If we are just checking for a icmp eq of a single bit and zext'ing it
        // to an integer, then shift the bit to the appropriate place and then
        // cast to integer to avoid the comparison.
        const APInt &Op1CV = CI->getValue();
    
        // sext (x <s  0) to i32 --> x>>s31      true if signbit set.
        // sext (x >s -1) to i32 --> (x>>s31)^-1  true if signbit clear.
        if ((Pred == ICmpInst::ICMP_SLT && Op1CV == 0) ||
            (Pred == ICmpInst::ICMP_SGT && Op1CV.isAllOnesValue())) {
          Value *In = ICI->getOperand(0);
          Value *Sh = ConstantInt::get(In->getType(),
                                       In->getType()->getPrimitiveSizeInBits()-1);
          In = InsertNewInstBefore(BinaryOperator::CreateAShr(In, Sh,
                                                          In->getName()+".lobit"),
                                   *ICI);
          if (In->getType() != SI.getType())
            In = CastInst::CreateIntegerCast(In, SI.getType(),
                                             true/*SExt*/, "tmp", ICI);
    
          if (Pred == ICmpInst::ICMP_SGT)
            In = InsertNewInstBefore(BinaryOperator::CreateNot(In,
                                       In->getName()+".not"), *ICI);
    
          return ReplaceInstUsesWith(SI, In);
        }
      }
    }

  if (CmpLHS == TrueVal && CmpRHS == FalseVal) {
    // Transform (X == Y) ? X : Y  -> Y
    if (Pred == ICmpInst::ICMP_EQ)
      return ReplaceInstUsesWith(SI, FalseVal);
    // Transform (X != Y) ? X : Y  -> X
    if (Pred == ICmpInst::ICMP_NE)
      return ReplaceInstUsesWith(SI, TrueVal);
    /// NOTE: if we wanted to, this is where to detect integer MIN/MAX

  } else if (CmpLHS == FalseVal && CmpRHS == TrueVal) {
    // Transform (X == Y) ? Y : X  -> X
    if (Pred == ICmpInst::ICMP_EQ)
      return ReplaceInstUsesWith(SI, FalseVal);
    // Transform (X != Y) ? Y : X  -> Y
    if (Pred == ICmpInst::ICMP_NE)
      return ReplaceInstUsesWith(SI, TrueVal);
    /// NOTE: if we wanted to, this is where to detect integer MIN/MAX
  }

  /// NOTE: if we wanted to, this is where to detect integer ABS

  return Changed ? &SI : 0;
}

Instruction *InstCombiner::visitSelectInst(SelectInst &SI) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  // select true, X, Y  -> X
  // select false, X, Y -> Y
  if (ConstantInt *C = dyn_cast<ConstantInt>(CondVal))
    return ReplaceInstUsesWith(SI, C->getZExtValue() ? TrueVal : FalseVal);

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

  if (SI.getType() == Type::Int1Ty) {
    if (ConstantInt *C = dyn_cast<ConstantInt>(TrueVal)) {
      if (C->getZExtValue()) {
        // Change: A = select B, true, C --> A = or B, C
        return BinaryOperator::CreateOr(CondVal, FalseVal);
      } else {
        // Change: A = select B, false, C --> A = and !B, C
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::CreateNot(CondVal,
                                             "not."+CondVal->getName()), SI);
        return BinaryOperator::CreateAnd(NotCond, FalseVal);
      }
    } else if (ConstantInt *C = dyn_cast<ConstantInt>(FalseVal)) {
      if (C->getZExtValue() == false) {
        // Change: A = select B, C, false --> A = and B, C
        return BinaryOperator::CreateAnd(CondVal, TrueVal);
      } else {
        // Change: A = select B, C, true --> A = or !B, C
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::CreateNot(CondVal,
                                             "not."+CondVal->getName()), SI);
        return BinaryOperator::CreateOr(NotCond, TrueVal);
      }
    }
    
    // select a, b, a  -> a&b
    // select a, a, b  -> a|b
    if (CondVal == TrueVal)
      return BinaryOperator::CreateOr(CondVal, FalseVal);
    else if (CondVal == FalseVal)
      return BinaryOperator::CreateAnd(CondVal, TrueVal);
  }

  // Selecting between two integer constants?
  if (ConstantInt *TrueValC = dyn_cast<ConstantInt>(TrueVal))
    if (ConstantInt *FalseValC = dyn_cast<ConstantInt>(FalseVal)) {
      // select C, 1, 0 -> zext C to int
      if (FalseValC->isZero() && TrueValC->getValue() == 1) {
        return CastInst::Create(Instruction::ZExt, CondVal, SI.getType());
      } else if (TrueValC->isZero() && FalseValC->getValue() == 1) {
        // select C, 0, 1 -> zext !C to int
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::CreateNot(CondVal,
                                               "not."+CondVal->getName()), SI);
        return CastInst::Create(Instruction::ZExt, NotCond, SI.getType());
      }

      if (ICmpInst *IC = dyn_cast<ICmpInst>(SI.getCondition())) {

        // (x <s 0) ? -1 : 0 -> ashr x, 31
        if (TrueValC->isAllOnesValue() && FalseValC->isZero())
          if (ConstantInt *CmpCst = dyn_cast<ConstantInt>(IC->getOperand(1))) {
            if (IC->getPredicate() == ICmpInst::ICMP_SLT && CmpCst->isZero()) {
              // The comparison constant and the result are not neccessarily the
              // same width. Make an all-ones value by inserting a AShr.
              Value *X = IC->getOperand(0);
              uint32_t Bits = X->getType()->getPrimitiveSizeInBits();
              Constant *ShAmt = ConstantInt::get(X->getType(), Bits-1);
              Instruction *SRA = BinaryOperator::Create(Instruction::AShr, X,
                                                        ShAmt, "ones");
              InsertNewInstBefore(SRA, SI);

              // Then cast to the appropriate width.
              return CastInst::CreateIntegerCast(SRA, SI.getType(), true);
            }
          }


        // If one of the constants is zero (we know they can't both be) and we
        // have an icmp instruction with zero, and we have an 'and' with the
        // non-constant value, eliminate this whole mess.  This corresponds to
        // cases like this: ((X & 27) ? 27 : 0)
        if (TrueValC->isZero() || FalseValC->isZero())
          if (IC->isEquality() && isa<ConstantInt>(IC->getOperand(1)) &&
              cast<Constant>(IC->getOperand(1))->isNullValue())
            if (Instruction *ICA = dyn_cast<Instruction>(IC->getOperand(0)))
              if (ICA->getOpcode() == Instruction::And &&
                  isa<ConstantInt>(ICA->getOperand(1)) &&
                  (ICA->getOperand(1) == TrueValC ||
                   ICA->getOperand(1) == FalseValC) &&
                  isOneBitSet(cast<ConstantInt>(ICA->getOperand(1)))) {
                // Okay, now we know that everything is set up, we just don't
                // know whether we have a icmp_ne or icmp_eq and whether the 
                // true or false val is the zero.
                bool ShouldNotVal = !TrueValC->isZero();
                ShouldNotVal ^= IC->getPredicate() == ICmpInst::ICMP_NE;
                Value *V = ICA;
                if (ShouldNotVal)
                  V = InsertNewInstBefore(BinaryOperator::Create(
                                  Instruction::Xor, V, ICA->getOperand(1)), SI);
                return ReplaceInstUsesWith(SI, V);
              }
      }
    }

  // See if we are selecting two values based on a comparison of the two values.
  if (FCmpInst *FCI = dyn_cast<FCmpInst>(CondVal)) {
    if (FCI->getOperand(0) == TrueVal && FCI->getOperand(1) == FalseVal) {
      // Transform (X == Y) ? X : Y  -> Y
      if (FCI->getPredicate() == FCmpInst::FCMP_OEQ) {
        // This is not safe in general for floating point:  
        // consider X== -0, Y== +0.
        // It becomes safe if either operand is a nonzero constant.
        ConstantFP *CFPt, *CFPf;
        if (((CFPt = dyn_cast<ConstantFP>(TrueVal)) &&
              !CFPt->getValueAPF().isZero()) ||
            ((CFPf = dyn_cast<ConstantFP>(FalseVal)) &&
             !CFPf->getValueAPF().isZero()))
        return ReplaceInstUsesWith(SI, FalseVal);
      }
      // Transform (X != Y) ? X : Y  -> X
      if (FCI->getPredicate() == FCmpInst::FCMP_ONE)
        return ReplaceInstUsesWith(SI, TrueVal);
      // NOTE: if we wanted to, this is where to detect MIN/MAX

    } else if (FCI->getOperand(0) == FalseVal && FCI->getOperand(1) == TrueVal){
      // Transform (X == Y) ? Y : X  -> X
      if (FCI->getPredicate() == FCmpInst::FCMP_OEQ) {
        // This is not safe in general for floating point:  
        // consider X== -0, Y== +0.
        // It becomes safe if either operand is a nonzero constant.
        ConstantFP *CFPt, *CFPf;
        if (((CFPt = dyn_cast<ConstantFP>(TrueVal)) &&
              !CFPt->getValueAPF().isZero()) ||
            ((CFPf = dyn_cast<ConstantFP>(FalseVal)) &&
             !CFPf->getValueAPF().isZero()))
          return ReplaceInstUsesWith(SI, FalseVal);
      }
      // Transform (X != Y) ? Y : X  -> Y
      if (FCI->getPredicate() == FCmpInst::FCMP_ONE)
        return ReplaceInstUsesWith(SI, TrueVal);
      // NOTE: if we wanted to, this is where to detect MIN/MAX
    }
    // NOTE: if we wanted to, this is where to detect ABS
  }

  // See if we are selecting two values based on a comparison of the two values.
  if (ICmpInst *ICI = dyn_cast<ICmpInst>(CondVal))
    if (Instruction *Result = visitSelectInstWithICmp(SI, ICI))
      return Result;

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
                    BinaryOperator::CreateNeg(SubOp->getOperand(1), "tmp"), SI);
            }

            Value *NewTrueOp = OtherAddOp;
            Value *NewFalseOp = NegVal;
            if (AddOp != TI)
              std::swap(NewTrueOp, NewFalseOp);
            Instruction *NewSel =
              SelectInst::Create(CondVal, NewTrueOp,
                                 NewFalseOp, SI.getName() + ".p");

            NewSel = InsertNewInstBefore(NewSel, SI);
            return BinaryOperator::CreateAdd(SubOp->getOperand(0), NewSel);
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
            Instruction *NewSel =
              SelectInst::Create(SI.getCondition(),
                                 TVI->getOperand(2-OpToFold), C);
            InsertNewInstBefore(NewSel, SI);
            NewSel->takeName(TVI);
            if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TVI))
              return BinaryOperator::Create(BO->getOpcode(), FalseVal, NewSel);
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
            Instruction *NewSel =
              SelectInst::Create(SI.getCondition(), C,
                                 FVI->getOperand(2-OpToFold));
            InsertNewInstBefore(NewSel, SI);
            NewSel->takeName(FVI);
            if (BinaryOperator *BO = dyn_cast<BinaryOperator>(FVI))
              return BinaryOperator::Create(BO->getOpcode(), TrueVal, NewSel);
            else
              assert(0 && "Unknown instruction!!");
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

/// EnforceKnownAlignment - If the specified pointer points to an object that
/// we control, modify the object's alignment to PrefAlign. This isn't
/// often possible though. If alignment is important, a more reliable approach
/// is to simply align all global variables and allocation instructions to
/// their preferred alignment from the beginning.
///
static unsigned EnforceKnownAlignment(Value *V,
                                      unsigned Align, unsigned PrefAlign) {

  User *U = dyn_cast<User>(V);
  if (!U) return Align;

  switch (getOpcode(U)) {
  default: break;
  case Instruction::BitCast:
    return EnforceKnownAlignment(U->getOperand(0), Align, PrefAlign);
  case Instruction::GetElementPtr: {
    // If all indexes are zero, it is just the alignment of the base pointer.
    bool AllZeroOperands = true;
    for (User::op_iterator i = U->op_begin() + 1, e = U->op_end(); i != e; ++i)
      if (!isa<Constant>(*i) ||
          !cast<Constant>(*i)->isNullValue()) {
        AllZeroOperands = false;
        break;
      }

    if (AllZeroOperands) {
      // Treat this like a bitcast.
      return EnforceKnownAlignment(U->getOperand(0), Align, PrefAlign);
    }
    break;
  }
  }

  if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    // If there is a large requested alignment and we can, bump up the alignment
    // of the global.
    if (!GV->isDeclaration()) {
      GV->setAlignment(PrefAlign);
      Align = PrefAlign;
    }
  } else if (AllocationInst *AI = dyn_cast<AllocationInst>(V)) {
    // If there is a requested alignment and if this is an alloca, round up.  We
    // don't do this for malloc, because some systems can't respect the request.
    if (isa<AllocaInst>(AI)) {
      AI->setAlignment(PrefAlign);
      Align = PrefAlign;
    }
  }

  return Align;
}

/// GetOrEnforceKnownAlignment - If the specified pointer has an alignment that
/// we can determine, return it, otherwise return 0.  If PrefAlign is specified,
/// and it is more than the alignment of the ultimate object, see if we can
/// increase the alignment of the ultimate object, making this check succeed.
unsigned InstCombiner::GetOrEnforceKnownAlignment(Value *V,
                                                  unsigned PrefAlign) {
  unsigned BitWidth = TD ? TD->getTypeSizeInBits(V->getType()) :
                      sizeof(PrefAlign) * CHAR_BIT;
  APInt Mask = APInt::getAllOnesValue(BitWidth);
  APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
  ComputeMaskedBits(V, Mask, KnownZero, KnownOne);
  unsigned TrailZ = KnownZero.countTrailingOnes();
  unsigned Align = 1u << std::min(BitWidth - 1, TrailZ);

  if (PrefAlign > Align)
    Align = EnforceKnownAlignment(V, Align, PrefAlign);
  
    // We don't need to make any adjustment.
  return Align;
}

Instruction *InstCombiner::SimplifyMemTransfer(MemIntrinsic *MI) {
  unsigned DstAlign = GetOrEnforceKnownAlignment(MI->getOperand(1));
  unsigned SrcAlign = GetOrEnforceKnownAlignment(MI->getOperand(2));
  unsigned MinAlign = std::min(DstAlign, SrcAlign);
  unsigned CopyAlign = MI->getAlignment()->getZExtValue();

  if (CopyAlign < MinAlign) {
    MI->setAlignment(ConstantInt::get(Type::Int32Ty, MinAlign));
    return MI;
  }
  
  // If MemCpyInst length is 1/2/4/8 bytes then replace memcpy with
  // load/store.
  ConstantInt *MemOpLength = dyn_cast<ConstantInt>(MI->getOperand(3));
  if (MemOpLength == 0) return 0;
  
  // Source and destination pointer types are always "i8*" for intrinsic.  See
  // if the size is something we can handle with a single primitive load/store.
  // A single load+store correctly handles overlapping memory in the memmove
  // case.
  unsigned Size = MemOpLength->getZExtValue();
  if (Size == 0) return MI;  // Delete this mem transfer.
  
  if (Size > 8 || (Size&(Size-1)))
    return 0;  // If not 1/2/4/8 bytes, exit.
  
  // Use an integer load+store unless we can find something better.
  Type *NewPtrTy = PointerType::getUnqual(IntegerType::get(Size<<3));
  
  // Memcpy forces the use of i8* for the source and destination.  That means
  // that if you're using memcpy to move one double around, you'll get a cast
  // from double* to i8*.  We'd much rather use a double load+store rather than
  // an i64 load+store, here because this improves the odds that the source or
  // dest address will be promotable.  See if we can find a better type than the
  // integer datatype.
  if (Value *Op = getBitCastOperand(MI->getOperand(1))) {
    const Type *SrcETy = cast<PointerType>(Op->getType())->getElementType();
    if (SrcETy->isSized() && TD->getTypeStoreSize(SrcETy) == Size) {
      // The SrcETy might be something like {{{double}}} or [1 x double].  Rip
      // down through these levels if so.
      while (!SrcETy->isSingleValueType()) {
        if (const StructType *STy = dyn_cast<StructType>(SrcETy)) {
          if (STy->getNumElements() == 1)
            SrcETy = STy->getElementType(0);
          else
            break;
        } else if (const ArrayType *ATy = dyn_cast<ArrayType>(SrcETy)) {
          if (ATy->getNumElements() == 1)
            SrcETy = ATy->getElementType();
          else
            break;
        } else
          break;
      }
      
      if (SrcETy->isSingleValueType())
        NewPtrTy = PointerType::getUnqual(SrcETy);
    }
  }
  
  
  // If the memcpy/memmove provides better alignment info than we can
  // infer, use it.
  SrcAlign = std::max(SrcAlign, CopyAlign);
  DstAlign = std::max(DstAlign, CopyAlign);
  
  Value *Src = InsertBitCastBefore(MI->getOperand(2), NewPtrTy, *MI);
  Value *Dest = InsertBitCastBefore(MI->getOperand(1), NewPtrTy, *MI);
  Instruction *L = new LoadInst(Src, "tmp", false, SrcAlign);
  InsertNewInstBefore(L, *MI);
  InsertNewInstBefore(new StoreInst(L, Dest, false, DstAlign), *MI);

  // Set the size of the copy to 0, it will be deleted on the next iteration.
  MI->setOperand(3, Constant::getNullValue(MemOpLength->getType()));
  return MI;
}

Instruction *InstCombiner::SimplifyMemSet(MemSetInst *MI) {
  unsigned Alignment = GetOrEnforceKnownAlignment(MI->getDest());
  if (MI->getAlignment()->getZExtValue() < Alignment) {
    MI->setAlignment(ConstantInt::get(Type::Int32Ty, Alignment));
    return MI;
  }
  
  // Extract the length and alignment and fill if they are constant.
  ConstantInt *LenC = dyn_cast<ConstantInt>(MI->getLength());
  ConstantInt *FillC = dyn_cast<ConstantInt>(MI->getValue());
  if (!LenC || !FillC || FillC->getType() != Type::Int8Ty)
    return 0;
  uint64_t Len = LenC->getZExtValue();
  Alignment = MI->getAlignment()->getZExtValue();
  
  // If the length is zero, this is a no-op
  if (Len == 0) return MI; // memset(d,c,0,a) -> noop
  
  // memset(s,c,n) -> store s, c (for n=1,2,4,8)
  if (Len <= 8 && isPowerOf2_32((uint32_t)Len)) {
    const Type *ITy = IntegerType::get(Len*8);  // n=1 -> i8.
    
    Value *Dest = MI->getDest();
    Dest = InsertBitCastBefore(Dest, PointerType::getUnqual(ITy), *MI);

    // Alignment 0 is identity for alignment 1 for memset, but not store.
    if (Alignment == 0) Alignment = 1;
    
    // Extract the fill value and store.
    uint64_t Fill = FillC->getZExtValue()*0x0101010101010101ULL;
    InsertNewInstBefore(new StoreInst(ConstantInt::get(ITy, Fill), Dest, false,
                                      Alignment), *MI);
    
    // Set the size of the copy to 0, it will be deleted on the next iteration.
    MI->setLength(Constant::getNullValue(LenC->getType()));
    return MI;
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
    if (MemMoveInst *MMI = dyn_cast<MemMoveInst>(MI)) {
      if (GlobalVariable *GVSrc = dyn_cast<GlobalVariable>(MMI->getSource()))
        if (GVSrc->isConstant()) {
          Module *M = CI.getParent()->getParent()->getParent();
          Intrinsic::ID MemCpyID = Intrinsic::memcpy;
          const Type *Tys[1];
          Tys[0] = CI.getOperand(3)->getType();
          CI.setOperand(0, 
                        Intrinsic::getDeclaration(M, MemCpyID, Tys, 1));
          Changed = true;
        }

      // memmove(x,x,size) -> noop.
      if (MMI->getSource() == MMI->getDest())
        return EraseInstFromFunction(CI);
    }

    // If we can determine a pointer alignment that is bigger than currently
    // set, update the alignment.
    if (isa<MemCpyInst>(MI) || isa<MemMoveInst>(MI)) {
      if (Instruction *I = SimplifyMemTransfer(MI))
        return I;
    } else if (MemSetInst *MSI = dyn_cast<MemSetInst>(MI)) {
      if (Instruction *I = SimplifyMemSet(MSI))
        return I;
    }
          
    if (Changed) return II;
  }
  
  switch (II->getIntrinsicID()) {
  default: break;
  case Intrinsic::bswap:
    // bswap(bswap(x)) -> x
    if (IntrinsicInst *Operand = dyn_cast<IntrinsicInst>(II->getOperand(1)))
      if (Operand->getIntrinsicID() == Intrinsic::bswap)
        return ReplaceInstUsesWith(CI, Operand->getOperand(1));
    break;
  case Intrinsic::ppc_altivec_lvx:
  case Intrinsic::ppc_altivec_lvxl:
  case Intrinsic::x86_sse_loadu_ps:
  case Intrinsic::x86_sse2_loadu_pd:
  case Intrinsic::x86_sse2_loadu_dq:
    // Turn PPC lvx     -> load if the pointer is known aligned.
    // Turn X86 loadups -> load if the pointer is known aligned.
    if (GetOrEnforceKnownAlignment(II->getOperand(1), 16) >= 16) {
      Value *Ptr = InsertBitCastBefore(II->getOperand(1),
                                       PointerType::getUnqual(II->getType()),
                                       CI);
      return new LoadInst(Ptr);
    }
    break;
  case Intrinsic::ppc_altivec_stvx:
  case Intrinsic::ppc_altivec_stvxl:
    // Turn stvx -> store if the pointer is known aligned.
    if (GetOrEnforceKnownAlignment(II->getOperand(2), 16) >= 16) {
      const Type *OpPtrTy = 
        PointerType::getUnqual(II->getOperand(1)->getType());
      Value *Ptr = InsertBitCastBefore(II->getOperand(2), OpPtrTy, CI);
      return new StoreInst(II->getOperand(1), Ptr);
    }
    break;
  case Intrinsic::x86_sse_storeu_ps:
  case Intrinsic::x86_sse2_storeu_pd:
  case Intrinsic::x86_sse2_storeu_dq:
    // Turn X86 storeu -> store if the pointer is known aligned.
    if (GetOrEnforceKnownAlignment(II->getOperand(1), 16) >= 16) {
      const Type *OpPtrTy = 
        PointerType::getUnqual(II->getOperand(2)->getType());
      Value *Ptr = InsertBitCastBefore(II->getOperand(1), OpPtrTy, CI);
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
    if (ConstantVector *Mask = dyn_cast<ConstantVector>(II->getOperand(3))) {
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
        Value *Op0 =InsertBitCastBefore(II->getOperand(1),Mask->getType(),CI);
        Value *Op1 =InsertBitCastBefore(II->getOperand(2),Mask->getType(),CI);
        Value *Result = UndefValue::get(Op0->getType());
        
        // Only extract each element once.
        Value *ExtractedElts[32];
        memset(ExtractedElts, 0, sizeof(ExtractedElts));
        
        for (unsigned i = 0; i != 16; ++i) {
          if (isa<UndefValue>(Mask->getOperand(i)))
            continue;
          unsigned Idx=cast<ConstantInt>(Mask->getOperand(i))->getZExtValue();
          Idx &= 31;  // Match the hardware behavior.
          
          if (ExtractedElts[Idx] == 0) {
            Instruction *Elt = 
              new ExtractElementInst(Idx < 16 ? Op0 : Op1, Idx&15, "tmp");
            InsertNewInstBefore(Elt, CI);
            ExtractedElts[Idx] = Elt;
          }
        
          // Insert this value into the result vector.
          Result = InsertElementInst::Create(Result, ExtractedElts[Idx],
                                             i, "tmp");
          InsertNewInstBefore(cast<Instruction>(Result), CI);
        }
        return CastInst::Create(Instruction::BitCast, Result, CI.getType());
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
    
    // Scan down this block to see if there is another stack restore in the
    // same block without an intervening call/alloca.
    BasicBlock::iterator BI = II;
    TerminatorInst *TI = II->getParent()->getTerminator();
    bool CannotRemove = false;
    for (++BI; &*BI != TI; ++BI) {
      if (isa<AllocaInst>(BI)) {
        CannotRemove = true;
        break;
      }
      if (CallInst *BCI = dyn_cast<CallInst>(BI)) {
        if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(BCI)) {
          // If there is a stackrestore below this one, remove this one.
          if (II->getIntrinsicID() == Intrinsic::stackrestore)
            return EraseInstFromFunction(CI);
          // Otherwise, ignore the intrinsic.
        } else {
          // If we found a non-intrinsic call, we can't remove the stack
          // restore.
          CannotRemove = true;
          break;
        }
      }
    }
    
    // If the stack restore is in a return/unwind block and if there are no
    // allocas or calls between the restore and the return, nuke the restore.
    if (!CannotRemove && (isa<ReturnInst>(TI) || isa<UnwindInst>(TI)))
      return EraseInstFromFunction(CI);
    break;
  }
  }

  return visitCallSite(II);
}

// InvokeInst simplification
//
Instruction *InstCombiner::visitInvokeInst(InvokeInst &II) {
  return visitCallSite(&II);
}

/// isSafeToEliminateVarargsCast - If this cast does not affect the value 
/// passed through the varargs area, we can eliminate the use of the cast.
static bool isSafeToEliminateVarargsCast(const CallSite CS,
                                         const CastInst * const CI,
                                         const TargetData * const TD,
                                         const int ix) {
  if (!CI->isLosslessCast())
    return false;

  // The size of ByVal arguments is derived from the type, so we
  // can't change to a type with a different size.  If the size were
  // passed explicitly we could avoid this check.
  if (!CS.paramHasAttr(ix, Attribute::ByVal))
    return true;

  const Type* SrcTy = 
            cast<PointerType>(CI->getOperand(0)->getType())->getElementType();
  const Type* DstTy = cast<PointerType>(CI->getType())->getElementType();
  if (!SrcTy->isSized() || !DstTy->isSized())
    return false;
  if (TD->getTypePaddedSize(SrcTy) != TD->getTypePaddedSize(DstTy))
    return false;
  return true;
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
      new StoreInst(ConstantInt::getTrue(),
                    UndefValue::get(PointerType::getUnqual(Type::Int1Ty)), 
                                    OldCall);
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
    new StoreInst(ConstantInt::getTrue(),
                  UndefValue::get(PointerType::getUnqual(Type::Int1Ty)),
                  CS.getInstruction());

    if (!CS.getInstruction()->use_empty())
      CS.getInstruction()->
        replaceAllUsesWith(UndefValue::get(CS.getInstruction()->getType()));

    if (InvokeInst *II = dyn_cast<InvokeInst>(CS.getInstruction())) {
      // Don't break the CFG, insert a dummy cond branch.
      BranchInst::Create(II->getNormalDest(), II->getUnwindDest(),
                         ConstantInt::getTrue(), II);
    }
    return EraseInstFromFunction(*CS.getInstruction());
  }

  if (BitCastInst *BC = dyn_cast<BitCastInst>(Callee))
    if (IntrinsicInst *In = dyn_cast<IntrinsicInst>(BC->getOperand(0)))
      if (In->getIntrinsicID() == Intrinsic::init_trampoline)
        return transformCallThroughTrampoline(CS);

  const PointerType *PTy = cast<PointerType>(Callee->getType());
  const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  if (FTy->isVarArg()) {
    int ix = FTy->getNumParams() + (isa<InvokeInst>(Callee) ? 3 : 1);
    // See if we can optimize any arguments passed through the varargs area of
    // the call.
    for (CallSite::arg_iterator I = CS.arg_begin()+FTy->getNumParams(),
           E = CS.arg_end(); I != E; ++I, ++ix) {
      CastInst *CI = dyn_cast<CastInst>(*I);
      if (CI && isSafeToEliminateVarargsCast(CS, CI, TD, ix)) {
        *I = CI->getOperand(0);
        Changed = true;
      }
    }
  }

  if (isa<InlineAsm>(Callee) && !CS.doesNotThrow()) {
    // Inline asm calls cannot throw - mark them 'nounwind'.
    CS.setDoesNotThrow();
    Changed = true;
  }

  return Changed ? CS.getInstruction() : 0;
}

// transformConstExprCastCall - If the callee is a constexpr cast of a function,
// attempt to move the cast to the arguments of the call/invoke.
//
bool InstCombiner::transformConstExprCastCall(CallSite CS) {
  if (!isa<ConstantExpr>(CS.getCalledValue())) return false;
  ConstantExpr *CE = cast<ConstantExpr>(CS.getCalledValue());
  if (CE->getOpcode() != Instruction::BitCast || 
      !isa<Function>(CE->getOperand(0)))
    return false;
  Function *Callee = cast<Function>(CE->getOperand(0));
  Instruction *Caller = CS.getInstruction();
  const AttrListPtr &CallerPAL = CS.getAttributes();

  // Okay, this is a cast from a function to a different type.  Unless doing so
  // would cause a type conversion of one of our arguments, change this call to
  // be a direct call with arguments casted to the appropriate types.
  //
  const FunctionType *FT = Callee->getFunctionType();
  const Type *OldRetTy = Caller->getType();
  const Type *NewRetTy = FT->getReturnType();

  if (isa<StructType>(NewRetTy))
    return false; // TODO: Handle multiple return values.

  // Check to see if we are changing the return type...
  if (OldRetTy != NewRetTy) {
    if (Callee->isDeclaration() &&
        // Conversion is ok if changing from one pointer type to another or from
        // a pointer to an integer of the same size.
        !((isa<PointerType>(OldRetTy) || OldRetTy == TD->getIntPtrType()) &&
          (isa<PointerType>(NewRetTy) || NewRetTy == TD->getIntPtrType())))
      return false;   // Cannot transform this return value.

    if (!Caller->use_empty() &&
        // void -> non-void is handled specially
        NewRetTy != Type::VoidTy && !CastInst::isCastable(NewRetTy, OldRetTy))
      return false;   // Cannot transform this return value.

    if (!CallerPAL.isEmpty() && !Caller->use_empty()) {
      Attributes RAttrs = CallerPAL.getRetAttributes();
      if (RAttrs & Attribute::typeIncompatible(NewRetTy))
        return false;   // Attribute not compatible with transformed value.
    }

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

    if (!CastInst::isCastable(ActTy, ParamTy))
      return false;   // Cannot transform this parameter value.

    if (CallerPAL.getParamAttributes(i + 1) 
        & Attribute::typeIncompatible(ParamTy))
      return false;   // Attribute not compatible with transformed value.

    // Converting from one pointer type to another or between a pointer and an
    // integer of the same size is safe even if we do not have a body.
    bool isConvertible = ActTy == ParamTy ||
      ((isa<PointerType>(ParamTy) || ParamTy == TD->getIntPtrType()) &&
       (isa<PointerType>(ActTy) || ActTy == TD->getIntPtrType()));
    if (Callee->isDeclaration() && !isConvertible) return false;
  }

  if (FT->getNumParams() < NumActualArgs && !FT->isVarArg() &&
      Callee->isDeclaration())
    return false;   // Do not delete arguments unless we have a function body.

  if (FT->getNumParams() < NumActualArgs && FT->isVarArg() &&
      !CallerPAL.isEmpty())
    // In this case we have more arguments than the new function type, but we
    // won't be dropping them.  Check that these extra arguments have attributes
    // that are compatible with being a vararg call argument.
    for (unsigned i = CallerPAL.getNumSlots(); i; --i) {
      if (CallerPAL.getSlot(i - 1).Index <= FT->getNumParams())
        break;
      Attributes PAttrs = CallerPAL.getSlot(i - 1).Attrs;
      if (PAttrs & Attribute::VarArgsIncompatible)
        return false;
    }

  // Okay, we decided that this is a safe thing to do: go ahead and start
  // inserting cast instructions as necessary...
  std::vector<Value*> Args;
  Args.reserve(NumActualArgs);
  SmallVector<AttributeWithIndex, 8> attrVec;
  attrVec.reserve(NumCommonArgs);

  // Get any return attributes.
  Attributes RAttrs = CallerPAL.getRetAttributes();

  // If the return value is not being used, the type may not be compatible
  // with the existing attributes.  Wipe out any problematic attributes.
  RAttrs &= ~Attribute::typeIncompatible(NewRetTy);

  // Add the new return attributes.
  if (RAttrs)
    attrVec.push_back(AttributeWithIndex::get(0, RAttrs));

  AI = CS.arg_begin();
  for (unsigned i = 0; i != NumCommonArgs; ++i, ++AI) {
    const Type *ParamTy = FT->getParamType(i);
    if ((*AI)->getType() == ParamTy) {
      Args.push_back(*AI);
    } else {
      Instruction::CastOps opcode = CastInst::getCastOpcode(*AI,
          false, ParamTy, false);
      CastInst *NewCast = CastInst::Create(opcode, *AI, ParamTy, "tmp");
      Args.push_back(InsertNewInstBefore(NewCast, *Caller));
    }

    // Add any parameter attributes.
    if (Attributes PAttrs = CallerPAL.getParamAttributes(i + 1))
      attrVec.push_back(AttributeWithIndex::get(i + 1, PAttrs));
  }

  // If the function takes more arguments than the call was taking, add them
  // now...
  for (unsigned i = NumCommonArgs; i != FT->getNumParams(); ++i)
    Args.push_back(Constant::getNullValue(FT->getParamType(i)));

  // If we are removing arguments to the function, emit an obnoxious warning...
  if (FT->getNumParams() < NumActualArgs) {
    if (!FT->isVarArg()) {
      cerr << "WARNING: While resolving call to function '"
           << Callee->getName() << "' arguments were dropped!\n";
    } else {
      // Add all of the arguments in their promoted form to the arg list...
      for (unsigned i = FT->getNumParams(); i != NumActualArgs; ++i, ++AI) {
        const Type *PTy = getPromotedType((*AI)->getType());
        if (PTy != (*AI)->getType()) {
          // Must promote to pass through va_arg area!
          Instruction::CastOps opcode = CastInst::getCastOpcode(*AI, false, 
                                                                PTy, false);
          Instruction *Cast = CastInst::Create(opcode, *AI, PTy, "tmp");
          InsertNewInstBefore(Cast, *Caller);
          Args.push_back(Cast);
        } else {
          Args.push_back(*AI);
        }

        // Add any parameter attributes.
        if (Attributes PAttrs = CallerPAL.getParamAttributes(i + 1))
          attrVec.push_back(AttributeWithIndex::get(i + 1, PAttrs));
      }
    }
  }

  if (Attributes FnAttrs =  CallerPAL.getFnAttributes())
    attrVec.push_back(AttributeWithIndex::get(~0, FnAttrs));

  if (NewRetTy == Type::VoidTy)
    Caller->setName("");   // Void type should not have a name.

  const AttrListPtr &NewCallerPAL = AttrListPtr::get(attrVec.begin(),attrVec.end());

  Instruction *NC;
  if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
    NC = InvokeInst::Create(Callee, II->getNormalDest(), II->getUnwindDest(),
                            Args.begin(), Args.end(),
                            Caller->getName(), Caller);
    cast<InvokeInst>(NC)->setCallingConv(II->getCallingConv());
    cast<InvokeInst>(NC)->setAttributes(NewCallerPAL);
  } else {
    NC = CallInst::Create(Callee, Args.begin(), Args.end(),
                          Caller->getName(), Caller);
    CallInst *CI = cast<CallInst>(Caller);
    if (CI->isTailCall())
      cast<CallInst>(NC)->setTailCall();
    cast<CallInst>(NC)->setCallingConv(CI->getCallingConv());
    cast<CallInst>(NC)->setAttributes(NewCallerPAL);
  }

  // Insert a cast of the return type as necessary.
  Value *NV = NC;
  if (OldRetTy != NV->getType() && !Caller->use_empty()) {
    if (NV->getType() != Type::VoidTy) {
      Instruction::CastOps opcode = CastInst::getCastOpcode(NC, false, 
                                                            OldRetTy, false);
      NV = NC = CastInst::Create(opcode, NC, OldRetTy, "tmp");

      // If this is an invoke instruction, we should insert it after the first
      // non-phi, instruction in the normal successor block.
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        BasicBlock::iterator I = II->getNormalDest()->getFirstNonPHI();
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
  Caller->eraseFromParent();
  RemoveFromWorkList(Caller);
  return true;
}

// transformCallThroughTrampoline - Turn a call to a function created by the
// init_trampoline intrinsic into a direct call to the underlying function.
//
Instruction *InstCombiner::transformCallThroughTrampoline(CallSite CS) {
  Value *Callee = CS.getCalledValue();
  const PointerType *PTy = cast<PointerType>(Callee->getType());
  const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  const AttrListPtr &Attrs = CS.getAttributes();

  // If the call already has the 'nest' attribute somewhere then give up -
  // otherwise 'nest' would occur twice after splicing in the chain.
  if (Attrs.hasAttrSomewhere(Attribute::Nest))
    return 0;

  IntrinsicInst *Tramp =
    cast<IntrinsicInst>(cast<BitCastInst>(Callee)->getOperand(0));

  Function *NestF = cast<Function>(Tramp->getOperand(2)->stripPointerCasts());
  const PointerType *NestFPTy = cast<PointerType>(NestF->getType());
  const FunctionType *NestFTy = cast<FunctionType>(NestFPTy->getElementType());

  const AttrListPtr &NestAttrs = NestF->getAttributes();
  if (!NestAttrs.isEmpty()) {
    unsigned NestIdx = 1;
    const Type *NestTy = 0;
    Attributes NestAttr = Attribute::None;

    // Look for a parameter marked with the 'nest' attribute.
    for (FunctionType::param_iterator I = NestFTy->param_begin(),
         E = NestFTy->param_end(); I != E; ++NestIdx, ++I)
      if (NestAttrs.paramHasAttr(NestIdx, Attribute::Nest)) {
        // Record the parameter type and any other attributes.
        NestTy = *I;
        NestAttr = NestAttrs.getParamAttributes(NestIdx);
        break;
      }

    if (NestTy) {
      Instruction *Caller = CS.getInstruction();
      std::vector<Value*> NewArgs;
      NewArgs.reserve(unsigned(CS.arg_end()-CS.arg_begin())+1);

      SmallVector<AttributeWithIndex, 8> NewAttrs;
      NewAttrs.reserve(Attrs.getNumSlots() + 1);

      // Insert the nest argument into the call argument list, which may
      // mean appending it.  Likewise for attributes.

      // Add any result attributes.
      if (Attributes Attr = Attrs.getRetAttributes())
        NewAttrs.push_back(AttributeWithIndex::get(0, Attr));

      {
        unsigned Idx = 1;
        CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
        do {
          if (Idx == NestIdx) {
            // Add the chain argument and attributes.
            Value *NestVal = Tramp->getOperand(3);
            if (NestVal->getType() != NestTy)
              NestVal = new BitCastInst(NestVal, NestTy, "nest", Caller);
            NewArgs.push_back(NestVal);
            NewAttrs.push_back(AttributeWithIndex::get(NestIdx, NestAttr));
          }

          if (I == E)
            break;

          // Add the original argument and attributes.
          NewArgs.push_back(*I);
          if (Attributes Attr = Attrs.getParamAttributes(Idx))
            NewAttrs.push_back
              (AttributeWithIndex::get(Idx + (Idx >= NestIdx), Attr));

          ++Idx, ++I;
        } while (1);
      }

      // Add any function attributes.
      if (Attributes Attr = Attrs.getFnAttributes())
        NewAttrs.push_back(AttributeWithIndex::get(~0, Attr));

      // The trampoline may have been bitcast to a bogus type (FTy).
      // Handle this by synthesizing a new function type, equal to FTy
      // with the chain parameter inserted.

      std::vector<const Type*> NewTypes;
      NewTypes.reserve(FTy->getNumParams()+1);

      // Insert the chain's type into the list of parameter types, which may
      // mean appending it.
      {
        unsigned Idx = 1;
        FunctionType::param_iterator I = FTy->param_begin(),
          E = FTy->param_end();

        do {
          if (Idx == NestIdx)
            // Add the chain's type.
            NewTypes.push_back(NestTy);

          if (I == E)
            break;

          // Add the original type.
          NewTypes.push_back(*I);

          ++Idx, ++I;
        } while (1);
      }

      // Replace the trampoline call with a direct call.  Let the generic
      // code sort out any function type mismatches.
      FunctionType *NewFTy =
        FunctionType::get(FTy->getReturnType(), NewTypes, FTy->isVarArg());
      Constant *NewCallee = NestF->getType() == PointerType::getUnqual(NewFTy) ?
        NestF : ConstantExpr::getBitCast(NestF, PointerType::getUnqual(NewFTy));
      const AttrListPtr &NewPAL = AttrListPtr::get(NewAttrs.begin(),NewAttrs.end());

      Instruction *NewCaller;
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        NewCaller = InvokeInst::Create(NewCallee,
                                       II->getNormalDest(), II->getUnwindDest(),
                                       NewArgs.begin(), NewArgs.end(),
                                       Caller->getName(), Caller);
        cast<InvokeInst>(NewCaller)->setCallingConv(II->getCallingConv());
        cast<InvokeInst>(NewCaller)->setAttributes(NewPAL);
      } else {
        NewCaller = CallInst::Create(NewCallee, NewArgs.begin(), NewArgs.end(),
                                     Caller->getName(), Caller);
        if (cast<CallInst>(Caller)->isTailCall())
          cast<CallInst>(NewCaller)->setTailCall();
        cast<CallInst>(NewCaller)->
          setCallingConv(cast<CallInst>(Caller)->getCallingConv());
        cast<CallInst>(NewCaller)->setAttributes(NewPAL);
      }
      if (Caller->getType() != Type::VoidTy && !Caller->use_empty())
        Caller->replaceAllUsesWith(NewCaller);
      Caller->eraseFromParent();
      RemoveFromWorkList(Caller);
      return 0;
    }
  }

  // Replace the trampoline call with a direct call.  Since there is no 'nest'
  // parameter, there is no need to adjust the argument list.  Let the generic
  // code sort out any function type mismatches.
  Constant *NewCallee =
    NestF->getType() == PTy ? NestF : ConstantExpr::getBitCast(NestF, PTy);
  CS.setCalledFunction(NewCallee);
  return CS.getInstruction();
}

/// FoldPHIArgBinOpIntoPHI - If we have something like phi [add (a,b), add(c,d)]
/// and if a/b/c/d and the add's all have a single use, turn this into two phi's
/// and a single binop.
Instruction *InstCombiner::FoldPHIArgBinOpIntoPHI(PHINode &PN) {
  Instruction *FirstInst = cast<Instruction>(PN.getIncomingValue(0));
  assert(isa<BinaryOperator>(FirstInst) || isa<CmpInst>(FirstInst));
  unsigned Opc = FirstInst->getOpcode();
  Value *LHSVal = FirstInst->getOperand(0);
  Value *RHSVal = FirstInst->getOperand(1);
    
  const Type *LHSType = LHSVal->getType();
  const Type *RHSType = RHSVal->getType();
  
  // Scan to see if all operands are the same opcode, all have one use, and all
  // kill their operands (i.e. the operands have one use).
  for (unsigned i = 1; i != PN.getNumIncomingValues(); ++i) {
    Instruction *I = dyn_cast<Instruction>(PN.getIncomingValue(i));
    if (!I || I->getOpcode() != Opc || !I->hasOneUse() ||
        // Verify type of the LHS matches so we don't fold cmp's of different
        // types or GEP's with different index types.
        I->getOperand(0)->getType() != LHSType ||
        I->getOperand(1)->getType() != RHSType)
      return 0;

    // If they are CmpInst instructions, check their predicates
    if (Opc == Instruction::ICmp || Opc == Instruction::FCmp)
      if (cast<CmpInst>(I)->getPredicate() !=
          cast<CmpInst>(FirstInst)->getPredicate())
        return 0;
    
    // Keep track of which operand needs a phi node.
    if (I->getOperand(0) != LHSVal) LHSVal = 0;
    if (I->getOperand(1) != RHSVal) RHSVal = 0;
  }
  
  // Otherwise, this is safe to transform!
  
  Value *InLHS = FirstInst->getOperand(0);
  Value *InRHS = FirstInst->getOperand(1);
  PHINode *NewLHS = 0, *NewRHS = 0;
  if (LHSVal == 0) {
    NewLHS = PHINode::Create(LHSType,
                             FirstInst->getOperand(0)->getName() + ".pn");
    NewLHS->reserveOperandSpace(PN.getNumOperands()/2);
    NewLHS->addIncoming(InLHS, PN.getIncomingBlock(0));
    InsertNewInstBefore(NewLHS, PN);
    LHSVal = NewLHS;
  }
  
  if (RHSVal == 0) {
    NewRHS = PHINode::Create(RHSType,
                             FirstInst->getOperand(1)->getName() + ".pn");
    NewRHS->reserveOperandSpace(PN.getNumOperands()/2);
    NewRHS->addIncoming(InRHS, PN.getIncomingBlock(0));
    InsertNewInstBefore(NewRHS, PN);
    RHSVal = NewRHS;
  }
  
  // Add all operands to the new PHIs.
  if (NewLHS || NewRHS) {
    for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
      Instruction *InInst = cast<Instruction>(PN.getIncomingValue(i));
      if (NewLHS) {
        Value *NewInLHS = InInst->getOperand(0);
        NewLHS->addIncoming(NewInLHS, PN.getIncomingBlock(i));
      }
      if (NewRHS) {
        Value *NewInRHS = InInst->getOperand(1);
        NewRHS->addIncoming(NewInRHS, PN.getIncomingBlock(i));
      }
    }
  }
    
  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(FirstInst))
    return BinaryOperator::Create(BinOp->getOpcode(), LHSVal, RHSVal);
  CmpInst *CIOp = cast<CmpInst>(FirstInst);
  return CmpInst::Create(CIOp->getOpcode(), CIOp->getPredicate(), LHSVal,
                         RHSVal);
}

Instruction *InstCombiner::FoldPHIArgGEPIntoPHI(PHINode &PN) {
  GetElementPtrInst *FirstInst =cast<GetElementPtrInst>(PN.getIncomingValue(0));
  
  SmallVector<Value*, 16> FixedOperands(FirstInst->op_begin(), 
                                        FirstInst->op_end());
  
  // Scan to see if all operands are the same opcode, all have one use, and all
  // kill their operands (i.e. the operands have one use).
  for (unsigned i = 1; i != PN.getNumIncomingValues(); ++i) {
    GetElementPtrInst *GEP= dyn_cast<GetElementPtrInst>(PN.getIncomingValue(i));
    if (!GEP || !GEP->hasOneUse() || GEP->getType() != FirstInst->getType() ||
      GEP->getNumOperands() != FirstInst->getNumOperands())
      return 0;

    // Compare the operand lists.
    for (unsigned op = 0, e = FirstInst->getNumOperands(); op != e; ++op) {
      if (FirstInst->getOperand(op) == GEP->getOperand(op))
        continue;
      
      // Don't merge two GEPs when two operands differ (introducing phi nodes)
      // if one of the PHIs has a constant for the index.  The index may be
      // substantially cheaper to compute for the constants, so making it a
      // variable index could pessimize the path.  This also handles the case
      // for struct indices, which must always be constant.
      if (isa<ConstantInt>(FirstInst->getOperand(op)) ||
          isa<ConstantInt>(GEP->getOperand(op)))
        return 0;
      
      if (FirstInst->getOperand(op)->getType() !=GEP->getOperand(op)->getType())
        return 0;
      FixedOperands[op] = 0;  // Needs a PHI.
    }
  }
  
  // Otherwise, this is safe to transform.  Insert PHI nodes for each operand
  // that is variable.
  SmallVector<PHINode*, 16> OperandPhis(FixedOperands.size());
  
  bool HasAnyPHIs = false;
  for (unsigned i = 0, e = FixedOperands.size(); i != e; ++i) {
    if (FixedOperands[i]) continue;  // operand doesn't need a phi.
    Value *FirstOp = FirstInst->getOperand(i);
    PHINode *NewPN = PHINode::Create(FirstOp->getType(),
                                     FirstOp->getName()+".pn");
    InsertNewInstBefore(NewPN, PN);
    
    NewPN->reserveOperandSpace(e);
    NewPN->addIncoming(FirstOp, PN.getIncomingBlock(0));
    OperandPhis[i] = NewPN;
    FixedOperands[i] = NewPN;
    HasAnyPHIs = true;
  }

  
  // Add all operands to the new PHIs.
  if (HasAnyPHIs) {
    for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
      GetElementPtrInst *InGEP =cast<GetElementPtrInst>(PN.getIncomingValue(i));
      BasicBlock *InBB = PN.getIncomingBlock(i);
      
      for (unsigned op = 0, e = OperandPhis.size(); op != e; ++op)
        if (PHINode *OpPhi = OperandPhis[op])
          OpPhi->addIncoming(InGEP->getOperand(op), InBB);
    }
  }
  
  Value *Base = FixedOperands[0];
  return GetElementPtrInst::Create(Base, FixedOperands.begin()+1,
                                   FixedOperands.end());
}


/// isSafeToSinkLoad - Return true if we know that it is safe sink the load out
/// of the block that defines it.  This means that it must be obvious the value
/// of the load is not changed from the point of the load to the end of the
/// block it is in.
///
/// Finally, it is safe, but not profitable, to sink a load targetting a
/// non-address-taken alloca.  Doing so will cause us to not promote the alloca
/// to a register.
static bool isSafeToSinkLoad(LoadInst *L) {
  BasicBlock::iterator BBI = L, E = L->getParent()->end();
  
  for (++BBI; BBI != E; ++BBI)
    if (BBI->mayWriteToMemory())
      return false;
  
  // Check for non-address taken alloca.  If not address-taken already, it isn't
  // profitable to do this xform.
  if (AllocaInst *AI = dyn_cast<AllocaInst>(L->getOperand(0))) {
    bool isAddressTaken = false;
    for (Value::use_iterator UI = AI->use_begin(), E = AI->use_end();
         UI != E; ++UI) {
      if (isa<LoadInst>(UI)) continue;
      if (StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
        // If storing TO the alloca, then the address isn't taken.
        if (SI->getOperand(1) == AI) continue;
      }
      isAddressTaken = true;
      break;
    }
    
    if (!isAddressTaken)
      return false;
  }
  
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
  } else if (isa<BinaryOperator>(FirstInst) || isa<CmpInst>(FirstInst)) {
    // Can fold binop, compare or shift here if the RHS is a constant, 
    // otherwise call FoldPHIArgBinOpIntoPHI.
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
    
    // If the PHI is of volatile loads and the load block has multiple
    // successors, sinking it would remove a load of the volatile value from
    // the path through the other successor.
    if (isVolatile &&
        LI->getParent()->getTerminator()->getNumSuccessors() != 1)
      return 0;
    
  } else if (isa<GetElementPtrInst>(FirstInst)) {
    return FoldPHIArgGEPIntoPHI(PN);
  } else {
    return 0;  // Cannot fold this operation.
  }

  // Check to see if all arguments are the same operation.
  for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
    if (!isa<Instruction>(PN.getIncomingValue(i))) return 0;
    Instruction *I = cast<Instruction>(PN.getIncomingValue(i));
    if (!I->hasOneUse() || !I->isSameOperationAs(FirstInst))
      return 0;
    if (CastSrcTy) {
      if (I->getOperand(0)->getType() != CastSrcTy)
        return 0;  // Cast operation must match.
    } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      // We can't sink the load if the loaded value could be modified between 
      // the load and the PHI.
      if (LI->isVolatile() != isVolatile ||
          LI->getParent() != PN.getIncomingBlock(i) ||
          !isSafeToSinkLoad(LI))
        return 0;
      
      // If the PHI is of volatile loads and the load block has multiple
      // successors, sinking it would remove a load of the volatile value from
      // the path through the other successor.
      if (isVolatile &&
          LI->getParent()->getTerminator()->getNumSuccessors() != 1)
        return 0;

      
    } else if (I->getOperand(1) != ConstantOp) {
      return 0;
    }
  }

  // Okay, they are all the same operation.  Create a new PHI node of the
  // correct type, and PHI together all of the LHS's of the instructions.
  PHINode *NewPN = PHINode::Create(FirstInst->getOperand(0)->getType(),
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
  if (CastInst* FirstCI = dyn_cast<CastInst>(FirstInst))
    return CastInst::Create(FirstCI->getOpcode(), PhiVal, PN.getType());
  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(FirstInst))
    return BinaryOperator::Create(BinOp->getOpcode(), PhiVal, ConstantOp);
  if (CmpInst *CIOp = dyn_cast<CmpInst>(FirstInst))
    return CmpInst::Create(CIOp->getOpcode(), CIOp->getPredicate(), 
                           PhiVal, ConstantOp);
  assert(isa<LoadInst>(FirstInst) && "Unknown operation");
  
  // If this was a volatile load that we are merging, make sure to loop through
  // and mark all the input loads as non-volatile.  If we don't do this, we will
  // insert a new volatile load and the old ones will not be deletable.
  if (isVolatile)
    for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
      cast<LoadInst>(PN.getIncomingValue(i))->setVolatile(false);
  
  return new LoadInst(PhiVal, "", isVolatile);
}

/// DeadPHICycle - Return true if this PHI node is only used by a PHI node cycle
/// that is dead.
static bool DeadPHICycle(PHINode *PN,
                         SmallPtrSet<PHINode*, 16> &PotentiallyDeadPHIs) {
  if (PN->use_empty()) return true;
  if (!PN->hasOneUse()) return false;

  // Remember this node, and if we find the cycle, return.
  if (!PotentiallyDeadPHIs.insert(PN))
    return true;
  
  // Don't scan crazily complex things.
  if (PotentiallyDeadPHIs.size() == 16)
    return false;

  if (PHINode *PU = dyn_cast<PHINode>(PN->use_back()))
    return DeadPHICycle(PU, PotentiallyDeadPHIs);

  return false;
}

/// PHIsEqualValue - Return true if this phi node is always equal to
/// NonPhiInVal.  This happens with mutually cyclic phi nodes like:
///   z = some value; x = phi (y, z); y = phi (x, z)
static bool PHIsEqualValue(PHINode *PN, Value *NonPhiInVal, 
                           SmallPtrSet<PHINode*, 16> &ValueEqualPHIs) {
  // See if we already saw this PHI node.
  if (!ValueEqualPHIs.insert(PN))
    return true;
  
  // Don't scan crazily complex things.
  if (ValueEqualPHIs.size() == 16)
    return false;
 
  // Scan the operands to see if they are either phi nodes or are equal to
  // the value.
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    Value *Op = PN->getIncomingValue(i);
    if (PHINode *OpPN = dyn_cast<PHINode>(Op)) {
      if (!PHIsEqualValue(OpPN, NonPhiInVal, ValueEqualPHIs))
        return false;
    } else if (Op != NonPhiInVal)
      return false;
  }
  
  return true;
}


// PHINode simplification
//
Instruction *InstCombiner::visitPHINode(PHINode &PN) {
  // If LCSSA is around, don't mess with Phi nodes
  if (MustPreserveLCSSA) return 0;
  
  if (Value *V = PN.hasConstantValue())
    return ReplaceInstUsesWith(PN, V);

  // If all PHI operands are the same operation, pull them through the PHI,
  // reducing code size.
  if (isa<Instruction>(PN.getIncomingValue(0)) &&
      isa<Instruction>(PN.getIncomingValue(1)) &&
      cast<Instruction>(PN.getIncomingValue(0))->getOpcode() ==
      cast<Instruction>(PN.getIncomingValue(1))->getOpcode() &&
      // FIXME: The hasOneUse check will fail for PHIs that use the value more
      // than themselves more than once.
      PN.getIncomingValue(0)->hasOneUse())
    if (Instruction *Result = FoldPHIArgOpIntoPHI(PN))
      return Result;

  // If this is a trivial cycle in the PHI node graph, remove it.  Basically, if
  // this PHI only has a single use (a PHI), and if that PHI only has one use (a
  // PHI)... break the cycle.
  if (PN.hasOneUse()) {
    Instruction *PHIUser = cast<Instruction>(PN.use_back());
    if (PHINode *PU = dyn_cast<PHINode>(PHIUser)) {
      SmallPtrSet<PHINode*, 16> PotentiallyDeadPHIs;
      PotentiallyDeadPHIs.insert(&PN);
      if (DeadPHICycle(PU, PotentiallyDeadPHIs))
        return ReplaceInstUsesWith(PN, UndefValue::get(PN.getType()));
    }
   
    // If this phi has a single use, and if that use just computes a value for
    // the next iteration of a loop, delete the phi.  This occurs with unused
    // induction variables, e.g. "for (int j = 0; ; ++j);".  Detecting this
    // common case here is good because the only other things that catch this
    // are induction variable analysis (sometimes) and ADCE, which is only run
    // late.
    if (PHIUser->hasOneUse() &&
        (isa<BinaryOperator>(PHIUser) || isa<GetElementPtrInst>(PHIUser)) &&
        PHIUser->use_back() == &PN) {
      return ReplaceInstUsesWith(PN, UndefValue::get(PN.getType()));
    }
  }

  // We sometimes end up with phi cycles that non-obviously end up being the
  // same value, for example:
  //   z = some value; x = phi (y, z); y = phi (x, z)
  // where the phi nodes don't necessarily need to be in the same block.  Do a
  // quick check to see if the PHI node only contains a single non-phi value, if
  // so, scan to see if the phi cycle is actually equal to that value.
  {
    unsigned InValNo = 0, NumOperandVals = PN.getNumIncomingValues();
    // Scan for the first non-phi operand.
    while (InValNo != NumOperandVals && 
           isa<PHINode>(PN.getIncomingValue(InValNo)))
      ++InValNo;

    if (InValNo != NumOperandVals) {
      Value *NonPhiInVal = PN.getOperand(InValNo);
      
      // Scan the rest of the operands to see if there are any conflicts, if so
      // there is no need to recursively scan other phis.
      for (++InValNo; InValNo != NumOperandVals; ++InValNo) {
        Value *OpVal = PN.getIncomingValue(InValNo);
        if (OpVal != NonPhiInVal && !isa<PHINode>(OpVal))
          break;
      }
      
      // If we scanned over all operands, then we have one unique value plus
      // phi values.  Scan PHI nodes to see if they all merge in each other or
      // the value.
      if (InValNo == NumOperandVals) {
        SmallPtrSet<PHINode*, 16> ValueEqualPHIs;
        if (PHIsEqualValue(&PN, NonPhiInVal, ValueEqualPHIs))
          return ReplaceInstUsesWith(PN, NonPhiInVal);
      }
    }
  }
  return 0;
}

static Value *InsertCastToIntPtrTy(Value *V, const Type *DTy,
                                   Instruction *InsertPoint,
                                   InstCombiner *IC) {
  unsigned PtrSize = DTy->getPrimitiveSizeInBits();
  unsigned VTySize = V->getType()->getPrimitiveSizeInBits();
  // We must cast correctly to the pointer type. Ensure that we
  // sign extend the integer value if it is smaller as this is
  // used for address computation.
  Instruction::CastOps opcode = 
     (VTySize < PtrSize ? Instruction::SExt :
      (VTySize == PtrSize ? Instruction::BitCast : Instruction::Trunc));
  return IC->InsertCastBefore(opcode, V, DTy, *InsertPoint);
}


Instruction *InstCombiner::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  Value *PtrOp = GEP.getOperand(0);
  // Is it 'getelementptr %P, i32 0'  or 'getelementptr %P'
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
  for (User::op_iterator i = GEP.op_begin() + 1, e = GEP.op_end();
       i != e; ++i, ++GTI) {
    if (isa<SequentialType>(*GTI)) {
      if (CastInst *CI = dyn_cast<CastInst>(*i)) {
        if (CI->getOpcode() == Instruction::ZExt ||
            CI->getOpcode() == Instruction::SExt) {
          const Type *SrcTy = CI->getOperand(0)->getType();
          // We can eliminate a cast from i32 to i64 iff the target 
          // is a 32-bit pointer target.
          if (SrcTy->getPrimitiveSizeInBits() >= TD->getPointerSizeInBits()) {
            MadeChange = true;
            *i = CI->getOperand(0);
          }
        }
      }
      // If we are using a wider index than needed for this platform, shrink it
      // to what we need.  If narrower, sign-extend it to what we need.
      // If the incoming value needs a cast instruction,
      // insert it.  This explicit cast can make subsequent optimizations more
      // obvious.
      Value *Op = *i;
      if (TD->getTypeSizeInBits(Op->getType()) > TD->getPointerSizeInBits()) {
        if (Constant *C = dyn_cast<Constant>(Op)) {
          *i = ConstantExpr::getTrunc(C, TD->getIntPtrType());
          MadeChange = true;
        } else {
          Op = InsertCastBefore(Instruction::Trunc, Op, TD->getIntPtrType(),
                                GEP);
          *i = Op;
          MadeChange = true;
        }
      } else if (TD->getTypeSizeInBits(Op->getType()) < TD->getPointerSizeInBits()) {
        if (Constant *C = dyn_cast<Constant>(Op)) {
          *i = ConstantExpr::getSExt(C, TD->getIntPtrType());
          MadeChange = true;
        } else {
          Op = InsertCastBefore(Instruction::SExt, Op, TD->getIntPtrType(),
                                GEP);
          *i = Op;
          MadeChange = true;
        }
      }
    }
  }
  if (MadeChange) return &GEP;

  // Combine Indices - If the source pointer to this getelementptr instruction
  // is a getelementptr instruction, combine the indices of the two
  // getelementptr instructions into a single instruction.
  //
  SmallVector<Value*, 8> SrcGEPOperands;
  if (User *Src = dyn_castGetElementPtr(PtrOp))
    SrcGEPOperands.append(Src->op_begin(), Src->op_end());

  if (!SrcGEPOperands.empty()) {
    // Note that if our source is a gep chain itself that we wait for that
    // chain to be resolved before we perform this transformation.  This
    // avoids us creating a TON of code in some cases.
    //
    if (isa<GetElementPtrInst>(SrcGEPOperands[0]) &&
        cast<Instruction>(SrcGEPOperands[0])->getNumOperands() == 2)
      return 0;   // Wait until our source is folded to completion.

    SmallVector<Value*, 8> Indices;

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
            SO1 = ConstantExpr::getIntegerCast(SO1C, GO1->getType(), true);
          } else if (Constant *GO1C = dyn_cast<Constant>(GO1)) {
            GO1 = ConstantExpr::getIntegerCast(GO1C, SO1->getType(), true);
          } else {
            unsigned PS = TD->getPointerSizeInBits();
            if (TD->getTypeSizeInBits(SO1->getType()) == PS) {
              // Convert GO1 to SO1's type.
              GO1 = InsertCastToIntPtrTy(GO1, SO1->getType(), &GEP, this);

            } else if (TD->getTypeSizeInBits(GO1->getType()) == PS) {
              // Convert SO1 to GO1's type.
              SO1 = InsertCastToIntPtrTy(SO1, GO1->getType(), &GEP, this);
            } else {
              const Type *PT = TD->getIntPtrType();
              SO1 = InsertCastToIntPtrTy(SO1, PT, &GEP, this);
              GO1 = InsertCastToIntPtrTy(GO1, PT, &GEP, this);
            }
          }
        }
        if (isa<Constant>(SO1) && isa<Constant>(GO1))
          Sum = ConstantExpr::getAdd(cast<Constant>(SO1), cast<Constant>(GO1));
        else {
          Sum = BinaryOperator::CreateAdd(SO1, GO1, PtrOp->getName()+".sum");
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
      return GetElementPtrInst::Create(SrcGEPOperands[0], Indices.begin(),
                                       Indices.end(), GEP.getName());

  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(PtrOp)) {
    // GEP of global variable.  If all of the indices for this GEP are
    // constants, we can promote this to a constexpr instead of an instruction.

    // Scan for nonconstants...
    SmallVector<Constant*, 8> Indices;
    User::op_iterator I = GEP.idx_begin(), E = GEP.idx_end();
    for (; I != E && isa<Constant>(*I); ++I)
      Indices.push_back(cast<Constant>(*I));

    if (I == E) {  // If they are all constants...
      Constant *CE = ConstantExpr::getGetElementPtr(GV,
                                                    &Indices[0],Indices.size());

      // Replace all uses of the GEP with the new constexpr...
      return ReplaceInstUsesWith(GEP, CE);
    }
  } else if (Value *X = getBitCastOperand(PtrOp)) {  // Is the operand a cast?
    if (!isa<PointerType>(X->getType())) {
      // Not interesting.  Source pointer must be a cast from pointer.
    } else if (HasZeroPointerIndex) {
      // transform: GEP (bitcast [10 x i8]* X to [0 x i8]*), i32 0, ...
      // into     : GEP [10 x i8]* X, i32 0, ...
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
      // %t = getelementptr i32* bitcast ([2 x i32]* %str to i32*), i32 %V
      // into:  %t1 = getelementptr [2 x i32]* %str, i32 0, i32 %V; bitcast
      const Type *SrcElTy = cast<PointerType>(X->getType())->getElementType();
      const Type *ResElTy=cast<PointerType>(PtrOp->getType())->getElementType();
      if (isa<ArrayType>(SrcElTy) &&
          TD->getTypePaddedSize(cast<ArrayType>(SrcElTy)->getElementType()) ==
          TD->getTypePaddedSize(ResElTy)) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(Type::Int32Ty);
        Idx[1] = GEP.getOperand(1);
        Value *V = InsertNewInstBefore(
               GetElementPtrInst::Create(X, Idx, Idx + 2, GEP.getName()), GEP);
        // V and GEP are both pointer types --> BitCast
        return new BitCastInst(V, GEP.getType());
      }
      
      // Transform things like:
      // getelementptr i8* bitcast ([100 x double]* X to i8*), i32 %tmp
      //   (where tmp = 8*tmp2) into:
      // getelementptr [100 x double]* %arr, i32 0, i32 %tmp2; bitcast
      
      if (isa<ArrayType>(SrcElTy) && ResElTy == Type::Int8Ty) {
        uint64_t ArrayEltSize =
            TD->getTypePaddedSize(cast<ArrayType>(SrcElTy)->getElementType());
        
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
            ConstantInt *ShAmt = cast<ConstantInt>(Inst->getOperand(1));
            uint32_t ShAmtVal = ShAmt->getLimitedValue(64);
            Scale = ConstantInt::get(Inst->getType(), 1ULL << ShAmtVal);
            NewIdx = Inst->getOperand(0);
          } else if (Inst->getOpcode() == Instruction::Mul &&
                     isa<ConstantInt>(Inst->getOperand(1))) {
            Scale = cast<ConstantInt>(Inst->getOperand(1));
            NewIdx = Inst->getOperand(0);
          }
        }
        
        // If the index will be to exactly the right offset with the scale taken
        // out, perform the transformation. Note, we don't know whether Scale is
        // signed or not. We'll use unsigned version of division/modulo
        // operation after making sure Scale doesn't have the sign bit set.
        if (Scale && Scale->getSExtValue() >= 0LL &&
            Scale->getZExtValue() % ArrayEltSize == 0) {
          Scale = ConstantInt::get(Scale->getType(),
                                   Scale->getZExtValue() / ArrayEltSize);
          if (Scale->getZExtValue() != 1) {
            Constant *C = ConstantExpr::getIntegerCast(Scale, NewIdx->getType(),
                                                       false /*ZExt*/);
            Instruction *Sc = BinaryOperator::CreateMul(NewIdx, C, "idxscale");
            NewIdx = InsertNewInstBefore(Sc, GEP);
          }

          // Insert the new GEP instruction.
          Value *Idx[2];
          Idx[0] = Constant::getNullValue(Type::Int32Ty);
          Idx[1] = NewIdx;
          Instruction *NewGEP =
            GetElementPtrInst::Create(X, Idx, Idx + 2, GEP.getName());
          NewGEP = InsertNewInstBefore(NewGEP, GEP);
          // The NewGEP must be pointer typed, so must the old one -> BitCast
          return new BitCastInst(NewGEP, GEP.getType());
        }
      }
    }
  }
  
  /// See if we can simplify:
  ///   X = bitcast A to B*
  ///   Y = gep X, <...constant indices...>
  /// into a gep of the original struct.  This is important for SROA and alias
  /// analysis of unions.  If "A" is also a bitcast, wait for A/X to be merged.
  if (BitCastInst *BCI = dyn_cast<BitCastInst>(PtrOp)) {
    if (!isa<BitCastInst>(BCI->getOperand(0)) && GEP.hasAllConstantIndices()) {
      // Determine how much the GEP moves the pointer.  We are guaranteed to get
      // a constant back from EmitGEPOffset.
      ConstantInt *OffsetV = cast<ConstantInt>(EmitGEPOffset(&GEP, GEP, *this));
      int64_t Offset = OffsetV->getSExtValue();
      
      // If this GEP instruction doesn't move the pointer, just replace the GEP
      // with a bitcast of the real input to the dest type.
      if (Offset == 0) {
        // If the bitcast is of an allocation, and the allocation will be
        // converted to match the type of the cast, don't touch this.
        if (isa<AllocationInst>(BCI->getOperand(0))) {
          // See if the bitcast simplifies, if so, don't nuke this GEP yet.
          if (Instruction *I = visitBitCast(*BCI)) {
            if (I != BCI) {
              I->takeName(BCI);
              BCI->getParent()->getInstList().insert(BCI, I);
              ReplaceInstUsesWith(*BCI, I);
            }
            return &GEP;
          }
        }
        return new BitCastInst(BCI->getOperand(0), GEP.getType());
      }
      
      // Otherwise, if the offset is non-zero, we need to find out if there is a
      // field at Offset in 'A's type.  If so, we can pull the cast through the
      // GEP.
      SmallVector<Value*, 8> NewIndices;
      const Type *InTy =
        cast<PointerType>(BCI->getOperand(0)->getType())->getElementType();
      if (FindElementAtOffset(InTy, Offset, NewIndices, TD)) {
        Instruction *NGEP =
           GetElementPtrInst::Create(BCI->getOperand(0), NewIndices.begin(),
                                     NewIndices.end());
        if (NGEP->getType() == GEP.getType()) return NGEP;
        InsertNewInstBefore(NGEP, GEP);
        NGEP->takeName(&GEP);
        return new BitCastInst(NGEP, GEP.getType());
      }
    }
  }    
    
  return 0;
}

Instruction *InstCombiner::visitAllocationInst(AllocationInst &AI) {
  // Convert: malloc Ty, C - where C is a constant != 1 into: malloc [C x Ty], 1
  if (AI.isArrayAllocation()) {  // Check C != 1
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
      Value *NullIdx = Constant::getNullValue(Type::Int32Ty);
      Value *Idx[2];
      Idx[0] = NullIdx;
      Idx[1] = NullIdx;
      Value *V = GetElementPtrInst::Create(New, Idx, Idx + 2,
                                           New->getName()+".sub", It);

      // Now make everything use the getelementptr instead of the original
      // allocation.
      return ReplaceInstUsesWith(AI, V);
    } else if (isa<UndefValue>(AI.getArraySize())) {
      return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));
    }
  }

  if (isa<AllocaInst>(AI) && AI.getAllocatedType()->isSized()) {
    // If alloca'ing a zero byte object, replace the alloca with a null pointer.
    // Note that we only do this for alloca's, because malloc should allocate and
    // return a unique pointer, even for a zero byte allocation.
    if (TD->getTypePaddedSize(AI.getAllocatedType()) == 0)
      return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));

    // If the alignment is 0 (unspecified), assign it the preferred alignment.
    if (AI.getAlignment() == 0)
      AI.setAlignment(TD->getPrefTypeAlignment(AI.getAllocatedType()));
  }

  return 0;
}

Instruction *InstCombiner::visitFreeInst(FreeInst &FI) {
  Value *Op = FI.getOperand(0);

  // free undef -> unreachable.
  if (isa<UndefValue>(Op)) {
    // Insert a new store to null because we cannot modify the CFG here.
    new StoreInst(ConstantInt::getTrue(),
                  UndefValue::get(PointerType::getUnqual(Type::Int1Ty)), &FI);
    return EraseInstFromFunction(FI);
  }
  
  // If we have 'free null' delete the instruction.  This can happen in stl code
  // when lots of inlining happens.
  if (isa<ConstantPointerNull>(Op))
    return EraseInstFromFunction(FI);
  
  // Change free <ty>* (cast <ty2>* X to <ty>*) into free <ty2>* X
  if (BitCastInst *CI = dyn_cast<BitCastInst>(Op)) {
    FI.setOperand(0, CI->getOperand(0));
    return &FI;
  }
  
  // Change free (gep X, 0,0,0,0) into free(X)
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(Op)) {
    if (GEPI->hasAllZeroIndices()) {
      AddToWorkList(GEPI);
      FI.setOperand(0, GEPI->getOperand(0));
      return &FI;
    }
  }
  
  // Change free(malloc) into nothing, if the malloc has a single use.
  if (MallocInst *MI = dyn_cast<MallocInst>(Op))
    if (MI->hasOneUse()) {
      EraseInstFromFunction(FI);
      return EraseInstFromFunction(*MI);
    }

  return 0;
}


/// InstCombineLoadCast - Fold 'load (cast P)' -> cast (load P)' when possible.
static Instruction *InstCombineLoadCast(InstCombiner &IC, LoadInst &LI,
                                        const TargetData *TD) {
  User *CI = cast<User>(LI.getOperand(0));
  Value *CastOp = CI->getOperand(0);

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(CI)) {
    // Instead of loading constant c string, use corresponding integer value
    // directly if string length is small enough.
    std::string Str;
    if (GetConstantStringInfo(CE->getOperand(0), Str) && !Str.empty()) {
      unsigned len = Str.length();
      const Type *Ty = cast<PointerType>(CE->getType())->getElementType();
      unsigned numBits = Ty->getPrimitiveSizeInBits();
      // Replace LI with immediate integer store.
      if ((numBits >> 3) == len + 1) {
        APInt StrVal(numBits, 0);
        APInt SingleChar(numBits, 0);
        if (TD->isLittleEndian()) {
          for (signed i = len-1; i >= 0; i--) {
            SingleChar = (uint64_t) Str[i];
            StrVal = (StrVal << 8) | SingleChar;
          }
        } else {
          for (unsigned i = 0; i < len; i++) {
            SingleChar = (uint64_t) Str[i];
            StrVal = (StrVal << 8) | SingleChar;
          }
          // Append NULL at the end.
          SingleChar = 0;
          StrVal = (StrVal << 8) | SingleChar;
        }
        Value *NL = ConstantInt::get(StrVal);
        return IC.ReplaceInstUsesWith(LI, NL);
      }
    }
  }

  const Type *DestPTy = cast<PointerType>(CI->getType())->getElementType();
  if (const PointerType *SrcTy = dyn_cast<PointerType>(CastOp->getType())) {
    const Type *SrcPTy = SrcTy->getElementType();

    if (DestPTy->isInteger() || isa<PointerType>(DestPTy) || 
         isa<VectorType>(DestPTy)) {
      // If the source is an array, the code below will not succeed.  Check to
      // see if a trivial 'gep P, 0, 0' will help matters.  Only do this for
      // constants.
      if (const ArrayType *ASrcTy = dyn_cast<ArrayType>(SrcPTy))
        if (Constant *CSrc = dyn_cast<Constant>(CastOp))
          if (ASrcTy->getNumElements() != 0) {
            Value *Idxs[2];
            Idxs[0] = Idxs[1] = Constant::getNullValue(Type::Int32Ty);
            CastOp = ConstantExpr::getGetElementPtr(CSrc, Idxs, 2);
            SrcTy = cast<PointerType>(CastOp->getType());
            SrcPTy = SrcTy->getElementType();
          }

      if ((SrcPTy->isInteger() || isa<PointerType>(SrcPTy) || 
            isa<VectorType>(SrcPTy)) &&
          // Do not allow turning this into a load of an integer, which is then
          // casted to a pointer, this pessimizes pointer analysis a lot.
          (isa<PointerType>(SrcPTy) == isa<PointerType>(LI.getType())) &&
          IC.getTargetData().getTypeSizeInBits(SrcPTy) ==
               IC.getTargetData().getTypeSizeInBits(DestPTy)) {

        // Okay, we are casting from one integer or pointer type to another of
        // the same size.  Instead of casting the pointer before the load, cast
        // the result of the loaded value.
        Value *NewLoad = IC.InsertNewInstBefore(new LoadInst(CastOp,
                                                             CI->getName(),
                                                         LI.isVolatile()),LI);
        // Now cast the result of the load.
        return new BitCastInst(NewLoad, LI.getType());
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
  // If it is an alloca it is always safe to load from.
  if (isa<AllocaInst>(V)) return true;

  // If it is a global variable it is mostly safe to load from.
  if (const GlobalValue *GV = dyn_cast<GlobalVariable>(V))
    // Don't try to evaluate aliases.  External weak GV can be null.
    return !isa<GlobalAlias>(GV) && !GV->hasExternalWeakLinkage();

  // Otherwise, be a little bit agressive by scanning the local block where we
  // want to check to see if the pointer is already being loaded or stored
  // from/to.  If so, the previous load or store would have already trapped,
  // so there is no harm doing an extra load (also, CSE will later eliminate
  // the load entirely).
  BasicBlock::iterator BBI = ScanFrom, E = ScanFrom->getParent()->begin();

  while (BBI != E) {
    --BBI;

    // If we see a free or a call (which might do a free) the pointer could be
    // marked invalid.
    if (isa<FreeInst>(BBI) || isa<CallInst>(BBI))
      return false;
    
    if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
      if (LI->getOperand(0) == V) return true;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(BBI)) {
      if (SI->getOperand(1) == V) return true;
    }

  }
  return false;
}

Instruction *InstCombiner::visitLoadInst(LoadInst &LI) {
  Value *Op = LI.getOperand(0);

  // Attempt to improve the alignment.
  unsigned KnownAlign = GetOrEnforceKnownAlignment(Op);
  if (KnownAlign >
      (LI.getAlignment() == 0 ? TD->getABITypeAlignment(LI.getType()) :
                                LI.getAlignment()))
    LI.setAlignment(KnownAlign);

  // load (cast X) --> cast (load X) iff safe
  if (isa<CastInst>(Op))
    if (Instruction *Res = InstCombineLoadCast(*this, LI, TD))
      return Res;

  // None of the following transforms are legal for volatile loads.
  if (LI.isVolatile()) return 0;
  
  // Do really simple store-to-load forwarding and load CSE, to catch cases
  // where there are several consequtive memory accesses to the same location,
  // separated by a few arithmetic operations.
  BasicBlock::iterator BBI = &LI;
  if (Value *AvailableVal = FindAvailableLoadedValue(Op, LI.getParent(), BBI,6))
    return ReplaceInstUsesWith(LI, AvailableVal);

  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(Op)) {
    const Value *GEPI0 = GEPI->getOperand(0);
    // TODO: Consider a target hook for valid address spaces for this xform.
    if (isa<ConstantPointerNull>(GEPI0) &&
        cast<PointerType>(GEPI0->getType())->getAddressSpace() == 0) {
      // Insert a new store to null instruction before the load to indicate
      // that this code is not reachable.  We do this instead of inserting
      // an unreachable instruction directly because we cannot modify the
      // CFG.
      new StoreInst(UndefValue::get(LI.getType()),
                    Constant::getNullValue(Op->getType()), &LI);
      return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
    }
  } 

  if (Constant *C = dyn_cast<Constant>(Op)) {
    // load null/undef -> undef
    // TODO: Consider a target hook for valid address spaces for this xform.
    if (isa<UndefValue>(C) || (C->isNullValue() && 
        cast<PointerType>(Op->getType())->getAddressSpace() == 0)) {
      // Insert a new store to null instruction before the load to indicate that
      // this code is not reachable.  We do this instead of inserting an
      // unreachable instruction directly because we cannot modify the CFG.
      new StoreInst(UndefValue::get(LI.getType()),
                    Constant::getNullValue(Op->getType()), &LI);
      return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
    }

    // Instcombine load (constant global) into the value loaded.
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Op))
      if (GV->isConstant() && !GV->isDeclaration())
        return ReplaceInstUsesWith(LI, GV->getInitializer());

    // Instcombine load (constantexpr_GEP global, 0, ...) into the value loaded.
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Op)) {
      if (CE->getOpcode() == Instruction::GetElementPtr) {
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(CE->getOperand(0)))
          if (GV->isConstant() && !GV->isDeclaration())
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

      } else if (CE->isCast()) {
        if (Instruction *Res = InstCombineLoadCast(*this, LI, TD))
          return Res;
      }
    }
  }
    
  // If this load comes from anywhere in a constant global, and if the global
  // is all undef or zero, we know what it loads.
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Op->getUnderlyingObject())){
    if (GV->isConstant() && GV->hasInitializer()) {
      if (GV->getInitializer()->isNullValue())
        return ReplaceInstUsesWith(LI, Constant::getNullValue(LI.getType()));
      else if (isa<UndefValue>(GV->getInitializer()))
        return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
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
        return SelectInst::Create(SI->getCondition(), V1, V2);
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

/// InstCombineStoreToCast - Fold store V, (cast P) -> store (cast V), P
/// when possible.  This makes it generally easy to do alias analysis and/or
/// SROA/mem2reg of the memory object.
static Instruction *InstCombineStoreToCast(InstCombiner &IC, StoreInst &SI) {
  User *CI = cast<User>(SI.getOperand(1));
  Value *CastOp = CI->getOperand(0);

  const Type *DestPTy = cast<PointerType>(CI->getType())->getElementType();
  const PointerType *SrcTy = dyn_cast<PointerType>(CastOp->getType());
  if (SrcTy == 0) return 0;
  
  const Type *SrcPTy = SrcTy->getElementType();

  if (!DestPTy->isInteger() && !isa<PointerType>(DestPTy))
    return 0;
  
  /// NewGEPIndices - If SrcPTy is an aggregate type, we can emit a "noop gep"
  /// to its first element.  This allows us to handle things like:
  ///   store i32 xxx, (bitcast {foo*, float}* %P to i32*)
  /// on 32-bit hosts.
  SmallVector<Value*, 4> NewGEPIndices;
  
  // If the source is an array, the code below will not succeed.  Check to
  // see if a trivial 'gep P, 0, 0' will help matters.  Only do this for
  // constants.
  if (isa<ArrayType>(SrcPTy) || isa<StructType>(SrcPTy)) {
    // Index through pointer.
    Constant *Zero = Constant::getNullValue(Type::Int32Ty);
    NewGEPIndices.push_back(Zero);
    
    while (1) {
      if (const StructType *STy = dyn_cast<StructType>(SrcPTy)) {
        if (!STy->getNumElements()) /* Struct can be empty {} */
          break;
        NewGEPIndices.push_back(Zero);
        SrcPTy = STy->getElementType(0);
      } else if (const ArrayType *ATy = dyn_cast<ArrayType>(SrcPTy)) {
        NewGEPIndices.push_back(Zero);
        SrcPTy = ATy->getElementType();
      } else {
        break;
      }
    }
    
    SrcTy = PointerType::get(SrcPTy, SrcTy->getAddressSpace());
  }

  if (!SrcPTy->isInteger() && !isa<PointerType>(SrcPTy))
    return 0;
  
  // If the pointers point into different address spaces or if they point to
  // values with different sizes, we can't do the transformation.
  if (SrcTy->getAddressSpace() != 
        cast<PointerType>(CI->getType())->getAddressSpace() ||
      IC.getTargetData().getTypeSizeInBits(SrcPTy) !=
      IC.getTargetData().getTypeSizeInBits(DestPTy))
    return 0;

  // Okay, we are casting from one integer or pointer type to another of
  // the same size.  Instead of casting the pointer before 
  // the store, cast the value to be stored.
  Value *NewCast;
  Value *SIOp0 = SI.getOperand(0);
  Instruction::CastOps opcode = Instruction::BitCast;
  const Type* CastSrcTy = SIOp0->getType();
  const Type* CastDstTy = SrcPTy;
  if (isa<PointerType>(CastDstTy)) {
    if (CastSrcTy->isInteger())
      opcode = Instruction::IntToPtr;
  } else if (isa<IntegerType>(CastDstTy)) {
    if (isa<PointerType>(SIOp0->getType()))
      opcode = Instruction::PtrToInt;
  }
  
  // SIOp0 is a pointer to aggregate and this is a store to the first field,
  // emit a GEP to index into its first field.
  if (!NewGEPIndices.empty()) {
    if (Constant *C = dyn_cast<Constant>(CastOp))
      CastOp = ConstantExpr::getGetElementPtr(C, &NewGEPIndices[0], 
                                              NewGEPIndices.size());
    else
      CastOp = IC.InsertNewInstBefore(
              GetElementPtrInst::Create(CastOp, NewGEPIndices.begin(),
                                        NewGEPIndices.end()), SI);
  }
  
  if (Constant *C = dyn_cast<Constant>(SIOp0))
    NewCast = ConstantExpr::getCast(opcode, C, CastDstTy);
  else
    NewCast = IC.InsertNewInstBefore(
      CastInst::Create(opcode, SIOp0, CastDstTy, SIOp0->getName()+".c"), 
      SI);
  return new StoreInst(NewCast, CastOp);
}

/// equivalentAddressValues - Test if A and B will obviously have the same
/// value. This includes recognizing that %t0 and %t1 will have the same
/// value in code like this:
///   %t0 = getelementptr @a, 0, 3
///   store i32 0, i32* %t0
///   %t1 = getelementptr @a, 0, 3
///   %t2 = load i32* %t1
///
static bool equivalentAddressValues(Value *A, Value *B) {
  // Test if the values are trivially equivalent.
  if (A == B) return true;
  
  // Test if the values come form identical arithmetic instructions.
  if (isa<BinaryOperator>(A) ||
      isa<CastInst>(A) ||
      isa<PHINode>(A) ||
      isa<GetElementPtrInst>(A))
    if (Instruction *BI = dyn_cast<Instruction>(B))
      if (cast<Instruction>(A)->isIdenticalTo(BI))
        return true;
  
  // Otherwise they may not be equivalent.
  return false;
}

Instruction *InstCombiner::visitStoreInst(StoreInst &SI) {
  Value *Val = SI.getOperand(0);
  Value *Ptr = SI.getOperand(1);

  if (isa<UndefValue>(Ptr)) {     // store X, undef -> noop (even if volatile)
    EraseInstFromFunction(SI);
    ++NumCombined;
    return 0;
  }
  
  // If the RHS is an alloca with a single use, zapify the store, making the
  // alloca dead.
  if (Ptr->hasOneUse() && !SI.isVolatile()) {
    if (isa<AllocaInst>(Ptr)) {
      EraseInstFromFunction(SI);
      ++NumCombined;
      return 0;
    }
    
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr))
      if (isa<AllocaInst>(GEP->getOperand(0)) &&
          GEP->getOperand(0)->hasOneUse()) {
        EraseInstFromFunction(SI);
        ++NumCombined;
        return 0;
      }
  }

  // Attempt to improve the alignment.
  unsigned KnownAlign = GetOrEnforceKnownAlignment(Ptr);
  if (KnownAlign >
      (SI.getAlignment() == 0 ? TD->getABITypeAlignment(Val->getType()) :
                                SI.getAlignment()))
    SI.setAlignment(KnownAlign);

  // Do really simple DSE, to catch cases where there are several consequtive
  // stores to the same location, separated by a few arithmetic operations. This
  // situation often occurs with bitfield accesses.
  BasicBlock::iterator BBI = &SI;
  for (unsigned ScanInsts = 6; BBI != SI.getParent()->begin() && ScanInsts;
       --ScanInsts) {
    --BBI;
    
    if (StoreInst *PrevSI = dyn_cast<StoreInst>(BBI)) {
      // Prev store isn't volatile, and stores to the same location?
      if (!PrevSI->isVolatile() &&equivalentAddressValues(PrevSI->getOperand(1),
                                                          SI.getOperand(1))) {
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
      if (LI == Val && equivalentAddressValues(LI->getOperand(0), Ptr) &&
          !SI.isVolatile()) {
        EraseInstFromFunction(SI);
        ++NumCombined;
        return 0;
      }
      // Otherwise, this is a load from some other location.  Stores before it
      // may not be dead.
      break;
    }
    
    // Don't skip over loads or things that can modify memory.
    if (BBI->mayWriteToMemory() || BBI->mayReadFromMemory())
      break;
  }
  
  
  if (SI.isVolatile()) return 0;  // Don't hack volatile stores.

  // store X, null    -> turns into 'unreachable' in SimplifyCFG
  if (isa<ConstantPointerNull>(Ptr)) {
    if (!isa<UndefValue>(Val)) {
      SI.setOperand(0, UndefValue::get(Val->getType()));
      if (Instruction *U = dyn_cast<Instruction>(Val))
        AddToWorkList(U);  // Dropped a use.
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
    if (CE->isCast())
      if (Instruction *Res = InstCombineStoreToCast(*this, SI))
        return Res;

  
  // If this store is the last instruction in the basic block, and if the block
  // ends with an unconditional branch, try to move it to the successor block.
  BBI = &SI; ++BBI;
  if (BranchInst *BI = dyn_cast<BranchInst>(BBI))
    if (BI->isUnconditional())
      if (SimplifyStoreAtEndOfBlock(SI))
        return 0;  // xform done!
  
  return 0;
}

/// SimplifyStoreAtEndOfBlock - Turn things like:
///   if () { *P = v1; } else { *P = v2 }
/// into a phi node with a store in the successor.
///
/// Simplify things like:
///   *P = v1; if () { *P = v2; }
/// into a phi node with a store in the successor.
///
bool InstCombiner::SimplifyStoreAtEndOfBlock(StoreInst &SI) {
  BasicBlock *StoreBB = SI.getParent();
  
  // Check to see if the successor block has exactly two incoming edges.  If
  // so, see if the other predecessor contains a store to the same location.
  // if so, insert a PHI node (if needed) and move the stores down.
  BasicBlock *DestBB = StoreBB->getTerminator()->getSuccessor(0);
  
  // Determine whether Dest has exactly two predecessors and, if so, compute
  // the other predecessor.
  pred_iterator PI = pred_begin(DestBB);
  BasicBlock *OtherBB = 0;
  if (*PI != StoreBB)
    OtherBB = *PI;
  ++PI;
  if (PI == pred_end(DestBB))
    return false;
  
  if (*PI != StoreBB) {
    if (OtherBB)
      return false;
    OtherBB = *PI;
  }
  if (++PI != pred_end(DestBB))
    return false;

  // Bail out if all the relevant blocks aren't distinct (this can happen,
  // for example, if SI is in an infinite loop)
  if (StoreBB == DestBB || OtherBB == DestBB)
    return false;

  // Verify that the other block ends in a branch and is not otherwise empty.
  BasicBlock::iterator BBI = OtherBB->getTerminator();
  BranchInst *OtherBr = dyn_cast<BranchInst>(BBI);
  if (!OtherBr || BBI == OtherBB->begin())
    return false;
  
  // If the other block ends in an unconditional branch, check for the 'if then
  // else' case.  there is an instruction before the branch.
  StoreInst *OtherStore = 0;
  if (OtherBr->isUnconditional()) {
    // If this isn't a store, or isn't a store to the same location, bail out.
    --BBI;
    OtherStore = dyn_cast<StoreInst>(BBI);
    if (!OtherStore || OtherStore->getOperand(1) != SI.getOperand(1))
      return false;
  } else {
    // Otherwise, the other block ended with a conditional branch. If one of the
    // destinations is StoreBB, then we have the if/then case.
    if (OtherBr->getSuccessor(0) != StoreBB && 
        OtherBr->getSuccessor(1) != StoreBB)
      return false;
    
    // Okay, we know that OtherBr now goes to Dest and StoreBB, so this is an
    // if/then triangle.  See if there is a store to the same ptr as SI that
    // lives in OtherBB.
    for (;; --BBI) {
      // Check to see if we find the matching store.
      if ((OtherStore = dyn_cast<StoreInst>(BBI))) {
        if (OtherStore->getOperand(1) != SI.getOperand(1))
          return false;
        break;
      }
      // If we find something that may be using or overwriting the stored
      // value, or if we run out of instructions, we can't do the xform.
      if (BBI->mayReadFromMemory() || BBI->mayWriteToMemory() ||
          BBI == OtherBB->begin())
        return false;
    }
    
    // In order to eliminate the store in OtherBr, we have to
    // make sure nothing reads or overwrites the stored value in
    // StoreBB.
    for (BasicBlock::iterator I = StoreBB->begin(); &*I != &SI; ++I) {
      // FIXME: This should really be AA driven.
      if (I->mayReadFromMemory() || I->mayWriteToMemory())
        return false;
    }
  }
  
  // Insert a PHI node now if we need it.
  Value *MergedVal = OtherStore->getOperand(0);
  if (MergedVal != SI.getOperand(0)) {
    PHINode *PN = PHINode::Create(MergedVal->getType(), "storemerge");
    PN->reserveOperandSpace(2);
    PN->addIncoming(SI.getOperand(0), SI.getParent());
    PN->addIncoming(OtherStore->getOperand(0), OtherBB);
    MergedVal = InsertNewInstBefore(PN, DestBB->front());
  }
  
  // Advance to a place where it is safe to insert the new store and
  // insert it.
  BBI = DestBB->getFirstNonPHI();
  InsertNewInstBefore(new StoreInst(MergedVal, SI.getOperand(1),
                                    OtherStore->isVolatile()), *BBI);
  
  // Nuke the old stores.
  EraseInstFromFunction(SI);
  EraseInstFromFunction(*OtherStore);
  ++NumCombined;
  return true;
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

  // Cannonicalize fcmp_one -> fcmp_oeq
  FCmpInst::Predicate FPred; Value *Y;
  if (match(&BI, m_Br(m_FCmp(FPred, m_Value(X), m_Value(Y)), 
                             TrueDest, FalseDest)))
    if ((FPred == FCmpInst::FCMP_ONE || FPred == FCmpInst::FCMP_OLE ||
         FPred == FCmpInst::FCMP_OGE) && BI.getCondition()->hasOneUse()) {
      FCmpInst *I = cast<FCmpInst>(BI.getCondition());
      FCmpInst::Predicate NewPred = FCmpInst::getInversePredicate(FPred);
      Instruction *NewSCC = new FCmpInst(NewPred, X, Y, "", I);
      NewSCC->takeName(I);
      // Swap Destinations and condition...
      BI.setCondition(NewSCC);
      BI.setSuccessor(0, FalseDest);
      BI.setSuccessor(1, TrueDest);
      RemoveFromWorkList(I);
      I->eraseFromParent();
      AddToWorkList(NewSCC);
      return &BI;
    }

  // Cannonicalize icmp_ne -> icmp_eq
  ICmpInst::Predicate IPred;
  if (match(&BI, m_Br(m_ICmp(IPred, m_Value(X), m_Value(Y)),
                      TrueDest, FalseDest)))
    if ((IPred == ICmpInst::ICMP_NE  || IPred == ICmpInst::ICMP_ULE ||
         IPred == ICmpInst::ICMP_SLE || IPred == ICmpInst::ICMP_UGE ||
         IPred == ICmpInst::ICMP_SGE) && BI.getCondition()->hasOneUse()) {
      ICmpInst *I = cast<ICmpInst>(BI.getCondition());
      ICmpInst::Predicate NewPred = ICmpInst::getInversePredicate(IPred);
      Instruction *NewSCC = new ICmpInst(NewPred, X, Y, "", I);
      NewSCC->takeName(I);
      // Swap Destinations and condition...
      BI.setCondition(NewSCC);
      BI.setSuccessor(0, FalseDest);
      BI.setSuccessor(1, TrueDest);
      RemoveFromWorkList(I);
      I->eraseFromParent();;
      AddToWorkList(NewSCC);
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
        AddToWorkList(I);
        return &SI;
      }
  }
  return 0;
}

Instruction *InstCombiner::visitExtractValueInst(ExtractValueInst &EV) {
  Value *Agg = EV.getAggregateOperand();

  if (!EV.hasIndices())
    return ReplaceInstUsesWith(EV, Agg);

  if (Constant *C = dyn_cast<Constant>(Agg)) {
    if (isa<UndefValue>(C))
      return ReplaceInstUsesWith(EV, UndefValue::get(EV.getType()));
      
    if (isa<ConstantAggregateZero>(C))
      return ReplaceInstUsesWith(EV, Constant::getNullValue(EV.getType()));

    if (isa<ConstantArray>(C) || isa<ConstantStruct>(C)) {
      // Extract the element indexed by the first index out of the constant
      Value *V = C->getOperand(*EV.idx_begin());
      if (EV.getNumIndices() > 1)
        // Extract the remaining indices out of the constant indexed by the
        // first index
        return ExtractValueInst::Create(V, EV.idx_begin() + 1, EV.idx_end());
      else
        return ReplaceInstUsesWith(EV, V);
    }
    return 0; // Can't handle other constants
  } 
  if (InsertValueInst *IV = dyn_cast<InsertValueInst>(Agg)) {
    // We're extracting from an insertvalue instruction, compare the indices
    const unsigned *exti, *exte, *insi, *inse;
    for (exti = EV.idx_begin(), insi = IV->idx_begin(),
         exte = EV.idx_end(), inse = IV->idx_end();
         exti != exte && insi != inse;
         ++exti, ++insi) {
      if (*insi != *exti)
        // The insert and extract both reference distinctly different elements.
        // This means the extract is not influenced by the insert, and we can
        // replace the aggregate operand of the extract with the aggregate
        // operand of the insert. i.e., replace
        // %I = insertvalue { i32, { i32 } } %A, { i32 } { i32 42 }, 1
        // %E = extractvalue { i32, { i32 } } %I, 0
        // with
        // %E = extractvalue { i32, { i32 } } %A, 0
        return ExtractValueInst::Create(IV->getAggregateOperand(),
                                        EV.idx_begin(), EV.idx_end());
    }
    if (exti == exte && insi == inse)
      // Both iterators are at the end: Index lists are identical. Replace
      // %B = insertvalue { i32, { i32 } } %A, i32 42, 1, 0
      // %C = extractvalue { i32, { i32 } } %B, 1, 0
      // with "i32 42"
      return ReplaceInstUsesWith(EV, IV->getInsertedValueOperand());
    if (exti == exte) {
      // The extract list is a prefix of the insert list. i.e. replace
      // %I = insertvalue { i32, { i32 } } %A, i32 42, 1, 0
      // %E = extractvalue { i32, { i32 } } %I, 1
      // with
      // %X = extractvalue { i32, { i32 } } %A, 1
      // %E = insertvalue { i32 } %X, i32 42, 0
      // by switching the order of the insert and extract (though the
      // insertvalue should be left in, since it may have other uses).
      Value *NewEV = InsertNewInstBefore(
        ExtractValueInst::Create(IV->getAggregateOperand(),
                                 EV.idx_begin(), EV.idx_end()),
        EV);
      return InsertValueInst::Create(NewEV, IV->getInsertedValueOperand(),
                                     insi, inse);
    }
    if (insi == inse)
      // The insert list is a prefix of the extract list
      // We can simply remove the common indices from the extract and make it
      // operate on the inserted value instead of the insertvalue result.
      // i.e., replace
      // %I = insertvalue { i32, { i32 } } %A, { i32 } { i32 42 }, 1
      // %E = extractvalue { i32, { i32 } } %I, 1, 0
      // with
      // %E extractvalue { i32 } { i32 42 }, 0
      return ExtractValueInst::Create(IV->getInsertedValueOperand(), 
                                      exti, exte);
  }
  // Can't simplify extracts from other values. Note that nested extracts are
  // already simplified implicitely by the above (extract ( extract (insert) )
  // will be translated into extract ( insert ( extract ) ) first and then just
  // the value inserted, if appropriate).
  return 0;
}

/// CheapToScalarize - Return true if the value is cheaper to scalarize than it
/// is to leave as a vector operation.
static bool CheapToScalarize(Value *V, bool isConstant) {
  if (isa<ConstantAggregateZero>(V)) 
    return true;
  if (ConstantVector *C = dyn_cast<ConstantVector>(V)) {
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
  if (CmpInst *CI = dyn_cast<CmpInst>(I))
    if (CI->hasOneUse() &&
        (CheapToScalarize(CI->getOperand(0), isConstant) ||
         CheapToScalarize(CI->getOperand(1), isConstant)))
      return true;
  
  return false;
}

/// Read and decode a shufflevector mask.
///
/// It turns undef elements into values that are larger than the number of
/// elements in the input.
static std::vector<unsigned> getShuffleMask(const ShuffleVectorInst *SVI) {
  unsigned NElts = SVI->getType()->getNumElements();
  if (isa<ConstantAggregateZero>(SVI->getOperand(2)))
    return std::vector<unsigned>(NElts, 0);
  if (isa<UndefValue>(SVI->getOperand(2)))
    return std::vector<unsigned>(NElts, 2*NElts);

  std::vector<unsigned> Result;
  const ConstantVector *CP = cast<ConstantVector>(SVI->getOperand(2));
  for (User::const_op_iterator i = CP->op_begin(), e = CP->op_end(); i!=e; ++i)
    if (isa<UndefValue>(*i))
      Result.push_back(NElts*2);  // undef -> 8
    else
      Result.push_back(cast<ConstantInt>(*i)->getZExtValue());
  return Result;
}

/// FindScalarElement - Given a vector and an element number, see if the scalar
/// value is already around as a register, for example if it were inserted then
/// extracted from the vector.
static Value *FindScalarElement(Value *V, unsigned EltNo) {
  assert(isa<VectorType>(V->getType()) && "Not looking at a vector?");
  const VectorType *PTy = cast<VectorType>(V->getType());
  unsigned Width = PTy->getNumElements();
  if (EltNo >= Width)  // Out of range access.
    return UndefValue::get(PTy->getElementType());
  
  if (isa<UndefValue>(V))
    return UndefValue::get(PTy->getElementType());
  else if (isa<ConstantAggregateZero>(V))
    return Constant::getNullValue(PTy->getElementType());
  else if (ConstantVector *CP = dyn_cast<ConstantVector>(V))
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
    unsigned LHSWidth =
      cast<VectorType>(SVI->getOperand(0)->getType())->getNumElements();
    unsigned InEl = getShuffleMask(SVI)[EltNo];
    if (InEl < LHSWidth)
      return FindScalarElement(SVI->getOperand(0), InEl);
    else if (InEl < LHSWidth*2)
      return FindScalarElement(SVI->getOperand(1), InEl - LHSWidth);
    else
      return UndefValue::get(PTy->getElementType());
  }
  
  // Otherwise, we don't know.
  return 0;
}

Instruction *InstCombiner::visitExtractElementInst(ExtractElementInst &EI) {
  // If vector val is undef, replace extract with scalar undef.
  if (isa<UndefValue>(EI.getOperand(0)))
    return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));

  // If vector val is constant 0, replace extract with scalar 0.
  if (isa<ConstantAggregateZero>(EI.getOperand(0)))
    return ReplaceInstUsesWith(EI, Constant::getNullValue(EI.getType()));
  
  if (ConstantVector *C = dyn_cast<ConstantVector>(EI.getOperand(0))) {
    // If vector val is constant with all elements the same, replace EI with
    // that element. When the elements are not identical, we cannot replace yet
    // (we do that below, but only when the index is constant).
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
    unsigned IndexVal = IdxC->getZExtValue();
    unsigned VectorWidth = 
      cast<VectorType>(EI.getOperand(0)->getType())->getNumElements();
      
    // If this is extracting an invalid index, turn this into undef, to avoid
    // crashing the code below.
    if (IndexVal >= VectorWidth)
      return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));
    
    // This instruction only demands the single element from the input vector.
    // If the input vector has a single use, simplify it based on this use
    // property.
    if (EI.getOperand(0)->hasOneUse() && VectorWidth != 1) {
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
    
    // If the this extractelement is directly using a bitcast from a vector of
    // the same number of elements, see if we can find the source element from
    // it.  In this case, we will end up needing to bitcast the scalars.
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(EI.getOperand(0))) {
      if (const VectorType *VT = 
              dyn_cast<VectorType>(BCI->getOperand(0)->getType()))
        if (VT->getNumElements() == VectorWidth)
          if (Value *Elt = FindScalarElement(BCI->getOperand(0), IndexVal))
            return new BitCastInst(Elt, EI.getType());
    }
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
          return BinaryOperator::Create(BO->getOpcode(), newEI0, newEI1);
        }
      } else if (isa<LoadInst>(I)) {
        unsigned AS = 
          cast<PointerType>(I->getOperand(0)->getType())->getAddressSpace();
        Value *Ptr = InsertBitCastBefore(I->getOperand(0),
                                         PointerType::get(EI.getType(), AS),EI);
        GetElementPtrInst *GEP =
          GetElementPtrInst::Create(Ptr, EI.getOperand(1), I->getName()+".gep");
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
        unsigned LHSWidth =
          cast<VectorType>(SVI->getOperand(0)->getType())->getNumElements();

        if (SrcIdx < LHSWidth)
          Src = SVI->getOperand(0);
        else if (SrcIdx < LHSWidth*2) {
          SrcIdx -= LHSWidth;
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
  unsigned NumElts = cast<VectorType>(V->getType())->getNumElements();

  if (isa<UndefValue>(V)) {
    Mask.assign(NumElts, UndefValue::get(Type::Int32Ty));
    return true;
  } else if (V == LHS) {
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(ConstantInt::get(Type::Int32Ty, i));
    return true;
  } else if (V == RHS) {
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(ConstantInt::get(Type::Int32Ty, i+NumElts));
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
        Mask[InsertedIdx] = UndefValue::get(Type::Int32Ty);
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
              Mask[InsertedIdx % NumElts] = 
                 ConstantInt::get(Type::Int32Ty, ExtractedIdx);
            } else {
              assert(EI->getOperand(0) == RHS);
              Mask[InsertedIdx % NumElts] = 
                ConstantInt::get(Type::Int32Ty, ExtractedIdx+NumElts);
              
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
  assert(isa<VectorType>(V->getType()) && 
         (RHS == 0 || V->getType() == RHS->getType()) &&
         "Invalid shuffle!");
  unsigned NumElts = cast<VectorType>(V->getType())->getNumElements();

  if (isa<UndefValue>(V)) {
    Mask.assign(NumElts, UndefValue::get(Type::Int32Ty));
    return V;
  } else if (isa<ConstantAggregateZero>(V)) {
    Mask.assign(NumElts, ConstantInt::get(Type::Int32Ty, 0));
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
          Mask[InsertedIdx % NumElts] = 
            ConstantInt::get(Type::Int32Ty, NumElts+ExtractedIdx);
          return V;
        }
        
        if (VecOp == RHS) {
          Value *V = CollectShuffleElements(EI->getOperand(0), Mask, RHS);
          // Everything but the extracted element is replaced with the RHS.
          for (unsigned i = 0; i != NumElts; ++i) {
            if (i != InsertedIdx)
              Mask[i] = ConstantInt::get(Type::Int32Ty, NumElts+i);
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
    Mask.push_back(ConstantInt::get(Type::Int32Ty, i));
  return V;
}

Instruction *InstCombiner::visitInsertElementInst(InsertElementInst &IE) {
  Value *VecOp    = IE.getOperand(0);
  Value *ScalarOp = IE.getOperand(1);
  Value *IdxOp    = IE.getOperand(2);
  
  // Inserting an undef or into an undefined place, remove this.
  if (isa<UndefValue>(ScalarOp) || isa<UndefValue>(IdxOp))
    ReplaceInstUsesWith(IE, VecOp);
  
  // If the inserted element was extracted from some other vector, and if the 
  // indexes are constant, try to turn this into a shufflevector operation.
  if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)) {
    if (isa<ConstantInt>(EI->getOperand(1)) && isa<ConstantInt>(IdxOp) &&
        EI->getOperand(0)->getType() == IE.getType()) {
      unsigned NumVectorElts = IE.getType()->getNumElements();
      unsigned ExtractedIdx =
        cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
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
          Mask.assign(NumVectorElts, UndefValue::get(Type::Int32Ty));
        else {
          assert(isa<ConstantAggregateZero>(VecOp) && "Unknown thing");
          Mask.assign(NumVectorElts, ConstantInt::get(Type::Int32Ty,
                                                       NumVectorElts));
        } 
        Mask[InsertedIdx] = ConstantInt::get(Type::Int32Ty, ExtractedIdx);
        return new ShuffleVectorInst(EI->getOperand(0), VecOp,
                                     ConstantVector::get(Mask));
      }
      
      // If this insertelement isn't used by some other insertelement, turn it
      // (and any insertelements it points to), into one big shuffle.
      if (!IE.hasOneUse() || !isa<InsertElementInst>(IE.use_back())) {
        std::vector<Constant*> Mask;
        Value *RHS = 0;
        Value *LHS = CollectShuffleElements(&IE, Mask, RHS);
        if (RHS == 0) RHS = UndefValue::get(LHS->getType());
        // We now have a shuffle of LHS, RHS, Mask.
        return new ShuffleVectorInst(LHS, RHS, ConstantVector::get(Mask));
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

  uint64_t UndefElts;
  unsigned VWidth = cast<VectorType>(SVI.getType())->getNumElements();

  if (VWidth != cast<VectorType>(LHS->getType())->getNumElements())
    return 0;

  uint64_t AllOnesEltMask = ~0ULL >> (64-VWidth);
  if (VWidth <= 64 &&
      SimplifyDemandedVectorElts(&SVI, AllOnesEltMask, UndefElts)) {
    LHS = SVI.getOperand(0);
    RHS = SVI.getOperand(1);
    MadeChange = true;
  }
  
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
        Elts.push_back(UndefValue::get(Type::Int32Ty));
      else {
        if ((Mask[i] >= e && isa<UndefValue>(RHS)) ||
            (Mask[i] <  e && isa<UndefValue>(LHS))) {
          Mask[i] = 2*e;     // Turn into undef.
          Elts.push_back(UndefValue::get(Type::Int32Ty));
        } else {
          Mask[i] = Mask[i] % e;  // Force to LHS.
          Elts.push_back(ConstantInt::get(Type::Int32Ty, Mask[i]));
        }
      }
    }
    SVI.setOperand(0, SVI.getOperand(1));
    SVI.setOperand(1, UndefValue::get(RHS->getType()));
    SVI.setOperand(2, ConstantVector::get(Elts));
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
        unsigned LHSInNElts =
          cast<VectorType>(LHSSVI->getOperand(0)->getType())->getNumElements();
        std::vector<Constant*> Elts;
        for (unsigned i = 0, e = NewMask.size(); i != e; ++i) {
          if (NewMask[i] >= LHSInNElts*2) {
            Elts.push_back(UndefValue::get(Type::Int32Ty));
          } else {
            Elts.push_back(ConstantInt::get(Type::Int32Ty, NewMask[i]));
          }
        }
        return new ShuffleVectorInst(LHSSVI->getOperand(0),
                                     LHSSVI->getOperand(1),
                                     ConstantVector::get(Elts));
      }
    }
  }

  return MadeChange ? &SVI : 0;
}




/// TryToSinkInstruction - Try to move the specified instruction from its
/// current block into the beginning of DestBlock, which can only happen if it's
/// safe to move the instruction past all of the instructions between it and the
/// end of its block.
static bool TryToSinkInstruction(Instruction *I, BasicBlock *DestBlock) {
  assert(I->hasOneUse() && "Invariants didn't hold!");

  // Cannot move control-flow-involving, volatile loads, vaarg, etc.
  if (isa<PHINode>(I) || I->mayWriteToMemory() || isa<TerminatorInst>(I))
    return false;

  // Do not sink alloca instructions out of the entry block.
  if (isa<AllocaInst>(I) && I->getParent() ==
        &DestBlock->getParent()->getEntryBlock())
    return false;

  // We can only sink load instructions if there is nothing between the load and
  // the end of block that could change the value.
  if (I->mayReadFromMemory()) {
    for (BasicBlock::iterator Scan = I, E = I->getParent()->end();
         Scan != E; ++Scan)
      if (Scan->mayWriteToMemory())
        return false;
  }

  BasicBlock::iterator InsertPos = DestBlock->getFirstNonPHI();

  I->moveBefore(InsertPos);
  ++NumSunkInst;
  return true;
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
                                       SmallPtrSet<BasicBlock*, 64> &Visited,
                                       InstCombiner &IC,
                                       const TargetData *TD) {
  SmallVector<BasicBlock*, 256> Worklist;
  Worklist.push_back(BB);

  while (!Worklist.empty()) {
    BB = Worklist.back();
    Worklist.pop_back();
    
    // We have now visited this block!  If we've already been here, ignore it.
    if (!Visited.insert(BB)) continue;

    DbgInfoIntrinsic *DBI_Prev = NULL;
    for (BasicBlock::iterator BBI = BB->begin(), E = BB->end(); BBI != E; ) {
      Instruction *Inst = BBI++;
      
      // DCE instruction if trivially dead.
      if (isInstructionTriviallyDead(Inst)) {
        ++NumDeadInst;
        DOUT << "IC: DCE: " << *Inst;
        Inst->eraseFromParent();
        continue;
      }
      
      // ConstantProp instruction if trivially constant.
      if (Constant *C = ConstantFoldInstruction(Inst, TD)) {
        DOUT << "IC: ConstFold to: " << *C << " from: " << *Inst;
        Inst->replaceAllUsesWith(C);
        ++NumConstProp;
        Inst->eraseFromParent();
        continue;
      }
     
      // If there are two consecutive llvm.dbg.stoppoint calls then
      // it is likely that the optimizer deleted code in between these
      // two intrinsics. 
      DbgInfoIntrinsic *DBI_Next = dyn_cast<DbgInfoIntrinsic>(Inst);
      if (DBI_Next) {
        if (DBI_Prev
            && DBI_Prev->getIntrinsicID() == llvm::Intrinsic::dbg_stoppoint
            && DBI_Next->getIntrinsicID() == llvm::Intrinsic::dbg_stoppoint) {
          IC.RemoveFromWorkList(DBI_Prev);
          DBI_Prev->eraseFromParent();
        }
        DBI_Prev = DBI_Next;
      }

      IC.AddToWorkList(Inst);
    }

    // Recursively visit successors.  If this is a branch or switch on a
    // constant, only visit the reachable successor.
    TerminatorInst *TI = BB->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      if (BI->isConditional() && isa<ConstantInt>(BI->getCondition())) {
        bool CondVal = cast<ConstantInt>(BI->getCondition())->getZExtValue();
        BasicBlock *ReachableBB = BI->getSuccessor(!CondVal);
        Worklist.push_back(ReachableBB);
        continue;
      }
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      if (ConstantInt *Cond = dyn_cast<ConstantInt>(SI->getCondition())) {
        // See if this is an explicit destination.
        for (unsigned i = 1, e = SI->getNumSuccessors(); i != e; ++i)
          if (SI->getCaseValue(i) == Cond) {
            BasicBlock *ReachableBB = SI->getSuccessor(i);
            Worklist.push_back(ReachableBB);
            continue;
          }
        
        // Otherwise it is the default destination.
        Worklist.push_back(SI->getSuccessor(0));
        continue;
      }
    }
    
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      Worklist.push_back(TI->getSuccessor(i));
  }
}

bool InstCombiner::DoOneIteration(Function &F, unsigned Iteration) {
  bool Changed = false;
  TD = &getAnalysis<TargetData>();
  
  DEBUG(DOUT << "\n\nINSTCOMBINE ITERATION #" << Iteration << " on "
             << F.getNameStr() << "\n");

  {
    // Do a depth-first traversal of the function, populate the worklist with
    // the reachable instructions.  Ignore blocks that are not reachable.  Keep
    // track of which blocks we visit.
    SmallPtrSet<BasicBlock*, 64> Visited;
    AddReachableCodeToWorklist(F.begin(), Visited, *this, TD);

    // Do a quick scan over the function.  If we find any blocks that are
    // unreachable, remove any instructions inside of them.  This prevents
    // the instcombine code from having to deal with some bad special cases.
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
      if (!Visited.count(BB)) {
        Instruction *Term = BB->getTerminator();
        while (Term != BB->begin()) {   // Remove instrs bottom-up
          BasicBlock::iterator I = Term; --I;

          DOUT << "IC: DCE: " << *I;
          ++NumDeadInst;

          if (!I->use_empty())
            I->replaceAllUsesWith(UndefValue::get(I->getType()));
          I->eraseFromParent();
          Changed = true;
        }
      }
  }

  while (!Worklist.empty()) {
    Instruction *I = RemoveOneFromWorkList();
    if (I == 0) continue;  // skip null values.

    // Check to see if we can DCE the instruction.
    if (isInstructionTriviallyDead(I)) {
      // Add operands to the worklist.
      if (I->getNumOperands() < 4)
        AddUsesToWorkList(*I);
      ++NumDeadInst;

      DOUT << "IC: DCE: " << *I;

      I->eraseFromParent();
      RemoveFromWorkList(I);
      Changed = true;
      continue;
    }

    // Instruction isn't dead, see if we can constant propagate it.
    if (Constant *C = ConstantFoldInstruction(I, TD)) {
      DOUT << "IC: ConstFold to: " << *C << " from: " << *I;

      // Add operands to the worklist.
      AddUsesToWorkList(*I);
      ReplaceInstUsesWith(*I, C);

      ++NumConstProp;
      I->eraseFromParent();
      RemoveFromWorkList(I);
      Changed = true;
      continue;
    }

    if (TD && I->getType()->getTypeID() == Type::VoidTyID) {
      // See if we can constant fold its operands.
      for (User::op_iterator i = I->op_begin(), e = I->op_end(); i != e; ++i)
        if (ConstantExpr *CE = dyn_cast<ConstantExpr>(i))
          if (Constant *NewC = ConstantFoldConstantExpression(CE, TD))
            if (NewC != CE) {
              i->set(NewC);
              Changed = true;
            }
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
#ifndef NDEBUG
    std::string OrigI;
#endif
    DEBUG(std::ostringstream SS; I->print(SS); OrigI = SS.str(););
    if (Instruction *Result = visit(*I)) {
      ++NumCombined;
      // Should we replace the old instruction with a new one?
      if (Result != I) {
        DOUT << "IC: Old = " << *I
             << "    New = " << *Result;

        // Everything uses the new instruction now.
        I->replaceAllUsesWith(Result);

        // Push the new instruction and any users onto the worklist.
        AddToWorkList(Result);
        AddUsersToWorkList(*Result);

        // Move the name to the new instruction first.
        Result->takeName(I);

        // Insert the new instruction into the basic block...
        BasicBlock *InstParent = I->getParent();
        BasicBlock::iterator InsertPos = I;

        if (!isa<PHINode>(Result))        // If combining a PHI, don't insert
          while (isa<PHINode>(InsertPos)) // middle of a block of PHIs.
            ++InsertPos;

        InstParent->getInstList().insert(InsertPos, Result);

        // Make sure that we reprocess all operands now that we reduced their
        // use counts.
        AddUsesToWorkList(*I);

        // Instructions can end up on the worklist more than once.  Make sure
        // we do not process an instruction that has been deleted.
        RemoveFromWorkList(I);

        // Erase the old instruction.
        InstParent->getInstList().erase(I);
      } else {
#ifndef NDEBUG
        DOUT << "IC: Mod = " << OrigI
             << "    New = " << *I;
#endif

        // If the instruction was modified, it's possible that it is now dead.
        // if so, remove it.
        if (isInstructionTriviallyDead(I)) {
          // Make sure we process all operands now that we are reducing their
          // use counts.
          AddUsesToWorkList(*I);

          // Instructions may end up in the worklist more than once.  Erase all
          // occurrences of this instruction.
          RemoveFromWorkList(I);
          I->eraseFromParent();
        } else {
          AddToWorkList(I);
          AddUsersToWorkList(*I);
        }
      }
      Changed = true;
    }
  }

  assert(WorklistMap.empty() && "Worklist empty, but map not?");
    
  // Do an explicit clear, this shrinks the map if needed.
  WorklistMap.clear();
  return Changed;
}


bool InstCombiner::runOnFunction(Function &F) {
  MustPreserveLCSSA = mustPreserveAnalysisID(LCSSAID);
  
  bool EverMadeChange = false;

  // Iterate while there is work to do.
  unsigned Iteration = 0;
  while (DoOneIteration(F, Iteration++))
    EverMadeChange = true;
  return EverMadeChange;
}

FunctionPass *llvm::createInstructionCombiningPass() {
  return new InstCombiner();
}
