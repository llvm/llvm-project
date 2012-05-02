//===- Reassociate.cpp - Reassociate binary expressions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass reassociates commutative expressions in an order that is designed
// to promote better constant propagation, GCSE, LICM, PRE, etc.
//
// For example: 4 + (x + 5) -> x + (4 + 5)
//
// In the implementation of this algorithm, constants are assigned rank = 0,
// function arguments are rank = 1, and other values are assigned ranks
// corresponding to the reverse post order traversal of current function
// (starting at 2), which effectively gives values in deep loops higher rank
// than values not in loops.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reassociate"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumLinear , "Number of insts linearized");
STATISTIC(NumChanged, "Number of insts reassociated");
STATISTIC(NumAnnihil, "Number of expr tree annihilated");
STATISTIC(NumFactor , "Number of multiplies factored");

namespace {
  struct ValueEntry {
    unsigned Rank;
    Value *Op;
    ValueEntry(unsigned R, Value *O) : Rank(R), Op(O) {}
  };
  inline bool operator<(const ValueEntry &LHS, const ValueEntry &RHS) {
    return LHS.Rank > RHS.Rank;   // Sort so that highest rank goes to start.
  }
}

#ifndef NDEBUG
/// PrintOps - Print out the expression identified in the Ops list.
///
static void PrintOps(Instruction *I, const SmallVectorImpl<ValueEntry> &Ops) {
  Module *M = I->getParent()->getParent()->getParent();
  dbgs() << Instruction::getOpcodeName(I->getOpcode()) << " "
       << *Ops[0].Op->getType() << '\t';
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    dbgs() << "[ ";
    WriteAsOperand(dbgs(), Ops[i].Op, false, M);
    dbgs() << ", #" << Ops[i].Rank << "] ";
  }
}
#endif
  
namespace {
  class Reassociate : public FunctionPass {
    DenseMap<BasicBlock*, unsigned> RankMap;
    DenseMap<AssertingVH<Value>, unsigned> ValueRankMap;
    SmallVector<WeakVH, 8> RedoInsts;
    SmallVector<WeakVH, 8> DeadInsts;
    bool MadeChange;
  public:
    static char ID; // Pass identification, replacement for typeid
    Reassociate() : FunctionPass(ID) {
      initializeReassociatePass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
  private:
    void BuildRankMap(Function &F);
    unsigned getRank(Value *V);
    Value *ReassociateExpression(BinaryOperator *I);
    void RewriteExprTree(BinaryOperator *I, SmallVectorImpl<ValueEntry> &Ops,
                         unsigned Idx = 0);
    Value *OptimizeExpression(BinaryOperator *I,
                              SmallVectorImpl<ValueEntry> &Ops);
    Value *OptimizeAdd(Instruction *I, SmallVectorImpl<ValueEntry> &Ops);
    void LinearizeExprTree(BinaryOperator *I, SmallVectorImpl<ValueEntry> &Ops);
    void LinearizeExpr(BinaryOperator *I);
    Value *RemoveFactorFromExpression(Value *V, Value *Factor);
    void ReassociateInst(BasicBlock::iterator &BBI);
    
    void RemoveDeadBinaryOp(Value *V);
  };
}

char Reassociate::ID = 0;
INITIALIZE_PASS(Reassociate, "reassociate",
                "Reassociate expressions", false, false)

// Public interface to the Reassociate pass
FunctionPass *llvm::createReassociatePass() { return new Reassociate(); }

void Reassociate::RemoveDeadBinaryOp(Value *V) {
  Instruction *Op = dyn_cast<Instruction>(V);
  if (!Op || !isa<BinaryOperator>(Op))
    return;
  
  Value *LHS = Op->getOperand(0), *RHS = Op->getOperand(1);
  
  ValueRankMap.erase(Op);
  DeadInsts.push_back(Op);
  RemoveDeadBinaryOp(LHS);
  RemoveDeadBinaryOp(RHS);
}


static bool isUnmovableInstruction(Instruction *I) {
  if (I->getOpcode() == Instruction::PHI ||
      I->getOpcode() == Instruction::Alloca ||
      I->getOpcode() == Instruction::Load ||
      I->getOpcode() == Instruction::Invoke ||
      (I->getOpcode() == Instruction::Call &&
       !isa<DbgInfoIntrinsic>(I)) ||
      I->getOpcode() == Instruction::UDiv || 
      I->getOpcode() == Instruction::SDiv ||
      I->getOpcode() == Instruction::FDiv ||
      I->getOpcode() == Instruction::URem ||
      I->getOpcode() == Instruction::SRem ||
      I->getOpcode() == Instruction::FRem)
    return true;
  return false;
}

void Reassociate::BuildRankMap(Function &F) {
  unsigned i = 2;

  // Assign distinct ranks to function arguments
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I)
    ValueRankMap[&*I] = ++i;

  ReversePostOrderTraversal<Function*> RPOT(&F);
  for (ReversePostOrderTraversal<Function*>::rpo_iterator I = RPOT.begin(),
         E = RPOT.end(); I != E; ++I) {
    BasicBlock *BB = *I;
    unsigned BBRank = RankMap[BB] = ++i << 16;

    // Walk the basic block, adding precomputed ranks for any instructions that
    // we cannot move.  This ensures that the ranks for these instructions are
    // all different in the block.
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (isUnmovableInstruction(I))
        ValueRankMap[&*I] = ++BBRank;
  }
}

unsigned Reassociate::getRank(Value *V) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) {
    if (isa<Argument>(V)) return ValueRankMap[V];   // Function argument.
    return 0;  // Otherwise it's a global or constant, rank 0.
  }

  if (unsigned Rank = ValueRankMap[I])
    return Rank;    // Rank already known?

  // If this is an expression, return the 1+MAX(rank(LHS), rank(RHS)) so that
  // we can reassociate expressions for code motion!  Since we do not recurse
  // for PHI nodes, we cannot have infinite recursion here, because there
  // cannot be loops in the value graph that do not go through PHI nodes.
  unsigned Rank = 0, MaxRank = RankMap[I->getParent()];
  for (unsigned i = 0, e = I->getNumOperands();
       i != e && Rank != MaxRank; ++i)
    Rank = std::max(Rank, getRank(I->getOperand(i)));

  // If this is a not or neg instruction, do not count it for rank.  This
  // assures us that X and ~X will have the same rank.
  if (!I->getType()->isIntegerTy() ||
      (!BinaryOperator::isNot(I) && !BinaryOperator::isNeg(I)))
    ++Rank;

  //DEBUG(dbgs() << "Calculated Rank[" << V->getName() << "] = "
  //     << Rank << "\n");

  return ValueRankMap[I] = Rank;
}

/// isReassociableOp - Return true if V is an instruction of the specified
/// opcode and if it only has one use.
static BinaryOperator *isReassociableOp(Value *V, unsigned Opcode) {
  if ((V->hasOneUse() || V->use_empty()) && isa<Instruction>(V) &&
      cast<Instruction>(V)->getOpcode() == Opcode)
    return cast<BinaryOperator>(V);
  return 0;
}

/// LowerNegateToMultiply - Replace 0-X with X*-1.
///
static Instruction *LowerNegateToMultiply(Instruction *Neg,
                         DenseMap<AssertingVH<Value>, unsigned> &ValueRankMap) {
  Constant *Cst = Constant::getAllOnesValue(Neg->getType());

  Instruction *Res = BinaryOperator::CreateMul(Neg->getOperand(1), Cst, "",Neg);
  ValueRankMap.erase(Neg);
  Res->takeName(Neg);
  Neg->replaceAllUsesWith(Res);
  Res->setDebugLoc(Neg->getDebugLoc());
  Neg->eraseFromParent();
  return Res;
}

// Given an expression of the form '(A+B)+(D+C)', turn it into '(((A+B)+C)+D)'.
// Note that if D is also part of the expression tree that we recurse to
// linearize it as well.  Besides that case, this does not recurse into A,B, or
// C.
void Reassociate::LinearizeExpr(BinaryOperator *I) {
  BinaryOperator *LHS = cast<BinaryOperator>(I->getOperand(0));
  BinaryOperator *RHS = cast<BinaryOperator>(I->getOperand(1));
  assert(isReassociableOp(LHS, I->getOpcode()) &&
         isReassociableOp(RHS, I->getOpcode()) &&
         "Not an expression that needs linearization?");

  DEBUG(dbgs() << "Linear" << *LHS << '\n' << *RHS << '\n' << *I << '\n');

  // Move the RHS instruction to live immediately before I, avoiding breaking
  // dominator properties.
  RHS->moveBefore(I);

  // Move operands around to do the linearization.
  I->setOperand(1, RHS->getOperand(0));
  RHS->setOperand(0, LHS);
  I->setOperand(0, RHS);

  // Conservatively clear all the optional flags, which may not hold
  // after the reassociation.
  I->clearSubclassOptionalData();
  LHS->clearSubclassOptionalData();
  RHS->clearSubclassOptionalData();

  ++NumLinear;
  MadeChange = true;
  DEBUG(dbgs() << "Linearized: " << *I << '\n');

  // If D is part of this expression tree, tail recurse.
  if (isReassociableOp(I->getOperand(1), I->getOpcode()))
    LinearizeExpr(I);
}


/// LinearizeExprTree - Given an associative binary expression tree, traverse
/// all of the uses putting it into canonical form.  This forces a left-linear
/// form of the expression (((a+b)+c)+d), and collects information about the
/// rank of the non-tree operands.
///
/// NOTE: These intentionally destroys the expression tree operands (turning
/// them into undef values) to reduce #uses of the values.  This means that the
/// caller MUST use something like RewriteExprTree to put the values back in.
///
void Reassociate::LinearizeExprTree(BinaryOperator *I,
                                    SmallVectorImpl<ValueEntry> &Ops) {
  Value *LHS = I->getOperand(0), *RHS = I->getOperand(1);
  unsigned Opcode = I->getOpcode();

  // First step, linearize the expression if it is in ((A+B)+(C+D)) form.
  BinaryOperator *LHSBO = isReassociableOp(LHS, Opcode);
  BinaryOperator *RHSBO = isReassociableOp(RHS, Opcode);

  // If this is a multiply expression tree and it contains internal negations,
  // transform them into multiplies by -1 so they can be reassociated.
  if (I->getOpcode() == Instruction::Mul) {
    if (!LHSBO && LHS->hasOneUse() && BinaryOperator::isNeg(LHS)) {
      LHS = LowerNegateToMultiply(cast<Instruction>(LHS), ValueRankMap);
      LHSBO = isReassociableOp(LHS, Opcode);
    }
    if (!RHSBO && RHS->hasOneUse() && BinaryOperator::isNeg(RHS)) {
      RHS = LowerNegateToMultiply(cast<Instruction>(RHS), ValueRankMap);
      RHSBO = isReassociableOp(RHS, Opcode);
    }
  }

  if (!LHSBO) {
    if (!RHSBO) {
      // Neither the LHS or RHS as part of the tree, thus this is a leaf.  As
      // such, just remember these operands and their rank.
      Ops.push_back(ValueEntry(getRank(LHS), LHS));
      Ops.push_back(ValueEntry(getRank(RHS), RHS));
      
      // Clear the leaves out.
      I->setOperand(0, UndefValue::get(I->getType()));
      I->setOperand(1, UndefValue::get(I->getType()));
      return;
    }
    
    // Turn X+(Y+Z) -> (Y+Z)+X
    std::swap(LHSBO, RHSBO);
    std::swap(LHS, RHS);
    bool Success = !I->swapOperands();
    assert(Success && "swapOperands failed");
    (void)Success;
    MadeChange = true;
  } else if (RHSBO) {
    // Turn (A+B)+(C+D) -> (((A+B)+C)+D).  This guarantees the RHS is not
    // part of the expression tree.
    LinearizeExpr(I);
    LHS = LHSBO = cast<BinaryOperator>(I->getOperand(0));
    RHS = I->getOperand(1);
    RHSBO = 0;
  }

  // Okay, now we know that the LHS is a nested expression and that the RHS is
  // not.  Perform reassociation.
  assert(!isReassociableOp(RHS, Opcode) && "LinearizeExpr failed!");

  // Move LHS right before I to make sure that the tree expression dominates all
  // values.
  LHSBO->moveBefore(I);

  // Linearize the expression tree on the LHS.
  LinearizeExprTree(LHSBO, Ops);

  // Remember the RHS operand and its rank.
  Ops.push_back(ValueEntry(getRank(RHS), RHS));
  
  // Clear the RHS leaf out.
  I->setOperand(1, UndefValue::get(I->getType()));
}

// RewriteExprTree - Now that the operands for this expression tree are
// linearized and optimized, emit them in-order.  This function is written to be
// tail recursive.
void Reassociate::RewriteExprTree(BinaryOperator *I,
                                  SmallVectorImpl<ValueEntry> &Ops,
                                  unsigned i) {
  if (i+2 == Ops.size()) {
    if (I->getOperand(0) != Ops[i].Op ||
        I->getOperand(1) != Ops[i+1].Op) {
      Value *OldLHS = I->getOperand(0);
      DEBUG(dbgs() << "RA: " << *I << '\n');
      I->setOperand(0, Ops[i].Op);
      I->setOperand(1, Ops[i+1].Op);

      // Clear all the optional flags, which may not hold after the
      // reassociation if the expression involved more than just this operation.
      if (Ops.size() != 2)
        I->clearSubclassOptionalData();

      DEBUG(dbgs() << "TO: " << *I << '\n');
      MadeChange = true;
      ++NumChanged;
      
      // If we reassociated a tree to fewer operands (e.g. (1+a+2) -> (a+3)
      // delete the extra, now dead, nodes.
      RemoveDeadBinaryOp(OldLHS);
    }
    return;
  }
  assert(i+2 < Ops.size() && "Ops index out of range!");

  if (I->getOperand(1) != Ops[i].Op) {
    DEBUG(dbgs() << "RA: " << *I << '\n');
    I->setOperand(1, Ops[i].Op);

    // Conservatively clear all the optional flags, which may not hold
    // after the reassociation.
    I->clearSubclassOptionalData();

    DEBUG(dbgs() << "TO: " << *I << '\n');
    MadeChange = true;
    ++NumChanged;
  }
  
  BinaryOperator *LHS = cast<BinaryOperator>(I->getOperand(0));
  assert(LHS->getOpcode() == I->getOpcode() &&
         "Improper expression tree!");
  
  // Compactify the tree instructions together with each other to guarantee
  // that the expression tree is dominated by all of Ops.
  LHS->moveBefore(I);
  RewriteExprTree(LHS, Ops, i+1);
}



// NegateValue - Insert instructions before the instruction pointed to by BI,
// that computes the negative version of the value specified.  The negative
// version of the value is returned, and BI is left pointing at the instruction
// that should be processed next by the reassociation pass.
//
static Value *NegateValue(Value *V, Instruction *BI) {
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getNeg(C);
  
  // We are trying to expose opportunity for reassociation.  One of the things
  // that we want to do to achieve this is to push a negation as deep into an
  // expression chain as possible, to expose the add instructions.  In practice,
  // this means that we turn this:
  //   X = -(A+12+C+D)   into    X = -A + -12 + -C + -D = -12 + -A + -C + -D
  // so that later, a: Y = 12+X could get reassociated with the -12 to eliminate
  // the constants.  We assume that instcombine will clean up the mess later if
  // we introduce tons of unnecessary negation instructions.
  //
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (I->getOpcode() == Instruction::Add && I->hasOneUse()) {
      // Push the negates through the add.
      I->setOperand(0, NegateValue(I->getOperand(0), BI));
      I->setOperand(1, NegateValue(I->getOperand(1), BI));

      // We must move the add instruction here, because the neg instructions do
      // not dominate the old add instruction in general.  By moving it, we are
      // assured that the neg instructions we just inserted dominate the 
      // instruction we are about to insert after them.
      //
      I->moveBefore(BI);
      I->setName(I->getName()+".neg");
      return I;
    }
  
  // Okay, we need to materialize a negated version of V with an instruction.
  // Scan the use lists of V to see if we have one already.
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E;++UI){
    User *U = *UI;
    if (!BinaryOperator::isNeg(U)) continue;

    // We found one!  Now we have to make sure that the definition dominates
    // this use.  We do this by moving it to the entry block (if it is a
    // non-instruction value) or right after the definition.  These negates will
    // be zapped by reassociate later, so we don't need much finesse here.
    BinaryOperator *TheNeg = cast<BinaryOperator>(U);

    // Verify that the negate is in this function, V might be a constant expr.
    if (TheNeg->getParent()->getParent() != BI->getParent()->getParent())
      continue;
    
    BasicBlock::iterator InsertPt;
    if (Instruction *InstInput = dyn_cast<Instruction>(V)) {
      if (InvokeInst *II = dyn_cast<InvokeInst>(InstInput)) {
        InsertPt = II->getNormalDest()->begin();
      } else {
        InsertPt = InstInput;
        ++InsertPt;
      }
      while (isa<PHINode>(InsertPt)) ++InsertPt;
    } else {
      InsertPt = TheNeg->getParent()->getParent()->getEntryBlock().begin();
    }
    TheNeg->moveBefore(InsertPt);
    return TheNeg;
  }

  // Insert a 'neg' instruction that subtracts the value from zero to get the
  // negation.
  return BinaryOperator::CreateNeg(V, V->getName() + ".neg", BI);
}

/// ShouldBreakUpSubtract - Return true if we should break up this subtract of
/// X-Y into (X + -Y).
static bool ShouldBreakUpSubtract(Instruction *Sub) {
  // If this is a negation, we can't split it up!
  if (BinaryOperator::isNeg(Sub))
    return false;
  
  // Don't bother to break this up unless either the LHS is an associable add or
  // subtract or if this is only used by one.
  if (isReassociableOp(Sub->getOperand(0), Instruction::Add) ||
      isReassociableOp(Sub->getOperand(0), Instruction::Sub))
    return true;
  if (isReassociableOp(Sub->getOperand(1), Instruction::Add) ||
      isReassociableOp(Sub->getOperand(1), Instruction::Sub))
    return true;
  if (Sub->hasOneUse() && 
      (isReassociableOp(Sub->use_back(), Instruction::Add) ||
       isReassociableOp(Sub->use_back(), Instruction::Sub)))
    return true;
    
  return false;
}

/// BreakUpSubtract - If we have (X-Y), and if either X is an add, or if this is
/// only used by an add, transform this into (X+(0-Y)) to promote better
/// reassociation.
static Instruction *BreakUpSubtract(Instruction *Sub,
                         DenseMap<AssertingVH<Value>, unsigned> &ValueRankMap) {
  // Convert a subtract into an add and a neg instruction. This allows sub
  // instructions to be commuted with other add instructions.
  //
  // Calculate the negative value of Operand 1 of the sub instruction,
  // and set it as the RHS of the add instruction we just made.
  //
  Value *NegVal = NegateValue(Sub->getOperand(1), Sub);
  Instruction *New =
    BinaryOperator::CreateAdd(Sub->getOperand(0), NegVal, "", Sub);
  New->takeName(Sub);

  // Everyone now refers to the add instruction.
  ValueRankMap.erase(Sub);
  Sub->replaceAllUsesWith(New);
  New->setDebugLoc(Sub->getDebugLoc());
  Sub->eraseFromParent();

  DEBUG(dbgs() << "Negated: " << *New << '\n');
  return New;
}

/// ConvertShiftToMul - If this is a shift of a reassociable multiply or is used
/// by one, change this into a multiply by a constant to assist with further
/// reassociation.
static Instruction *ConvertShiftToMul(Instruction *Shl,
                         DenseMap<AssertingVH<Value>, unsigned> &ValueRankMap) {
  // If an operand of this shift is a reassociable multiply, or if the shift
  // is used by a reassociable multiply or add, turn into a multiply.
  if (isReassociableOp(Shl->getOperand(0), Instruction::Mul) ||
      (Shl->hasOneUse() && 
       (isReassociableOp(Shl->use_back(), Instruction::Mul) ||
        isReassociableOp(Shl->use_back(), Instruction::Add)))) {
    Constant *MulCst = ConstantInt::get(Shl->getType(), 1);
    MulCst = ConstantExpr::getShl(MulCst, cast<Constant>(Shl->getOperand(1)));
    
    Instruction *Mul =
      BinaryOperator::CreateMul(Shl->getOperand(0), MulCst, "", Shl);
    ValueRankMap.erase(Shl);
    Mul->takeName(Shl);
    Shl->replaceAllUsesWith(Mul);
    Mul->setDebugLoc(Shl->getDebugLoc());
    Shl->eraseFromParent();
    return Mul;
  }
  return 0;
}

// Scan backwards and forwards among values with the same rank as element i to
// see if X exists.  If X does not exist, return i.  This is useful when
// scanning for 'x' when we see '-x' because they both get the same rank.
static unsigned FindInOperandList(SmallVectorImpl<ValueEntry> &Ops, unsigned i,
                                  Value *X) {
  unsigned XRank = Ops[i].Rank;
  unsigned e = Ops.size();
  for (unsigned j = i+1; j != e && Ops[j].Rank == XRank; ++j)
    if (Ops[j].Op == X)
      return j;
  // Scan backwards.
  for (unsigned j = i-1; j != ~0U && Ops[j].Rank == XRank; --j)
    if (Ops[j].Op == X)
      return j;
  return i;
}

/// EmitAddTreeOfValues - Emit a tree of add instructions, summing Ops together
/// and returning the result.  Insert the tree before I.
static Value *EmitAddTreeOfValues(Instruction *I,
                                  SmallVectorImpl<WeakVH> &Ops){
  if (Ops.size() == 1) return Ops.back();
  
  Value *V1 = Ops.back();
  Ops.pop_back();
  Value *V2 = EmitAddTreeOfValues(I, Ops);
  return BinaryOperator::CreateAdd(V2, V1, "tmp", I);
}

/// RemoveFactorFromExpression - If V is an expression tree that is a 
/// multiplication sequence, and if this sequence contains a multiply by Factor,
/// remove Factor from the tree and return the new tree.
Value *Reassociate::RemoveFactorFromExpression(Value *V, Value *Factor) {
  BinaryOperator *BO = isReassociableOp(V, Instruction::Mul);
  if (!BO) return 0;
  
  SmallVector<ValueEntry, 8> Factors;
  LinearizeExprTree(BO, Factors);

  bool FoundFactor = false;
  bool NeedsNegate = false;
  for (unsigned i = 0, e = Factors.size(); i != e; ++i) {
    if (Factors[i].Op == Factor) {
      FoundFactor = true;
      Factors.erase(Factors.begin()+i);
      break;
    }
    
    // If this is a negative version of this factor, remove it.
    if (ConstantInt *FC1 = dyn_cast<ConstantInt>(Factor))
      if (ConstantInt *FC2 = dyn_cast<ConstantInt>(Factors[i].Op))
        if (FC1->getValue() == -FC2->getValue()) {
          FoundFactor = NeedsNegate = true;
          Factors.erase(Factors.begin()+i);
          break;
        }
  }
  
  if (!FoundFactor) {
    // Make sure to restore the operands to the expression tree.
    RewriteExprTree(BO, Factors);
    return 0;
  }
  
  BasicBlock::iterator InsertPt = BO; ++InsertPt;
  
  // If this was just a single multiply, remove the multiply and return the only
  // remaining operand.
  if (Factors.size() == 1) {
    ValueRankMap.erase(BO);
    DeadInsts.push_back(BO);
    V = Factors[0].Op;
  } else {
    RewriteExprTree(BO, Factors);
    V = BO;
  }
  
  if (NeedsNegate)
    V = BinaryOperator::CreateNeg(V, "neg", InsertPt);
  
  return V;
}

/// FindSingleUseMultiplyFactors - If V is a single-use multiply, recursively
/// add its operands as factors, otherwise add V to the list of factors.
///
/// Ops is the top-level list of add operands we're trying to factor.
static void FindSingleUseMultiplyFactors(Value *V,
                                         SmallVectorImpl<Value*> &Factors,
                                       const SmallVectorImpl<ValueEntry> &Ops,
                                         bool IsRoot) {
  BinaryOperator *BO;
  if (!(V->hasOneUse() || V->use_empty()) || // More than one use.
      !(BO = dyn_cast<BinaryOperator>(V)) ||
      BO->getOpcode() != Instruction::Mul) {
    Factors.push_back(V);
    return;
  }
  
  // If this value has a single use because it is another input to the add
  // tree we're reassociating and we dropped its use, it actually has two
  // uses and we can't factor it.
  if (!IsRoot) {
    for (unsigned i = 0, e = Ops.size(); i != e; ++i)
      if (Ops[i].Op == V) {
        Factors.push_back(V);
        return;
      }
  }
  
  
  // Otherwise, add the LHS and RHS to the list of factors.
  FindSingleUseMultiplyFactors(BO->getOperand(1), Factors, Ops, false);
  FindSingleUseMultiplyFactors(BO->getOperand(0), Factors, Ops, false);
}

/// OptimizeAndOrXor - Optimize a series of operands to an 'and', 'or', or 'xor'
/// instruction.  This optimizes based on identities.  If it can be reduced to
/// a single Value, it is returned, otherwise the Ops list is mutated as
/// necessary.
static Value *OptimizeAndOrXor(unsigned Opcode,
                               SmallVectorImpl<ValueEntry> &Ops) {
  // Scan the operand lists looking for X and ~X pairs, along with X,X pairs.
  // If we find any, we can simplify the expression. X&~X == 0, X|~X == -1.
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    // First, check for X and ~X in the operand list.
    assert(i < Ops.size());
    if (BinaryOperator::isNot(Ops[i].Op)) {    // Cannot occur for ^.
      Value *X = BinaryOperator::getNotArgument(Ops[i].Op);
      unsigned FoundX = FindInOperandList(Ops, i, X);
      if (FoundX != i) {
        if (Opcode == Instruction::And)   // ...&X&~X = 0
          return Constant::getNullValue(X->getType());
        
        if (Opcode == Instruction::Or)    // ...|X|~X = -1
          return Constant::getAllOnesValue(X->getType());
      }
    }
    
    // Next, check for duplicate pairs of values, which we assume are next to
    // each other, due to our sorting criteria.
    assert(i < Ops.size());
    if (i+1 != Ops.size() && Ops[i+1].Op == Ops[i].Op) {
      if (Opcode == Instruction::And || Opcode == Instruction::Or) {
        // Drop duplicate values for And and Or.
        Ops.erase(Ops.begin()+i);
        --i; --e;
        ++NumAnnihil;
        continue;
      }
      
      // Drop pairs of values for Xor.
      assert(Opcode == Instruction::Xor);
      if (e == 2)
        return Constant::getNullValue(Ops[0].Op->getType());
      
      // Y ^ X^X -> Y
      Ops.erase(Ops.begin()+i, Ops.begin()+i+2);
      i -= 1; e -= 2;
      ++NumAnnihil;
    }
  }
  return 0;
}

/// OptimizeAdd - Optimize a series of operands to an 'add' instruction.  This
/// optimizes based on identities.  If it can be reduced to a single Value, it
/// is returned, otherwise the Ops list is mutated as necessary.
Value *Reassociate::OptimizeAdd(Instruction *I,
                                SmallVectorImpl<ValueEntry> &Ops) {
  // Scan the operand lists looking for X and -X pairs.  If we find any, we
  // can simplify the expression. X+-X == 0.  While we're at it, scan for any
  // duplicates.  We want to canonicalize Y+Y+Y+Z -> 3*Y+Z.
  //
  // TODO: We could handle "X + ~X" -> "-1" if we wanted, since "-X = ~X+1".
  //
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    Value *TheOp = Ops[i].Op;
    // Check to see if we've seen this operand before.  If so, we factor all
    // instances of the operand together.  Due to our sorting criteria, we know
    // that these need to be next to each other in the vector.
    if (i+1 != Ops.size() && Ops[i+1].Op == TheOp) {
      // Rescan the list, remove all instances of this operand from the expr.
      unsigned NumFound = 0;
      do {
        Ops.erase(Ops.begin()+i);
        ++NumFound;
      } while (i != Ops.size() && Ops[i].Op == TheOp);
      
      DEBUG(errs() << "\nFACTORING [" << NumFound << "]: " << *TheOp << '\n');
      ++NumFactor;
      
      // Insert a new multiply.
      Value *Mul = ConstantInt::get(cast<IntegerType>(I->getType()), NumFound);
      Mul = BinaryOperator::CreateMul(TheOp, Mul, "factor", I);
      
      // Now that we have inserted a multiply, optimize it. This allows us to
      // handle cases that require multiple factoring steps, such as this:
      // (X*2) + (X*2) + (X*2) -> (X*2)*3 -> X*6
      RedoInsts.push_back(Mul);
      
      // If every add operand was a duplicate, return the multiply.
      if (Ops.empty())
        return Mul;
      
      // Otherwise, we had some input that didn't have the dupe, such as
      // "A + A + B" -> "A*2 + B".  Add the new multiply to the list of
      // things being added by this operation.
      Ops.insert(Ops.begin(), ValueEntry(getRank(Mul), Mul));
      
      --i;
      e = Ops.size();
      continue;
    }
    
    // Check for X and -X in the operand list.
    if (!BinaryOperator::isNeg(TheOp))
      continue;
    
    Value *X = BinaryOperator::getNegArgument(TheOp);
    unsigned FoundX = FindInOperandList(Ops, i, X);
    if (FoundX == i)
      continue;
    
    // Remove X and -X from the operand list.
    if (Ops.size() == 2)
      return Constant::getNullValue(X->getType());
    
    Ops.erase(Ops.begin()+i);
    if (i < FoundX)
      --FoundX;
    else
      --i;   // Need to back up an extra one.
    Ops.erase(Ops.begin()+FoundX);
    ++NumAnnihil;
    --i;     // Revisit element.
    e -= 2;  // Removed two elements.
  }
  
  // Scan the operand list, checking to see if there are any common factors
  // between operands.  Consider something like A*A+A*B*C+D.  We would like to
  // reassociate this to A*(A+B*C)+D, which reduces the number of multiplies.
  // To efficiently find this, we count the number of times a factor occurs
  // for any ADD operands that are MULs.
  DenseMap<Value*, unsigned> FactorOccurrences;
  
  // Keep track of each multiply we see, to avoid triggering on (X*4)+(X*4)
  // where they are actually the same multiply.
  unsigned MaxOcc = 0;
  Value *MaxOccVal = 0;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    BinaryOperator *BOp = dyn_cast<BinaryOperator>(Ops[i].Op);
    if (BOp == 0 || BOp->getOpcode() != Instruction::Mul || !BOp->use_empty())
      continue;
    
    // Compute all of the factors of this added value.
    SmallVector<Value*, 8> Factors;
    FindSingleUseMultiplyFactors(BOp, Factors, Ops, true);
    assert(Factors.size() > 1 && "Bad linearize!");
    
    // Add one to FactorOccurrences for each unique factor in this op.
    SmallPtrSet<Value*, 8> Duplicates;
    for (unsigned i = 0, e = Factors.size(); i != e; ++i) {
      Value *Factor = Factors[i];
      if (!Duplicates.insert(Factor)) continue;
      
      unsigned Occ = ++FactorOccurrences[Factor];
      if (Occ > MaxOcc) { MaxOcc = Occ; MaxOccVal = Factor; }
      
      // If Factor is a negative constant, add the negated value as a factor
      // because we can percolate the negate out.  Watch for minint, which
      // cannot be positivified.
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Factor))
        if (CI->isNegative() && !CI->isMinValue(true)) {
          Factor = ConstantInt::get(CI->getContext(), -CI->getValue());
          assert(!Duplicates.count(Factor) &&
                 "Shouldn't have two constant factors, missed a canonicalize");
          
          unsigned Occ = ++FactorOccurrences[Factor];
          if (Occ > MaxOcc) { MaxOcc = Occ; MaxOccVal = Factor; }
        }
    }
  }
  
  // If any factor occurred more than one time, we can pull it out.
  if (MaxOcc > 1) {
    DEBUG(errs() << "\nFACTORING [" << MaxOcc << "]: " << *MaxOccVal << '\n');
    ++NumFactor;

    // Create a new instruction that uses the MaxOccVal twice.  If we don't do
    // this, we could otherwise run into situations where removing a factor
    // from an expression will drop a use of maxocc, and this can cause 
    // RemoveFactorFromExpression on successive values to behave differently.
    Instruction *DummyInst = BinaryOperator::CreateAdd(MaxOccVal, MaxOccVal);
    SmallVector<WeakVH, 4> NewMulOps;
    for (unsigned i = 0; i != Ops.size(); ++i) {
      // Only try to remove factors from expressions we're allowed to.
      BinaryOperator *BOp = dyn_cast<BinaryOperator>(Ops[i].Op);
      if (BOp == 0 || BOp->getOpcode() != Instruction::Mul || !BOp->use_empty())
        continue;
      
      if (Value *V = RemoveFactorFromExpression(Ops[i].Op, MaxOccVal)) {
        // The factorized operand may occur several times.  Convert them all in
        // one fell swoop.
        for (unsigned j = Ops.size(); j != i;) {
          --j;
          if (Ops[j].Op == Ops[i].Op) {
            NewMulOps.push_back(V);
            Ops.erase(Ops.begin()+j);
          }
        }
        --i;
      }
    }
    
    // No need for extra uses anymore.
    delete DummyInst;

    unsigned NumAddedValues = NewMulOps.size();
    Value *V = EmitAddTreeOfValues(I, NewMulOps);

    // Now that we have inserted the add tree, optimize it. This allows us to
    // handle cases that require multiple factoring steps, such as this:
    // A*A*B + A*A*C   -->   A*(A*B+A*C)   -->   A*(A*(B+C))
    assert(NumAddedValues > 1 && "Each occurrence should contribute a value");
    (void)NumAddedValues;
    V = ReassociateExpression(cast<BinaryOperator>(V));

    // Create the multiply.
    Value *V2 = BinaryOperator::CreateMul(V, MaxOccVal, "tmp", I);

    // Rerun associate on the multiply in case the inner expression turned into
    // a multiply.  We want to make sure that we keep things in canonical form.
    V2 = ReassociateExpression(cast<BinaryOperator>(V2));
    
    // If every add operand included the factor (e.g. "A*B + A*C"), then the
    // entire result expression is just the multiply "A*(B+C)".
    if (Ops.empty())
      return V2;
    
    // Otherwise, we had some input that didn't have the factor, such as
    // "A*B + A*C + D" -> "A*(B+C) + D".  Add the new multiply to the list of
    // things being added by this operation.
    Ops.insert(Ops.begin(), ValueEntry(getRank(V2), V2));
  }
  
  return 0;
}

Value *Reassociate::OptimizeExpression(BinaryOperator *I,
                                       SmallVectorImpl<ValueEntry> &Ops) {
  // Now that we have the linearized expression tree, try to optimize it.
  // Start by folding any constants that we found.
  bool IterateOptimization = false;
  if (Ops.size() == 1) return Ops[0].Op;

  unsigned Opcode = I->getOpcode();
  
  if (Constant *V1 = dyn_cast<Constant>(Ops[Ops.size()-2].Op))
    if (Constant *V2 = dyn_cast<Constant>(Ops.back().Op)) {
      Ops.pop_back();
      Ops.back().Op = ConstantExpr::get(Opcode, V1, V2);
      return OptimizeExpression(I, Ops);
    }

  // Check for destructive annihilation due to a constant being used.
  if (ConstantInt *CstVal = dyn_cast<ConstantInt>(Ops.back().Op))
    switch (Opcode) {
    default: break;
    case Instruction::And:
      if (CstVal->isZero())                  // X & 0 -> 0
        return CstVal;
      if (CstVal->isAllOnesValue())          // X & -1 -> X
        Ops.pop_back();
      break;
    case Instruction::Mul:
      if (CstVal->isZero()) {                // X * 0 -> 0
        ++NumAnnihil;
        return CstVal;
      }
        
      if (cast<ConstantInt>(CstVal)->isOne())
        Ops.pop_back();                      // X * 1 -> X
      break;
    case Instruction::Or:
      if (CstVal->isAllOnesValue())          // X | -1 -> -1
        return CstVal;
      // FALLTHROUGH!
    case Instruction::Add:
    case Instruction::Xor:
      if (CstVal->isZero())                  // X [|^+] 0 -> X
        Ops.pop_back();
      break;
    }
  if (Ops.size() == 1) return Ops[0].Op;

  // Handle destructive annihilation due to identities between elements in the
  // argument list here.
  switch (Opcode) {
  default: break;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    unsigned NumOps = Ops.size();
    if (Value *Result = OptimizeAndOrXor(Opcode, Ops))
      return Result;
    IterateOptimization |= Ops.size() != NumOps;
    break;
  }

  case Instruction::Add: {
    unsigned NumOps = Ops.size();
    if (Value *Result = OptimizeAdd(I, Ops))
      return Result;
    IterateOptimization |= Ops.size() != NumOps;
  }

    break;
  //case Instruction::Mul:
  }

  if (IterateOptimization)
    return OptimizeExpression(I, Ops);
  return 0;
}


/// ReassociateInst - Inspect and reassociate the instruction at the
/// given position, post-incrementing the position.
void Reassociate::ReassociateInst(BasicBlock::iterator &BBI) {
  Instruction *BI = BBI++;
  if (BI->getOpcode() == Instruction::Shl &&
      isa<ConstantInt>(BI->getOperand(1)))
    if (Instruction *NI = ConvertShiftToMul(BI, ValueRankMap)) {
      MadeChange = true;
      BI = NI;
    }

  // Reject cases where it is pointless to do this.
  if (!isa<BinaryOperator>(BI) || BI->getType()->isFloatingPointTy() || 
      BI->getType()->isVectorTy())
    return;  // Floating point ops are not associative.

  // Do not reassociate boolean (i1) expressions.  We want to preserve the
  // original order of evaluation for short-circuited comparisons that
  // SimplifyCFG has folded to AND/OR expressions.  If the expression
  // is not further optimized, it is likely to be transformed back to a
  // short-circuited form for code gen, and the source order may have been
  // optimized for the most likely conditions.
  if (BI->getType()->isIntegerTy(1))
    return;

  // If this is a subtract instruction which is not already in negate form,
  // see if we can convert it to X+-Y.
  if (BI->getOpcode() == Instruction::Sub) {
    if (ShouldBreakUpSubtract(BI)) {
      BI = BreakUpSubtract(BI, ValueRankMap);
      // Reset the BBI iterator in case BreakUpSubtract changed the
      // instruction it points to.
      BBI = BI;
      ++BBI;
      MadeChange = true;
    } else if (BinaryOperator::isNeg(BI)) {
      // Otherwise, this is a negation.  See if the operand is a multiply tree
      // and if this is not an inner node of a multiply tree.
      if (isReassociableOp(BI->getOperand(1), Instruction::Mul) &&
          (!BI->hasOneUse() ||
           !isReassociableOp(BI->use_back(), Instruction::Mul))) {
        BI = LowerNegateToMultiply(BI, ValueRankMap);
        MadeChange = true;
      }
    }
  }

  // If this instruction is a commutative binary operator, process it.
  if (!BI->isAssociative()) return;
  BinaryOperator *I = cast<BinaryOperator>(BI);

  // If this is an interior node of a reassociable tree, ignore it until we
  // get to the root of the tree, to avoid N^2 analysis.
  if (I->hasOneUse() && isReassociableOp(I->use_back(), I->getOpcode()))
    return;

  // If this is an add tree that is used by a sub instruction, ignore it 
  // until we process the subtract.
  if (I->hasOneUse() && I->getOpcode() == Instruction::Add &&
      cast<Instruction>(I->use_back())->getOpcode() == Instruction::Sub)
    return;

  ReassociateExpression(I);
}

Value *Reassociate::ReassociateExpression(BinaryOperator *I) {
  
  // First, walk the expression tree, linearizing the tree, collecting the
  // operand information.
  SmallVector<ValueEntry, 8> Ops;
  LinearizeExprTree(I, Ops);
  
  DEBUG(dbgs() << "RAIn:\t"; PrintOps(I, Ops); dbgs() << '\n');
  
  // Now that we have linearized the tree to a list and have gathered all of
  // the operands and their ranks, sort the operands by their rank.  Use a
  // stable_sort so that values with equal ranks will have their relative
  // positions maintained (and so the compiler is deterministic).  Note that
  // this sorts so that the highest ranking values end up at the beginning of
  // the vector.
  std::stable_sort(Ops.begin(), Ops.end());
  
  // OptimizeExpression - Now that we have the expression tree in a convenient
  // sorted form, optimize it globally if possible.
  if (Value *V = OptimizeExpression(I, Ops)) {
    // This expression tree simplified to something that isn't a tree,
    // eliminate it.
    DEBUG(dbgs() << "Reassoc to scalar: " << *V << '\n');
    I->replaceAllUsesWith(V);
    if (Instruction *VI = dyn_cast<Instruction>(V))
      VI->setDebugLoc(I->getDebugLoc());
    RemoveDeadBinaryOp(I);
    ++NumAnnihil;
    return V;
  }
  
  // We want to sink immediates as deeply as possible except in the case where
  // this is a multiply tree used only by an add, and the immediate is a -1.
  // In this case we reassociate to put the negation on the outside so that we
  // can fold the negation into the add: (-X)*Y + Z -> Z-X*Y
  if (I->getOpcode() == Instruction::Mul && I->hasOneUse() &&
      cast<Instruction>(I->use_back())->getOpcode() == Instruction::Add &&
      isa<ConstantInt>(Ops.back().Op) &&
      cast<ConstantInt>(Ops.back().Op)->isAllOnesValue()) {
    ValueEntry Tmp = Ops.pop_back_val();
    Ops.insert(Ops.begin(), Tmp);
  }
  
  DEBUG(dbgs() << "RAOut:\t"; PrintOps(I, Ops); dbgs() << '\n');
  
  if (Ops.size() == 1) {
    // This expression tree simplified to something that isn't a tree,
    // eliminate it.
    I->replaceAllUsesWith(Ops[0].Op);
    if (Instruction *OI = dyn_cast<Instruction>(Ops[0].Op))
      OI->setDebugLoc(I->getDebugLoc());
    RemoveDeadBinaryOp(I);
    return Ops[0].Op;
  }
  
  // Now that we ordered and optimized the expressions, splat them back into
  // the expression tree, removing any unneeded nodes.
  RewriteExprTree(I, Ops);
  return I;
}


bool Reassociate::runOnFunction(Function &F) {
  // Recalculate the rank map for F
  BuildRankMap(F);

  MadeChange = false;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
    for (BasicBlock::iterator BBI = FI->begin(); BBI != FI->end(); )
      ReassociateInst(BBI);

  // Now that we're done, revisit any instructions which are likely to
  // have secondary reassociation opportunities.
  while (!RedoInsts.empty())
    if (Value *V = RedoInsts.pop_back_val()) {
      BasicBlock::iterator BBI = cast<Instruction>(V);
      ReassociateInst(BBI);
    }

  // Now that we're done, delete any instructions which are no longer used.
  while (!DeadInsts.empty())
    if (Value *V = DeadInsts.pop_back_val())
      RecursivelyDeleteTriviallyDeadInstructions(V);

  // We are done with the rank map.
  RankMap.clear();
  ValueRankMap.clear();
  return MadeChange;
}

