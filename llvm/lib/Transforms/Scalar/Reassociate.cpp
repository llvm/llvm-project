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
// to promote better constant propagation, GCSE, LICM, PRE...
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
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include <algorithm>
#include <map>
using namespace llvm;

STATISTIC(NumLinear , "Number of insts linearized");
STATISTIC(NumChanged, "Number of insts reassociated");
STATISTIC(NumAnnihil, "Number of expr tree annihilated");
STATISTIC(NumFactor , "Number of multiplies factored");

namespace {
  struct VISIBILITY_HIDDEN ValueEntry {
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
static void PrintOps(Instruction *I, const std::vector<ValueEntry> &Ops) {
  Module *M = I->getParent()->getParent()->getParent();
  cerr << Instruction::getOpcodeName(I->getOpcode()) << " "
       << *Ops[0].Op->getType();
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    WriteAsOperand(*cerr.stream() << " ", Ops[i].Op, false, M);
    cerr << "," << Ops[i].Rank;
  }
}
#endif
  
namespace {
  class VISIBILITY_HIDDEN Reassociate : public FunctionPass {
    std::map<BasicBlock*, unsigned> RankMap;
    std::map<Value*, unsigned> ValueRankMap;
    bool MadeChange;
  public:
    static char ID; // Pass identification, replacement for typeid
    Reassociate() : FunctionPass(&ID) {}

    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
  private:
    void BuildRankMap(Function &F);
    unsigned getRank(Value *V);
    void ReassociateExpression(BinaryOperator *I);
    void RewriteExprTree(BinaryOperator *I, std::vector<ValueEntry> &Ops,
                         unsigned Idx = 0);
    Value *OptimizeExpression(BinaryOperator *I, std::vector<ValueEntry> &Ops);
    void LinearizeExprTree(BinaryOperator *I, std::vector<ValueEntry> &Ops);
    void LinearizeExpr(BinaryOperator *I);
    Value *RemoveFactorFromExpression(Value *V, Value *Factor);
    void ReassociateBB(BasicBlock *BB);
    
    void RemoveDeadBinaryOp(Value *V);
  };
}

char Reassociate::ID = 0;
static RegisterPass<Reassociate> X("reassociate", "Reassociate expressions");

// Public interface to the Reassociate pass
FunctionPass *llvm::createReassociatePass() { return new Reassociate(); }

void Reassociate::RemoveDeadBinaryOp(Value *V) {
  Instruction *Op = dyn_cast<Instruction>(V);
  if (!Op || !isa<BinaryOperator>(Op) || !isa<CmpInst>(Op) || !Op->use_empty())
    return;
  
  Value *LHS = Op->getOperand(0), *RHS = Op->getOperand(1);
  RemoveDeadBinaryOp(LHS);
  RemoveDeadBinaryOp(RHS);
}


static bool isUnmovableInstruction(Instruction *I) {
  if (I->getOpcode() == Instruction::PHI ||
      I->getOpcode() == Instruction::Alloca ||
      I->getOpcode() == Instruction::Load ||
      I->getOpcode() == Instruction::Malloc ||
      I->getOpcode() == Instruction::Invoke ||
      I->getOpcode() == Instruction::Call ||
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
    ValueRankMap[I] = ++i;

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
        ValueRankMap[I] = ++BBRank;
  }
}

unsigned Reassociate::getRank(Value *V) {
  if (isa<Argument>(V)) return ValueRankMap[V];   // Function argument...

  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) return 0;  // Otherwise it's a global or constant, rank 0.

  unsigned &CachedRank = ValueRankMap[I];
  if (CachedRank) return CachedRank;    // Rank already known?

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
  if (!I->getType()->isInteger() ||
      (!BinaryOperator::isNot(I) && !BinaryOperator::isNeg(I)))
    ++Rank;

  //DOUT << "Calculated Rank[" << V->getName() << "] = "
  //     << Rank << "\n";

  return CachedRank = Rank;
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
static Instruction *LowerNegateToMultiply(Instruction *Neg) {
  Constant *Cst = ConstantInt::getAllOnesValue(Neg->getType());

  Instruction *Res = BinaryOperator::CreateMul(Neg->getOperand(1), Cst, "",Neg);
  Res->takeName(Neg);
  Neg->replaceAllUsesWith(Res);
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

  DOUT << "Linear" << *LHS << *RHS << *I;

  // Move the RHS instruction to live immediately before I, avoiding breaking
  // dominator properties.
  RHS->moveBefore(I);

  // Move operands around to do the linearization.
  I->setOperand(1, RHS->getOperand(0));
  RHS->setOperand(0, LHS);
  I->setOperand(0, RHS);

  ++NumLinear;
  MadeChange = true;
  DOUT << "Linearized: " << *I;

  // If D is part of this expression tree, tail recurse.
  if (isReassociableOp(I->getOperand(1), I->getOpcode()))
    LinearizeExpr(I);
}


/// LinearizeExprTree - Given an associative binary expression tree, traverse
/// all of the uses putting it into canonical form.  This forces a left-linear
/// form of the the expression (((a+b)+c)+d), and collects information about the
/// rank of the non-tree operands.
///
/// NOTE: These intentionally destroys the expression tree operands (turning
/// them into undef values) to reduce #uses of the values.  This means that the
/// caller MUST use something like RewriteExprTree to put the values back in.
///
void Reassociate::LinearizeExprTree(BinaryOperator *I,
                                    std::vector<ValueEntry> &Ops) {
  Value *LHS = I->getOperand(0), *RHS = I->getOperand(1);
  unsigned Opcode = I->getOpcode();

  // First step, linearize the expression if it is in ((A+B)+(C+D)) form.
  BinaryOperator *LHSBO = isReassociableOp(LHS, Opcode);
  BinaryOperator *RHSBO = isReassociableOp(RHS, Opcode);

  // If this is a multiply expression tree and it contains internal negations,
  // transform them into multiplies by -1 so they can be reassociated.
  if (I->getOpcode() == Instruction::Mul) {
    if (!LHSBO && LHS->hasOneUse() && BinaryOperator::isNeg(LHS)) {
      LHS = LowerNegateToMultiply(cast<Instruction>(LHS));
      LHSBO = isReassociableOp(LHS, Opcode);
    }
    if (!RHSBO && RHS->hasOneUse() && BinaryOperator::isNeg(RHS)) {
      RHS = LowerNegateToMultiply(cast<Instruction>(RHS));
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
    } else {
      // Turn X+(Y+Z) -> (Y+Z)+X
      std::swap(LHSBO, RHSBO);
      std::swap(LHS, RHS);
      bool Success = !I->swapOperands();
      assert(Success && "swapOperands failed");
      Success = false;
      MadeChange = true;
    }
  } else if (RHSBO) {
    // Turn (A+B)+(C+D) -> (((A+B)+C)+D).  This guarantees the the RHS is not
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
                                  std::vector<ValueEntry> &Ops,
                                  unsigned i) {
  if (i+2 == Ops.size()) {
    if (I->getOperand(0) != Ops[i].Op ||
        I->getOperand(1) != Ops[i+1].Op) {
      Value *OldLHS = I->getOperand(0);
      DOUT << "RA: " << *I;
      I->setOperand(0, Ops[i].Op);
      I->setOperand(1, Ops[i+1].Op);
      DOUT << "TO: " << *I;
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
    DOUT << "RA: " << *I;
    I->setOperand(1, Ops[i].Op);
    DOUT << "TO: " << *I;
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
  // We are trying to expose opportunity for reassociation.  One of the things
  // that we want to do to achieve this is to push a negation as deep into an
  // expression chain as possible, to expose the add instructions.  In practice,
  // this means that we turn this:
  //   X = -(A+12+C+D)   into    X = -A + -12 + -C + -D = -12 + -A + -C + -D
  // so that later, a: Y = 12+X could get reassociated with the -12 to eliminate
  // the constants.  We assume that instcombine will clean up the mess later if
  // we introduce tons of unnecessary negation instructions...
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

  // Insert a 'neg' instruction that subtracts the value from zero to get the
  // negation.
  //
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
static Instruction *BreakUpSubtract(Instruction *Sub) {
  // Convert a subtract into an add and a neg instruction... so that sub
  // instructions can be commuted with other add instructions...
  //
  // Calculate the negative value of Operand 1 of the sub instruction...
  // and set it as the RHS of the add instruction we just made...
  //
  Value *NegVal = NegateValue(Sub->getOperand(1), Sub);
  Instruction *New =
    BinaryOperator::CreateAdd(Sub->getOperand(0), NegVal, "", Sub);
  New->takeName(Sub);

  // Everyone now refers to the add instruction.
  Sub->replaceAllUsesWith(New);
  Sub->eraseFromParent();

  DOUT << "Negated: " << *New;
  return New;
}

/// ConvertShiftToMul - If this is a shift of a reassociable multiply or is used
/// by one, change this into a multiply by a constant to assist with further
/// reassociation.
static Instruction *ConvertShiftToMul(Instruction *Shl) {
  // If an operand of this shift is a reassociable multiply, or if the shift
  // is used by a reassociable multiply or add, turn into a multiply.
  if (isReassociableOp(Shl->getOperand(0), Instruction::Mul) ||
      (Shl->hasOneUse() && 
       (isReassociableOp(Shl->use_back(), Instruction::Mul) ||
        isReassociableOp(Shl->use_back(), Instruction::Add)))) {
    Constant *MulCst = ConstantInt::get(Shl->getType(), 1);
    MulCst = ConstantExpr::getShl(MulCst, cast<Constant>(Shl->getOperand(1)));
    
    Instruction *Mul = BinaryOperator::CreateMul(Shl->getOperand(0), MulCst,
                                                 "", Shl);
    Mul->takeName(Shl);
    Shl->replaceAllUsesWith(Mul);
    Shl->eraseFromParent();
    return Mul;
  }
  return 0;
}

// Scan backwards and forwards among values with the same rank as element i to
// see if X exists.  If X does not exist, return i.
static unsigned FindInOperandList(std::vector<ValueEntry> &Ops, unsigned i,
                                  Value *X) {
  unsigned XRank = Ops[i].Rank;
  unsigned e = Ops.size();
  for (unsigned j = i+1; j != e && Ops[j].Rank == XRank; ++j)
    if (Ops[j].Op == X)
      return j;
  // Scan backwards
  for (unsigned j = i-1; j != ~0U && Ops[j].Rank == XRank; --j)
    if (Ops[j].Op == X)
      return j;
  return i;
}

/// EmitAddTreeOfValues - Emit a tree of add instructions, summing Ops together
/// and returning the result.  Insert the tree before I.
static Value *EmitAddTreeOfValues(Instruction *I, std::vector<Value*> &Ops) {
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
  
  std::vector<ValueEntry> Factors;
  LinearizeExprTree(BO, Factors);

  bool FoundFactor = false;
  for (unsigned i = 0, e = Factors.size(); i != e; ++i)
    if (Factors[i].Op == Factor) {
      FoundFactor = true;
      Factors.erase(Factors.begin()+i);
      break;
    }
  if (!FoundFactor) {
    // Make sure to restore the operands to the expression tree.
    RewriteExprTree(BO, Factors);
    return 0;
  }
  
  if (Factors.size() == 1) return Factors[0].Op;
  
  RewriteExprTree(BO, Factors);
  return BO;
}

/// FindSingleUseMultiplyFactors - If V is a single-use multiply, recursively
/// add its operands as factors, otherwise add V to the list of factors.
static void FindSingleUseMultiplyFactors(Value *V,
                                         std::vector<Value*> &Factors) {
  BinaryOperator *BO;
  if ((!V->hasOneUse() && !V->use_empty()) ||
      !(BO = dyn_cast<BinaryOperator>(V)) ||
      BO->getOpcode() != Instruction::Mul) {
    Factors.push_back(V);
    return;
  }
  
  // Otherwise, add the LHS and RHS to the list of factors.
  FindSingleUseMultiplyFactors(BO->getOperand(1), Factors);
  FindSingleUseMultiplyFactors(BO->getOperand(0), Factors);
}



Value *Reassociate::OptimizeExpression(BinaryOperator *I,
                                       std::vector<ValueEntry> &Ops) {
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
      if (CstVal->isZero()) {                // ... & 0 -> 0
        ++NumAnnihil;
        return CstVal;
      } else if (CstVal->isAllOnesValue()) { // ... & -1 -> ...
        Ops.pop_back();
      }
      break;
    case Instruction::Mul:
      if (CstVal->isZero()) {                // ... * 0 -> 0
        ++NumAnnihil;
        return CstVal;
      } else if (cast<ConstantInt>(CstVal)->isOne()) {
        Ops.pop_back();                      // ... * 1 -> ...
      }
      break;
    case Instruction::Or:
      if (CstVal->isAllOnesValue()) {        // ... | -1 -> -1
        ++NumAnnihil;
        return CstVal;
      }
      // FALLTHROUGH!
    case Instruction::Add:
    case Instruction::Xor:
      if (CstVal->isZero())                  // ... [|^+] 0 -> ...
        Ops.pop_back();
      break;
    }
  if (Ops.size() == 1) return Ops[0].Op;

  // Handle destructive annihilation do to identities between elements in the
  // argument list here.
  switch (Opcode) {
  default: break;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    // Scan the operand lists looking for X and ~X pairs, along with X,X pairs.
    // If we find any, we can simplify the expression. X&~X == 0, X|~X == -1.
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      // First, check for X and ~X in the operand list.
      assert(i < Ops.size());
      if (BinaryOperator::isNot(Ops[i].Op)) {    // Cannot occur for ^.
        Value *X = BinaryOperator::getNotArgument(Ops[i].Op);
        unsigned FoundX = FindInOperandList(Ops, i, X);
        if (FoundX != i) {
          if (Opcode == Instruction::And) {   // ...&X&~X = 0
            ++NumAnnihil;
            return Constant::getNullValue(X->getType());
          } else if (Opcode == Instruction::Or) {   // ...|X|~X = -1
            ++NumAnnihil;
            return ConstantInt::getAllOnesValue(X->getType());
          }
        }
      }

      // Next, check for duplicate pairs of values, which we assume are next to
      // each other, due to our sorting criteria.
      assert(i < Ops.size());
      if (i+1 != Ops.size() && Ops[i+1].Op == Ops[i].Op) {
        if (Opcode == Instruction::And || Opcode == Instruction::Or) {
          // Drop duplicate values.
          Ops.erase(Ops.begin()+i);
          --i; --e;
          IterateOptimization = true;
          ++NumAnnihil;
        } else {
          assert(Opcode == Instruction::Xor);
          if (e == 2) {
            ++NumAnnihil;
            return Constant::getNullValue(Ops[0].Op->getType());
          }
          // ... X^X -> ...
          Ops.erase(Ops.begin()+i, Ops.begin()+i+2);
          i -= 1; e -= 2;
          IterateOptimization = true;
          ++NumAnnihil;
        }
      }
    }
    break;

  case Instruction::Add:
    // Scan the operand lists looking for X and -X pairs.  If we find any, we
    // can simplify the expression. X+-X == 0.
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      assert(i < Ops.size());
      // Check for X and -X in the operand list.
      if (BinaryOperator::isNeg(Ops[i].Op)) {
        Value *X = BinaryOperator::getNegArgument(Ops[i].Op);
        unsigned FoundX = FindInOperandList(Ops, i, X);
        if (FoundX != i) {
          // Remove X and -X from the operand list.
          if (Ops.size() == 2) {
            ++NumAnnihil;
            return Constant::getNullValue(X->getType());
          } else {
            Ops.erase(Ops.begin()+i);
            if (i < FoundX)
              --FoundX;
            else
              --i;   // Need to back up an extra one.
            Ops.erase(Ops.begin()+FoundX);
            IterateOptimization = true;
            ++NumAnnihil;
            --i;     // Revisit element.
            e -= 2;  // Removed two elements.
          }
        }
      }
    }
    

    // Scan the operand list, checking to see if there are any common factors
    // between operands.  Consider something like A*A+A*B*C+D.  We would like to
    // reassociate this to A*(A+B*C)+D, which reduces the number of multiplies.
    // To efficiently find this, we count the number of times a factor occurs
    // for any ADD operands that are MULs.
    std::map<Value*, unsigned> FactorOccurrences;
    unsigned MaxOcc = 0;
    Value *MaxOccVal = 0;
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      if (BinaryOperator *BOp = dyn_cast<BinaryOperator>(Ops[i].Op)) {
        if (BOp->getOpcode() == Instruction::Mul && BOp->use_empty()) {
          // Compute all of the factors of this added value.
          std::vector<Value*> Factors;
          FindSingleUseMultiplyFactors(BOp, Factors);
          assert(Factors.size() > 1 && "Bad linearize!");

          // Add one to FactorOccurrences for each unique factor in this op.
          if (Factors.size() == 2) {
            unsigned Occ = ++FactorOccurrences[Factors[0]];
            if (Occ > MaxOcc) { MaxOcc = Occ; MaxOccVal = Factors[0]; }
            if (Factors[0] != Factors[1]) {   // Don't double count A*A.
              Occ = ++FactorOccurrences[Factors[1]];
              if (Occ > MaxOcc) { MaxOcc = Occ; MaxOccVal = Factors[1]; }
            }
          } else {
            std::set<Value*> Duplicates;
            for (unsigned i = 0, e = Factors.size(); i != e; ++i) {
              if (Duplicates.insert(Factors[i]).second) {
                unsigned Occ = ++FactorOccurrences[Factors[i]];
                if (Occ > MaxOcc) { MaxOcc = Occ; MaxOccVal = Factors[i]; }
              }
            }
          }
        }
      }
    }

    // If any factor occurred more than one time, we can pull it out.
    if (MaxOcc > 1) {
      DOUT << "\nFACTORING [" << MaxOcc << "]: " << *MaxOccVal << "\n";
      
      // Create a new instruction that uses the MaxOccVal twice.  If we don't do
      // this, we could otherwise run into situations where removing a factor
      // from an expression will drop a use of maxocc, and this can cause 
      // RemoveFactorFromExpression on successive values to behave differently.
      Instruction *DummyInst = BinaryOperator::CreateAdd(MaxOccVal, MaxOccVal);
      std::vector<Value*> NewMulOps;
      for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
        if (Value *V = RemoveFactorFromExpression(Ops[i].Op, MaxOccVal)) {
          NewMulOps.push_back(V);
          Ops.erase(Ops.begin()+i);
          --i; --e;
        }
      }
      
      // No need for extra uses anymore.
      delete DummyInst;

      unsigned NumAddedValues = NewMulOps.size();
      Value *V = EmitAddTreeOfValues(I, NewMulOps);
      Value *V2 = BinaryOperator::CreateMul(V, MaxOccVal, "tmp", I);

      // Now that we have inserted V and its sole use, optimize it. This allows
      // us to handle cases that require multiple factoring steps, such as this:
      // A*A*B + A*A*C   -->   A*(A*B+A*C)   -->   A*(A*(B+C))
      if (NumAddedValues > 1)
        ReassociateExpression(cast<BinaryOperator>(V));
      
      ++NumFactor;
      
      if (Ops.empty())
        return V2;

      // Add the new value to the list of things being added.
      Ops.insert(Ops.begin(), ValueEntry(getRank(V2), V2));
      
      // Rewrite the tree so that there is now a use of V.
      RewriteExprTree(I, Ops);
      return OptimizeExpression(I, Ops);
    }
    break;
  //case Instruction::Mul:
  }

  if (IterateOptimization)
    return OptimizeExpression(I, Ops);
  return 0;
}


/// ReassociateBB - Inspect all of the instructions in this basic block,
/// reassociating them as we go.
void Reassociate::ReassociateBB(BasicBlock *BB) {
  for (BasicBlock::iterator BBI = BB->begin(); BBI != BB->end(); ) {
    Instruction *BI = BBI++;
    if (BI->getOpcode() == Instruction::Shl &&
        isa<ConstantInt>(BI->getOperand(1)))
      if (Instruction *NI = ConvertShiftToMul(BI)) {
        MadeChange = true;
        BI = NI;
      }

    // Reject cases where it is pointless to do this.
    if (!isa<BinaryOperator>(BI) || BI->getType()->isFloatingPoint() || 
        isa<VectorType>(BI->getType()))
      continue;  // Floating point ops are not associative.

    // If this is a subtract instruction which is not already in negate form,
    // see if we can convert it to X+-Y.
    if (BI->getOpcode() == Instruction::Sub) {
      if (ShouldBreakUpSubtract(BI)) {
        BI = BreakUpSubtract(BI);
        MadeChange = true;
      } else if (BinaryOperator::isNeg(BI)) {
        // Otherwise, this is a negation.  See if the operand is a multiply tree
        // and if this is not an inner node of a multiply tree.
        if (isReassociableOp(BI->getOperand(1), Instruction::Mul) &&
            (!BI->hasOneUse() ||
             !isReassociableOp(BI->use_back(), Instruction::Mul))) {
          BI = LowerNegateToMultiply(BI);
          MadeChange = true;
        }
      }
    }

    // If this instruction is a commutative binary operator, process it.
    if (!BI->isAssociative()) continue;
    BinaryOperator *I = cast<BinaryOperator>(BI);

    // If this is an interior node of a reassociable tree, ignore it until we
    // get to the root of the tree, to avoid N^2 analysis.
    if (I->hasOneUse() && isReassociableOp(I->use_back(), I->getOpcode()))
      continue;

    // If this is an add tree that is used by a sub instruction, ignore it 
    // until we process the subtract.
    if (I->hasOneUse() && I->getOpcode() == Instruction::Add &&
        cast<Instruction>(I->use_back())->getOpcode() == Instruction::Sub)
      continue;

    ReassociateExpression(I);
  }
}

void Reassociate::ReassociateExpression(BinaryOperator *I) {
  
  // First, walk the expression tree, linearizing the tree, collecting
  std::vector<ValueEntry> Ops;
  LinearizeExprTree(I, Ops);
  
  DOUT << "RAIn:\t"; DEBUG(PrintOps(I, Ops)); DOUT << "\n";
  
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
    DOUT << "Reassoc to scalar: " << *V << "\n";
    I->replaceAllUsesWith(V);
    RemoveDeadBinaryOp(I);
    return;
  }
  
  // We want to sink immediates as deeply as possible except in the case where
  // this is a multiply tree used only by an add, and the immediate is a -1.
  // In this case we reassociate to put the negation on the outside so that we
  // can fold the negation into the add: (-X)*Y + Z -> Z-X*Y
  if (I->getOpcode() == Instruction::Mul && I->hasOneUse() &&
      cast<Instruction>(I->use_back())->getOpcode() == Instruction::Add &&
      isa<ConstantInt>(Ops.back().Op) &&
      cast<ConstantInt>(Ops.back().Op)->isAllOnesValue()) {
    Ops.insert(Ops.begin(), Ops.back());
    Ops.pop_back();
  }
  
  DOUT << "RAOut:\t"; DEBUG(PrintOps(I, Ops)); DOUT << "\n";
  
  if (Ops.size() == 1) {
    // This expression tree simplified to something that isn't a tree,
    // eliminate it.
    I->replaceAllUsesWith(Ops[0].Op);
    RemoveDeadBinaryOp(I);
  } else {
    // Now that we ordered and optimized the expressions, splat them back into
    // the expression tree, removing any unneeded nodes.
    RewriteExprTree(I, Ops);
  }
}


bool Reassociate::runOnFunction(Function &F) {
  // Recalculate the rank map for F
  BuildRankMap(F);

  MadeChange = false;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
    ReassociateBB(FI);

  // We are done with the rank map...
  RankMap.clear();
  ValueRankMap.clear();
  return MadeChange;
}

