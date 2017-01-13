//===---- NewGVN.cpp - Global Value Numbering Pass --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the new LLVM's Global Value Numbering pass.
/// GVN partitions values computed by a function into congruence classes.
/// Values ending up in the same congruence class are guaranteed to be the same
/// for every execution of the program. In that respect, congruency is a
/// compile-time approximation of equivalence of values at runtime.
/// The algorithm implemented here uses a sparse formulation and it's based
/// on the ideas described in the paper:
/// "A Sparse Algorithm for Predicated Global Value Numbering" from
/// Karthik Gargi.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/NewGVN.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/PredIteratorCache.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVNExpression.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/MemorySSA.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <unordered_map>
#include <utility>
#include <vector>
using namespace llvm;
using namespace PatternMatch;
using namespace llvm::GVNExpression;

#define DEBUG_TYPE "newgvn"

STATISTIC(NumGVNInstrDeleted, "Number of instructions deleted");
STATISTIC(NumGVNBlocksDeleted, "Number of blocks deleted");
STATISTIC(NumGVNOpsSimplified, "Number of Expressions simplified");
STATISTIC(NumGVNPhisAllSame, "Number of PHIs whos arguments are all the same");
STATISTIC(NumGVNMaxIterations,
          "Maximum Number of iterations it took to converge GVN");

//===----------------------------------------------------------------------===//
//                                GVN Pass
//===----------------------------------------------------------------------===//

// Anchor methods.
namespace llvm {
namespace GVNExpression {
Expression::~Expression() = default;
BasicExpression::~BasicExpression() = default;
CallExpression::~CallExpression() = default;
LoadExpression::~LoadExpression() = default;
StoreExpression::~StoreExpression() = default;
AggregateValueExpression::~AggregateValueExpression() = default;
PHIExpression::~PHIExpression() = default;
}
}

// Congruence classes represent the set of expressions/instructions
// that are all the same *during some scope in the function*.
// That is, because of the way we perform equality propagation, and
// because of memory value numbering, it is not correct to assume
// you can willy-nilly replace any member with any other at any
// point in the function.
//
// For any Value in the Member set, it is valid to replace any dominated member
// with that Value.
//
// Every congruence class has a leader, and the leader is used to
// symbolize instructions in a canonical way (IE every operand of an
// instruction that is a member of the same congruence class will
// always be replaced with leader during symbolization).
// To simplify symbolization, we keep the leader as a constant if class can be
// proved to be a constant value.
// Otherwise, the leader is a randomly chosen member of the value set, it does
// not matter which one is chosen.
// Each congruence class also has a defining expression,
// though the expression may be null.  If it exists, it can be used for forward
// propagation and reassociation of values.
//
struct CongruenceClass {
  using MemberSet = SmallPtrSet<Value *, 4>;
  unsigned ID;
  // Representative leader.
  Value *RepLeader = nullptr;
  // Defining Expression.
  const Expression *DefiningExpr = nullptr;
  // Actual members of this class.
  MemberSet Members;

  // True if this class has no members left.  This is mainly used for assertion
  // purposes, and for skipping empty classes.
  bool Dead = false;

  // Number of stores in this congruence class.
  // This is used so we can detect store equivalence changes properly.
  int StoreCount = 0;

  explicit CongruenceClass(unsigned ID) : ID(ID) {}
  CongruenceClass(unsigned ID, Value *Leader, const Expression *E)
      : ID(ID), RepLeader(Leader), DefiningExpr(E) {}
};

namespace llvm {
template <> struct DenseMapInfo<const Expression *> {
  static const Expression *getEmptyKey() {
    auto Val = static_cast<uintptr_t>(-1);
    Val <<= PointerLikeTypeTraits<const Expression *>::NumLowBitsAvailable;
    return reinterpret_cast<const Expression *>(Val);
  }
  static const Expression *getTombstoneKey() {
    auto Val = static_cast<uintptr_t>(~1U);
    Val <<= PointerLikeTypeTraits<const Expression *>::NumLowBitsAvailable;
    return reinterpret_cast<const Expression *>(Val);
  }
  static unsigned getHashValue(const Expression *V) {
    return static_cast<unsigned>(V->getHashValue());
  }
  static bool isEqual(const Expression *LHS, const Expression *RHS) {
    if (LHS == RHS)
      return true;
    if (LHS == getTombstoneKey() || RHS == getTombstoneKey() ||
        LHS == getEmptyKey() || RHS == getEmptyKey())
      return false;
    return *LHS == *RHS;
  }
};
} // end namespace llvm

class NewGVN : public FunctionPass {
  DominatorTree *DT;
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;
  AssumptionCache *AC;
  AliasAnalysis *AA;
  MemorySSA *MSSA;
  MemorySSAWalker *MSSAWalker;
  BumpPtrAllocator ExpressionAllocator;
  ArrayRecycler<Value *> ArgRecycler;

  // Congruence class info.
  CongruenceClass *InitialClass;
  std::vector<CongruenceClass *> CongruenceClasses;
  unsigned NextCongruenceNum;

  // Value Mappings.
  DenseMap<Value *, CongruenceClass *> ValueToClass;
  DenseMap<Value *, const Expression *> ValueToExpression;

  // A table storing which memorydefs/phis represent a memory state provably
  // equivalent to another memory state.
  // We could use the congruence class machinery, but the MemoryAccess's are
  // abstract memory states, so they can only ever be equivalent to each other,
  // and not to constants, etc.
  DenseMap<const MemoryAccess *, MemoryAccess *> MemoryAccessEquiv;

  // Expression to class mapping.
  using ExpressionClassMap = DenseMap<const Expression *, CongruenceClass *>;
  ExpressionClassMap ExpressionToClass;

  // Which values have changed as a result of leader changes.
  SmallPtrSet<Value *, 8> LeaderChanges;

  // Reachability info.
  using BlockEdge = BasicBlockEdge;
  DenseSet<BlockEdge> ReachableEdges;
  SmallPtrSet<const BasicBlock *, 8> ReachableBlocks;

  // This is a bitvector because, on larger functions, we may have
  // thousands of touched instructions at once (entire blocks,
  // instructions with hundreds of uses, etc).  Even with optimization
  // for when we mark whole blocks as touched, when this was a
  // SmallPtrSet or DenseSet, for some functions, we spent >20% of all
  // the time in GVN just managing this list.  The bitvector, on the
  // other hand, efficiently supports test/set/clear of both
  // individual and ranges, as well as "find next element" This
  // enables us to use it as a worklist with essentially 0 cost.
  BitVector TouchedInstructions;

  DenseMap<const BasicBlock *, std::pair<unsigned, unsigned>> BlockInstRange;
  DenseMap<const DomTreeNode *, std::pair<unsigned, unsigned>>
      DominatedInstRange;

#ifndef NDEBUG
  // Debugging for how many times each block and instruction got processed.
  DenseMap<const Value *, unsigned> ProcessedCount;
#endif

  // DFS info.
  DenseMap<const BasicBlock *, std::pair<int, int>> DFSDomMap;
  DenseMap<const Value *, unsigned> InstrDFS;
  SmallVector<Value *, 32> DFSToInstr;

  // Deletion info.
  SmallPtrSet<Instruction *, 8> InstructionsToErase;

public:
  static char ID; // Pass identification, replacement for typeid.
  NewGVN() : FunctionPass(ID) {
    initializeNewGVNPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;
  bool runGVN(Function &F, DominatorTree *DT, AssumptionCache *AC,
              TargetLibraryInfo *TLI, AliasAnalysis *AA, MemorySSA *MSSA);

private:
  // This transformation requires dominator postdominator info.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<MemorySSAWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();

    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }

  // Expression handling.
  const Expression *createExpression(Instruction *, const BasicBlock *);
  const Expression *createBinaryExpression(unsigned, Type *, Value *, Value *,
                                           const BasicBlock *);
  PHIExpression *createPHIExpression(Instruction *);
  const VariableExpression *createVariableExpression(Value *);
  const ConstantExpression *createConstantExpression(Constant *);
  const Expression *createVariableOrConstant(Value *V, const BasicBlock *B);
  const UnknownExpression *createUnknownExpression(Instruction *);
  const StoreExpression *createStoreExpression(StoreInst *, MemoryAccess *,
                                               const BasicBlock *);
  LoadExpression *createLoadExpression(Type *, Value *, LoadInst *,
                                       MemoryAccess *, const BasicBlock *);

  const CallExpression *createCallExpression(CallInst *, MemoryAccess *,
                                             const BasicBlock *);
  const AggregateValueExpression *
  createAggregateValueExpression(Instruction *, const BasicBlock *);
  bool setBasicExpressionInfo(Instruction *, BasicExpression *,
                              const BasicBlock *);

  // Congruence class handling.
  CongruenceClass *createCongruenceClass(Value *Leader, const Expression *E) {
    auto *result = new CongruenceClass(NextCongruenceNum++, Leader, E);
    CongruenceClasses.emplace_back(result);
    return result;
  }

  CongruenceClass *createSingletonCongruenceClass(Value *Member) {
    CongruenceClass *CClass = createCongruenceClass(Member, nullptr);
    CClass->Members.insert(Member);
    ValueToClass[Member] = CClass;
    return CClass;
  }
  void initializeCongruenceClasses(Function &F);

  // Value number an Instruction or MemoryPhi.
  void valueNumberMemoryPhi(MemoryPhi *);
  void valueNumberInstruction(Instruction *);

  // Symbolic evaluation.
  const Expression *checkSimplificationResults(Expression *, Instruction *,
                                               Value *);
  const Expression *performSymbolicEvaluation(Value *, const BasicBlock *);
  const Expression *performSymbolicLoadEvaluation(Instruction *,
                                                  const BasicBlock *);
  const Expression *performSymbolicStoreEvaluation(Instruction *,
                                                   const BasicBlock *);
  const Expression *performSymbolicCallEvaluation(Instruction *,
                                                  const BasicBlock *);
  const Expression *performSymbolicPHIEvaluation(Instruction *,
                                                 const BasicBlock *);
  bool setMemoryAccessEquivTo(MemoryAccess *From, MemoryAccess *To);
  const Expression *performSymbolicAggrValueEvaluation(Instruction *,
                                                       const BasicBlock *);

  // Congruence finding.
  // Templated to allow them to work both on BB's and BB-edges.
  template <class T>
  Value *lookupOperandLeader(Value *, const User *, const T &) const;
  void performCongruenceFinding(Value *, const Expression *);
  void moveValueToNewCongruenceClass(Value *, CongruenceClass *,
                                     CongruenceClass *);
  // Reachability handling.
  void updateReachableEdge(BasicBlock *, BasicBlock *);
  void processOutgoingEdges(TerminatorInst *, BasicBlock *);
  bool isOnlyReachableViaThisEdge(const BasicBlockEdge &) const;
  Value *findConditionEquivalence(Value *, BasicBlock *) const;
  MemoryAccess *lookupMemoryAccessEquiv(MemoryAccess *) const;

  // Elimination.
  struct ValueDFS;
  void convertDenseToDFSOrdered(CongruenceClass::MemberSet &,
                                SmallVectorImpl<ValueDFS> &);

  bool eliminateInstructions(Function &);
  void replaceInstruction(Instruction *, Value *);
  void markInstructionForDeletion(Instruction *);
  void deleteInstructionsInBlock(BasicBlock *);

  // New instruction creation.
  void handleNewInstruction(Instruction *){};

  // Various instruction touch utilities
  void markUsersTouched(Value *);
  void markMemoryUsersTouched(MemoryAccess *);
  void markLeaderChangeTouched(CongruenceClass *CC);

  // Utilities.
  void cleanupTables();
  std::pair<unsigned, unsigned> assignDFSNumbers(BasicBlock *, unsigned);
  void updateProcessedCount(Value *V);
  void verifyMemoryCongruency() const;
  bool singleReachablePHIPath(const MemoryAccess *, const MemoryAccess *) const;
};

char NewGVN::ID = 0;

// createGVNPass - The public interface to this file.
FunctionPass *llvm::createNewGVNPass() { return new NewGVN(); }

template <typename T>
static bool equalsLoadStoreHelper(const T &LHS, const Expression &RHS) {
  if ((!isa<LoadExpression>(RHS) && !isa<StoreExpression>(RHS)) ||
      !LHS.BasicExpression::equals(RHS)) {
    return false;
  } else if (const auto *L = dyn_cast<LoadExpression>(&RHS)) {
    if (LHS.getDefiningAccess() != L->getDefiningAccess())
      return false;
  } else if (const auto *S = dyn_cast<StoreExpression>(&RHS)) {
    if (LHS.getDefiningAccess() != S->getDefiningAccess())
      return false;
  }
  return true;
}

bool LoadExpression::equals(const Expression &Other) const {
  return equalsLoadStoreHelper(*this, Other);
}

bool StoreExpression::equals(const Expression &Other) const {
  return equalsLoadStoreHelper(*this, Other);
}

#ifndef NDEBUG
static std::string getBlockName(const BasicBlock *B) {
  return DOTGraphTraits<const Function *>::getSimpleNodeLabel(B, nullptr);
}
#endif

INITIALIZE_PASS_BEGIN(NewGVN, "newgvn", "Global Value Numbering", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_END(NewGVN, "newgvn", "Global Value Numbering", false, false)

PHIExpression *NewGVN::createPHIExpression(Instruction *I) {
  BasicBlock *PHIBlock = I->getParent();
  auto *PN = cast<PHINode>(I);
  auto *E =
      new (ExpressionAllocator) PHIExpression(PN->getNumOperands(), PHIBlock);

  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(I->getType());
  E->setOpcode(I->getOpcode());

  auto ReachablePhiArg = [&](const Use &U) {
    return ReachableBlocks.count(PN->getIncomingBlock(U));
  };

  // Filter out unreachable operands
  auto Filtered = make_filter_range(PN->operands(), ReachablePhiArg);

  std::transform(Filtered.begin(), Filtered.end(), op_inserter(E),
                 [&](const Use &U) -> Value * {
                   // Don't try to transform self-defined phis.
                   if (U == PN)
                     return PN;
                   const BasicBlockEdge BBE(PN->getIncomingBlock(U), PHIBlock);
                   return lookupOperandLeader(U, I, BBE);
                 });
  return E;
}

// Set basic expression info (Arguments, type, opcode) for Expression
// E from Instruction I in block B.
bool NewGVN::setBasicExpressionInfo(Instruction *I, BasicExpression *E,
                                    const BasicBlock *B) {
  bool AllConstant = true;
  if (auto *GEP = dyn_cast<GetElementPtrInst>(I))
    E->setType(GEP->getSourceElementType());
  else
    E->setType(I->getType());
  E->setOpcode(I->getOpcode());
  E->allocateOperands(ArgRecycler, ExpressionAllocator);

  // Transform the operand array into an operand leader array, and keep track of
  // whether all members are constant.
  std::transform(I->op_begin(), I->op_end(), op_inserter(E), [&](Value *O) {
    auto Operand = lookupOperandLeader(O, I, B);
    AllConstant &= isa<Constant>(Operand);
    return Operand;
  });

  return AllConstant;
}

const Expression *NewGVN::createBinaryExpression(unsigned Opcode, Type *T,
                                                 Value *Arg1, Value *Arg2,
                                                 const BasicBlock *B) {
  auto *E = new (ExpressionAllocator) BasicExpression(2);

  E->setType(T);
  E->setOpcode(Opcode);
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  if (Instruction::isCommutative(Opcode)) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    if (Arg1 > Arg2)
      std::swap(Arg1, Arg2);
  }
  E->op_push_back(lookupOperandLeader(Arg1, nullptr, B));
  E->op_push_back(lookupOperandLeader(Arg2, nullptr, B));

  Value *V = SimplifyBinOp(Opcode, E->getOperand(0), E->getOperand(1), *DL, TLI,
                           DT, AC);
  if (const Expression *SimplifiedE = checkSimplificationResults(E, nullptr, V))
    return SimplifiedE;
  return E;
}

// Take a Value returned by simplification of Expression E/Instruction
// I, and see if it resulted in a simpler expression. If so, return
// that expression.
// TODO: Once finished, this should not take an Instruction, we only
// use it for printing.
const Expression *NewGVN::checkSimplificationResults(Expression *E,
                                                     Instruction *I, Value *V) {
  if (!V)
    return nullptr;
  if (auto *C = dyn_cast<Constant>(V)) {
    if (I)
      DEBUG(dbgs() << "Simplified " << *I << " to "
                   << " constant " << *C << "\n");
    NumGVNOpsSimplified++;
    assert(isa<BasicExpression>(E) &&
           "We should always have had a basic expression here");

    cast<BasicExpression>(E)->deallocateOperands(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    return createConstantExpression(C);
  } else if (isa<Argument>(V) || isa<GlobalVariable>(V)) {
    if (I)
      DEBUG(dbgs() << "Simplified " << *I << " to "
                   << " variable " << *V << "\n");
    cast<BasicExpression>(E)->deallocateOperands(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    return createVariableExpression(V);
  }

  CongruenceClass *CC = ValueToClass.lookup(V);
  if (CC && CC->DefiningExpr) {
    if (I)
      DEBUG(dbgs() << "Simplified " << *I << " to "
                   << " expression " << *V << "\n");
    NumGVNOpsSimplified++;
    assert(isa<BasicExpression>(E) &&
           "We should always have had a basic expression here");
    cast<BasicExpression>(E)->deallocateOperands(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    return CC->DefiningExpr;
  }
  return nullptr;
}

const Expression *NewGVN::createExpression(Instruction *I,
                                           const BasicBlock *B) {

  auto *E = new (ExpressionAllocator) BasicExpression(I->getNumOperands());

  bool AllConstant = setBasicExpressionInfo(I, E, B);

  if (I->isCommutative()) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    assert(I->getNumOperands() == 2 && "Unsupported commutative instruction!");
    if (E->getOperand(0) > E->getOperand(1))
      E->swapOperands(0, 1);
  }

  // Perform simplificaiton
  // TODO: Right now we only check to see if we get a constant result.
  // We may get a less than constant, but still better, result for
  // some operations.
  // IE
  //  add 0, x -> x
  //  and x, x -> x
  // We should handle this by simply rewriting the expression.
  if (auto *CI = dyn_cast<CmpInst>(I)) {
    // Sort the operand value numbers so x<y and y>x get the same value
    // number.
    CmpInst::Predicate Predicate = CI->getPredicate();
    if (E->getOperand(0) > E->getOperand(1)) {
      E->swapOperands(0, 1);
      Predicate = CmpInst::getSwappedPredicate(Predicate);
    }
    E->setOpcode((CI->getOpcode() << 8) | Predicate);
    // TODO: 25% of our time is spent in SimplifyCmpInst with pointer operands
    // TODO: Since we noop bitcasts, we may need to check types before
    // simplifying, so that we don't end up simplifying based on a wrong
    // type assumption. We should clean this up so we can use constants of the
    // wrong type

    assert(I->getOperand(0)->getType() == I->getOperand(1)->getType() &&
           "Wrong types on cmp instruction");
    if ((E->getOperand(0)->getType() == I->getOperand(0)->getType() &&
         E->getOperand(1)->getType() == I->getOperand(1)->getType())) {
      Value *V = SimplifyCmpInst(Predicate, E->getOperand(0), E->getOperand(1),
                                 *DL, TLI, DT, AC);
      if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
        return SimplifiedE;
    }
  } else if (isa<SelectInst>(I)) {
    if (isa<Constant>(E->getOperand(0)) ||
        (E->getOperand(1)->getType() == I->getOperand(1)->getType() &&
         E->getOperand(2)->getType() == I->getOperand(2)->getType())) {
      Value *V = SimplifySelectInst(E->getOperand(0), E->getOperand(1),
                                    E->getOperand(2), *DL, TLI, DT, AC);
      if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
        return SimplifiedE;
    }
  } else if (I->isBinaryOp()) {
    Value *V = SimplifyBinOp(E->getOpcode(), E->getOperand(0), E->getOperand(1),
                             *DL, TLI, DT, AC);
    if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
      return SimplifiedE;
  } else if (auto *BI = dyn_cast<BitCastInst>(I)) {
    Value *V = SimplifyInstruction(BI, *DL, TLI, DT, AC);
    if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
      return SimplifiedE;
  } else if (isa<GetElementPtrInst>(I)) {
    Value *V = SimplifyGEPInst(E->getType(),
                               ArrayRef<Value *>(E->op_begin(), E->op_end()),
                               *DL, TLI, DT, AC);
    if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
      return SimplifiedE;
  } else if (AllConstant) {
    // We don't bother trying to simplify unless all of the operands
    // were constant.
    // TODO: There are a lot of Simplify*'s we could call here, if we
    // wanted to.  The original motivating case for this code was a
    // zext i1 false to i8, which we don't have an interface to
    // simplify (IE there is no SimplifyZExt).

    SmallVector<Constant *, 8> C;
    for (Value *Arg : E->operands())
      C.emplace_back(cast<Constant>(Arg));

    if (Value *V = ConstantFoldInstOperands(I, C, *DL, TLI))
      if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
        return SimplifiedE;
  }
  return E;
}

const AggregateValueExpression *
NewGVN::createAggregateValueExpression(Instruction *I, const BasicBlock *B) {
  if (auto *II = dyn_cast<InsertValueInst>(I)) {
    auto *E = new (ExpressionAllocator)
        AggregateValueExpression(I->getNumOperands(), II->getNumIndices());
    setBasicExpressionInfo(I, E, B);
    E->allocateIntOperands(ExpressionAllocator);
    std::copy(II->idx_begin(), II->idx_end(), int_op_inserter(E));
    return E;
  } else if (auto *EI = dyn_cast<ExtractValueInst>(I)) {
    auto *E = new (ExpressionAllocator)
        AggregateValueExpression(I->getNumOperands(), EI->getNumIndices());
    setBasicExpressionInfo(EI, E, B);
    E->allocateIntOperands(ExpressionAllocator);
    std::copy(EI->idx_begin(), EI->idx_end(), int_op_inserter(E));
    return E;
  }
  llvm_unreachable("Unhandled type of aggregate value operation");
}

const VariableExpression *NewGVN::createVariableExpression(Value *V) {
  auto *E = new (ExpressionAllocator) VariableExpression(V);
  E->setOpcode(V->getValueID());
  return E;
}

const Expression *NewGVN::createVariableOrConstant(Value *V,
                                                   const BasicBlock *B) {
  auto Leader = lookupOperandLeader(V, nullptr, B);
  if (auto *C = dyn_cast<Constant>(Leader))
    return createConstantExpression(C);
  return createVariableExpression(Leader);
}

const ConstantExpression *NewGVN::createConstantExpression(Constant *C) {
  auto *E = new (ExpressionAllocator) ConstantExpression(C);
  E->setOpcode(C->getValueID());
  return E;
}

const UnknownExpression *NewGVN::createUnknownExpression(Instruction *I) {
  auto *E = new (ExpressionAllocator) UnknownExpression(I);
  E->setOpcode(I->getOpcode());
  return E;
}

const CallExpression *NewGVN::createCallExpression(CallInst *CI,
                                                   MemoryAccess *HV,
                                                   const BasicBlock *B) {
  // FIXME: Add operand bundles for calls.
  auto *E =
      new (ExpressionAllocator) CallExpression(CI->getNumOperands(), CI, HV);
  setBasicExpressionInfo(CI, E, B);
  return E;
}

// See if we have a congruence class and leader for this operand, and if so,
// return it. Otherwise, return the operand itself.
template <class T>
Value *NewGVN::lookupOperandLeader(Value *V, const User *U, const T &B) const {
  CongruenceClass *CC = ValueToClass.lookup(V);
  if (CC && (CC != InitialClass))
    return CC->RepLeader;
  return V;
}

MemoryAccess *NewGVN::lookupMemoryAccessEquiv(MemoryAccess *MA) const {
  MemoryAccess *Result = MemoryAccessEquiv.lookup(MA);
  return Result ? Result : MA;
}

LoadExpression *NewGVN::createLoadExpression(Type *LoadType, Value *PointerOp,
                                             LoadInst *LI, MemoryAccess *DA,
                                             const BasicBlock *B) {
  auto *E = new (ExpressionAllocator) LoadExpression(1, LI, DA);
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(LoadType);

  // Give store and loads same opcode so they value number together.
  E->setOpcode(0);
  E->op_push_back(lookupOperandLeader(PointerOp, LI, B));
  if (LI)
    E->setAlignment(LI->getAlignment());

  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load).
  return E;
}

const StoreExpression *NewGVN::createStoreExpression(StoreInst *SI,
                                                     MemoryAccess *DA,
                                                     const BasicBlock *B) {
  auto *E =
      new (ExpressionAllocator) StoreExpression(SI->getNumOperands(), SI, DA);
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(SI->getValueOperand()->getType());

  // Give store and loads same opcode so they value number together.
  E->setOpcode(0);
  E->op_push_back(lookupOperandLeader(SI->getPointerOperand(), SI, B));

  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load).
  return E;
}

// Utility function to check whether the congruence class has a member other
// than the given instruction.
bool hasMemberOtherThanUs(const CongruenceClass *CC, Instruction *I) {
  // Either it has more than one store, in which case it must contain something
  // other than us (because it's indexed by value), or if it only has one store
  // right now, that member should not be us.
  return CC->StoreCount > 1 || CC->Members.count(I) == 0;
}

const Expression *NewGVN::performSymbolicStoreEvaluation(Instruction *I,
                                                         const BasicBlock *B) {
  // Unlike loads, we never try to eliminate stores, so we do not check if they
  // are simple and avoid value numbering them.
  auto *SI = cast<StoreInst>(I);
  MemoryAccess *StoreAccess = MSSA->getMemoryAccess(SI);
  // See if we are defined by a previous store expression, it already has a
  // value, and it's the same value as our current store. FIXME: Right now, we
  // only do this for simple stores, we should expand to cover memcpys, etc.
  if (SI->isSimple()) {
    // Get the expression, if any, for the RHS of the MemoryDef.
    MemoryAccess *StoreRHS = lookupMemoryAccessEquiv(
        cast<MemoryDef>(StoreAccess)->getDefiningAccess());
    const Expression *OldStore = createStoreExpression(SI, StoreRHS, B);
    CongruenceClass *CC = ExpressionToClass.lookup(OldStore);
    // Basically, check if the congruence class the store is in is defined by a
    // store that isn't us, and has the same value.  MemorySSA takes care of
    // ensuring the store has the same memory state as us already.
    if (CC && CC->DefiningExpr && isa<StoreExpression>(CC->DefiningExpr) &&
        CC->RepLeader == lookupOperandLeader(SI->getValueOperand(), SI, B) &&
        hasMemberOtherThanUs(CC, I))
      return createStoreExpression(SI, StoreRHS, B);
  }

  return createStoreExpression(SI, StoreAccess, B);
}

const Expression *NewGVN::performSymbolicLoadEvaluation(Instruction *I,
                                                        const BasicBlock *B) {
  auto *LI = cast<LoadInst>(I);

  // We can eliminate in favor of non-simple loads, but we won't be able to
  // eliminate the loads themselves.
  if (!LI->isSimple())
    return nullptr;

  Value *LoadAddressLeader = lookupOperandLeader(LI->getPointerOperand(), I, B);
  // Load of undef is undef.
  if (isa<UndefValue>(LoadAddressLeader))
    return createConstantExpression(UndefValue::get(LI->getType()));

  MemoryAccess *DefiningAccess = MSSAWalker->getClobberingMemoryAccess(I);

  if (!MSSA->isLiveOnEntryDef(DefiningAccess)) {
    if (auto *MD = dyn_cast<MemoryDef>(DefiningAccess)) {
      Instruction *DefiningInst = MD->getMemoryInst();
      // If the defining instruction is not reachable, replace with undef.
      if (!ReachableBlocks.count(DefiningInst->getParent()))
        return createConstantExpression(UndefValue::get(LI->getType()));
    }
  }

  const Expression *E =
      createLoadExpression(LI->getType(), LI->getPointerOperand(), LI,
                           lookupMemoryAccessEquiv(DefiningAccess), B);
  return E;
}

// Evaluate read only and pure calls, and create an expression result.
const Expression *NewGVN::performSymbolicCallEvaluation(Instruction *I,
                                                        const BasicBlock *B) {
  auto *CI = cast<CallInst>(I);
  if (AA->doesNotAccessMemory(CI))
    return createCallExpression(CI, nullptr, B);
  if (AA->onlyReadsMemory(CI)) {
    MemoryAccess *DefiningAccess = MSSAWalker->getClobberingMemoryAccess(CI);
    return createCallExpression(CI, lookupMemoryAccessEquiv(DefiningAccess), B);
  }
  return nullptr;
}

// Update the memory access equivalence table to say that From is equal to To,
// and return true if this is different from what already existed in the table.
bool NewGVN::setMemoryAccessEquivTo(MemoryAccess *From, MemoryAccess *To) {
  DEBUG(dbgs() << "Setting " << *From << " equivalent to ");
  if (!To)
    DEBUG(dbgs() << "itself");
  else
    DEBUG(dbgs() << *To);
  DEBUG(dbgs() << "\n");
  auto LookupResult = MemoryAccessEquiv.find(From);
  bool Changed = false;
  // If it's already in the table, see if the value changed.
  if (LookupResult != MemoryAccessEquiv.end()) {
    if (To && LookupResult->second != To) {
      // It wasn't equivalent before, and now it is.
      LookupResult->second = To;
      Changed = true;
    } else if (!To) {
      // It used to be equivalent to something, and now it's not.
      MemoryAccessEquiv.erase(LookupResult);
      Changed = true;
    }
  } else {
    assert(!To &&
           "Memory equivalence should never change from nothing to something");
  }

  return Changed;
}
// Evaluate PHI nodes symbolically, and create an expression result.
const Expression *NewGVN::performSymbolicPHIEvaluation(Instruction *I,
                                                       const BasicBlock *B) {
  auto *E = cast<PHIExpression>(createPHIExpression(I));
  // We match the semantics of SimplifyPhiNode from InstructionSimplify here.

  // See if all arguaments are the same.
  // We track if any were undef because they need special handling.
  bool HasUndef = false;
  auto Filtered = make_filter_range(E->operands(), [&](const Value *Arg) {
    if (Arg == I)
      return false;
    if (isa<UndefValue>(Arg)) {
      HasUndef = true;
      return false;
    }
    return true;
  });
  // If we are left with no operands, it's undef
  if (Filtered.begin() == Filtered.end()) {
    DEBUG(dbgs() << "Simplified PHI node " << *I << " to undef"
                 << "\n");
    E->deallocateOperands(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    return createConstantExpression(UndefValue::get(I->getType()));
  }
  Value *AllSameValue = *(Filtered.begin());
  ++Filtered.begin();
  // Can't use std::equal here, sadly, because filter.begin moves.
  if (llvm::all_of(Filtered, [AllSameValue](const Value *V) {
        return V == AllSameValue;
      })) {
    // In LLVM's non-standard representation of phi nodes, it's possible to have
    // phi nodes with cycles (IE dependent on other phis that are .... dependent
    // on the original phi node), especially in weird CFG's where some arguments
    // are unreachable, or uninitialized along certain paths.  This can cause
    // infinite loops during evaluation. We work around this by not trying to
    // really evaluate them independently, but instead using a variable
    // expression to say if one is equivalent to the other.
    // We also special case undef, so that if we have an undef, we can't use the
    // common value unless it dominates the phi block.
    if (HasUndef) {
      // Only have to check for instructions
      if (auto *AllSameInst = dyn_cast<Instruction>(AllSameValue))
        if (!DT->dominates(AllSameInst, I))
          return E;
    }

    NumGVNPhisAllSame++;
    DEBUG(dbgs() << "Simplified PHI node " << *I << " to " << *AllSameValue
                 << "\n");
    E->deallocateOperands(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    if (auto *C = dyn_cast<Constant>(AllSameValue))
      return createConstantExpression(C);
    return createVariableExpression(AllSameValue);
  }
  return E;
}

const Expression *
NewGVN::performSymbolicAggrValueEvaluation(Instruction *I,
                                           const BasicBlock *B) {
  if (auto *EI = dyn_cast<ExtractValueInst>(I)) {
    auto *II = dyn_cast<IntrinsicInst>(EI->getAggregateOperand());
    if (II && EI->getNumIndices() == 1 && *EI->idx_begin() == 0) {
      unsigned Opcode = 0;
      // EI might be an extract from one of our recognised intrinsics. If it
      // is we'll synthesize a semantically equivalent expression instead on
      // an extract value expression.
      switch (II->getIntrinsicID()) {
      case Intrinsic::sadd_with_overflow:
      case Intrinsic::uadd_with_overflow:
        Opcode = Instruction::Add;
        break;
      case Intrinsic::ssub_with_overflow:
      case Intrinsic::usub_with_overflow:
        Opcode = Instruction::Sub;
        break;
      case Intrinsic::smul_with_overflow:
      case Intrinsic::umul_with_overflow:
        Opcode = Instruction::Mul;
        break;
      default:
        break;
      }

      if (Opcode != 0) {
        // Intrinsic recognized. Grab its args to finish building the
        // expression.
        assert(II->getNumArgOperands() == 2 &&
               "Expect two args for recognised intrinsics.");
        return createBinaryExpression(Opcode, EI->getType(),
                                      II->getArgOperand(0),
                                      II->getArgOperand(1), B);
      }
    }
  }

  return createAggregateValueExpression(I, B);
}

// Substitute and symbolize the value before value numbering.
const Expression *NewGVN::performSymbolicEvaluation(Value *V,
                                                    const BasicBlock *B) {
  const Expression *E = nullptr;
  if (auto *C = dyn_cast<Constant>(V))
    E = createConstantExpression(C);
  else if (isa<Argument>(V) || isa<GlobalVariable>(V)) {
    E = createVariableExpression(V);
  } else {
    // TODO: memory intrinsics.
    // TODO: Some day, we should do the forward propagation and reassociation
    // parts of the algorithm.
    auto *I = cast<Instruction>(V);
    switch (I->getOpcode()) {
    case Instruction::ExtractValue:
    case Instruction::InsertValue:
      E = performSymbolicAggrValueEvaluation(I, B);
      break;
    case Instruction::PHI:
      E = performSymbolicPHIEvaluation(I, B);
      break;
    case Instruction::Call:
      E = performSymbolicCallEvaluation(I, B);
      break;
    case Instruction::Store:
      E = performSymbolicStoreEvaluation(I, B);
      break;
    case Instruction::Load:
      E = performSymbolicLoadEvaluation(I, B);
      break;
    case Instruction::BitCast: {
      E = createExpression(I, B);
    } break;

    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::ICmp:
    case Instruction::FCmp:
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::Select:
    case Instruction::ExtractElement:
    case Instruction::InsertElement:
    case Instruction::ShuffleVector:
    case Instruction::GetElementPtr:
      E = createExpression(I, B);
      break;
    default:
      return nullptr;
    }
  }
  return E;
}

// There is an edge from 'Src' to 'Dst'.  Return true if every path from
// the entry block to 'Dst' passes via this edge.  In particular 'Dst'
// must not be reachable via another edge from 'Src'.
bool NewGVN::isOnlyReachableViaThisEdge(const BasicBlockEdge &E) const {

  // While in theory it is interesting to consider the case in which Dst has
  // more than one predecessor, because Dst might be part of a loop which is
  // only reachable from Src, in practice it is pointless since at the time
  // GVN runs all such loops have preheaders, which means that Dst will have
  // been changed to have only one predecessor, namely Src.
  const BasicBlock *Pred = E.getEnd()->getSinglePredecessor();
  const BasicBlock *Src = E.getStart();
  assert((!Pred || Pred == Src) && "No edge between these basic blocks!");
  (void)Src;
  return Pred != nullptr;
}

void NewGVN::markUsersTouched(Value *V) {
  // Now mark the users as touched.
  for (auto *User : V->users()) {
    assert(isa<Instruction>(User) && "Use of value not within an instruction?");
    TouchedInstructions.set(InstrDFS[User]);
  }
}

void NewGVN::markMemoryUsersTouched(MemoryAccess *MA) {
  for (auto U : MA->users()) {
    if (auto *MUD = dyn_cast<MemoryUseOrDef>(U))
      TouchedInstructions.set(InstrDFS[MUD->getMemoryInst()]);
    else
      TouchedInstructions.set(InstrDFS[U]);
  }
}

// Touch the instructions that need to be updated after a congruence class has a
// leader change, and mark changed values.
void NewGVN::markLeaderChangeTouched(CongruenceClass *CC) {
  for (auto M : CC->Members) {
    if (auto *I = dyn_cast<Instruction>(M))
      TouchedInstructions.set(InstrDFS[I]);
    LeaderChanges.insert(M);
  }
}

// Move a value, currently in OldClass, to be part of NewClass
// Update OldClass for the move (including changing leaders, etc)
void NewGVN::moveValueToNewCongruenceClass(Value *V, CongruenceClass *OldClass,
                                           CongruenceClass *NewClass) {
  DEBUG(dbgs() << "New congruence class for " << V << " is " << NewClass->ID
               << "\n");
  OldClass->Members.erase(V);
  NewClass->Members.insert(V);
  if (isa<StoreInst>(V)) {
    --OldClass->StoreCount;
    assert(OldClass->StoreCount >= 0);
    ++NewClass->StoreCount;
    assert(NewClass->StoreCount > 0);
  }

  ValueToClass[V] = NewClass;
  // See if we destroyed the class or need to swap leaders.
  if (OldClass->Members.empty() && OldClass != InitialClass) {
    if (OldClass->DefiningExpr) {
      OldClass->Dead = true;
      DEBUG(dbgs() << "Erasing expression " << OldClass->DefiningExpr
                   << " from table\n");
      ExpressionToClass.erase(OldClass->DefiningExpr);
    }
  } else if (OldClass->RepLeader == V) {
    // When the leader changes, the value numbering of
    // everything may change due to symbolization changes, so we need to
    // reprocess.
    OldClass->RepLeader = *(OldClass->Members.begin());
    markLeaderChangeTouched(OldClass);
  }
}

// Perform congruence finding on a given value numbering expression.
void NewGVN::performCongruenceFinding(Value *V, const Expression *E) {
  ValueToExpression[V] = E;
  // This is guaranteed to return something, since it will at least find
  // INITIAL.

  CongruenceClass *VClass = ValueToClass[V];
  assert(VClass && "Should have found a vclass");
  // Dead classes should have been eliminated from the mapping.
  assert(!VClass->Dead && "Found a dead class");

  CongruenceClass *EClass;
  if (const auto *VE = dyn_cast<VariableExpression>(E)) {
    EClass = ValueToClass[VE->getVariableValue()];
  } else {
    auto lookupResult = ExpressionToClass.insert({E, nullptr});

    // If it's not in the value table, create a new congruence class.
    if (lookupResult.second) {
      CongruenceClass *NewClass = createCongruenceClass(nullptr, E);
      auto place = lookupResult.first;
      place->second = NewClass;

      // Constants and variables should always be made the leader.
      if (const auto *CE = dyn_cast<ConstantExpression>(E)) {
        NewClass->RepLeader = CE->getConstantValue();
      } else if (const auto *SE = dyn_cast<StoreExpression>(E)) {
        StoreInst *SI = SE->getStoreInst();
        NewClass->RepLeader =
            lookupOperandLeader(SI->getValueOperand(), SI, SI->getParent());
      } else {
        NewClass->RepLeader = V;
      }
      assert(!isa<VariableExpression>(E) &&
             "VariableExpression should have been handled already");

      EClass = NewClass;
      DEBUG(dbgs() << "Created new congruence class for " << *V
                   << " using expression " << *E << " at " << NewClass->ID
                   << " and leader " << *(NewClass->RepLeader) << "\n");
      DEBUG(dbgs() << "Hash value was " << E->getHashValue() << "\n");
    } else {
      EClass = lookupResult.first->second;
      if (isa<ConstantExpression>(E))
        assert(isa<Constant>(EClass->RepLeader) &&
               "Any class with a constant expression should have a "
               "constant leader");

      assert(EClass && "Somehow don't have an eclass");

      assert(!EClass->Dead && "We accidentally looked up a dead class");
    }
  }
  bool ClassChanged = VClass != EClass;
  bool LeaderChanged = LeaderChanges.erase(V);
  if (ClassChanged || LeaderChanged) {
    DEBUG(dbgs() << "Found class " << EClass->ID << " for expression " << E
                 << "\n");

    if (ClassChanged)

      moveValueToNewCongruenceClass(V, VClass, EClass);


    markUsersTouched(V);
    if (auto *I = dyn_cast<Instruction>(V)) {
      if (MemoryAccess *MA = MSSA->getMemoryAccess(I)) {
        // If this is a MemoryDef, we need to update the equivalence table. If
        // we determined the expression is congruent to a different memory
        // state, use that different memory state.  If we determined it didn't,
        // we update that as well.  Right now, we only support store
        // expressions.
        if (!isa<MemoryUse>(MA) && isa<StoreExpression>(E) &&
            EClass->Members.size() != 1) {
          auto *DefAccess = cast<StoreExpression>(E)->getDefiningAccess();
          setMemoryAccessEquivTo(MA, DefAccess != MA ? DefAccess : nullptr);
        } else {
          setMemoryAccessEquivTo(MA, nullptr);
        }
        markMemoryUsersTouched(MA);
      }
    }
  } else if (StoreInst *SI = dyn_cast<StoreInst>(V)) {
    // There is, sadly, one complicating thing for stores.  Stores do not
    // produce values, only consume them.  However, in order to make loads and
    // stores value number the same, we ignore the value operand of the store.
    // But the value operand will still be the leader of our class, and thus, it
    // may change.  Because the store is a use, the store will get reprocessed,
    // but nothing will change about it, and so nothing above will catch it
    // (since the class will not change).  In order to make sure everything ends
    // up okay, we need to recheck the leader of the class.  Since stores of
    // different values value number differently due to different memorydefs, we
    // are guaranteed the leader is always the same between stores in the same
    // class.
    DEBUG(dbgs() << "Checking store leader\n");
    auto ProperLeader =
        lookupOperandLeader(SI->getValueOperand(), SI, SI->getParent());
    if (EClass->RepLeader != ProperLeader) {
      DEBUG(dbgs() << "Store leader changed, fixing\n");
      EClass->RepLeader = ProperLeader;
      markLeaderChangeTouched(EClass);
      markMemoryUsersTouched(MSSA->getMemoryAccess(SI));
    }
  }
}

// Process the fact that Edge (from, to) is reachable, including marking
// any newly reachable blocks and instructions for processing.
void NewGVN::updateReachableEdge(BasicBlock *From, BasicBlock *To) {
  // Check if the Edge was reachable before.
  if (ReachableEdges.insert({From, To}).second) {
    // If this block wasn't reachable before, all instructions are touched.
    if (ReachableBlocks.insert(To).second) {
      DEBUG(dbgs() << "Block " << getBlockName(To) << " marked reachable\n");
      const auto &InstRange = BlockInstRange.lookup(To);
      TouchedInstructions.set(InstRange.first, InstRange.second);
    } else {
      DEBUG(dbgs() << "Block " << getBlockName(To)
                   << " was reachable, but new edge {" << getBlockName(From)
                   << "," << getBlockName(To) << "} to it found\n");

      // We've made an edge reachable to an existing block, which may
      // impact predicates. Otherwise, only mark the phi nodes as touched, as
      // they are the only thing that depend on new edges. Anything using their
      // values will get propagated to if necessary.
      if (MemoryAccess *MemPhi = MSSA->getMemoryAccess(To))
        TouchedInstructions.set(InstrDFS[MemPhi]);

      auto BI = To->begin();
      while (isa<PHINode>(BI)) {
        TouchedInstructions.set(InstrDFS[&*BI]);
        ++BI;
      }
    }
  }
}

// Given a predicate condition (from a switch, cmp, or whatever) and a block,
// see if we know some constant value for it already.
Value *NewGVN::findConditionEquivalence(Value *Cond, BasicBlock *B) const {
  auto Result = lookupOperandLeader(Cond, nullptr, B);
  if (isa<Constant>(Result))
    return Result;
  return nullptr;
}

// Process the outgoing edges of a block for reachability.
void NewGVN::processOutgoingEdges(TerminatorInst *TI, BasicBlock *B) {
  // Evaluate reachability of terminator instruction.
  BranchInst *BR;
  if ((BR = dyn_cast<BranchInst>(TI)) && BR->isConditional()) {
    Value *Cond = BR->getCondition();
    Value *CondEvaluated = findConditionEquivalence(Cond, B);
    if (!CondEvaluated) {
      if (auto *I = dyn_cast<Instruction>(Cond)) {
        const Expression *E = createExpression(I, B);
        if (const auto *CE = dyn_cast<ConstantExpression>(E)) {
          CondEvaluated = CE->getConstantValue();
        }
      } else if (isa<ConstantInt>(Cond)) {
        CondEvaluated = Cond;
      }
    }
    ConstantInt *CI;
    BasicBlock *TrueSucc = BR->getSuccessor(0);
    BasicBlock *FalseSucc = BR->getSuccessor(1);
    if (CondEvaluated && (CI = dyn_cast<ConstantInt>(CondEvaluated))) {
      if (CI->isOne()) {
        DEBUG(dbgs() << "Condition for Terminator " << *TI
                     << " evaluated to true\n");
        updateReachableEdge(B, TrueSucc);
      } else if (CI->isZero()) {
        DEBUG(dbgs() << "Condition for Terminator " << *TI
                     << " evaluated to false\n");
        updateReachableEdge(B, FalseSucc);
      }
    } else {
      updateReachableEdge(B, TrueSucc);
      updateReachableEdge(B, FalseSucc);
    }
  } else if (auto *SI = dyn_cast<SwitchInst>(TI)) {
    // For switches, propagate the case values into the case
    // destinations.

    // Remember how many outgoing edges there are to every successor.
    SmallDenseMap<BasicBlock *, unsigned, 16> SwitchEdges;

    Value *SwitchCond = SI->getCondition();
    Value *CondEvaluated = findConditionEquivalence(SwitchCond, B);
    // See if we were able to turn this switch statement into a constant.
    if (CondEvaluated && isa<ConstantInt>(CondEvaluated)) {
      auto *CondVal = cast<ConstantInt>(CondEvaluated);
      // We should be able to get case value for this.
      auto CaseVal = SI->findCaseValue(CondVal);
      if (CaseVal.getCaseSuccessor() == SI->getDefaultDest()) {
        // We proved the value is outside of the range of the case.
        // We can't do anything other than mark the default dest as reachable,
        // and go home.
        updateReachableEdge(B, SI->getDefaultDest());
        return;
      }
      // Now get where it goes and mark it reachable.
      BasicBlock *TargetBlock = CaseVal.getCaseSuccessor();
      updateReachableEdge(B, TargetBlock);
    } else {
      for (unsigned i = 0, e = SI->getNumSuccessors(); i != e; ++i) {
        BasicBlock *TargetBlock = SI->getSuccessor(i);
        ++SwitchEdges[TargetBlock];
        updateReachableEdge(B, TargetBlock);
      }
    }
  } else {
    // Otherwise this is either unconditional, or a type we have no
    // idea about. Just mark successors as reachable.
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
      BasicBlock *TargetBlock = TI->getSuccessor(i);
      updateReachableEdge(B, TargetBlock);
    }

    // This also may be a memory defining terminator, in which case, set it
    // equivalent to nothing.
    if (MemoryAccess *MA = MSSA->getMemoryAccess(TI))
      setMemoryAccessEquivTo(MA, nullptr);
  }
}

// The algorithm initially places the values of the routine in the INITIAL
// congruence
// class. The leader of INITIAL is the undetermined value `TOP`.
// When the algorithm has finished, values still in INITIAL are unreachable.
void NewGVN::initializeCongruenceClasses(Function &F) {
  // FIXME now i can't remember why this is 2
  NextCongruenceNum = 2;
  // Initialize all other instructions to be in INITIAL class.
  CongruenceClass::MemberSet InitialValues;
  InitialClass = createCongruenceClass(nullptr, nullptr);
  for (auto &B : F) {
    if (auto *MP = MSSA->getMemoryAccess(&B))
      MemoryAccessEquiv.insert({MP, MSSA->getLiveOnEntryDef()});

    for (auto &I : B) {
      InitialValues.insert(&I);
      ValueToClass[&I] = InitialClass;
      // All memory accesses are equivalent to live on entry to start. They must
      // be initialized to something so that initial changes are noticed. For
      // the maximal answer, we initialize them all to be the same as
      // liveOnEntry.  Note that to save time, we only initialize the
      // MemoryDef's for stores and all MemoryPhis to be equal.  Right now, no
      // other expression can generate a memory equivalence.  If we start
      // handling memcpy/etc, we can expand this.
      if (isa<StoreInst>(&I)) {
        MemoryAccessEquiv.insert(
            {MSSA->getMemoryAccess(&I), MSSA->getLiveOnEntryDef()});
        ++InitialClass->StoreCount;
        assert(InitialClass->StoreCount > 0);
      }
    }
  }
  InitialClass->Members.swap(InitialValues);

  // Initialize arguments to be in their own unique congruence classes
  for (auto &FA : F.args())
    createSingletonCongruenceClass(&FA);
}

void NewGVN::cleanupTables() {
  for (unsigned i = 0, e = CongruenceClasses.size(); i != e; ++i) {
    DEBUG(dbgs() << "Congruence class " << CongruenceClasses[i]->ID << " has "
                 << CongruenceClasses[i]->Members.size() << " members\n");
    // Make sure we delete the congruence class (probably worth switching to
    // a unique_ptr at some point.
    delete CongruenceClasses[i];
    CongruenceClasses[i] = nullptr;
  }

  ValueToClass.clear();
  ArgRecycler.clear(ExpressionAllocator);
  ExpressionAllocator.Reset();
  CongruenceClasses.clear();
  ExpressionToClass.clear();
  ValueToExpression.clear();
  ReachableBlocks.clear();
  ReachableEdges.clear();
#ifndef NDEBUG
  ProcessedCount.clear();
#endif
  DFSDomMap.clear();
  InstrDFS.clear();
  InstructionsToErase.clear();

  DFSToInstr.clear();
  BlockInstRange.clear();
  TouchedInstructions.clear();
  DominatedInstRange.clear();
  MemoryAccessEquiv.clear();
}

std::pair<unsigned, unsigned> NewGVN::assignDFSNumbers(BasicBlock *B,
                                                       unsigned Start) {
  unsigned End = Start;
  if (MemoryAccess *MemPhi = MSSA->getMemoryAccess(B)) {
    InstrDFS[MemPhi] = End++;
    DFSToInstr.emplace_back(MemPhi);
  }

  for (auto &I : *B) {
    InstrDFS[&I] = End++;
    DFSToInstr.emplace_back(&I);
  }

  // All of the range functions taken half-open ranges (open on the end side).
  // So we do not subtract one from count, because at this point it is one
  // greater than the last instruction.
  return std::make_pair(Start, End);
}

void NewGVN::updateProcessedCount(Value *V) {
#ifndef NDEBUG
  if (ProcessedCount.count(V) == 0) {
    ProcessedCount.insert({V, 1});
  } else {
    ProcessedCount[V] += 1;
    assert(ProcessedCount[V] < 100 &&
           "Seem to have processed the same Value a lot");
  }
#endif
}
// Evaluate MemoryPhi nodes symbolically, just like PHI nodes
void NewGVN::valueNumberMemoryPhi(MemoryPhi *MP) {
  // If all the arguments are the same, the MemoryPhi has the same value as the
  // argument.
  // Filter out unreachable blocks from our operands.
  auto Filtered = make_filter_range(MP->operands(), [&](const Use &U) {
    return ReachableBlocks.count(MP->getIncomingBlock(U));
  });

  assert(Filtered.begin() != Filtered.end() &&
         "We should not be processing a MemoryPhi in a completely "
         "unreachable block");

  // Transform the remaining operands into operand leaders.
  // FIXME: mapped_iterator should have a range version.
  auto LookupFunc = [&](const Use &U) {
    return lookupMemoryAccessEquiv(cast<MemoryAccess>(U));
  };
  auto MappedBegin = map_iterator(Filtered.begin(), LookupFunc);
  auto MappedEnd = map_iterator(Filtered.end(), LookupFunc);

  // and now check if all the elements are equal.
  // Sadly, we can't use std::equals since these are random access iterators.
  MemoryAccess *AllSameValue = *MappedBegin;
  ++MappedBegin;
  bool AllEqual = std::all_of(
      MappedBegin, MappedEnd,
      [&AllSameValue](const MemoryAccess *V) { return V == AllSameValue; });

  if (AllEqual)
    DEBUG(dbgs() << "Memory Phi value numbered to " << *AllSameValue << "\n");
  else
    DEBUG(dbgs() << "Memory Phi value numbered to itself\n");

  if (setMemoryAccessEquivTo(MP, AllEqual ? AllSameValue : nullptr))
    markMemoryUsersTouched(MP);
}

// Value number a single instruction, symbolically evaluating, performing
// congruence finding, and updating mappings.
void NewGVN::valueNumberInstruction(Instruction *I) {
  DEBUG(dbgs() << "Processing instruction " << *I << "\n");
  if (isInstructionTriviallyDead(I, TLI)) {
    DEBUG(dbgs() << "Skipping unused instruction\n");
    markInstructionForDeletion(I);
    return;
  }
  if (!I->isTerminator()) {
    const auto *Symbolized = performSymbolicEvaluation(I, I->getParent());
    // If we couldn't come up with a symbolic expression, use the unknown
    // expression
    if (Symbolized == nullptr)
      Symbolized = createUnknownExpression(I);
    performCongruenceFinding(I, Symbolized);
  } else {
    // Handle terminators that return values. All of them produce values we
    // don't currently understand.
    if (!I->getType()->isVoidTy()) {
      auto *Symbolized = createUnknownExpression(I);
      performCongruenceFinding(I, Symbolized);
    }
    processOutgoingEdges(dyn_cast<TerminatorInst>(I), I->getParent());
  }
}

// Check if there is a path, using single or equal argument phi nodes, from
// First to Second.
bool NewGVN::singleReachablePHIPath(const MemoryAccess *First,
                                    const MemoryAccess *Second) const {
  if (First == Second)
    return true;

  if (auto *FirstDef = dyn_cast<MemoryUseOrDef>(First)) {
    auto *DefAccess = FirstDef->getDefiningAccess();
    return singleReachablePHIPath(DefAccess, Second);
  } else {
    auto *MP = cast<MemoryPhi>(First);
    auto ReachableOperandPred = [&](const Use &U) {
      return ReachableBlocks.count(MP->getIncomingBlock(U));
    };
    auto FilteredPhiArgs =
        make_filter_range(MP->operands(), ReachableOperandPred);
    SmallVector<const Value *, 32> OperandList;
    std::copy(FilteredPhiArgs.begin(), FilteredPhiArgs.end(),
              std::back_inserter(OperandList));
    bool Okay = OperandList.size() == 1;
    if (!Okay)
      Okay = std::equal(OperandList.begin(), OperandList.end(),
                        OperandList.begin());
    if (Okay)
      return singleReachablePHIPath(cast<MemoryAccess>(OperandList[0]), Second);
    return false;
  }
}

// Verify the that the memory equivalence table makes sense relative to the
// congruence classes.  Note that this checking is not perfect, and is currently
// subject to very rare false negatives. It is only useful for testing/debugging.
void NewGVN::verifyMemoryCongruency() const {
  // Anything equivalent in the memory access table should be in the same
  // congruence class.

  // Filter out the unreachable and trivially dead entries, because they may
  // never have been updated if the instructions were not processed.
  auto ReachableAccessPred =
      [&](const std::pair<const MemoryAccess *, MemoryAccess *> Pair) {
        bool Result = ReachableBlocks.count(Pair.first->getBlock());
        if (!Result)
          return false;
        if (auto *MemDef = dyn_cast<MemoryDef>(Pair.first))
          return !isInstructionTriviallyDead(MemDef->getMemoryInst());
        return true;
      };

  auto Filtered = make_filter_range(MemoryAccessEquiv, ReachableAccessPred);
  for (auto KV : Filtered) {
    assert(KV.first != KV.second &&
           "We added a useless equivalence to the memory equivalence table");
    // Unreachable instructions may not have changed because we never process
    // them.
    if (!ReachableBlocks.count(KV.first->getBlock()))
      continue;
    if (auto *FirstMUD = dyn_cast<MemoryUseOrDef>(KV.first)) {
      auto *SecondMUD = dyn_cast<MemoryUseOrDef>(KV.second);
      if (FirstMUD && SecondMUD)
        assert((singleReachablePHIPath(FirstMUD, SecondMUD) ||
               ValueToClass.lookup(FirstMUD->getMemoryInst()) ==
                       ValueToClass.lookup(SecondMUD->getMemoryInst())) &&
                   "The instructions for these memory operations should have "
                   "been in the same congruence class or reachable through"
                   "a single argument phi");
    } else if (auto *FirstMP = dyn_cast<MemoryPhi>(KV.first)) {

      // We can only sanely verify that MemoryDefs in the operand list all have
      // the same class.
      auto ReachableOperandPred = [&](const Use &U) {
        return ReachableBlocks.count(FirstMP->getIncomingBlock(U)) &&
               isa<MemoryDef>(U);

      };
      // All arguments should in the same class, ignoring unreachable arguments
      auto FilteredPhiArgs =
          make_filter_range(FirstMP->operands(), ReachableOperandPred);
      SmallVector<const CongruenceClass *, 16> PhiOpClasses;
      std::transform(FilteredPhiArgs.begin(), FilteredPhiArgs.end(),
                     std::back_inserter(PhiOpClasses), [&](const Use &U) {
                       const MemoryDef *MD = cast<MemoryDef>(U);
                       return ValueToClass.lookup(MD->getMemoryInst());
                     });
      assert(std::equal(PhiOpClasses.begin(), PhiOpClasses.end(),
                        PhiOpClasses.begin()) &&
             "All MemoryPhi arguments should be in the same class");
    }
  }
}

// This is the main transformation entry point.
bool NewGVN::runGVN(Function &F, DominatorTree *_DT, AssumptionCache *_AC,
                    TargetLibraryInfo *_TLI, AliasAnalysis *_AA,
                    MemorySSA *_MSSA) {
  bool Changed = false;
  DT = _DT;
  AC = _AC;
  TLI = _TLI;
  AA = _AA;
  MSSA = _MSSA;
  DL = &F.getParent()->getDataLayout();
  MSSAWalker = MSSA->getWalker();

  // Count number of instructions for sizing of hash tables, and come
  // up with a global dfs numbering for instructions.
  unsigned ICount = 1;
  // Add an empty instruction to account for the fact that we start at 1
  DFSToInstr.emplace_back(nullptr);
  // Note: We want RPO traversal of the blocks, which is not quite the same as
  // dominator tree order, particularly with regard whether backedges get
  // visited first or second, given a block with multiple successors.
  // If we visit in the wrong order, we will end up performing N times as many
  // iterations.
  // The dominator tree does guarantee that, for a given dom tree node, it's
  // parent must occur before it in the RPO ordering. Thus, we only need to sort
  // the siblings.
  DenseMap<const DomTreeNode *, unsigned> RPOOrdering;
  ReversePostOrderTraversal<Function *> RPOT(&F);
  unsigned Counter = 0;
  for (auto &B : RPOT) {
    auto *Node = DT->getNode(B);
    assert(Node && "RPO and Dominator tree should have same reachability");
    RPOOrdering[Node] = ++Counter;
  }
  // Sort dominator tree children arrays into RPO.
  for (auto &B : RPOT) {
    auto *Node = DT->getNode(B);
    if (Node->getChildren().size() > 1)
      std::sort(Node->begin(), Node->end(),
                [&RPOOrdering](const DomTreeNode *A, const DomTreeNode *B) {
                  return RPOOrdering[A] < RPOOrdering[B];
                });
  }

  // Now a standard depth first ordering of the domtree is equivalent to RPO.
  auto DFI = df_begin(DT->getRootNode());
  for (auto DFE = df_end(DT->getRootNode()); DFI != DFE; ++DFI) {
    BasicBlock *B = DFI->getBlock();
    const auto &BlockRange = assignDFSNumbers(B, ICount);
    BlockInstRange.insert({B, BlockRange});
    ICount += BlockRange.second - BlockRange.first;
  }

  // Handle forward unreachable blocks and figure out which blocks
  // have single preds.
  for (auto &B : F) {
    // Assign numbers to unreachable blocks.
    if (!DFI.nodeVisited(DT->getNode(&B))) {
      const auto &BlockRange = assignDFSNumbers(&B, ICount);
      BlockInstRange.insert({&B, BlockRange});
      ICount += BlockRange.second - BlockRange.first;
    }
  }

  TouchedInstructions.resize(ICount);
  DominatedInstRange.reserve(F.size());
  // Ensure we don't end up resizing the expressionToClass map, as
  // that can be quite expensive. At most, we have one expression per
  // instruction.
  ExpressionToClass.reserve(ICount);

  // Initialize the touched instructions to include the entry block.
  const auto &InstRange = BlockInstRange.lookup(&F.getEntryBlock());
  TouchedInstructions.set(InstRange.first, InstRange.second);
  ReachableBlocks.insert(&F.getEntryBlock());

  initializeCongruenceClasses(F);

  unsigned int Iterations = 0;
  // We start out in the entry block.
  BasicBlock *LastBlock = &F.getEntryBlock();
  while (TouchedInstructions.any()) {
    ++Iterations;
    // Walk through all the instructions in all the blocks in RPO.
    for (int InstrNum = TouchedInstructions.find_first(); InstrNum != -1;
         InstrNum = TouchedInstructions.find_next(InstrNum)) {
      assert(InstrNum != 0 && "Bit 0 should never be set, something touched an "
                              "instruction not in the lookup table");
      Value *V = DFSToInstr[InstrNum];
      BasicBlock *CurrBlock = nullptr;

      if (auto *I = dyn_cast<Instruction>(V))
        CurrBlock = I->getParent();
      else if (auto *MP = dyn_cast<MemoryPhi>(V))
        CurrBlock = MP->getBlock();
      else
        llvm_unreachable("DFSToInstr gave us an unknown type of instruction");

      // If we hit a new block, do reachability processing.
      if (CurrBlock != LastBlock) {
        LastBlock = CurrBlock;
        bool BlockReachable = ReachableBlocks.count(CurrBlock);
        const auto &CurrInstRange = BlockInstRange.lookup(CurrBlock);

        // If it's not reachable, erase any touched instructions and move on.
        if (!BlockReachable) {
          TouchedInstructions.reset(CurrInstRange.first, CurrInstRange.second);
          DEBUG(dbgs() << "Skipping instructions in block "
                       << getBlockName(CurrBlock)
                       << " because it is unreachable\n");
          continue;
        }
        updateProcessedCount(CurrBlock);
      }

      if (auto *MP = dyn_cast<MemoryPhi>(V)) {
        DEBUG(dbgs() << "Processing MemoryPhi " << *MP << "\n");
        valueNumberMemoryPhi(MP);
      } else if (auto *I = dyn_cast<Instruction>(V)) {
        valueNumberInstruction(I);
      } else {
        llvm_unreachable("Should have been a MemoryPhi or Instruction");
      }
      updateProcessedCount(V);
      // Reset after processing (because we may mark ourselves as touched when
      // we propagate equalities).
      TouchedInstructions.reset(InstrNum);
    }
  }
  NumGVNMaxIterations = std::max(NumGVNMaxIterations.getValue(), Iterations);
#ifndef NDEBUG
  verifyMemoryCongruency();
#endif
  Changed |= eliminateInstructions(F);

  // Delete all instructions marked for deletion.
  for (Instruction *ToErase : InstructionsToErase) {
    if (!ToErase->use_empty())
      ToErase->replaceAllUsesWith(UndefValue::get(ToErase->getType()));

    ToErase->eraseFromParent();
  }

  // Delete all unreachable blocks.
  auto UnreachableBlockPred = [&](const BasicBlock &BB) {
    return !ReachableBlocks.count(&BB);
  };

  for (auto &BB : make_filter_range(F, UnreachableBlockPred)) {
    DEBUG(dbgs() << "We believe block " << getBlockName(&BB)
                 << " is unreachable\n");
    deleteInstructionsInBlock(&BB);
    Changed = true;
  }

  cleanupTables();
  return Changed;
}

bool NewGVN::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  return runGVN(F, &getAnalysis<DominatorTreeWrapperPass>().getDomTree(),
                &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F),
                &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(),
                &getAnalysis<AAResultsWrapperPass>().getAAResults(),
                &getAnalysis<MemorySSAWrapperPass>().getMSSA());
}

PreservedAnalyses NewGVNPass::run(Function &F, AnalysisManager<Function> &AM) {
  NewGVN Impl;

  // Apparently the order in which we get these results matter for
  // the old GVN (see Chandler's comment in GVN.cpp). I'll keep
  // the same order here, just in case.
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &AA = AM.getResult<AAManager>(F);
  auto &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();
  bool Changed = Impl.runGVN(F, &DT, &AC, &TLI, &AA, &MSSA);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<GlobalsAA>();
  return PA;
}

// Return true if V is a value that will always be available (IE can
// be placed anywhere) in the function.  We don't do globals here
// because they are often worse to put in place.
// TODO: Separate cost from availability
static bool alwaysAvailable(Value *V) {
  return isa<Constant>(V) || isa<Argument>(V);
}

// Get the basic block from an instruction/value.
static BasicBlock *getBlockForValue(Value *V) {
  if (auto *I = dyn_cast<Instruction>(V))
    return I->getParent();
  return nullptr;
}

struct NewGVN::ValueDFS {
  int DFSIn = 0;
  int DFSOut = 0;
  int LocalNum = 0;
  // Only one of these will be set.
  Value *Val = nullptr;
  Use *U = nullptr;

  bool operator<(const ValueDFS &Other) const {
    // It's not enough that any given field be less than - we have sets
    // of fields that need to be evaluated together to give a proper ordering.
    // For example, if you have;
    // DFS (1, 3)
    // Val 0
    // DFS (1, 2)
    // Val 50
    // We want the second to be less than the first, but if we just go field
    // by field, we will get to Val 0 < Val 50 and say the first is less than
    // the second. We only want it to be less than if the DFS orders are equal.
    //
    // Each LLVM instruction only produces one value, and thus the lowest-level
    // differentiator that really matters for the stack (and what we use as as a
    // replacement) is the local dfs number.
    // Everything else in the structure is instruction level, and only affects
    // the order in which we will replace operands of a given instruction.
    //
    // For a given instruction (IE things with equal dfsin, dfsout, localnum),
    // the order of replacement of uses does not matter.
    // IE given,
    //  a = 5
    //  b = a + a
    // When you hit b, you will have two valuedfs with the same dfsin, out, and
    // localnum.
    // The .val will be the same as well.
    // The .u's will be different.
    // You will replace both, and it does not matter what order you replace them
    // in (IE whether you replace operand 2, then operand 1, or operand 1, then
    // operand 2).
    // Similarly for the case of same dfsin, dfsout, localnum, but different
    // .val's
    //  a = 5
    //  b  = 6
    //  c = a + b
    // in c, we will a valuedfs for a, and one for b,with everything the same
    // but .val  and .u.
    // It does not matter what order we replace these operands in.
    // You will always end up with the same IR, and this is guaranteed.
    return std::tie(DFSIn, DFSOut, LocalNum, Val, U) <
           std::tie(Other.DFSIn, Other.DFSOut, Other.LocalNum, Other.Val,
                    Other.U);
  }
};

void NewGVN::convertDenseToDFSOrdered(
    CongruenceClass::MemberSet &Dense,
    SmallVectorImpl<ValueDFS> &DFSOrderedSet) {
  for (auto D : Dense) {
    // First add the value.
    BasicBlock *BB = getBlockForValue(D);
    // Constants are handled prior to ever calling this function, so
    // we should only be left with instructions as members.
    assert(BB && "Should have figured out a basic block for value");
    ValueDFS VD;

    std::pair<int, int> DFSPair = DFSDomMap[BB];
    assert(DFSPair.first != -1 && DFSPair.second != -1 && "Invalid DFS Pair");
    VD.DFSIn = DFSPair.first;
    VD.DFSOut = DFSPair.second;
    VD.Val = D;
    // If it's an instruction, use the real local dfs number.
    if (auto *I = dyn_cast<Instruction>(D))
      VD.LocalNum = InstrDFS[I];
    else
      llvm_unreachable("Should have been an instruction");

    DFSOrderedSet.emplace_back(VD);

    // Now add the users.
    for (auto &U : D->uses()) {
      if (auto *I = dyn_cast<Instruction>(U.getUser())) {
        ValueDFS VD;
        // Put the phi node uses in the incoming block.
        BasicBlock *IBlock;
        if (auto *P = dyn_cast<PHINode>(I)) {
          IBlock = P->getIncomingBlock(U);
          // Make phi node users appear last in the incoming block
          // they are from.
          VD.LocalNum = InstrDFS.size() + 1;
        } else {
          IBlock = I->getParent();
          VD.LocalNum = InstrDFS[I];
        }
        std::pair<int, int> DFSPair = DFSDomMap[IBlock];
        VD.DFSIn = DFSPair.first;
        VD.DFSOut = DFSPair.second;
        VD.U = &U;
        DFSOrderedSet.emplace_back(VD);
      }
    }
  }
}

static void patchReplacementInstruction(Instruction *I, Value *Repl) {
  // Patch the replacement so that it is not more restrictive than the value
  // being replaced.
  auto *Op = dyn_cast<BinaryOperator>(I);
  auto *ReplOp = dyn_cast<BinaryOperator>(Repl);

  if (Op && ReplOp)
    ReplOp->andIRFlags(Op);

  if (auto *ReplInst = dyn_cast<Instruction>(Repl)) {
    // FIXME: If both the original and replacement value are part of the
    // same control-flow region (meaning that the execution of one
    // guarentees the executation of the other), then we can combine the
    // noalias scopes here and do better than the general conservative
    // answer used in combineMetadata().

    // In general, GVN unifies expressions over different control-flow
    // regions, and so we need a conservative combination of the noalias
    // scopes.
    unsigned KnownIDs[] = {
        LLVMContext::MD_tbaa,           LLVMContext::MD_alias_scope,
        LLVMContext::MD_noalias,        LLVMContext::MD_range,
        LLVMContext::MD_fpmath,         LLVMContext::MD_invariant_load,
        LLVMContext::MD_invariant_group};
    combineMetadata(ReplInst, I, KnownIDs);
  }
}

static void patchAndReplaceAllUsesWith(Instruction *I, Value *Repl) {
  patchReplacementInstruction(I, Repl);
  I->replaceAllUsesWith(Repl);
}

void NewGVN::deleteInstructionsInBlock(BasicBlock *BB) {
  DEBUG(dbgs() << "  BasicBlock Dead:" << *BB);
  ++NumGVNBlocksDeleted;

  // Check to see if there are non-terminating instructions to delete.
  if (isa<TerminatorInst>(BB->begin()))
    return;

  // Delete the instructions backwards, as it has a reduced likelihood of having
  // to update as many def-use and use-def chains. Start after the terminator.
  auto StartPoint = BB->rbegin();
  ++StartPoint;
  // Note that we explicitly recalculate BB->rend() on each iteration,
  // as it may change when we remove the first instruction.
  for (BasicBlock::reverse_iterator I(StartPoint); I != BB->rend();) {
    Instruction &Inst = *I++;
    if (!Inst.use_empty())
      Inst.replaceAllUsesWith(UndefValue::get(Inst.getType()));
    if (isa<LandingPadInst>(Inst))
      continue;

    Inst.eraseFromParent();
    ++NumGVNInstrDeleted;
  }
}

void NewGVN::markInstructionForDeletion(Instruction *I) {
  DEBUG(dbgs() << "Marking " << *I << " for deletion\n");
  InstructionsToErase.insert(I);
}

void NewGVN::replaceInstruction(Instruction *I, Value *V) {

  DEBUG(dbgs() << "Replacing " << *I << " with " << *V << "\n");
  patchAndReplaceAllUsesWith(I, V);
  // We save the actual erasing to avoid invalidating memory
  // dependencies until we are done with everything.
  markInstructionForDeletion(I);
}

namespace {

// This is a stack that contains both the value and dfs info of where
// that value is valid.
class ValueDFSStack {
public:
  Value *back() const { return ValueStack.back(); }
  std::pair<int, int> dfs_back() const { return DFSStack.back(); }

  void push_back(Value *V, int DFSIn, int DFSOut) {
    ValueStack.emplace_back(V);
    DFSStack.emplace_back(DFSIn, DFSOut);
  }
  bool empty() const { return DFSStack.empty(); }
  bool isInScope(int DFSIn, int DFSOut) const {
    if (empty())
      return false;
    return DFSIn >= DFSStack.back().first && DFSOut <= DFSStack.back().second;
  }

  void popUntilDFSScope(int DFSIn, int DFSOut) {

    // These two should always be in sync at this point.
    assert(ValueStack.size() == DFSStack.size() &&
           "Mismatch between ValueStack and DFSStack");
    while (
        !DFSStack.empty() &&
        !(DFSIn >= DFSStack.back().first && DFSOut <= DFSStack.back().second)) {
      DFSStack.pop_back();
      ValueStack.pop_back();
    }
  }

private:
  SmallVector<Value *, 8> ValueStack;
  SmallVector<std::pair<int, int>, 8> DFSStack;
};
}

bool NewGVN::eliminateInstructions(Function &F) {
  // This is a non-standard eliminator. The normal way to eliminate is
  // to walk the dominator tree in order, keeping track of available
  // values, and eliminating them.  However, this is mildly
  // pointless. It requires doing lookups on every instruction,
  // regardless of whether we will ever eliminate it.  For
  // instructions part of most singleton congruence classes, we know we
  // will never eliminate them.

  // Instead, this eliminator looks at the congruence classes directly, sorts
  // them into a DFS ordering of the dominator tree, and then we just
  // perform elimination straight on the sets by walking the congruence
  // class member uses in order, and eliminate the ones dominated by the
  // last member.   This is worst case O(E log E) where E = number of
  // instructions in a single congruence class.  In theory, this is all
  // instructions.   In practice, it is much faster, as most instructions are
  // either in singleton congruence classes or can't possibly be eliminated
  // anyway (if there are no overlapping DFS ranges in class).
  // When we find something not dominated, it becomes the new leader
  // for elimination purposes.
  // TODO: If we wanted to be faster, We could remove any members with no
  // overlapping ranges while sorting, as we will never eliminate anything
  // with those members, as they don't dominate anything else in our set.

  bool AnythingReplaced = false;

  // Since we are going to walk the domtree anyway, and we can't guarantee the
  // DFS numbers are updated, we compute some ourselves.
  DT->updateDFSNumbers();

  for (auto &B : F) {
    if (!ReachableBlocks.count(&B)) {
      for (const auto S : successors(&B)) {
        for (auto II = S->begin(); isa<PHINode>(II); ++II) {
          auto &Phi = cast<PHINode>(*II);
          DEBUG(dbgs() << "Replacing incoming value of " << *II << " for block "
                       << getBlockName(&B)
                       << " with undef due to it being unreachable\n");
          for (auto &Operand : Phi.incoming_values())
            if (Phi.getIncomingBlock(Operand) == &B)
              Operand.set(UndefValue::get(Phi.getType()));
        }
      }
    }
    DomTreeNode *Node = DT->getNode(&B);
    if (Node)
      DFSDomMap[&B] = {Node->getDFSNumIn(), Node->getDFSNumOut()};
  }

  for (CongruenceClass *CC : CongruenceClasses) {
    // FIXME: We should eventually be able to replace everything still
    // in the initial class with undef, as they should be unreachable.
    // Right now, initial still contains some things we skip value
    // numbering of (UNREACHABLE's, for example).
    if (CC == InitialClass || CC->Dead)
      continue;
    assert(CC->RepLeader && "We should have had a leader");

    // If this is a leader that is always available, and it's a
    // constant or has no equivalences, just replace everything with
    // it. We then update the congruence class with whatever members
    // are left.
    if (alwaysAvailable(CC->RepLeader)) {
      SmallPtrSet<Value *, 4> MembersLeft;
      for (auto M : CC->Members) {

        Value *Member = M;

        // Void things have no uses we can replace.
        if (Member == CC->RepLeader || Member->getType()->isVoidTy()) {
          MembersLeft.insert(Member);
          continue;
        }

        DEBUG(dbgs() << "Found replacement " << *(CC->RepLeader) << " for "
                     << *Member << "\n");
        // Due to equality propagation, these may not always be
        // instructions, they may be real values.  We don't really
        // care about trying to replace the non-instructions.
        if (auto *I = dyn_cast<Instruction>(Member)) {
          assert(CC->RepLeader != I &&
                 "About to accidentally remove our leader");
          replaceInstruction(I, CC->RepLeader);
          AnythingReplaced = true;

          continue;
        } else {
          MembersLeft.insert(I);
        }
      }
      CC->Members.swap(MembersLeft);

    } else {
      DEBUG(dbgs() << "Eliminating in congruence class " << CC->ID << "\n");
      // If this is a singleton, we can skip it.
      if (CC->Members.size() != 1) {

        // This is a stack because equality replacement/etc may place
        // constants in the middle of the member list, and we want to use
        // those constant values in preference to the current leader, over
        // the scope of those constants.
        ValueDFSStack EliminationStack;

        // Convert the members to DFS ordered sets and then merge them.
        SmallVector<ValueDFS, 8> DFSOrderedSet;
        convertDenseToDFSOrdered(CC->Members, DFSOrderedSet);

        // Sort the whole thing.
        std::sort(DFSOrderedSet.begin(), DFSOrderedSet.end());

        for (auto &VD : DFSOrderedSet) {
          int MemberDFSIn = VD.DFSIn;
          int MemberDFSOut = VD.DFSOut;
          Value *Member = VD.Val;
          Use *MemberUse = VD.U;

          if (Member) {
            // We ignore void things because we can't get a value from them.
            // FIXME: We could actually use this to kill dead stores that are
            // dominated by equivalent earlier stores.
            if (Member->getType()->isVoidTy())
              continue;
          }

          if (EliminationStack.empty()) {
            DEBUG(dbgs() << "Elimination Stack is empty\n");
          } else {
            DEBUG(dbgs() << "Elimination Stack Top DFS numbers are ("
                         << EliminationStack.dfs_back().first << ","
                         << EliminationStack.dfs_back().second << ")\n");
          }

          DEBUG(dbgs() << "Current DFS numbers are (" << MemberDFSIn << ","
                       << MemberDFSOut << ")\n");
          // First, we see if we are out of scope or empty.  If so,
          // and there equivalences, we try to replace the top of
          // stack with equivalences (if it's on the stack, it must
          // not have been eliminated yet).
          // Then we synchronize to our current scope, by
          // popping until we are back within a DFS scope that
          // dominates the current member.
          // Then, what happens depends on a few factors
          // If the stack is now empty, we need to push
          // If we have a constant or a local equivalence we want to
          // start using, we also push.
          // Otherwise, we walk along, processing members who are
          // dominated by this scope, and eliminate them.
          bool ShouldPush =
              Member && (EliminationStack.empty() || isa<Constant>(Member));
          bool OutOfScope =
              !EliminationStack.isInScope(MemberDFSIn, MemberDFSOut);

          if (OutOfScope || ShouldPush) {
            // Sync to our current scope.
            EliminationStack.popUntilDFSScope(MemberDFSIn, MemberDFSOut);
            ShouldPush |= Member && EliminationStack.empty();
            if (ShouldPush) {
              EliminationStack.push_back(Member, MemberDFSIn, MemberDFSOut);
            }
          }

          // If we get to this point, and the stack is empty we must have a use
          // with nothing we can use to eliminate it, just skip it.
          if (EliminationStack.empty())
            continue;

          // Skip the Value's, we only want to eliminate on their uses.
          if (Member)
            continue;
          Value *Result = EliminationStack.back();

          // Don't replace our existing users with ourselves.
          if (MemberUse->get() == Result)
            continue;

          DEBUG(dbgs() << "Found replacement " << *Result << " for "
                       << *MemberUse->get() << " in " << *(MemberUse->getUser())
                       << "\n");

          // If we replaced something in an instruction, handle the patching of
          // metadata.
          if (auto *ReplacedInst = dyn_cast<Instruction>(MemberUse->get()))
            patchReplacementInstruction(ReplacedInst, Result);

          assert(isa<Instruction>(MemberUse->getUser()));
          MemberUse->set(Result);
          AnythingReplaced = true;
        }
      }
    }

    // Cleanup the congruence class.
    SmallPtrSet<Value *, 4> MembersLeft;
    for (Value *Member : CC->Members) {
      if (Member->getType()->isVoidTy()) {
        MembersLeft.insert(Member);
        continue;
      }

      if (auto *MemberInst = dyn_cast<Instruction>(Member)) {
        if (isInstructionTriviallyDead(MemberInst)) {
          // TODO: Don't mark loads of undefs.
          markInstructionForDeletion(MemberInst);
          continue;
        }
      }
      MembersLeft.insert(Member);
    }
    CC->Members.swap(MembersLeft);
  }

  return AnythingReplaced;
}
