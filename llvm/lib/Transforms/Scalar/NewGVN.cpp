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
/// A brief overview of the algorithm: The algorithm is essentially the same as
/// the standard RPO value numbering algorithm (a good reference is the paper
/// "SCC based value numbering" by L. Taylor Simpson) with one major difference:
/// The RPO algorithm proceeds, on every iteration, to process every reachable
/// block and every instruction in that block.  This is because the standard RPO
/// algorithm does not track what things have the same value number, it only
/// tracks what the value number of a given operation is (the mapping is
/// operation -> value number).  Thus, when a value number of an operation
/// changes, it must reprocess everything to ensure all uses of a value number
/// get updated properly.  In constrast, the sparse algorithm we use *also*
/// tracks what operations have a given value number (IE it also tracks the
/// reverse mapping from value number -> operations with that value number), so
/// that it only needs to reprocess the instructions that are affected when
/// something's value number changes.  The rest of the algorithm is devoted to
/// performing symbolic evaluation, forward propagation, and simplification of
/// operations based on the value numbers deduced so far.
///
/// We also do not perform elimination by using any published algorithm.  All
/// published algorithms are O(Instructions). Instead, we use a technique that
/// is O(number of operations with the same value number), enabling us to skip
/// trying to eliminate things that have unique value numbers.
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
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVNExpression.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/MemorySSA.h"
#include "llvm/Transforms/Utils/PredicateInfo.h"
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
STATISTIC(NumGVNLeaderChanges, "Number of leader changes");
STATISTIC(NumGVNSortedLeaderChanges, "Number of sorted leader changes");
STATISTIC(NumGVNAvoidedSortedLeaderChanges,
          "Number of avoided sorted leader changes");
STATISTIC(NumGVNNotMostDominatingLeader,
          "Number of times a member dominated it's new classes' leader");
STATISTIC(NumGVNDeadStores, "Number of redundant/dead stores eliminated");
DEBUG_COUNTER(VNCounter, "newgvn-vn",
              "Controls which instructions are value numbered")
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
  // If this is represented by a store, the value.
  Value *RepStoredValue = nullptr;
  // If this class contains MemoryDefs, what is the represented memory state.
  MemoryAccess *RepMemoryAccess = nullptr;
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

  // The most dominating leader after our current leader, because the member set
  // is not sorted and is expensive to keep sorted all the time.
  std::pair<Value *, unsigned int> NextLeader = {nullptr, ~0U};

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

namespace {
class NewGVN : public FunctionPass {
  DominatorTree *DT;
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;
  AssumptionCache *AC;
  AliasAnalysis *AA;
  MemorySSA *MSSA;
  MemorySSAWalker *MSSAWalker;
  std::unique_ptr<PredicateInfo> PredInfo;
  BumpPtrAllocator ExpressionAllocator;
  ArrayRecycler<Value *> ArgRecycler;

  // Number of function arguments, used by ranking
  unsigned int NumFuncArgs;

  // Congruence class info.

  // This class is called INITIAL in the paper. It is the class everything
  // startsout in, and represents any value. Being an optimistic analysis,
  // anything in the INITIAL class has the value TOP, which is indeterminate and
  // equivalent to everything.
  CongruenceClass *InitialClass;
  std::vector<CongruenceClass *> CongruenceClasses;
  unsigned NextCongruenceNum;

  // Value Mappings.
  DenseMap<Value *, CongruenceClass *> ValueToClass;
  DenseMap<Value *, const Expression *> ValueToExpression;

  // Mapping from predicate info we used to the instructions we used it with.
  // In order to correctly ensure propagation, we must keep track of what
  // comparisons we used, so that when the values of the comparisons change, we
  // propagate the information to the places we used the comparison.
  DenseMap<const Value *, SmallPtrSet<Instruction *, 2>> PredicateToUsers;

  // A table storing which memorydefs/phis represent a memory state provably
  // equivalent to another memory state.
  // We could use the congruence class machinery, but the MemoryAccess's are
  // abstract memory states, so they can only ever be equivalent to each other,
  // and not to constants, etc.
  DenseMap<const MemoryAccess *, CongruenceClass *> MemoryAccessToClass;

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
  // This contains a mapping from Instructions to DFS numbers.
  // The numbering starts at 1. An instruction with DFS number zero
  // means that the instruction is dead.
  DenseMap<const Value *, unsigned> InstrDFS;

  // This contains the mapping DFS numbers to instructions.
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
  const Expression *createExpression(Instruction *);
  const Expression *createBinaryExpression(unsigned, Type *, Value *, Value *);
  PHIExpression *createPHIExpression(Instruction *);
  const VariableExpression *createVariableExpression(Value *);
  const ConstantExpression *createConstantExpression(Constant *);
  const Expression *createVariableOrConstant(Value *V);
  const UnknownExpression *createUnknownExpression(Instruction *);
  const StoreExpression *createStoreExpression(StoreInst *, MemoryAccess *);
  LoadExpression *createLoadExpression(Type *, Value *, LoadInst *,
                                       MemoryAccess *);
  const CallExpression *createCallExpression(CallInst *, MemoryAccess *);
  const AggregateValueExpression *createAggregateValueExpression(Instruction *);
  bool setBasicExpressionInfo(Instruction *, BasicExpression *);

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
  const Expression *performSymbolicEvaluation(Value *);
  const Expression *performSymbolicLoadEvaluation(Instruction *);
  const Expression *performSymbolicStoreEvaluation(Instruction *);
  const Expression *performSymbolicCallEvaluation(Instruction *);
  const Expression *performSymbolicPHIEvaluation(Instruction *);
  const Expression *performSymbolicAggrValueEvaluation(Instruction *);
  const Expression *performSymbolicCmpEvaluation(Instruction *);
  const Expression *performSymbolicPredicateInfoEvaluation(Instruction *);

  // Congruence finding.
  Value *lookupOperandLeader(Value *) const;
  void performCongruenceFinding(Instruction *, const Expression *);
  void moveValueToNewCongruenceClass(Instruction *, CongruenceClass *,
                                     CongruenceClass *);
  bool setMemoryAccessEquivTo(MemoryAccess *From, CongruenceClass *To);
  MemoryAccess *lookupMemoryAccessEquiv(MemoryAccess *) const;
  bool isMemoryAccessTop(const MemoryAccess *) const;

  // Ranking
  unsigned int getRank(const Value *) const;
  bool shouldSwapOperands(const Value *, const Value *) const;

  // Reachability handling.
  void updateReachableEdge(BasicBlock *, BasicBlock *);
  void processOutgoingEdges(TerminatorInst *, BasicBlock *);
  Value *findConditionEquivalence(Value *) const;

  // Elimination.
  struct ValueDFS;
  void convertDenseToDFSOrdered(const CongruenceClass::MemberSet &,
                                SmallVectorImpl<ValueDFS> &);
  void convertDenseToLoadsAndStores(const CongruenceClass::MemberSet &,
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
  void markPredicateUsersTouched(Instruction *);
  void markLeaderChangeTouched(CongruenceClass *CC);
  void addPredicateUsers(const PredicateBase *, Instruction *);

  // Utilities.
  void cleanupTables();
  std::pair<unsigned, unsigned> assignDFSNumbers(BasicBlock *, unsigned);
  void updateProcessedCount(Value *V);
  void verifyMemoryCongruency() const;
  void verifyComparisons(Function &F);
  bool singleReachablePHIPath(const MemoryAccess *, const MemoryAccess *) const;
};
} // end anonymous namespace

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
  bool Result = equalsLoadStoreHelper(*this, Other);
  // Make sure that store vs store includes the value operand.
  if (Result)
    if (const auto *S = dyn_cast<StoreExpression>(&Other))
      if (getStoredValue() != S->getStoredValue())
        return false;
  return Result;
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

  // Filter out unreachable phi operands.
  auto Filtered = make_filter_range(PN->operands(), [&](const Use &U) {
    return ReachableBlocks.count(PN->getIncomingBlock(U));
  });

  std::transform(Filtered.begin(), Filtered.end(), op_inserter(E),
                 [&](const Use &U) -> Value * {
                   // Don't try to transform self-defined phis.
                   if (U == PN)
                     return PN;
                   return lookupOperandLeader(U);
                 });
  return E;
}

// Set basic expression info (Arguments, type, opcode) for Expression
// E from Instruction I in block B.
bool NewGVN::setBasicExpressionInfo(Instruction *I, BasicExpression *E) {
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
    auto Operand = lookupOperandLeader(O);
    AllConstant &= isa<Constant>(Operand);
    return Operand;
  });

  return AllConstant;
}

const Expression *NewGVN::createBinaryExpression(unsigned Opcode, Type *T,
                                                 Value *Arg1, Value *Arg2) {
  auto *E = new (ExpressionAllocator) BasicExpression(2);

  E->setType(T);
  E->setOpcode(Opcode);
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  if (Instruction::isCommutative(Opcode)) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    if (shouldSwapOperands(Arg1, Arg2))
      std::swap(Arg1, Arg2);
  }
  E->op_push_back(lookupOperandLeader(Arg1));
  E->op_push_back(lookupOperandLeader(Arg2));

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

const Expression *NewGVN::createExpression(Instruction *I) {
  auto *E = new (ExpressionAllocator) BasicExpression(I->getNumOperands());

  bool AllConstant = setBasicExpressionInfo(I, E);

  if (I->isCommutative()) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    assert(I->getNumOperands() == 2 && "Unsupported commutative instruction!");
    if (shouldSwapOperands(E->getOperand(0), E->getOperand(1)))
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
    if (shouldSwapOperands(E->getOperand(0), E->getOperand(1))) {
      E->swapOperands(0, 1);
      Predicate = CmpInst::getSwappedPredicate(Predicate);
    }
    E->setOpcode((CI->getOpcode() << 8) | Predicate);
    // TODO: 25% of our time is spent in SimplifyCmpInst with pointer operands
    assert(I->getOperand(0)->getType() == I->getOperand(1)->getType() &&
           "Wrong types on cmp instruction");
    assert((E->getOperand(0)->getType() == I->getOperand(0)->getType() &&
            E->getOperand(1)->getType() == I->getOperand(1)->getType()));
    Value *V = SimplifyCmpInst(Predicate, E->getOperand(0), E->getOperand(1),
                               *DL, TLI, DT, AC);
    if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
      return SimplifiedE;
  } else if (isa<SelectInst>(I)) {
    if (isa<Constant>(E->getOperand(0)) ||
        E->getOperand(0) == E->getOperand(1)) {
      assert(E->getOperand(1)->getType() == I->getOperand(1)->getType() &&
             E->getOperand(2)->getType() == I->getOperand(2)->getType());
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
NewGVN::createAggregateValueExpression(Instruction *I) {
  if (auto *II = dyn_cast<InsertValueInst>(I)) {
    auto *E = new (ExpressionAllocator)
        AggregateValueExpression(I->getNumOperands(), II->getNumIndices());
    setBasicExpressionInfo(I, E);
    E->allocateIntOperands(ExpressionAllocator);
    std::copy(II->idx_begin(), II->idx_end(), int_op_inserter(E));
    return E;
  } else if (auto *EI = dyn_cast<ExtractValueInst>(I)) {
    auto *E = new (ExpressionAllocator)
        AggregateValueExpression(I->getNumOperands(), EI->getNumIndices());
    setBasicExpressionInfo(EI, E);
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

const Expression *NewGVN::createVariableOrConstant(Value *V) {
  if (auto *C = dyn_cast<Constant>(V))
    return createConstantExpression(C);
  return createVariableExpression(V);
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
                                                   MemoryAccess *HV) {
  // FIXME: Add operand bundles for calls.
  auto *E =
      new (ExpressionAllocator) CallExpression(CI->getNumOperands(), CI, HV);
  setBasicExpressionInfo(CI, E);
  return E;
}

// See if we have a congruence class and leader for this operand, and if so,
// return it. Otherwise, return the operand itself.
Value *NewGVN::lookupOperandLeader(Value *V) const {
  CongruenceClass *CC = ValueToClass.lookup(V);
  if (CC) {
    // Everything in INITIAL is represneted by undef, as it can be any value.
    // We do have to make sure we get the type right though, so we can't set the
    // RepLeader to undef.
    if (CC == InitialClass)
      return UndefValue::get(V->getType());
    return CC->RepStoredValue ? CC->RepStoredValue : CC->RepLeader;
  }

  return V;
}

MemoryAccess *NewGVN::lookupMemoryAccessEquiv(MemoryAccess *MA) const {
  auto *CC = MemoryAccessToClass.lookup(MA);
  if (CC && CC->RepMemoryAccess)
    return CC->RepMemoryAccess;
  // FIXME: We need to audit all the places that current set a nullptr To, and
  // fix them.  There should always be *some* congruence class, even if it is
  // singular.  Right now, we don't bother setting congruence classes for
  // anything but stores, which means we have to return the original access
  // here.  Otherwise, this should be unreachable.
  return MA;
}

// Return true if the MemoryAccess is really equivalent to everything. This is
// equivalent to the lattice value "TOP" in most lattices.  This is the initial
// state of all memory accesses.
bool NewGVN::isMemoryAccessTop(const MemoryAccess *MA) const {
  return MemoryAccessToClass.lookup(MA) == InitialClass;
}

LoadExpression *NewGVN::createLoadExpression(Type *LoadType, Value *PointerOp,
                                             LoadInst *LI, MemoryAccess *DA) {
  auto *E = new (ExpressionAllocator) LoadExpression(1, LI, DA);
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(LoadType);

  // Give store and loads same opcode so they value number together.
  E->setOpcode(0);
  E->op_push_back(lookupOperandLeader(PointerOp));
  if (LI)
    E->setAlignment(LI->getAlignment());

  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load).
  return E;
}

const StoreExpression *NewGVN::createStoreExpression(StoreInst *SI,
                                                     MemoryAccess *DA) {
  auto *StoredValueLeader = lookupOperandLeader(SI->getValueOperand());
  auto *E = new (ExpressionAllocator)
      StoreExpression(SI->getNumOperands(), SI, StoredValueLeader, DA);
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(SI->getValueOperand()->getType());

  // Give store and loads same opcode so they value number together.
  E->setOpcode(0);
  E->op_push_back(lookupOperandLeader(SI->getPointerOperand()));

  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load).
  return E;
}

const Expression *NewGVN::performSymbolicStoreEvaluation(Instruction *I) {
  // Unlike loads, we never try to eliminate stores, so we do not check if they
  // are simple and avoid value numbering them.
  auto *SI = cast<StoreInst>(I);
  MemoryAccess *StoreAccess = MSSA->getMemoryAccess(SI);
  // Get the expression, if any, for the RHS of the MemoryDef.
  MemoryAccess *StoreRHS = lookupMemoryAccessEquiv(
      cast<MemoryDef>(StoreAccess)->getDefiningAccess());
  // If we are defined by ourselves, use the live on entry def.
  if (StoreRHS == StoreAccess)
    StoreRHS = MSSA->getLiveOnEntryDef();

  if (SI->isSimple()) {
    // See if we are defined by a previous store expression, it already has a
    // value, and it's the same value as our current store. FIXME: Right now, we
    // only do this for simple stores, we should expand to cover memcpys, etc.
    const Expression *OldStore = createStoreExpression(SI, StoreRHS);
    CongruenceClass *CC = ExpressionToClass.lookup(OldStore);
    // Basically, check if the congruence class the store is in is defined by a
    // store that isn't us, and has the same value.  MemorySSA takes care of
    // ensuring the store has the same memory state as us already.
    // The RepStoredValue gets nulled if all the stores disappear in a class, so
    // we don't need to check if the class contains a store besides us.
    if (CC && CC->RepStoredValue == lookupOperandLeader(SI->getValueOperand()))
      return createStoreExpression(SI, StoreRHS);
    // Also check if our value operand is defined by a load of the same memory
    // location, and the memory state is the same as it was then
    // (otherwise, it could have been overwritten later. See test32 in
    // transforms/DeadStoreElimination/simple.ll)
    if (LoadInst *LI = dyn_cast<LoadInst>(SI->getValueOperand())) {
      if ((lookupOperandLeader(LI->getPointerOperand()) ==
           lookupOperandLeader(SI->getPointerOperand())) &&
          (lookupMemoryAccessEquiv(
               MSSA->getMemoryAccess(LI)->getDefiningAccess()) == StoreRHS))
        return createVariableExpression(LI);
    }
  }
  return createStoreExpression(SI, StoreAccess);
}

const Expression *NewGVN::performSymbolicLoadEvaluation(Instruction *I) {
  auto *LI = cast<LoadInst>(I);

  // We can eliminate in favor of non-simple loads, but we won't be able to
  // eliminate the loads themselves.
  if (!LI->isSimple())
    return nullptr;

  Value *LoadAddressLeader = lookupOperandLeader(LI->getPointerOperand());
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
                           lookupMemoryAccessEquiv(DefiningAccess));
  return E;
}

const Expression *
NewGVN::performSymbolicPredicateInfoEvaluation(Instruction *I) {
  auto *PI = PredInfo->getPredicateInfoFor(I);
  if (!PI)
    return nullptr;

  DEBUG(dbgs() << "Found predicate info from instruction !\n");

  auto *PWC = dyn_cast<PredicateWithCondition>(PI);
  if (!PWC)
    return nullptr;

  auto *CopyOf = I->getOperand(0);
  auto *Cond = PWC->Condition;

  // If this a copy of the condition, it must be either true or false depending
  // on the predicate info type and edge
  if (CopyOf == Cond) {
    // We should not need to add predicate users because the predicate info is
    // already a use of this operand.
    if (isa<PredicateAssume>(PI))
      return createConstantExpression(ConstantInt::getTrue(Cond->getType()));
    if (auto *PBranch = dyn_cast<PredicateBranch>(PI)) {
      if (PBranch->TrueEdge)
        return createConstantExpression(ConstantInt::getTrue(Cond->getType()));
      return createConstantExpression(ConstantInt::getFalse(Cond->getType()));
    }
    if (auto *PSwitch = dyn_cast<PredicateSwitch>(PI))
      return createConstantExpression(cast<Constant>(PSwitch->CaseValue));
  }

  // Not a copy of the condition, so see what the predicates tell us about this
  // value.  First, though, we check to make sure the value is actually a copy
  // of one of the condition operands. It's possible, in certain cases, for it
  // to be a copy of a predicateinfo copy. In particular, if two branch
  // operations use the same condition, and one branch dominates the other, we
  // will end up with a copy of a copy.  This is currently a small deficiency in
  // predicateinfo.  What will end up happening here is that we will value
  // number both copies the same anyway.

  // Everything below relies on the condition being a comparison.
  auto *Cmp = dyn_cast<CmpInst>(Cond);
  if (!Cmp)
    return nullptr;

  if (CopyOf != Cmp->getOperand(0) && CopyOf != Cmp->getOperand(1)) {
    DEBUG(dbgs() << "Copy is not of any condition operands!");
    return nullptr;
  }
  Value *FirstOp = lookupOperandLeader(Cmp->getOperand(0));
  Value *SecondOp = lookupOperandLeader(Cmp->getOperand(1));
  bool SwappedOps = false;
  // Sort the ops
  if (shouldSwapOperands(FirstOp, SecondOp)) {
    std::swap(FirstOp, SecondOp);
    SwappedOps = true;
  }
  CmpInst::Predicate Predicate =
      SwappedOps ? Cmp->getSwappedPredicate() : Cmp->getPredicate();

  if (isa<PredicateAssume>(PI)) {
    // If the comparison is true when the operands are equal, then we know the
    // operands are equal, because assumes must always be true.
    if (CmpInst::isTrueWhenEqual(Predicate)) {
      addPredicateUsers(PI, I);
      return createVariableOrConstant(FirstOp);
    }
  }
  if (const auto *PBranch = dyn_cast<PredicateBranch>(PI)) {
    // If we are *not* a copy of the comparison, we may equal to the other
    // operand when the predicate implies something about equality of
    // operations.  In particular, if the comparison is true/false when the
    // operands are equal, and we are on the right edge, we know this operation
    // is equal to something.
    if ((PBranch->TrueEdge && Predicate == CmpInst::ICMP_EQ) ||
        (!PBranch->TrueEdge && Predicate == CmpInst::ICMP_NE)) {
      addPredicateUsers(PI, I);
      return createVariableOrConstant(FirstOp);
    }
    // Handle the special case of floating point.
    if (((PBranch->TrueEdge && Predicate == CmpInst::FCMP_OEQ) ||
         (!PBranch->TrueEdge && Predicate == CmpInst::FCMP_UNE)) &&
        isa<ConstantFP>(FirstOp) && !cast<ConstantFP>(FirstOp)->isZero()) {
      addPredicateUsers(PI, I);
      return createConstantExpression(cast<Constant>(FirstOp));
    }
  }
  return nullptr;
}

// Evaluate read only and pure calls, and create an expression result.
const Expression *NewGVN::performSymbolicCallEvaluation(Instruction *I) {
  auto *CI = cast<CallInst>(I);
  if (auto *II = dyn_cast<IntrinsicInst>(I)) {
    // Instrinsics with the returned attribute are copies of arguments.
    if (auto *ReturnedValue = II->getReturnedArgOperand()) {
      if (II->getIntrinsicID() == Intrinsic::ssa_copy)
        if (const auto *Result = performSymbolicPredicateInfoEvaluation(I))
          return Result;
      return createVariableOrConstant(ReturnedValue);
    }
  }
  if (AA->doesNotAccessMemory(CI)) {
    return createCallExpression(CI, nullptr);
  } else if (AA->onlyReadsMemory(CI)) {
    MemoryAccess *DefiningAccess = MSSAWalker->getClobberingMemoryAccess(CI);
    return createCallExpression(CI, lookupMemoryAccessEquiv(DefiningAccess));
  }
  return nullptr;
}

// Update the memory access equivalence table to say that From is equal to To,
// and return true if this is different from what already existed in the table.
// FIXME: We need to audit all the places that current set a nullptr To, and fix
// them. There should always be *some* congruence class, even if it is singular.
bool NewGVN::setMemoryAccessEquivTo(MemoryAccess *From, CongruenceClass *To) {
  DEBUG(dbgs() << "Setting " << *From);
  if (To) {
    DEBUG(dbgs() << " equivalent to congruence class ");
    DEBUG(dbgs() << To->ID << " with current memory access leader ");
    DEBUG(dbgs() << *To->RepMemoryAccess);
  } else {
    DEBUG(dbgs() << " equivalent to itself");
  }
  DEBUG(dbgs() << "\n");

  auto LookupResult = MemoryAccessToClass.find(From);
  bool Changed = false;
  // If it's already in the table, see if the value changed.
  if (LookupResult != MemoryAccessToClass.end()) {
    if (To && LookupResult->second != To) {
      // It wasn't equivalent before, and now it is.
      LookupResult->second = To;
      Changed = true;
    } else if (!To) {
      // It used to be equivalent to something, and now it's not.
      MemoryAccessToClass.erase(LookupResult);
      Changed = true;
    }
  } else {
    assert(!To &&
           "Memory equivalence should never change from nothing to something");
  }

  return Changed;
}
// Evaluate PHI nodes symbolically, and create an expression result.
const Expression *NewGVN::performSymbolicPHIEvaluation(Instruction *I) {
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
    return createVariableOrConstant(AllSameValue);
  }
  return E;
}

const Expression *NewGVN::performSymbolicAggrValueEvaluation(Instruction *I) {
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
        return createBinaryExpression(
            Opcode, EI->getType(), II->getArgOperand(0), II->getArgOperand(1));
      }
    }
  }

  return createAggregateValueExpression(I);
}
const Expression *NewGVN::performSymbolicCmpEvaluation(Instruction *I) {
  auto *CI = dyn_cast<CmpInst>(I);
  // See if our operands are equal to those of a previous predicate, and if so,
  // if it implies true or false.
  auto Op0 = lookupOperandLeader(CI->getOperand(0));
  auto Op1 = lookupOperandLeader(CI->getOperand(1));
  auto OurPredicate = CI->getPredicate();
  if (shouldSwapOperands(Op0, Op1)) {
    std::swap(Op0, Op1);
    OurPredicate = CI->getSwappedPredicate();
  }

  // Avoid processing the same info twice
  const PredicateBase *LastPredInfo = nullptr;
  // See if we know something about the comparison itself, like it is the target
  // of an assume.
  auto *CmpPI = PredInfo->getPredicateInfoFor(I);
  if (dyn_cast_or_null<PredicateAssume>(CmpPI))
    return createConstantExpression(ConstantInt::getTrue(CI->getType()));

  if (Op0 == Op1) {
    // This condition does not depend on predicates, no need to add users
    if (CI->isTrueWhenEqual())
      return createConstantExpression(ConstantInt::getTrue(CI->getType()));
    else if (CI->isFalseWhenEqual())
      return createConstantExpression(ConstantInt::getFalse(CI->getType()));
  }

  // NOTE: Because we are comparing both operands here and below, and using
  // previous comparisons, we rely on fact that predicateinfo knows to mark
  // comparisons that use renamed operands as users of the earlier comparisons.
  // It is *not* enough to just mark predicateinfo renamed operands as users of
  // the earlier comparisons, because the *other* operand may have changed in a
  // previous iteration.
  // Example:
  // icmp slt %a, %b
  // %b.0 = ssa.copy(%b)
  // false branch:
  // icmp slt %c, %b.0

  // %c and %a may start out equal, and thus, the code below will say the second
  // %icmp is false.  c may become equal to something else, and in that case the
  // %second icmp *must* be reexamined, but would not if only the renamed
  // %operands are considered users of the icmp.

  // *Currently* we only check one level of comparisons back, and only mark one
  // level back as touched when changes appen .  If you modify this code to look
  // back farther through comparisons, you *must* mark the appropriate
  // comparisons as users in PredicateInfo.cpp, or you will cause bugs.  See if
  // we know something just from the operands themselves

  // See if our operands have predicate info, so that we may be able to derive
  // something from a previous comparison.
  for (const auto &Op : CI->operands()) {
    auto *PI = PredInfo->getPredicateInfoFor(Op);
    if (const auto *PBranch = dyn_cast_or_null<PredicateBranch>(PI)) {
      if (PI == LastPredInfo)
        continue;
      LastPredInfo = PI;

      // TODO: Along the false edge, we may know more things too, like icmp of
      // same operands is false.
      // TODO: We only handle actual comparison conditions below, not and/or.
      auto *BranchCond = dyn_cast<CmpInst>(PBranch->Condition);
      if (!BranchCond)
        continue;
      auto *BranchOp0 = lookupOperandLeader(BranchCond->getOperand(0));
      auto *BranchOp1 = lookupOperandLeader(BranchCond->getOperand(1));
      auto BranchPredicate = BranchCond->getPredicate();
      if (shouldSwapOperands(BranchOp0, BranchOp1)) {
        std::swap(BranchOp0, BranchOp1);
        BranchPredicate = BranchCond->getSwappedPredicate();
      }
      if (BranchOp0 == Op0 && BranchOp1 == Op1) {
        if (PBranch->TrueEdge) {
          // If we know the previous predicate is true and we are in the true
          // edge then we may be implied true or false.
          if (CmpInst::isImpliedTrueByMatchingCmp(OurPredicate,
                                                  BranchPredicate)) {
            addPredicateUsers(PI, I);
            return createConstantExpression(
                ConstantInt::getTrue(CI->getType()));
          }

          if (CmpInst::isImpliedFalseByMatchingCmp(OurPredicate,
                                                   BranchPredicate)) {
            addPredicateUsers(PI, I);
            return createConstantExpression(
                ConstantInt::getFalse(CI->getType()));
          }

        } else {
          // Just handle the ne and eq cases, where if we have the same
          // operands, we may know something.
          if (BranchPredicate == OurPredicate) {
            addPredicateUsers(PI, I);
            // Same predicate, same ops,we know it was false, so this is false.
            return createConstantExpression(
                ConstantInt::getFalse(CI->getType()));
          } else if (BranchPredicate ==
                     CmpInst::getInversePredicate(OurPredicate)) {
            addPredicateUsers(PI, I);
            // Inverse predicate, we know the other was false, so this is true.
            // FIXME: Double check this
            return createConstantExpression(
                ConstantInt::getTrue(CI->getType()));
          }
        }
      }
    }
  }
  // Create expression will take care of simplifyCmpInst
  return createExpression(I);
}

// Substitute and symbolize the value before value numbering.
const Expression *NewGVN::performSymbolicEvaluation(Value *V) {
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
      E = performSymbolicAggrValueEvaluation(I);
      break;
    case Instruction::PHI:
      E = performSymbolicPHIEvaluation(I);
      break;
    case Instruction::Call:
      E = performSymbolicCallEvaluation(I);
      break;
    case Instruction::Store:
      E = performSymbolicStoreEvaluation(I);
      break;
    case Instruction::Load:
      E = performSymbolicLoadEvaluation(I);
      break;
    case Instruction::BitCast: {
      E = createExpression(I);
    } break;
    case Instruction::ICmp:
    case Instruction::FCmp: {
      E = performSymbolicCmpEvaluation(I);
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
      E = createExpression(I);
      break;
    default:
      return nullptr;
    }
  }
  return E;
}

void NewGVN::markUsersTouched(Value *V) {
  // Now mark the users as touched.
  for (auto *User : V->users()) {
    assert(isa<Instruction>(User) && "Use of value not within an instruction?");
    TouchedInstructions.set(InstrDFS.lookup(User));
  }
}

void NewGVN::markMemoryUsersTouched(MemoryAccess *MA) {
  for (auto U : MA->users()) {
    if (auto *MUD = dyn_cast<MemoryUseOrDef>(U))
      TouchedInstructions.set(InstrDFS.lookup(MUD->getMemoryInst()));
    else
      TouchedInstructions.set(InstrDFS.lookup(U));
  }
}

// Add I to the set of users of a given predicate.
void NewGVN::addPredicateUsers(const PredicateBase *PB, Instruction *I) {
  if (auto *PBranch = dyn_cast<PredicateBranch>(PB))
    PredicateToUsers[PBranch->Condition].insert(I);
  else if (auto *PAssume = dyn_cast<PredicateBranch>(PB))
    PredicateToUsers[PAssume->Condition].insert(I);
}

// Touch all the predicates that depend on this instruction.
void NewGVN::markPredicateUsersTouched(Instruction *I) {
  const auto Result = PredicateToUsers.find(I);
  if (Result != PredicateToUsers.end())
    for (auto *User : Result->second)
      TouchedInstructions.set(InstrDFS.lookup(User));
}

// Touch the instructions that need to be updated after a congruence class has a
// leader change, and mark changed values.
void NewGVN::markLeaderChangeTouched(CongruenceClass *CC) {
  for (auto M : CC->Members) {
    if (auto *I = dyn_cast<Instruction>(M))
      TouchedInstructions.set(InstrDFS.lookup(I));
    LeaderChanges.insert(M);
  }
}

// Move a value, currently in OldClass, to be part of NewClass
// Update OldClass for the move (including changing leaders, etc)
void NewGVN::moveValueToNewCongruenceClass(Instruction *I,
                                           CongruenceClass *OldClass,
                                           CongruenceClass *NewClass) {
  DEBUG(dbgs() << "New congruence class for " << I << " is " << NewClass->ID
               << "\n");

  if (I == OldClass->NextLeader.first)
    OldClass->NextLeader = {nullptr, ~0U};

  // It's possible, though unlikely, for us to discover equivalences such
  // that the current leader does not dominate the old one.
  // This statistic tracks how often this happens.
  // We assert on phi nodes when this happens, currently, for debugging, because
  // we want to make sure we name phi node cycles properly.
  if (isa<Instruction>(NewClass->RepLeader) && NewClass->RepLeader &&
      I != NewClass->RepLeader &&
      DT->properlyDominates(
          I->getParent(),
          cast<Instruction>(NewClass->RepLeader)->getParent())) {
    ++NumGVNNotMostDominatingLeader;
    assert(!isa<PHINode>(I) &&
           "New class for instruction should not be dominated by instruction");
  }

  if (NewClass->RepLeader != I) {
    auto DFSNum = InstrDFS.lookup(I);
    if (DFSNum < NewClass->NextLeader.second)
      NewClass->NextLeader = {I, DFSNum};
  }

  OldClass->Members.erase(I);
  NewClass->Members.insert(I);
  MemoryAccess *StoreAccess = nullptr;
  if (auto *SI = dyn_cast<StoreInst>(I)) {
    StoreAccess = MSSA->getMemoryAccess(SI);
    --OldClass->StoreCount;
    assert(OldClass->StoreCount >= 0);
    ++NewClass->StoreCount;
    assert(NewClass->StoreCount > 0);
    if (!NewClass->RepMemoryAccess) {
      // If we don't have a representative memory access, it better be the only
      // store in there.
      assert(NewClass->StoreCount == 1);
      NewClass->RepMemoryAccess = StoreAccess;
    }
    setMemoryAccessEquivTo(StoreAccess, NewClass);
  }

  ValueToClass[I] = NewClass;
  // See if we destroyed the class or need to swap leaders.
  if (OldClass->Members.empty() && OldClass != InitialClass) {
    if (OldClass->DefiningExpr) {
      OldClass->Dead = true;
      DEBUG(dbgs() << "Erasing expression " << OldClass->DefiningExpr
                   << " from table\n");
      ExpressionToClass.erase(OldClass->DefiningExpr);
    }
  } else if (OldClass->RepLeader == I) {
    // When the leader changes, the value numbering of
    // everything may change due to symbolization changes, so we need to
    // reprocess.
    DEBUG(dbgs() << "Leader change!\n");
    ++NumGVNLeaderChanges;
    // Destroy the stored value if there are no more stores to represent it.
    if (OldClass->StoreCount == 0) {
      if (OldClass->RepStoredValue != nullptr)
        OldClass->RepStoredValue = nullptr;
      if (OldClass->RepMemoryAccess != nullptr)
        OldClass->RepMemoryAccess = nullptr;
    }

    // If we destroy the old access leader, we have to effectively destroy the
    // congruence class.  When it comes to scalars, anything with the same value
    // is as good as any other.  That means that one leader is as good as
    // another, and as long as you have some leader for the value, you are
    // good.. When it comes to *memory states*, only one particular thing really
    // represents the definition of a given memory state.  Once it goes away, we
    // need to re-evaluate which pieces of memory are really still
    // equivalent. The best way to do this is to re-value number things.  The
    // only way to really make that happen is to destroy the rest of the class.
    // In order to effectively destroy the class, we reset ExpressionToClass for
    // each by using the ValueToExpression mapping.  The members later get
    // marked as touched due to the leader change.  We will create new
    // congruence classes, and the pieces that are still equivalent will end
    // back together in a new class.  If this becomes too expensive, it is
    // possible to use a versioning scheme for the congruence classes to avoid
    // the expressions finding this old class.
    if (OldClass->StoreCount > 0 && OldClass->RepMemoryAccess == StoreAccess) {
      DEBUG(dbgs() << "Kicking everything out of class " << OldClass->ID
                   << " because memory access leader changed");
      for (auto Member : OldClass->Members)
        ExpressionToClass.erase(ValueToExpression.lookup(Member));
    }

    // We don't need to sort members if there is only 1, and we don't care about
    // sorting the INITIAL class because everything either gets out of it or is
    // unreachable.
    if (OldClass->Members.size() == 1 || OldClass == InitialClass) {
      OldClass->RepLeader = *(OldClass->Members.begin());
    } else if (OldClass->NextLeader.first) {
      ++NumGVNAvoidedSortedLeaderChanges;
      OldClass->RepLeader = OldClass->NextLeader.first;
      OldClass->NextLeader = {nullptr, ~0U};
    } else {
      ++NumGVNSortedLeaderChanges;
      // TODO: If this ends up to slow, we can maintain a dual structure for
      // member testing/insertion, or keep things mostly sorted, and sort only
      // here, or ....
      std::pair<Value *, unsigned> MinDFS = {nullptr, ~0U};
      for (const auto X : OldClass->Members) {
        auto DFSNum = InstrDFS.lookup(X);
        if (DFSNum < MinDFS.second)
          MinDFS = {X, DFSNum};
      }
      OldClass->RepLeader = MinDFS.first;
    }
    markLeaderChangeTouched(OldClass);
  }
}

// Perform congruence finding on a given value numbering expression.
void NewGVN::performCongruenceFinding(Instruction *I, const Expression *E) {
  ValueToExpression[I] = E;
  // This is guaranteed to return something, since it will at least find
  // TOP.

  CongruenceClass *IClass = ValueToClass[I];
  assert(IClass && "Should have found a IClass");
  // Dead classes should have been eliminated from the mapping.
  assert(!IClass->Dead && "Found a dead class");

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
        NewClass->RepLeader = SI;
        NewClass->RepStoredValue = lookupOperandLeader(SI->getValueOperand());
        // The RepMemoryAccess field will be filled in properly by the
        // moveValueToNewCongruenceClass call.
      } else {
        NewClass->RepLeader = I;
      }
      assert(!isa<VariableExpression>(E) &&
             "VariableExpression should have been handled already");

      EClass = NewClass;
      DEBUG(dbgs() << "Created new congruence class for " << *I
                   << " using expression " << *E << " at " << NewClass->ID
                   << " and leader " << *(NewClass->RepLeader));
      if (NewClass->RepStoredValue)
        DEBUG(dbgs() << " and stored value " << *(NewClass->RepStoredValue));
      DEBUG(dbgs() << "\n");
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
  bool ClassChanged = IClass != EClass;
  bool LeaderChanged = LeaderChanges.erase(I);
  if (ClassChanged || LeaderChanged) {
    DEBUG(dbgs() << "Found class " << EClass->ID << " for expression " << E
                 << "\n");

    if (ClassChanged)
      moveValueToNewCongruenceClass(I, IClass, EClass);
    markUsersTouched(I);
    if (MemoryAccess *MA = MSSA->getMemoryAccess(I))
      markMemoryUsersTouched(MA);
    if (auto *CI = dyn_cast<CmpInst>(I))
      markPredicateUsersTouched(CI);
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
        TouchedInstructions.set(InstrDFS.lookup(MemPhi));

      auto BI = To->begin();
      while (isa<PHINode>(BI)) {
        TouchedInstructions.set(InstrDFS.lookup(&*BI));
        ++BI;
      }
    }
  }
}

// Given a predicate condition (from a switch, cmp, or whatever) and a block,
// see if we know some constant value for it already.
Value *NewGVN::findConditionEquivalence(Value *Cond) const {
  auto Result = lookupOperandLeader(Cond);
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
    Value *CondEvaluated = findConditionEquivalence(Cond);
    if (!CondEvaluated) {
      if (auto *I = dyn_cast<Instruction>(Cond)) {
        const Expression *E = createExpression(I);
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
    Value *CondEvaluated = findConditionEquivalence(SwitchCond);
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
// congruence class. The leader of INITIAL is the undetermined value `TOP`.
// When the algorithm has finished, values still in INITIAL are unreachable.
void NewGVN::initializeCongruenceClasses(Function &F) {
  // FIXME now i can't remember why this is 2
  NextCongruenceNum = 2;
  // Initialize all other instructions to be in INITIAL class.
  CongruenceClass::MemberSet InitialValues;
  InitialClass = createCongruenceClass(nullptr, nullptr);
  InitialClass->RepMemoryAccess = MSSA->getLiveOnEntryDef();
  for (auto &B : F) {
    if (auto *MP = MSSA->getMemoryAccess(&B))
      MemoryAccessToClass[MP] = InitialClass;

    for (auto &I : B) {
      // Don't insert void terminators into the class. We don't value number
      // them, and they just end up sitting in INITIAL.
      if (isa<TerminatorInst>(I) && I.getType()->isVoidTy())
        continue;
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
        MemoryAccessToClass[MSSA->getMemoryAccess(&I)] = InitialClass;
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
  InstrDFS.clear();
  InstructionsToErase.clear();
  DFSToInstr.clear();
  BlockInstRange.clear();
  TouchedInstructions.clear();
  DominatedInstRange.clear();
  MemoryAccessToClass.clear();
  PredicateToUsers.clear();
}

std::pair<unsigned, unsigned> NewGVN::assignDFSNumbers(BasicBlock *B,
                                                       unsigned Start) {
  unsigned End = Start;
  if (MemoryAccess *MemPhi = MSSA->getMemoryAccess(B)) {
    InstrDFS[MemPhi] = End++;
    DFSToInstr.emplace_back(MemPhi);
  }

  for (auto &I : *B) {
    // There's no need to call isInstructionTriviallyDead more than once on
    // an instruction. Therefore, once we know that an instruction is dead
    // we change its DFS number so that it doesn't get value numbered.
    if (isInstructionTriviallyDead(&I, TLI)) {
      InstrDFS[&I] = 0;
      DEBUG(dbgs() << "Skipping trivially dead instruction " << I << "\n");
      markInstructionForDeletion(&I);
      continue;
    }

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
    ++ProcessedCount[V];
    assert(ProcessedCount[V] < 100 &&
           "Seem to have processed the same Value a lot");
  }
#endif
}
// Evaluate MemoryPhi nodes symbolically, just like PHI nodes
void NewGVN::valueNumberMemoryPhi(MemoryPhi *MP) {
  // If all the arguments are the same, the MemoryPhi has the same value as the
  // argument.
  // Filter out unreachable blocks and self phis from our operands.
  auto Filtered = make_filter_range(MP->operands(), [&](const Use &U) {
    return lookupMemoryAccessEquiv(cast<MemoryAccess>(U)) != MP &&
           !isMemoryAccessTop(cast<MemoryAccess>(U)) &&
           ReachableBlocks.count(MP->getIncomingBlock(U));
  });
  // If all that is left is nothing, our memoryphi is undef. We keep it as
  // InitialClass.  Note: The only case this should happen is if we have at
  // least one self-argument.
  if (Filtered.begin() == Filtered.end()) {
    if (setMemoryAccessEquivTo(MP, InitialClass))
      markMemoryUsersTouched(MP);
    return;
  }

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

  if (setMemoryAccessEquivTo(
          MP, AllEqual ? MemoryAccessToClass.lookup(AllSameValue) : nullptr))
    markMemoryUsersTouched(MP);
}

// Value number a single instruction, symbolically evaluating, performing
// congruence finding, and updating mappings.
void NewGVN::valueNumberInstruction(Instruction *I) {
  DEBUG(dbgs() << "Processing instruction " << *I << "\n");
  if (!I->isTerminator()) {
    const Expression *Symbolized = nullptr;
    if (DebugCounter::shouldExecute(VNCounter)) {
      Symbolized = performSymbolicEvaluation(I);
    } else {
      // Mark the instruction as unused so we don't value number it again.
      InstrDFS[I] = 0;
    }
    // If we couldn't come up with a symbolic expression, use the unknown
    // expression
    if (Symbolized == nullptr)
      Symbolized = createUnknownExpression(I);
    performCongruenceFinding(I, Symbolized);
  } else {
    // Handle terminators that return values. All of them produce values we
    // don't currently understand.  We don't place non-value producing
    // terminators in a class.
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
// subject to very rare false negatives. It is only useful for
// testing/debugging.
void NewGVN::verifyMemoryCongruency() const {
  // Anything equivalent in the memory access table should be in the same
  // congruence class.

  // Filter out the unreachable and trivially dead entries, because they may
  // never have been updated if the instructions were not processed.
  auto ReachableAccessPred =
      [&](const std::pair<const MemoryAccess *, CongruenceClass *> Pair) {
        bool Result = ReachableBlocks.count(Pair.first->getBlock());
        if (!Result)
          return false;
        if (auto *MemDef = dyn_cast<MemoryDef>(Pair.first))
          return !isInstructionTriviallyDead(MemDef->getMemoryInst());
        return true;
      };

  auto Filtered = make_filter_range(MemoryAccessToClass, ReachableAccessPred);
  for (auto KV : Filtered) {
    // Unreachable instructions may not have changed because we never process
    // them.
    if (!ReachableBlocks.count(KV.first->getBlock()))
      continue;
    if (auto *FirstMUD = dyn_cast<MemoryUseOrDef>(KV.first)) {
      auto *SecondMUD = dyn_cast<MemoryUseOrDef>(KV.second->RepMemoryAccess);
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

// Re-evaluate all the comparisons after value numbering and ensure they don't
// change. If they changed, we didn't mark them touched properly.
void NewGVN::verifyComparisons(Function &F) {
#ifndef NDEBUG
  for (auto &BB : F) {
    if (!ReachableBlocks.count(&BB))
      continue;
    for (auto &I : BB) {
      if (InstrDFS.lookup(&I) == 0)
        continue;
      if (isa<CmpInst>(&I)) {
        auto *CurrentVal = ValueToClass.lookup(&I);
        valueNumberInstruction(&I);
        assert(CurrentVal == ValueToClass.lookup(&I) &&
               "Re-evaluating comparison changed value");
      }
    }
  }
#endif
}

// This is the main transformation entry point.
bool NewGVN::runGVN(Function &F, DominatorTree *_DT, AssumptionCache *_AC,
                    TargetLibraryInfo *_TLI, AliasAnalysis *_AA,
                    MemorySSA *_MSSA) {
  bool Changed = false;
  NumFuncArgs = F.arg_size();
  DT = _DT;
  AC = _AC;
  TLI = _TLI;
  AA = _AA;
  MSSA = _MSSA;
  PredInfo = make_unique<PredicateInfo>(F, *DT, *AC);
  DL = &F.getParent()->getDataLayout();
  MSSAWalker = MSSA->getWalker();

  // Count number of instructions for sizing of hash tables, and come
  // up with a global dfs numbering for instructions.
  unsigned ICount = 1;
  // Add an empty instruction to account for the fact that we start at 1
  DFSToInstr.emplace_back(nullptr);
  // Note: We want ideal RPO traversal of the blocks, which is not quite the
  // same as dominator tree order, particularly with regard whether backedges
  // get visited first or second, given a block with multiple successors.
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
    // TODO: As we hit a new block, we should push and pop equalities into a
    // table lookupOperandLeader can use, to catch things PredicateInfo
    // might miss, like edge-only equivalences.
    for (int InstrNum = TouchedInstructions.find_first(); InstrNum != -1;
         InstrNum = TouchedInstructions.find_next(InstrNum)) {

      // This instruction was found to be dead. We don't bother looking
      // at it again.
      if (InstrNum == 0) {
        TouchedInstructions.reset(InstrNum);
        continue;
      }

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
  verifyComparisons(F);
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

// This function converts the set of members for a congruence class from values,
// to sets of defs and uses with associated DFS info.
void NewGVN::convertDenseToDFSOrdered(
    const CongruenceClass::MemberSet &Dense,
    SmallVectorImpl<ValueDFS> &DFSOrderedSet) {
  for (auto D : Dense) {
    // First add the value.
    BasicBlock *BB = getBlockForValue(D);
    // Constants are handled prior to ever calling this function, so
    // we should only be left with instructions as members.
    assert(BB && "Should have figured out a basic block for value");
    ValueDFS VD;
    DomTreeNode *DomNode = DT->getNode(BB);
    VD.DFSIn = DomNode->getDFSNumIn();
    VD.DFSOut = DomNode->getDFSNumOut();
    // If it's a store, use the leader of the value operand.
    if (auto *SI = dyn_cast<StoreInst>(D)) {
      auto Leader = lookupOperandLeader(SI->getValueOperand());
      VD.Val = alwaysAvailable(Leader) ? Leader : SI->getValueOperand();
    } else {
      VD.Val = D;
    }

    if (auto *I = dyn_cast<Instruction>(D))
      VD.LocalNum = InstrDFS.lookup(I);
    else
      llvm_unreachable("Should have been an instruction");

    DFSOrderedSet.emplace_back(VD);

    // Now add the uses.
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
          VD.LocalNum = InstrDFS.lookup(I);
        }

        // Skip uses in unreachable blocks, as we're going
        // to delete them.
        if (ReachableBlocks.count(IBlock) == 0)
          continue;

        DomTreeNode *DomNode = DT->getNode(IBlock);
        VD.DFSIn = DomNode->getDFSNumIn();
        VD.DFSOut = DomNode->getDFSNumOut();
        VD.U = &U;
        DFSOrderedSet.emplace_back(VD);
      }
    }
  }
}

// This function converts the set of members for a congruence class from values,
// to the set of defs for loads and stores, with associated DFS info.
void NewGVN::convertDenseToLoadsAndStores(
    const CongruenceClass::MemberSet &Dense,
    SmallVectorImpl<ValueDFS> &LoadsAndStores) {
  for (auto D : Dense) {
    if (!isa<LoadInst>(D) && !isa<StoreInst>(D))
      continue;

    BasicBlock *BB = getBlockForValue(D);
    ValueDFS VD;
    DomTreeNode *DomNode = DT->getNode(BB);
    VD.DFSIn = DomNode->getDFSNumIn();
    VD.DFSOut = DomNode->getDFSNumOut();
    VD.Val = D;

    // If it's an instruction, use the real local dfs number.
    if (auto *I = dyn_cast<Instruction>(D))
      VD.LocalNum = InstrDFS.lookup(I);
    else
      llvm_unreachable("Should have been an instruction");

    LoadsAndStores.emplace_back(VD);
  }
}

static void patchReplacementInstruction(Instruction *I, Value *Repl) {
  auto *ReplInst = dyn_cast<Instruction>(Repl);
  if (!ReplInst)
    return;

  // Patch the replacement so that it is not more restrictive than the value
  // being replaced.
  // Note that if 'I' is a load being replaced by some operation,
  // for example, by an arithmetic operation, then andIRFlags()
  // would just erase all math flags from the original arithmetic
  // operation, which is clearly not wanted and not needed.
  if (!isa<LoadInst>(I))
    ReplInst->andIRFlags(I);

  // FIXME: If both the original and replacement value are part of the
  // same control-flow region (meaning that the execution of one
  // guarantees the execution of the other), then we can combine the
  // noalias scopes here and do better than the general conservative
  // answer used in combineMetadata().

  // In general, GVN unifies expressions over different control-flow
  // regions, and so we need a conservative combination of the noalias
  // scopes.
  static const unsigned KnownIDs[] = {
      LLVMContext::MD_tbaa,           LLVMContext::MD_alias_scope,
      LLVMContext::MD_noalias,        LLVMContext::MD_range,
      LLVMContext::MD_fpmath,         LLVMContext::MD_invariant_load,
      LLVMContext::MD_invariant_group};
  combineMetadata(ReplInst, I, KnownIDs);
}

static void patchAndReplaceAllUsesWith(Instruction *I, Value *Repl) {
  patchReplacementInstruction(I, Repl);
  I->replaceAllUsesWith(Repl);
}

void NewGVN::deleteInstructionsInBlock(BasicBlock *BB) {
  DEBUG(dbgs() << "  BasicBlock Dead:" << *BB);
  ++NumGVNBlocksDeleted;

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
  // Now insert something that simplifycfg will turn into an unreachable.
  Type *Int8Ty = Type::getInt8Ty(BB->getContext());
  new StoreInst(UndefValue::get(Int8Ty),
                Constant::getNullValue(Int8Ty->getPointerTo()),
                BB->getTerminator());
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
  }

  for (CongruenceClass *CC : reverse(CongruenceClasses)) {
    // Track the equivalent store info so we can decide whether to try
    // dead store elimination.
    SmallVector<ValueDFS, 8> PossibleDeadStores;

    if (CC->Dead)
      continue;
    // Everything still in the INITIAL class is unreachable or dead.
    if (CC == InitialClass) {
#ifndef NDEBUG
      for (auto M : CC->Members)
        assert((!ReachableBlocks.count(cast<Instruction>(M)->getParent()) ||
                InstructionsToErase.count(cast<Instruction>(M))) &&
               "Everything in INITIAL should be unreachable or dead at this "
               "point");
#endif
      continue;
    }

    assert(CC->RepLeader && "We should have had a leader");

    // If this is a leader that is always available, and it's a
    // constant or has no equivalences, just replace everything with
    // it. We then update the congruence class with whatever members
    // are left.
    Value *Leader = CC->RepStoredValue ? CC->RepStoredValue : CC->RepLeader;
    if (alwaysAvailable(Leader)) {
      SmallPtrSet<Value *, 4> MembersLeft;
      for (auto M : CC->Members) {
        Value *Member = M;
        // Void things have no uses we can replace.
        if (Member == CC->RepLeader || Member->getType()->isVoidTy()) {
          MembersLeft.insert(Member);
          continue;
        }
        DEBUG(dbgs() << "Found replacement " << *(Leader) << " for " << *Member
                     << "\n");
        // Due to equality propagation, these may not always be
        // instructions, they may be real values.  We don't really
        // care about trying to replace the non-instructions.
        if (auto *I = dyn_cast<Instruction>(Member)) {
          assert(Leader != I && "About to accidentally remove our leader");
          replaceInstruction(I, Leader);
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

          // We ignore void things because we can't get a value from them.
          if (Member && Member->getType()->isVoidTy())
            continue;

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
          bool ShouldPush = Member && EliminationStack.empty();
          bool OutOfScope =
              !EliminationStack.isInScope(MemberDFSIn, MemberDFSOut);

          if (OutOfScope || ShouldPush) {
            // Sync to our current scope.
            EliminationStack.popUntilDFSScope(MemberDFSIn, MemberDFSOut);
            bool ShouldPush = Member && EliminationStack.empty();
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
          if (auto *ReplacedInst = dyn_cast<Instruction>(MemberUse->get())) {
            // Skip this if we are replacing predicateinfo with its original
            // operand, as we already know we can just drop it.
            auto *PI = PredInfo->getPredicateInfoFor(ReplacedInst);
            if (!PI || Result != PI->OriginalOp)
              patchReplacementInstruction(ReplacedInst, Result);
          }

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

    // If we have possible dead stores to look at, try to eliminate them.
    if (CC->StoreCount > 0) {
      convertDenseToLoadsAndStores(CC->Members, PossibleDeadStores);
      std::sort(PossibleDeadStores.begin(), PossibleDeadStores.end());
      ValueDFSStack EliminationStack;
      for (auto &VD : PossibleDeadStores) {
        int MemberDFSIn = VD.DFSIn;
        int MemberDFSOut = VD.DFSOut;
        Instruction *Member = cast<Instruction>(VD.Val);
        if (EliminationStack.empty() ||
            !EliminationStack.isInScope(MemberDFSIn, MemberDFSOut)) {
          // Sync to our current scope.
          EliminationStack.popUntilDFSScope(MemberDFSIn, MemberDFSOut);
          if (EliminationStack.empty()) {
            EliminationStack.push_back(Member, MemberDFSIn, MemberDFSOut);
            continue;
          }
        }
        // We already did load elimination, so nothing to do here.
        if (isa<LoadInst>(Member))
          continue;
        assert(!EliminationStack.empty());
        Instruction *Leader = cast<Instruction>(EliminationStack.back());
        (void)Leader;
        assert(DT->dominates(Leader->getParent(), Member->getParent()));
        // Member is dominater by Leader, and thus dead
        DEBUG(dbgs() << "Marking dead store " << *Member
                     << " that is dominated by " << *Leader << "\n");
        markInstructionForDeletion(Member);
        CC->Members.erase(Member);
        ++NumGVNDeadStores;
      }
    }
  }

  return AnythingReplaced;
}

// This function provides global ranking of operations so that we can place them
// in a canonical order.  Note that rank alone is not necessarily enough for a
// complete ordering, as constants all have the same rank.  However, generally,
// we will simplify an operation with all constants so that it doesn't matter
// what order they appear in.
unsigned int NewGVN::getRank(const Value *V) const {
  // Prefer undef to anything else
  if (isa<UndefValue>(V))
    return 0;
  if (isa<Constant>(V))
    return 1;
  else if (auto *A = dyn_cast<Argument>(V))
    return 2 + A->getArgNo();

  // Need to shift the instruction DFS by number of arguments + 3 to account for
  // the constant and argument ranking above.
  unsigned Result = InstrDFS.lookup(V);
  if (Result > 0)
    return 3 + NumFuncArgs + Result;
  // Unreachable or something else, just return a really large number.
  return ~0;
}

// This is a function that says whether two commutative operations should
// have their order swapped when canonicalizing.
bool NewGVN::shouldSwapOperands(const Value *A, const Value *B) const {
  // Because we only care about a total ordering, and don't rewrite expressions
  // in this order, we order by rank, which will give a strict weak ordering to
  // everything but constants, and then we order by pointer address.
  return std::make_pair(getRank(A), A) > std::make_pair(getRank(B), B);
}
