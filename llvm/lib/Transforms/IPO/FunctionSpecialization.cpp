//===- FunctionSpecialization.cpp - Function Specialization ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/FunctionSpecialization.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueLattice.h"
#include "llvm/Analysis/ValueLatticeUtils.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SCCPSolver.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include <cmath>

using namespace llvm;

#define DEBUG_TYPE "function-specialization"

STATISTIC(NumSpecsCreated, "Number of specializations created");

static cl::opt<bool> ForceSpecialization(
    "force-specialization", cl::init(false), cl::Hidden, cl::desc(
    "Force function specialization for every call site with a constant "
    "argument"));

static cl::opt<unsigned> MaxClones(
    "funcspec-max-clones", cl::init(3), cl::Hidden, cl::desc(
    "The maximum number of clones allowed for a single function "
    "specialization"));

static cl::opt<unsigned>
    MaxDiscoveryIterations("funcspec-max-discovery-iterations", cl::init(100),
                           cl::Hidden,
                           cl::desc("The maximum number of iterations allowed "
                                    "when searching for transitive "
                                    "phis"));

static cl::opt<unsigned> MaxIncomingPhiValues(
    "funcspec-max-incoming-phi-values", cl::init(8), cl::Hidden,
    cl::desc("The maximum number of incoming values a PHI node can have to be "
             "considered during the specialization bonus estimation"));

static cl::opt<unsigned> MaxBlockPredecessors(
    "funcspec-max-block-predecessors", cl::init(2), cl::Hidden, cl::desc(
    "The maximum number of predecessors a basic block can have to be "
    "considered during the estimation of dead code"));

static cl::opt<unsigned> MinFunctionSize(
    "funcspec-min-function-size", cl::init(500), cl::Hidden,
    cl::desc("Don't specialize functions that have less than this number of "
             "instructions"));

static cl::opt<unsigned> MaxCodeSizeGrowth(
    "funcspec-max-codesize-growth", cl::init(3), cl::Hidden, cl::desc(
    "Maximum codesize growth allowed per function"));

static cl::opt<unsigned> MinCodeSizeSavings(
    "funcspec-min-codesize-savings", cl::init(20), cl::Hidden,
    cl::desc("Reject specializations whose codesize savings are less than this "
             "much percent of the original function size"));

static cl::opt<unsigned> MinLatencySavings(
    "funcspec-min-latency-savings", cl::init(20), cl::Hidden,
    cl::desc("Reject specializations whose latency savings are less than this "
             "much percent of the original function size"));

static cl::opt<unsigned> MinInliningBonus(
    "funcspec-min-inlining-bonus", cl::init(300), cl::Hidden,
    cl::desc("Reject specializations whose inlining bonus is less than this "
             "much percent of the original function size"));

static cl::opt<bool> SpecializeOnAddress(
    "funcspec-on-address", cl::init(false), cl::Hidden, cl::desc(
    "Enable function specialization on the address of global values"));

static cl::opt<bool> SpecializeLiteralConstant(
    "funcspec-for-literal-constant", cl::init(true), cl::Hidden,
    cl::desc(
        "Enable specialization of functions that take a literal constant as an "
        "argument"));

extern cl::opt<bool> ProfcheckDisableMetadataFixes;

bool InstCostVisitor::canEliminateSuccessor(BasicBlock *BB,
                                            BasicBlock *Succ) const {
  unsigned I = 0;
  return all_of(predecessors(Succ), [&I, BB, Succ, this](BasicBlock *Pred) {
    return I++ < MaxBlockPredecessors &&
           (Pred == BB || Pred == Succ || !isBlockExecutable(Pred));
  });
}

// Estimates the codesize savings due to dead code after constant propagation.
// \p WorkList represents the basic blocks of a specialization which will
// eventually become dead once we replace instructions that are known to be
// constants. The successors of such blocks are added to the list as long as
// the \p Solver found they were executable prior to specialization, and only
// if all their predecessors are dead.
Cost InstCostVisitor::estimateBasicBlocks(
                          SmallVectorImpl<BasicBlock *> &WorkList) {
  Cost CodeSize = 0;
  // Accumulate the codesize savings of each basic block.
  while (!WorkList.empty()) {
    BasicBlock *BB = WorkList.pop_back_val();

    // These blocks are considered dead as far as the InstCostVisitor
    // is concerned. They haven't been proven dead yet by the Solver,
    // but may become if we propagate the specialization arguments.
    assert(Solver.isBlockExecutable(BB) && "BB already found dead by IPSCCP!");
    if (!DeadBlocks.insert(BB).second)
      continue;

    for (Instruction &I : *BB) {
      // If it's a known constant we have already accounted for it.
      if (KnownConstants.contains(&I))
        continue;

      Cost C = TTI.getInstructionCost(&I, TargetTransformInfo::TCK_CodeSize);

      LLVM_DEBUG(dbgs() << "FnSpecialization:     CodeSize " << C
                        << " for user " << I << "\n");
      CodeSize += C;
    }

    // Keep adding dead successors to the list as long as they are
    // executable and only reachable from dead blocks.
    for (BasicBlock *SuccBB : successors(BB))
      if (isBlockExecutable(SuccBB) && canEliminateSuccessor(BB, SuccBB))
        WorkList.push_back(SuccBB);
  }
  return CodeSize;
}

Constant *InstCostVisitor::findConstantFor(Value *V) const {
  if (auto *C = dyn_cast<Constant>(V))
    return C;
  if (auto *C = Solver.getConstantOrNull(V))
    return C;
  return KnownConstants.lookup(V);
}

Cost InstCostVisitor::getCodeSizeSavingsFromPendingPHIs() {
  Cost CodeSize;
  while (!PendingPHIs.empty()) {
    Instruction *Phi = PendingPHIs.pop_back_val();
    // The pending PHIs could have been proven dead by now.
    if (isBlockExecutable(Phi->getParent()))
      CodeSize += getCodeSizeSavingsForUser(Phi);
  }
  return CodeSize;
}

/// Compute the codesize savings for replacing argument \p A with constant \p C.
Cost InstCostVisitor::getCodeSizeSavingsForArg(Argument *A, Constant *C) {
  LLVM_DEBUG(dbgs() << "FnSpecialization: Analysing bonus for constant: "
                    << C->getNameOrAsOperand() << "\n");
  Cost CodeSize;
  for (auto *U : A->users())
    if (auto *UI = dyn_cast<Instruction>(U))
      if (isBlockExecutable(UI->getParent()))
        CodeSize += getCodeSizeSavingsForUser(UI, A, C);

  LLVM_DEBUG(dbgs() << "FnSpecialization:   Accumulated bonus {CodeSize = "
                    << CodeSize << "} for argument " << *A << "\n");
  return CodeSize;
}

/// Compute the latency savings from replacing all arguments with constants for
/// a specialization candidate. As this function computes the latency savings
/// for all Instructions in KnownConstants at once, it should be called only
/// after every instruction has been visited, i.e. after:
///
/// * getCodeSizeSavingsForArg has been run for every constant argument of a
///   specialization candidate
///
/// * getCodeSizeSavingsFromPendingPHIs has been run
///
/// to ensure that the latency savings are calculated for all Instructions we
/// have visited and found to be constant.
Cost InstCostVisitor::getLatencySavingsForKnownConstants() {
  auto &BFI = GetBFI(*F);
  Cost TotalLatency = 0;

  for (auto Pair : KnownConstants) {
    Instruction *I = dyn_cast<Instruction>(Pair.first);
    if (!I)
      continue;

    uint64_t Weight = BFI.getBlockFreq(I->getParent()).getFrequency() /
                      BFI.getEntryFreq().getFrequency();

    Cost Latency =
        Weight * TTI.getInstructionCost(I, TargetTransformInfo::TCK_Latency);

    LLVM_DEBUG(dbgs() << "FnSpecialization:     {Latency = " << Latency
                      << "} for instruction " << *I << "\n");

    TotalLatency += Latency;
  }

  return TotalLatency;
}

Cost InstCostVisitor::getCodeSizeSavingsForUser(Instruction *User, Value *Use,
                                                Constant *C) {
  // We have already propagated a constant for this user.
  if (KnownConstants.contains(User))
    return 0;

  // Cache the iterator before visiting.
  LastVisited = Use ? KnownConstants.insert({Use, C}).first
                    : KnownConstants.end();

  Cost CodeSize = 0;
  if (auto *I = dyn_cast<SwitchInst>(User)) {
    CodeSize = estimateSwitchInst(*I);
  } else if (auto *I = dyn_cast<BranchInst>(User)) {
    CodeSize = estimateBranchInst(*I);
  } else {
    C = visit(*User);
    if (!C)
      return 0;
  }

  // Even though it doesn't make sense to bind switch and branch instructions
  // with a constant, unlike any other instruction type, it prevents estimating
  // their bonus multiple times.
  KnownConstants.insert({User, C});

  CodeSize += TTI.getInstructionCost(User, TargetTransformInfo::TCK_CodeSize);

  LLVM_DEBUG(dbgs() << "FnSpecialization:     {CodeSize = " << CodeSize
                    << "} for user " << *User << "\n");

  for (auto *U : User->users())
    if (auto *UI = dyn_cast<Instruction>(U))
      if (UI != User && isBlockExecutable(UI->getParent()))
        CodeSize += getCodeSizeSavingsForUser(UI, User, C);

  return CodeSize;
}

Cost InstCostVisitor::estimateSwitchInst(SwitchInst &I) {
  assert(LastVisited != KnownConstants.end() && "Invalid iterator!");

  if (I.getCondition() != LastVisited->first)
    return 0;

  auto *C = dyn_cast<ConstantInt>(LastVisited->second);
  if (!C)
    return 0;

  BasicBlock *Succ = I.findCaseValue(C)->getCaseSuccessor();
  // Initialize the worklist with the dead basic blocks. These are the
  // destination labels which are different from the one corresponding
  // to \p C. They should be executable and have a unique predecessor.
  SmallVector<BasicBlock *> WorkList;
  for (const auto &Case : I.cases()) {
    BasicBlock *BB = Case.getCaseSuccessor();
    if (BB != Succ && isBlockExecutable(BB) &&
        canEliminateSuccessor(I.getParent(), BB))
      WorkList.push_back(BB);
  }

  return estimateBasicBlocks(WorkList);
}

Cost InstCostVisitor::estimateBranchInst(BranchInst &I) {
  assert(LastVisited != KnownConstants.end() && "Invalid iterator!");

  if (I.getCondition() != LastVisited->first)
    return 0;

  BasicBlock *Succ = I.getSuccessor(LastVisited->second->isOneValue());
  // Initialize the worklist with the dead successor as long as
  // it is executable and has a unique predecessor.
  SmallVector<BasicBlock *> WorkList;
  if (isBlockExecutable(Succ) && canEliminateSuccessor(I.getParent(), Succ))
    WorkList.push_back(Succ);

  return estimateBasicBlocks(WorkList);
}

bool InstCostVisitor::discoverTransitivelyIncomingValues(
    Constant *Const, PHINode *Root, DenseSet<PHINode *> &TransitivePHIs) {

  SmallVector<PHINode *, 64> WorkList;
  WorkList.push_back(Root);
  unsigned Iter = 0;

  while (!WorkList.empty()) {
    PHINode *PN = WorkList.pop_back_val();

    if (++Iter > MaxDiscoveryIterations ||
        PN->getNumIncomingValues() > MaxIncomingPhiValues)
      return false;

    if (!TransitivePHIs.insert(PN).second)
      continue;

    for (unsigned I = 0, E = PN->getNumIncomingValues(); I != E; ++I) {
      Value *V = PN->getIncomingValue(I);

      // Disregard self-references and dead incoming values.
      if (auto *Inst = dyn_cast<Instruction>(V))
        if (Inst == PN || !isBlockExecutable(PN->getIncomingBlock(I)))
          continue;

      if (Constant *C = findConstantFor(V)) {
        // Not all incoming values are the same constant. Bail immediately.
        if (C != Const)
          return false;
        continue;
      }

      if (auto *Phi = dyn_cast<PHINode>(V)) {
        WorkList.push_back(Phi);
        continue;
      }

      // We can't reason about anything else.
      return false;
    }
  }
  return true;
}

Constant *InstCostVisitor::visitPHINode(PHINode &I) {
  if (I.getNumIncomingValues() > MaxIncomingPhiValues)
    return nullptr;

  bool Inserted = VisitedPHIs.insert(&I).second;
  Constant *Const = nullptr;
  bool HaveSeenIncomingPHI = false;

  for (unsigned Idx = 0, E = I.getNumIncomingValues(); Idx != E; ++Idx) {
    Value *V = I.getIncomingValue(Idx);

    // Disregard self-references and dead incoming values.
    if (auto *Inst = dyn_cast<Instruction>(V))
      if (Inst == &I || !isBlockExecutable(I.getIncomingBlock(Idx)))
        continue;

    if (Constant *C = findConstantFor(V)) {
      if (!Const)
        Const = C;
      // Not all incoming values are the same constant. Bail immediately.
      if (C != Const)
        return nullptr;
      continue;
    }

    if (Inserted) {
      // First time we are seeing this phi. We will retry later, after
      // all the constant arguments have been propagated. Bail for now.
      PendingPHIs.push_back(&I);
      return nullptr;
    }

    if (isa<PHINode>(V)) {
      // Perhaps it is a Transitive Phi. We will confirm later.
      HaveSeenIncomingPHI = true;
      continue;
    }

    // We can't reason about anything else.
    return nullptr;
  }

  if (!Const)
    return nullptr;

  if (!HaveSeenIncomingPHI)
    return Const;

  DenseSet<PHINode *> TransitivePHIs;
  if (!discoverTransitivelyIncomingValues(Const, &I, TransitivePHIs))
    return nullptr;

  return Const;
}

Constant *InstCostVisitor::visitFreezeInst(FreezeInst &I) {
  assert(LastVisited != KnownConstants.end() && "Invalid iterator!");

  if (isGuaranteedNotToBeUndefOrPoison(LastVisited->second))
    return LastVisited->second;
  return nullptr;
}

Constant *InstCostVisitor::visitCallBase(CallBase &I) {
  assert(LastVisited != KnownConstants.end() && "Invalid iterator!");

  Function *F = I.getCalledFunction();
  if (!F || !canConstantFoldCallTo(&I, F))
    return nullptr;

  SmallVector<Constant *, 8> Operands;
  Operands.reserve(I.getNumOperands());

  for (unsigned Idx = 0, E = I.getNumOperands() - 1; Idx != E; ++Idx) {
    Value *V = I.getOperand(Idx);
    if (isa<MetadataAsValue>(V))
      return nullptr;
    Constant *C = findConstantFor(V);
    if (!C)
      return nullptr;
    Operands.push_back(C);
  }

  auto Ops = ArrayRef(Operands.begin(), Operands.end());
  return ConstantFoldCall(&I, F, Ops);
}

Constant *InstCostVisitor::visitLoadInst(LoadInst &I) {
  assert(LastVisited != KnownConstants.end() && "Invalid iterator!");

  if (isa<ConstantPointerNull>(LastVisited->second))
    return nullptr;
  return ConstantFoldLoadFromConstPtr(LastVisited->second, I.getType(), DL);
}

Constant *InstCostVisitor::visitGetElementPtrInst(GetElementPtrInst &I) {
  SmallVector<Constant *, 8> Operands;
  Operands.reserve(I.getNumOperands());

  for (unsigned Idx = 0, E = I.getNumOperands(); Idx != E; ++Idx) {
    Value *V = I.getOperand(Idx);
    Constant *C = findConstantFor(V);
    if (!C)
      return nullptr;
    Operands.push_back(C);
  }

  auto Ops = ArrayRef(Operands.begin(), Operands.end());
  return ConstantFoldInstOperands(&I, Ops, DL);
}

Constant *InstCostVisitor::visitSelectInst(SelectInst &I) {
  assert(LastVisited != KnownConstants.end() && "Invalid iterator!");

  if (I.getCondition() == LastVisited->first) {
    Value *V = LastVisited->second->isZeroValue() ? I.getFalseValue()
                                                  : I.getTrueValue();
    return findConstantFor(V);
  }
  if (Constant *Condition = findConstantFor(I.getCondition()))
    if ((I.getTrueValue() == LastVisited->first && Condition->isOneValue()) ||
        (I.getFalseValue() == LastVisited->first && Condition->isZeroValue()))
      return LastVisited->second;
  return nullptr;
}

Constant *InstCostVisitor::visitCastInst(CastInst &I) {
  return ConstantFoldCastOperand(I.getOpcode(), LastVisited->second,
                                 I.getType(), DL);
}

Constant *InstCostVisitor::visitCmpInst(CmpInst &I) {
  assert(LastVisited != KnownConstants.end() && "Invalid iterator!");

  Constant *Const = LastVisited->second;
  bool ConstOnRHS = I.getOperand(1) == LastVisited->first;
  Value *V = ConstOnRHS ? I.getOperand(0) : I.getOperand(1);
  Constant *Other = findConstantFor(V);

  if (Other) {
    if (ConstOnRHS)
      std::swap(Const, Other);
    return ConstantFoldCompareInstOperands(I.getPredicate(), Const, Other, DL);
  }

  // If we haven't found Other to be a specific constant value, we may still be
  // able to constant fold using information from the lattice value.
  const ValueLatticeElement &ConstLV = ValueLatticeElement::get(Const);
  const ValueLatticeElement &OtherLV = Solver.getLatticeValueFor(V);
  auto &V1State = ConstOnRHS ? OtherLV : ConstLV;
  auto &V2State = ConstOnRHS ? ConstLV : OtherLV;
  return V1State.getCompare(I.getPredicate(), I.getType(), V2State, DL);
}

Constant *InstCostVisitor::visitUnaryOperator(UnaryOperator &I) {
  assert(LastVisited != KnownConstants.end() && "Invalid iterator!");

  return ConstantFoldUnaryOpOperand(I.getOpcode(), LastVisited->second, DL);
}

Constant *InstCostVisitor::visitBinaryOperator(BinaryOperator &I) {
  assert(LastVisited != KnownConstants.end() && "Invalid iterator!");

  bool ConstOnRHS = I.getOperand(1) == LastVisited->first;
  Value *V = ConstOnRHS ? I.getOperand(0) : I.getOperand(1);
  Constant *Other = findConstantFor(V);
  Value *OtherVal = Other ? Other : V;
  Value *ConstVal = LastVisited->second;

  if (ConstOnRHS)
    std::swap(ConstVal, OtherVal);

  return dyn_cast_or_null<Constant>(
      simplifyBinOp(I.getOpcode(), ConstVal, OtherVal, SimplifyQuery(DL)));
}

Constant *FunctionSpecializer::getPromotableAlloca(AllocaInst *Alloca,
                                                   CallInst *Call) {
  Value *StoreValue = nullptr;
  for (auto *User : Alloca->users()) {
    // We can't use llvm::isAllocaPromotable() as that would fail because of
    // the usage in the CallInst, which is what we check here.
    if (User == Call)
      continue;

    if (auto *Store = dyn_cast<StoreInst>(User)) {
      // This is a duplicate store, bail out.
      if (StoreValue || Store->isVolatile())
        return nullptr;
      StoreValue = Store->getValueOperand();
      continue;
    }
    // Bail if there is any other unknown usage.
    return nullptr;
  }

  if (!StoreValue)
    return nullptr;

  return getCandidateConstant(StoreValue);
}

// A constant stack value is an AllocaInst that has a single constant
// value stored to it. Return this constant if such an alloca stack value
// is a function argument.
Constant *FunctionSpecializer::getConstantStackValue(CallInst *Call,
                                                     Value *Val) {
  if (!Val)
    return nullptr;
  Val = Val->stripPointerCasts();
  if (auto *ConstVal = dyn_cast<ConstantInt>(Val))
    return ConstVal;
  auto *Alloca = dyn_cast<AllocaInst>(Val);
  if (!Alloca || !Alloca->getAllocatedType()->isIntegerTy())
    return nullptr;
  return getPromotableAlloca(Alloca, Call);
}

// To support specializing recursive functions, it is important to propagate
// constant arguments because after a first iteration of specialisation, a
// reduced example may look like this:
//
//     define internal void @RecursiveFn(i32* arg1) {
//       %temp = alloca i32, align 4
//       store i32 2 i32* %temp, align 4
//       call void @RecursiveFn.1(i32* nonnull %temp)
//       ret void
//     }
//
// Before a next iteration, we need to propagate the constant like so
// which allows further specialization in next iterations.
//
//     @funcspec.arg = internal constant i32 2
//
//     define internal void @someFunc(i32* arg1) {
//       call void @otherFunc(i32* nonnull @funcspec.arg)
//       ret void
//     }
//
// See if there are any new constant values for the callers of \p F via
// stack variables and promote them to global variables.
void FunctionSpecializer::promoteConstantStackValues(Function *F) {
  for (User *U : F->users()) {

    auto *Call = dyn_cast<CallInst>(U);
    if (!Call)
      continue;

    if (!Solver.isBlockExecutable(Call->getParent()))
      continue;

    for (const Use &U : Call->args()) {
      unsigned Idx = Call->getArgOperandNo(&U);
      Value *ArgOp = Call->getArgOperand(Idx);
      Type *ArgOpType = ArgOp->getType();

      if (!Call->onlyReadsMemory(Idx) || !ArgOpType->isPointerTy())
        continue;

      auto *ConstVal = getConstantStackValue(Call, ArgOp);
      if (!ConstVal)
        continue;

      Value *GV = new GlobalVariable(M, ConstVal->getType(), true,
                                     GlobalValue::InternalLinkage, ConstVal,
                                     "specialized.arg." + Twine(++NGlobals));
      Call->setArgOperand(Idx, GV);
    }
  }
}

// The SCCP solver inserts bitcasts for PredicateInfo. These interfere with the
// promoteConstantStackValues() optimization.
static void removeSSACopy(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &Inst : llvm::make_early_inc_range(BB)) {
      auto *BC = dyn_cast<BitCastInst>(&Inst);
      if (!BC || BC->getType() != BC->getOperand(0)->getType())
        continue;
      Inst.replaceAllUsesWith(BC->getOperand(0));
      Inst.eraseFromParent();
    }
  }
}

/// Remove any ssa_copy intrinsics that may have been introduced.
void FunctionSpecializer::cleanUpSSA() {
  for (Function *F : Specializations)
    removeSSACopy(*F);
}


template <> struct llvm::DenseMapInfo<SpecSig> {
  static inline SpecSig getEmptyKey() { return {~0U, {}}; }

  static inline SpecSig getTombstoneKey() { return {~1U, {}}; }

  static unsigned getHashValue(const SpecSig &S) {
    return static_cast<unsigned>(hash_value(S));
  }

  static bool isEqual(const SpecSig &LHS, const SpecSig &RHS) {
    return LHS == RHS;
  }
};

FunctionSpecializer::~FunctionSpecializer() {
  LLVM_DEBUG(
    if (NumSpecsCreated > 0)
      dbgs() << "FnSpecialization: Created " << NumSpecsCreated
             << " specializations in module " << M.getName() << "\n");
  // Eliminate dead code.
  removeDeadFunctions();
  cleanUpSSA();
}

/// Get the unsigned Value of given Cost object. Assumes the Cost is always
/// non-negative, which is true for both TCK_CodeSize and TCK_Latency, and
/// always Valid.
static unsigned getCostValue(const Cost &C) {
  int64_t Value = C.getValue();

  assert(Value >= 0 && "CodeSize and Latency cannot be negative");
  // It is safe to down cast since we know the arguments cannot be negative and
  // Cost is of type int64_t.
  return static_cast<unsigned>(Value);
}

/// Attempt to specialize functions in the module to enable constant
/// propagation across function boundaries.
///
/// \returns true if at least one function is specialized.
bool FunctionSpecializer::run() {
  // Find possible specializations for each function.
  SpecMap SM;
  SmallVector<Spec, 32> AllSpecs;
  unsigned NumCandidates = 0;
  for (Function &F : M) {
    if (!isCandidateFunction(&F))
      continue;

    auto [It, Inserted] = FunctionMetrics.try_emplace(&F);
    CodeMetrics &Metrics = It->second;
    //Analyze the function.
    if (Inserted) {
      SmallPtrSet<const Value *, 32> EphValues;
      CodeMetrics::collectEphemeralValues(&F, &GetAC(F), EphValues);
      for (BasicBlock &BB : F)
        Metrics.analyzeBasicBlock(&BB, GetTTI(F), EphValues);
    }

    // When specializing literal constants is enabled, always require functions
    // to be larger than MinFunctionSize, to prevent excessive specialization.
    const bool RequireMinSize =
        !ForceSpecialization &&
        (SpecializeLiteralConstant || !F.hasFnAttribute(Attribute::NoInline));

    // If the code metrics reveal that we shouldn't duplicate the function,
    // or if the code size implies that this function is easy to get inlined,
    // then we shouldn't specialize it.
    if (Metrics.notDuplicatable || !Metrics.NumInsts.isValid() ||
        (RequireMinSize && Metrics.NumInsts < MinFunctionSize))
      continue;

    // When specialization on literal constants is disabled, only consider
    // recursive functions when running multiple times to save wasted analysis,
    // as we will not be able to specialize on any newly found literal constant
    // return values.
    if (!SpecializeLiteralConstant && !Inserted && !Metrics.isRecursive)
      continue;

    int64_t Sz = Metrics.NumInsts.getValue();
    assert(Sz > 0 && "CodeSize should be positive");
    // It is safe to down cast from int64_t, NumInsts is always positive.
    unsigned FuncSize = static_cast<unsigned>(Sz);

    LLVM_DEBUG(dbgs() << "FnSpecialization: Specialization cost for "
                      << F.getName() << " is " << FuncSize << "\n");

    if (Inserted && Metrics.isRecursive)
      promoteConstantStackValues(&F);

    if (!findSpecializations(&F, FuncSize, AllSpecs, SM)) {
      LLVM_DEBUG(
          dbgs() << "FnSpecialization: No possible specializations found for "
                 << F.getName() << "\n");
      continue;
    }

    ++NumCandidates;
  }

  if (!NumCandidates) {
    LLVM_DEBUG(
        dbgs()
        << "FnSpecialization: No possible specializations found in module\n");
    return false;
  }

  // Choose the most profitable specialisations, which fit in the module
  // specialization budget, which is derived from maximum number of
  // specializations per specialization candidate function.
  auto CompareScore = [&AllSpecs](unsigned I, unsigned J) {
    if (AllSpecs[I].Score != AllSpecs[J].Score)
      return AllSpecs[I].Score > AllSpecs[J].Score;
    return I > J;
  };
  const unsigned NSpecs =
      std::min(NumCandidates * MaxClones, unsigned(AllSpecs.size()));
  SmallVector<unsigned> BestSpecs(NSpecs + 1);
  std::iota(BestSpecs.begin(), BestSpecs.begin() + NSpecs, 0);
  if (AllSpecs.size() > NSpecs) {
    LLVM_DEBUG(dbgs() << "FnSpecialization: Number of candidates exceed "
                      << "the maximum number of clones threshold.\n"
                      << "FnSpecialization: Specializing the "
                      << NSpecs
                      << " most profitable candidates.\n");
    std::make_heap(BestSpecs.begin(), BestSpecs.begin() + NSpecs, CompareScore);
    for (unsigned I = NSpecs, N = AllSpecs.size(); I < N; ++I) {
      BestSpecs[NSpecs] = I;
      std::push_heap(BestSpecs.begin(), BestSpecs.end(), CompareScore);
      std::pop_heap(BestSpecs.begin(), BestSpecs.end(), CompareScore);
    }
  }

  LLVM_DEBUG(dbgs() << "FnSpecialization: List of specializations \n";
             for (unsigned I = 0; I < NSpecs; ++I) {
               const Spec &S = AllSpecs[BestSpecs[I]];
               dbgs() << "FnSpecialization: Function " << S.F->getName()
                      << " , score " << S.Score << "\n";
               for (const ArgInfo &Arg : S.Sig.Args)
                 dbgs() << "FnSpecialization:   FormalArg = "
                        << Arg.Formal->getNameOrAsOperand()
                        << ", ActualArg = " << Arg.Actual->getNameOrAsOperand()
                        << "\n";
             });

  // Create the chosen specializations.
  SmallPtrSet<Function *, 8> OriginalFuncs;
  SmallVector<Function *> Clones;
  for (unsigned I = 0; I < NSpecs; ++I) {
    Spec &S = AllSpecs[BestSpecs[I]];

    // Accumulate the codesize growth for the function, now we are creating the
    // specialization.
    FunctionGrowth[S.F] += S.CodeSize;

    S.Clone = createSpecialization(S.F, S.Sig);

    // Update the known call sites to call the clone.
    for (CallBase *Call : S.CallSites) {
      Function *Clone = S.Clone;
      LLVM_DEBUG(dbgs() << "FnSpecialization: Redirecting " << *Call
                        << " to call " << Clone->getName() << "\n");
      Call->setCalledFunction(S.Clone);
      auto &BFI = GetBFI(*Call->getFunction());
      std::optional<uint64_t> Count =
          BFI.getBlockProfileCount(Call->getParent());
      if (Count && !ProfcheckDisableMetadataFixes) {
        std::optional<llvm::Function::ProfileCount> MaybeCloneCount =
            Clone->getEntryCount();
        assert(MaybeCloneCount && "Clone entry count was not set!");
        uint64_t CallCount = *Count + MaybeCloneCount->getCount();
        Clone->setEntryCount(CallCount);
        if (std::optional<llvm::Function::ProfileCount> MaybeOriginalCount =
                S.F->getEntryCount()) {
          uint64_t OriginalCount = MaybeOriginalCount->getCount();
          if (OriginalCount >= *Count) {
            S.F->setEntryCount(OriginalCount - *Count);
          } else {
            // This should generally not happen as that would mean there are
            // more computed calls to the function than what was recorded.
            LLVM_DEBUG(S.F->setEntryCount(0));
          }
        }
      }
    }

    Clones.push_back(S.Clone);
    OriginalFuncs.insert(S.F);
  }

  Solver.solveWhileResolvedUndefsIn(Clones);

  // Update the rest of the call sites - these are the recursive calls, calls
  // to discarded specialisations and calls that may match a specialisation
  // after the solver runs.
  for (Function *F : OriginalFuncs) {
    auto [Begin, End] = SM[F];
    updateCallSites(F, AllSpecs.begin() + Begin, AllSpecs.begin() + End);
  }

  for (Function *F : Clones) {
    if (F->getReturnType()->isVoidTy())
      continue;
    if (F->getReturnType()->isStructTy()) {
      auto *STy = cast<StructType>(F->getReturnType());
      if (!Solver.isStructLatticeConstant(F, STy))
        continue;
    } else {
      auto It = Solver.getTrackedRetVals().find(F);
      assert(It != Solver.getTrackedRetVals().end() &&
             "Return value ought to be tracked");
      if (SCCPSolver::isOverdefined(It->second))
        continue;
    }
    for (User *U : F->users()) {
      if (auto *CS = dyn_cast<CallBase>(U)) {
        //The user instruction does not call our function.
        if (CS->getCalledFunction() != F)
          continue;
        Solver.resetLatticeValueFor(CS);
      }
    }
  }

  // Rerun the solver to notify the users of the modified callsites.
  Solver.solveWhileResolvedUndefs();

  for (Function *F : OriginalFuncs)
    if (FunctionMetrics[F].isRecursive)
      promoteConstantStackValues(F);

  return true;
}

void FunctionSpecializer::removeDeadFunctions() {
  for (Function *F : DeadFunctions) {
    LLVM_DEBUG(dbgs() << "FnSpecialization: Removing dead function "
                      << F->getName() << "\n");
    if (FAM)
      FAM->clear(*F, F->getName());

    // Remove all the callsites that were proven unreachable once, and replace
    // them with poison.
    for (User *U : make_early_inc_range(F->users())) {
      assert((isa<CallInst>(U) || isa<InvokeInst>(U)) &&
             "User of dead function must be call or invoke");
      Instruction *CS = cast<Instruction>(U);
      CS->replaceAllUsesWith(PoisonValue::get(CS->getType()));
      CS->eraseFromParent();
    }
    F->eraseFromParent();
  }
  DeadFunctions.clear();
}

/// Clone the function \p F and remove the ssa_copy intrinsics added by
/// the SCCPSolver in the cloned version.
static Function *cloneCandidateFunction(Function *F, unsigned NSpecs) {
  ValueToValueMapTy Mappings;
  Function *Clone = CloneFunction(F, Mappings);
  Clone->setName(F->getName() + ".specialized." + Twine(NSpecs));
  removeSSACopy(*Clone);
  return Clone;
}

bool FunctionSpecializer::findSpecializations(Function *F, unsigned FuncSize,
                                              SmallVectorImpl<Spec> &AllSpecs,
                                              SpecMap &SM) {
  // A mapping from a specialisation signature to the index of the respective
  // entry in the all specialisation array. Used to ensure uniqueness of
  // specialisations.
  DenseMap<SpecSig, unsigned> UniqueSpecs;

  // Get a list of interesting arguments.
  SmallVector<Argument *> Args;
  for (Argument &Arg : F->args())
    if (isArgumentInteresting(&Arg))
      Args.push_back(&Arg);

  if (Args.empty())
    return false;

  for (User *U : F->users()) {
    if (!isa<CallInst>(U) && !isa<InvokeInst>(U))
      continue;
    auto &CS = *cast<CallBase>(U);

    // The user instruction does not call our function.
    if (CS.getCalledFunction() != F)
      continue;

    // If the call site has attribute minsize set, that callsite won't be
    // specialized.
    if (CS.hasFnAttr(Attribute::MinSize))
      continue;

    // If the parent of the call site will never be executed, we don't need
    // to worry about the passed value.
    if (!Solver.isBlockExecutable(CS.getParent()))
      continue;

    // Examine arguments and create a specialisation candidate from the
    // constant operands of this call site.
    SpecSig S;
    for (Argument *A : Args) {
      Constant *C = getCandidateConstant(CS.getArgOperand(A->getArgNo()));
      if (!C)
        continue;
      LLVM_DEBUG(dbgs() << "FnSpecialization: Found interesting argument "
                        << A->getName() << " : " << C->getNameOrAsOperand()
                        << "\n");
      S.Args.push_back({A, C});
    }

    if (S.Args.empty())
      continue;

    // Check if we have encountered the same specialisation already.
    if (auto It = UniqueSpecs.find(S); It != UniqueSpecs.end()) {
      // Existing specialisation. Add the call to the list to rewrite, unless
      // it's a recursive call. A specialisation, generated because of a
      // recursive call may end up as not the best specialisation for all
      // the cloned instances of this call, which result from specialising
      // functions. Hence we don't rewrite the call directly, but match it with
      // the best specialisation once all specialisations are known.
      if (CS.getFunction() == F)
        continue;
      const unsigned Index = It->second;
      AllSpecs[Index].CallSites.push_back(&CS);
    } else {
      // Calculate the specialisation gain.
      Cost CodeSize;
      unsigned Score = 0;
      InstCostVisitor Visitor = getInstCostVisitorFor(F);
      for (ArgInfo &A : S.Args) {
        CodeSize += Visitor.getCodeSizeSavingsForArg(A.Formal, A.Actual);
        Score += getInliningBonus(A.Formal, A.Actual);
      }
      CodeSize += Visitor.getCodeSizeSavingsFromPendingPHIs();

      unsigned CodeSizeSavings = getCostValue(CodeSize);
      unsigned SpecSize = FuncSize - CodeSizeSavings;

      auto IsProfitable = [&]() -> bool {
        // No check required.
        if (ForceSpecialization)
          return true;

        LLVM_DEBUG(
            dbgs() << "FnSpecialization: Specialization bonus {Inlining = "
                   << Score << " (" << (Score * 100 / FuncSize) << "%)}\n");

        // Minimum inlining bonus.
        if (Score > MinInliningBonus * FuncSize / 100)
          return true;

        LLVM_DEBUG(
            dbgs() << "FnSpecialization: Specialization bonus {CodeSize = "
                   << CodeSizeSavings << " ("
                   << (CodeSizeSavings * 100 / FuncSize) << "%)}\n");

        // Minimum codesize savings.
        if (CodeSizeSavings < MinCodeSizeSavings * FuncSize / 100)
          return false;

        // Lazily compute the Latency, to avoid unnecessarily computing BFI.
        unsigned LatencySavings =
            getCostValue(Visitor.getLatencySavingsForKnownConstants());

        LLVM_DEBUG(
            dbgs() << "FnSpecialization: Specialization bonus {Latency = "
                   << LatencySavings << " ("
                   << (LatencySavings * 100 / FuncSize) << "%)}\n");

        // Minimum latency savings.
        if (LatencySavings < MinLatencySavings * FuncSize / 100)
          return false;
        // Maximum codesize growth.
        if ((FunctionGrowth[F] + SpecSize) / FuncSize > MaxCodeSizeGrowth)
          return false;

        Score += std::max(CodeSizeSavings, LatencySavings);
        return true;
      };

      // Discard unprofitable specialisations.
      if (!IsProfitable())
        continue;

      // Create a new specialisation entry.
      auto &Spec = AllSpecs.emplace_back(F, S, Score, SpecSize);
      if (CS.getFunction() != F)
        Spec.CallSites.push_back(&CS);
      const unsigned Index = AllSpecs.size() - 1;
      UniqueSpecs[S] = Index;
      if (auto [It, Inserted] = SM.try_emplace(F, Index, Index + 1); !Inserted)
        It->second.second = Index + 1;
    }
  }

  return !UniqueSpecs.empty();
}

bool FunctionSpecializer::isCandidateFunction(Function *F) {
  if (F->isDeclaration() || F->arg_empty())
    return false;

  if (F->hasFnAttribute(Attribute::NoDuplicate))
    return false;

  // Do not specialize the cloned function again.
  if (Specializations.contains(F))
    return false;

  // If we're optimizing the function for size, we shouldn't specialize it.
  if (shouldOptimizeForSize(F, nullptr, nullptr, PGSOQueryType::IRPass))
    return false;

  // Exit if the function is not executable. There's no point in specializing
  // a dead function.
  if (!Solver.isBlockExecutable(&F->getEntryBlock()))
    return false;

  // It wastes time to specialize a function which would get inlined finally.
  if (F->hasFnAttribute(Attribute::AlwaysInline))
    return false;

  LLVM_DEBUG(dbgs() << "FnSpecialization: Try function: " << F->getName()
                    << "\n");
  return true;
}

Function *FunctionSpecializer::createSpecialization(Function *F,
                                                    const SpecSig &S) {
  Function *Clone = cloneCandidateFunction(F, Specializations.size() + 1);

  // The original function does not neccessarily have internal linkage, but the
  // clone must.
  Clone->setLinkage(GlobalValue::InternalLinkage);

  if (F->getEntryCount() && !ProfcheckDisableMetadataFixes)
    Clone->setEntryCount(0);

  // Initialize the lattice state of the arguments of the function clone,
  // marking the argument on which we specialized the function constant
  // with the given value.
  Solver.setLatticeValueForSpecializationArguments(Clone, S.Args);
  Solver.markBlockExecutable(&Clone->front());
  Solver.addArgumentTrackedFunction(Clone);
  Solver.addTrackedFunction(Clone);

  // Mark all the specialized functions
  Specializations.insert(Clone);
  ++NumSpecsCreated;

  return Clone;
}

/// Compute the inlining bonus for replacing argument \p A with constant \p C.
/// The below heuristic is only concerned with exposing inlining
/// opportunities via indirect call promotion. If the argument is not a
/// (potentially casted) function pointer, give up.
unsigned FunctionSpecializer::getInliningBonus(Argument *A, Constant *C) {
  Function *CalledFunction = dyn_cast<Function>(C->stripPointerCasts());
  if (!CalledFunction)
    return 0;

  // Get TTI for the called function (used for the inline cost).
  auto &CalleeTTI = (GetTTI)(*CalledFunction);

  // Look at all the call sites whose called value is the argument.
  // Specializing the function on the argument would allow these indirect
  // calls to be promoted to direct calls. If the indirect call promotion
  // would likely enable the called function to be inlined, specializing is a
  // good idea.
  int InliningBonus = 0;
  for (User *U : A->users()) {
    if (!isa<CallInst>(U) && !isa<InvokeInst>(U))
      continue;
    auto *CS = cast<CallBase>(U);
    if (CS->getCalledOperand() != A)
      continue;
    if (CS->getFunctionType() != CalledFunction->getFunctionType())
      continue;

    // Get the cost of inlining the called function at this call site. Note
    // that this is only an estimate. The called function may eventually
    // change in a way that leads to it not being inlined here, even though
    // inlining looks profitable now. For example, one of its called
    // functions may be inlined into it, making the called function too large
    // to be inlined into this call site.
    //
    // We apply a boost for performing indirect call promotion by increasing
    // the default threshold by the threshold for indirect calls.
    auto Params = getInlineParams();
    Params.DefaultThreshold += InlineConstants::IndirectCallThreshold;
    InlineCost IC =
        getInlineCost(*CS, CalledFunction, Params, CalleeTTI, GetAC, GetTLI);

    // We clamp the bonus for this call to be between zero and the default
    // threshold.
    if (IC.isAlways())
      InliningBonus += Params.DefaultThreshold;
    else if (IC.isVariable() && IC.getCostDelta() > 0)
      InliningBonus += IC.getCostDelta();

    LLVM_DEBUG(dbgs() << "FnSpecialization:   Inlining bonus " << InliningBonus
                      << " for user " << *U << "\n");
  }

  return InliningBonus > 0 ? static_cast<unsigned>(InliningBonus) : 0;
}

/// Determine if it is possible to specialise the function for constant values
/// of the formal parameter \p A.
bool FunctionSpecializer::isArgumentInteresting(Argument *A) {
  // No point in specialization if the argument is unused.
  if (A->user_empty())
    return false;

  Type *Ty = A->getType();
  if (!Ty->isPointerTy() && (!SpecializeLiteralConstant ||
      (!Ty->isIntegerTy() && !Ty->isFloatingPointTy() && !Ty->isStructTy())))
    return false;

  // SCCP solver does not record an argument that will be constructed on
  // stack.
  if (A->hasByValAttr() && !A->getParent()->onlyReadsMemory())
    return false;

  // For non-argument-tracked functions every argument is overdefined.
  if (!Solver.isArgumentTrackedFunction(A->getParent()))
    return true;

  // Check the lattice value and decide if we should attemt to specialize,
  // based on this argument. No point in specialization, if the lattice value
  // is already a constant.
  bool IsOverdefined = Ty->isStructTy()
    ? any_of(Solver.getStructLatticeValueFor(A), SCCPSolver::isOverdefined)
    : SCCPSolver::isOverdefined(Solver.getLatticeValueFor(A));

  LLVM_DEBUG(
    if (IsOverdefined)
      dbgs() << "FnSpecialization: Found interesting parameter "
             << A->getNameOrAsOperand() << "\n";
    else
      dbgs() << "FnSpecialization: Nothing to do, parameter "
             << A->getNameOrAsOperand() << " is already constant\n";
  );
  return IsOverdefined;
}

/// Check if the value \p V  (an actual argument) is a constant or can only
/// have a constant value. Return that constant.
Constant *FunctionSpecializer::getCandidateConstant(Value *V) {
  if (isa<PoisonValue>(V))
    return nullptr;

  // Select for possible specialisation values that are constants or
  // are deduced to be constants or constant ranges with a single element.
  Constant *C = dyn_cast<Constant>(V);
  if (!C)
    C = Solver.getConstantOrNull(V);

  // Don't specialize on (anything derived from) the address of a non-constant
  // global variable, unless explicitly enabled.
  if (C && C->getType()->isPointerTy() && !C->isNullValue())
    if (auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(C));
        GV && !(GV->isConstant() || SpecializeOnAddress))
      return nullptr;

  return C;
}

void FunctionSpecializer::updateCallSites(Function *F, const Spec *Begin,
                                          const Spec *End) {
  // Collect the call sites that need updating.
  SmallVector<CallBase *> ToUpdate;
  for (User *U : F->users())
    if (auto *CS = dyn_cast<CallBase>(U);
        CS && CS->getCalledFunction() == F &&
        Solver.isBlockExecutable(CS->getParent()))
      ToUpdate.push_back(CS);

  unsigned NCallsLeft = ToUpdate.size();
  for (CallBase *CS : ToUpdate) {
    bool ShouldDecrementCount = CS->getFunction() == F;

    // Find the best matching specialisation.
    const Spec *BestSpec = nullptr;
    for (const Spec &S : make_range(Begin, End)) {
      if (!S.Clone || (BestSpec && S.Score <= BestSpec->Score))
        continue;

      if (any_of(S.Sig.Args, [CS, this](const ArgInfo &Arg) {
            unsigned ArgNo = Arg.Formal->getArgNo();
            return getCandidateConstant(CS->getArgOperand(ArgNo)) != Arg.Actual;
          }))
        continue;

      BestSpec = &S;
    }

    if (BestSpec) {
      LLVM_DEBUG(dbgs() << "FnSpecialization: Redirecting " << *CS
                        << " to call " << BestSpec->Clone->getName() << "\n");
      CS->setCalledFunction(BestSpec->Clone);
      ShouldDecrementCount = true;
    }

    if (ShouldDecrementCount)
      --NCallsLeft;
  }

  // If the function has been completely specialized, the original function
  // is no longer needed. Mark it unreachable.
  // NOTE: If the address of a function is taken, we cannot treat it as dead
  // function.
  if (NCallsLeft == 0 && Solver.isArgumentTrackedFunction(F) &&
      !F->hasAddressTaken()) {
    Solver.markFunctionUnreachable(F);
    DeadFunctions.insert(F);
  }
}
