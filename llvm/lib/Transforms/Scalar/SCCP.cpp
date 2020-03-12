//===- SCCP.cpp - Sparse Conditional Constant Propagation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements sparse conditional constant propagation and merging:
//
// Specifically, this:
//   * Assumes values are constant unless proven otherwise
//   * Assumes BasicBlocks are dead unless proven otherwise
//   * Proves values to be constant, and replaces them with constants
//   * Proves conditional branches to be unconditional
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueLattice.h"
#include "llvm/Analysis/ValueLatticeUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PredicateInfo.h"
#include <cassert>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "sccp"

STATISTIC(NumInstRemoved, "Number of instructions removed");
STATISTIC(NumDeadBlocks , "Number of basic blocks unreachable");

STATISTIC(IPNumInstRemoved, "Number of instructions removed by IPSCCP");
STATISTIC(IPNumArgsElimed ,"Number of arguments constant propagated by IPSCCP");
STATISTIC(IPNumGlobalConst, "Number of globals found to be constant by IPSCCP");

namespace {

// Use ValueLatticeElement as LatticeVal.
using LatticeVal = ValueLatticeElement;

// Helper to check if \p LV is either a constant or a constant
// range with a single element. This should cover exactly the same cases as the
// old LatticeVal::isConstant() and is intended to be used in the transition
// from LatticeVal to LatticeValueElement.
bool isConstant(const LatticeVal &LV) {
  return LV.isConstant() ||
         (LV.isConstantRange() && LV.getConstantRange().isSingleElement());
}

// Helper to check if \p LV is either overdefined or a constant range with more
// than a single element. This should cover exactly the same cases as the old
// LatticeVal::isOverdefined() and is intended to be used in the transition from
// LatticeVal to LatticeValueElement.
bool isOverdefined(const LatticeVal &LV) {
  return LV.isOverdefined() ||
         (LV.isConstantRange() && !LV.getConstantRange().isSingleElement());
}

//===----------------------------------------------------------------------===//
//
/// SCCPSolver - This class is a general purpose solver for Sparse Conditional
/// Constant Propagation.
///
class SCCPSolver : public InstVisitor<SCCPSolver> {
  const DataLayout &DL;
  std::function<const TargetLibraryInfo &(Function &)> GetTLI;
  SmallPtrSet<BasicBlock *, 8> BBExecutable; // The BBs that are executable.
  DenseMap<Value *, LatticeVal> ValueState;  // The state each value is in.

  /// StructValueState - This maintains ValueState for values that have
  /// StructType, for example for formal arguments, calls, insertelement, etc.
  DenseMap<std::pair<Value *, unsigned>, LatticeVal> StructValueState;

  /// GlobalValue - If we are tracking any values for the contents of a global
  /// variable, we keep a mapping from the constant accessor to the element of
  /// the global, to the currently known value.  If the value becomes
  /// overdefined, it's entry is simply removed from this map.
  DenseMap<GlobalVariable *, LatticeVal> TrackedGlobals;

  /// TrackedRetVals - If we are tracking arguments into and the return
  /// value out of a function, it will have an entry in this map, indicating
  /// what the known return value for the function is.
  MapVector<Function *, LatticeVal> TrackedRetVals;

  /// TrackedMultipleRetVals - Same as TrackedRetVals, but used for functions
  /// that return multiple values.
  MapVector<std::pair<Function *, unsigned>, LatticeVal> TrackedMultipleRetVals;

  /// MRVFunctionsTracked - Each function in TrackedMultipleRetVals is
  /// represented here for efficient lookup.
  SmallPtrSet<Function *, 16> MRVFunctionsTracked;

  /// MustTailFunctions - Each function here is a callee of non-removable
  /// musttail call site.
  SmallPtrSet<Function *, 16> MustTailCallees;

  /// TrackingIncomingArguments - This is the set of functions for whose
  /// arguments we make optimistic assumptions about and try to prove as
  /// constants.
  SmallPtrSet<Function *, 16> TrackingIncomingArguments;

  /// The reason for two worklists is that overdefined is the lowest state
  /// on the lattice, and moving things to overdefined as fast as possible
  /// makes SCCP converge much faster.
  ///
  /// By having a separate worklist, we accomplish this because everything
  /// possibly overdefined will become overdefined at the soonest possible
  /// point.
  SmallVector<Value *, 64> OverdefinedInstWorkList;
  SmallVector<Value *, 64> InstWorkList;

  // The BasicBlock work list
  SmallVector<BasicBlock *, 64>  BBWorkList;

  /// KnownFeasibleEdges - Entries in this set are edges which have already had
  /// PHI nodes retriggered.
  using Edge = std::pair<BasicBlock *, BasicBlock *>;
  DenseSet<Edge> KnownFeasibleEdges;

  DenseMap<Function *, AnalysisResultsForFn> AnalysisResults;
  DenseMap<Value *, SmallPtrSet<User *, 2>> AdditionalUsers;

  LLVMContext &Ctx;

public:
  void addAnalysis(Function &F, AnalysisResultsForFn A) {
    AnalysisResults.insert({&F, std::move(A)});
  }

  const PredicateBase *getPredicateInfoFor(Instruction *I) {
    auto A = AnalysisResults.find(I->getParent()->getParent());
    if (A == AnalysisResults.end())
      return nullptr;
    return A->second.PredInfo->getPredicateInfoFor(I);
  }

  DomTreeUpdater getDTU(Function &F) {
    auto A = AnalysisResults.find(&F);
    assert(A != AnalysisResults.end() && "Need analysis results for function.");
    return {A->second.DT, A->second.PDT, DomTreeUpdater::UpdateStrategy::Lazy};
  }

  SCCPSolver(const DataLayout &DL,
             std::function<const TargetLibraryInfo &(Function &)> GetTLI,
             LLVMContext &Ctx)
      : DL(DL), GetTLI(std::move(GetTLI)), Ctx(Ctx) {}

  /// MarkBlockExecutable - This method can be used by clients to mark all of
  /// the blocks that are known to be intrinsically live in the processed unit.
  ///
  /// This returns true if the block was not considered live before.
  bool MarkBlockExecutable(BasicBlock *BB) {
    if (!BBExecutable.insert(BB).second)
      return false;
    LLVM_DEBUG(dbgs() << "Marking Block Executable: " << BB->getName() << '\n');
    BBWorkList.push_back(BB);  // Add the block to the work list!
    return true;
  }

  /// TrackValueOfGlobalVariable - Clients can use this method to
  /// inform the SCCPSolver that it should track loads and stores to the
  /// specified global variable if it can.  This is only legal to call if
  /// performing Interprocedural SCCP.
  void TrackValueOfGlobalVariable(GlobalVariable *GV) {
    // We only track the contents of scalar globals.
    if (GV->getValueType()->isSingleValueType()) {
      LatticeVal &IV = TrackedGlobals[GV];
      if (!isa<UndefValue>(GV->getInitializer()))
        IV.markConstant(GV->getInitializer());
    }
  }

  /// AddTrackedFunction - If the SCCP solver is supposed to track calls into
  /// and out of the specified function (which cannot have its address taken),
  /// this method must be called.
  void AddTrackedFunction(Function *F) {
    // Add an entry, F -> undef.
    if (auto *STy = dyn_cast<StructType>(F->getReturnType())) {
      MRVFunctionsTracked.insert(F);
      for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
        TrackedMultipleRetVals.insert(std::make_pair(std::make_pair(F, i),
                                                     LatticeVal()));
    } else
      TrackedRetVals.insert(std::make_pair(F, LatticeVal()));
  }

  /// AddMustTailCallee - If the SCCP solver finds that this function is called
  /// from non-removable musttail call site.
  void AddMustTailCallee(Function *F) {
    MustTailCallees.insert(F);
  }

  /// Returns true if the given function is called from non-removable musttail
  /// call site.
  bool isMustTailCallee(Function *F) {
    return MustTailCallees.count(F);
  }

  void AddArgumentTrackedFunction(Function *F) {
    TrackingIncomingArguments.insert(F);
  }

  /// Returns true if the given function is in the solver's set of
  /// argument-tracked functions.
  bool isArgumentTrackedFunction(Function *F) {
    return TrackingIncomingArguments.count(F);
  }

  /// Solve - Solve for constants and executable blocks.
  void Solve();

  /// ResolvedUndefsIn - While solving the dataflow for a function, we assume
  /// that branches on undef values cannot reach any of their successors.
  /// However, this is not a safe assumption.  After we solve dataflow, this
  /// method should be use to handle this.  If this returns true, the solver
  /// should be rerun.
  bool ResolvedUndefsIn(Function &F);

  bool isBlockExecutable(BasicBlock *BB) const {
    return BBExecutable.count(BB);
  }

  // isEdgeFeasible - Return true if the control flow edge from the 'From' basic
  // block to the 'To' basic block is currently feasible.
  bool isEdgeFeasible(BasicBlock *From, BasicBlock *To);

  std::vector<LatticeVal> getStructLatticeValueFor(Value *V) const {
    std::vector<LatticeVal> StructValues;
    auto *STy = dyn_cast<StructType>(V->getType());
    assert(STy && "getStructLatticeValueFor() can be called only on structs");
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      auto I = StructValueState.find(std::make_pair(V, i));
      assert(I != StructValueState.end() && "Value not in valuemap!");
      StructValues.push_back(I->second);
    }
    return StructValues;
  }

  const LatticeVal &getLatticeValueFor(Value *V) const {
    assert(!V->getType()->isStructTy() &&
           "Should use getStructLatticeValueFor");
    DenseMap<Value *, LatticeVal>::const_iterator I = ValueState.find(V);
    assert(I != ValueState.end() &&
           "V not found in ValueState nor Paramstate map!");
    return I->second;
  }

  /// getTrackedRetVals - Get the inferred return value map.
  const MapVector<Function*, LatticeVal> &getTrackedRetVals() {
    return TrackedRetVals;
  }

  /// getTrackedGlobals - Get and return the set of inferred initializers for
  /// global variables.
  const DenseMap<GlobalVariable*, LatticeVal> &getTrackedGlobals() {
    return TrackedGlobals;
  }

  /// getMRVFunctionsTracked - Get the set of functions which return multiple
  /// values tracked by the pass.
  const SmallPtrSet<Function *, 16> getMRVFunctionsTracked() {
    return MRVFunctionsTracked;
  }

  /// getMustTailCallees - Get the set of functions which are called
  /// from non-removable musttail call sites.
  const SmallPtrSet<Function *, 16> getMustTailCallees() {
    return MustTailCallees;
  }

  /// markOverdefined - Mark the specified value overdefined.  This
  /// works with both scalars and structs.
  void markOverdefined(Value *V) {
    if (auto *STy = dyn_cast<StructType>(V->getType()))
      for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
        markOverdefined(getStructValueState(V, i), V);
    else
      markOverdefined(ValueState[V], V);
  }

  // isStructLatticeConstant - Return true if all the lattice values
  // corresponding to elements of the structure are constants,
  // false otherwise.
  bool isStructLatticeConstant(Function *F, StructType *STy) {
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      const auto &It = TrackedMultipleRetVals.find(std::make_pair(F, i));
      assert(It != TrackedMultipleRetVals.end());
      LatticeVal LV = It->second;
      if (!isConstant(LV))
        return false;
    }
    return true;
  }

  /// Helper to return a Constant if \p LV is either a constant or a constant
  /// range with a single element.
  Constant *getConstant(const LatticeVal &LV) const {
    if (LV.isConstant())
      return LV.getConstant();

    if (LV.isConstantRange()) {
      auto &CR = LV.getConstantRange();
      if (CR.getSingleElement())
        return ConstantInt::get(Ctx, *CR.getSingleElement());
    }
    return nullptr;
  }

private:
  ConstantInt *getConstantInt(const LatticeVal &IV) const {
    return dyn_cast_or_null<ConstantInt>(getConstant(IV));
  }

  // pushToWorkList - Helper for markConstant/markOverdefined
  void pushToWorkList(LatticeVal &IV, Value *V) {
    if (isOverdefined(IV))
      return OverdefinedInstWorkList.push_back(V);
    InstWorkList.push_back(V);
  }

  // Helper to push \p V to the worklist, after updating it to \p IV. Also
  // prints a debug message with the updated value.
  void pushToWorkListMsg(LatticeVal &IV, Value *V) {
    LLVM_DEBUG(dbgs() << "updated " << IV << ": " << *V << '\n');
    if (isOverdefined(IV))
      return OverdefinedInstWorkList.push_back(V);
    InstWorkList.push_back(V);
  }

  // markConstant - Make a value be marked as "constant".  If the value
  // is not already a constant, add it to the instruction work list so that
  // the users of the instruction are updated later.
  bool markConstant(LatticeVal &IV, Value *V, Constant *C) {
    if (!IV.markConstant(C)) return false;
    LLVM_DEBUG(dbgs() << "markConstant: " << *C << ": " << *V << '\n');
    pushToWorkList(IV, V);
    return true;
  }

  bool markConstant(Value *V, Constant *C) {
    assert(!V->getType()->isStructTy() && "structs should use mergeInValue");
    return markConstant(ValueState[V], V, C);
  }

  // markOverdefined - Make a value be marked as "overdefined". If the
  // value is not already overdefined, add it to the overdefined instruction
  // work list so that the users of the instruction are updated later.
  bool markOverdefined(LatticeVal &IV, Value *V) {
    if (!IV.markOverdefined()) return false;

    LLVM_DEBUG(dbgs() << "markOverdefined: ";
               if (auto *F = dyn_cast<Function>(V)) dbgs()
               << "Function '" << F->getName() << "'\n";
               else dbgs() << *V << '\n');
    // Only instructions go on the work list
    pushToWorkList(IV, V);
    return true;
  }

  bool mergeInValue(LatticeVal &IV, Value *V, LatticeVal MergeWithV,
                    bool Widen = true) {
    // Do a simple form of widening, to avoid extending a range repeatedly in a
    // loop. If IV is a constant range, it means we already set it once. If
    // MergeWithV would extend IV, mark V as overdefined.
    if (Widen && IV.isConstantRange() && MergeWithV.isConstantRange() &&
        !IV.getConstantRange().contains(MergeWithV.getConstantRange())) {
      markOverdefined(IV, V);
      return true;
    }
    if (IV.mergeIn(MergeWithV, DL)) {
      pushToWorkList(IV, V);
      LLVM_DEBUG(dbgs() << "Merged " << MergeWithV << " into " << *V << " : "
                        << IV << "\n");
      return true;
    }
    return false;
  }

  bool mergeInValue(Value *V, LatticeVal MergeWithV, bool Widen = true) {
    assert(!V->getType()->isStructTy() &&
           "non-structs should use markConstant");
    return mergeInValue(ValueState[V], V, MergeWithV, Widen);
  }

  /// getValueState - Return the LatticeVal object that corresponds to the
  /// value.  This function handles the case when the value hasn't been seen yet
  /// by properly seeding constants etc.
  LatticeVal &getValueState(Value *V) {
    assert(!V->getType()->isStructTy() && "Should use getStructValueState");

    std::pair<DenseMap<Value*, LatticeVal>::iterator, bool> I =
      ValueState.insert(std::make_pair(V, LatticeVal()));
    LatticeVal &LV = I.first->second;

    if (!I.second)
      return LV;  // Common case, already in the map.

    if (auto *C = dyn_cast<Constant>(V)) {
      // Undef values remain unknown.
      if (!isa<UndefValue>(V))
        LV.markConstant(C);          // Constants are constant
    }

    // All others are underdefined by default.
    return LV;
  }

  LatticeVal toLatticeVal(const ValueLatticeElement &V, Type *T) {
    LatticeVal Res;
    if (V.isUndefined())
      return Res;

    if (V.isConstant()) {
      Res.markConstant(V.getConstant());
      return Res;
    }
    if (V.isConstantRange() && V.getConstantRange().isSingleElement()) {
      Res.markConstant(
          ConstantInt::get(T, *V.getConstantRange().getSingleElement()));
      return Res;
    }
    Res.markOverdefined();
    return Res;
  }

  /// getStructValueState - Return the LatticeVal object that corresponds to the
  /// value/field pair.  This function handles the case when the value hasn't
  /// been seen yet by properly seeding constants etc.
  LatticeVal &getStructValueState(Value *V, unsigned i) {
    assert(V->getType()->isStructTy() && "Should use getValueState");
    assert(i < cast<StructType>(V->getType())->getNumElements() &&
           "Invalid element #");

    std::pair<DenseMap<std::pair<Value*, unsigned>, LatticeVal>::iterator,
              bool> I = StructValueState.insert(
                        std::make_pair(std::make_pair(V, i), LatticeVal()));
    LatticeVal &LV = I.first->second;

    if (!I.second)
      return LV;  // Common case, already in the map.

    if (auto *C = dyn_cast<Constant>(V)) {
      Constant *Elt = C->getAggregateElement(i);

      if (!Elt)
        LV.markOverdefined();      // Unknown sort of constant.
      else if (isa<UndefValue>(Elt))
        ; // Undef values remain unknown.
      else
        LV.markConstant(Elt);      // Constants are constant.
    }

    // All others are underdefined by default.
    return LV;
  }

  /// markEdgeExecutable - Mark a basic block as executable, adding it to the BB
  /// work list if it is not already executable.
  bool markEdgeExecutable(BasicBlock *Source, BasicBlock *Dest) {
    if (!KnownFeasibleEdges.insert(Edge(Source, Dest)).second)
      return false;  // This edge is already known to be executable!

    if (!MarkBlockExecutable(Dest)) {
      // If the destination is already executable, we just made an *edge*
      // feasible that wasn't before.  Revisit the PHI nodes in the block
      // because they have potentially new operands.
      LLVM_DEBUG(dbgs() << "Marking Edge Executable: " << Source->getName()
                        << " -> " << Dest->getName() << '\n');

      for (PHINode &PN : Dest->phis())
        visitPHINode(PN);
    }
    return true;
  }

  // getFeasibleSuccessors - Return a vector of booleans to indicate which
  // successors are reachable from a given terminator instruction.
  void getFeasibleSuccessors(Instruction &TI, SmallVectorImpl<bool> &Succs);

  // OperandChangedState - This method is invoked on all of the users of an
  // instruction that was just changed state somehow.  Based on this
  // information, we need to update the specified user of this instruction.
  void OperandChangedState(Instruction *I) {
    if (BBExecutable.count(I->getParent()))   // Inst is executable?
      visit(*I);
  }

  // Add U as additional user of V.
  void addAdditionalUser(Value *V, User *U) {
    auto Iter = AdditionalUsers.insert({V, {}});
    Iter.first->second.insert(U);
  }

  // Mark I's users as changed, including AdditionalUsers.
  void markUsersAsChanged(Value *I) {
    for (User *U : I->users())
      if (auto *UI = dyn_cast<Instruction>(U))
        OperandChangedState(UI);

    auto Iter = AdditionalUsers.find(I);
    if (Iter != AdditionalUsers.end()) {
      for (User *U : Iter->second)
        if (auto *UI = dyn_cast<Instruction>(U))
          OperandChangedState(UI);
    }
  }

private:
  friend class InstVisitor<SCCPSolver>;

  // visit implementations - Something changed in this instruction.  Either an
  // operand made a transition, or the instruction is newly executable.  Change
  // the value type of I to reflect these changes if appropriate.
  void visitPHINode(PHINode &I);

  // Terminators

  void visitReturnInst(ReturnInst &I);
  void visitTerminator(Instruction &TI);

  void visitCastInst(CastInst &I);
  void visitSelectInst(SelectInst &I);
  void visitUnaryOperator(Instruction &I);
  void visitBinaryOperator(Instruction &I);
  void visitCmpInst(CmpInst &I);
  void visitExtractValueInst(ExtractValueInst &EVI);
  void visitInsertValueInst(InsertValueInst &IVI);

  void visitCatchSwitchInst(CatchSwitchInst &CPI) {
    markOverdefined(&CPI);
    visitTerminator(CPI);
  }

  // Instructions that cannot be folded away.

  void visitStoreInst     (StoreInst &I);
  void visitLoadInst      (LoadInst &I);
  void visitGetElementPtrInst(GetElementPtrInst &I);

  void visitCallInst      (CallInst &I) {
    visitCallSite(&I);
  }

  void visitInvokeInst    (InvokeInst &II) {
    visitCallSite(&II);
    visitTerminator(II);
  }

  void visitCallBrInst    (CallBrInst &CBI) {
    visitCallSite(&CBI);
    visitTerminator(CBI);
  }

  void visitCallSite      (CallSite CS);
  void visitResumeInst    (ResumeInst &I) { /*returns void*/ }
  void visitUnreachableInst(UnreachableInst &I) { /*returns void*/ }
  void visitFenceInst     (FenceInst &I) { /*returns void*/ }

  void visitInstruction(Instruction &I) {
    // All the instructions we don't do any special handling for just
    // go to overdefined.
    LLVM_DEBUG(dbgs() << "SCCP: Don't know how to handle: " << I << '\n');
    markOverdefined(&I);
  }
};

} // end anonymous namespace

// getFeasibleSuccessors - Return a vector of booleans to indicate which
// successors are reachable from a given terminator instruction.
void SCCPSolver::getFeasibleSuccessors(Instruction &TI,
                                       SmallVectorImpl<bool> &Succs) {
  Succs.resize(TI.getNumSuccessors());
  if (auto *BI = dyn_cast<BranchInst>(&TI)) {
    if (BI->isUnconditional()) {
      Succs[0] = true;
      return;
    }

    LatticeVal BCValue = getValueState(BI->getCondition());
    ConstantInt *CI = getConstantInt(BCValue);
    if (!CI) {
      // Overdefined condition variables, and branches on unfoldable constant
      // conditions, mean the branch could go either way.
      if (!BCValue.isUnknown())
        Succs[0] = Succs[1] = true;
      return;
    }

    // Constant condition variables mean the branch can only go a single way.
    Succs[CI->isZero()] = true;
    return;
  }

  // Unwinding instructions successors are always executable.
  if (TI.isExceptionalTerminator()) {
    Succs.assign(TI.getNumSuccessors(), true);
    return;
  }

  if (auto *SI = dyn_cast<SwitchInst>(&TI)) {
    if (!SI->getNumCases()) {
      Succs[0] = true;
      return;
    }
    LatticeVal SCValue = getValueState(SI->getCondition());
    ConstantInt *CI = getConstantInt(SCValue);

    if (!CI) {   // Overdefined or unknown condition?
      // All destinations are executable!
      if (!SCValue.isUnknown())
        Succs.assign(TI.getNumSuccessors(), true);
      return;
    }

    Succs[SI->findCaseValue(CI)->getSuccessorIndex()] = true;
    return;
  }

  // In case of indirect branch and its address is a blockaddress, we mark
  // the target as executable.
  if (auto *IBR = dyn_cast<IndirectBrInst>(&TI)) {
    // Casts are folded by visitCastInst.
    LatticeVal IBRValue = getValueState(IBR->getAddress());
    BlockAddress *Addr = dyn_cast_or_null<BlockAddress>(getConstant(IBRValue));
    if (!Addr) {   // Overdefined or unknown condition?
      // All destinations are executable!
      if (!IBRValue.isUnknown())
        Succs.assign(TI.getNumSuccessors(), true);
      return;
    }

    BasicBlock* T = Addr->getBasicBlock();
    assert(Addr->getFunction() == T->getParent() &&
           "Block address of a different function ?");
    for (unsigned i = 0; i < IBR->getNumSuccessors(); ++i) {
      // This is the target.
      if (IBR->getDestination(i) == T) {
        Succs[i] = true;
        return;
      }
    }

    // If we didn't find our destination in the IBR successor list, then we
    // have undefined behavior. Its ok to assume no successor is executable.
    return;
  }

  // In case of callbr, we pessimistically assume that all successors are
  // feasible.
  if (isa<CallBrInst>(&TI)) {
    Succs.assign(TI.getNumSuccessors(), true);
    return;
  }

  LLVM_DEBUG(dbgs() << "Unknown terminator instruction: " << TI << '\n');
  llvm_unreachable("SCCP: Don't know how to handle this terminator!");
}

// isEdgeFeasible - Return true if the control flow edge from the 'From' basic
// block to the 'To' basic block is currently feasible.
bool SCCPSolver::isEdgeFeasible(BasicBlock *From, BasicBlock *To) {
  // Check if we've called markEdgeExecutable on the edge yet. (We could
  // be more aggressive and try to consider edges which haven't been marked
  // yet, but there isn't any need.)
  return KnownFeasibleEdges.count(Edge(From, To));
}

// visit Implementations - Something changed in this instruction, either an
// operand made a transition, or the instruction is newly executable.  Change
// the value type of I to reflect these changes if appropriate.  This method
// makes sure to do the following actions:
//
// 1. If a phi node merges two constants in, and has conflicting value coming
//    from different branches, or if the PHI node merges in an overdefined
//    value, then the PHI node becomes overdefined.
// 2. If a phi node merges only constants in, and they all agree on value, the
//    PHI node becomes a constant value equal to that.
// 3. If V <- x (op) y && isConstant(x) && isConstant(y) V = Constant
// 4. If V <- x (op) y && (isOverdefined(x) || isOverdefined(y)) V = Overdefined
// 5. If V <- MEM or V <- CALL or V <- (unknown) then V = Overdefined
// 6. If a conditional branch has a value that is constant, make the selected
//    destination executable
// 7. If a conditional branch has a value that is overdefined, make all
//    successors executable.
void SCCPSolver::visitPHINode(PHINode &PN) {
  // If this PN returns a struct, just mark the result overdefined.
  // TODO: We could do a lot better than this if code actually uses this.
  if (PN.getType()->isStructTy())
    return (void)markOverdefined(&PN);

  if (isOverdefined(getValueState(&PN))) {
    return (void)markOverdefined(&PN);
  }

  // Super-extra-high-degree PHI nodes are unlikely to ever be marked constant,
  // and slow us down a lot.  Just mark them overdefined.
  if (PN.getNumIncomingValues() > 64)
    return (void)markOverdefined(&PN);

  // Look at all of the executable operands of the PHI node.  If any of them
  // are overdefined, the PHI becomes overdefined as well.  If they are all
  // constant, and they agree with each other, the PHI becomes the identical
  // constant.  If they are constant and don't agree, the PHI is overdefined.
  // If there are no executable operands, the PHI remains unknown.
  Constant *OperandVal = nullptr;
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
    LatticeVal IV = getValueState(PN.getIncomingValue(i));
    if (IV.isUnknown()) continue;  // Doesn't influence PHI node.

    if (!isEdgeFeasible(PN.getIncomingBlock(i), PN.getParent()))
      continue;

    if (isOverdefined(IV)) // PHI node becomes overdefined!
      return (void)markOverdefined(&PN);

    if (!OperandVal) {   // Grab the first value.
      OperandVal = getConstant(IV);
      continue;
    }

    // There is already a reachable operand.  If we conflict with it,
    // then the PHI node becomes overdefined.  If we agree with it, we
    // can continue on.

    // Check to see if there are two different constants merging, if so, the PHI
    // node is overdefined.
    if (getConstant(IV) != OperandVal)
      return (void)markOverdefined(&PN);
  }

  // If we exited the loop, this means that the PHI node only has constant
  // arguments that agree with each other(and OperandVal is the constant) or
  // OperandVal is null because there are no defined incoming arguments.  If
  // this is the case, the PHI remains unknown.
  if (OperandVal)
    markConstant(&PN, OperandVal);      // Acquire operand value
}

void SCCPSolver::visitReturnInst(ReturnInst &I) {
  if (I.getNumOperands() == 0) return;  // ret void

  // ResolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (isOverdefined(ValueState[&I]))
    return;

  Function *F = I.getParent()->getParent();
  Value *ResultOp = I.getOperand(0);

  // If we are tracking the return value of this function, merge it in.
  if (!TrackedRetVals.empty() && !ResultOp->getType()->isStructTy()) {
    MapVector<Function*, LatticeVal>::iterator TFRVI =
      TrackedRetVals.find(F);
    if (TFRVI != TrackedRetVals.end()) {
      mergeInValue(TFRVI->second, F, getValueState(ResultOp));
      return;
    }
  }

  // Handle functions that return multiple values.
  if (!TrackedMultipleRetVals.empty()) {
    if (auto *STy = dyn_cast<StructType>(ResultOp->getType()))
      if (MRVFunctionsTracked.count(F))
        for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
          mergeInValue(TrackedMultipleRetVals[std::make_pair(F, i)], F,
                       getStructValueState(ResultOp, i));
  }
}

void SCCPSolver::visitTerminator(Instruction &TI) {
  SmallVector<bool, 16> SuccFeasible;
  getFeasibleSuccessors(TI, SuccFeasible);

  BasicBlock *BB = TI.getParent();

  // Mark all feasible successors executable.
  for (unsigned i = 0, e = SuccFeasible.size(); i != e; ++i)
    if (SuccFeasible[i])
      markEdgeExecutable(BB, TI.getSuccessor(i));
}

void SCCPSolver::visitCastInst(CastInst &I) {
  // ResolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (isOverdefined(ValueState[&I]))
    return;

  LatticeVal OpSt = getValueState(I.getOperand(0));
  if (Constant *OpC = getConstant(OpSt)) {
    // Fold the constant as we build.
    Constant *C = ConstantFoldCastOperand(I.getOpcode(), OpC, I.getType(), DL);
    if (isa<UndefValue>(C))
      return;
    // Propagate constant value
    markConstant(&I, C);
  } else if (!OpSt.isUnknown())
    markOverdefined(&I);
}

void SCCPSolver::visitExtractValueInst(ExtractValueInst &EVI) {
  // ResolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (isOverdefined(ValueState[&EVI]))
    return;

  // If this returns a struct, mark all elements over defined, we don't track
  // structs in structs.
  if (EVI.getType()->isStructTy())
    return (void)markOverdefined(&EVI);

  // If this is extracting from more than one level of struct, we don't know.
  if (EVI.getNumIndices() != 1)
    return (void)markOverdefined(&EVI);

  Value *AggVal = EVI.getAggregateOperand();
  if (AggVal->getType()->isStructTy()) {
    unsigned i = *EVI.idx_begin();
    LatticeVal EltVal = getStructValueState(AggVal, i);
    mergeInValue(getValueState(&EVI), &EVI, EltVal);
  } else {
    // Otherwise, must be extracting from an array.
    return (void)markOverdefined(&EVI);
  }
}

void SCCPSolver::visitInsertValueInst(InsertValueInst &IVI) {
  auto *STy = dyn_cast<StructType>(IVI.getType());
  if (!STy)
    return (void)markOverdefined(&IVI);

  // ResolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (isOverdefined(ValueState[&IVI]))
    return;

  // If this has more than one index, we can't handle it, drive all results to
  // undef.
  if (IVI.getNumIndices() != 1)
    return (void)markOverdefined(&IVI);

  Value *Aggr = IVI.getAggregateOperand();
  unsigned Idx = *IVI.idx_begin();

  // Compute the result based on what we're inserting.
  for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
    // This passes through all values that aren't the inserted element.
    if (i != Idx) {
      LatticeVal EltVal = getStructValueState(Aggr, i);
      mergeInValue(getStructValueState(&IVI, i), &IVI, EltVal);
      continue;
    }

    Value *Val = IVI.getInsertedValueOperand();
    if (Val->getType()->isStructTy())
      // We don't track structs in structs.
      markOverdefined(getStructValueState(&IVI, i), &IVI);
    else {
      LatticeVal InVal = getValueState(Val);
      mergeInValue(getStructValueState(&IVI, i), &IVI, InVal);
    }
  }
}

void SCCPSolver::visitSelectInst(SelectInst &I) {
  // If this select returns a struct, just mark the result overdefined.
  // TODO: We could do a lot better than this if code actually uses this.
  if (I.getType()->isStructTy())
    return (void)markOverdefined(&I);

  // ResolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (isOverdefined(ValueState[&I]))
    return;

  LatticeVal CondValue = getValueState(I.getCondition());
  if (CondValue.isUnknown())
    return;

  if (ConstantInt *CondCB = getConstantInt(CondValue)) {
    Value *OpVal = CondCB->isZero() ? I.getFalseValue() : I.getTrueValue();
    mergeInValue(&I, getValueState(OpVal));
    return;
  }

  // Otherwise, the condition is overdefined or a constant we can't evaluate.
  // See if we can produce something better than overdefined based on the T/F
  // value.
  LatticeVal TVal = getValueState(I.getTrueValue());
  LatticeVal FVal = getValueState(I.getFalseValue());

  // select ?, C, C -> C.
  if (isConstant(TVal) && isConstant(FVal) &&
      getConstant(TVal) == getConstant(FVal))
    return (void)markConstant(&I, getConstant(FVal));

  if (TVal.isUnknown())   // select ?, undef, X -> X.
    return (void)mergeInValue(&I, FVal);
  if (FVal.isUnknown())   // select ?, X, undef -> X.
    return (void)mergeInValue(&I, TVal);
  markOverdefined(&I);
}

// Handle Unary Operators.
void SCCPSolver::visitUnaryOperator(Instruction &I) {
  LatticeVal V0State = getValueState(I.getOperand(0));

  LatticeVal &IV = ValueState[&I];
  // ResolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (isOverdefined(IV)) return;

  if (isConstant(V0State)) {
    Constant *C = ConstantExpr::get(I.getOpcode(), getConstant(V0State));

    // op Y -> undef.
    if (isa<UndefValue>(C))
      return;
    return (void)markConstant(IV, &I, C);
  }

  // If something is undef, wait for it to resolve.
  if (!isOverdefined(V0State))
    return;

  markOverdefined(&I);
}

// Handle Binary Operators.
void SCCPSolver::visitBinaryOperator(Instruction &I) {
  LatticeVal V1State = getValueState(I.getOperand(0));
  LatticeVal V2State = getValueState(I.getOperand(1));

  LatticeVal &IV = ValueState[&I];
  if (isOverdefined(IV)) {
    markOverdefined(&I);
    return;
  }

  if (isConstant(V1State) && isConstant(V2State)) {
    Constant *C = ConstantExpr::get(I.getOpcode(), getConstant(V1State),
                                    getConstant(V2State));
    // X op Y -> undef.
    if (isa<UndefValue>(C))
      return;
    return (void)markConstant(IV, &I, C);
  }

  // If something is undef, wait for it to resolve.
  if (V1State.isUnknown() || V2State.isUnknown())
    return;

  // Otherwise, one of our operands is overdefined.  Try to produce something
  // better than overdefined with some tricks.
  // If this is 0 / Y, it doesn't matter that the second operand is
  // overdefined, and we can replace it with zero.
  if (I.getOpcode() == Instruction::UDiv || I.getOpcode() == Instruction::SDiv)
    if (isConstant(V1State) && getConstant(V1State)->isNullValue())
      return (void)markConstant(IV, &I, getConstant(V1State));

  // If this is:
  // -> AND/MUL with 0
  // -> OR with -1
  // it doesn't matter that the other operand is overdefined.
  if (I.getOpcode() == Instruction::And || I.getOpcode() == Instruction::Mul ||
      I.getOpcode() == Instruction::Or) {
    LatticeVal *NonOverdefVal = nullptr;
    if (!isOverdefined(V1State))
      NonOverdefVal = &V1State;

    else if (!isOverdefined(V2State))
      NonOverdefVal = &V2State;
    if (NonOverdefVal) {
      if (NonOverdefVal->isUnknown())
        return;

      if (I.getOpcode() == Instruction::And ||
          I.getOpcode() == Instruction::Mul) {
        // X and 0 = 0
        // X * 0 = 0
        if (getConstant(*NonOverdefVal)->isNullValue())
          return (void)markConstant(IV, &I, getConstant(*NonOverdefVal));
      } else {
        // X or -1 = -1
        if (ConstantInt *CI = getConstantInt(*NonOverdefVal))
          if (CI->isMinusOne())
            return (void)markConstant(IV, &I, CI);
      }
    }
  }

  markOverdefined(&I);
}

// Handle ICmpInst instruction.
void SCCPSolver::visitCmpInst(CmpInst &I) {
  // Do not cache this lookup, getValueState calls later in the function might
  // invalidate the reference.
  if (isOverdefined(ValueState[&I])) {
    markOverdefined(&I);
    return;
  }

  Value *Op1 = I.getOperand(0);
  Value *Op2 = I.getOperand(1);

  // For parameters, use ParamState which includes constant range info if
  // available.
  auto V1State = getValueState(Op1);
  auto V2State = getValueState(Op2);

  Constant *C = V1State.getCompare(I.getPredicate(), I.getType(), V2State);
  if (C) {
    if (isa<UndefValue>(C))
      return;
    LatticeVal CV;
    CV.markConstant(C);
    mergeInValue(&I, CV);
    return;
  }

  // If operands are still unknown, wait for it to resolve.
  if ((V1State.isUnknown() || V2State.isUnknown()) &&
      !isConstant(ValueState[&I]))
    return;

  markOverdefined(&I);
}

// Handle getelementptr instructions.  If all operands are constants then we
// can turn this into a getelementptr ConstantExpr.
void SCCPSolver::visitGetElementPtrInst(GetElementPtrInst &I) {
  if (isOverdefined(ValueState[&I])) return;

  SmallVector<Constant*, 8> Operands;
  Operands.reserve(I.getNumOperands());

  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    LatticeVal State = getValueState(I.getOperand(i));
    if (State.isUnknown())
      return;  // Operands are not resolved yet.

    if (isOverdefined(State))
      return (void)markOverdefined(&I);

    if (Constant *C = getConstant(State)) {
      Operands.push_back(C);
      continue;
    }

    return (void)markOverdefined(&I);
  }

  Constant *Ptr = Operands[0];
  auto Indices = makeArrayRef(Operands.begin() + 1, Operands.end());
  Constant *C =
      ConstantExpr::getGetElementPtr(I.getSourceElementType(), Ptr, Indices);
  if (isa<UndefValue>(C))
      return;
  markConstant(&I, C);
}

void SCCPSolver::visitStoreInst(StoreInst &SI) {
  // If this store is of a struct, ignore it.
  if (SI.getOperand(0)->getType()->isStructTy())
    return;

  if (TrackedGlobals.empty() || !isa<GlobalVariable>(SI.getOperand(1)))
    return;

  // ResolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (isOverdefined(ValueState[&SI]))
    return;

  GlobalVariable *GV = cast<GlobalVariable>(SI.getOperand(1));
  DenseMap<GlobalVariable*, LatticeVal>::iterator I = TrackedGlobals.find(GV);
  if (I == TrackedGlobals.end())
    return;

  // Get the value we are storing into the global, then merge it.
  mergeInValue(I->second, GV, getValueState(SI.getOperand(0)));
  if (isOverdefined(I->second))
    TrackedGlobals.erase(I);      // No need to keep tracking this!
}

// Handle load instructions.  If the operand is a constant pointer to a constant
// global, we can replace the load with the loaded constant value!
void SCCPSolver::visitLoadInst(LoadInst &I) {
  // If this load is of a struct, just mark the result overdefined.
  if (I.getType()->isStructTy())
    return (void)markOverdefined(&I);

  // ResolvedUndefsIn might mark I as overdefined. Bail out, even if we would
  // discover a concrete value later.
  if (isOverdefined(ValueState[&I]))
    return;

  LatticeVal PtrVal = getValueState(I.getOperand(0));
  if (PtrVal.isUnknown()) return;   // The pointer is not resolved yet!

  LatticeVal &IV = ValueState[&I];

  if (!isConstant(PtrVal) || I.isVolatile())
    return (void)markOverdefined(IV, &I);

  Constant *Ptr = getConstant(PtrVal);

  // load null is undefined.
  if (isa<ConstantPointerNull>(Ptr)) {
    if (NullPointerIsDefined(I.getFunction(), I.getPointerAddressSpace()))
      return (void)markOverdefined(IV, &I);
    else
      return;
  }

  // Transform load (constant global) into the value loaded.
  if (auto *GV = dyn_cast<GlobalVariable>(Ptr)) {
    if (!TrackedGlobals.empty()) {
      // If we are tracking this global, merge in the known value for it.
      DenseMap<GlobalVariable*, LatticeVal>::iterator It =
        TrackedGlobals.find(GV);
      if (It != TrackedGlobals.end()) {
        mergeInValue(IV, &I, It->second);
        return;
      }
    }
  }

  // Transform load from a constant into a constant if possible.
  if (Constant *C = ConstantFoldLoadFromConstPtr(Ptr, I.getType(), DL)) {
    if (isa<UndefValue>(C))
      return;
    return (void)markConstant(IV, &I, C);
  }

  // Otherwise we cannot say for certain what value this load will produce.
  // Bail out.
  markOverdefined(IV, &I);
}

void SCCPSolver::visitCallSite(CallSite CS) {
  Function *F = CS.getCalledFunction();
  Instruction *I = CS.getInstruction();

  if (auto *II = dyn_cast<IntrinsicInst>(I)) {
    if (II->getIntrinsicID() == Intrinsic::ssa_copy) {
      if (isOverdefined(ValueState[I]))
        return;

      auto *PI = getPredicateInfoFor(I);
      if (!PI)
        return;

      Value *CopyOf = I->getOperand(0);
      auto *PBranch = dyn_cast<PredicateBranch>(PI);
      if (!PBranch) {
        mergeInValue(ValueState[I], I, getValueState(CopyOf));
        return;
      }

      Value *Cond = PBranch->Condition;

      // Everything below relies on the condition being a comparison.
      auto *Cmp = dyn_cast<CmpInst>(Cond);
      if (!Cmp) {
        mergeInValue(ValueState[I], I, getValueState(CopyOf));
        return;
      }

      Value *CmpOp0 = Cmp->getOperand(0);
      Value *CmpOp1 = Cmp->getOperand(1);
      if (CopyOf != CmpOp0 && CopyOf != CmpOp1) {
        mergeInValue(ValueState[I], I, getValueState(CopyOf));
        return;
      }

      if (CmpOp0 != CopyOf)
        std::swap(CmpOp0, CmpOp1);

      LatticeVal OriginalVal = getValueState(CopyOf);
      LatticeVal EqVal = getValueState(CmpOp1);
      LatticeVal &IV = ValueState[I];
      if (PBranch->TrueEdge && Cmp->getPredicate() == CmpInst::ICMP_EQ) {
        addAdditionalUser(CmpOp1, I);
        if (isConstant(OriginalVal))
          mergeInValue(IV, I, OriginalVal);
        else
          mergeInValue(IV, I, EqVal);
        return;
      }
      if (!PBranch->TrueEdge && Cmp->getPredicate() == CmpInst::ICMP_NE) {
        addAdditionalUser(CmpOp1, I);
        if (isConstant(OriginalVal))
          mergeInValue(IV, I, OriginalVal);
        else
          mergeInValue(IV, I, EqVal);
        return;
      }

      return (void)mergeInValue(IV, I, getValueState(CopyOf));
    }
  }

  // The common case is that we aren't tracking the callee, either because we
  // are not doing interprocedural analysis or the callee is indirect, or is
  // external.  Handle these cases first.
  if (!F || F->isDeclaration()) {
CallOverdefined:
    // Void return and not tracking callee, just bail.
    if (I->getType()->isVoidTy()) return;

    // Otherwise, if we have a single return value case, and if the function is
    // a declaration, maybe we can constant fold it.
    if (F && F->isDeclaration() && !I->getType()->isStructTy() &&
        canConstantFoldCallTo(cast<CallBase>(CS.getInstruction()), F)) {
      SmallVector<Constant*, 8> Operands;
      for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
           AI != E; ++AI) {
        if (AI->get()->getType()->isStructTy())
          return markOverdefined(I); // Can't handle struct args.
        LatticeVal State = getValueState(*AI);

        if (State.isUnknown())
          return;  // Operands are not resolved yet.
        if (isOverdefined(State))
          return (void)markOverdefined(I);
        assert(isConstant(State) && "Unknown state!");
        Operands.push_back(getConstant(State));
      }

      if (isOverdefined(getValueState(I)))
        return;

      // If we can constant fold this, mark the result of the call as a
      // constant.
      if (Constant *C = ConstantFoldCall(cast<CallBase>(CS.getInstruction()), F,
                                         Operands, &GetTLI(*F))) {
        // call -> undef.
        if (isa<UndefValue>(C))
          return;
        return (void)markConstant(I, C);
      }
    }

    // Otherwise, we don't know anything about this call, mark it overdefined.
    return (void)markOverdefined(I);
  }

  // If this is a local function that doesn't have its address taken, mark its
  // entry block executable and merge in the actual arguments to the call into
  // the formal arguments of the function.
  if (!TrackingIncomingArguments.empty() && TrackingIncomingArguments.count(F)){
    MarkBlockExecutable(&F->front());

    // Propagate information from this call site into the callee.
    CallSite::arg_iterator CAI = CS.arg_begin();
    for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
         AI != E; ++AI, ++CAI) {
      // If this argument is byval, and if the function is not readonly, there
      // will be an implicit copy formed of the input aggregate.
      if (AI->hasByValAttr() && !F->onlyReadsMemory()) {
        markOverdefined(&*AI);
        continue;
      }

      if (auto *STy = dyn_cast<StructType>(AI->getType())) {
        for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
          LatticeVal CallArg = getStructValueState(*CAI, i);
          mergeInValue(getStructValueState(&*AI, i), &*AI, CallArg, false);
        }
      } else
        mergeInValue(&*AI, getValueState(*CAI), false);
    }
  }

  // If this is a single/zero retval case, see if we're tracking the function.
  if (auto *STy = dyn_cast<StructType>(F->getReturnType())) {
    if (!MRVFunctionsTracked.count(F))
      goto CallOverdefined;  // Not tracking this callee.

    // If we are tracking this callee, propagate the result of the function
    // into this call site.
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
      mergeInValue(getStructValueState(I, i), I,
                   TrackedMultipleRetVals[std::make_pair(F, i)]);
  } else {
    MapVector<Function*, LatticeVal>::iterator TFRVI = TrackedRetVals.find(F);
    if (TFRVI == TrackedRetVals.end())
      goto CallOverdefined;  // Not tracking this callee.

    // If so, propagate the return value of the callee into this call result.
    mergeInValue(I, TFRVI->second);
  }
}

void SCCPSolver::Solve() {
  // Process the work lists until they are empty!
  while (!BBWorkList.empty() || !InstWorkList.empty() ||
         !OverdefinedInstWorkList.empty()) {
    // Process the overdefined instruction's work list first, which drives other
    // things to overdefined more quickly.
    while (!OverdefinedInstWorkList.empty()) {
      Value *I = OverdefinedInstWorkList.pop_back_val();

      LLVM_DEBUG(dbgs() << "\nPopped off OI-WL: " << *I << '\n');

      // "I" got into the work list because it either made the transition from
      // bottom to constant, or to overdefined.
      //
      // Anything on this worklist that is overdefined need not be visited
      // since all of its users will have already been marked as overdefined
      // Update all of the users of this instruction's value.
      //
      markUsersAsChanged(I);
    }

    // Process the instruction work list.
    while (!InstWorkList.empty()) {
      Value *I = InstWorkList.pop_back_val();

      LLVM_DEBUG(dbgs() << "\nPopped off I-WL: " << *I << '\n');

      // "I" got into the work list because it made the transition from undef to
      // constant.
      //
      // Anything on this worklist that is overdefined need not be visited
      // since all of its users will have already been marked as overdefined.
      // Update all of the users of this instruction's value.
      //
      if (I->getType()->isStructTy() || !isOverdefined(getValueState(I)))
        markUsersAsChanged(I);
    }

    // Process the basic block work list.
    while (!BBWorkList.empty()) {
      BasicBlock *BB = BBWorkList.back();
      BBWorkList.pop_back();

      LLVM_DEBUG(dbgs() << "\nPopped off BBWL: " << *BB << '\n');

      // Notify all instructions in this basic block that they are newly
      // executable.
      visit(BB);
    }
  }
}

/// ResolvedUndefsIn - While solving the dataflow for a function, we assume
/// that branches on undef values cannot reach any of their successors.
/// However, this is not a safe assumption.  After we solve dataflow, this
/// method should be use to handle this.  If this returns true, the solver
/// should be rerun.
///
/// This method handles this by finding an unresolved branch and marking it one
/// of the edges from the block as being feasible, even though the condition
/// doesn't say it would otherwise be.  This allows SCCP to find the rest of the
/// CFG and only slightly pessimizes the analysis results (by marking one,
/// potentially infeasible, edge feasible).  This cannot usefully modify the
/// constraints on the condition of the branch, as that would impact other users
/// of the value.
///
/// This scan also checks for values that use undefs. It conservatively marks
/// them as overdefined.
bool SCCPSolver::ResolvedUndefsIn(Function &F) {
  for (BasicBlock &BB : F) {
    if (!BBExecutable.count(&BB))
      continue;

    for (Instruction &I : BB) {
      // Look for instructions which produce undef values.
      if (I.getType()->isVoidTy()) continue;

      if (auto *STy = dyn_cast<StructType>(I.getType())) {
        // Only a few things that can be structs matter for undef.

        // Tracked calls must never be marked overdefined in ResolvedUndefsIn.
        if (CallSite CS = CallSite(&I))
          if (Function *F = CS.getCalledFunction())
            if (MRVFunctionsTracked.count(F))
              continue;

        // extractvalue and insertvalue don't need to be marked; they are
        // tracked as precisely as their operands.
        if (isa<ExtractValueInst>(I) || isa<InsertValueInst>(I))
          continue;
        // Send the results of everything else to overdefined.  We could be
        // more precise than this but it isn't worth bothering.
        for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
          LatticeVal &LV = getStructValueState(&I, i);
          if (LV.isUnknown())
            markOverdefined(LV, &I);
        }
        continue;
      }

      LatticeVal &LV = getValueState(&I);
      if (!LV.isUnknown())
        continue;

      // There are two reasons a call can have an undef result
      // 1. It could be tracked.
      // 2. It could be constant-foldable.
      // Because of the way we solve return values, tracked calls must
      // never be marked overdefined in ResolvedUndefsIn.
      if (CallSite CS = CallSite(&I))
        if (Function *F = CS.getCalledFunction())
          if (TrackedRetVals.count(F))
            continue;

      if (isa<LoadInst>(I)) {
        // A load here means one of two things: a load of undef from a global,
        // a load from an unknown pointer.  Either way, having it return undef
        // is okay.
        continue;
      }

      markOverdefined(&I);
      return true;
    }

    // Check to see if we have a branch or switch on an undefined value.  If so
    // we force the branch to go one way or the other to make the successor
    // values live.  It doesn't really matter which way we force it.
    Instruction *TI = BB.getTerminator();
    if (auto *BI = dyn_cast<BranchInst>(TI)) {
      if (!BI->isConditional()) continue;
      if (!getValueState(BI->getCondition()).isUnknown())
        continue;

      // If the input to SCCP is actually branch on undef, fix the undef to
      // false.
      if (isa<UndefValue>(BI->getCondition())) {
        BI->setCondition(ConstantInt::getFalse(BI->getContext()));
        markEdgeExecutable(&BB, TI->getSuccessor(1));
        return true;
      }

      // Otherwise, it is a branch on a symbolic value which is currently
      // considered to be undef.  Make sure some edge is executable, so a
      // branch on "undef" always flows somewhere.
      // FIXME: Distinguish between dead code and an LLVM "undef" value.
      BasicBlock *DefaultSuccessor = TI->getSuccessor(1);
      if (markEdgeExecutable(&BB, DefaultSuccessor))
        return true;

      continue;
    }

   if (auto *IBR = dyn_cast<IndirectBrInst>(TI)) {
      // Indirect branch with no successor ?. Its ok to assume it branches
      // to no target.
      if (IBR->getNumSuccessors() < 1)
        continue;

      if (!getValueState(IBR->getAddress()).isUnknown())
        continue;

      // If the input to SCCP is actually branch on undef, fix the undef to
      // the first successor of the indirect branch.
      if (isa<UndefValue>(IBR->getAddress())) {
        IBR->setAddress(BlockAddress::get(IBR->getSuccessor(0)));
        markEdgeExecutable(&BB, IBR->getSuccessor(0));
        return true;
      }

      // Otherwise, it is a branch on a symbolic value which is currently
      // considered to be undef.  Make sure some edge is executable, so a
      // branch on "undef" always flows somewhere.
      // FIXME: IndirectBr on "undef" doesn't actually need to go anywhere:
      // we can assume the branch has undefined behavior instead.
      BasicBlock *DefaultSuccessor = IBR->getSuccessor(0);
      if (markEdgeExecutable(&BB, DefaultSuccessor))
        return true;

      continue;
    }

    if (auto *SI = dyn_cast<SwitchInst>(TI)) {
      if (!SI->getNumCases() || !getValueState(SI->getCondition()).isUnknown())
        continue;

      // If the input to SCCP is actually switch on undef, fix the undef to
      // the first constant.
      if (isa<UndefValue>(SI->getCondition())) {
        SI->setCondition(SI->case_begin()->getCaseValue());
        markEdgeExecutable(&BB, SI->case_begin()->getCaseSuccessor());
        return true;
      }

      // Otherwise, it is a branch on a symbolic value which is currently
      // considered to be undef.  Make sure some edge is executable, so a
      // branch on "undef" always flows somewhere.
      // FIXME: Distinguish between dead code and an LLVM "undef" value.
      BasicBlock *DefaultSuccessor = SI->case_begin()->getCaseSuccessor();
      if (markEdgeExecutable(&BB, DefaultSuccessor))
        return true;

      continue;
    }
  }

  return false;
}

static bool tryToReplaceWithConstant(SCCPSolver &Solver, Value *V) {
  Constant *Const = nullptr;
  if (V->getType()->isStructTy()) {
    std::vector<LatticeVal> IVs = Solver.getStructLatticeValueFor(V);
    if (any_of(IVs, [](const LatticeVal &LV) { return isOverdefined(LV); }))
      return false;
    std::vector<Constant *> ConstVals;
    auto *ST = cast<StructType>(V->getType());
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
      LatticeVal V = IVs[i];
      ConstVals.push_back(isConstant(V)
                              ? Solver.getConstant(V)
                              : UndefValue::get(ST->getElementType(i)));
    }
    Const = ConstantStruct::get(ST, ConstVals);
  } else {
    const LatticeVal &IV = Solver.getLatticeValueFor(V);
    if (isOverdefined(IV))
      return false;

    Const =
        isConstant(IV) ? Solver.getConstant(IV) : UndefValue::get(V->getType());
  }
  assert(Const && "Constant is nullptr here!");

  // Replacing `musttail` instructions with constant breaks `musttail` invariant
  // unless the call itself can be removed
  CallInst *CI = dyn_cast<CallInst>(V);
  if (CI && CI->isMustTailCall() && !CI->isSafeToRemove()) {
    CallSite CS(CI);
    Function *F = CS.getCalledFunction();

    // Don't zap returns of the callee
    if (F)
      Solver.AddMustTailCallee(F);

    LLVM_DEBUG(dbgs() << "  Can\'t treat the result of musttail call : " << *CI
                      << " as a constant\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "  Constant: " << *Const << " = " << *V << '\n');

  // Replaces all of the uses of a variable with uses of the constant.
  V->replaceAllUsesWith(Const);
  return true;
}

// runSCCP() - Run the Sparse Conditional Constant Propagation algorithm,
// and return true if the function was modified.
static bool runSCCP(Function &F, const DataLayout &DL,
                    const TargetLibraryInfo *TLI) {
  LLVM_DEBUG(dbgs() << "SCCP on function '" << F.getName() << "'\n");
  SCCPSolver Solver(
      DL, [TLI](Function &F) -> const TargetLibraryInfo & { return *TLI; },
      F.getContext());

  // Mark the first block of the function as being executable.
  Solver.MarkBlockExecutable(&F.front());

  // Mark all arguments to the function as being overdefined.
  for (Argument &AI : F.args())
    Solver.markOverdefined(&AI);

  // Solve for constants.
  bool ResolvedUndefs = true;
  while (ResolvedUndefs) {
    Solver.Solve();
    LLVM_DEBUG(dbgs() << "RESOLVING UNDEFs\n");
    ResolvedUndefs = Solver.ResolvedUndefsIn(F);
  }

  bool MadeChanges = false;

  // If we decided that there are basic blocks that are dead in this function,
  // delete their contents now.  Note that we cannot actually delete the blocks,
  // as we cannot modify the CFG of the function.

  for (BasicBlock &BB : F) {
    if (!Solver.isBlockExecutable(&BB)) {
      LLVM_DEBUG(dbgs() << "  BasicBlock Dead:" << BB);

      ++NumDeadBlocks;
      NumInstRemoved += removeAllNonTerminatorAndEHPadInstructions(&BB);

      MadeChanges = true;
      continue;
    }

    // Iterate over all of the instructions in a function, replacing them with
    // constants if we have found them to be of constant values.
    for (BasicBlock::iterator BI = BB.begin(), E = BB.end(); BI != E;) {
      Instruction *Inst = &*BI++;
      if (Inst->getType()->isVoidTy() || Inst->isTerminator())
        continue;

      if (tryToReplaceWithConstant(Solver, Inst)) {
        if (isInstructionTriviallyDead(Inst))
          Inst->eraseFromParent();
        // Hey, we just changed something!
        MadeChanges = true;
        ++NumInstRemoved;
      }
    }
  }

  return MadeChanges;
}

PreservedAnalyses SCCPPass::run(Function &F, FunctionAnalysisManager &AM) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  if (!runSCCP(F, DL, &TLI))
    return PreservedAnalyses::all();

  auto PA = PreservedAnalyses();
  PA.preserve<GlobalsAA>();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

namespace {

//===--------------------------------------------------------------------===//
//
/// SCCP Class - This class uses the SCCPSolver to implement a per-function
/// Sparse Conditional Constant Propagator.
///
class SCCPLegacyPass : public FunctionPass {
public:
  // Pass identification, replacement for typeid
  static char ID;

  SCCPLegacyPass() : FunctionPass(ID) {
    initializeSCCPLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.setPreservesCFG();
  }

  // runOnFunction - Run the Sparse Conditional Constant Propagation
  // algorithm, and return true if the function was modified.
  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    const DataLayout &DL = F.getParent()->getDataLayout();
    const TargetLibraryInfo *TLI =
        &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    return runSCCP(F, DL, TLI);
  }
};

} // end anonymous namespace

char SCCPLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(SCCPLegacyPass, "sccp",
                      "Sparse Conditional Constant Propagation", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(SCCPLegacyPass, "sccp",
                    "Sparse Conditional Constant Propagation", false, false)

// createSCCPPass - This is the public interface to this file.
FunctionPass *llvm::createSCCPPass() { return new SCCPLegacyPass(); }

static void findReturnsToZap(Function &F,
                             SmallVector<ReturnInst *, 8> &ReturnsToZap,
                             SCCPSolver &Solver) {
  // We can only do this if we know that nothing else can call the function.
  if (!Solver.isArgumentTrackedFunction(&F))
    return;

  // There is a non-removable musttail call site of this function. Zapping
  // returns is not allowed.
  if (Solver.isMustTailCallee(&F)) {
    LLVM_DEBUG(dbgs() << "Can't zap returns of the function : " << F.getName()
                      << " due to present musttail call of it\n");
    return;
  }

  assert(
      all_of(F.users(),
             [&Solver](User *U) {
               if (isa<Instruction>(U) &&
                   !Solver.isBlockExecutable(cast<Instruction>(U)->getParent()))
                 return true;
               // Non-callsite uses are not impacted by zapping. Also, constant
               // uses (like blockaddresses) could stuck around, without being
               // used in the underlying IR, meaning we do not have lattice
               // values for them.
               if (!CallSite(U))
                 return true;
               if (U->getType()->isStructTy()) {
                 return all_of(
                     Solver.getStructLatticeValueFor(U),
                     [](const LatticeVal &LV) { return !isOverdefined(LV); });
               }
               return !isOverdefined(Solver.getLatticeValueFor(U));
             }) &&
      "We can only zap functions where all live users have a concrete value");

  for (BasicBlock &BB : F) {
    if (CallInst *CI = BB.getTerminatingMustTailCall()) {
      LLVM_DEBUG(dbgs() << "Can't zap return of the block due to present "
                        << "musttail call : " << *CI << "\n");
      (void)CI;
      return;
    }

    if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator()))
      if (!isa<UndefValue>(RI->getOperand(0)))
        ReturnsToZap.push_back(RI);
  }
}

// Update the condition for terminators that are branching on indeterminate
// values, forcing them to use a specific edge.
static void forceIndeterminateEdge(Instruction* I, SCCPSolver &Solver) {
  BasicBlock *Dest = nullptr;
  Constant *C = nullptr;
  if (SwitchInst *SI = dyn_cast<SwitchInst>(I)) {
    if (!isa<ConstantInt>(SI->getCondition())) {
      // Indeterminate switch; use first case value.
      Dest = SI->case_begin()->getCaseSuccessor();
      C = SI->case_begin()->getCaseValue();
    }
  } else if (BranchInst *BI = dyn_cast<BranchInst>(I)) {
    if (!isa<ConstantInt>(BI->getCondition())) {
      // Indeterminate branch; use false.
      Dest = BI->getSuccessor(1);
      C = ConstantInt::getFalse(BI->getContext());
    }
  } else if (IndirectBrInst *IBR = dyn_cast<IndirectBrInst>(I)) {
    if (!isa<BlockAddress>(IBR->getAddress()->stripPointerCasts())) {
      // Indeterminate indirectbr; use successor 0.
      Dest = IBR->getSuccessor(0);
      C = BlockAddress::get(IBR->getSuccessor(0));
    }
  } else {
    llvm_unreachable("Unexpected terminator instruction");
  }
  if (C) {
    assert(Solver.isEdgeFeasible(I->getParent(), Dest) &&
           "Didn't find feasible edge?");
    (void)Dest;

    I->setOperand(0, C);
  }
}

bool llvm::runIPSCCP(
    Module &M, const DataLayout &DL,
    std::function<const TargetLibraryInfo &(Function &)> GetTLI,
    function_ref<AnalysisResultsForFn(Function &)> getAnalysis) {
  SCCPSolver Solver(DL, GetTLI, M.getContext());

  // Loop over all functions, marking arguments to those with their addresses
  // taken or that are external as overdefined.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    Solver.addAnalysis(F, getAnalysis(F));

    // Determine if we can track the function's return values. If so, add the
    // function to the solver's set of return-tracked functions.
    if (canTrackReturnsInterprocedurally(&F))
      Solver.AddTrackedFunction(&F);

    // Determine if we can track the function's arguments. If so, add the
    // function to the solver's set of argument-tracked functions.
    if (canTrackArgumentsInterprocedurally(&F)) {
      Solver.AddArgumentTrackedFunction(&F);
      continue;
    }

    // Assume the function is called.
    Solver.MarkBlockExecutable(&F.front());

    // Assume nothing about the incoming arguments.
    for (Argument &AI : F.args())
      Solver.markOverdefined(&AI);
  }

  // Determine if we can track any of the module's global variables. If so, add
  // the global variables we can track to the solver's set of tracked global
  // variables.
  for (GlobalVariable &G : M.globals()) {
    G.removeDeadConstantUsers();
    if (canTrackGlobalVariableInterprocedurally(&G))
      Solver.TrackValueOfGlobalVariable(&G);
  }

  // Solve for constants.
  bool ResolvedUndefs = true;
  Solver.Solve();
  while (ResolvedUndefs) {
    LLVM_DEBUG(dbgs() << "RESOLVING UNDEFS\n");
    ResolvedUndefs = false;
    for (Function &F : M)
      if (Solver.ResolvedUndefsIn(F)) {
        // We run Solve() after we resolved an undef in a function, because
        // we might deduce a fact that eliminates an undef in another function.
        Solver.Solve();
        ResolvedUndefs = true;
      }
  }

  bool MadeChanges = false;

  // Iterate over all of the instructions in the module, replacing them with
  // constants if we have found them to be of constant values.

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    SmallVector<BasicBlock *, 512> BlocksToErase;

    if (Solver.isBlockExecutable(&F.front()))
      for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end(); AI != E;
           ++AI) {
        if (!AI->use_empty() && tryToReplaceWithConstant(Solver, &*AI)) {
          ++IPNumArgsElimed;
          continue;
        }
      }

    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
      if (!Solver.isBlockExecutable(&*BB)) {
        LLVM_DEBUG(dbgs() << "  BasicBlock Dead:" << *BB);
        ++NumDeadBlocks;

        MadeChanges = true;

        if (&*BB != &F.front())
          BlocksToErase.push_back(&*BB);
        continue;
      }

      for (BasicBlock::iterator BI = BB->begin(), E = BB->end(); BI != E; ) {
        Instruction *Inst = &*BI++;
        if (Inst->getType()->isVoidTy())
          continue;
        if (tryToReplaceWithConstant(Solver, Inst)) {
          if (Inst->isSafeToRemove())
            Inst->eraseFromParent();
          // Hey, we just changed something!
          MadeChanges = true;
          ++IPNumInstRemoved;
        }
      }
    }

    DomTreeUpdater DTU = Solver.getDTU(F);
    // Change dead blocks to unreachable. We do it after replacing constants
    // in all executable blocks, because changeToUnreachable may remove PHI
    // nodes in executable blocks we found values for. The function's entry
    // block is not part of BlocksToErase, so we have to handle it separately.
    for (BasicBlock *BB : BlocksToErase) {
      NumInstRemoved +=
          changeToUnreachable(BB->getFirstNonPHI(), /*UseLLVMTrap=*/false,
                              /*PreserveLCSSA=*/false, &DTU);
    }
    if (!Solver.isBlockExecutable(&F.front()))
      NumInstRemoved += changeToUnreachable(F.front().getFirstNonPHI(),
                                            /*UseLLVMTrap=*/false,
                                            /*PreserveLCSSA=*/false, &DTU);

    // Now that all instructions in the function are constant folded,
    // use ConstantFoldTerminator to get rid of in-edges, record DT updates and
    // delete dead BBs.
    for (BasicBlock *DeadBB : BlocksToErase) {
      // If there are any PHI nodes in this successor, drop entries for BB now.
      for (Value::user_iterator UI = DeadBB->user_begin(),
                                UE = DeadBB->user_end();
           UI != UE;) {
        // Grab the user and then increment the iterator early, as the user
        // will be deleted. Step past all adjacent uses from the same user.
        auto *I = dyn_cast<Instruction>(*UI);
        do { ++UI; } while (UI != UE && *UI == I);

        // Ignore blockaddress users; BasicBlock's dtor will handle them.
        if (!I) continue;

        // If we have forced an edge for an indeterminate value, then force the
        // terminator to fold to that edge.
        forceIndeterminateEdge(I, Solver);
        BasicBlock *InstBB = I->getParent();
        bool Folded = ConstantFoldTerminator(InstBB,
                                             /*DeleteDeadConditions=*/false,
                                             /*TLI=*/nullptr, &DTU);
        assert(Folded &&
              "Expect TermInst on constantint or blockaddress to be folded");
        (void) Folded;
        // If we folded the terminator to an unconditional branch to another
        // dead block, replace it with Unreachable, to avoid trying to fold that
        // branch again.
        BranchInst *BI = cast<BranchInst>(InstBB->getTerminator());
        if (BI && BI->isUnconditional() &&
            !Solver.isBlockExecutable(BI->getSuccessor(0))) {
          InstBB->getTerminator()->eraseFromParent();
          new UnreachableInst(InstBB->getContext(), InstBB);
        }
      }
      // Mark dead BB for deletion.
      DTU.deleteBB(DeadBB);
    }

    for (BasicBlock &BB : F) {
      for (BasicBlock::iterator BI = BB.begin(), E = BB.end(); BI != E;) {
        Instruction *Inst = &*BI++;
        if (Solver.getPredicateInfoFor(Inst)) {
          if (auto *II = dyn_cast<IntrinsicInst>(Inst)) {
            if (II->getIntrinsicID() == Intrinsic::ssa_copy) {
              Value *Op = II->getOperand(0);
              Inst->replaceAllUsesWith(Op);
              Inst->eraseFromParent();
            }
          }
        }
      }
    }
  }

  // If we inferred constant or undef return values for a function, we replaced
  // all call uses with the inferred value.  This means we don't need to bother
  // actually returning anything from the function.  Replace all return
  // instructions with return undef.
  //
  // Do this in two stages: first identify the functions we should process, then
  // actually zap their returns.  This is important because we can only do this
  // if the address of the function isn't taken.  In cases where a return is the
  // last use of a function, the order of processing functions would affect
  // whether other functions are optimizable.
  SmallVector<ReturnInst*, 8> ReturnsToZap;

  const MapVector<Function*, LatticeVal> &RV = Solver.getTrackedRetVals();
  for (const auto &I : RV) {
    Function *F = I.first;
    if (isOverdefined(I.second) || F->getReturnType()->isVoidTy())
      continue;
    findReturnsToZap(*F, ReturnsToZap, Solver);
  }

  for (auto F : Solver.getMRVFunctionsTracked()) {
    assert(F->getReturnType()->isStructTy() &&
           "The return type should be a struct");
    StructType *STy = cast<StructType>(F->getReturnType());
    if (Solver.isStructLatticeConstant(F, STy))
      findReturnsToZap(*F, ReturnsToZap, Solver);
  }

  // Zap all returns which we've identified as zap to change.
  for (unsigned i = 0, e = ReturnsToZap.size(); i != e; ++i) {
    Function *F = ReturnsToZap[i]->getParent()->getParent();
    ReturnsToZap[i]->setOperand(0, UndefValue::get(F->getReturnType()));
  }

  // If we inferred constant or undef values for globals variables, we can
  // delete the global and any stores that remain to it.
  const DenseMap<GlobalVariable*, LatticeVal> &TG = Solver.getTrackedGlobals();
  for (DenseMap<GlobalVariable*, LatticeVal>::const_iterator I = TG.begin(),
         E = TG.end(); I != E; ++I) {
    GlobalVariable *GV = I->first;
    if (isOverdefined(I->second))
      continue;
    LLVM_DEBUG(dbgs() << "Found that GV '" << GV->getName()
                      << "' is constant!\n");
    while (!GV->use_empty()) {
      StoreInst *SI = cast<StoreInst>(GV->user_back());
      SI->eraseFromParent();
    }
    M.getGlobalList().erase(GV);
    ++IPNumGlobalConst;
  }

  return MadeChanges;
}
