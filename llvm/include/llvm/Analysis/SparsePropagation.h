//===- SparsePropagation.h - Sparse Conditional Property Propagation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an abstract sparse conditional propagation algorithm,
// modeled after SCCP, but with a customizable lattice function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SPARSEPROPAGATION_H
#define LLVM_ANALYSIS_SPARSEPROPAGATION_H

#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include <set>

#define DEBUG_TYPE "sparseprop"

namespace llvm {

template <class LatticeVal> class SparseSolver;

/// AbstractLatticeFunction - This class is implemented by the dataflow instance
/// to specify what the lattice values are and how they handle merges etc.  This
/// gives the client the power to compute lattice values from instructions,
/// constants, etc.  The current requirement is that lattice values must be
/// copyable.  At the moment, nothing tries to avoid copying.


template <class LatticeVal> class AbstractLatticeFunction {
private:
  LatticeVal UndefVal, OverdefinedVal, UntrackedVal;

public:
  AbstractLatticeFunction(LatticeVal undefVal, LatticeVal overdefinedVal,
                          LatticeVal untrackedVal) {
    UndefVal = undefVal;
    OverdefinedVal = overdefinedVal;
    UntrackedVal = untrackedVal;
  }

  virtual ~AbstractLatticeFunction() = default;

  LatticeVal getUndefVal()       const { return UndefVal; }
  LatticeVal getOverdefinedVal() const { return OverdefinedVal; }
  LatticeVal getUntrackedVal()   const { return UntrackedVal; }

  /// IsUntrackedValue - If the specified Value is something that is obviously
  /// uninteresting to the analysis (and would always return UntrackedVal),
  /// this function can return true to avoid pointless work.
  virtual bool IsUntrackedValue(Value *V) { return false; }

  /// ComputeConstant - Given a constant value, compute and return a lattice
  /// value corresponding to the specified constant.
  virtual LatticeVal ComputeConstant(Constant *C) {
    return getOverdefinedVal(); // always safe
  }

  /// IsSpecialCasedPHI - Given a PHI node, determine whether this PHI node is
  /// one that the we want to handle through ComputeInstructionState.
  virtual bool IsSpecialCasedPHI(PHINode *PN) { return false; }

  /// GetConstant - If the specified lattice value is representable as an LLVM
  /// constant value, return it.  Otherwise return null.  The returned value
  /// must be in the same LLVM type as Val.
  virtual Constant *GetConstant(LatticeVal LV, Value *Val,
                                SparseSolver<LatticeVal> &SS) {
    return nullptr;
  }

  /// ComputeArgument - Given a formal argument value, compute and return a
  /// lattice value corresponding to the specified argument.
  virtual LatticeVal ComputeArgument(Argument *I) {
    return getOverdefinedVal(); // always safe
  }

  /// MergeValues - Compute and return the merge of the two specified lattice
  /// values.  Merging should only move one direction down the lattice to
  /// guarantee convergence (toward overdefined).
  virtual LatticeVal MergeValues(LatticeVal X, LatticeVal Y) {
    return getOverdefinedVal(); // always safe, never useful.
  }

  /// ComputeInstructionState - Given an instruction and a vector of its operand
  /// values, compute the result value of the instruction.
  virtual LatticeVal ComputeInstructionState(Instruction &I,
                                             SparseSolver<LatticeVal> &SS) {
    return getOverdefinedVal(); // always safe, never useful.
  }

  /// PrintValue - Render the specified lattice value to the specified stream.
  virtual void PrintValue(LatticeVal V, raw_ostream &OS);
};

/// SparseSolver - This class is a general purpose solver for Sparse Conditional
/// Propagation with a programmable lattice function.
template <class LatticeVal> class SparseSolver {

  /// LatticeFunc - This is the object that knows the lattice and how to
  /// compute transfer functions.
  AbstractLatticeFunction<LatticeVal> *LatticeFunc;

  /// ValueState - Holds the lattice state associated with LLVM values.
  DenseMap<Value *, LatticeVal> ValueState;

  /// BBExecutable - Holds the basic blocks that are executable.
  SmallPtrSet<BasicBlock *, 16> BBExecutable;

  /// ValueWorkList - Holds values that should be processed.
  SmallVector<Value *, 64> ValueWorkList;

  /// BBWorkList - Holds basic blocks that should be processed.
  SmallVector<BasicBlock *, 64> BBWorkList;

  using Edge = std::pair<BasicBlock *, BasicBlock *>;

  /// KnownFeasibleEdges - Entries in this set are edges which have already had
  /// PHI nodes retriggered.
  std::set<Edge> KnownFeasibleEdges;

public:
  explicit SparseSolver(AbstractLatticeFunction<LatticeVal> *Lattice)
      : LatticeFunc(Lattice) {}
  SparseSolver(const SparseSolver &) = delete;
  SparseSolver &operator=(const SparseSolver &) = delete;

  /// Solve - Solve for constants and executable blocks.
  void Solve(Function &F);

  void Print(Function &F, raw_ostream &OS) const;

  /// getExistingValueState - Return the LatticeVal object corresponding to the
  /// given value from the ValueState map. If the value is not in the map,
  /// UntrackedVal is returned, unlike the getValueState method.
  LatticeVal getExistingValueState(Value *V) const {
    auto I = ValueState.find(V);
    return I != ValueState.end() ? I->second : LatticeFunc->getUntrackedVal();
  }

  /// getValueState - Return the LatticeVal object corresponding to the given
  /// value from the ValueState map. If the value is not in the map, its state
  /// is initialized.
  LatticeVal getValueState(Value *V);

  /// isEdgeFeasible - Return true if the control flow edge from the 'From'
  /// basic block to the 'To' basic block is currently feasible.  If
  /// AggressiveUndef is true, then this treats values with unknown lattice
  /// values as undefined.  This is generally only useful when solving the
  /// lattice, not when querying it.
  bool isEdgeFeasible(BasicBlock *From, BasicBlock *To,
                      bool AggressiveUndef = false);

  /// isBlockExecutable - Return true if there are any known feasible
  /// edges into the basic block.  This is generally only useful when
  /// querying the lattice.
  bool isBlockExecutable(BasicBlock *BB) const {
    return BBExecutable.count(BB);
  }

private:
  /// UpdateState - When the state for some instruction is potentially updated,
  /// this function notices and adds I to the worklist if needed.
  void UpdateState(Instruction &Inst, LatticeVal V);

  /// MarkBlockExecutable - This method can be used by clients to mark all of
  /// the blocks that are known to be intrinsically live in the processed unit.
  void MarkBlockExecutable(BasicBlock *BB);

  /// markEdgeExecutable - Mark a basic block as executable, adding it to the BB
  /// work list if it is not already executable.
  void markEdgeExecutable(BasicBlock *Source, BasicBlock *Dest);

  /// getFeasibleSuccessors - Return a vector of booleans to indicate which
  /// successors are reachable from a given terminator instruction.
  void getFeasibleSuccessors(TerminatorInst &TI, SmallVectorImpl<bool> &Succs,
                             bool AggressiveUndef);

  void visitInst(Instruction &I);
  void visitPHINode(PHINode &I);
  void visitTerminatorInst(TerminatorInst &TI);
};

//===----------------------------------------------------------------------===//
//                  AbstractLatticeFunction Implementation
//===----------------------------------------------------------------------===//

template <class LatticeVal>
void AbstractLatticeFunction<LatticeVal>::PrintValue(LatticeVal V,
                                                     raw_ostream &OS) {
  if (V == UndefVal)
    OS << "undefined";
  else if (V == OverdefinedVal)
    OS << "overdefined";
  else if (V == UntrackedVal)
    OS << "untracked";
  else
    OS << "unknown lattice value";
}

//===----------------------------------------------------------------------===//
//                          SparseSolver Implementation
//===----------------------------------------------------------------------===//

template <class LatticeVal>
LatticeVal SparseSolver<LatticeVal>::getValueState(Value *V) {
  auto I = ValueState.find(V);
  if (I != ValueState.end())
    return I->second; // Common case, in the map

  LatticeVal LV;
  if (LatticeFunc->IsUntrackedValue(V))
    return LatticeFunc->getUntrackedVal();
  else if (Constant *C = dyn_cast<Constant>(V))
    LV = LatticeFunc->ComputeConstant(C);
  else if (Argument *A = dyn_cast<Argument>(V))
    LV = LatticeFunc->ComputeArgument(A);
  else if (!isa<Instruction>(V))
    // All other non-instructions are overdefined.
    LV = LatticeFunc->getOverdefinedVal();
  else
    // All instructions are underdefined by default.
    LV = LatticeFunc->getUndefVal();

  // If this value is untracked, don't add it to the map.
  if (LV == LatticeFunc->getUntrackedVal())
    return LV;
  return ValueState[V] = LV;
}

template <class LatticeVal>
void SparseSolver<LatticeVal>::UpdateState(Instruction &Inst, LatticeVal V) {
  auto I = ValueState.find(&Inst);
  if (I != ValueState.end() && I->second == V)
    return; // No change.

  // An update.  Visit uses of I.
  ValueState[&Inst] = V;
  ValueWorkList.push_back(&Inst);
}

template <class LatticeVal>
void SparseSolver<LatticeVal>::MarkBlockExecutable(BasicBlock *BB) {
  DEBUG(dbgs() << "Marking Block Executable: " << BB->getName() << "\n");
  BBExecutable.insert(BB);  // Basic block is executable!
  BBWorkList.push_back(BB); // Add the block to the work list!
}

template <class LatticeVal>
void SparseSolver<LatticeVal>::markEdgeExecutable(BasicBlock *Source,
                                                  BasicBlock *Dest) {
  if (!KnownFeasibleEdges.insert(Edge(Source, Dest)).second)
    return; // This edge is already known to be executable!

  DEBUG(dbgs() << "Marking Edge Executable: " << Source->getName() << " -> "
               << Dest->getName() << "\n");

  if (BBExecutable.count(Dest)) {
    // The destination is already executable, but we just made an edge
    // feasible that wasn't before.  Revisit the PHI nodes in the block
    // because they have potentially new operands.
    for (BasicBlock::iterator I = Dest->begin(); isa<PHINode>(I); ++I)
      visitPHINode(*cast<PHINode>(I));
  } else {
    MarkBlockExecutable(Dest);
  }
}

template <class LatticeVal>
void SparseSolver<LatticeVal>::getFeasibleSuccessors(
    TerminatorInst &TI, SmallVectorImpl<bool> &Succs, bool AggressiveUndef) {
  Succs.resize(TI.getNumSuccessors());
  if (TI.getNumSuccessors() == 0)
    return;

  if (BranchInst *BI = dyn_cast<BranchInst>(&TI)) {
    if (BI->isUnconditional()) {
      Succs[0] = true;
      return;
    }

    LatticeVal BCValue;
    if (AggressiveUndef)
      BCValue = getValueState(BI->getCondition());
    else
      BCValue = getExistingValueState(BI->getCondition());

    if (BCValue == LatticeFunc->getOverdefinedVal() ||
        BCValue == LatticeFunc->getUntrackedVal()) {
      // Overdefined condition variables can branch either way.
      Succs[0] = Succs[1] = true;
      return;
    }

    // If undefined, neither is feasible yet.
    if (BCValue == LatticeFunc->getUndefVal())
      return;

    Constant *C = LatticeFunc->GetConstant(BCValue, BI->getCondition(), *this);
    if (!C || !isa<ConstantInt>(C)) {
      // Non-constant values can go either way.
      Succs[0] = Succs[1] = true;
      return;
    }

    // Constant condition variables mean the branch can only go a single way
    Succs[C->isNullValue()] = true;
    return;
  }

  if (isa<InvokeInst>(TI)) {
    // Invoke instructions successors are always executable.
    // TODO: Could ask the lattice function if the value can throw.
    Succs[0] = Succs[1] = true;
    return;
  }

  if (isa<IndirectBrInst>(TI)) {
    Succs.assign(Succs.size(), true);
    return;
  }

  SwitchInst &SI = cast<SwitchInst>(TI);
  LatticeVal SCValue;
  if (AggressiveUndef)
    SCValue = getValueState(SI.getCondition());
  else
    SCValue = getExistingValueState(SI.getCondition());

  if (SCValue == LatticeFunc->getOverdefinedVal() ||
      SCValue == LatticeFunc->getUntrackedVal()) {
    // All destinations are executable!
    Succs.assign(TI.getNumSuccessors(), true);
    return;
  }

  // If undefined, neither is feasible yet.
  if (SCValue == LatticeFunc->getUndefVal())
    return;

  Constant *C = LatticeFunc->GetConstant(SCValue, SI.getCondition(), *this);
  if (!C || !isa<ConstantInt>(C)) {
    // All destinations are executable!
    Succs.assign(TI.getNumSuccessors(), true);
    return;
  }
  SwitchInst::CaseHandle Case = *SI.findCaseValue(cast<ConstantInt>(C));
  Succs[Case.getSuccessorIndex()] = true;
}

template <class LatticeVal>
bool SparseSolver<LatticeVal>::isEdgeFeasible(BasicBlock *From, BasicBlock *To,
                                              bool AggressiveUndef) {
  SmallVector<bool, 16> SuccFeasible;
  TerminatorInst *TI = From->getTerminator();
  getFeasibleSuccessors(*TI, SuccFeasible, AggressiveUndef);

  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
    if (TI->getSuccessor(i) == To && SuccFeasible[i])
      return true;

  return false;
}

template <class LatticeVal>
void SparseSolver<LatticeVal>::visitTerminatorInst(TerminatorInst &TI) {
  SmallVector<bool, 16> SuccFeasible;
  getFeasibleSuccessors(TI, SuccFeasible, true);

  BasicBlock *BB = TI.getParent();

  // Mark all feasible successors executable...
  for (unsigned i = 0, e = SuccFeasible.size(); i != e; ++i)
    if (SuccFeasible[i])
      markEdgeExecutable(BB, TI.getSuccessor(i));
}

template <class LatticeVal>
void SparseSolver<LatticeVal>::visitPHINode(PHINode &PN) {
  // The lattice function may store more information on a PHINode than could be
  // computed from its incoming values.  For example, SSI form stores its sigma
  // functions as PHINodes with a single incoming value.
  if (LatticeFunc->IsSpecialCasedPHI(&PN)) {
    LatticeVal IV = LatticeFunc->ComputeInstructionState(PN, *this);
    if (IV != LatticeFunc->getUntrackedVal())
      UpdateState(PN, IV);
    return;
  }

  LatticeVal PNIV = getValueState(&PN);
  LatticeVal Overdefined = LatticeFunc->getOverdefinedVal();

  // If this value is already overdefined (common) just return.
  if (PNIV == Overdefined || PNIV == LatticeFunc->getUntrackedVal())
    return; // Quick exit

  // Super-extra-high-degree PHI nodes are unlikely to ever be interesting,
  // and slow us down a lot.  Just mark them overdefined.
  if (PN.getNumIncomingValues() > 64) {
    UpdateState(PN, Overdefined);
    return;
  }

  // Look at all of the executable operands of the PHI node.  If any of them
  // are overdefined, the PHI becomes overdefined as well.  Otherwise, ask the
  // transfer function to give us the merge of the incoming values.
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
    // If the edge is not yet known to be feasible, it doesn't impact the PHI.
    if (!isEdgeFeasible(PN.getIncomingBlock(i), PN.getParent(), true))
      continue;

    // Merge in this value.
    LatticeVal OpVal = getValueState(PN.getIncomingValue(i));
    if (OpVal != PNIV)
      PNIV = LatticeFunc->MergeValues(PNIV, OpVal);

    if (PNIV == Overdefined)
      break; // Rest of input values don't matter.
  }

  // Update the PHI with the compute value, which is the merge of the inputs.
  UpdateState(PN, PNIV);
}

template <class LatticeVal>
void SparseSolver<LatticeVal>::visitInst(Instruction &I) {
  // PHIs are handled by the propagation logic, they are never passed into the
  // transfer functions.
  if (PHINode *PN = dyn_cast<PHINode>(&I))
    return visitPHINode(*PN);

  // Otherwise, ask the transfer function what the result is.  If this is
  // something that we care about, remember it.
  LatticeVal IV = LatticeFunc->ComputeInstructionState(I, *this);
  if (IV != LatticeFunc->getUntrackedVal())
    UpdateState(I, IV);

  if (TerminatorInst *TI = dyn_cast<TerminatorInst>(&I))
    visitTerminatorInst(*TI);
}

template <class LatticeVal> void SparseSolver<LatticeVal>::Solve(Function &F) {
  MarkBlockExecutable(&F.getEntryBlock());

  // Process the work lists until they are empty!
  while (!BBWorkList.empty() || !ValueWorkList.empty()) {
    // Process the value work list.
    while (!ValueWorkList.empty()) {
      Value *V = ValueWorkList.back();
      ValueWorkList.pop_back();

      DEBUG(dbgs() << "\nPopped off V-WL: " << *V << "\n");

      // "V" got into the work list because it made a transition. See if any
      // users are both live and in need of updating.
      for (User *U : V->users())
        if (Instruction *Inst = dyn_cast<Instruction>(U))
          if (BBExecutable.count(Inst->getParent())) // Inst is executable?
            visitInst(*Inst);
    }

    // Process the basic block work list.
    while (!BBWorkList.empty()) {
      BasicBlock *BB = BBWorkList.back();
      BBWorkList.pop_back();

      DEBUG(dbgs() << "\nPopped off BBWL: " << *BB);

      // Notify all instructions in this basic block that they are newly
      // executable.
      for (Instruction &I : *BB)
        visitInst(I);
    }
  }
}

template <class LatticeVal>
void SparseSolver<LatticeVal>::Print(Function &F, raw_ostream &OS) const {
  OS << "\nFUNCTION: " << F.getName() << "\n";
  for (auto &BB : F) {
    if (!BBExecutable.count(&BB))
      OS << "INFEASIBLE: ";
    OS << "\t";
    if (BB.hasName())
      OS << BB.getName() << ":\n";
    else
      OS << "; anon bb\n";
    for (auto &I : BB) {
      LatticeFunc->PrintValue(getExistingValueState(&I), OS);
      OS << I << "\n";
    }

    OS << "\n";
  }
}
} // end namespace llvm

#undef DEBUG_TYPE

#endif // LLVM_ANALYSIS_SPARSEPROPAGATION_H
