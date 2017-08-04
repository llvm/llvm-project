//===------ ForwardOpTree.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Move instructions between statements.
//
//===----------------------------------------------------------------------===//

#include "polly/ForwardOpTree.h"

#include "polly/ScopBuilder.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/VirtualInstruction.h"
#include "llvm/Analysis/ValueTracking.h"

#define DEBUG_TYPE "polly-delicm"

using namespace polly;
using namespace llvm;

STATISTIC(TotalInstructionsCopied, "Number of copied instructions");
STATISTIC(TotalReadOnlyCopied, "Number of copied read-only accesses");
STATISTIC(TotalForwardedTrees, "Number of forwarded operand trees");
STATISTIC(TotalModifiedStmts,
          "Number of statements with at least one forwarded tree");

STATISTIC(ScopsModified, "Number of SCoPs with at least one forwarded tree");

namespace {

/// The state of whether an operand tree was/can be forwarded.
///
/// The items apply to an instructions and its operand tree with the instruction
/// as the root element. If the value in question is not an instruction in the
/// SCoP, it can be a leaf of an instruction's operand tree.
enum ForwardingDecision {
  /// The root instruction or value cannot be forwarded at all.
  FD_CannotForward,

  /// The root instruction or value can be forwarded as a leaf of a larger
  /// operand tree.
  /// It does not make sense to move the value itself, it would just replace it
  /// by a use of itself. For instance, a constant "5" used in a statement can
  /// be forwarded, but it would just replace it by the same constant "5".
  /// However, it makes sense to move as an operand of
  ///
  ///   %add = add 5, 5
  ///
  /// where "5" is moved as part of a larger operand tree. "5" would be placed
  /// (disregarding for a moment that literal constants don't have a location
  /// and can be used anywhere) into the same statement as %add would.
  FD_CanForwardLeaf,

  /// The root instruction can be forwarded in a non-trivial way. This requires
  /// the operand tree root to be an instruction in some statement.
  FD_CanForwardTree,

  /// Used to indicate that a forwarding has be carried out successfully.
  FD_DidForward,

  /// A forwarding method cannot be applied to the operand tree.
  /// The difference to FD_CannotForward is that there might be other methods
  /// that can handle it.
  /// The conditions that make an operand tree applicable must be checked even
  /// with DoIt==true because a method following the one that returned
  /// FD_NotApplicable might have returned FD_CanForwardTree.
  FD_NotApplicable
};

/// Implementation of operand tree forwarding for a specific SCoP.
///
/// For a statement that requires a scalar value (through a value read
/// MemoryAccess), see if its operand can be moved into the statement. If so,
/// the MemoryAccess is removed and the all the operand tree instructions are
/// moved into the statement. All original instructions are left in the source
/// statements. The simplification pass can clean these up.
class ForwardOpTreeImpl {
private:
  /// The SCoP we are currently processing.
  Scop *S;

  /// LoopInfo is required for VirtualUse.
  LoopInfo *LI;

  /// How many instructions have been copied to other statements.
  int NumInstructionsCopied = 0;

  /// How many read-only accesses have been copied.
  int NumReadOnlyCopied = 0;

  /// How many operand trees have been forwarded.
  int NumForwardedTrees = 0;

  /// Number of statements with at least one forwarded operand tree.
  int NumModifiedStmts = 0;

  /// Whether we carried out at least one change to the SCoP.
  bool Modified = false;

  void printStatistics(raw_ostream &OS, int Indent = 0) {
    OS.indent(Indent) << "Statistics {\n";
    OS.indent(Indent + 4) << "Instructions copied: " << NumInstructionsCopied
                          << '\n';
    OS.indent(Indent + 4) << "Read-only accesses copied: " << NumReadOnlyCopied
                          << '\n';
    OS.indent(Indent + 4) << "Operand trees forwarded: " << NumForwardedTrees
                          << '\n';
    OS.indent(Indent + 4) << "Statements with forwarded operand trees: "
                          << NumModifiedStmts << '\n';
    OS.indent(Indent) << "}\n";
  }

  void printStatements(llvm::raw_ostream &OS, int Indent = 0) const {
    OS.indent(Indent) << "After statements {\n";
    for (auto &Stmt : *S) {
      OS.indent(Indent + 4) << Stmt.getBaseName() << "\n";
      for (auto *MA : Stmt)
        MA->print(OS);

      OS.indent(Indent + 12);
      Stmt.printInstructions(OS);
    }
    OS.indent(Indent) << "}\n";
  }

  /// Forwards a speculatively executable instruction.
  ///
  /// If the instruction itself cannot be executed speculatively, returns
  /// FD_NotApplicable.
  ///
  /// The parameters the same as for
  /// @see forwardTree()
  ForwardingDecision forwardSpeculatable(ScopStmt *TargetStmt,
                                         Instruction *UseInst,
                                         ScopStmt *UseStmt, Loop *UseLoop,
                                         bool DoIt) {
    // PHIs, unless synthesizable, are not yet supported.
    if (isa<PHINode>(UseInst))
      return FD_NotApplicable;

    // Compatible instructions must satisfy the following conditions:
    // 1. Idempotent (instruction will be copied, not moved; although its
    //    original instance might be removed by simplification)
    // 2. Not access memory (There might be memory writes between)
    // 3. Not cause undefined behaviour (we might copy to a location when the
    //    original instruction was no executed; this is currently not possible
    //    because we do not forward PHINodes)
    // 4. Not leak memory if executed multiple times (i.e. malloc)
    //
    // Instruction::mayHaveSideEffects is not sufficient because it considers
    // malloc to not have side-effects. llvm::isSafeToSpeculativelyExecute is
    // not sufficient because it allows memory accesses.
    if (mayBeMemoryDependent(*UseInst))
      return FD_NotApplicable;

    Loop *DefLoop = LI->getLoopFor(UseInst->getParent());
    ScopStmt *DefStmt = S->getStmtFor(UseInst);
    assert(DefStmt && "Value must be defined somewhere");

    if (DoIt) {
      // To ensure the right order, prepend this instruction before its
      // operands. This ensures that its operands are inserted before the
      // instruction using them.
      // TODO: The operand tree is not really a tree, but a DAG. We should be
      // able to handle DAGs without duplication.
      TargetStmt->prependInstruction(UseInst);
      NumInstructionsCopied++;
      TotalInstructionsCopied++;
    }

    for (Value *OpVal : UseInst->operand_values()) {
      ForwardingDecision OpDecision =
          forwardTree(TargetStmt, OpVal, DefStmt, DefLoop, DoIt);
      switch (OpDecision) {
      case FD_CannotForward:
        assert(!DoIt);
        return FD_CannotForward;

      case FD_CanForwardLeaf:
      case FD_CanForwardTree:
        assert(!DoIt);
        break;

      case FD_DidForward:
        assert(DoIt);
        break;

      case FD_NotApplicable:
        llvm_unreachable("forwardTree should never return FD_NotApplicable");
      }
    }

    if (DoIt)
      return FD_DidForward;
    return FD_CanForwardTree;
  }

  /// Determines whether an operand tree can be forwarded or carries out a
  /// forwarding, depending on the @p DoIt flag.
  ///
  /// @param TargetStmt The statement the operand tree will be copied to.
  /// @param UseVal     The value (usually an instruction) which is root of an
  ///                   operand tree.
  /// @param UseStmt    The statement that uses @p UseVal.
  /// @param UseLoop    The loop @p UseVal is used in.
  /// @param DoIt       If false, only determine whether an operand tree can be
  ///                   forwarded. If true, carry out the forwarding. Do not use
  ///                   DoIt==true if an operand tree is not known to be
  ///                   forwardable.
  ///
  /// @return If DoIt==false, return whether the operand tree can be forwarded.
  ///         If DoIt==true, return FD_DidForward.
  ForwardingDecision forwardTree(ScopStmt *TargetStmt, Value *UseVal,
                                 ScopStmt *UseStmt, Loop *UseLoop, bool DoIt) {
    VirtualUse VUse = VirtualUse::create(UseStmt, UseLoop, UseVal, true);
    switch (VUse.getKind()) {
    case VirtualUse::Constant:
    case VirtualUse::Block:
    case VirtualUse::Hoisted:
      // These can be used anywhere without special considerations.
      if (DoIt)
        return FD_DidForward;
      return FD_CanForwardLeaf;

    case VirtualUse::Synthesizable: {
      // ScopExpander will take care for of generating the code at the new
      // location.
      if (DoIt)
        return FD_DidForward;

      // Check if the value is synthesizable at the new location as well. This
      // might be possible when leaving a loop for which ScalarEvolution is
      // unable to derive the exit value for.
      // TODO: If there is a LCSSA PHI at the loop exit, use that one.
      // If the SCEV contains a SCEVAddRecExpr, we currently depend on that we
      // do not forward past its loop header. This would require us to use a
      // previous loop induction variable instead the current one. We currently
      // do not allow forwarding PHI nodes, thus this should never occur (the
      // only exception where no phi is necessary being an unreachable loop
      // without edge from the outside).
      VirtualUse TargetUse = VirtualUse::create(
          S, TargetStmt, TargetStmt->getSurroundingLoop(), UseVal, true);
      if (TargetUse.getKind() == VirtualUse::Synthesizable)
        return FD_CanForwardLeaf;

      DEBUG(dbgs() << "    Synthesizable would not be synthesizable anymore: "
                   << *UseVal << "\n");
      return FD_CannotForward;
    }

    case VirtualUse::ReadOnly:
      // Note that we cannot return FD_CanForwardTree here. With a operand tree
      // depth of 0, UseVal is the use in TargetStmt that we try to replace.
      // With -polly-analyze-read-only-scalars=true we would ensure the
      // existence of a MemoryAccess (which already exists for a leaf) and be
      // removed again by tryForwardTree because it's goal is to remove this
      // scalar MemoryAccess. It interprets FD_CanForwardTree as the permission
      // to do so.
      if (!DoIt)
        return FD_CanForwardLeaf;

      // If we model read-only scalars, we need to create a MemoryAccess for it.
      if (ModelReadOnlyScalars)
        TargetStmt->ensureValueRead(UseVal);

      NumReadOnlyCopied++;
      TotalReadOnlyCopied++;
      return FD_DidForward;

    case VirtualUse::Intra:
    case VirtualUse::Inter:
      auto Inst = cast<Instruction>(UseVal);

      ForwardingDecision SpeculativeResult =
          forwardSpeculatable(TargetStmt, Inst, UseStmt, UseLoop, DoIt);
      if (SpeculativeResult != FD_NotApplicable)
        return SpeculativeResult;

      // When no method is found to forward the operand tree, we effectively
      // cannot handle it.
      DEBUG(dbgs() << "    Cannot forward instruction: " << *Inst << "\n");
      return FD_CannotForward;
    }

    llvm_unreachable("Case unhandled");
  }

  /// Try to forward an operand tree rooted in @p RA.
  bool tryForwardTree(MemoryAccess *RA) {
    assert(RA->isLatestScalarKind());
    DEBUG(dbgs() << "Trying to forward operand tree " << RA << "...\n");

    ScopStmt *Stmt = RA->getStatement();
    Loop *InLoop = Stmt->getSurroundingLoop();

    ForwardingDecision Assessment =
        forwardTree(Stmt, RA->getAccessValue(), Stmt, InLoop, false);
    assert(Assessment != FD_DidForward);
    if (Assessment != FD_CanForwardTree)
      return false;

    ForwardingDecision Execution =
        forwardTree(Stmt, RA->getAccessValue(), Stmt, InLoop, true);
    assert(Execution == FD_DidForward &&
           "A previous positive assessment must also be executable");
    (void)Execution;

    Stmt->removeSingleMemoryAccess(RA);
    return true;
  }

public:
  ForwardOpTreeImpl(Scop *S, LoopInfo *LI) : S(S), LI(LI) {}

  /// Return which SCoP this instance is processing.
  Scop *getScop() const { return S; }

  /// Run the algorithm: Use value read accesses as operand tree roots and try
  /// to forward them into the statement.
  bool forwardOperandTrees() {
    for (ScopStmt &Stmt : *S) {
      // Currently we cannot modify the instruction list of region statements.
      if (!Stmt.isBlockStmt())
        continue;

      bool StmtModified = false;

      // Because we are modifying the MemoryAccess list, collect them first to
      // avoid iterator invalidation.
      SmallVector<MemoryAccess *, 16> Accs;
      for (MemoryAccess *RA : Stmt) {
        if (!RA->isRead())
          continue;
        if (!RA->isLatestScalarKind())
          continue;

        Accs.push_back(RA);
      }

      for (MemoryAccess *RA : Accs) {
        if (tryForwardTree(RA)) {
          Modified = true;
          StmtModified = true;
          NumForwardedTrees++;
          TotalForwardedTrees++;
        }
      }

      if (StmtModified) {
        NumModifiedStmts++;
        TotalModifiedStmts++;
      }
    }

    if (Modified)
      ScopsModified++;
    return Modified;
  }

  /// Print the pass result, performed transformations and the SCoP after the
  /// transformation.
  void print(llvm::raw_ostream &OS, int Indent = 0) {
    printStatistics(OS, Indent);

    if (!Modified) {
      // This line can easily be checked in regression tests.
      OS << "ForwardOpTree executed, but did not modify anything\n";
      return;
    }

    printStatements(OS, Indent);
  }
};

/// Pass that redirects scalar reads to array elements that are known to contain
/// the same value.
///
/// This reduces the number of scalar accesses and therefore potentially
/// increases the freedom of the scheduler. In the ideal case, all reads of a
/// scalar definition are redirected (We currently do not care about removing
/// the write in this case).  This is also useful for the main DeLICM pass as
/// there are less scalars to be mapped.
class ForwardOpTree : public ScopPass {
private:
  ForwardOpTree(const ForwardOpTree &) = delete;
  const ForwardOpTree &operator=(const ForwardOpTree &) = delete;

  /// The pass implementation, also holding per-scop data.
  std::unique_ptr<ForwardOpTreeImpl> Impl;

public:
  static char ID;

  explicit ForwardOpTree() : ScopPass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitive<ScopInfoRegionPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesAll();
  }

  virtual bool runOnScop(Scop &S) override {
    // Free resources for previous SCoP's computation, if not yet done.
    releaseMemory();

    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    Impl = make_unique<ForwardOpTreeImpl>(&S, &LI);

    DEBUG(dbgs() << "Forwarding operand trees...\n");
    Impl->forwardOperandTrees();

    DEBUG(dbgs() << "\nFinal Scop:\n");
    DEBUG(dbgs() << S);

    return false;
  }

  virtual void printScop(raw_ostream &OS, Scop &S) const override {
    if (!Impl)
      return;

    assert(Impl->getScop() == &S);
    Impl->print(OS);
  }

  virtual void releaseMemory() override { Impl.reset(); }

}; // class ForwardOpTree

char ForwardOpTree::ID;
} // anonymous namespace

ScopPass *polly::createForwardOpTreePass() { return new ForwardOpTree(); }

INITIALIZE_PASS_BEGIN(ForwardOpTree, "polly-optree",
                      "Polly - Forward operand tree", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(ForwardOpTree, "polly-optree",
                    "Polly - Forward operand tree", false, false)
