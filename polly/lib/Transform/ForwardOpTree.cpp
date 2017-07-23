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

#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/VirtualInstruction.h"
#include "llvm/Analysis/ValueTracking.h"

#define DEBUG_TYPE "polly-delicm"

using namespace polly;
using namespace llvm;

STATISTIC(TotalInstructionsCopied, "Number of copied instructions");
STATISTIC(TotalForwardedTrees, "Number of forwarded operand trees");
STATISTIC(TotalModifiedStmts,
          "Number of statements with at least one forwarded tree");

STATISTIC(ScopsModified, "Number of SCoPs with at least one forwarded tree");

namespace {

/// The state of whether an operand tree was/can be forwarded.
enum ForwardingDecision {
  FD_CannotForward,
  FD_CanForward,
  FD_DidForward,
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
  /// @return When DoIt==true, return whether the operand tree can be forwarded.
  ///         When DoIt==false, return FD_DidForward.
  ForwardingDecision canForwardTree(ScopStmt *TargetStmt, Value *UseVal,
                                    ScopStmt *UseStmt, Loop *UseLoop,
                                    bool DoIt) {

    // PHis are not yet supported.
    if (isa<PHINode>(UseVal)) {
      DEBUG(dbgs() << "    Cannot forward PHI: " << *UseVal << "\n");
      return FD_CannotForward;
    }

    VirtualUse VUse = VirtualUse::create(UseStmt, UseLoop, UseVal, true);
    switch (VUse.getKind()) {
    case VirtualUse::Constant:
    case VirtualUse::Block:
    case VirtualUse::Hoisted:
      // These can be used anywhere without special considerations.
      if (DoIt)
        return FD_DidForward;
      return FD_CanForward;

    case VirtualUse::Synthesizable:
      // Not supported yet.
      DEBUG(dbgs() << "    Cannot forward synthesizable: " << *UseVal << "\n");
      return FD_CannotForward;

    case VirtualUse::ReadOnly:
      // Not supported yet.
      DEBUG(dbgs() << "    Cannot forward read-only val: " << *UseVal << "\n");
      return FD_CannotForward;

    case VirtualUse::Intra:
    case VirtualUse::Inter:
      auto Inst = cast<Instruction>(UseVal);

      // Compatible instructions must satisfy the following conditions:
      // 1. Idempotent (instruction will be copied, not moved; although its
      // original instance might be removed by simplification)
      // 2. Not access memory (There might be memory writes between)
      // 3. Not cause undefined behaviour (we might copy to a location when the
      // original instruction was no executed; this is currently not possible
      // because we do not forward PHINodes)
      // 4. Not leak memory if executed multiple times (I am looking at you,
      // malloc!)
      //
      // Instruction::mayHaveSideEffects is not sufficient because it considers
      // malloc to not have side-effects. llvm::isSafeToSpeculativelyExecute is
      // not sufficient because it allows memory accesses.
      if (mayBeMemoryDependent(*Inst)) {
        DEBUG(dbgs() << "    Cannot forward side-effect instruction: " << *Inst
                     << "\n");
        return FD_CannotForward;
      }

      Loop *DefLoop = LI->getLoopFor(Inst->getParent());
      ScopStmt *DefStmt = S->getStmtFor(Inst);
      assert(DefStmt && "Value must be defined somewhere");

      if (DoIt) {
        // To ensure the right order, prepend this instruction before its
        // operands. This ensures that its operands are inserted before the
        // instruction using them.
        // TODO: The operand tree is not really a tree, but a DAG. We should be
        // able to handle DAGs without duplication.
        TargetStmt->prependInstrunction(Inst);
        NumInstructionsCopied++;
        TotalInstructionsCopied++;
      }

      for (Value *OpVal : Inst->operand_values()) {
        ForwardingDecision OpDecision =
            canForwardTree(TargetStmt, OpVal, DefStmt, DefLoop, DoIt);
        switch (OpDecision) {
        case FD_CannotForward:
          assert(!DoIt);
          return FD_CannotForward;

        case FD_CanForward:
          assert(!DoIt);
          break;

        case FD_DidForward:
          assert(DoIt);
          break;
        }
      }

      if (DoIt)
        return FD_DidForward;
      return FD_CanForward;
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
        canForwardTree(Stmt, RA->getAccessValue(), Stmt, InLoop, false);
    assert(Assessment != FD_DidForward);
    if (Assessment == FD_CannotForward)
      return false;

    ForwardingDecision Execution =
        canForwardTree(Stmt, RA->getAccessValue(), Stmt, InLoop, true);
    assert(Execution == FD_DidForward);

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
