//===- llvm/CodeGen/SSAIfConv.h - SSAIfConv ----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The SSAIfConv class performs if-conversion on SSA form machine code after
// determining if it is possible. The class contains no heuristics; external
// code should be used to determine when if-conversion is a good idea.
//
// SSAIfConv can convert both triangles and diamonds:
//
//   Triangle: Head              Diamond: Head
//              | \                       /  \_
//              |  \                     /    |
//              |  [TF]BB              FBB    TBB
//              |  /                     \    /
//              | /                       \  /
//             Tail                       Tail
//
// Instructions in the conditional blocks TBB and/or FBB are spliced into the
// Head block, and phis in the Tail block are converted to select instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SparseSet.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineTraceMetrics.h"

#ifndef LLVM_CODEGEN_SSA_IF_CONV_H
#define LLVM_CODEGEN_SSA_IF_CONV_H
namespace llvm {
class SSAIfConv {
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  MachineDominatorTree *DomTree;
  MachineLoopInfo *Loops;
  MachineTraceMetrics *Traces;

public:
  /// The block containing the conditional branch.
  MachineBasicBlock *Head;

  /// The block containing phis after the if-then-else.
  MachineBasicBlock *Tail;

  /// The 'true' conditional block as determined by analyzeBranch.
  MachineBasicBlock *TBB;

  /// The 'false' conditional block as determined by analyzeBranch.
  MachineBasicBlock *FBB;

  /// isTriangle - When there is no 'else' block, either TBB or FBB will be
  /// equal to Tail.
  bool isTriangle() const { return TBB == Tail || FBB == Tail; }

  /// Returns the Tail predecessor for the True side.
  MachineBasicBlock *getTPred() const { return TBB == Tail ? Head : TBB; }

  /// Returns the Tail predecessor for the  False side.
  MachineBasicBlock *getFPred() const { return FBB == Tail ? Head : FBB; }

  /// Information about each phi in the Tail block.
  struct PHIInfo {
    MachineInstr *PHI;
    unsigned TReg = 0, FReg = 0;
    // Latencies from Cond+Branch, TReg, and FReg to DstReg.
    int CondCycles = 0, TCycles = 0, FCycles = 0;

    PHIInfo(MachineInstr *phi) : PHI(phi) {}
  };

  SmallVector<PHIInfo, 8> PHIs;

  /// The branch condition determined by analyzeBranch.
  SmallVector<MachineOperand, 4> Cond;

  struct PredicationStrategyBase {
    virtual bool canConvertIf(MachineBasicBlock *Head, MachineBasicBlock *TBB,
                              MachineBasicBlock *FBB, MachineBasicBlock *Tail,
                              ArrayRef<MachineOperand> Cond) {
      return true;
    }
    virtual bool canPredicateInstr(const MachineInstr &I) = 0;
    /// Apply cost model and heuristics to the if-conversion in IfConv.
    /// Return true if the conversion is a good idea.
    virtual bool shouldConvertIf(SSAIfConv &) = 0;
    virtual void predicateBlock(MachineBasicBlock *MBB,
                                ArrayRef<MachineOperand> Cond,
                                bool Reverse) = 0;
    virtual ~PredicationStrategyBase() = default;
  };

  PredicationStrategyBase &Predicate;

public:
  SSAIfConv(PredicationStrategyBase &Predicate, MachineFunction &MF,
            MachineDominatorTree *DomTree, MachineLoopInfo *Loops,
            MachineTraceMetrics *Traces = nullptr);

  bool run();

  MachineTraceMetrics::Ensemble *getEnsemble(MachineTraceStrategy S);

private:
  /// Instructions in Head that define values used by the conditional blocks.
  /// The hoisted instructions must be inserted after these instructions.
  SmallPtrSet<MachineInstr *, 8> InsertAfter;

  /// Register units clobbered by the conditional blocks.
  BitVector ClobberedRegUnits;

  // Scratch pad for findInsertionPoint.
  SparseSet<unsigned> LiveRegUnits;

  /// Insertion point in Head for speculatively executed instructions form TBB
  /// and FBB.
  MachineBasicBlock::iterator InsertionPoint;

  /// Return true if all non-terminator instructions in MBB can be safely
  /// predicated.
  bool canPredicateInstrs(MachineBasicBlock *MBB);

  /// Scan through instruction dependencies and update InsertAfter array.
  /// Return false if any dependency is incompatible with if conversion.
  bool InstrDependenciesAllowIfConv(MachineInstr *I);

  /// Find a valid insertion point in Head.
  bool findInsertionPoint();

  /// Replace PHI instructions in Tail with selects.
  void replacePHIInstrs();

  /// Insert selects and rewrite PHI operands to use them.
  void rewritePHIOperands();

  /// canConvertIf - If the sub-CFG headed by MBB can be if-converted,
  /// initialize the internal state, and return true.
  bool canConvertIf(MachineBasicBlock *MBB);

  /// convertIf - If-convert the last block passed to canConvertIf(), assuming
  /// it is possible. Add any blocks that are to be erased to RemoveBlocks.
  void convertIf(SmallVectorImpl<MachineBasicBlock *> &RemoveBlocks);

  /// Attempt repeated if-conversion on MBB, return true if successful.
  bool tryConvertIf(MachineBasicBlock *);

  /// Invalidate MachineTraceMetrics before if-conversion.
  void invalidateTraces();

  /// Update the dominator tree after if-conversion erased some blocks.
  void updateDomTree(ArrayRef<MachineBasicBlock *> Removed);

  /// Update LoopInfo after if-conversion.
  void updateLoops(ArrayRef<MachineBasicBlock *> Removed);
};

} // namespace llvm

#endif