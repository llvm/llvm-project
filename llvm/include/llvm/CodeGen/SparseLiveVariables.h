//===-- SparseLiveVariables.h - Sparse Live Variable Analysis ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a sparse, target-independent liveness analysis pass.
//
// Design Decisions & Computational Sparsity:
// 1. Data Structure Sparsity: Virtual registers in LLVM have very large IDs
//    (starting at 2^30). Tracking their liveness using a dense BitVector would
//    allocate an exorbitant amount of memory. This pass uses SparseBitVector
//    to lazily allocate small chunks, keeping the memory footprint tiny even
//    for sparse, high-ID virtual registers.
//
// 2. Computational Sparsity (Statelessness): Unlike the legacy LiveVariables
//    pass which computes and caches full liveness intervals and records every
//    Def/Kill point globally, this pass strictly computes only the block-level
//    boundary conditions (LiveIn and LiveOut). Instruction-level liveness
//    is intentionally not cached. Instead, queries are evaluated "on the fly"
//    by walking backward from the block's LiveOut set using a LivenessTracker.
//    This trades a small amount of compute for significant memory savings and
//    eliminates the maintenance burden of massive state caches.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CODEGEN_SPARSELIVEVARIABLES_H
#define LLVM_CODEGEN_SPARSELIVEVARIABLES_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/PassRegistry.h"

namespace llvm {

/// SparseLiveVariables - A target-independent liveness analysis pass.
///
/// This pass computes block-level live-in and live-out sets using a
/// SparseBitVector representation. It is designed to be a lightweight,
/// memory-efficient alternative to the legacy LiveVariables pass. It operates
/// as a read-only analysis but provides mutation APIs (`updateLiveIns`)
/// to explicitly update the IR state if desired.
class SparseLiveVariables {
public:
  SparseLiveVariables() = default;

  struct BlockInfo {
    SparseBitVector<> LiveIn;
    SparseBitVector<> LiveOut;
  };

  /// Liveness information per block, indexed by MBB->getNumber().
  /// We size this vector to MF.getNumBlockIDs() to guarantee bounds safety.
  /// Note: Block numbers may have gaps if blocks were deleted. We intentionally
  /// leave these gap indices empty (unused) rather than calling
  /// MF.RenumberBlocks(), as doing so would mutate the function and invalidate
  /// other analysis passes.
  std::vector<BlockInfo> BlockLiveness;

  /// LivenessTracker - A utility class for backward liveness traversal.
  ///
  /// This class tracks the liveness state of registers as you step backward
  /// through a MachineBasicBlock. It is initialized with the Live-Out set of
  /// the block and updated by calling `stepBackward(MI)` on each instruction.
  ///
  /// Note: The SparseLiveVariables pass itself is stateless at the instruction
  /// level. To query instruction-level liveness dynamically, you must use this
  /// tracker or the `isLiveAt`/`isLiveAfter` methods (which internally use it).
  class LivenessTracker {
    SparseBitVector<> LiveRegs;
    const MachineRegisterInfo *MRI;

  public:
    bool isTrackableRegister(Register Reg) const {
      if (Reg.isVirtual())
        return true;
      if (Reg.isPhysical()) {
        if (MRI->isReserved(Reg))
          return false;
        return true;
      }
      return false;
    }

    LivenessTracker(const SparseBitVector<> &LiveOut,
                    const MachineRegisterInfo *MRI)
        : LiveRegs(LiveOut), MRI(MRI) {}

    void stepBackward(const MachineInstr &MI) {
      if (MI.isDebugInstr())
        return;

      SmallVector<Register, 4> Uses;
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;
        Register Reg = MO.getReg();
        if (!Reg.isValid() || !isTrackableRegister(Reg))
          continue;

        if (MO.isDef())
          LiveRegs.reset(Reg.id());
        if (!MI.isPHI() && MO.isUse())
          Uses.push_back(Reg);
      }
      for (Register Reg : Uses)
        LiveRegs.set(Reg.id());
    }

    bool isLive(Register Reg) const {
      if (!Reg.isValid())
        return false;
      return LiveRegs.test(Reg.id());
    }

    const SparseBitVector<> &getLiveSet() const { return LiveRegs; }
  };

  /// Returns true if the given MachineBasicBlock has been analyzed.
  bool hasAnalyzed(const MachineBasicBlock *MBB) const {
    return MBB->getNumber() >= 0 &&
           (size_t)MBB->getNumber() < BlockLiveness.size();
  }

  /// Returns the computed Live-In set for the given MachineBasicBlock.
  const SparseBitVector<> &getLiveInSet(const MachineBasicBlock *MBB) const {
    assert(hasAnalyzed(MBB) && "Block not analyzed");
    return BlockLiveness[MBB->getNumber()].LiveIn;
  }

  /// Returns the computed Live-Out set for the given MachineBasicBlock.
  const SparseBitVector<> &getLiveOutSet(const MachineBasicBlock *MBB) const {
    assert(hasAnalyzed(MBB) && "Block not analyzed");
    return BlockLiveness[MBB->getNumber()].LiveOut;
  }

  /// Validates the computed block liveness against existing MachineBasicBlock
  /// live-ins.
  /// \brief Incrementally recomputes liveness for a specific register.
  ///
  /// This performs a highly localized $O(V + E)$ recomputation of liveness for
  /// the given register. It instantly clears the register's liveness from all
  /// blocks, then reseeds and propagates it from its remaining uses. This
  /// safely bypasses the "phantom live range" problem where loops artificially
  /// keep a register alive even after its uses have been deleted.
  ///
  /// \param Reg The virtual register to recompute.
  /// \param IgnoreMI An optional instruction to ignore during recomputation
  ///                 (useful for removing liveness before an instruction is
  ///                 physically deleted).
  void recomputeRegisterLiveness(Register Reg,
                                 MachineInstr *IgnoreMI = nullptr);

  /// \brief Update liveness after a pass adds a new instruction.
  void addInstruction(MachineInstr &MI, MachineBasicBlock *MBB);

  /// \brief Update liveness before a pass removes an instruction.
  ///
  /// The pass must call this *before* calling MI.eraseFromParent().
  void removeInstruction(MachineInstr &MI);

  /// \brief Update liveness after a pass moves an instruction.
  void handleMove(MachineInstr &MI, MachineBasicBlock *OldBB,
                  MachineBasicBlock *NewBB);

  /// \brief Update the live-in list of each MachineBasicBlock.
  ///
  /// This mutates the underlying `MachineBasicBlock` structures to sync their
  /// state.
  void verifyLiveness(const MachineFunction &MF) const;

  /// Update the live-ins of all basic blocks in MF based on computed liveness.
  void updateLiveIns(MachineFunction &MF) const;

  void analyze(MachineFunction &MF);

private:
  const MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;

  void propagateGrowth(Register Reg, MachineBasicBlock *StartBB,
                       MachineInstr *IgnoreMI = nullptr);
  void propagateShrinkage(Register Reg, MachineBasicBlock *StartBB,
                          MachineInstr *IgnoreMI = nullptr);

  bool evaluateLiveIn(Register Reg, MachineBasicBlock *MBB,
                      MachineInstr *IgnoreMI = nullptr) const;
  bool isLiveOut(Register Reg, MachineBasicBlock *MBB,
                 MachineInstr *IgnoreMI = nullptr) const;
  void reevaluateLiveIn(Register Reg, MachineBasicBlock *MBB,
                        MachineInstr *IgnoreMI = nullptr);
};

class SparseLiveVariablesAnalysis
    : public AnalysisInfoMixin<SparseLiveVariablesAnalysis> {
  friend AnalysisInfoMixin<SparseLiveVariablesAnalysis>;
  static AnalysisKey Key;

public:
  using Result = SparseLiveVariables;
  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &);
};

class SparseLiveVariablesWrapperPass : public MachineFunctionPass {
  SparseLiveVariables LV;

public:
  static char ID;

  SparseLiveVariablesWrapperPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    LV.analyze(MF);
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  SparseLiveVariables &getLV() { return LV; }

  StringRef getPassName() const override {
    return "Sparse Live Variable Analysis";
  }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_SPARSELIVEVARIABLES_H
