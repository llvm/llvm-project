//===- AMDGPUMirDivergenceAnalysis.h -  Mir Divergence Analysis -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// The divergence analysis determines which instructions and branches are
// divergent given a set of divergent source instructions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AMDGPUMirSyncDependenceAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Pass.h"
#include <vector>

namespace llvm {
class raw_ostream;
class TargetTransformInfo;
class MachineRegisterInfo;
class SIInstrInfo;
class SIRegisterInfo;
class MachineOperand;
class MachineBasicBlock;

using Module_ = void;
class TargetTransformInfo;
using ValueTy = unsigned;
using PHINode_ = MachineInstr;

/// \brief Generic divergence analysis for reducible CFGs.
///
/// This analysis propagates divergence in a data-parallel context from sources
/// of divergence to all users. It requires reducible CFGs. All assignments
/// should be in SSA form.
class DivergenceAnalysis {
public:
  /// \brief This instance will analyze the whole function \p F or the loop \p
  /// RegionLoop.
  ///
  /// \param RegionLoop if non-null the analysis is restricted to \p RegionLoop.
  /// Otherwise the whole function is analyzed.
  /// \param IsLCSSAForm whether the analysis may assume that the IR in the
  /// region in in LCSSA form.
  DivergenceAnalysis(const llvm::MachineFunction &F,
                     const MachineLoop *RegionLoop,
                     const MachineDominatorTree &DT,
                     const MachinePostDominatorTree &PDT,
                     const MachineLoopInfo &LI, SyncDependenceAnalysis &SDA,
                     bool IsLCSSAForm,
                     // AMDGPU change begin.
                     DivergentJoinMapTy &JoinMap
                     // AMDGPU change end.
  );

  /// \brief The loop that defines the analyzed region (if any).
  const MachineLoop *getRegionLoop() const { return RegionLoop; }
  const llvm::MachineFunction &getFunction() const { return F; }

  /// \brief Whether \p BB is part of the region.
  bool inRegion(const MachineBasicBlock &BB) const;
  /// \brief Whether \p I is part of the region.
  bool inRegion(const MachineInstr &I) const;

  /// \brief Mark \p UniVal as a value that is always uniform.
  void addUniformOverride(const ValueTy UniVal);
  void addUniformOverride(const MachineInstr &I);

  /// \brief Mark \p DivVal as a value that is always divergent.
  void markDivergent(const ValueTy DivVal);
  void markDivergent(const MachineInstr &I);

  /// \brief Propagate divergence to all instructions in the region.
  /// Divergence is seeded by calls to \p markDivergent.
  void compute();

  /// \brief Whether any value was marked or analyzed to be divergent.
  bool hasDetectedDivergence() const { return !DivergentValues.empty(); }

  /// \brief Whether \p Val will always return a uniform value regardless of its
  /// operands
  bool isAlwaysUniform(const ValueTy Val) const;

  /// \brief Whether \p Val is a divergent value
  bool isDivergent(const ValueTy Val) const;
  bool isDivergent(const MachineInstr &I) const;

  void print(llvm::raw_ostream &OS, const Module_ *) const;

private:
  bool isDivergent(const llvm::MachineOperand &MO) const;
  bool updateTerminator(const MachineInstr &Term) const;
  bool updatePHINode(const PHINode_ &Phi) const;
  bool updateVCndMask(const MachineInstr &VCndMask) const;
  bool
  isBitUniform(const MachineInstr &I,
               llvm::DenseMap<const MachineInstr *, bool> &Processed) const;
  bool
  isBitUniform(const MachineInstr &I, const llvm::MachineOperand &UseMO,
               llvm::DenseMap<const MachineInstr *, bool> &Processed) const;

  /// \brief Computes whether \p Inst is divergent based on the
  /// divergence of its operands.
  ///
  /// \returns Whether \p Inst is divergent.
  ///
  /// This should only be called for non-phi, non-terminator instructions.
  bool updateNormalInstruction(const MachineInstr &Inst) const;

  /// \brief Mark users of live-out users as divergent.
  ///
  /// \param LoopHeader the header of the divergent loop.
  ///
  /// Marks all users of live-out values of the loop headed by \p LoopHeader
  /// as divergent and puts them on the worklist.
  void taintLoopLiveOuts(const MachineBasicBlock &LoopHeader);

  /// \brief Push all users of \p Val (in the region) to the worklist
  void pushUsers(const ValueTy I);
  void pushUsers(const MachineInstr &I);

  void pushInstruction(const MachineInstr &I);
  /// \brief Push all phi nodes in @block to the worklist
  void pushPHINodes(const MachineBasicBlock &Block);

  /// \brief Mark \p Block as join divergent
  ///
  /// A block is join divergent if two threads may reach it from different
  /// incoming blocks at the same time.
  void markBlockJoinDivergent(const MachineBasicBlock &Block) {
    DivergentJoinBlocks.insert(&Block);
  }

  /// \brief Whether \p Val is divergent when read in \p ObservingBlock.
  bool isTemporalDivergent(
      const MachineBasicBlock &ObservingBlock, const ValueTy Val,
      const MachineBasicBlock &incomingBlock) const; // AMDGPU change

  /// \brief Whether \p Block is join divergent
  ///
  /// (see markBlockJoinDivergent).
  bool isJoinDivergent(const MachineBasicBlock &Block) const {
    return DivergentJoinBlocks.find(&Block) != DivergentJoinBlocks.end();
  }

  /// \brief Propagate control-induced divergence to users (phi nodes and
  /// instructions).
  //
  // \param JoinBlock is a divergent loop exit or join point of two disjoint
  // paths.
  // \returns Whether \p JoinBlock is a divergent loop exit of \p TermLoop.
  bool propagateJoinDivergence(const MachineBasicBlock &JoinBlock,
                               const MachineLoop *TermLoop);

  /// \brief Propagate induced value divergence due to control divergence in \p
  /// Term.
  void propagateBranchDivergence(const MachineInstr &Term);

  /// \brief Propagate induced value divergence due to exec update caused by \p
  /// SaveExec.
  void propagateExecControlFlowDivergence(const MachineInstr &SaveExec);

  /// \brief Propagate divergent caused by a divergent loop exit.
  ///
  /// \param ExitingLoop is a divergent loop.
  void propagateLoopDivergence(const MachineLoop &ExitingLoop);

private:
  const llvm::MachineFunction &F;
  const llvm::MachineRegisterInfo &MRI;
  const llvm::SIRegisterInfo *SIRI;
  const llvm::SIInstrInfo *SIII;
  // If regionLoop != nullptr, analysis is only performed within \p RegionLoop.
  // Otw, analyze the whole function
  const MachineLoop *RegionLoop;

  const MachineDominatorTree &DT;
  const MachinePostDominatorTree &PDT;
  const MachineLoopInfo &LI;

  // Recognized divergent loops
  llvm::DenseSet<const MachineLoop *> DivergentLoops;

  // AMDGPU change begin
  // Save block pair which divergent disjoint.
  // A
  // | \
  // |  \
  // B   C
  // |  /
  //  D
  // When A is divergent branch, B and C are divergent join at D.
  // Then DivergentJoinMap[B].count(C) > 0 and
  // DivergentJoinMap[C].count(B) > 0.
  DivergentJoinMapTy &DivergentJoinMap;
  // AMDGPU change end

  // The SDA links divergent branches to divergent control-flow joins.
  SyncDependenceAnalysis &SDA;

  // Use simplified code path for LCSSA form.
  bool IsLCSSAForm;

  // Set of known-uniform values.
  llvm::DenseSet<unsigned> UniformOverrides;
  llvm::DenseSet<const llvm::MachineInstr *> UniformOverridesInsts;

  // Blocks with joining divergent control from different predecessors.
  llvm::DenseSet<const MachineBasicBlock *> DivergentJoinBlocks;

  // Detected/marked divergent values.
  llvm::DenseSet<unsigned> DivergentValues;
  llvm::DenseSet<const llvm::MachineInstr *> DivergentInsts;

  // Mir change for EXEC control flow.
  // Map from MBB to the exec region it belongs too.
  // A exec region is begin with
  // S_MOV_B64 sreg, exec
  // end with
  // S_MOV_B64 exec, sreg
  // Inside the region, exec might be updated to make control flow with exec.
  struct ExecRegion {
    const llvm::MachineInstr *begin;
    const llvm::MachineInstr *end;
    std::vector<const llvm::MachineBasicBlock *> blocks;
    bool bPropagated = false;
    ExecRegion(const llvm::MachineInstr *b, const llvm::MachineInstr *e)
        : begin(b), end(e), bPropagated(false) {}
  };
  llvm::DenseMap<const llvm::MachineBasicBlock *, ExecRegion *> ExecRegionMap;

  // Internal worklist for divergence propagation.
  std::vector<const llvm::MachineInstr *> Worklist;
};

/// \brief Divergence analysis frontend for GPU kernels.
class MirGPUDivergenceAnalysis {
  // AMDGPU change begin
  // Save block pair which divergent disjoint.
  // A
  // | \
  // |  \
  // B   C
  // |  /
  //  D
  // When A is divergent branch, B and C are divergent join at D.
  // Then DivergentJoinMap[B].count(C) > 0 and
  // DivergentJoinMap[C].count(B) > 0.
  DivergentJoinMapTy DivergentJoinMap;
  // AMDGPU change end
  SyncDependenceAnalysis SDA;
  DivergenceAnalysis DA;

public:
  /// Runs the divergence analysis on @F, a GPU kernel
  MirGPUDivergenceAnalysis(llvm::MachineFunction &F,
                           const MachineDominatorTree &DT,
                           const MachinePostDominatorTree &PDT,
                           const MachineLoopInfo &LI);

  /// Whether any divergence was detected.
  bool hasDivergence() const { return DA.hasDetectedDivergence(); }

  /// The GPU kernel this analysis result is for
  const llvm::MachineFunction &getFunction() const { return DA.getFunction(); }

  /// Whether \p I is divergent.
  bool isDivergent(const MachineInstr *I) const;

  /// Whether \p I is uniform/non-divergent
  bool isUniform(const MachineInstr *I) const { return !isDivergent(I); }

  /// Print all divergent values in the kernel.
  void print(llvm::raw_ostream &OS, const Module_ *) const;
};

} // namespace llvm
