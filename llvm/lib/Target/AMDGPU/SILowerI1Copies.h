//===-- SILowerI1Copies.h --------------------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Interface definition of the PhiLoweringHelper class that implements lane
/// mask merging algorithm for divergent i1 phis.
//
//===----------------------------------------------------------------------===//

#include "AMDGPULaneMaskUtils.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"

namespace llvm {
class MachineIDFSSAUpdater;
namespace AMDGPU {
class LoopFinder;
class PhiIncomingAnalysis;
/// Incoming for lane mask phi as machine instruction, incoming register \p Reg
/// and incoming block \p Block are taken from machine instruction.
/// \p UpdatedReg (if valid) is \p Reg lane mask merged with another lane mask.
struct Incoming {
  Register Reg;
  MachineBasicBlock *Block;
  Register UpdatedReg;

  Incoming(Register Reg, MachineBasicBlock *Block, Register UpdatedReg)
      : Reg(Reg), Block(Block), UpdatedReg(UpdatedReg) {}
};

Register createLaneMaskReg(MachineRegisterInfo *MRI,
                           MachineRegisterInfo::VRegAttrs LaneMaskRegAttrs);

class PhiLoweringHelper {
public:
  PhiLoweringHelper(MachineFunction &MF, MachineDominatorTree &DT,
                    MachinePostDominatorTree &PDT);
  virtual ~PhiLoweringHelper() = default;

protected:
  bool IsWave32 = false;
  MachineFunction &MF;
  MachineDominatorTree &DT;
  MachinePostDominatorTree &PDT;
  MachineRegisterInfo *MRI = nullptr;
  const GCNSubtarget *ST = nullptr;
  const SIInstrInfo *TII = nullptr;
  MachineRegisterInfo::VRegAttrs LaneMaskRegAttrs;
  const AMDGPU::LaneMaskConstants *LMC = nullptr;

#ifndef NDEBUG
  DenseSet<Register> PhiRegisters;
#endif

public:
  bool lowerPhis();
  bool isConstantLaneMask(Register Reg, bool &Val) const;

  /// Insert at \p I: \p DstReg = (\p PrevReg & ~EXEC) | (\p CurReg & EXEC).
  /// Constant 0 / all-ones / undef inputs omit the corresponding ANDN2, AND,
  /// or OR. Non-constant sources are passed through constrainOrCopyToLaneMask.
  void buildMergeLaneMasks(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, const DebugLoc &DL,
                           Register DstReg, Register PrevReg, Register CurReg);

  /// Return \p Reg if it is already a wave-sized SGPR, a physreg, or SDAG
  /// vreg_1. Otherwise (GISel generic s1) COPY it into a lane-mask vreg and
  /// return that.
  Register constrainOrCopyToLaneMask(Register Reg);

  /// Merge the \p Incomings lane masks into \p DstReg, the value owned by \p
  /// MBB. Builds the per-predecessor merges and leaves \p SSAUpdater calculated
  /// so the caller can query the merged value with getValueInMiddleOfBlock. \p
  /// Incomings is sorted and its UpdatedReg fields are filled in.
  void mergeIncomingLaneMasks(Register DstReg, MachineBasicBlock &MBB,
                              SmallVectorImpl<Incoming> &Incomings,
                              MachineIDFSSAUpdater &SSAUpdater, LoopFinder &LF,
                              PhiIncomingAnalysis &PIA);
  MachineBasicBlock::iterator
  getSaluInsertionAtEnd(MachineBasicBlock &MBB) const;

  void initializeLaneMaskRegisterAttributes(Register LaneMask) {
    LaneMaskRegAttrs = MRI->getVRegAttrs(LaneMask);
  }

  void
  initializeLaneMaskRegisterAttributes(MachineRegisterInfo::VRegAttrs Attrs) {
    LaneMaskRegAttrs = Attrs;
  }

  bool isLaneMaskReg(Register Reg) const {
    // GISel generic s1 is virtual but has no register class yet.
    // Physical VCC/EXEC are lane masks and must not use VRegInfo.
    if (Reg.isVirtual() && !MRI->getRegClassOrNull(Reg))
      return false;
    return TII->getRegisterInfo().isSGPRReg(*MRI, Reg) &&
           TII->getRegisterInfo().getRegSizeInBits(Reg, *MRI) ==
               ST->getWavefrontSize();
  }

  // Helpers from lowerPhis that are different between sdag and global-isel.

  virtual void markAsLaneMask(Register DstReg) const = 0;
  virtual void getCandidatesForLowering(
      SmallVectorImpl<MachineInstr *> &Vreg1Phis) const = 0;
  virtual void
  collectIncomingValuesFromPhi(const MachineInstr *MI,
                               SmallVectorImpl<Incoming> &Incomings) const = 0;
  virtual void replaceDstReg(Register NewReg, Register OldReg,
                             MachineBasicBlock *MBB) = 0;
  virtual void constrainAsLaneMask(Incoming &In) = 0;
};
} // namespace AMDGPU
} // end namespace llvm
