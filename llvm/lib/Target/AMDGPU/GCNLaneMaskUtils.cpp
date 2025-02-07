//===- GCNLaneMaskUtils.cpp --------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GCNLaneMaskUtils.h"

#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

/// Obtain a reference to the global wavefront-size dependent constants
/// based on \p wavefrontSize.
const GCNLaneMaskConstants *
GCNLaneMaskUtils::getConsts(unsigned WavefrontSize) {
  static const GCNLaneMaskConstants Wave32 = {
      AMDGPU::EXEC_LO,    AMDGPU::VCC_LO,         &AMDGPU::SReg_32RegClass,
      AMDGPU::S_MOV_B32,  AMDGPU::S_MOV_B32_term, AMDGPU::S_AND_B32,
      AMDGPU::S_OR_B32,   AMDGPU::S_XOR_B32,      AMDGPU::S_ANDN2_B32,
      AMDGPU::S_ORN2_B32, AMDGPU::S_CSELECT_B32,
  };
  static const GCNLaneMaskConstants Wave64 = {
      AMDGPU::EXEC,
      AMDGPU::VCC,
      &AMDGPU::SReg_64RegClass,
      AMDGPU::S_MOV_B64,
      AMDGPU::S_MOV_B64_term,
      AMDGPU::S_AND_B64,
      AMDGPU::S_OR_B64,
      AMDGPU::S_XOR_B64,
      AMDGPU::S_ANDN2_B64,
      AMDGPU::S_ORN2_B64,
      AMDGPU::S_CSELECT_B64,
  };
  assert(WavefrontSize == 32 || WavefrontSize == 64);
  return WavefrontSize == 32 ? &Wave32 : &Wave64;
}

/// Obtain a reference to the global wavefront-size dependent constants
/// based on the wavefront-size of \p function.
const GCNLaneMaskConstants *GCNLaneMaskUtils::getConsts(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  return getConsts(ST.getWavefrontSize());
}

/// Check whether the register could be a lane-mask register.
///
/// It does not distinguish between lane-masks and scalar registers that happen
/// to have the right bitsize.
bool GCNLaneMaskUtils::maybeLaneMask(Register Reg) const {
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  return TII->getRegisterInfo().isSGPRReg(MRI, Reg) &&
         TII->getRegisterInfo().getRegSizeInBits(Reg, MRI) ==
             ST.getWavefrontSize();
}

/// Determine whether the lane-mask register \p Reg is a wave-wide constant.
/// If so, the value is stored in \p Val.
bool GCNLaneMaskUtils::isConstantLaneMask(Register Reg, bool &Val) const {
  MachineRegisterInfo &MRI = MF->getRegInfo();

  const MachineInstr *MI;
  for (;;) {
    MI = MRI.getVRegDef(Reg);
    if (!MI) {
      // This can happen when called from GCNLaneMaskUpdater, where Reg can
      // be a placeholder that has not yet been filled in.
      return false;
    }

    if (MI->getOpcode() == AMDGPU::IMPLICIT_DEF)
      return true;

    if (MI->getOpcode() != AMDGPU::COPY)
      break;

    Reg = MI->getOperand(1).getReg();
    if (!Register::isVirtualRegister(Reg))
      return false;
    if (!maybeLaneMask(Reg))
      return false;
  }

  if (MI->getOpcode() != Constants->OpMov)
    return false;

  if (!MI->getOperand(1).isImm())
    return false;

  int64_t Imm = MI->getOperand(1).getImm();
  if (Imm == 0) {
    Val = false;
    return true;
  }
  if (Imm == -1) {
    Val = true;
    return true;
  }

  return false;
}

/// Create a virtual lanemask register.
Register GCNLaneMaskUtils::createLaneMaskReg() const {
  MachineRegisterInfo &MRI = MF->getRegInfo();
  return MRI.createVirtualRegister(Constants->RegClass);
}

/// Insert the moral equivalent of
///
///    DstReg = (PrevReg & ~EXEC) | (CurReg & EXEC)
///
/// before \p I in basic block \p MBB. Some simplifications are applied on the
/// fly based on constant inputs and analysis via \p LMA, and further
/// simplifications can be requested in "accumulating" mode.
///
/// \param DstReg The virtual register into which the merged mask is written.
/// \param PrevReg The virtual register with the "previous" lane mask value;
///                may be null to indicate an undef value.
/// \param CurReg The virtual register with the "current" lane mask value to
///               be merged into "previous".
/// \param LMA If non-null, used to test whether CurReg may already be a subset
///            of EXEC.
/// \param accumulating Indicates that we should assume PrevReg is already
///                     properly masked, i.e. use PrevReg directly instead of
///                     (PrevReg & ~EXEC), and don't add extra 1-bits to DstReg
///                     beyond (CurReg & EXEC).
void GCNLaneMaskUtils::buildMergeLaneMasks(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator I,
                                           const DebugLoc &DL, Register DstReg,
                                           Register PrevReg, Register CurReg,
                                           GCNLaneMaskAnalysis *LMA,
                                           bool accumulating) const {
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  bool PrevVal = false;
  bool PrevConstant = !PrevReg || isConstantLaneMask(PrevReg, PrevVal);
  bool CurVal = false;
  bool CurConstant = isConstantLaneMask(CurReg, CurVal);

  assert(PrevReg || !accumulating);

  if (PrevConstant && CurConstant) {
    if (PrevVal == CurVal) {
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DstReg).addReg(CurReg);
    } else if (CurVal) {
      // If PrevReg is undef, prefer to propagate a full constant.
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DstReg)
          .addReg(PrevReg ? Constants->RegExec : CurReg);
    } else {
      BuildMI(MBB, I, DL, TII->get(Constants->OpXor), DstReg)
          .addReg(Constants->RegExec)
          .addImm(-1);
    }
    return;
  }

  MachineInstr *PrevMaskedBuilt = nullptr;
  MachineInstr *CurMaskedBuilt = nullptr;
  Register PrevMaskedReg;
  Register CurMaskedReg;
  if (!PrevConstant) {
    if (accumulating || (CurConstant && CurVal)) {
      PrevMaskedReg = PrevReg;
    } else {
      PrevMaskedReg = createLaneMaskReg();
      PrevMaskedBuilt =
          BuildMI(MBB, I, DL, TII->get(Constants->OpAndN2), PrevMaskedReg)
              .addReg(PrevReg)
              .addReg(Constants->RegExec);
    }
  }
  if (!CurConstant) {
    if ((PrevConstant && PrevVal) ||
        (LMA && LMA->isSubsetOfExec(CurReg, MBB))) {
      CurMaskedReg = CurReg;
    } else {
      CurMaskedReg = createLaneMaskReg();
      CurMaskedBuilt =
          BuildMI(MBB, I, DL, TII->get(Constants->OpAnd), CurMaskedReg)
              .addReg(CurReg)
              .addReg(Constants->RegExec);
    }
  }

  // TODO-NOW: reevaluate the masking logic in case of CurConstant && CurVal &&
  // accumulating

  if (PrevConstant && !PrevVal) {
    if (CurMaskedBuilt) {
      CurMaskedBuilt->getOperand(0).setReg(DstReg);
    } else {
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DstReg).addReg(CurMaskedReg);
    }
  } else if (CurConstant && !CurVal) {
    if (PrevMaskedBuilt) {
      PrevMaskedBuilt->getOperand(0).setReg(DstReg);
    } else {
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), DstReg).addReg(PrevMaskedReg);
    }
  } else if (PrevConstant && PrevVal) {
    BuildMI(MBB, I, DL, TII->get(Constants->OpOrN2), DstReg)
        .addReg(CurMaskedReg)
        .addReg(Constants->RegExec);
  } else {
    BuildMI(MBB, I, DL, TII->get(Constants->OpOr), DstReg)
        .addReg(PrevMaskedReg)
        .addReg(CurMaskedReg ? CurMaskedReg : Constants->RegExec);
  }
}

/// Conservatively determine whether the \p Reg is a subset of EXEC for
/// \p UseBlock, i.e. it returns true if it can statically prove that
/// (Reg & EXEC) == Reg when used in \p UseBlock.
bool GCNLaneMaskAnalysis::isSubsetOfExec(Register Reg,
                                         MachineBasicBlock &UseBlock,
                                         unsigned RemainingDepth) {
  MachineRegisterInfo &MRI = LMU.function()->getRegInfo();
  MachineInstr *DefInstr = nullptr;

  for (;;) {
    if (!Register::isVirtualRegister(Reg)) {
      if (Reg == LMU.consts().RegExec &&
          (!DefInstr || DefInstr->getParent() == &UseBlock))
        return true;
      return false;
    }

    DefInstr = MRI.getVRegDef(Reg);
    if (DefInstr->getOpcode() == AMDGPU::COPY) {
      Reg = DefInstr->getOperand(1).getReg();
      continue;
    }

    if (DefInstr->getOpcode() == LMU.consts().OpMov) {
      if (DefInstr->getOperand(1).isImm() &&
          DefInstr->getOperand(1).getImm() == 0)
        return true;
      return false;
    }

    break;
  }

  if (DefInstr->getParent() != &UseBlock)
    return false;

  auto CacheIt = SubsetOfExec.find(Reg);
  if (CacheIt != SubsetOfExec.end())
    return CacheIt->second;

  // V_CMP_xx always return a subset of EXEC.
  if (DefInstr->isCompare() &&
      (SIInstrInfo::isVOPC(*DefInstr) || SIInstrInfo::isVOP3(*DefInstr))) {
    SubsetOfExec[Reg] = true;
    return true;
  }

  if (!RemainingDepth--)
    return false;

  bool LikeOr = DefInstr->getOpcode() == LMU.consts().OpOr ||
                DefInstr->getOpcode() == LMU.consts().OpXor ||
                DefInstr->getOpcode() == LMU.consts().OpCSelect;
  bool IsAnd = DefInstr->getOpcode() == LMU.consts().OpAnd;
  bool IsAndN2 = DefInstr->getOpcode() == LMU.consts().OpAndN2;
  if ((LikeOr || IsAnd || IsAndN2) &&
      (DefInstr->getOperand(1).isReg() && DefInstr->getOperand(2).isReg())) {
    bool FirstIsSubset = isSubsetOfExec(DefInstr->getOperand(1).getReg(),
                                        UseBlock, RemainingDepth);
    if (!FirstIsSubset && (LikeOr || IsAndN2))
      return SubsetOfExec.try_emplace(Reg, false).first->second;

    if (FirstIsSubset && (IsAnd || IsAndN2)) {
      SubsetOfExec[Reg] = true;
      return true;
    }

    bool SecondIsSubset = isSubsetOfExec(DefInstr->getOperand(2).getReg(),
                                         UseBlock, RemainingDepth);
    if (!SecondIsSubset)
      return SubsetOfExec.try_emplace(Reg, false).first->second;

    SubsetOfExec[Reg] = true;
    return true;
  }

  return false;
}

/// Initialize the updater.
void GCNLaneMaskUpdater::init(Register Reg) {
  Processed = false;
  Blocks.clear();
  //SSAUpdater.Initialize(LMU.consts().RegClass);
  SSAUpdater.Initialize(Reg);
}

/// Optional cleanup, may remove stray instructions.
void GCNLaneMaskUpdater::cleanup() {
  Processed = false;
  Blocks.clear();

  MachineRegisterInfo &MRI = LMU.function()->getRegInfo();

  if (ZeroReg && MRI.use_empty(ZeroReg)) {
    MRI.getVRegDef(ZeroReg)->eraseFromParent();
    ZeroReg = {};
  }

  for (MachineInstr *MI : PotentiallyDead) {
    Register DefReg = MI->getOperand(0).getReg();
    if (MRI.use_empty(DefReg))
      MI->eraseFromParent();
  }
  PotentiallyDead.clear();
}

/// Indicate that a reset should occur in the given block.
///
/// Can be called multiple times for the same block, flags accumulate.
void GCNLaneMaskUpdater::addReset(MachineBasicBlock &Block, ResetFlags Flags) {
  assert(!Processed);

  auto BlockIt = findBlockInfo(Block);
  if (BlockIt == Blocks.end()) {
    Blocks.emplace_back(&Block);
    BlockIt = Blocks.end() - 1;
  }

  BlockIt->Flags |= Flags;
}

/// Indicate that a new value is available in \p block. Lane mask bits
/// (per-thread boolean values) are updated.
///
/// \param Value A virtual lane mask register; the lane bits are masked by the
///              block's effective EXEC.
void GCNLaneMaskUpdater::addAvailable(MachineBasicBlock &Block,
                                      Register Value) {
  assert(!Processed);

  auto BlockIt = findBlockInfo(Block);
  if (BlockIt == Blocks.end()) {
    Blocks.emplace_back(&Block);
    BlockIt = Blocks.end() - 1;
  }
  assert(!BlockIt->Value);

  BlockIt->Value = Value;
}

/// Return the value in the middle of the block, i.e. before any change that
/// was registered via \ref addAvailable.
Register GCNLaneMaskUpdater::getValueInMiddleOfBlock(MachineBasicBlock &Block) {
  if (!Processed)
    process();
  return SSAUpdater.GetValueInMiddleOfBlock(&Block);
}

/// Return the value at the end of the given block, i.e. after any change that
/// was registered via \ref addAvailable.
///
/// Note: If \p Block is the reset block in accumulating mode with ResetAtEnd
///       reset mode, then this value will be 0. You likely want
///       \ref getPreReset instead.
Register GCNLaneMaskUpdater::getValueAtEndOfBlock(MachineBasicBlock &Block) {
  if (!Processed)
    process();
  return SSAUpdater.GetValueAtEndOfBlock(&Block);
}

/// Return the value in \p Block after the value merge (if any).
Register GCNLaneMaskUpdater::getValueAfterMerge(MachineBasicBlock &Block) {
  if (!Processed)
    process();

  auto BlockIt = findBlockInfo(Block);
  if (BlockIt != Blocks.end()) {
    if (BlockIt->Merged)
      return BlockIt->Merged;
    if (BlockIt->Flags & ResetInMiddle)
      return ZeroReg;
  }

  // We didn't merge anything in the block, but the block may still be
  // ResetAtEnd, in which case we need the pre-reset value.
  return SSAUpdater.GetValueInMiddleOfBlock(&Block);
}

/// Determine whether \p MI defines and/or uses SCC.
static void instrDefsUsesSCC(const MachineInstr &MI, bool &Def, bool &Use) {
  Def = false;
  Use = false;

  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.getReg() == AMDGPU::SCC) {
      if (MO.isUse())
        Use = true;
      else
        Def = true;
    }
  }
}

/// Return a point at the end of the given \p MBB to insert SALU instructions
/// for lane mask calculation. Take terminators and SCC into account.
static MachineBasicBlock::iterator
getSaluInsertionAtEnd(MachineBasicBlock &MBB) {
  auto InsertionPt = MBB.getFirstTerminator();
  bool TerminatorsUseSCC = false;
  for (auto I = InsertionPt, E = MBB.end(); I != E; ++I) {
    bool DefsSCC;
    instrDefsUsesSCC(*I, DefsSCC, TerminatorsUseSCC);
    if (TerminatorsUseSCC || DefsSCC)
      break;
  }

  if (!TerminatorsUseSCC)
    return InsertionPt;

  while (InsertionPt != MBB.begin()) {
    InsertionPt--;

    bool DefSCC, UseSCC;
    instrDefsUsesSCC(*InsertionPt, DefSCC, UseSCC);
    if (DefSCC)
      return InsertionPt;
  }

  // We should have at least seen an IMPLICIT_DEF or COPY
  llvm_unreachable("SCC used by terminator but no def in block");
}

/// Internal method to insert merge instructions.
void GCNLaneMaskUpdater::process() {
  MachineRegisterInfo &MRI = LMU.function()->getRegInfo();
  const SIInstrInfo *TII =
      LMU.function()->getSubtarget<GCNSubtarget>().getInstrInfo();
  MachineBasicBlock &Entry = LMU.function()->front();

  // Prepare an all-zero value for the default and reset in accumulating mode.
  if (Accumulating && !ZeroReg) {
    ZeroReg = LMU.createLaneMaskReg();
    BuildMI(Entry, Entry.getFirstTerminator(), {}, TII->get(LMU.consts().OpMov),
            ZeroReg)
        .addImm(0);
  }

  // Add available values.
  for (BlockInfo &Info : Blocks) {
    assert(Accumulating || !Info.Flags);
    assert(Info.Flags || Info.Value);

    if (Info.Value)
      Info.Merged = LMU.createLaneMaskReg();

    SSAUpdater.AddAvailableValue(
        Info.Block,
        (Info.Value && !(Info.Flags & ResetAtEnd)) ? Info.Merged : ZeroReg);
  }

  if (Accumulating && !SSAUpdater.HasValueForBlock(&Entry))
    SSAUpdater.AddAvailableValue(&Entry, ZeroReg);

  // Once the SSA updater is ready, we can fill in all merge code, relying
  // on the SSA updater to insert required PHIs.
  for (BlockInfo &Info : Blocks) {
    if (!Info.Value)
      continue;

    // Determine the "previous" value, if any.
    Register Previous;
    if (Info.Block != &LMU.function()->front() &&
        !(Info.Flags & ResetInMiddle)) {
      Previous = SSAUpdater.GetValueInMiddleOfBlock(Info.Block);
      if (Accumulating) {
        assert(!MRI.getVRegDef(Previous) ||
               MRI.getVRegDef(Previous)->getOpcode() != AMDGPU::IMPLICIT_DEF);
      } else {
        MachineInstr *PrevInstr = MRI.getVRegDef(Previous);
        if (PrevInstr && PrevInstr->getOpcode() == AMDGPU::IMPLICIT_DEF) {
          PotentiallyDead.insert(PrevInstr);
          Previous = {};
        }
      }
    } else {
      if (Accumulating)
        Previous = ZeroReg;
    }

    // Insert merge logic.
    MachineBasicBlock::iterator insertPt = getSaluInsertionAtEnd(*Info.Block);
    LMU.buildMergeLaneMasks(*Info.Block, insertPt, {}, Info.Merged, Previous,
                            Info.Value, LMA, Accumulating);

    if (Info.Flags & ResetAtEnd) {
      MachineInstr *mergeInstr = MRI.getVRegDef(Info.Merged);
      if (mergeInstr->getOpcode() == AMDGPU::COPY &&
          mergeInstr->getOperand(1).getReg().isVirtual()) {
        assert(MRI.use_empty(Info.Merged));
        Info.Merged = mergeInstr->getOperand(1).getReg();
        mergeInstr->eraseFromParent();
      }
    }
  }

  Processed = true;
}

/// Find a block in the \ref Blocks structure.
SmallVectorImpl<GCNLaneMaskUpdater::BlockInfo>::iterator
GCNLaneMaskUpdater::findBlockInfo(MachineBasicBlock &Block) {
  return llvm::find_if(
      Blocks, [&](const auto &Entry) { return Entry.Block == &Block; });
}
