//===-- SIRegisterInfo.cpp - SI Register Information ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief SI implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "SIRegisterInfo.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

SIRegisterInfo::SIRegisterInfo() : AMDGPURegisterInfo() {
  unsigned NumRegPressureSets = getNumRegPressureSets();

  SGPR32SetID = NumRegPressureSets;
  VGPR32SetID = NumRegPressureSets;
  for (unsigned i = 0; i < NumRegPressureSets; ++i) {
    if (strncmp("SGPR_32", getRegPressureSetName(i), 7) == 0)
      SGPR32SetID = i;
    else if (strncmp("VGPR_32", getRegPressureSetName(i), 7) == 0)
      VGPR32SetID = i;
  }
  assert(SGPR32SetID < NumRegPressureSets &&
         VGPR32SetID < NumRegPressureSets);
}

void SIRegisterInfo::reserveRegisterTuples(BitVector &Reserved, unsigned Reg) const {
  MCRegAliasIterator R(Reg, this, true);

  for (; R.isValid(); ++R)
    Reserved.set(*R);
}

unsigned SIRegisterInfo::reservedPrivateSegmentBufferReg(
  const MachineFunction &MF) const {
  const AMDGPUSubtarget &ST = MF.getSubtarget<AMDGPUSubtarget>();
  if (ST.hasSGPRInitBug()) {
    // Leave space for flat_scr, xnack_mask, vcc, and alignment
    unsigned BaseIdx = AMDGPUSubtarget::FIXED_SGPR_COUNT_FOR_INIT_BUG - 8 - 4;
    unsigned BaseReg(AMDGPU::SGPR_32RegClass.getRegister(BaseIdx));
    return getMatchingSuperReg(BaseReg, AMDGPU::sub0, &AMDGPU::SReg_128RegClass);
  }

  if (ST.getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS) {
    // 96/97 need to be reserved for flat_scr, 98/99 for xnack_mask, and
    // 100/101 for vcc. This is the next sgpr128 down.
    return AMDGPU::SGPR92_SGPR93_SGPR94_SGPR95;
  }

  return AMDGPU::SGPR96_SGPR97_SGPR98_SGPR99;
}

unsigned SIRegisterInfo::reservedPrivateSegmentWaveByteOffsetReg(
  const MachineFunction &MF) const {
  const AMDGPUSubtarget &ST = MF.getSubtarget<AMDGPUSubtarget>();
  if (ST.hasSGPRInitBug()) {
    unsigned Idx = AMDGPUSubtarget::FIXED_SGPR_COUNT_FOR_INIT_BUG - 6 - 1;
    return AMDGPU::SGPR_32RegClass.getRegister(Idx);
  }

  if (ST.getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS) {
    // Next register before reservations for flat_scr, xnack_mask, vcc,
    // and scratch resource.
    return AMDGPU::SGPR91;
  }

  return AMDGPU::SGPR95;
}

BitVector SIRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(AMDGPU::INDIRECT_BASE_ADDR);

  // EXEC_LO and EXEC_HI could be allocated and used as regular register, but
  // this seems likely to result in bugs, so I'm marking them as reserved.
  reserveRegisterTuples(Reserved, AMDGPU::EXEC);
  reserveRegisterTuples(Reserved, AMDGPU::FLAT_SCR);

  // Reserve the last 2 registers so we will always have at least 2 more that
  // will physically contain VCC.
  reserveRegisterTuples(Reserved, AMDGPU::SGPR102_SGPR103);

  const AMDGPUSubtarget &ST = MF.getSubtarget<AMDGPUSubtarget>();

  if (ST.getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS) {
    // SI/CI have 104 SGPRs. VI has 102. We need to shift down the reservation
    // for VCC/XNACK_MASK/FLAT_SCR.
    //
    // TODO The SGPRs that alias to XNACK_MASK could be used as general purpose
    // SGPRs when the XNACK feature is not used. This is currently not done
    // because the code that counts SGPRs cannot account for such holes.
    reserveRegisterTuples(Reserved, AMDGPU::SGPR96_SGPR97);
    reserveRegisterTuples(Reserved, AMDGPU::SGPR98_SGPR99);
    reserveRegisterTuples(Reserved, AMDGPU::SGPR100_SGPR101);
  }

  // Tonga and Iceland can only allocate a fixed number of SGPRs due
  // to a hw bug.
  if (ST.hasSGPRInitBug()) {
    unsigned NumSGPRs = AMDGPU::SGPR_32RegClass.getNumRegs();
    // Reserve some SGPRs for FLAT_SCRATCH, XNACK_MASK, and VCC (6 SGPRs).
    unsigned Limit = AMDGPUSubtarget::FIXED_SGPR_COUNT_FOR_INIT_BUG - 6;

    for (unsigned i = Limit; i < NumSGPRs; ++i) {
      unsigned Reg = AMDGPU::SGPR_32RegClass.getRegister(i);
      reserveRegisterTuples(Reserved, Reg);
    }
  }

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  unsigned ScratchWaveOffsetReg = MFI->getScratchWaveOffsetReg();
  if (ScratchWaveOffsetReg != AMDGPU::NoRegister) {
    // Reserve 1 SGPR for scratch wave offset in case we need to spill.
    reserveRegisterTuples(Reserved, ScratchWaveOffsetReg);
  }

  unsigned ScratchRSrcReg = MFI->getScratchRSrcReg();
  if (ScratchRSrcReg != AMDGPU::NoRegister) {
    // Reserve 4 SGPRs for the scratch buffer resource descriptor in case we need
    // to spill.
    // TODO: May need to reserve a VGPR if doing LDS spilling.
    reserveRegisterTuples(Reserved, ScratchRSrcReg);
    assert(!isSubRegister(ScratchRSrcReg, ScratchWaveOffsetReg));
  }

  return Reserved;
}

unsigned SIRegisterInfo::getRegPressureSetLimit(const MachineFunction &MF,
                                                unsigned Idx) const {
  const AMDGPUSubtarget &STI = MF.getSubtarget<AMDGPUSubtarget>();
  // FIXME: We should adjust the max number of waves based on LDS size.
  unsigned SGPRLimit = getNumSGPRsAllowed(STI.getGeneration(),
                                          STI.getMaxWavesPerCU());
  unsigned VGPRLimit = getNumVGPRsAllowed(STI.getMaxWavesPerCU());

  unsigned VSLimit = SGPRLimit + VGPRLimit;

  for (regclass_iterator I = regclass_begin(), E = regclass_end();
       I != E; ++I) {
    const TargetRegisterClass *RC = *I;

    unsigned NumSubRegs = std::max((int)RC->getSize() / 4, 1);
    unsigned Limit;

    if (isPseudoRegClass(RC)) {
      // FIXME: This is a hack. We should never be considering the pressure of
      // these since no virtual register should ever have this class.
      Limit = VSLimit;
    } else if (isSGPRClass(RC)) {
      Limit = SGPRLimit / NumSubRegs;
    } else {
      Limit = VGPRLimit / NumSubRegs;
    }

    const int *Sets = getRegClassPressureSets(RC);
    assert(Sets);
    for (unsigned i = 0; Sets[i] != -1; ++i) {
      if (Sets[i] == (int)Idx)
        return Limit;
    }
  }
  return 256;
}

bool SIRegisterInfo::requiresRegisterScavenging(const MachineFunction &Fn) const {
  return Fn.getFrameInfo()->hasStackObjects();
}

static unsigned getNumSubRegsForSpillOp(unsigned Op) {

  switch (Op) {
  case AMDGPU::SI_SPILL_S512_SAVE:
  case AMDGPU::SI_SPILL_S512_RESTORE:
  case AMDGPU::SI_SPILL_V512_SAVE:
  case AMDGPU::SI_SPILL_V512_RESTORE:
    return 16;
  case AMDGPU::SI_SPILL_S256_SAVE:
  case AMDGPU::SI_SPILL_S256_RESTORE:
  case AMDGPU::SI_SPILL_V256_SAVE:
  case AMDGPU::SI_SPILL_V256_RESTORE:
    return 8;
  case AMDGPU::SI_SPILL_S128_SAVE:
  case AMDGPU::SI_SPILL_S128_RESTORE:
  case AMDGPU::SI_SPILL_V128_SAVE:
  case AMDGPU::SI_SPILL_V128_RESTORE:
    return 4;
  case AMDGPU::SI_SPILL_V96_SAVE:
  case AMDGPU::SI_SPILL_V96_RESTORE:
    return 3;
  case AMDGPU::SI_SPILL_S64_SAVE:
  case AMDGPU::SI_SPILL_S64_RESTORE:
  case AMDGPU::SI_SPILL_V64_SAVE:
  case AMDGPU::SI_SPILL_V64_RESTORE:
    return 2;
  case AMDGPU::SI_SPILL_S32_SAVE:
  case AMDGPU::SI_SPILL_S32_RESTORE:
  case AMDGPU::SI_SPILL_V32_SAVE:
  case AMDGPU::SI_SPILL_V32_RESTORE:
    return 1;
  default: llvm_unreachable("Invalid spill opcode");
  }
}

void SIRegisterInfo::buildScratchLoadStore(MachineBasicBlock::iterator MI,
                                           unsigned LoadStoreOp,
                                           unsigned Value,
                                           unsigned ScratchRsrcReg,
                                           unsigned ScratchOffset,
                                           int64_t Offset,
                                           RegScavenger *RS) const {

  MachineBasicBlock *MBB = MI->getParent();
  const MachineFunction *MF = MI->getParent()->getParent();
  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo *>(MF->getSubtarget().getInstrInfo());
  LLVMContext &Ctx = MF->getFunction()->getContext();
  DebugLoc DL = MI->getDebugLoc();
  bool IsLoad = TII->get(LoadStoreOp).mayLoad();

  bool RanOutOfSGPRs = false;
  bool Scavenged = false;
  unsigned SOffset = ScratchOffset;

  unsigned NumSubRegs = getNumSubRegsForSpillOp(MI->getOpcode());
  unsigned Size = NumSubRegs * 4;

  if (!isUInt<12>(Offset + Size)) {
    SOffset = RS->scavengeRegister(&AMDGPU::SGPR_32RegClass, MI, 0);
    if (SOffset == AMDGPU::NoRegister) {
      RanOutOfSGPRs = true;
      SOffset = AMDGPU::SGPR0;
    } else {
      Scavenged = true;
    }
    BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_ADD_U32), SOffset)
            .addReg(ScratchOffset)
            .addImm(Offset);
    Offset = 0;
  }

  if (RanOutOfSGPRs)
    Ctx.emitError("Ran out of SGPRs for spilling VGPRS");

  for (unsigned i = 0, e = NumSubRegs; i != e; ++i, Offset += 4) {
    unsigned SubReg = NumSubRegs > 1 ?
        getPhysRegSubReg(Value, &AMDGPU::VGPR_32RegClass, i) :
        Value;

    unsigned SOffsetRegState = 0;
    if (i + 1 == e && Scavenged)
      SOffsetRegState |= RegState::Kill;

    BuildMI(*MBB, MI, DL, TII->get(LoadStoreOp))
      .addReg(SubReg, getDefRegState(IsLoad))
      .addReg(ScratchRsrcReg)
      .addReg(SOffset, SOffsetRegState)
      .addImm(Offset)
      .addImm(0) // glc
      .addImm(0) // slc
      .addImm(0) // tfe
      .addReg(Value, RegState::Implicit | getDefRegState(IsLoad))
      .setMemRefs(MI->memoperands_begin(), MI->memoperands_end());
  }
}

void SIRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                        int SPAdj, unsigned FIOperandNum,
                                        RegScavenger *RS) const {
  MachineFunction *MF = MI->getParent()->getParent();
  MachineBasicBlock *MBB = MI->getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  MachineFrameInfo *FrameInfo = MF->getFrameInfo();
  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo *>(MF->getSubtarget().getInstrInfo());
  DebugLoc DL = MI->getDebugLoc();

  MachineOperand &FIOp = MI->getOperand(FIOperandNum);
  int Index = MI->getOperand(FIOperandNum).getIndex();

  switch (MI->getOpcode()) {
    // SGPR register spill
    case AMDGPU::SI_SPILL_S512_SAVE:
    case AMDGPU::SI_SPILL_S256_SAVE:
    case AMDGPU::SI_SPILL_S128_SAVE:
    case AMDGPU::SI_SPILL_S64_SAVE:
    case AMDGPU::SI_SPILL_S32_SAVE: {
      unsigned NumSubRegs = getNumSubRegsForSpillOp(MI->getOpcode());

      for (unsigned i = 0, e = NumSubRegs; i < e; ++i) {
        unsigned SubReg = getPhysRegSubReg(MI->getOperand(0).getReg(),
                                           &AMDGPU::SGPR_32RegClass, i);
        struct SIMachineFunctionInfo::SpilledReg Spill =
            MFI->getSpilledReg(MF, Index, i);

        BuildMI(*MBB, MI, DL,
                TII->getMCOpcodeFromPseudo(AMDGPU::V_WRITELANE_B32),
                Spill.VGPR)
                .addReg(SubReg)
                .addImm(Spill.Lane);

        // FIXME: Since this spills to another register instead of an actual
        // frame index, we should delete the frame index when all references to
        // it are fixed.
      }
      MI->eraseFromParent();
      break;
    }

    // SGPR register restore
    case AMDGPU::SI_SPILL_S512_RESTORE:
    case AMDGPU::SI_SPILL_S256_RESTORE:
    case AMDGPU::SI_SPILL_S128_RESTORE:
    case AMDGPU::SI_SPILL_S64_RESTORE:
    case AMDGPU::SI_SPILL_S32_RESTORE: {
      unsigned NumSubRegs = getNumSubRegsForSpillOp(MI->getOpcode());

      for (unsigned i = 0, e = NumSubRegs; i < e; ++i) {
        unsigned SubReg = getPhysRegSubReg(MI->getOperand(0).getReg(),
                                           &AMDGPU::SGPR_32RegClass, i);
        struct SIMachineFunctionInfo::SpilledReg Spill =
            MFI->getSpilledReg(MF, Index, i);

        BuildMI(*MBB, MI, DL,
                TII->getMCOpcodeFromPseudo(AMDGPU::V_READLANE_B32),
                SubReg)
                .addReg(Spill.VGPR)
                .addImm(Spill.Lane)
                .addReg(MI->getOperand(0).getReg(), RegState::ImplicitDefine);
      }

      // TODO: only do this when it is needed
      switch (MF->getSubtarget<AMDGPUSubtarget>().getGeneration()) {
      case AMDGPUSubtarget::SOUTHERN_ISLANDS:
        // "VALU writes SGPR" -> "SMRD reads that SGPR" needs 4 wait states
        // ("S_NOP 3") on SI
        TII->insertWaitStates(MI, 4);
        break;
      case AMDGPUSubtarget::SEA_ISLANDS:
        break;
      default: // VOLCANIC_ISLANDS and later
        // "VALU writes SGPR -> VMEM reads that SGPR" needs 5 wait states
        // ("S_NOP 4") on VI and later. This also applies to VALUs which write
        // VCC, but we're unlikely to see VMEM use VCC.
        TII->insertWaitStates(MI, 5);
      }

      MI->eraseFromParent();
      break;
    }

    // VGPR register spill
    case AMDGPU::SI_SPILL_V512_SAVE:
    case AMDGPU::SI_SPILL_V256_SAVE:
    case AMDGPU::SI_SPILL_V128_SAVE:
    case AMDGPU::SI_SPILL_V96_SAVE:
    case AMDGPU::SI_SPILL_V64_SAVE:
    case AMDGPU::SI_SPILL_V32_SAVE:
      buildScratchLoadStore(MI, AMDGPU::BUFFER_STORE_DWORD_OFFSET,
            TII->getNamedOperand(*MI, AMDGPU::OpName::src)->getReg(),
            TII->getNamedOperand(*MI, AMDGPU::OpName::scratch_rsrc)->getReg(),
            TII->getNamedOperand(*MI, AMDGPU::OpName::scratch_offset)->getReg(),
             FrameInfo->getObjectOffset(Index), RS);
      MI->eraseFromParent();
      break;
    case AMDGPU::SI_SPILL_V32_RESTORE:
    case AMDGPU::SI_SPILL_V64_RESTORE:
    case AMDGPU::SI_SPILL_V96_RESTORE:
    case AMDGPU::SI_SPILL_V128_RESTORE:
    case AMDGPU::SI_SPILL_V256_RESTORE:
    case AMDGPU::SI_SPILL_V512_RESTORE: {
      buildScratchLoadStore(MI, AMDGPU::BUFFER_LOAD_DWORD_OFFSET,
            TII->getNamedOperand(*MI, AMDGPU::OpName::dst)->getReg(),
            TII->getNamedOperand(*MI, AMDGPU::OpName::scratch_rsrc)->getReg(),
            TII->getNamedOperand(*MI, AMDGPU::OpName::scratch_offset)->getReg(),
            FrameInfo->getObjectOffset(Index), RS);
      MI->eraseFromParent();
      break;
    }

    default: {
      int64_t Offset = FrameInfo->getObjectOffset(Index);
      FIOp.ChangeToImmediate(Offset);
      if (!TII->isImmOperandLegal(MI, FIOperandNum, FIOp)) {
        unsigned TmpReg = RS->scavengeRegister(&AMDGPU::VGPR_32RegClass, MI, SPAdj);
        BuildMI(*MBB, MI, MI->getDebugLoc(),
                TII->get(AMDGPU::V_MOV_B32_e32), TmpReg)
                .addImm(Offset);
        FIOp.ChangeToRegister(TmpReg, false, false, true);
      }
    }
  }
}

unsigned SIRegisterInfo::getHWRegIndex(unsigned Reg) const {
  return getEncodingValue(Reg) & 0xff;
}

// FIXME: This is very slow. It might be worth creating a map from physreg to
// register class.
const TargetRegisterClass *SIRegisterInfo::getPhysRegClass(unsigned Reg) const {
  assert(!TargetRegisterInfo::isVirtualRegister(Reg));

  static const TargetRegisterClass *const BaseClasses[] = {
    &AMDGPU::VGPR_32RegClass,
    &AMDGPU::SReg_32RegClass,
    &AMDGPU::VReg_64RegClass,
    &AMDGPU::SReg_64RegClass,
    &AMDGPU::VReg_96RegClass,
    &AMDGPU::VReg_128RegClass,
    &AMDGPU::SReg_128RegClass,
    &AMDGPU::VReg_256RegClass,
    &AMDGPU::SReg_256RegClass,
    &AMDGPU::VReg_512RegClass,
    &AMDGPU::SReg_512RegClass,
    &AMDGPU::SCC_CLASSRegClass,
  };

  for (const TargetRegisterClass *BaseClass : BaseClasses) {
    if (BaseClass->contains(Reg)) {
      return BaseClass;
    }
  }
  return nullptr;
}

// TODO: It might be helpful to have some target specific flags in
// TargetRegisterClass to mark which classes are VGPRs to make this trivial.
bool SIRegisterInfo::hasVGPRs(const TargetRegisterClass *RC) const {
  switch (RC->getSize()) {
  case 0: return false;
  case 1: return false;
  case 4:
    return getCommonSubClass(&AMDGPU::VGPR_32RegClass, RC) != nullptr;
  case 8:
    return getCommonSubClass(&AMDGPU::VReg_64RegClass, RC) != nullptr;
  case 12:
    return getCommonSubClass(&AMDGPU::VReg_96RegClass, RC) != nullptr;
  case 16:
    return getCommonSubClass(&AMDGPU::VReg_128RegClass, RC) != nullptr;
  case 32:
    return getCommonSubClass(&AMDGPU::VReg_256RegClass, RC) != nullptr;
  case 64:
    return getCommonSubClass(&AMDGPU::VReg_512RegClass, RC) != nullptr;
  default:
    llvm_unreachable("Invalid register class size");
  }
}

const TargetRegisterClass *SIRegisterInfo::getEquivalentVGPRClass(
                                         const TargetRegisterClass *SRC) const {
  switch (SRC->getSize()) {
  case 4:
    return &AMDGPU::VGPR_32RegClass;
  case 8:
    return &AMDGPU::VReg_64RegClass;
  case 12:
    return &AMDGPU::VReg_96RegClass;
  case 16:
    return &AMDGPU::VReg_128RegClass;
  case 32:
    return &AMDGPU::VReg_256RegClass;
  case 64:
    return &AMDGPU::VReg_512RegClass;
  default:
    llvm_unreachable("Invalid register class size");
  }
}

const TargetRegisterClass *SIRegisterInfo::getEquivalentSGPRClass(
                                         const TargetRegisterClass *VRC) const {
  switch (VRC->getSize()) {
  case 4:
    return &AMDGPU::SGPR_32RegClass;
  case 8:
    return &AMDGPU::SReg_64RegClass;
  case 16:
    return &AMDGPU::SReg_128RegClass;
  case 32:
    return &AMDGPU::SReg_256RegClass;
  case 64:
    return &AMDGPU::SReg_512RegClass;
  default:
    llvm_unreachable("Invalid register class size");
  }
}

const TargetRegisterClass *SIRegisterInfo::getSubRegClass(
                         const TargetRegisterClass *RC, unsigned SubIdx) const {
  if (SubIdx == AMDGPU::NoSubRegister)
    return RC;

  // We can assume that each lane corresponds to one 32-bit register.
  unsigned Count = countPopulation(getSubRegIndexLaneMask(SubIdx));
  if (isSGPRClass(RC)) {
    switch (Count) {
    case 1:
      return &AMDGPU::SGPR_32RegClass;
    case 2:
      return &AMDGPU::SReg_64RegClass;
    case 4:
      return &AMDGPU::SReg_128RegClass;
    case 8:
      return &AMDGPU::SReg_256RegClass;
    case 16: /* fall-through */
    default:
      llvm_unreachable("Invalid sub-register class size");
    }
  } else {
    switch (Count) {
    case 1:
      return &AMDGPU::VGPR_32RegClass;
    case 2:
      return &AMDGPU::VReg_64RegClass;
    case 3:
      return &AMDGPU::VReg_96RegClass;
    case 4:
      return &AMDGPU::VReg_128RegClass;
    case 8:
      return &AMDGPU::VReg_256RegClass;
    case 16: /* fall-through */
    default:
      llvm_unreachable("Invalid sub-register class size");
    }
  }
}

bool SIRegisterInfo::shouldRewriteCopySrc(
  const TargetRegisterClass *DefRC,
  unsigned DefSubReg,
  const TargetRegisterClass *SrcRC,
  unsigned SrcSubReg) const {
  // We want to prefer the smallest register class possible, so we don't want to
  // stop and rewrite on anything that looks like a subregister
  // extract. Operations mostly don't care about the super register class, so we
  // only want to stop on the most basic of copies between the smae register
  // class.
  //
  // e.g. if we have something like
  // vreg0 = ...
  // vreg1 = ...
  // vreg2 = REG_SEQUENCE vreg0, sub0, vreg1, sub1, vreg2, sub2
  // vreg3 = COPY vreg2, sub0
  //
  // We want to look through the COPY to find:
  //  => vreg3 = COPY vreg0

  // Plain copy.
  return getCommonSubClass(DefRC, SrcRC) != nullptr;
}

unsigned SIRegisterInfo::getPhysRegSubReg(unsigned Reg,
                                          const TargetRegisterClass *SubRC,
                                          unsigned Channel) const {

  switch (Reg) {
    case AMDGPU::VCC:
      switch(Channel) {
        case 0: return AMDGPU::VCC_LO;
        case 1: return AMDGPU::VCC_HI;
        default: llvm_unreachable("Invalid SubIdx for VCC");
      }

  case AMDGPU::FLAT_SCR:
    switch (Channel) {
    case 0:
      return AMDGPU::FLAT_SCR_LO;
    case 1:
      return AMDGPU::FLAT_SCR_HI;
    default:
      llvm_unreachable("Invalid SubIdx for FLAT_SCR");
    }
    break;

  case AMDGPU::EXEC:
    switch (Channel) {
    case 0:
      return AMDGPU::EXEC_LO;
    case 1:
      return AMDGPU::EXEC_HI;
    default:
      llvm_unreachable("Invalid SubIdx for EXEC");
    }
    break;
  }

  const TargetRegisterClass *RC = getPhysRegClass(Reg);
  // 32-bit registers don't have sub-registers, so we can just return the
  // Reg.  We need to have this check here, because the calculation below
  // using getHWRegIndex() will fail with special 32-bit registers like
  // VCC_LO, VCC_HI, EXEC_LO, EXEC_HI and M0.
  if (RC->getSize() == 4) {
    assert(Channel == 0);
    return Reg;
  }

  unsigned Index = getHWRegIndex(Reg);
  return SubRC->getRegister(Index + Channel);
}

bool SIRegisterInfo::opCanUseLiteralConstant(unsigned OpType) const {
  return OpType == AMDGPU::OPERAND_REG_IMM32;
}

bool SIRegisterInfo::opCanUseInlineConstant(unsigned OpType) const {
  if (opCanUseLiteralConstant(OpType))
    return true;

  return OpType == AMDGPU::OPERAND_REG_INLINE_C;
}

// FIXME: Most of these are flexible with HSA and we don't need to reserve them
// as input registers if unused. Whether the dispatch ptr is necessary should be
// easy to detect from used intrinsics. Scratch setup is harder to know.
unsigned SIRegisterInfo::getPreloadedValue(const MachineFunction &MF,
                                           enum PreloadedValue Value) const {

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const AMDGPUSubtarget &ST = MF.getSubtarget<AMDGPUSubtarget>();
  (void)ST;
  switch (Value) {
  case SIRegisterInfo::WORKGROUP_ID_X:
    assert(MFI->hasWorkGroupIDX());
    return MFI->WorkGroupIDXSystemSGPR;
  case SIRegisterInfo::WORKGROUP_ID_Y:
    assert(MFI->hasWorkGroupIDY());
    return MFI->WorkGroupIDYSystemSGPR;
  case SIRegisterInfo::WORKGROUP_ID_Z:
    assert(MFI->hasWorkGroupIDZ());
    return MFI->WorkGroupIDZSystemSGPR;
  case SIRegisterInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET:
    return MFI->PrivateSegmentWaveByteOffsetSystemSGPR;
  case SIRegisterInfo::PRIVATE_SEGMENT_BUFFER:
    assert(ST.isAmdHsaOS() && "Non-HSA ABI currently uses relocations");
    assert(MFI->hasPrivateSegmentBuffer());
    return MFI->PrivateSegmentBufferUserSGPR;
  case SIRegisterInfo::KERNARG_SEGMENT_PTR:
    assert(MFI->hasKernargSegmentPtr());
    return MFI->KernargSegmentPtrUserSGPR;
  case SIRegisterInfo::DISPATCH_ID:
    llvm_unreachable("unimplemented");
  case SIRegisterInfo::FLAT_SCRATCH_INIT:
    assert(MFI->hasFlatScratchInit());
    return MFI->FlatScratchInitUserSGPR;
  case SIRegisterInfo::DISPATCH_PTR:
    assert(MFI->hasDispatchPtr());
    return MFI->DispatchPtrUserSGPR;
  case SIRegisterInfo::QUEUE_PTR:
    llvm_unreachable("not implemented");
  case SIRegisterInfo::WORKITEM_ID_X:
    assert(MFI->hasWorkItemIDX());
    return AMDGPU::VGPR0;
  case SIRegisterInfo::WORKITEM_ID_Y:
    assert(MFI->hasWorkItemIDY());
    return AMDGPU::VGPR1;
  case SIRegisterInfo::WORKITEM_ID_Z:
    assert(MFI->hasWorkItemIDZ());
    return AMDGPU::VGPR2;
  }
  llvm_unreachable("unexpected preloaded value type");
}

/// \brief Returns a register that is not used at any point in the function.
///        If all registers are used, then this function will return
//         AMDGPU::NoRegister.
unsigned SIRegisterInfo::findUnusedRegister(const MachineRegisterInfo &MRI,
                                           const TargetRegisterClass *RC) const {
  for (unsigned Reg : *RC)
    if (!MRI.isPhysRegUsed(Reg))
      return Reg;
  return AMDGPU::NoRegister;
}

unsigned SIRegisterInfo::getNumVGPRsAllowed(unsigned WaveCount) const {
  switch(WaveCount) {
    case 10: return 24;
    case 9:  return 28;
    case 8:  return 32;
    case 7:  return 36;
    case 6:  return 40;
    case 5:  return 48;
    case 4:  return 64;
    case 3:  return 84;
    case 2:  return 128;
    default: return 256;
  }
}

unsigned SIRegisterInfo::getNumSGPRsAllowed(AMDGPUSubtarget::Generation gen,
                                            unsigned WaveCount) const {
  if (gen >= AMDGPUSubtarget::VOLCANIC_ISLANDS) {
    switch (WaveCount) {
      case 10: return 80;
      case 9:  return 80;
      case 8:  return 96;
      default: return 102;
    }
  } else {
    switch(WaveCount) {
      case 10: return 48;
      case 9:  return 56;
      case 8:  return 64;
      case 7:  return 72;
      case 6:  return 80;
      case 5:  return 96;
      default: return 103;
    }
  }
}
