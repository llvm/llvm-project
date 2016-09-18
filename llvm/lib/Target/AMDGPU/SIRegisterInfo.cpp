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
#include "AMDGPUSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

static bool hasPressureSet(const int *PSets, unsigned PSetID) {
  for (unsigned i = 0; PSets[i] != -1; ++i) {
    if (PSets[i] == (int)PSetID)
      return true;
  }
  return false;
}

void SIRegisterInfo::classifyPressureSet(unsigned PSetID, unsigned Reg,
                                         BitVector &PressureSets) const {
  for (MCRegUnitIterator U(Reg, this); U.isValid(); ++U) {
    const int *PSets = getRegUnitPressureSets(*U);
    if (hasPressureSet(PSets, PSetID)) {
      PressureSets.set(PSetID);
      break;
    }
  }
}

SIRegisterInfo::SIRegisterInfo() : AMDGPURegisterInfo(),
                                   SGPRPressureSets(getNumRegPressureSets()),
                                   VGPRPressureSets(getNumRegPressureSets()) {
  unsigned NumRegPressureSets = getNumRegPressureSets();

  SGPRSetID = NumRegPressureSets;
  VGPRSetID = NumRegPressureSets;

  for (unsigned i = 0; i < NumRegPressureSets; ++i) {
    classifyPressureSet(i, AMDGPU::SGPR0, SGPRPressureSets);
    classifyPressureSet(i, AMDGPU::VGPR0, VGPRPressureSets);
  }

  // Determine the number of reg units for each pressure set.
  std::vector<unsigned> PressureSetRegUnits(NumRegPressureSets, 0);
  for (unsigned i = 0, e = getNumRegUnits(); i != e; ++i) {
    const int *PSets = getRegUnitPressureSets(i);
    for (unsigned j = 0; PSets[j] != -1; ++j) {
      ++PressureSetRegUnits[PSets[j]];
    }
  }

  unsigned VGPRMax = 0, SGPRMax = 0;
  for (unsigned i = 0; i < NumRegPressureSets; ++i) {
    if (isVGPRPressureSet(i) && PressureSetRegUnits[i] > VGPRMax) {
      VGPRSetID = i;
      VGPRMax = PressureSetRegUnits[i];
      continue;
    }
    if (isSGPRPressureSet(i) && PressureSetRegUnits[i] > SGPRMax) {
      SGPRSetID = i;
      SGPRMax = PressureSetRegUnits[i];
    }
  }

  assert(SGPRSetID < NumRegPressureSets &&
         VGPRSetID < NumRegPressureSets);
}

void SIRegisterInfo::reserveRegisterTuples(BitVector &Reserved, unsigned Reg) const {
  MCRegAliasIterator R(Reg, this, true);

  for (; R.isValid(); ++R)
    Reserved.set(*R);
}

unsigned SIRegisterInfo::reservedPrivateSegmentBufferReg(
  const MachineFunction &MF) const {
  unsigned BaseIdx = alignDown(getMaxNumSGPRs(MF), 4) - 4;
  unsigned BaseReg(AMDGPU::SGPR_32RegClass.getRegister(BaseIdx));
  return getMatchingSuperReg(BaseReg, AMDGPU::sub0, &AMDGPU::SReg_128RegClass);
}

unsigned SIRegisterInfo::reservedPrivateSegmentWaveByteOffsetReg(
  const MachineFunction &MF) const {
  unsigned RegCount = getMaxNumSGPRs(MF);
  unsigned Reg;

  // Try to place it in a hole after PrivateSegmentbufferReg.
  if (RegCount & 3) {
    // We cannot put the segment buffer in (Idx - 4) ... (Idx - 1) due to
    // alignment constraints, so we have a hole where can put the wave offset.
    Reg = RegCount - 1;
  } else {
    // We can put the segment buffer in (Idx - 4) ... (Idx - 1) and put the
    // wave offset before it.
    Reg = RegCount - 5;
  }
  return AMDGPU::SGPR_32RegClass.getRegister(Reg);
}

BitVector SIRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(AMDGPU::INDIRECT_BASE_ADDR);

  // EXEC_LO and EXEC_HI could be allocated and used as regular register, but
  // this seems likely to result in bugs, so I'm marking them as reserved.
  reserveRegisterTuples(Reserved, AMDGPU::EXEC);
  reserveRegisterTuples(Reserved, AMDGPU::FLAT_SCR);

  // Reserve Trap Handler registers - support is not implemented in Codegen.
  reserveRegisterTuples(Reserved, AMDGPU::TBA);
  reserveRegisterTuples(Reserved, AMDGPU::TMA);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP0_TTMP1);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP2_TTMP3);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP4_TTMP5);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP6_TTMP7);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP8_TTMP9);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP10_TTMP11);

  unsigned MaxNumSGPRs = getMaxNumSGPRs(MF);
  unsigned TotalNumSGPRs = AMDGPU::SGPR_32RegClass.getNumRegs();
  for (unsigned i = MaxNumSGPRs; i < TotalNumSGPRs; ++i) {
    unsigned Reg = AMDGPU::SGPR_32RegClass.getRegister(i);
    reserveRegisterTuples(Reserved, Reg);
  }

  unsigned MaxNumVGPRs = getMaxNumVGPRs(MF);
  unsigned TotalNumVGPRs = AMDGPU::VGPR_32RegClass.getNumRegs();
  for (unsigned i = MaxNumVGPRs; i < TotalNumVGPRs; ++i) {
    unsigned Reg = AMDGPU::VGPR_32RegClass.getRegister(i);
    reserveRegisterTuples(Reserved, Reg);
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

bool SIRegisterInfo::requiresRegisterScavenging(const MachineFunction &Fn) const {
  return Fn.getFrameInfo().hasStackObjects();
}

bool
SIRegisterInfo::requiresFrameIndexScavenging(const MachineFunction &MF) const {
  return MF.getFrameInfo().hasStackObjects();
}

bool SIRegisterInfo::requiresVirtualBaseRegisters(
  const MachineFunction &) const {
  // There are no special dedicated stack or frame pointers.
  return true;
}

bool SIRegisterInfo::trackLivenessAfterRegAlloc(const MachineFunction &MF) const {
  // This helps catch bugs as verifier errors.
  return true;
}

int64_t SIRegisterInfo::getFrameIndexInstrOffset(const MachineInstr *MI,
                                                 int Idx) const {
  if (!SIInstrInfo::isMUBUF(*MI))
    return 0;

  assert(Idx == AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                           AMDGPU::OpName::vaddr) &&
         "Should never see frame index on non-address operand");

  int OffIdx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                          AMDGPU::OpName::offset);
  return MI->getOperand(OffIdx).getImm();
}

bool SIRegisterInfo::needsFrameBaseReg(MachineInstr *MI, int64_t Offset) const {
  return MI->mayLoadOrStore();
}

void SIRegisterInfo::materializeFrameBaseRegister(MachineBasicBlock *MBB,
                                                  unsigned BaseReg,
                                                  int FrameIdx,
                                                  int64_t Offset) const {
  MachineBasicBlock::iterator Ins = MBB->begin();
  DebugLoc DL; // Defaults to "unknown"

  if (Ins != MBB->end())
    DL = Ins->getDebugLoc();

  MachineFunction *MF = MBB->getParent();
  const SISubtarget &Subtarget = MF->getSubtarget<SISubtarget>();
  const SIInstrInfo *TII = Subtarget.getInstrInfo();

  if (Offset == 0) {
    BuildMI(*MBB, Ins, DL, TII->get(AMDGPU::V_MOV_B32_e32), BaseReg)
      .addFrameIndex(FrameIdx);
    return;
  }

  MachineRegisterInfo &MRI = MF->getRegInfo();
  unsigned UnusedCarry = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);
  unsigned OffsetReg = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);

  BuildMI(*MBB, Ins, DL, TII->get(AMDGPU::S_MOV_B32), OffsetReg)
    .addImm(Offset);
  BuildMI(*MBB, Ins, DL, TII->get(AMDGPU::V_ADD_I32_e64), BaseReg)
    .addReg(UnusedCarry, RegState::Define | RegState::Dead)
    .addReg(OffsetReg, RegState::Kill)
    .addFrameIndex(FrameIdx);
}

void SIRegisterInfo::resolveFrameIndex(MachineInstr &MI, unsigned BaseReg,
                                       int64_t Offset) const {

  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction *MF = MBB->getParent();
  const SISubtarget &Subtarget = MF->getSubtarget<SISubtarget>();
  const SIInstrInfo *TII = Subtarget.getInstrInfo();

#ifndef NDEBUG
  // FIXME: Is it possible to be storing a frame index to itself?
  bool SeenFI = false;
  for (const MachineOperand &MO: MI.operands()) {
    if (MO.isFI()) {
      if (SeenFI)
        llvm_unreachable("should not see multiple frame indices");

      SeenFI = true;
    }
  }
#endif

  MachineOperand *FIOp = TII->getNamedOperand(MI, AMDGPU::OpName::vaddr);
  assert(FIOp && FIOp->isFI() && "frame index must be address operand");

  assert(TII->isMUBUF(MI));

  MachineOperand *OffsetOp = TII->getNamedOperand(MI, AMDGPU::OpName::offset);
  int64_t NewOffset = OffsetOp->getImm() + Offset;
  assert(isUInt<12>(NewOffset) && "offset should be legal");

  FIOp->ChangeToRegister(BaseReg, false);
  OffsetOp->setImm(NewOffset);
}

bool SIRegisterInfo::isFrameOffsetLegal(const MachineInstr *MI,
                                        unsigned BaseReg,
                                        int64_t Offset) const {
  return SIInstrInfo::isMUBUF(*MI) && isUInt<12>(Offset);
}

const TargetRegisterClass *SIRegisterInfo::getPointerRegClass(
  const MachineFunction &MF, unsigned Kind) const {
  // This is inaccurate. It depends on the instruction and address space. The
  // only place where we should hit this is for dealing with frame indexes /
  // private accesses, so this is correct in that case.
  return &AMDGPU::VGPR_32RegClass;
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
                                           const MachineOperand *SrcDst,
                                           unsigned ScratchRsrcReg,
                                           unsigned ScratchOffset,
                                           int64_t Offset,
                                           RegScavenger *RS) const {

  unsigned Value = SrcDst->getReg();
  bool IsKill = SrcDst->isKill();
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction *MF = MI->getParent()->getParent();
  const SISubtarget &ST =  MF->getSubtarget<SISubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();

  DebugLoc DL = MI->getDebugLoc();
  bool IsStore = MI->mayStore();

  bool RanOutOfSGPRs = false;
  bool Scavenged = false;
  unsigned SOffset = ScratchOffset;
  unsigned OriginalImmOffset = Offset;

  unsigned NumSubRegs = getNumSubRegsForSpillOp(MI->getOpcode());
  unsigned Size = NumSubRegs * 4;

  if (!isUInt<12>(Offset + Size)) {
    SOffset = AMDGPU::NoRegister;

    // We don't have access to the register scavenger if this function is called
    // during  PEI::scavengeFrameVirtualRegs().
    if (RS)
      SOffset = RS->FindUnusedReg(&AMDGPU::SGPR_32RegClass);

    if (SOffset == AMDGPU::NoRegister) {
      // There are no free SGPRs, and since we are in the process of spilling
      // VGPRs too.  Since we need a VGPR in order to spill SGPRs (this is true
      // on SI/CI and on VI it is true until we implement spilling using scalar
      // stores), we have no way to free up an SGPR.  Our solution here is to
      // add the offset directly to the ScratchOffset register, and then
      // subtract the offset after the spill to return ScratchOffset to it's
      // original value.
      RanOutOfSGPRs = true;
      SOffset = ScratchOffset;
    } else {
      Scavenged = true;
    }
    BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_ADD_U32), SOffset)
            .addReg(ScratchOffset)
            .addImm(Offset);
    Offset = 0;
  }

  for (unsigned i = 0, e = NumSubRegs; i != e; ++i, Offset += 4) {
    unsigned SubReg = NumSubRegs == 1 ?
      Value : getSubReg(Value, getSubRegFromChannel(i));

    unsigned SOffsetRegState = 0;
    unsigned SrcDstRegState = getDefRegState(!IsStore);
    if (i + 1 == e) {
      SOffsetRegState |= getKillRegState(Scavenged);
      // The last implicit use carries the "Kill" flag.
      SrcDstRegState |= getKillRegState(IsKill);
    }

    BuildMI(*MBB, MI, DL, TII->get(LoadStoreOp))
      .addReg(SubReg, getDefRegState(!IsStore))
      .addReg(ScratchRsrcReg)
      .addReg(SOffset, SOffsetRegState)
      .addImm(Offset)
      .addImm(0) // glc
      .addImm(0) // slc
      .addImm(0) // tfe
      .addReg(Value, RegState::Implicit | SrcDstRegState)
      .setMemRefs(MI->memoperands_begin(), MI->memoperands_end());
  }
  if (RanOutOfSGPRs) {
    // Subtract the offset we added to the ScratchOffset register.
    BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_SUB_U32), ScratchOffset)
            .addReg(ScratchOffset)
            .addImm(OriginalImmOffset);
  }
}

void SIRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                        int SPAdj, unsigned FIOperandNum,
                                        RegScavenger *RS) const {
  MachineFunction *MF = MI->getParent()->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  MachineBasicBlock *MBB = MI->getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  MachineFrameInfo &FrameInfo = MF->getFrameInfo();
  const SISubtarget &ST =  MF->getSubtarget<SISubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
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
      unsigned TmpReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
      unsigned SuperReg = MI->getOperand(0).getReg();
      bool IsKill = MI->getOperand(0).isKill();

      // SubReg carries the "Kill" flag when SubReg == SuperReg.
      unsigned SubKillState = getKillRegState((NumSubRegs == 1) && IsKill);
      for (unsigned i = 0, e = NumSubRegs; i < e; ++i) {
        unsigned SubReg = NumSubRegs == 1 ?
          SuperReg : getSubReg(SuperReg, getSubRegFromChannel(i));

        struct SIMachineFunctionInfo::SpilledReg Spill =
            MFI->getSpilledReg(MF, Index, i);
        if (Spill.hasReg()) {
          if (SuperReg == AMDGPU::M0) {
            assert(NumSubRegs == 1);
            unsigned CopyM0
              = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
            BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_MOV_B32), CopyM0)
              .addReg(SuperReg, getKillRegState(IsKill));

            // The real spill now kills the temp copy.
            SubReg = SuperReg = CopyM0;
            IsKill = true;
          }

          BuildMI(*MBB, MI, DL,
                  TII->getMCOpcodeFromPseudo(AMDGPU::V_WRITELANE_B32),
                  Spill.VGPR)
                  .addReg(SubReg, getKillRegState(IsKill))
                  .addImm(Spill.Lane);

          // FIXME: Since this spills to another register instead of an actual
          // frame index, we should delete the frame index when all references to
          // it are fixed.
        } else {
          // Spill SGPR to a frame index.
          // FIXME we should use S_STORE_DWORD here for VI.
          MachineInstrBuilder Mov
            = BuildMI(*MBB, MI, DL, TII->get(AMDGPU::V_MOV_B32_e32), TmpReg)
            .addReg(SubReg, SubKillState);


          // There could be undef components of a spilled super register.
          // TODO: Can we detect this and skip the spill?
          if (NumSubRegs > 1) {
            // The last implicit use of the SuperReg carries the "Kill" flag.
            unsigned SuperKillState = 0;
            if (i + 1 == e)
              SuperKillState |= getKillRegState(IsKill);
            Mov.addReg(SuperReg, RegState::Implicit | SuperKillState);
          }

          unsigned Size = FrameInfo.getObjectSize(Index);
          unsigned Align = FrameInfo.getObjectAlignment(Index);
          MachinePointerInfo PtrInfo
              = MachinePointerInfo::getFixedStack(*MF, Index);
          MachineMemOperand *MMO
              = MF->getMachineMemOperand(PtrInfo, MachineMemOperand::MOStore,
                                         Size, Align);
          BuildMI(*MBB, MI, DL, TII->get(AMDGPU::SI_SPILL_V32_SAVE))
                  .addReg(TmpReg, RegState::Kill)         // src
                  .addFrameIndex(Index)                   // vaddr
                  .addReg(MFI->getScratchRSrcReg())       // srrsrc
                  .addReg(MFI->getScratchWaveOffsetReg()) // soffset
                  .addImm(i * 4)                          // offset
                  .addMemOperand(MMO);
        }
      }
      MI->eraseFromParent();
      MFI->addToSpilledSGPRs(NumSubRegs);
      break;
    }

    // SGPR register restore
    case AMDGPU::SI_SPILL_S512_RESTORE:
    case AMDGPU::SI_SPILL_S256_RESTORE:
    case AMDGPU::SI_SPILL_S128_RESTORE:
    case AMDGPU::SI_SPILL_S64_RESTORE:
    case AMDGPU::SI_SPILL_S32_RESTORE: {
      unsigned NumSubRegs = getNumSubRegsForSpillOp(MI->getOpcode());
      unsigned TmpReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
      unsigned SuperReg = MI->getOperand(0).getReg();

      // m0 is not allowed as with readlane/writelane, so a temporary SGPR and
      // extra copy is needed.
      bool IsM0 = (SuperReg == AMDGPU::M0);
      if (IsM0) {
        assert(NumSubRegs == 1);
        SuperReg = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
      }

      for (unsigned i = 0, e = NumSubRegs; i < e; ++i) {
        unsigned SubReg = NumSubRegs == 1 ?
          SuperReg : getSubReg(SuperReg, getSubRegFromChannel(i));

        SIMachineFunctionInfo::SpilledReg Spill
          = MFI->getSpilledReg(MF, Index, i);

        if (Spill.hasReg()) {
          BuildMI(*MBB, MI, DL,
                  TII->getMCOpcodeFromPseudo(AMDGPU::V_READLANE_B32),
                  SubReg)
                  .addReg(Spill.VGPR)
                  .addImm(Spill.Lane)
                  .addReg(MI->getOperand(0).getReg(), RegState::ImplicitDefine);
        } else {
          // Restore SGPR from a stack slot.
          // FIXME: We should use S_LOAD_DWORD here for VI.

          unsigned Align = FrameInfo.getObjectAlignment(Index);
          unsigned Size = FrameInfo.getObjectSize(Index);

          MachinePointerInfo PtrInfo
              = MachinePointerInfo::getFixedStack(*MF, Index);

          MachineMemOperand *MMO = MF->getMachineMemOperand(
              PtrInfo, MachineMemOperand::MOLoad, Size, Align);

          BuildMI(*MBB, MI, DL, TII->get(AMDGPU::SI_SPILL_V32_RESTORE), TmpReg)
                  .addFrameIndex(Index)                   // vaddr
                  .addReg(MFI->getScratchRSrcReg())       // srsrc
                  .addReg(MFI->getScratchWaveOffsetReg()) // soffset
                  .addImm(i * 4)                          // offset
                  .addMemOperand(MMO);
          BuildMI(*MBB, MI, DL,
                  TII->get(AMDGPU::V_READFIRSTLANE_B32), SubReg)
                  .addReg(TmpReg, RegState::Kill)
                  .addReg(MI->getOperand(0).getReg(), RegState::ImplicitDefine);
        }
      }

      if (IsM0 && SuperReg != AMDGPU::M0) {
        BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
          .addReg(SuperReg);
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
            TII->getNamedOperand(*MI, AMDGPU::OpName::vdata),
            TII->getNamedOperand(*MI, AMDGPU::OpName::srsrc)->getReg(),
            TII->getNamedOperand(*MI, AMDGPU::OpName::soffset)->getReg(),
            FrameInfo.getObjectOffset(Index) +
            TII->getNamedOperand(*MI, AMDGPU::OpName::offset)->getImm(), RS);
      MI->eraseFromParent();
      MFI->addToSpilledVGPRs(getNumSubRegsForSpillOp(MI->getOpcode()));
      break;
    case AMDGPU::SI_SPILL_V32_RESTORE:
    case AMDGPU::SI_SPILL_V64_RESTORE:
    case AMDGPU::SI_SPILL_V96_RESTORE:
    case AMDGPU::SI_SPILL_V128_RESTORE:
    case AMDGPU::SI_SPILL_V256_RESTORE:
    case AMDGPU::SI_SPILL_V512_RESTORE: {
      buildScratchLoadStore(MI, AMDGPU::BUFFER_LOAD_DWORD_OFFSET,
            TII->getNamedOperand(*MI, AMDGPU::OpName::vdata),
            TII->getNamedOperand(*MI, AMDGPU::OpName::srsrc)->getReg(),
            TII->getNamedOperand(*MI, AMDGPU::OpName::soffset)->getReg(),
            FrameInfo.getObjectOffset(Index) +
            TII->getNamedOperand(*MI, AMDGPU::OpName::offset)->getImm(), RS);
      MI->eraseFromParent();
      break;
    }

    default: {
      int64_t Offset = FrameInfo.getObjectOffset(Index);
      FIOp.ChangeToImmediate(Offset);
      if (!TII->isImmOperandLegal(*MI, FIOperandNum, FIOp)) {
        unsigned TmpReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
        BuildMI(*MBB, MI, MI->getDebugLoc(),
                TII->get(AMDGPU::V_MOV_B32_e32), TmpReg)
                .addImm(Offset);
        FIOp.ChangeToRegister(TmpReg, false, false, true);
      }
    }
  }
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

bool SIRegisterInfo::opCanUseLiteralConstant(unsigned OpType) const {
  return OpType == AMDGPU::OPERAND_REG_IMM32_INT ||
         OpType == AMDGPU::OPERAND_REG_IMM32_FP;
}

bool SIRegisterInfo::opCanUseInlineConstant(unsigned OpType) const {
  if (opCanUseLiteralConstant(OpType))
    return true;

  return OpType == AMDGPU::OPERAND_REG_INLINE_C_INT ||
         OpType == AMDGPU::OPERAND_REG_INLINE_C_FP;
}

// FIXME: Most of these are flexible with HSA and we don't need to reserve them
// as input registers if unused. Whether the dispatch ptr is necessary should be
// easy to detect from used intrinsics. Scratch setup is harder to know.
unsigned SIRegisterInfo::getPreloadedValue(const MachineFunction &MF,
                                           enum PreloadedValue Value) const {

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
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
    assert(ST.isAmdCodeObjectV2() &&
           "Non-CodeObjectV2 ABI currently uses relocations");
    assert(MFI->hasPrivateSegmentBuffer());
    return MFI->PrivateSegmentBufferUserSGPR;
  case SIRegisterInfo::KERNARG_SEGMENT_PTR:
    assert(MFI->hasKernargSegmentPtr());
    return MFI->KernargSegmentPtrUserSGPR;
  case SIRegisterInfo::DISPATCH_ID:
    assert(MFI->hasDispatchID());
    return MFI->DispatchIDUserSGPR;
  case SIRegisterInfo::FLAT_SCRATCH_INIT:
    assert(MFI->hasFlatScratchInit());
    return MFI->FlatScratchInitUserSGPR;
  case SIRegisterInfo::DISPATCH_PTR:
    assert(MFI->hasDispatchPtr());
    return MFI->DispatchPtrUserSGPR;
  case SIRegisterInfo::QUEUE_PTR:
    assert(MFI->hasQueuePtr());
    return MFI->QueuePtrUserSGPR;
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
unsigned
SIRegisterInfo::findUnusedRegister(const MachineRegisterInfo &MRI,
                                   const TargetRegisterClass *RC,
                                   const MachineFunction &MF) const {

  for (unsigned Reg : *RC)
    if (MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg))
      return Reg;
  return AMDGPU::NoRegister;
}

bool SIRegisterInfo::isVGPR(const MachineRegisterInfo &MRI,
                            unsigned Reg) const {
  const TargetRegisterClass *RC;
  if (TargetRegisterInfo::isVirtualRegister(Reg))
    RC = MRI.getRegClass(Reg);
  else
    RC = getPhysRegClass(Reg);

  return hasVGPRs(RC);
}

unsigned SIRegisterInfo::getTotalNumSGPRs(const SISubtarget &ST) const {
  if (ST.getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS)
    return 800;
  return 512;
}

unsigned SIRegisterInfo::getNumAddressableSGPRs(const SISubtarget &ST) const {
  if (ST.getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS)
    return 102;
  return 104;
}

unsigned SIRegisterInfo::getNumReservedSGPRs(const SISubtarget &ST) const {
  if (ST.getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS)
    return 6; // VCC, FLAT_SCRATCH, XNACK.
  return 2; // VCC.
}

unsigned SIRegisterInfo::getMinNumSGPRs(const SISubtarget &ST,
                                        unsigned WavesPerEU) const {
  if (ST.getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS) {
    switch (WavesPerEU) {
      case 0:  return 0;
      case 10: return 0;
      case 9:  return 0;
      case 8:  return 81;
      default: return 97;
    }
  } else {
    switch (WavesPerEU) {
      case 0:  return 0;
      case 10: return 0;
      case 9:  return 49;
      case 8:  return 57;
      case 7:  return 65;
      case 6:  return 73;
      case 5:  return 81;
      default: return 97;
    }
  }
}

unsigned SIRegisterInfo::getMaxNumSGPRs(const SISubtarget &ST,
                                        unsigned WavesPerEU) const {
  if (ST.getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS) {
    switch (WavesPerEU) {
      case 0:  return 80;
      case 10: return 80;
      case 9:  return 80;
      case 8:  return 96;
      default: return getNumAddressableSGPRs(ST);
    }
  } else {
    switch (WavesPerEU) {
      case 0:  return 48;
      case 10: return 48;
      case 9:  return 56;
      case 8:  return 64;
      case 7:  return 72;
      case 6:  return 80;
      case 5:  return 96;
      default: return getNumAddressableSGPRs(ST);
    }
  }
}

unsigned SIRegisterInfo::getMaxNumSGPRs(const MachineFunction &MF) const {
  const Function &F = *MF.getFunction();

  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();

  // Compute maximum number of SGPRs function can use using default/requested
  // minimum number of waves per execution unit.
  std::pair<unsigned, unsigned> WavesPerEU = MFI.getWavesPerEU();
  unsigned MaxNumSGPRs = getMaxNumSGPRs(ST, WavesPerEU.first);

  // Check if maximum number of SGPRs was explicitly requested using
  // "amdgpu-num-sgpr" attribute.
  if (F.hasFnAttribute("amdgpu-num-sgpr")) {
    unsigned Requested = AMDGPU::getIntegerAttribute(
      F, "amdgpu-num-sgpr", MaxNumSGPRs);

    // Make sure requested value does not violate subtarget's specifications.
    if (Requested && Requested <= getNumReservedSGPRs(ST))
      Requested = 0;

    // Make sure requested value is compatible with values implied by
    // default/requested minimum/maximum number of waves per execution unit.
    if (Requested && Requested > getMaxNumSGPRs(ST, WavesPerEU.first))
      Requested = 0;
    if (WavesPerEU.second &&
        Requested && Requested < getMinNumSGPRs(ST, WavesPerEU.second))
      Requested = 0;

    if (Requested)
      MaxNumSGPRs = Requested;
  }

  if (ST.hasSGPRInitBug())
    MaxNumSGPRs = SISubtarget::FIXED_SGPR_COUNT_FOR_INIT_BUG;

  return MaxNumSGPRs - getNumReservedSGPRs(ST);
}

unsigned SIRegisterInfo::getNumDebuggerReservedVGPRs(
  const SISubtarget &ST) const {
  if (ST.debuggerReserveRegs())
    return 4;
  return 0;
}

unsigned SIRegisterInfo::getMinNumVGPRs(unsigned WavesPerEU) const {
  switch (WavesPerEU) {
    case 0:  return 0;
    case 10: return 0;
    case 9:  return 25;
    case 8:  return 29;
    case 7:  return 33;
    case 6:  return 37;
    case 5:  return 41;
    case 4:  return 49;
    case 3:  return 65;
    case 2:  return 85;
    default: return 129;
  }
}

unsigned SIRegisterInfo::getMaxNumVGPRs(unsigned WavesPerEU) const {
  switch (WavesPerEU) {
    case 0:  return 24;
    case 10: return 24;
    case 9:  return 28;
    case 8:  return 32;
    case 7:  return 36;
    case 6:  return 40;
    case 5:  return 48;
    case 4:  return 64;
    case 3:  return 84;
    case 2:  return 128;
    default: return getTotalNumVGPRs();
  }
}

unsigned SIRegisterInfo::getMaxNumVGPRs(const MachineFunction &MF) const {
  const Function &F = *MF.getFunction();

  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();

  // Compute maximum number of VGPRs function can use using default/requested
  // minimum number of waves per execution unit.
  std::pair<unsigned, unsigned> WavesPerEU = MFI.getWavesPerEU();
  unsigned MaxNumVGPRs = getMaxNumVGPRs(WavesPerEU.first);

  // Check if maximum number of VGPRs was explicitly requested using
  // "amdgpu-num-vgpr" attribute.
  if (F.hasFnAttribute("amdgpu-num-vgpr")) {
    unsigned Requested = AMDGPU::getIntegerAttribute(
      F, "amdgpu-num-vgpr", MaxNumVGPRs);

    // Make sure requested value does not violate subtarget's specifications.
    if (Requested && Requested <= getNumDebuggerReservedVGPRs(ST))
      Requested = 0;

    // Make sure requested value is compatible with values implied by
    // default/requested minimum/maximum number of waves per execution unit.
    if (Requested && Requested > getMaxNumVGPRs(WavesPerEU.first))
      Requested = 0;
    if (WavesPerEU.second &&
        Requested && Requested < getMinNumVGPRs(WavesPerEU.second))
      Requested = 0;

    if (Requested)
      MaxNumVGPRs = Requested;
  }

  return MaxNumVGPRs - getNumDebuggerReservedVGPRs(ST);
}
