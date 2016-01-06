//===----------------------- SIFrameLowering.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//

#include "SIFrameLowering.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"

using namespace llvm;


static bool hasOnlySGPRSpills(const SIMachineFunctionInfo *FuncInfo,
                              const MachineFrameInfo *FrameInfo) {
  if (!FuncInfo->hasSpilledSGPRs())
    return false;

  if (FuncInfo->hasSpilledVGPRs())
    return false;

  for (int I = FrameInfo->getObjectIndexBegin(),
         E = FrameInfo->getObjectIndexEnd(); I != E; ++I) {
    if (!FrameInfo->isSpillSlotObjectIndex(I))
      return false;
  }

  return true;
}

static ArrayRef<MCPhysReg> getAllSGPR128() {
  return makeArrayRef(AMDGPU::SReg_128RegClass.begin(),
                      AMDGPU::SReg_128RegClass.getNumRegs());
}

static ArrayRef<MCPhysReg> getAllSGPRs() {
  return makeArrayRef(AMDGPU::SGPR_32RegClass.begin(),
                      AMDGPU::SGPR_32RegClass.getNumRegs());
}

void SIFrameLowering::emitPrologue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  if (!MF.getFrameInfo()->hasStackObjects())
    return;

  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");

  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  // If we only have SGPR spills, we won't actually be using scratch memory
  // since these spill to VGPRs.
  //
  // FIXME: We should be cleaning up these unused SGPR spill frame indices
  // somewhere.
  if (hasOnlySGPRSpills(MFI, MF.getFrameInfo()))
    return;

  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const SIRegisterInfo *TRI = &TII->getRegisterInfo();
  const AMDGPUSubtarget &ST = MF.getSubtarget<AMDGPUSubtarget>();

  // We need to insert initialization of the scratch resource descriptor.
  unsigned ScratchRsrcReg = MFI->getScratchRSrcReg();
  assert(ScratchRsrcReg != AMDGPU::NoRegister);

  unsigned ScratchWaveOffsetReg = MFI->getScratchWaveOffsetReg();
  assert(ScratchWaveOffsetReg != AMDGPU::NoRegister);

  unsigned PreloadedScratchWaveOffsetReg = TRI->getPreloadedValue(
    MF, SIRegisterInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);

  unsigned PreloadedPrivateBufferReg = AMDGPU::NoRegister;
  if (ST.isAmdHsaOS()) {
    PreloadedPrivateBufferReg = TRI->getPreloadedValue(
      MF, SIRegisterInfo::PRIVATE_SEGMENT_BUFFER);
  }

  // If we reserved the original input registers, we don't need to copy to the
  // reserved registers.
  if (ScratchRsrcReg == PreloadedPrivateBufferReg) {
    // We should always reserve these 5 registers at the same time.
    assert(ScratchWaveOffsetReg == PreloadedScratchWaveOffsetReg &&
           "scratch wave offset and private segment buffer inconsistent");
    return;
  }


  // We added live-ins during argument lowering, but since they were not used
  // they were deleted. We're adding the uses now, so add them back.
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MRI.addLiveIn(PreloadedScratchWaveOffsetReg);
  MBB.addLiveIn(PreloadedScratchWaveOffsetReg);

  if (ST.isAmdHsaOS()) {
    MRI.addLiveIn(PreloadedPrivateBufferReg);
    MBB.addLiveIn(PreloadedPrivateBufferReg);
  }

  if (!ST.hasSGPRInitBug()) {
    // We reserved the last registers for this. Shift it down to the end of those
    // which were actually used.
    //
    // FIXME: It might be safer to use a pseudoregister before replacement.

    // FIXME: We should be able to eliminate unused input registers. We only
    // cannot do this for the resources required for scratch access. For now we
    // skip over user SGPRs and may leave unused holes.

    // We find the resource first because it has an alignment requirement.
    if (ScratchRsrcReg == TRI->reservedPrivateSegmentBufferReg(MF)) {
      MachineRegisterInfo &MRI = MF.getRegInfo();

      unsigned NumPreloaded = MFI->getNumPreloadedSGPRs() / 4;
      // Skip the last 2 elements because the last one is reserved for VCC, and
      // this is the 2nd to last element already.
      for (MCPhysReg Reg : getAllSGPR128().drop_back(2).slice(NumPreloaded)) {
        // Pick the first unallocated one. Make sure we don't clobber the other
        // reserved input we needed.
        if (!MRI.isPhysRegUsed(Reg)) {
          assert(MRI.isAllocatable(Reg));
          MRI.replaceRegWith(ScratchRsrcReg, Reg);
          ScratchRsrcReg = Reg;
          MFI->setScratchRSrcReg(ScratchRsrcReg);
          break;
        }
      }
    }

    if (ScratchWaveOffsetReg == TRI->reservedPrivateSegmentWaveByteOffsetReg(MF)) {
      MachineRegisterInfo &MRI = MF.getRegInfo();
      // Skip the last 2 elements because the last one is reserved for VCC, and
      // this is the 2nd to last element already.
      unsigned NumPreloaded = MFI->getNumPreloadedSGPRs();
      for (MCPhysReg Reg : getAllSGPRs().drop_back(6).slice(NumPreloaded)) {
        // Pick the first unallocated SGPR. Be careful not to pick an alias of the
        // scratch descriptor, since we haven’t added its uses yet.
        if (!MRI.isPhysRegUsed(Reg)) {
          assert(MRI.isAllocatable(Reg) &&
                !TRI->isSubRegisterEq(ScratchRsrcReg, Reg));

          MRI.replaceRegWith(ScratchWaveOffsetReg, Reg);
          ScratchWaveOffsetReg = Reg;
          MFI->setScratchWaveOffsetReg(ScratchWaveOffsetReg);
          break;
        }
      }
    }
  }


  assert(!TRI->isSubRegister(ScratchRsrcReg, ScratchWaveOffsetReg));

  const MCInstrDesc &SMovB32 = TII->get(AMDGPU::S_MOV_B32);
  MachineBasicBlock::iterator I = MBB.begin();
  DebugLoc DL;

  if (PreloadedScratchWaveOffsetReg != ScratchWaveOffsetReg) {
    // Make sure we emit the copy for the offset first. We may have chosen to copy
    // the buffer resource into a register that aliases the input offset register.
    BuildMI(MBB, I, DL, SMovB32, ScratchWaveOffsetReg)
      .addReg(PreloadedScratchWaveOffsetReg, RegState::Kill);
  }

  if (ST.isAmdHsaOS()) {
    // Insert copies from argument register.
    assert(
      !TRI->isSubRegisterEq(PreloadedPrivateBufferReg, ScratchRsrcReg) &&
      !TRI->isSubRegisterEq(PreloadedPrivateBufferReg, ScratchWaveOffsetReg));

    unsigned Rsrc01 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub0_sub1);
    unsigned Rsrc23 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub2_sub3);

    unsigned Lo = TRI->getSubReg(PreloadedPrivateBufferReg, AMDGPU::sub0_sub1);
    unsigned Hi = TRI->getSubReg(PreloadedPrivateBufferReg, AMDGPU::sub2_sub3);

    const MCInstrDesc &SMovB64 = TII->get(AMDGPU::S_MOV_B64);

    BuildMI(MBB, I, DL, SMovB64, Rsrc01)
      .addReg(Lo, RegState::Kill);
    BuildMI(MBB, I, DL, SMovB64, Rsrc23)
      .addReg(Hi, RegState::Kill);
  } else {
    unsigned Rsrc0 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub0);
    unsigned Rsrc1 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub1);
    unsigned Rsrc2 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub2);
    unsigned Rsrc3 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub3);

    // Use relocations to get the pointer, and setup the other bits manually.
    uint64_t Rsrc23 = TII->getScratchRsrcWords23();
    BuildMI(MBB, I, DL, SMovB32, Rsrc0)
      .addExternalSymbol("SCRATCH_RSRC_DWORD0")
      .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

    BuildMI(MBB, I, DL, SMovB32, Rsrc1)
      .addExternalSymbol("SCRATCH_RSRC_DWORD1")
      .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

    BuildMI(MBB, I, DL, SMovB32, Rsrc2)
      .addImm(Rsrc23 & 0xffffffff)
      .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

    BuildMI(MBB, I, DL, SMovB32, Rsrc3)
      .addImm(Rsrc23 >> 32)
      .addReg(ScratchRsrcReg, RegState::ImplicitDefine);
  }

  // Make the register selected live throughout the function.
  for (MachineBasicBlock &OtherBB : MF) {
    if (&OtherBB == &MBB)
      continue;

    OtherBB.addLiveIn(ScratchRsrcReg);
    OtherBB.addLiveIn(ScratchWaveOffsetReg);
  }
}

void SIFrameLowering::processFunctionBeforeFrameFinalized(
  MachineFunction &MF,
  RegScavenger *RS) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();

  if (!MFI->hasStackObjects())
    return;

  bool MayNeedScavengingEmergencySlot = MFI->hasStackObjects();

  assert((RS || !MayNeedScavengingEmergencySlot) &&
         "RegScavenger required if spilling");

  if (MayNeedScavengingEmergencySlot) {
    int ScavengeFI = MFI->CreateSpillStackObject(
      AMDGPU::SGPR_32RegClass.getSize(),
      AMDGPU::SGPR_32RegClass.getAlignment());
    RS->addScavengingFrameIndex(ScavengeFI);
  }
}
