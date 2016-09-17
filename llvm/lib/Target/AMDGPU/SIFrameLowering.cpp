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
#include "AMDGPUSubtarget.h"

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"

using namespace llvm;


static bool hasOnlySGPRSpills(const SIMachineFunctionInfo *FuncInfo,
                              const MachineFrameInfo &MFI) {
  return FuncInfo->hasSpilledSGPRs() &&
    (!FuncInfo->hasSpilledVGPRs() && !FuncInfo->hasNonSpillStackObjects());
}

static ArrayRef<MCPhysReg> getAllSGPR128(const MachineFunction &MF,
                                         const SIRegisterInfo *TRI) {
  return makeArrayRef(AMDGPU::SGPR_128RegClass.begin(),
                      TRI->getMaxNumSGPRs(MF) / 4);
}

static ArrayRef<MCPhysReg> getAllSGPRs(const MachineFunction &MF,
                                       const SIRegisterInfo *TRI) {
  return makeArrayRef(AMDGPU::SGPR_32RegClass.begin(),
                      TRI->getMaxNumSGPRs(MF));
}

void SIFrameLowering::emitFlatScratchInit(const SIInstrInfo *TII,
                                          const SIRegisterInfo* TRI,
                                          MachineFunction &MF,
                                          MachineBasicBlock &MBB) const {
  // We don't need this if we only have spills since there is no user facing
  // scratch.

  // TODO: If we know we don't have flat instructions earlier, we can omit
  // this from the input registers.
  //
  // TODO: We only need to know if we access scratch space through a flat
  // pointer. Because we only detect if flat instructions are used at all,
  // this will be used more often than necessary on VI.

  // Debug location must be unknown since the first debug location is used to
  // determine the end of the prologue.
  DebugLoc DL;
  MachineBasicBlock::iterator I = MBB.begin();

  unsigned FlatScratchInitReg
    = TRI->getPreloadedValue(MF, SIRegisterInfo::FLAT_SCRATCH_INIT);

  MachineRegisterInfo &MRI = MF.getRegInfo();
  MRI.addLiveIn(FlatScratchInitReg);
  MBB.addLiveIn(FlatScratchInitReg);

  // Copy the size in bytes.
  unsigned FlatScrInitHi = TRI->getSubReg(FlatScratchInitReg, AMDGPU::sub1);
  BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), AMDGPU::FLAT_SCR_LO)
    .addReg(FlatScrInitHi, RegState::Kill);

  unsigned FlatScrInitLo = TRI->getSubReg(FlatScratchInitReg, AMDGPU::sub0);

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned ScratchWaveOffsetReg = MFI->getScratchWaveOffsetReg();


  // Add wave offset in bytes to private base offset.
  // See comment in AMDKernelCodeT.h for enable_sgpr_flat_scratch_init.
  BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ADD_U32), FlatScrInitLo)
    .addReg(FlatScrInitLo)
    .addReg(ScratchWaveOffsetReg);

  // Convert offset to 256-byte units.
  BuildMI(MBB, I, DL, TII->get(AMDGPU::S_LSHR_B32), AMDGPU::FLAT_SCR_HI)
    .addReg(FlatScrInitLo, RegState::Kill)
    .addImm(8);
}

unsigned SIFrameLowering::getReservedPrivateSegmentBufferReg(
  const SISubtarget &ST,
  const SIInstrInfo *TII,
  const SIRegisterInfo *TRI,
  SIMachineFunctionInfo *MFI,
  MachineFunction &MF) const {

  // We need to insert initialization of the scratch resource descriptor.
  unsigned ScratchRsrcReg = MFI->getScratchRSrcReg();
  assert(ScratchRsrcReg != AMDGPU::NoRegister);

  if (ST.hasSGPRInitBug() ||
      ScratchRsrcReg != TRI->reservedPrivateSegmentBufferReg(MF))
    return ScratchRsrcReg;

  // We reserved the last registers for this. Shift it down to the end of those
  // which were actually used.
  //
  // FIXME: It might be safer to use a pseudoregister before replacement.

  // FIXME: We should be able to eliminate unused input registers. We only
  // cannot do this for the resources required for scratch access. For now we
  // skip over user SGPRs and may leave unused holes.

  // We find the resource first because it has an alignment requirement.

  MachineRegisterInfo &MRI = MF.getRegInfo();

  unsigned NumPreloaded = MFI->getNumPreloadedSGPRs() / 4;
  // Skip the last 2 elements because the last one is reserved for VCC, and
  // this is the 2nd to last element already.
  for (MCPhysReg Reg : getAllSGPR128(MF, TRI).drop_back(2).slice(NumPreloaded)) {
    // Pick the first unallocated one. Make sure we don't clobber the other
    // reserved input we needed.
    if (!MRI.isPhysRegUsed(Reg)) {
      assert(MRI.isAllocatable(Reg));
      MRI.replaceRegWith(ScratchRsrcReg, Reg);
      MFI->setScratchRSrcReg(Reg);
      return Reg;
    }
  }

  return ScratchRsrcReg;
}

unsigned SIFrameLowering::getReservedPrivateSegmentWaveByteOffsetReg(
  const SISubtarget &ST,
  const SIInstrInfo *TII,
  const SIRegisterInfo *TRI,
  SIMachineFunctionInfo *MFI,
  MachineFunction &MF) const {
  unsigned ScratchWaveOffsetReg = MFI->getScratchWaveOffsetReg();
  if (ST.hasSGPRInitBug() ||
      ScratchWaveOffsetReg != TRI->reservedPrivateSegmentWaveByteOffsetReg(MF))
    return ScratchWaveOffsetReg;

  unsigned ScratchRsrcReg = MFI->getScratchRSrcReg();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned NumPreloaded = MFI->getNumPreloadedSGPRs();

  // We need to drop register from the end of the list that we cannot use
  // for the scratch wave offset.
  // + 2 s102 and s103 do not exist on VI.
  // + 2 for vcc
  // + 2 for xnack_mask
  // + 2 for flat_scratch
  // + 4 for registers reserved for scratch resource register
  // + 1 for register reserved for scratch wave offset.  (By exluding this
  //     register from the list to consider, it means that when this
  //     register is being used for the scratch wave offset and there
  //     are no other free SGPRs, then the value will stay in this register.
  // ----
  //  13
  for (MCPhysReg Reg : getAllSGPRs(MF, TRI).drop_back(13).slice(NumPreloaded)) {
    // Pick the first unallocated SGPR. Be careful not to pick an alias of the
    // scratch descriptor, since we haven’t added its uses yet.
    if (!MRI.isPhysRegUsed(Reg)) {
      if (!MRI.isAllocatable(Reg) ||
          TRI->isSubRegisterEq(ScratchRsrcReg, Reg))
        continue;

      MRI.replaceRegWith(ScratchWaveOffsetReg, Reg);
      MFI->setScratchWaveOffsetReg(Reg);
      return Reg;
    }
  }

  return ScratchWaveOffsetReg;
}

void SIFrameLowering::emitPrologue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  // Emit debugger prologue if "amdgpu-debugger-emit-prologue" attribute was
  // specified.
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  if (ST.debuggerEmitPrologue())
    emitDebuggerPrologue(MF, MBB);

  if (!MF.getFrameInfo().hasStackObjects())
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

  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = &TII->getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  unsigned ScratchRsrcReg
    = getReservedPrivateSegmentBufferReg(ST, TII, TRI, MFI, MF);
  unsigned ScratchWaveOffsetReg
    = getReservedPrivateSegmentWaveByteOffsetReg(ST, TII, TRI, MFI, MF);
  assert(ScratchRsrcReg != AMDGPU::NoRegister);
  assert(ScratchWaveOffsetReg != AMDGPU::NoRegister);
  assert(!TRI->isSubRegister(ScratchRsrcReg, ScratchWaveOffsetReg));

  if (MFI->hasFlatScratchInit())
    emitFlatScratchInit(TII, TRI, MF, MBB);

  // We need to insert initialization of the scratch resource descriptor.
  unsigned PreloadedScratchWaveOffsetReg = TRI->getPreloadedValue(
    MF, SIRegisterInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);

  unsigned PreloadedPrivateBufferReg = AMDGPU::NoRegister;
  if (ST.isAmdCodeObjectV2()) {
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
  MRI.addLiveIn(PreloadedScratchWaveOffsetReg);
  MBB.addLiveIn(PreloadedScratchWaveOffsetReg);

  if (ST.isAmdCodeObjectV2()) {
    MRI.addLiveIn(PreloadedPrivateBufferReg);
    MBB.addLiveIn(PreloadedPrivateBufferReg);
  }

  // Make the register selected live throughout the function.
  for (MachineBasicBlock &OtherBB : MF) {
    if (&OtherBB == &MBB)
      continue;

    OtherBB.addLiveIn(ScratchRsrcReg);
    OtherBB.addLiveIn(ScratchWaveOffsetReg);
  }

  DebugLoc DL;
  MachineBasicBlock::iterator I = MBB.begin();

  if (PreloadedScratchWaveOffsetReg != ScratchWaveOffsetReg) {
    // Make sure we emit the copy for the offset first. We may have chosen to
    // copy the buffer resource into a register that aliases the input offset
    // register.
    BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), ScratchWaveOffsetReg)
      .addReg(PreloadedScratchWaveOffsetReg, RegState::Kill);
  }

  if (ST.isAmdCodeObjectV2()) {
    // Insert copies from argument register.
    assert(
      !TRI->isSubRegisterEq(PreloadedPrivateBufferReg, ScratchRsrcReg) &&
      !TRI->isSubRegisterEq(PreloadedPrivateBufferReg, ScratchWaveOffsetReg));

    BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), ScratchRsrcReg)
      .addReg(PreloadedPrivateBufferReg, RegState::Kill);
  } else {
    const MCInstrDesc &SMovB32 = TII->get(AMDGPU::S_MOV_B32);

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
}

void SIFrameLowering::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {

}

void SIFrameLowering::processFunctionBeforeFrameFinalized(
  MachineFunction &MF,
  RegScavenger *RS) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();

  if (!MFI.hasStackObjects())
    return;

  bool MayNeedScavengingEmergencySlot = MFI.hasStackObjects();

  assert((RS || !MayNeedScavengingEmergencySlot) &&
         "RegScavenger required if spilling");

  if (MayNeedScavengingEmergencySlot) {
    int ScavengeFI = MFI.CreateStackObject(
      AMDGPU::SGPR_32RegClass.getSize(),
      AMDGPU::SGPR_32RegClass.getAlignment(), false);
    RS->addScavengingFrameIndex(ScavengeFI);
  }
}

void SIFrameLowering::emitDebuggerPrologue(MachineFunction &MF,
                                           MachineBasicBlock &MBB) const {
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = &TII->getRegisterInfo();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  MachineBasicBlock::iterator I = MBB.begin();
  DebugLoc DL;

  // For each dimension:
  for (unsigned i = 0; i < 3; ++i) {
    // Get work group ID SGPR, and make it live-in again.
    unsigned WorkGroupIDSGPR = MFI->getWorkGroupIDSGPR(i);
    MF.getRegInfo().addLiveIn(WorkGroupIDSGPR);
    MBB.addLiveIn(WorkGroupIDSGPR);

    // Since SGPRs are spilled into VGPRs, copy work group ID SGPR to VGPR in
    // order to spill it to scratch.
    unsigned WorkGroupIDVGPR =
      MF.getRegInfo().createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BuildMI(MBB, I, DL, TII->get(AMDGPU::V_MOV_B32_e32), WorkGroupIDVGPR)
      .addReg(WorkGroupIDSGPR);

    // Spill work group ID.
    int WorkGroupIDObjectIdx = MFI->getDebuggerWorkGroupIDStackObjectIndex(i);
    TII->storeRegToStackSlot(MBB, I, WorkGroupIDVGPR, false,
      WorkGroupIDObjectIdx, &AMDGPU::VGPR_32RegClass, TRI);

    // Get work item ID VGPR, and make it live-in again.
    unsigned WorkItemIDVGPR = MFI->getWorkItemIDVGPR(i);
    MF.getRegInfo().addLiveIn(WorkItemIDVGPR);
    MBB.addLiveIn(WorkItemIDVGPR);

    // Spill work item ID.
    int WorkItemIDObjectIdx = MFI->getDebuggerWorkItemIDStackObjectIndex(i);
    TII->storeRegToStackSlot(MBB, I, WorkItemIDVGPR, false,
      WorkItemIDObjectIdx, &AMDGPU::VGPR_32RegClass, TRI);
  }
}
