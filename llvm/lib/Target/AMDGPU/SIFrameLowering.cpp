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


static ArrayRef<MCPhysReg> getAllSGPR128(const SISubtarget &ST,
                                         const MachineFunction &MF) {
  return makeArrayRef(AMDGPU::SGPR_128RegClass.begin(),
                      ST.getMaxNumSGPRs(MF) / 4);
}

static ArrayRef<MCPhysReg> getAllSGPRs(const SISubtarget &ST,
                                       const MachineFunction &MF) {
  return makeArrayRef(AMDGPU::SGPR_32RegClass.begin(),
                      ST.getMaxNumSGPRs(MF));
}

void SIFrameLowering::emitFlatScratchInit(const SISubtarget &ST,
                                          MachineFunction &MF,
                                          MachineBasicBlock &MBB) const {
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo* TRI = &TII->getRegisterInfo();

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

  unsigned FlatScrInitLo = TRI->getSubReg(FlatScratchInitReg, AMDGPU::sub0);
  unsigned FlatScrInitHi = TRI->getSubReg(FlatScratchInitReg, AMDGPU::sub1);

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned ScratchWaveOffsetReg = MFI->getScratchWaveOffsetReg();

  // Do a 64-bit pointer add.
  if (ST.flatScratchIsPointer()) {
    BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ADD_U32), AMDGPU::FLAT_SCR_LO)
      .addReg(FlatScrInitLo)
      .addReg(ScratchWaveOffsetReg);
    BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ADDC_U32), AMDGPU::FLAT_SCR_HI)
      .addReg(FlatScrInitHi)
      .addImm(0);

    return;
  }

  // Copy the size in bytes.
  BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), AMDGPU::FLAT_SCR_LO)
    .addReg(FlatScrInitHi, RegState::Kill);

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
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // We need to insert initialization of the scratch resource descriptor.
  unsigned ScratchRsrcReg = MFI->getScratchRSrcReg();
  if (ScratchRsrcReg == AMDGPU::NoRegister ||
      !MRI.isPhysRegUsed(ScratchRsrcReg))
    return AMDGPU::NoRegister;

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

  unsigned NumPreloaded = (MFI->getNumPreloadedSGPRs() + 3) / 4;
  ArrayRef<MCPhysReg> AllSGPR128s = getAllSGPR128(ST, MF);
  AllSGPR128s = AllSGPR128s.slice(std::min(static_cast<unsigned>(AllSGPR128s.size()), NumPreloaded));

  // Skip the last N reserved elements because they should have already been
  // reserved for VCC etc.
  for (MCPhysReg Reg : AllSGPR128s) {
    // Pick the first unallocated one. Make sure we don't clobber the other
    // reserved input we needed.
    if (!MRI.isPhysRegUsed(Reg) && MRI.isAllocatable(Reg)) {
      MRI.replaceRegWith(ScratchRsrcReg, Reg);
      MFI->setScratchRSrcReg(Reg);
      return Reg;
    }
  }

  return ScratchRsrcReg;
}

// Shift down registers reserved for the scratch wave offset and stack pointer
// SGPRs.
std::pair<unsigned, unsigned>
SIFrameLowering::getReservedPrivateSegmentWaveByteOffsetReg(
  const SISubtarget &ST,
  const SIInstrInfo *TII,
  const SIRegisterInfo *TRI,
  SIMachineFunctionInfo *MFI,
  MachineFunction &MF) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned ScratchWaveOffsetReg = MFI->getScratchWaveOffsetReg();

  // No replacement necessary.
  if (ScratchWaveOffsetReg == AMDGPU::NoRegister ||
      !MRI.isPhysRegUsed(ScratchWaveOffsetReg)) {
    assert(MFI->getStackPtrOffsetReg() == AMDGPU::NoRegister);
    return std::make_pair(AMDGPU::NoRegister, AMDGPU::NoRegister);
  }

  unsigned SPReg = MFI->getStackPtrOffsetReg();
  if (ST.hasSGPRInitBug())
    return std::make_pair(ScratchWaveOffsetReg, SPReg);

  unsigned NumPreloaded = MFI->getNumPreloadedSGPRs();

  ArrayRef<MCPhysReg> AllSGPRs = getAllSGPRs(ST, MF);
  if (NumPreloaded > AllSGPRs.size())
    return std::make_pair(ScratchWaveOffsetReg, SPReg);

  AllSGPRs = AllSGPRs.slice(NumPreloaded);

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
  // + 1 if stack pointer is used.
  // ----
  //  13 (+1)
  unsigned ReservedRegCount = 13;
  if (SPReg != AMDGPU::NoRegister)
    ++ReservedRegCount;

  if (AllSGPRs.size() < ReservedRegCount)
    return std::make_pair(ScratchWaveOffsetReg, SPReg);

  bool HandledScratchWaveOffsetReg =
    ScratchWaveOffsetReg != TRI->reservedPrivateSegmentWaveByteOffsetReg(MF);

  for (MCPhysReg Reg : AllSGPRs.drop_back(ReservedRegCount)) {
    // Pick the first unallocated SGPR. Be careful not to pick an alias of the
    // scratch descriptor, since we haven’t added its uses yet.
    if (!MRI.isPhysRegUsed(Reg) && MRI.isAllocatable(Reg)) {
      if (!HandledScratchWaveOffsetReg) {
        HandledScratchWaveOffsetReg = true;

        MRI.replaceRegWith(ScratchWaveOffsetReg, Reg);
        MFI->setScratchWaveOffsetReg(Reg);
        ScratchWaveOffsetReg = Reg;
      } else {
        if (SPReg == AMDGPU::NoRegister)
          break;

        MRI.replaceRegWith(SPReg, Reg);
        MFI->setStackPtrOffsetReg(Reg);
        SPReg = Reg;
        break;
      }
    }
  }

  return std::make_pair(ScratchWaveOffsetReg, SPReg);
}

void SIFrameLowering::emitPrologue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  // Emit debugger prologue if "amdgpu-debugger-emit-prologue" attribute was
  // specified.
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  auto AMDGPUASI = ST.getAMDGPUAS();
  if (ST.debuggerEmitPrologue())
    emitDebuggerPrologue(MF, MBB);

  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");

  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  // If we only have SGPR spills, we won't actually be using scratch memory
  // since these spill to VGPRs.
  //
  // FIXME: We should be cleaning up these unused SGPR spill frame indices
  // somewhere.

  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = &TII->getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // We need to do the replacement of the private segment buffer and wave offset
  // register even if there are no stack objects. There could be stores to undef
  // or a constant without an associated object.

  // FIXME: We still have implicit uses on SGPR spill instructions in case they
  // need to spill to vector memory. It's likely that will not happen, but at
  // this point it appears we need the setup. This part of the prolog should be
  // emitted after frame indices are eliminated.

  if (MF.getFrameInfo().hasStackObjects() && MFI->hasFlatScratchInit())
    emitFlatScratchInit(ST, MF, MBB);

  unsigned SPReg = MFI->getStackPtrOffsetReg();
  if (SPReg != AMDGPU::NoRegister) {
    DebugLoc DL;
    int64_t StackSize = MF.getFrameInfo().getStackSize();

    if (StackSize == 0) {
      BuildMI(MBB, MBB.begin(), DL, TII->get(AMDGPU::COPY), SPReg)
        .addReg(MFI->getScratchWaveOffsetReg());
    } else {
      BuildMI(MBB, MBB.begin(), DL, TII->get(AMDGPU::S_ADD_U32), SPReg)
        .addReg(MFI->getScratchWaveOffsetReg())
        .addImm(StackSize * ST.getWavefrontSize());
    }
  }

  unsigned ScratchRsrcReg
    = getReservedPrivateSegmentBufferReg(ST, TII, TRI, MFI, MF);

  unsigned ScratchWaveOffsetReg;
  std::tie(ScratchWaveOffsetReg, SPReg)
    = getReservedPrivateSegmentWaveByteOffsetReg(ST, TII, TRI, MFI, MF);

  // It's possible to have uses of only ScratchWaveOffsetReg without
  // ScratchRsrcReg if it's only used for the initialization of flat_scratch,
  // but the inverse is not true.
  if (ScratchWaveOffsetReg == AMDGPU::NoRegister) {
    assert(ScratchRsrcReg == AMDGPU::NoRegister);
    return;
  }

  // We need to insert initialization of the scratch resource descriptor.
  unsigned PreloadedScratchWaveOffsetReg = TRI->getPreloadedValue(
    MF, SIRegisterInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);

  unsigned PreloadedPrivateBufferReg = AMDGPU::NoRegister;
  if (ST.isAmdCodeObjectV2(MF) || ST.isMesaGfxShader(MF)) {
    PreloadedPrivateBufferReg = TRI->getPreloadedValue(
      MF, SIRegisterInfo::PRIVATE_SEGMENT_BUFFER);
  }

  bool OffsetRegUsed = MRI.isPhysRegUsed(ScratchWaveOffsetReg);
  bool ResourceRegUsed = ScratchRsrcReg != AMDGPU::NoRegister &&
                         MRI.isPhysRegUsed(ScratchRsrcReg);

  // We added live-ins during argument lowering, but since they were not used
  // they were deleted. We're adding the uses now, so add them back.
  if (OffsetRegUsed) {
    assert(PreloadedScratchWaveOffsetReg != AMDGPU::NoRegister &&
           "scratch wave offset input is required");
    MRI.addLiveIn(PreloadedScratchWaveOffsetReg);
    MBB.addLiveIn(PreloadedScratchWaveOffsetReg);
  }

  if (ResourceRegUsed && PreloadedPrivateBufferReg != AMDGPU::NoRegister) {
    assert(ST.isAmdCodeObjectV2(MF) || ST.isMesaGfxShader(MF));
    MRI.addLiveIn(PreloadedPrivateBufferReg);
    MBB.addLiveIn(PreloadedPrivateBufferReg);
  }

  // Make the register selected live throughout the function.
  for (MachineBasicBlock &OtherBB : MF) {
    if (&OtherBB == &MBB)
      continue;

    if (OffsetRegUsed)
      OtherBB.addLiveIn(ScratchWaveOffsetReg);

    if (ResourceRegUsed)
      OtherBB.addLiveIn(ScratchRsrcReg);
  }

  DebugLoc DL;
  MachineBasicBlock::iterator I = MBB.begin();

  // If we reserved the original input registers, we don't need to copy to the
  // reserved registers.

  bool CopyBuffer = ResourceRegUsed &&
    PreloadedPrivateBufferReg != AMDGPU::NoRegister &&
    ST.isAmdCodeObjectV2(MF) &&
    ScratchRsrcReg != PreloadedPrivateBufferReg;

  // This needs to be careful of the copying order to avoid overwriting one of
  // the input registers before it's been copied to it's final
  // destination. Usually the offset should be copied first.
  bool CopyBufferFirst = TRI->isSubRegisterEq(PreloadedPrivateBufferReg,
                                              ScratchWaveOffsetReg);
  if (CopyBuffer && CopyBufferFirst) {
    BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), ScratchRsrcReg)
      .addReg(PreloadedPrivateBufferReg, RegState::Kill);
  }

  if (OffsetRegUsed &&
      PreloadedScratchWaveOffsetReg != ScratchWaveOffsetReg) {
    BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), ScratchWaveOffsetReg)
      .addReg(PreloadedScratchWaveOffsetReg, RegState::Kill);
  }

  if (CopyBuffer && !CopyBufferFirst) {
    BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), ScratchRsrcReg)
      .addReg(PreloadedPrivateBufferReg, RegState::Kill);
  }

  if (ResourceRegUsed && (ST.isMesaGfxShader(MF) || (PreloadedPrivateBufferReg == AMDGPU::NoRegister))) {
    assert(!ST.isAmdCodeObjectV2(MF));
    const MCInstrDesc &SMovB32 = TII->get(AMDGPU::S_MOV_B32);

    unsigned Rsrc2 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub2);
    unsigned Rsrc3 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub3);

    // Use relocations to get the pointer, and setup the other bits manually.
    uint64_t Rsrc23 = TII->getScratchRsrcWords23();

    if (MFI->hasPrivateMemoryInputPtr()) {
      unsigned Rsrc01 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub0_sub1);

      if (AMDGPU::isCompute(MF.getFunction()->getCallingConv())) {
        const MCInstrDesc &Mov64 = TII->get(AMDGPU::S_MOV_B64);

        BuildMI(MBB, I, DL, Mov64, Rsrc01)
          .addReg(PreloadedPrivateBufferReg)
          .addReg(ScratchRsrcReg, RegState::ImplicitDefine);
      } else {
        const MCInstrDesc &LoadDwordX2 = TII->get(AMDGPU::S_LOAD_DWORDX2_IMM);

        PointerType *PtrTy =
          PointerType::get(Type::getInt64Ty(MF.getFunction()->getContext()),
                           AMDGPUASI.CONSTANT_ADDRESS);
        MachinePointerInfo PtrInfo(UndefValue::get(PtrTy));
        auto MMO = MF.getMachineMemOperand(PtrInfo,
                                           MachineMemOperand::MOLoad |
                                           MachineMemOperand::MOInvariant |
                                           MachineMemOperand::MODereferenceable,
                                           0, 0);
        BuildMI(MBB, I, DL, LoadDwordX2, Rsrc01)
          .addReg(PreloadedPrivateBufferReg)
          .addImm(0) // offset
          .addImm(0) // glc
          .addMemOperand(MMO)
          .addReg(ScratchRsrcReg, RegState::ImplicitDefine);
      }
    } else {
      unsigned Rsrc0 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub0);
      unsigned Rsrc1 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub1);

      BuildMI(MBB, I, DL, SMovB32, Rsrc0)
        .addExternalSymbol("SCRATCH_RSRC_DWORD0")
        .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

      BuildMI(MBB, I, DL, SMovB32, Rsrc1)
        .addExternalSymbol("SCRATCH_RSRC_DWORD1")
        .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

    }

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

static bool allStackObjectsAreDead(const MachineFrameInfo &MFI) {
  for (int I = MFI.getObjectIndexBegin(), E = MFI.getObjectIndexEnd();
       I != E; ++I) {
    if (!MFI.isDeadObjectIndex(I))
      return false;
  }

  return true;
}

int SIFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                            unsigned &FrameReg) const {
  const SIRegisterInfo *RI = MF.getSubtarget<SISubtarget>().getRegisterInfo();

  FrameReg = RI->getFrameRegister(MF);
  return MF.getFrameInfo().getObjectOffset(FI);
}

void SIFrameLowering::processFunctionBeforeFrameFinalized(
  MachineFunction &MF,
  RegScavenger *RS) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();

  if (!MFI.hasStackObjects())
    return;

  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  bool AllSGPRSpilledToVGPRs = false;

  if (TRI.spillSGPRToVGPR() && FuncInfo->hasSpilledSGPRs()) {
    AllSGPRSpilledToVGPRs = true;

    // Process all SGPR spills before frame offsets are finalized. Ideally SGPRs
    // are spilled to VGPRs, in which case we can eliminate the stack usage.
    //
    // XXX - This operates under the assumption that only other SGPR spills are
    // users of the frame index. I'm not 100% sure this is correct. The
    // StackColoring pass has a comment saying a future improvement would be to
    // merging of allocas with spill slots, but for now according to
    // MachineFrameInfo isSpillSlot can't alias any other object.
    for (MachineBasicBlock &MBB : MF) {
      MachineBasicBlock::iterator Next;
      for (auto I = MBB.begin(), E = MBB.end(); I != E; I = Next) {
        MachineInstr &MI = *I;
        Next = std::next(I);

        if (TII->isSGPRSpill(MI)) {
          int FI = TII->getNamedOperand(MI, AMDGPU::OpName::addr)->getIndex();
          if (FuncInfo->allocateSGPRSpillToVGPR(MF, FI)) {
            bool Spilled = TRI.eliminateSGPRToVGPRSpillFrameIndex(MI, FI, RS);
            (void)Spilled;
            assert(Spilled && "failed to spill SGPR to VGPR when allocated");
          } else
            AllSGPRSpilledToVGPRs = false;
        }
      }
    }

    FuncInfo->removeSGPRToVGPRFrameIndices(MFI);
  }

  // FIXME: The other checks should be redundant with allStackObjectsAreDead,
  // but currently hasNonSpillStackObjects is set only from source
  // allocas. Stack temps produced from legalization are not counted currently.
  if (FuncInfo->hasNonSpillStackObjects() || FuncInfo->hasSpilledVGPRs() ||
      !AllSGPRSpilledToVGPRs || !allStackObjectsAreDead(MFI)) {
    assert(RS && "RegScavenger required if spilling");

    // We force this to be at offset 0 so no user object ever has 0 as an
    // address, so we may use 0 as an invalid pointer value. This is because
    // LLVM assumes 0 is an invalid pointer in address space 0. Because alloca
    // is required to be address space 0, we are forced to accept this for
    // now. Ideally we could have the stack in another address space with 0 as a
    // valid pointer, and -1 as the null value.
    //
    // This will also waste additional space when user stack objects require > 4
    // byte alignment.
    //
    // The main cost here is losing the offset for addressing modes. However
    // this also ensures we shouldn't need a register for the offset when
    // emergency scavenging.
    int ScavengeFI = MFI.CreateFixedObject(
      TRI.getSpillSize(AMDGPU::SGPR_32RegClass), 0, false);
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
