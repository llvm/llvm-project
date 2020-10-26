//===-- SIRegisterInfo.cpp - SI Register Information ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// SI implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "SIRegisterInfo.h"
#include "AMDGPURegisterBankInfo.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "MCTargetDesc/AMDGPUInstPrinter.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include <vector>

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "AMDGPUGenRegisterInfo.inc"

static cl::opt<bool> EnableSpillSGPRToVGPR(
  "amdgpu-spill-sgpr-to-vgpr",
  cl::desc("Enable spilling VGPRs to SGPRs"),
  cl::ReallyHidden,
  cl::init(true));

std::array<std::vector<int16_t>, 16> SIRegisterInfo::RegSplitParts;
std::array<std::array<uint16_t, 32>, 9> SIRegisterInfo::SubRegFromChannelTable;

// Map numbers of DWORDs to indexes in SubRegFromChannelTable.
// Valid indexes are shifted 1, such that a 0 mapping means unsupported.
// e.g. for 8 DWORDs (256-bit), SubRegFromChannelTableWidthMap[8] = 8,
//      meaning index 7 in SubRegFromChannelTable.
static const std::array<unsigned, 17> SubRegFromChannelTableWidthMap = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 9};

SIRegisterInfo::SIRegisterInfo(const GCNSubtarget &ST)
    : AMDGPUGenRegisterInfo(AMDGPU::PC_REG, ST.getAMDGPUDwarfFlavour()), ST(ST),
      SpillSGPRToVGPR(EnableSpillSGPRToVGPR), isWave32(ST.isWave32()) {

  assert(getSubRegIndexLaneMask(AMDGPU::sub0).getAsInteger() == 3 &&
         getSubRegIndexLaneMask(AMDGPU::sub31).getAsInteger() == (3ULL << 62) &&
         (getSubRegIndexLaneMask(AMDGPU::lo16) |
          getSubRegIndexLaneMask(AMDGPU::hi16)).getAsInteger() ==
           getSubRegIndexLaneMask(AMDGPU::sub0).getAsInteger() &&
         "getNumCoveredRegs() will not work with generated subreg masks!");

  RegPressureIgnoredUnits.resize(getNumRegUnits());
  RegPressureIgnoredUnits.set(*MCRegUnitIterator(AMDGPU::M0, this));
  for (auto Reg : AMDGPU::VGPR_HI16RegClass)
    RegPressureIgnoredUnits.set(*MCRegUnitIterator(Reg, this));

  // HACK: Until this is fully tablegen'd.
  static llvm::once_flag InitializeRegSplitPartsFlag;

  static auto InitializeRegSplitPartsOnce = [this]() {
    for (unsigned Idx = 1, E = getNumSubRegIndices() - 1; Idx < E; ++Idx) {
      unsigned Size = getSubRegIdxSize(Idx);
      if (Size & 31)
        continue;
      std::vector<int16_t> &Vec = RegSplitParts[Size / 32 - 1];
      unsigned Pos = getSubRegIdxOffset(Idx);
      if (Pos % Size)
        continue;
      Pos /= Size;
      if (Vec.empty()) {
        unsigned MaxNumParts = 1024 / Size; // Maximum register is 1024 bits.
        Vec.resize(MaxNumParts);
      }
      Vec[Pos] = Idx;
    }
  };

  static llvm::once_flag InitializeSubRegFromChannelTableFlag;

  static auto InitializeSubRegFromChannelTableOnce = [this]() {
    for (auto &Row : SubRegFromChannelTable)
      Row.fill(AMDGPU::NoSubRegister);
    for (uint16_t Idx = 1; Idx < getNumSubRegIndices(); ++Idx) {
      unsigned Width = AMDGPUSubRegIdxRanges[Idx].Size / 32;
      unsigned Offset = AMDGPUSubRegIdxRanges[Idx].Offset / 32;
      assert(Width < SubRegFromChannelTableWidthMap.size());
      Width = SubRegFromChannelTableWidthMap[Width];
      if (Width == 0)
        continue;
      unsigned TableIdx = Width - 1;
      assert(TableIdx < SubRegFromChannelTable.size());
      assert(Offset < SubRegFromChannelTable[TableIdx].size());
      SubRegFromChannelTable[TableIdx][Offset] = Idx;
    }
  };

  llvm::call_once(InitializeRegSplitPartsFlag, InitializeRegSplitPartsOnce);
  llvm::call_once(InitializeSubRegFromChannelTableFlag,
                  InitializeSubRegFromChannelTableOnce);
}

void SIRegisterInfo::reserveRegisterTuples(BitVector &Reserved,
                                           MCRegister Reg) const {
  MCRegAliasIterator R(Reg, this, true);

  for (; R.isValid(); ++R)
    Reserved.set(*R);
}

// Forced to be here by one .inc
const MCPhysReg *SIRegisterInfo::getCalleeSavedRegs(
  const MachineFunction *MF) const {
  CallingConv::ID CC = MF->getFunction().getCallingConv();
  switch (CC) {
  case CallingConv::C:
  case CallingConv::Fast:
  case CallingConv::Cold:
    return CSR_AMDGPU_HighRegs_SaveList;
  default: {
    // Dummy to not crash RegisterClassInfo.
    static const MCPhysReg NoCalleeSavedReg = AMDGPU::NoRegister;
    return &NoCalleeSavedReg;
  }
  }
}

const MCPhysReg *
SIRegisterInfo::getCalleeSavedRegsViaCopy(const MachineFunction *MF) const {
  return nullptr;
}

const uint32_t *SIRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                                     CallingConv::ID CC) const {
  switch (CC) {
  case CallingConv::C:
  case CallingConv::Fast:
  case CallingConv::Cold:
    return CSR_AMDGPU_HighRegs_RegMask;
  default:
    return nullptr;
  }
}

const uint32_t *SIRegisterInfo::getNoPreservedMask() const {
  return CSR_AMDGPU_NoRegs_RegMask;
}

Register SIRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const SIFrameLowering *TFI =
      MF.getSubtarget<GCNSubtarget>().getFrameLowering();
  const SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  // During ISel lowering we always reserve the stack pointer in entry
  // functions, but never actually want to reference it when accessing our own
  // frame. If we need a frame pointer we use it, but otherwise we can just use
  // an immediate "0" which we represent by returning NoRegister.
  if (FuncInfo->isEntryFunction()) {
    return TFI->hasFP(MF) ? FuncInfo->getFrameOffsetReg() : Register();
  }
  return TFI->hasFP(MF) ? FuncInfo->getFrameOffsetReg()
                        : FuncInfo->getStackPtrOffsetReg();
}

bool SIRegisterInfo::hasBasePointer(const MachineFunction &MF) const {
  // When we need stack realignment, we can't reference off of the
  // stack pointer, so we reserve a base pointer.
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MFI.getNumFixedObjects() && needsStackRealignment(MF);
}

Register SIRegisterInfo::getBaseRegister() const { return AMDGPU::SGPR34; }

const uint32_t *SIRegisterInfo::getAllVGPRRegMask() const {
  return CSR_AMDGPU_AllVGPRs_RegMask;
}

const uint32_t *SIRegisterInfo::getAllAllocatableSRegMask() const {
  return CSR_AMDGPU_AllAllocatableSRegs_RegMask;
}

unsigned SIRegisterInfo::getSubRegFromChannel(unsigned Channel,
                                              unsigned NumRegs) {
  assert(NumRegs < SubRegFromChannelTableWidthMap.size());
  unsigned NumRegIndex = SubRegFromChannelTableWidthMap[NumRegs];
  assert(NumRegIndex && "Not implemented");
  assert(Channel < SubRegFromChannelTable[NumRegIndex - 1].size());
  return SubRegFromChannelTable[NumRegIndex - 1][Channel];
}

MCRegister SIRegisterInfo::reservedPrivateSegmentBufferReg(
  const MachineFunction &MF) const {
  unsigned BaseIdx = alignDown(ST.getMaxNumSGPRs(MF), 4) - 4;
  MCRegister BaseReg(AMDGPU::SGPR_32RegClass.getRegister(BaseIdx));
  return getMatchingSuperReg(BaseReg, AMDGPU::sub0, &AMDGPU::SGPR_128RegClass);
}

BitVector SIRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(AMDGPU::MODE);

  // EXEC_LO and EXEC_HI could be allocated and used as regular register, but
  // this seems likely to result in bugs, so I'm marking them as reserved.
  reserveRegisterTuples(Reserved, AMDGPU::EXEC);
  reserveRegisterTuples(Reserved, AMDGPU::FLAT_SCR);

  // M0 has to be reserved so that llvm accepts it as a live-in into a block.
  reserveRegisterTuples(Reserved, AMDGPU::M0);

  // Reserve src_vccz, src_execz, src_scc.
  reserveRegisterTuples(Reserved, AMDGPU::SRC_VCCZ);
  reserveRegisterTuples(Reserved, AMDGPU::SRC_EXECZ);
  reserveRegisterTuples(Reserved, AMDGPU::SRC_SCC);

  // Reserve the memory aperture registers.
  reserveRegisterTuples(Reserved, AMDGPU::SRC_SHARED_BASE);
  reserveRegisterTuples(Reserved, AMDGPU::SRC_SHARED_LIMIT);
  reserveRegisterTuples(Reserved, AMDGPU::SRC_PRIVATE_BASE);
  reserveRegisterTuples(Reserved, AMDGPU::SRC_PRIVATE_LIMIT);

  // Reserve src_pops_exiting_wave_id - support is not implemented in Codegen.
  reserveRegisterTuples(Reserved, AMDGPU::SRC_POPS_EXITING_WAVE_ID);

  // Reserve xnack_mask registers - support is not implemented in Codegen.
  reserveRegisterTuples(Reserved, AMDGPU::XNACK_MASK);

  // Reserve lds_direct register - support is not implemented in Codegen.
  reserveRegisterTuples(Reserved, AMDGPU::LDS_DIRECT);

  // Reserve Trap Handler registers - support is not implemented in Codegen.
  reserveRegisterTuples(Reserved, AMDGPU::TBA);
  reserveRegisterTuples(Reserved, AMDGPU::TMA);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP0_TTMP1);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP2_TTMP3);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP4_TTMP5);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP6_TTMP7);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP8_TTMP9);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP10_TTMP11);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP12_TTMP13);
  reserveRegisterTuples(Reserved, AMDGPU::TTMP14_TTMP15);

  // Reserve null register - it shall never be allocated
  reserveRegisterTuples(Reserved, AMDGPU::SGPR_NULL);

  // Disallow vcc_hi allocation in wave32. It may be allocated but most likely
  // will result in bugs.
  if (isWave32) {
    Reserved.set(AMDGPU::VCC);
    Reserved.set(AMDGPU::VCC_HI);
  }

  unsigned MaxNumSGPRs = ST.getMaxNumSGPRs(MF);
  unsigned TotalNumSGPRs = AMDGPU::SGPR_32RegClass.getNumRegs();
  for (unsigned i = MaxNumSGPRs; i < TotalNumSGPRs; ++i) {
    unsigned Reg = AMDGPU::SGPR_32RegClass.getRegister(i);
    reserveRegisterTuples(Reserved, Reg);
  }

  unsigned MaxNumVGPRs = ST.getMaxNumVGPRs(MF);
  unsigned TotalNumVGPRs = AMDGPU::VGPR_32RegClass.getNumRegs();
  for (unsigned i = MaxNumVGPRs; i < TotalNumVGPRs; ++i) {
    unsigned Reg = AMDGPU::VGPR_32RegClass.getRegister(i);
    reserveRegisterTuples(Reserved, Reg);
    Reg = AMDGPU::AGPR_32RegClass.getRegister(i);
    reserveRegisterTuples(Reserved, Reg);
  }

  for (auto Reg : AMDGPU::SReg_32RegClass) {
    Reserved.set(getSubReg(Reg, AMDGPU::hi16));
    Register Low = getSubReg(Reg, AMDGPU::lo16);
    // This is to prevent BB vcc liveness errors.
    if (!AMDGPU::SGPR_LO16RegClass.contains(Low))
      Reserved.set(Low);
  }

  for (auto Reg : AMDGPU::AGPR_32RegClass) {
    Reserved.set(getSubReg(Reg, AMDGPU::hi16));
  }

  // Reserve all the rest AGPRs if there are no instructions to use it.
  if (!ST.hasMAIInsts()) {
    for (unsigned i = 0; i < MaxNumVGPRs; ++i) {
      unsigned Reg = AMDGPU::AGPR_32RegClass.getRegister(i);
      reserveRegisterTuples(Reserved, Reg);
    }
  }

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  Register ScratchRSrcReg = MFI->getScratchRSrcReg();
  if (ScratchRSrcReg != AMDGPU::NoRegister) {
    // Reserve 4 SGPRs for the scratch buffer resource descriptor in case we need
    // to spill.
    // TODO: May need to reserve a VGPR if doing LDS spilling.
    reserveRegisterTuples(Reserved, ScratchRSrcReg);
  }

  // We have to assume the SP is needed in case there are calls in the function,
  // which is detected after the function is lowered. If we aren't really going
  // to need SP, don't bother reserving it.
  MCRegister StackPtrReg = MFI->getStackPtrOffsetReg();

  if (StackPtrReg) {
    reserveRegisterTuples(Reserved, StackPtrReg);
    assert(!isSubRegister(ScratchRSrcReg, StackPtrReg));
  }

  MCRegister FrameReg = MFI->getFrameOffsetReg();
  if (FrameReg) {
    reserveRegisterTuples(Reserved, FrameReg);
    assert(!isSubRegister(ScratchRSrcReg, FrameReg));
  }

  if (hasBasePointer(MF)) {
    MCRegister BasePtrReg = getBaseRegister();
    reserveRegisterTuples(Reserved, BasePtrReg);
    assert(!isSubRegister(ScratchRSrcReg, BasePtrReg));
  }

  for (MCRegister Reg : MFI->WWMReservedRegs) {
    reserveRegisterTuples(Reserved, Reg);
  }

  // FIXME: Stop using reserved registers for this.
  for (MCPhysReg Reg : MFI->getAGPRSpillVGPRs())
    reserveRegisterTuples(Reserved, Reg);

  for (MCPhysReg Reg : MFI->getVGPRSpillAGPRs())
    reserveRegisterTuples(Reserved, Reg);

  for (auto SSpill : MFI->getSGPRSpillVGPRs())
    reserveRegisterTuples(Reserved, SSpill.VGPR);

  return Reserved;
}

bool SIRegisterInfo::canRealignStack(const MachineFunction &MF) const {
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  // On entry, the base address is 0, so it can't possibly need any more
  // alignment.

  // FIXME: Should be able to specify the entry frame alignment per calling
  // convention instead.
  if (Info->isEntryFunction())
    return false;

  return TargetRegisterInfo::canRealignStack(MF);
}

bool SIRegisterInfo::requiresRegisterScavenging(const MachineFunction &Fn) const {
  const SIMachineFunctionInfo *Info = Fn.getInfo<SIMachineFunctionInfo>();
  if (Info->isEntryFunction()) {
    const MachineFrameInfo &MFI = Fn.getFrameInfo();
    return MFI.hasStackObjects() || MFI.hasCalls();
  }

  // May need scavenger for dealing with callee saved registers.
  return true;
}

bool SIRegisterInfo::requiresFrameIndexScavenging(
  const MachineFunction &MF) const {
  // Do not use frame virtual registers. They used to be used for SGPRs, but
  // once we reach PrologEpilogInserter, we can no longer spill SGPRs. If the
  // scavenger fails, we can increment/decrement the necessary SGPRs to avoid a
  // spill.
  return false;
}

bool SIRegisterInfo::requiresFrameIndexReplacementScavenging(
  const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MFI.hasStackObjects();
}

bool SIRegisterInfo::requiresVirtualBaseRegisters(
  const MachineFunction &) const {
  // There are no special dedicated stack or frame pointers.
  return true;
}

int64_t SIRegisterInfo::getScratchInstrOffset(const MachineInstr *MI) const {
  assert(SIInstrInfo::isMUBUF(*MI) || SIInstrInfo::isFLATScratch(*MI));

  int OffIdx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                          AMDGPU::OpName::offset);
  return MI->getOperand(OffIdx).getImm();
}

int64_t SIRegisterInfo::getFrameIndexInstrOffset(const MachineInstr *MI,
                                                 int Idx) const {
  if (!SIInstrInfo::isMUBUF(*MI) && !SIInstrInfo::isFLATScratch(*MI))
    return 0;

  assert((Idx == AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                            AMDGPU::OpName::vaddr) ||
         (Idx == AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                            AMDGPU::OpName::saddr))) &&
         "Should never see frame index on non-address operand");

  return getScratchInstrOffset(MI);
}

bool SIRegisterInfo::needsFrameBaseReg(MachineInstr *MI, int64_t Offset) const {
  if (!MI->mayLoadOrStore())
    return false;

  int64_t FullOffset = Offset + getScratchInstrOffset(MI);

  if (SIInstrInfo::isMUBUF(*MI))
    return !SIInstrInfo::isLegalMUBUFImmOffset(FullOffset);

  const SIInstrInfo *TII = ST.getInstrInfo();
  return TII->isLegalFLATOffset(FullOffset, AMDGPUAS::PRIVATE_ADDRESS, true);
}

void SIRegisterInfo::materializeFrameBaseRegister(MachineBasicBlock *MBB,
                                                  Register BaseReg,
                                                  int FrameIdx,
                                                  int64_t Offset) const {
  MachineBasicBlock::iterator Ins = MBB->begin();
  DebugLoc DL; // Defaults to "unknown"

  if (Ins != MBB->end())
    DL = Ins->getDebugLoc();

  MachineFunction *MF = MBB->getParent();
  const SIInstrInfo *TII = ST.getInstrInfo();
  unsigned MovOpc = ST.enableFlatScratch() ? AMDGPU::S_MOV_B32
                                           : AMDGPU::V_MOV_B32_e32;

  if (Offset == 0) {
    BuildMI(*MBB, Ins, DL, TII->get(MovOpc), BaseReg)
      .addFrameIndex(FrameIdx);
    return;
  }

  MachineRegisterInfo &MRI = MF->getRegInfo();
  Register OffsetReg = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);

  Register FIReg = MRI.createVirtualRegister(
      ST.enableFlatScratch() ? &AMDGPU::SReg_32_XM0RegClass
                             : &AMDGPU::VGPR_32RegClass);

  BuildMI(*MBB, Ins, DL, TII->get(AMDGPU::S_MOV_B32), OffsetReg)
    .addImm(Offset);
  BuildMI(*MBB, Ins, DL, TII->get(MovOpc), FIReg)
    .addFrameIndex(FrameIdx);

  if (ST.enableFlatScratch() ) {
    BuildMI(*MBB, Ins, DL, TII->get(AMDGPU::S_ADD_U32), BaseReg)
        .addReg(OffsetReg, RegState::Kill)
        .addReg(FIReg);
    return;
  }

  TII->getAddNoCarry(*MBB, Ins, DL, BaseReg)
    .addReg(OffsetReg, RegState::Kill)
    .addReg(FIReg)
    .addImm(0); // clamp bit
}

void SIRegisterInfo::resolveFrameIndex(MachineInstr &MI, Register BaseReg,
                                       int64_t Offset) const {
  const SIInstrInfo *TII = ST.getInstrInfo();
  bool IsFlat = TII->isFLATScratch(MI);

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

  MachineOperand *FIOp =
      TII->getNamedOperand(MI, IsFlat ? AMDGPU::OpName::saddr
                                      : AMDGPU::OpName::vaddr);

  MachineOperand *OffsetOp = TII->getNamedOperand(MI, AMDGPU::OpName::offset);
  int64_t NewOffset = OffsetOp->getImm() + Offset;

#ifndef NDEBUG
  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction *MF = MBB->getParent();
  assert(FIOp && FIOp->isFI() && "frame index must be address operand");
  assert(TII->isMUBUF(MI) || TII->isFLATScratch(MI));

  if (IsFlat) {
    assert(TII->isLegalFLATOffset(NewOffset, AMDGPUAS::PRIVATE_ADDRESS, true) &&
           "offset should be legal");
    FIOp->ChangeToRegister(BaseReg, false);
    OffsetOp->setImm(NewOffset);
    return;
  }

  MachineOperand *SOffset = TII->getNamedOperand(MI, AMDGPU::OpName::soffset);
  assert((SOffset->isReg() &&
          SOffset->getReg() ==
              MF->getInfo<SIMachineFunctionInfo>()->getStackPtrOffsetReg()) ||
         (SOffset->isImm() && SOffset->getImm() == 0));
#endif

  assert(SIInstrInfo::isLegalMUBUFImmOffset(NewOffset) &&
         "offset should be legal");

  FIOp->ChangeToRegister(BaseReg, false);
  OffsetOp->setImm(NewOffset);
}

bool SIRegisterInfo::isFrameOffsetLegal(const MachineInstr *MI,
                                        Register BaseReg,
                                        int64_t Offset) const {
  if (!SIInstrInfo::isMUBUF(*MI) && !!SIInstrInfo::isFLATScratch(*MI))
    return false;

  int64_t NewOffset = Offset + getScratchInstrOffset(MI);

  if (SIInstrInfo::isMUBUF(*MI))
    return SIInstrInfo::isLegalMUBUFImmOffset(NewOffset);

  const SIInstrInfo *TII = ST.getInstrInfo();
  return TII->isLegalFLATOffset(NewOffset, AMDGPUAS::PRIVATE_ADDRESS, true);
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
  case AMDGPU::SI_SPILL_S1024_SAVE:
  case AMDGPU::SI_SPILL_S1024_RESTORE:
  case AMDGPU::SI_SPILL_V1024_SAVE:
  case AMDGPU::SI_SPILL_V1024_RESTORE:
  case AMDGPU::SI_SPILL_A1024_SAVE:
  case AMDGPU::SI_SPILL_A1024_RESTORE:
    return 32;
  case AMDGPU::SI_SPILL_S512_SAVE:
  case AMDGPU::SI_SPILL_S512_RESTORE:
  case AMDGPU::SI_SPILL_V512_SAVE:
  case AMDGPU::SI_SPILL_V512_RESTORE:
  case AMDGPU::SI_SPILL_A512_SAVE:
  case AMDGPU::SI_SPILL_A512_RESTORE:
    return 16;
  case AMDGPU::SI_SPILL_S256_SAVE:
  case AMDGPU::SI_SPILL_S256_RESTORE:
  case AMDGPU::SI_SPILL_V256_SAVE:
  case AMDGPU::SI_SPILL_V256_RESTORE:
  case AMDGPU::SI_SPILL_A256_SAVE:
  case AMDGPU::SI_SPILL_A256_RESTORE:
    return 8;
  case AMDGPU::SI_SPILL_S192_SAVE:
  case AMDGPU::SI_SPILL_S192_RESTORE:
  case AMDGPU::SI_SPILL_V192_SAVE:
  case AMDGPU::SI_SPILL_V192_RESTORE:
  case AMDGPU::SI_SPILL_A192_SAVE:
  case AMDGPU::SI_SPILL_A192_RESTORE:
    return 6;
  case AMDGPU::SI_SPILL_S160_SAVE:
  case AMDGPU::SI_SPILL_S160_RESTORE:
  case AMDGPU::SI_SPILL_V160_SAVE:
  case AMDGPU::SI_SPILL_V160_RESTORE:
  case AMDGPU::SI_SPILL_A160_SAVE:
  case AMDGPU::SI_SPILL_A160_RESTORE:
    return 5;
  case AMDGPU::SI_SPILL_S128_SAVE:
  case AMDGPU::SI_SPILL_S128_RESTORE:
  case AMDGPU::SI_SPILL_V128_SAVE:
  case AMDGPU::SI_SPILL_V128_RESTORE:
  case AMDGPU::SI_SPILL_A128_SAVE:
  case AMDGPU::SI_SPILL_A128_RESTORE:
    return 4;
  case AMDGPU::SI_SPILL_S96_SAVE:
  case AMDGPU::SI_SPILL_S96_RESTORE:
  case AMDGPU::SI_SPILL_V96_SAVE:
  case AMDGPU::SI_SPILL_V96_RESTORE:
  case AMDGPU::SI_SPILL_A96_SAVE:
  case AMDGPU::SI_SPILL_A96_RESTORE:
    return 3;
  case AMDGPU::SI_SPILL_S64_SAVE:
  case AMDGPU::SI_SPILL_S64_RESTORE:
  case AMDGPU::SI_SPILL_V64_SAVE:
  case AMDGPU::SI_SPILL_V64_RESTORE:
  case AMDGPU::SI_SPILL_A64_SAVE:
  case AMDGPU::SI_SPILL_A64_RESTORE:
    return 2;
  case AMDGPU::SI_SPILL_S32_SAVE:
  case AMDGPU::SI_SPILL_S32_RESTORE:
  case AMDGPU::SI_SPILL_V32_SAVE:
  case AMDGPU::SI_SPILL_V32_RESTORE:
  case AMDGPU::SI_SPILL_A32_SAVE:
  case AMDGPU::SI_SPILL_A32_RESTORE:
    return 1;
  default: llvm_unreachable("Invalid spill opcode");
  }
}

static int getOffsetMUBUFStore(unsigned Opc) {
  switch (Opc) {
  case AMDGPU::BUFFER_STORE_DWORD_OFFEN:
    return AMDGPU::BUFFER_STORE_DWORD_OFFSET;
  case AMDGPU::BUFFER_STORE_BYTE_OFFEN:
    return AMDGPU::BUFFER_STORE_BYTE_OFFSET;
  case AMDGPU::BUFFER_STORE_SHORT_OFFEN:
    return AMDGPU::BUFFER_STORE_SHORT_OFFSET;
  case AMDGPU::BUFFER_STORE_DWORDX2_OFFEN:
    return AMDGPU::BUFFER_STORE_DWORDX2_OFFSET;
  case AMDGPU::BUFFER_STORE_DWORDX4_OFFEN:
    return AMDGPU::BUFFER_STORE_DWORDX4_OFFSET;
  case AMDGPU::BUFFER_STORE_SHORT_D16_HI_OFFEN:
    return AMDGPU::BUFFER_STORE_SHORT_D16_HI_OFFSET;
  case AMDGPU::BUFFER_STORE_BYTE_D16_HI_OFFEN:
    return AMDGPU::BUFFER_STORE_BYTE_D16_HI_OFFSET;
  default:
    return -1;
  }
}

static int getOffsetMUBUFLoad(unsigned Opc) {
  switch (Opc) {
  case AMDGPU::BUFFER_LOAD_DWORD_OFFEN:
    return AMDGPU::BUFFER_LOAD_DWORD_OFFSET;
  case AMDGPU::BUFFER_LOAD_UBYTE_OFFEN:
    return AMDGPU::BUFFER_LOAD_UBYTE_OFFSET;
  case AMDGPU::BUFFER_LOAD_SBYTE_OFFEN:
    return AMDGPU::BUFFER_LOAD_SBYTE_OFFSET;
  case AMDGPU::BUFFER_LOAD_USHORT_OFFEN:
    return AMDGPU::BUFFER_LOAD_USHORT_OFFSET;
  case AMDGPU::BUFFER_LOAD_SSHORT_OFFEN:
    return AMDGPU::BUFFER_LOAD_SSHORT_OFFSET;
  case AMDGPU::BUFFER_LOAD_DWORDX2_OFFEN:
    return AMDGPU::BUFFER_LOAD_DWORDX2_OFFSET;
  case AMDGPU::BUFFER_LOAD_DWORDX4_OFFEN:
    return AMDGPU::BUFFER_LOAD_DWORDX4_OFFSET;
  case AMDGPU::BUFFER_LOAD_UBYTE_D16_OFFEN:
    return AMDGPU::BUFFER_LOAD_UBYTE_D16_OFFSET;
  case AMDGPU::BUFFER_LOAD_UBYTE_D16_HI_OFFEN:
    return AMDGPU::BUFFER_LOAD_UBYTE_D16_HI_OFFSET;
  case AMDGPU::BUFFER_LOAD_SBYTE_D16_OFFEN:
    return AMDGPU::BUFFER_LOAD_SBYTE_D16_OFFSET;
  case AMDGPU::BUFFER_LOAD_SBYTE_D16_HI_OFFEN:
    return AMDGPU::BUFFER_LOAD_SBYTE_D16_HI_OFFSET;
  case AMDGPU::BUFFER_LOAD_SHORT_D16_OFFEN:
    return AMDGPU::BUFFER_LOAD_SHORT_D16_OFFSET;
  case AMDGPU::BUFFER_LOAD_SHORT_D16_HI_OFFEN:
    return AMDGPU::BUFFER_LOAD_SHORT_D16_HI_OFFSET;
  default:
    return -1;
  }
}

static MachineInstrBuilder spillVGPRtoAGPR(const GCNSubtarget &ST,
                                           MachineBasicBlock::iterator MI,
                                           int Index,
                                           unsigned Lane,
                                           unsigned ValueReg,
                                           bool IsKill) {
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction *MF = MI->getParent()->getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  const SIInstrInfo *TII = ST.getInstrInfo();

  MCPhysReg Reg = MFI->getVGPRToAGPRSpill(Index, Lane);

  if (Reg == AMDGPU::NoRegister)
    return MachineInstrBuilder();

  bool IsStore = MI->mayStore();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  auto *TRI = static_cast<const SIRegisterInfo*>(MRI.getTargetRegisterInfo());

  unsigned Dst = IsStore ? Reg : ValueReg;
  unsigned Src = IsStore ? ValueReg : Reg;
  unsigned Opc = (IsStore ^ TRI->isVGPR(MRI, Reg)) ? AMDGPU::V_ACCVGPR_WRITE_B32
                                                   : AMDGPU::V_ACCVGPR_READ_B32;

  return BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(Opc), Dst)
           .addReg(Src, getKillRegState(IsKill));
}

// This differs from buildSpillLoadStore by only scavenging a VGPR. It does not
// need to handle the case where an SGPR may need to be spilled while spilling.
static bool buildMUBUFOffsetLoadStore(const GCNSubtarget &ST,
                                      MachineFrameInfo &MFI,
                                      MachineBasicBlock::iterator MI,
                                      int Index,
                                      int64_t Offset) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  MachineBasicBlock *MBB = MI->getParent();
  const DebugLoc &DL = MI->getDebugLoc();
  bool IsStore = MI->mayStore();

  unsigned Opc = MI->getOpcode();
  int LoadStoreOp = IsStore ?
    getOffsetMUBUFStore(Opc) : getOffsetMUBUFLoad(Opc);
  if (LoadStoreOp == -1)
    return false;

  const MachineOperand *Reg = TII->getNamedOperand(*MI, AMDGPU::OpName::vdata);
  if (spillVGPRtoAGPR(ST, MI, Index, 0, Reg->getReg(), false).getInstr())
    return true;

  MachineInstrBuilder NewMI =
      BuildMI(*MBB, MI, DL, TII->get(LoadStoreOp))
          .add(*Reg)
          .add(*TII->getNamedOperand(*MI, AMDGPU::OpName::srsrc))
          .add(*TII->getNamedOperand(*MI, AMDGPU::OpName::soffset))
          .addImm(Offset)
          .addImm(0) // glc
          .addImm(0) // slc
          .addImm(0) // tfe
          .addImm(0) // dlc
          .addImm(0) // swz
          .cloneMemRefs(*MI);

  const MachineOperand *VDataIn = TII->getNamedOperand(*MI,
                                                       AMDGPU::OpName::vdata_in);
  if (VDataIn)
    NewMI.add(*VDataIn);
  return true;
}

void SIRegisterInfo::buildSpillLoadStore(MachineBasicBlock::iterator MI,
                                         unsigned LoadStoreOp,
                                         int Index,
                                         Register ValueReg,
                                         bool IsKill,
                                         MCRegister ScratchRsrcReg,
                                         MCRegister ScratchOffsetReg,
                                         int64_t InstOffset,
                                         MachineMemOperand *MMO,
                                         RegScavenger *RS) const {
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction *MF = MI->getParent()->getParent();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const MachineFrameInfo &MFI = MF->getFrameInfo();
  const SIMachineFunctionInfo *FuncInfo = MF->getInfo<SIMachineFunctionInfo>();

  const MCInstrDesc *Desc = &TII->get(LoadStoreOp);
  const DebugLoc &DL = MI->getDebugLoc();
  bool IsStore = Desc->mayStore();
  bool IsFlat = TII->isFLATScratch(LoadStoreOp);

  bool Scavenged = false;
  MCRegister SOffset = ScratchOffsetReg;

  const unsigned EltSize = 4;
  const TargetRegisterClass *RC = getRegClassForReg(MF->getRegInfo(), ValueReg);
  unsigned NumSubRegs = AMDGPU::getRegBitWidth(RC->getID()) / (EltSize * CHAR_BIT);
  unsigned Size = NumSubRegs * EltSize;
  int64_t Offset = InstOffset + MFI.getObjectOffset(Index);
  int64_t MaxOffset = Offset + Size - EltSize;
  int64_t ScratchOffsetRegDelta = 0;

  Align Alignment = MFI.getObjectAlign(Index);
  const MachinePointerInfo &BasePtrInfo = MMO->getPointerInfo();

  assert((Offset % EltSize) == 0 && "unexpected VGPR spill offset");

  bool IsOffsetLegal = IsFlat
      ? TII->isLegalFLATOffset(MaxOffset, AMDGPUAS::PRIVATE_ADDRESS, true)
      : SIInstrInfo::isLegalMUBUFImmOffset(MaxOffset);
  if (!IsOffsetLegal || (IsFlat && !SOffset && !ST.hasFlatScratchSTMode())) {
    SOffset = MCRegister();

    // We currently only support spilling VGPRs to EltSize boundaries, meaning
    // we can simplify the adjustment of Offset here to just scale with
    // WavefrontSize.
    if (!IsFlat)
      Offset *= ST.getWavefrontSize();

    // We don't have access to the register scavenger if this function is called
    // during  PEI::scavengeFrameVirtualRegs().
    if (RS)
      SOffset = RS->scavengeRegister(&AMDGPU::SGPR_32RegClass, MI, 0, false);

    if (!SOffset) {
      // There are no free SGPRs, and since we are in the process of spilling
      // VGPRs too.  Since we need a VGPR in order to spill SGPRs (this is true
      // on SI/CI and on VI it is true until we implement spilling using scalar
      // stores), we have no way to free up an SGPR.  Our solution here is to
      // add the offset directly to the ScratchOffset or StackPtrOffset
      // register, and then subtract the offset after the spill to return the
      // register to it's original value.
      if (!ScratchOffsetReg)
        ScratchOffsetReg = FuncInfo->getStackPtrOffsetReg();
      SOffset = ScratchOffsetReg;
      ScratchOffsetRegDelta = Offset;
    } else {
      Scavenged = true;
    }

    if (!SOffset)
      report_fatal_error("could not scavenge SGPR to spill in entry function");

    if (ScratchOffsetReg == AMDGPU::NoRegister) {
      BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_MOV_B32), SOffset)
          .addImm(Offset);
    } else {
      BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_ADD_U32), SOffset)
          .addReg(ScratchOffsetReg)
          .addImm(Offset);
    }

    Offset = 0;
  }

  if (IsFlat && SOffset == AMDGPU::NoRegister) {
    assert(AMDGPU::getNamedOperandIdx(LoadStoreOp, AMDGPU::OpName::vaddr) < 0
           && "Unexpected vaddr for flat scratch with a FI operand");

    assert(ST.hasFlatScratchSTMode());
    LoadStoreOp = AMDGPU::getFlatScratchInstSTfromSS(LoadStoreOp);
    Desc = &TII->get(LoadStoreOp);
  }

  Register TmpReg;

  // FIXME: Flat scratch does not have to be limited to a dword per store.
  for (unsigned i = 0, e = NumSubRegs; i != e; ++i, Offset += EltSize) {
    Register SubReg = NumSubRegs == 1
                          ? Register(ValueReg)
                          : getSubReg(ValueReg, getSubRegFromChannel(i));

    unsigned SOffsetRegState = 0;
    unsigned SrcDstRegState = getDefRegState(!IsStore);
    if (i + 1 == e) {
      SOffsetRegState |= getKillRegState(Scavenged);
      // The last implicit use carries the "Kill" flag.
      SrcDstRegState |= getKillRegState(IsKill);
    }

    // Make sure the whole register is defined if there are undef components by
    // adding an implicit def of the super-reg on the first instruction.
    const bool NeedSuperRegDef = NumSubRegs > 1 && IsStore && i == 0;

    auto MIB = spillVGPRtoAGPR(ST, MI, Index, i, SubReg, IsKill);

    if (!MIB.getInstr()) {
      unsigned FinalReg = SubReg;

      const bool IsAGPR = hasAGPRs(RC);
      if (IsAGPR) {
        if (!TmpReg) {
          assert(RS && "Needs to have RegScavenger to spill an AGPR!");
          // FIXME: change to scavengeRegisterBackwards()
          TmpReg = RS->scavengeRegister(&AMDGPU::VGPR_32RegClass, MI, 0);
          RS->setRegUsed(TmpReg);
        }
        if (IsStore) {
          auto AccRead = BuildMI(*MBB, MI, DL, TII->get(AMDGPU::V_ACCVGPR_READ_B32), TmpReg)
            .addReg(SubReg, getKillRegState(IsKill));
          if (NeedSuperRegDef)
            AccRead.addReg(ValueReg, RegState::ImplicitDefine);
        }
        SubReg = TmpReg;
      }

      MachinePointerInfo PInfo = BasePtrInfo.getWithOffset(EltSize * i);
      MachineMemOperand *NewMMO =
          MF->getMachineMemOperand(PInfo, MMO->getFlags(), EltSize,
                                   commonAlignment(Alignment, EltSize * i));

      MIB = BuildMI(*MBB, MI, DL, *Desc)
                .addReg(SubReg,
                        getDefRegState(!IsStore) | getKillRegState(IsKill));
      if (!IsFlat)
        MIB.addReg(ScratchRsrcReg);

      if (SOffset == AMDGPU::NoRegister) {
        if (!IsFlat)
          MIB.addImm(0);
      } else {
        MIB.addReg(SOffset, SOffsetRegState);
      }
      MIB.addImm(Offset)
          .addImm(0) // glc
          .addImm(0) // slc
          .addImm(0); // tfe for MUBUF or dlc for FLAT
      if (!IsFlat)
        MIB.addImm(0) // dlc
           .addImm(0); // swz
      MIB.addMemOperand(NewMMO);

      if (!IsAGPR && NeedSuperRegDef)
        MIB.addReg(ValueReg, RegState::ImplicitDefine);

      if (!IsStore && TmpReg != AMDGPU::NoRegister)
        MIB = BuildMI(*MBB, MI, DL, TII->get(AMDGPU::V_ACCVGPR_WRITE_B32),
                      FinalReg)
          .addReg(TmpReg, RegState::Kill);
    } else {
      if (NeedSuperRegDef)
        MIB.addReg(ValueReg, RegState::ImplicitDefine);
    }

    if (NumSubRegs > 1) {
      MIB.addReg(ValueReg, RegState::Implicit | SrcDstRegState);
    }
  }

  if (ScratchOffsetRegDelta != 0) {
    // Subtract the offset we added to the ScratchOffset register.
    BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_SUB_U32), SOffset)
        .addReg(SOffset)
        .addImm(ScratchOffsetRegDelta);
  }
}

// Generate a VMEM access which loads or stores the VGPR containing an SGPR
// spill such that all the lanes set in VGPRLanes are loaded or stored.
// This generates exec mask manipulation and will use SGPRs available in MI
// or VGPR lanes in the VGPR to save and restore the exec mask.
void SIRegisterInfo::buildSGPRSpillLoadStore(MachineBasicBlock::iterator MI,
                                             int Index, int Offset,
                                             unsigned EltSize, Register VGPR,
                                             int64_t VGPRLanes,
                                             RegScavenger *RS,
                                             bool IsLoad) const {
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction *MF = MBB->getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  const SIInstrInfo *TII = ST.getInstrInfo();

  Register SuperReg = MI->getOperand(0).getReg();
  const TargetRegisterClass *RC = getPhysRegClass(SuperReg);
  ArrayRef<int16_t> SplitParts = getRegSplitParts(RC, EltSize);
  unsigned NumSubRegs = SplitParts.empty() ? 1 : SplitParts.size();
  unsigned FirstPart = Offset * 32;
  unsigned ExecLane = 0;

  bool IsKill = MI->getOperand(0).isKill();
  const DebugLoc &DL = MI->getDebugLoc();

  // Cannot handle load/store to EXEC
  assert(SuperReg != AMDGPU::EXEC_LO && SuperReg != AMDGPU::EXEC_HI &&
         SuperReg != AMDGPU::EXEC && "exec should never spill");

  // On Wave32 only handle EXEC_LO.
  // On Wave64 only update EXEC_HI if there is sufficent space for a copy.
  bool OnlyExecLo = isWave32 || NumSubRegs == 1 || SuperReg == AMDGPU::EXEC_HI;

  unsigned ExecMovOpc = OnlyExecLo ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;
  Register ExecReg = OnlyExecLo ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
  Register SavedExecReg;

  // Backup EXEC
  if (OnlyExecLo) {
    SavedExecReg = NumSubRegs == 1
                       ? SuperReg
                       : getSubReg(SuperReg, SplitParts[FirstPart + ExecLane]);
  } else {
    // If src/dst is an odd size it is possible subreg0 is not aligned.
    for (; ExecLane < (NumSubRegs - 1); ++ExecLane) {
      SavedExecReg = getMatchingSuperReg(
          getSubReg(SuperReg, SplitParts[FirstPart + ExecLane]), AMDGPU::sub0,
          &AMDGPU::SReg_64_XEXECRegClass);
      if (SavedExecReg)
        break;
    }
  }
  assert(SavedExecReg);
  BuildMI(*MBB, MI, DL, TII->get(ExecMovOpc), SavedExecReg).addReg(ExecReg);

  // Setup EXEC
  BuildMI(*MBB, MI, DL, TII->get(ExecMovOpc), ExecReg).addImm(VGPRLanes);

  // Load/store VGPR
  MachineFrameInfo &FrameInfo = MF->getFrameInfo();
  assert(FrameInfo.getStackID(Index) != TargetStackID::SGPRSpill);

  Register FrameReg = FrameInfo.isFixedObjectIndex(Index) && hasBasePointer(*MF)
                          ? getBaseRegister()
                          : getFrameRegister(*MF);

  Align Alignment = FrameInfo.getObjectAlign(Index);
  MachinePointerInfo PtrInfo =
      MachinePointerInfo::getFixedStack(*MF, Index);
  MachineMemOperand *MMO = MF->getMachineMemOperand(
      PtrInfo, IsLoad ? MachineMemOperand::MOLoad : MachineMemOperand::MOStore,
      EltSize, Alignment);

  if (IsLoad) {
    unsigned Opc = ST.enableFlatScratch() ? AMDGPU::SCRATCH_LOAD_DWORD_SADDR
                                          : AMDGPU::BUFFER_LOAD_DWORD_OFFSET;
    buildSpillLoadStore(MI, Opc,
          Index,
          VGPR, false,
          MFI->getScratchRSrcReg(), FrameReg,
          Offset * EltSize, MMO,
          RS);
  } else {
    unsigned Opc = ST.enableFlatScratch() ? AMDGPU::SCRATCH_STORE_DWORD_SADDR
                                          : AMDGPU::BUFFER_STORE_DWORD_OFFSET;
    buildSpillLoadStore(MI, Opc, Index, VGPR,
                        IsKill, MFI->getScratchRSrcReg(), FrameReg,
                        Offset * EltSize, MMO, RS);
    // This only ever adds one VGPR spill
    MFI->addToSpilledVGPRs(1);
  }

  // Restore EXEC
  BuildMI(*MBB, MI, DL, TII->get(ExecMovOpc), ExecReg)
      .addReg(SavedExecReg, getKillRegState(IsLoad || IsKill));

  // Restore clobbered SGPRs
  if (IsLoad) {
    // Nothing to do; register will be overwritten
  } else if (!IsKill) {
    // Restore SGPRs from appropriate VGPR lanes
    if (!OnlyExecLo) {
      BuildMI(*MBB, MI, DL, TII->getMCOpcodeFromPseudo(AMDGPU::V_READLANE_B32),
              getSubReg(SuperReg, SplitParts[FirstPart + ExecLane + 1]))
          .addReg(VGPR)
          .addImm(ExecLane + 1);
    }
    BuildMI(*MBB, MI, DL, TII->getMCOpcodeFromPseudo(AMDGPU::V_READLANE_B32),
            NumSubRegs == 1
                ? SavedExecReg
                : getSubReg(SuperReg, SplitParts[FirstPart + ExecLane]))
        .addReg(VGPR, RegState::Kill)
        .addImm(ExecLane);
  }
}

bool SIRegisterInfo::spillSGPR(MachineBasicBlock::iterator MI,
                               int Index,
                               RegScavenger *RS,
                               bool OnlyToVGPR) const {
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction *MF = MBB->getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  DenseSet<Register> SGPRSpillVGPRDefinedSet; // FIXME: This should be removed

  ArrayRef<SIMachineFunctionInfo::SpilledReg> VGPRSpills
    = MFI->getSGPRToVGPRSpills(Index);
  bool SpillToVGPR = !VGPRSpills.empty();
  if (OnlyToVGPR && !SpillToVGPR)
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();

  Register SuperReg = MI->getOperand(0).getReg();
  bool IsKill = MI->getOperand(0).isKill();
  const DebugLoc &DL = MI->getDebugLoc();

  assert(SpillToVGPR || (SuperReg != MFI->getStackPtrOffsetReg() &&
                         SuperReg != MFI->getFrameOffsetReg()));

  assert(SuperReg != AMDGPU::M0 && "m0 should never spill");
  assert(SuperReg != AMDGPU::EXEC_LO && SuperReg != AMDGPU::EXEC_HI &&
         SuperReg != AMDGPU::EXEC && "exec should never spill");

  unsigned EltSize = 4;
  const TargetRegisterClass *RC = getPhysRegClass(SuperReg);

  ArrayRef<int16_t> SplitParts = getRegSplitParts(RC, EltSize);
  unsigned NumSubRegs = SplitParts.empty() ? 1 : SplitParts.size();

  if (SpillToVGPR) {
    for (unsigned i = 0, e = NumSubRegs; i < e; ++i) {
      Register SubReg =
          NumSubRegs == 1 ? SuperReg : getSubReg(SuperReg, SplitParts[i]);
      SIMachineFunctionInfo::SpilledReg Spill = VGPRSpills[i];

      bool UseKill = IsKill && i == NumSubRegs - 1;

      // During SGPR spilling to VGPR, determine if the VGPR is defined. The
      // only circumstance in which we say it is undefined is when it is the
      // first spill to this VGPR in the first basic block.
      bool VGPRDefined = true;
      if (MBB == &MF->front())
        VGPRDefined = !SGPRSpillVGPRDefinedSet.insert(Spill.VGPR).second;

      // Mark the "old value of vgpr" input undef only if this is the first sgpr
      // spill to this specific vgpr in the first basic block.
      auto MIB = BuildMI(*MBB, MI, DL,
              TII->getMCOpcodeFromPseudo(AMDGPU::V_WRITELANE_B32),
              Spill.VGPR)
        .addReg(SubReg, getKillRegState(UseKill))
        .addImm(Spill.Lane)
        .addReg(Spill.VGPR, VGPRDefined ? 0 : RegState::Undef);

      if (i == 0 && NumSubRegs > 1) {
        // We may be spilling a super-register which is only partially defined,
        // and need to ensure later spills think the value is defined.
        MIB.addReg(SuperReg, RegState::ImplicitDefine);
      }

      if (NumSubRegs > 1)
        MIB.addReg(SuperReg, getKillRegState(UseKill) | RegState::Implicit);

      // FIXME: Since this spills to another register instead of an actual
      // frame index, we should delete the frame index when all references to
      // it are fixed.
    }
  } else {
    // Scavenged temporary VGPR to use. It must be scavenged once for any number
    // of spilled subregs.
    Register TmpVGPR = RS->scavengeRegister(&AMDGPU::VGPR_32RegClass, MI, 0);
    RS->setRegUsed(TmpVGPR);

    // SubReg carries the "Kill" flag when SubReg == SuperReg.
    unsigned SubKillState = getKillRegState((NumSubRegs == 1) && IsKill);

    unsigned PerVGPR = 32;
    unsigned NumVGPRs = (NumSubRegs + (PerVGPR - 1)) / PerVGPR;
    int64_t VGPRLanes = (1LL << std::min(PerVGPR, NumSubRegs)) - 1LL;

    for (unsigned Offset = 0; Offset < NumVGPRs; ++Offset) {
      unsigned TmpVGPRFlags = RegState::Undef;

      // Write sub registers into the VGPR
      for (unsigned i = Offset * PerVGPR,
                    e = std::min((Offset + 1) * PerVGPR, NumSubRegs);
           i < e; ++i) {
        Register SubReg =
            NumSubRegs == 1 ? SuperReg : getSubReg(SuperReg, SplitParts[i]);

        MachineInstrBuilder WriteLane =
            BuildMI(*MBB, MI, DL,
                    TII->getMCOpcodeFromPseudo(AMDGPU::V_WRITELANE_B32),
                    TmpVGPR)
                .addReg(SubReg, SubKillState)
                .addImm(i % PerVGPR)
                .addReg(TmpVGPR, TmpVGPRFlags);
        TmpVGPRFlags = 0;

        // There could be undef components of a spilled super register.
        // TODO: Can we detect this and skip the spill?
        if (NumSubRegs > 1) {
          // The last implicit use of the SuperReg carries the "Kill" flag.
          unsigned SuperKillState = 0;
          if (i + 1 == NumSubRegs)
            SuperKillState |= getKillRegState(IsKill);
          WriteLane.addReg(SuperReg, RegState::Implicit | SuperKillState);
        }
      }

      // Write out VGPR
      buildSGPRSpillLoadStore(MI, Index, Offset, EltSize, TmpVGPR, VGPRLanes,
                              RS, false);
    }
  }

  MI->eraseFromParent();
  MFI->addToSpilledSGPRs(NumSubRegs);
  return true;
}

bool SIRegisterInfo::restoreSGPR(MachineBasicBlock::iterator MI,
                                 int Index,
                                 RegScavenger *RS,
                                 bool OnlyToVGPR) const {
  MachineFunction *MF = MI->getParent()->getParent();
  MachineBasicBlock *MBB = MI->getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();

  ArrayRef<SIMachineFunctionInfo::SpilledReg> VGPRSpills
    = MFI->getSGPRToVGPRSpills(Index);
  bool SpillToVGPR = !VGPRSpills.empty();
  if (OnlyToVGPR && !SpillToVGPR)
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();
  const DebugLoc &DL = MI->getDebugLoc();

  Register SuperReg = MI->getOperand(0).getReg();

  assert(SuperReg != AMDGPU::M0 && "m0 should never spill");
  assert(SuperReg != AMDGPU::EXEC_LO && SuperReg != AMDGPU::EXEC_HI &&
         SuperReg != AMDGPU::EXEC && "exec should never spill");

  unsigned EltSize = 4;

  const TargetRegisterClass *RC = getPhysRegClass(SuperReg);

  ArrayRef<int16_t> SplitParts = getRegSplitParts(RC, EltSize);
  unsigned NumSubRegs = SplitParts.empty() ? 1 : SplitParts.size();

  if (SpillToVGPR) {
    for (unsigned i = 0, e = NumSubRegs; i < e; ++i) {
      Register SubReg =
          NumSubRegs == 1 ? SuperReg : getSubReg(SuperReg, SplitParts[i]);

      SIMachineFunctionInfo::SpilledReg Spill = VGPRSpills[i];
      auto MIB =
        BuildMI(*MBB, MI, DL, TII->getMCOpcodeFromPseudo(AMDGPU::V_READLANE_B32),
                SubReg)
        .addReg(Spill.VGPR)
        .addImm(Spill.Lane);
      if (NumSubRegs > 1 && i == 0)
        MIB.addReg(SuperReg, RegState::ImplicitDefine);
    }
  } else {
    Register TmpVGPR = RS->scavengeRegister(&AMDGPU::VGPR_32RegClass, MI, 0);
    RS->setRegUsed(TmpVGPR);

    unsigned PerVGPR = 32;
    unsigned NumVGPRs = (NumSubRegs + (PerVGPR - 1)) / PerVGPR;
    int64_t VGPRLanes = (1LL << std::min(PerVGPR, NumSubRegs)) - 1LL;

    for (unsigned Offset = 0; Offset < NumVGPRs; ++Offset) {
      // Load in VGPR data
      buildSGPRSpillLoadStore(MI, Index, Offset, EltSize, TmpVGPR, VGPRLanes,
                              RS, true);

      // Unpack lanes
      for (unsigned i = Offset * PerVGPR,
                    e = std::min((Offset + 1) * PerVGPR, NumSubRegs);
           i < e; ++i) {
        Register SubReg =
            NumSubRegs == 1 ? SuperReg : getSubReg(SuperReg, SplitParts[i]);

        bool LastSubReg = (i + 1 == e);
        auto MIB =
            BuildMI(*MBB, MI, DL,
                    TII->getMCOpcodeFromPseudo(AMDGPU::V_READLANE_B32), SubReg)
                .addReg(TmpVGPR, getKillRegState(LastSubReg))
                .addImm(i);
        if (NumSubRegs > 1 && i == 0)
          MIB.addReg(SuperReg, RegState::ImplicitDefine);
      }
    }
  }

  MI->eraseFromParent();
  return true;
}

/// Special case of eliminateFrameIndex. Returns true if the SGPR was spilled to
/// a VGPR and the stack slot can be safely eliminated when all other users are
/// handled.
bool SIRegisterInfo::eliminateSGPRToVGPRSpillFrameIndex(
  MachineBasicBlock::iterator MI,
  int FI,
  RegScavenger *RS) const {
  switch (MI->getOpcode()) {
  case AMDGPU::SI_SPILL_S1024_SAVE:
  case AMDGPU::SI_SPILL_S512_SAVE:
  case AMDGPU::SI_SPILL_S256_SAVE:
  case AMDGPU::SI_SPILL_S192_SAVE:
  case AMDGPU::SI_SPILL_S160_SAVE:
  case AMDGPU::SI_SPILL_S128_SAVE:
  case AMDGPU::SI_SPILL_S96_SAVE:
  case AMDGPU::SI_SPILL_S64_SAVE:
  case AMDGPU::SI_SPILL_S32_SAVE:
    return spillSGPR(MI, FI, RS, true);
  case AMDGPU::SI_SPILL_S1024_RESTORE:
  case AMDGPU::SI_SPILL_S512_RESTORE:
  case AMDGPU::SI_SPILL_S256_RESTORE:
  case AMDGPU::SI_SPILL_S192_RESTORE:
  case AMDGPU::SI_SPILL_S160_RESTORE:
  case AMDGPU::SI_SPILL_S128_RESTORE:
  case AMDGPU::SI_SPILL_S96_RESTORE:
  case AMDGPU::SI_SPILL_S64_RESTORE:
  case AMDGPU::SI_SPILL_S32_RESTORE:
    return restoreSGPR(MI, FI, RS, true);
  default:
    llvm_unreachable("not an SGPR spill instruction");
  }
}

void SIRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                        int SPAdj, unsigned FIOperandNum,
                                        RegScavenger *RS) const {
  MachineFunction *MF = MI->getParent()->getParent();
  MachineBasicBlock *MBB = MI->getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  MachineFrameInfo &FrameInfo = MF->getFrameInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  DebugLoc DL = MI->getDebugLoc();

  assert(SPAdj == 0 && "unhandled SP adjustment in call sequence?");

  MachineOperand &FIOp = MI->getOperand(FIOperandNum);
  int Index = MI->getOperand(FIOperandNum).getIndex();

  Register FrameReg = FrameInfo.isFixedObjectIndex(Index) && hasBasePointer(*MF)
                          ? getBaseRegister()
                          : getFrameRegister(*MF);

  switch (MI->getOpcode()) {
    // SGPR register spill
    case AMDGPU::SI_SPILL_S1024_SAVE:
    case AMDGPU::SI_SPILL_S512_SAVE:
    case AMDGPU::SI_SPILL_S256_SAVE:
    case AMDGPU::SI_SPILL_S192_SAVE:
    case AMDGPU::SI_SPILL_S160_SAVE:
    case AMDGPU::SI_SPILL_S128_SAVE:
    case AMDGPU::SI_SPILL_S96_SAVE:
    case AMDGPU::SI_SPILL_S64_SAVE:
    case AMDGPU::SI_SPILL_S32_SAVE: {
      spillSGPR(MI, Index, RS);
      break;
    }

    // SGPR register restore
    case AMDGPU::SI_SPILL_S1024_RESTORE:
    case AMDGPU::SI_SPILL_S512_RESTORE:
    case AMDGPU::SI_SPILL_S256_RESTORE:
    case AMDGPU::SI_SPILL_S192_RESTORE:
    case AMDGPU::SI_SPILL_S160_RESTORE:
    case AMDGPU::SI_SPILL_S128_RESTORE:
    case AMDGPU::SI_SPILL_S96_RESTORE:
    case AMDGPU::SI_SPILL_S64_RESTORE:
    case AMDGPU::SI_SPILL_S32_RESTORE: {
      restoreSGPR(MI, Index, RS);
      break;
    }

    // VGPR register spill
    case AMDGPU::SI_SPILL_V1024_SAVE:
    case AMDGPU::SI_SPILL_V512_SAVE:
    case AMDGPU::SI_SPILL_V256_SAVE:
    case AMDGPU::SI_SPILL_V160_SAVE:
    case AMDGPU::SI_SPILL_V128_SAVE:
    case AMDGPU::SI_SPILL_V96_SAVE:
    case AMDGPU::SI_SPILL_V64_SAVE:
    case AMDGPU::SI_SPILL_V32_SAVE:
    case AMDGPU::SI_SPILL_A1024_SAVE:
    case AMDGPU::SI_SPILL_A512_SAVE:
    case AMDGPU::SI_SPILL_A256_SAVE:
    case AMDGPU::SI_SPILL_A192_SAVE:
    case AMDGPU::SI_SPILL_A160_SAVE:
    case AMDGPU::SI_SPILL_A128_SAVE:
    case AMDGPU::SI_SPILL_A96_SAVE:
    case AMDGPU::SI_SPILL_A64_SAVE:
    case AMDGPU::SI_SPILL_A32_SAVE: {
      const MachineOperand *VData = TII->getNamedOperand(*MI,
                                                         AMDGPU::OpName::vdata);
      assert(TII->getNamedOperand(*MI, AMDGPU::OpName::soffset)->getReg() ==
             MFI->getStackPtrOffsetReg());

      unsigned Opc = ST.enableFlatScratch() ? AMDGPU::SCRATCH_STORE_DWORD_SADDR
                                            : AMDGPU::BUFFER_STORE_DWORD_OFFSET;
      buildSpillLoadStore(MI, Opc,
            Index,
            VData->getReg(), VData->isKill(),
            TII->getNamedOperand(*MI, AMDGPU::OpName::srsrc)->getReg(),
            FrameReg,
            TII->getNamedOperand(*MI, AMDGPU::OpName::offset)->getImm(),
            *MI->memoperands_begin(),
            RS);
      MFI->addToSpilledVGPRs(getNumSubRegsForSpillOp(MI->getOpcode()));
      MI->eraseFromParent();
      break;
    }
    case AMDGPU::SI_SPILL_V32_RESTORE:
    case AMDGPU::SI_SPILL_V64_RESTORE:
    case AMDGPU::SI_SPILL_V96_RESTORE:
    case AMDGPU::SI_SPILL_V128_RESTORE:
    case AMDGPU::SI_SPILL_V160_RESTORE:
    case AMDGPU::SI_SPILL_V256_RESTORE:
    case AMDGPU::SI_SPILL_V512_RESTORE:
    case AMDGPU::SI_SPILL_V1024_RESTORE:
    case AMDGPU::SI_SPILL_A32_RESTORE:
    case AMDGPU::SI_SPILL_A64_RESTORE:
    case AMDGPU::SI_SPILL_A96_RESTORE:
    case AMDGPU::SI_SPILL_A128_RESTORE:
    case AMDGPU::SI_SPILL_A160_RESTORE:
    case AMDGPU::SI_SPILL_A192_RESTORE:
    case AMDGPU::SI_SPILL_A256_RESTORE:
    case AMDGPU::SI_SPILL_A512_RESTORE:
    case AMDGPU::SI_SPILL_A1024_RESTORE: {
      const MachineOperand *VData = TII->getNamedOperand(*MI,
                                                         AMDGPU::OpName::vdata);
      assert(TII->getNamedOperand(*MI, AMDGPU::OpName::soffset)->getReg() ==
             MFI->getStackPtrOffsetReg());

      unsigned Opc = ST.enableFlatScratch() ? AMDGPU::SCRATCH_LOAD_DWORD_SADDR
                                            : AMDGPU::BUFFER_LOAD_DWORD_OFFSET;
      buildSpillLoadStore(MI, Opc,
            Index,
            VData->getReg(), VData->isKill(),
            TII->getNamedOperand(*MI, AMDGPU::OpName::srsrc)->getReg(),
            FrameReg,
            TII->getNamedOperand(*MI, AMDGPU::OpName::offset)->getImm(),
            *MI->memoperands_begin(),
            RS);
      MI->eraseFromParent();
      break;
    }

    default: {
      const DebugLoc &DL = MI->getDebugLoc();

      int64_t Offset = FrameInfo.getObjectOffset(Index);
      if (ST.enableFlatScratch()) {
        if (TII->isFLATScratch(*MI)) {
          // The offset is always swizzled, just replace it
          if (FrameReg)
            FIOp.ChangeToRegister(FrameReg, false);

          if (!Offset)
            return;

          MachineOperand *OffsetOp =
            TII->getNamedOperand(*MI, AMDGPU::OpName::offset);
          int64_t NewOffset = Offset + OffsetOp->getImm();
          if (TII->isLegalFLATOffset(NewOffset, AMDGPUAS::PRIVATE_ADDRESS,
                                     true)) {
            OffsetOp->setImm(NewOffset);
            if (FrameReg)
              return;
            Offset = 0;
          }

          assert(!TII->getNamedOperand(*MI, AMDGPU::OpName::vaddr) &&
                 "Unexpected vaddr for flat scratch with a FI operand");

          // On GFX10 we have ST mode to use no registers for an address.
          // Otherwise we need to materialize 0 into an SGPR.
          if (!Offset && ST.hasFlatScratchSTMode()) {
            unsigned Opc = MI->getOpcode();
            unsigned NewOpc = AMDGPU::getFlatScratchInstSTfromSS(Opc);
            MI->RemoveOperand(
                AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::saddr));
            MI->setDesc(TII->get(NewOpc));
            return;
          }
        }

        if (!FrameReg) {
          FIOp.ChangeToImmediate(Offset);
          if (TII->isImmOperandLegal(*MI, FIOperandNum, FIOp))
            return;
        }

        // We need to use register here. Check if we can use an SGPR or need
        // a VGPR.
        FIOp.ChangeToRegister(AMDGPU::M0, false);
        bool UseSGPR = TII->isOperandLegal(*MI, FIOperandNum, &FIOp);

        if (!Offset && FrameReg && UseSGPR) {
          FIOp.setReg(FrameReg);
          return;
        }

        const TargetRegisterClass *RC = UseSGPR ? &AMDGPU::SReg_32_XM0RegClass
                                                : &AMDGPU::VGPR_32RegClass;

        Register TmpReg = RS->scavengeRegister(RC, MI, 0, !UseSGPR);
        FIOp.setReg(TmpReg);
        FIOp.setIsKill(true);

        if ((!FrameReg || !Offset) && TmpReg) {
          unsigned Opc = UseSGPR ? AMDGPU::S_MOV_B32 : AMDGPU::V_MOV_B32_e32;
          auto MIB = BuildMI(*MBB, MI, DL, TII->get(Opc), TmpReg);
          if (FrameReg)
            MIB.addReg(FrameReg);
          else
            MIB.addImm(Offset);

          return;
        }

        Register TmpSReg =
            UseSGPR ? TmpReg
                    : RS->scavengeRegister(&AMDGPU::SReg_32_XM0RegClass, MI, 0,
                                           !UseSGPR);

        // TODO: for flat scratch another attempt can be made with a VGPR index
        //       if no SGPRs can be scavenged.
        if ((!TmpSReg && !FrameReg) || (!TmpReg && !UseSGPR))
          report_fatal_error("Cannot scavenge register in FI elimination!");

        if (!TmpSReg) {
          // Use frame register and restore it after.
          TmpSReg = FrameReg;
          FIOp.setReg(FrameReg);
          FIOp.setIsKill(false);
        }

        BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_ADD_U32), TmpSReg)
          .addReg(FrameReg)
          .addImm(Offset);

        if (!UseSGPR)
          BuildMI(*MBB, MI, DL, TII->get(AMDGPU::V_MOV_B32_e32), TmpReg)
            .addReg(TmpSReg, RegState::Kill);

        if (TmpSReg == FrameReg) {
          // Undo frame register modification.
          BuildMI(*MBB, std::next(MI), DL, TII->get(AMDGPU::S_SUB_U32),
                  FrameReg)
            .addReg(FrameReg)
            .addImm(Offset);
        }

        return;
      }

      bool IsMUBUF = TII->isMUBUF(*MI);

      if (!IsMUBUF && !MFI->isEntryFunction()) {
        // Convert to a swizzled stack address by scaling by the wave size.
        //
        // In an entry function/kernel the offset is already swizzled.

        bool IsCopy = MI->getOpcode() == AMDGPU::V_MOV_B32_e32;
        Register ResultReg =
            IsCopy ? MI->getOperand(0).getReg()
                   : RS->scavengeRegister(&AMDGPU::VGPR_32RegClass, MI, 0);

        int64_t Offset = FrameInfo.getObjectOffset(Index);
        if (Offset == 0) {
          // XXX - This never happens because of emergency scavenging slot at 0?
          BuildMI(*MBB, MI, DL, TII->get(AMDGPU::V_LSHRREV_B32_e64), ResultReg)
            .addImm(ST.getWavefrontSizeLog2())
            .addReg(FrameReg);
        } else {
          if (auto MIB = TII->getAddNoCarry(*MBB, MI, DL, ResultReg, *RS)) {
            // Reuse ResultReg in intermediate step.
            Register ScaledReg = ResultReg;

            BuildMI(*MBB, *MIB, DL, TII->get(AMDGPU::V_LSHRREV_B32_e64),
                    ScaledReg)
              .addImm(ST.getWavefrontSizeLog2())
              .addReg(FrameReg);

            const bool IsVOP2 = MIB->getOpcode() == AMDGPU::V_ADD_U32_e32;

            // TODO: Fold if use instruction is another add of a constant.
            if (IsVOP2 || AMDGPU::isInlinableLiteral32(Offset, ST.hasInv2PiInlineImm())) {
              // FIXME: This can fail
              MIB.addImm(Offset);
              MIB.addReg(ScaledReg, RegState::Kill);
              if (!IsVOP2)
                MIB.addImm(0); // clamp bit
            } else {
              assert(MIB->getOpcode() == AMDGPU::V_ADD_CO_U32_e64 &&
                     "Need to reuse carry out register");

              // Use scavenged unused carry out as offset register.
              Register ConstOffsetReg;
              if (!isWave32)
                ConstOffsetReg = getSubReg(MIB.getReg(1), AMDGPU::sub0);
              else
                ConstOffsetReg = MIB.getReg(1);

              BuildMI(*MBB, *MIB, DL, TII->get(AMDGPU::S_MOV_B32), ConstOffsetReg)
                .addImm(Offset);
              MIB.addReg(ConstOffsetReg, RegState::Kill);
              MIB.addReg(ScaledReg, RegState::Kill);
              MIB.addImm(0); // clamp bit
            }
          } else {
            // We have to produce a carry out, and there isn't a free SGPR pair
            // for it. We can keep the whole computation on the SALU to avoid
            // clobbering an additional register at the cost of an extra mov.

            // We may have 1 free scratch SGPR even though a carry out is
            // unavailable. Only one additional mov is needed.
            Register TmpScaledReg =
                RS->scavengeRegister(&AMDGPU::SReg_32_XM0RegClass, MI, 0, false);
            Register ScaledReg = TmpScaledReg.isValid() ? TmpScaledReg : FrameReg;

            BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_LSHR_B32), ScaledReg)
              .addReg(FrameReg)
              .addImm(ST.getWavefrontSizeLog2());
            BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_ADD_U32), ScaledReg)
              .addReg(ScaledReg, RegState::Kill)
              .addImm(Offset);
            BuildMI(*MBB, MI, DL, TII->get(AMDGPU::COPY), ResultReg)
              .addReg(ScaledReg, RegState::Kill);

            // If there were truly no free SGPRs, we need to undo everything.
            if (!TmpScaledReg.isValid()) {
              BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_SUB_U32), ScaledReg)
                .addReg(ScaledReg, RegState::Kill)
                .addImm(Offset);
              BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_LSHL_B32), ScaledReg)
                .addReg(FrameReg)
                .addImm(ST.getWavefrontSizeLog2());
            }
          }
        }

        // Don't introduce an extra copy if we're just materializing in a mov.
        if (IsCopy)
          MI->eraseFromParent();
        else
          FIOp.ChangeToRegister(ResultReg, false, false, true);
        return;
      }

      if (IsMUBUF) {
        // Disable offen so we don't need a 0 vgpr base.
        assert(static_cast<int>(FIOperandNum) ==
               AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                          AMDGPU::OpName::vaddr));

        auto &SOffset = *TII->getNamedOperand(*MI, AMDGPU::OpName::soffset);
        assert((SOffset.isReg() &&
                SOffset.getReg() == MFI->getStackPtrOffsetReg()) ||
               (SOffset.isImm() && SOffset.getImm() == 0));
        if (SOffset.isReg()) {
          if (FrameReg == AMDGPU::NoRegister) {
            SOffset.ChangeToImmediate(0);
          } else {
            SOffset.setReg(FrameReg);
          }
        } else if (SOffset.isImm() && FrameReg != AMDGPU::NoRegister) {
          SOffset.ChangeToRegister(FrameReg, false);
        }

        int64_t Offset = FrameInfo.getObjectOffset(Index);
        int64_t OldImm
          = TII->getNamedOperand(*MI, AMDGPU::OpName::offset)->getImm();
        int64_t NewOffset = OldImm + Offset;

        if (SIInstrInfo::isLegalMUBUFImmOffset(NewOffset) &&
            buildMUBUFOffsetLoadStore(ST, FrameInfo, MI, Index, NewOffset)) {
          MI->eraseFromParent();
          return;
        }
      }

      // If the offset is simply too big, don't convert to a scratch wave offset
      // relative index.

      FIOp.ChangeToImmediate(Offset);
      if (!TII->isImmOperandLegal(*MI, FIOperandNum, FIOp)) {
        Register TmpReg = RS->scavengeRegister(&AMDGPU::VGPR_32RegClass, MI, 0);
        BuildMI(*MBB, MI, DL, TII->get(AMDGPU::V_MOV_B32_e32), TmpReg)
          .addImm(Offset);
        FIOp.ChangeToRegister(TmpReg, false, false, true);
      }
    }
  }
}

StringRef SIRegisterInfo::getRegAsmName(MCRegister Reg) const {
  return AMDGPUInstPrinter::getRegisterName(Reg);
}

const TargetRegisterClass *
SIRegisterInfo::getVGPRClassForBitWidth(unsigned BitWidth) {
  if (BitWidth == 1)
    return &AMDGPU::VReg_1RegClass;
  if (BitWidth <= 16)
    return &AMDGPU::VGPR_LO16RegClass;
  if (BitWidth <= 32)
    return &AMDGPU::VGPR_32RegClass;
  if (BitWidth <= 64)
    return &AMDGPU::VReg_64RegClass;
  if (BitWidth <= 96)
    return &AMDGPU::VReg_96RegClass;
  if (BitWidth <= 128)
    return &AMDGPU::VReg_128RegClass;
  if (BitWidth <= 160)
    return &AMDGPU::VReg_160RegClass;
  if (BitWidth <= 192)
    return &AMDGPU::VReg_192RegClass;
  if (BitWidth <= 256)
    return &AMDGPU::VReg_256RegClass;
  if (BitWidth <= 512)
    return &AMDGPU::VReg_512RegClass;
  if (BitWidth <= 1024)
    return &AMDGPU::VReg_1024RegClass;

  return nullptr;
}

const TargetRegisterClass *
SIRegisterInfo::getAGPRClassForBitWidth(unsigned BitWidth) {
  if (BitWidth <= 16)
    return &AMDGPU::AGPR_LO16RegClass;
  if (BitWidth <= 32)
    return &AMDGPU::AGPR_32RegClass;
  if (BitWidth <= 64)
    return &AMDGPU::AReg_64RegClass;
  if (BitWidth <= 96)
    return &AMDGPU::AReg_96RegClass;
  if (BitWidth <= 128)
    return &AMDGPU::AReg_128RegClass;
  if (BitWidth <= 160)
    return &AMDGPU::AReg_160RegClass;
  if (BitWidth <= 192)
    return &AMDGPU::AReg_192RegClass;
  if (BitWidth <= 256)
    return &AMDGPU::AReg_256RegClass;
  if (BitWidth <= 512)
    return &AMDGPU::AReg_512RegClass;
  if (BitWidth <= 1024)
    return &AMDGPU::AReg_1024RegClass;

  return nullptr;
}

const TargetRegisterClass *
SIRegisterInfo::getSGPRClassForBitWidth(unsigned BitWidth) {
  if (BitWidth <= 16)
    return &AMDGPU::SGPR_LO16RegClass;
  if (BitWidth <= 32)
    return &AMDGPU::SReg_32RegClass;
  if (BitWidth <= 64)
    return &AMDGPU::SReg_64RegClass;
  if (BitWidth <= 96)
    return &AMDGPU::SGPR_96RegClass;
  if (BitWidth <= 128)
    return &AMDGPU::SGPR_128RegClass;
  if (BitWidth <= 160)
    return &AMDGPU::SGPR_160RegClass;
  if (BitWidth <= 192)
    return &AMDGPU::SGPR_192RegClass;
  if (BitWidth <= 256)
    return &AMDGPU::SGPR_256RegClass;
  if (BitWidth <= 512)
    return &AMDGPU::SGPR_512RegClass;
  if (BitWidth <= 1024)
    return &AMDGPU::SGPR_1024RegClass;

  return nullptr;
}

// FIXME: This is very slow. It might be worth creating a map from physreg to
// register class.
const TargetRegisterClass *
SIRegisterInfo::getPhysRegClass(MCRegister Reg) const {
  static const TargetRegisterClass *const BaseClasses[] = {
    &AMDGPU::VGPR_LO16RegClass,
    &AMDGPU::VGPR_HI16RegClass,
    &AMDGPU::SReg_LO16RegClass,
    &AMDGPU::AGPR_LO16RegClass,
    &AMDGPU::VGPR_32RegClass,
    &AMDGPU::SReg_32RegClass,
    &AMDGPU::AGPR_32RegClass,
    &AMDGPU::VReg_64RegClass,
    &AMDGPU::SReg_64RegClass,
    &AMDGPU::AReg_64RegClass,
    &AMDGPU::VReg_96RegClass,
    &AMDGPU::SReg_96RegClass,
    &AMDGPU::AReg_96RegClass,
    &AMDGPU::VReg_128RegClass,
    &AMDGPU::SReg_128RegClass,
    &AMDGPU::AReg_128RegClass,
    &AMDGPU::VReg_160RegClass,
    &AMDGPU::SReg_160RegClass,
    &AMDGPU::AReg_160RegClass,
    &AMDGPU::VReg_192RegClass,
    &AMDGPU::SReg_192RegClass,
    &AMDGPU::AReg_192RegClass,
    &AMDGPU::VReg_256RegClass,
    &AMDGPU::SReg_256RegClass,
    &AMDGPU::AReg_256RegClass,
    &AMDGPU::VReg_512RegClass,
    &AMDGPU::SReg_512RegClass,
    &AMDGPU::AReg_512RegClass,
    &AMDGPU::SReg_1024RegClass,
    &AMDGPU::VReg_1024RegClass,
    &AMDGPU::AReg_1024RegClass,
    &AMDGPU::SCC_CLASSRegClass,
    &AMDGPU::Pseudo_SReg_32RegClass,
    &AMDGPU::Pseudo_SReg_128RegClass,
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
  unsigned Size = getRegSizeInBits(*RC);
  if (Size == 16) {
    return getCommonSubClass(&AMDGPU::VGPR_LO16RegClass, RC) != nullptr ||
           getCommonSubClass(&AMDGPU::VGPR_HI16RegClass, RC) != nullptr;
  }
  const TargetRegisterClass *VRC = getVGPRClassForBitWidth(Size);
  if (!VRC) {
    assert(Size < 32 && "Invalid register class size");
    return false;
  }
  return getCommonSubClass(VRC, RC) != nullptr;
}

bool SIRegisterInfo::hasAGPRs(const TargetRegisterClass *RC) const {
  unsigned Size = getRegSizeInBits(*RC);
  if (Size < 16)
    return false;
  const TargetRegisterClass *ARC = getAGPRClassForBitWidth(Size);
  if (!ARC) {
    assert(getVGPRClassForBitWidth(Size) && "Invalid register class size");
    return false;
  }
  return getCommonSubClass(ARC, RC) != nullptr;
}

const TargetRegisterClass *
SIRegisterInfo::getEquivalentVGPRClass(const TargetRegisterClass *SRC) const {
  unsigned Size = getRegSizeInBits(*SRC);
  const TargetRegisterClass *VRC = getVGPRClassForBitWidth(Size);
  assert(VRC && "Invalid register class size");
  return VRC;
}

const TargetRegisterClass *
SIRegisterInfo::getEquivalentAGPRClass(const TargetRegisterClass *SRC) const {
  unsigned Size = getRegSizeInBits(*SRC);
  const TargetRegisterClass *ARC = getAGPRClassForBitWidth(Size);
  assert(ARC && "Invalid register class size");
  return ARC;
}

const TargetRegisterClass *
SIRegisterInfo::getEquivalentSGPRClass(const TargetRegisterClass *VRC) const {
  unsigned Size = getRegSizeInBits(*VRC);
  if (Size == 32)
    return &AMDGPU::SGPR_32RegClass;
  const TargetRegisterClass *SRC = getSGPRClassForBitWidth(Size);
  assert(SRC && "Invalid register class size");
  return SRC;
}

const TargetRegisterClass *SIRegisterInfo::getSubRegClass(
                         const TargetRegisterClass *RC, unsigned SubIdx) const {
  if (SubIdx == AMDGPU::NoSubRegister)
    return RC;

  // We can assume that each lane corresponds to one 32-bit register.
  unsigned Size = getNumChannelsFromSubReg(SubIdx) * 32;
  if (isSGPRClass(RC)) {
    if (Size == 32)
      RC = &AMDGPU::SGPR_32RegClass;
    else
      RC = getSGPRClassForBitWidth(Size);
  } else if (hasAGPRs(RC)) {
    RC = getAGPRClassForBitWidth(Size);
  } else {
    RC = getVGPRClassForBitWidth(Size);
  }
  assert(RC && "Invalid sub-register class size");
  return RC;
}

bool SIRegisterInfo::opCanUseInlineConstant(unsigned OpType) const {
  if (OpType >= AMDGPU::OPERAND_REG_INLINE_AC_FIRST &&
      OpType <= AMDGPU::OPERAND_REG_INLINE_AC_LAST)
    return !ST.hasMFMAInlineLiteralBug();

  return OpType >= AMDGPU::OPERAND_SRC_FIRST &&
         OpType <= AMDGPU::OPERAND_SRC_LAST;
}

bool SIRegisterInfo::shouldRewriteCopySrc(
  const TargetRegisterClass *DefRC,
  unsigned DefSubReg,
  const TargetRegisterClass *SrcRC,
  unsigned SrcSubReg) const {
  // We want to prefer the smallest register class possible, so we don't want to
  // stop and rewrite on anything that looks like a subregister
  // extract. Operations mostly don't care about the super register class, so we
  // only want to stop on the most basic of copies between the same register
  // class.
  //
  // e.g. if we have something like
  // %0 = ...
  // %1 = ...
  // %2 = REG_SEQUENCE %0, sub0, %1, sub1, %2, sub2
  // %3 = COPY %2, sub0
  //
  // We want to look through the COPY to find:
  //  => %3 = COPY %0

  // Plain copy.
  return getCommonSubClass(DefRC, SrcRC) != nullptr;
}

/// Returns a lowest register that is not used at any point in the function.
///        If all registers are used, then this function will return
///         AMDGPU::NoRegister. If \p ReserveHighestVGPR = true, then return
///         highest unused register.
MCRegister SIRegisterInfo::findUnusedRegister(const MachineRegisterInfo &MRI,
                                              const TargetRegisterClass *RC,
                                              const MachineFunction &MF,
                                              bool ReserveHighestVGPR) const {
  if (ReserveHighestVGPR) {
    for (MCRegister Reg : reverse(*RC))
      if (MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg))
        return Reg;
  } else {
    for (MCRegister Reg : *RC)
      if (MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg))
        return Reg;
  }
  return MCRegister();
}

ArrayRef<int16_t> SIRegisterInfo::getRegSplitParts(const TargetRegisterClass *RC,
                                                   unsigned EltSize) const {
  const unsigned RegBitWidth = AMDGPU::getRegBitWidth(*RC->MC);
  assert(RegBitWidth >= 32 && RegBitWidth <= 1024);

  const unsigned RegDWORDs = RegBitWidth / 32;
  const unsigned EltDWORDs = EltSize / 4;
  assert(RegSplitParts.size() + 1 >= EltDWORDs);

  const std::vector<int16_t> &Parts = RegSplitParts[EltDWORDs - 1];
  const unsigned NumParts = RegDWORDs / EltDWORDs;

  return makeArrayRef(Parts.data(), NumParts);
}

const TargetRegisterClass*
SIRegisterInfo::getRegClassForReg(const MachineRegisterInfo &MRI,
                                  Register Reg) const {
  return Reg.isVirtual() ? MRI.getRegClass(Reg) : getPhysRegClass(Reg);
}

bool SIRegisterInfo::isVGPR(const MachineRegisterInfo &MRI,
                            Register Reg) const {
  const TargetRegisterClass *RC = getRegClassForReg(MRI, Reg);
  // Registers without classes are unaddressable, SGPR-like registers.
  return RC && hasVGPRs(RC);
}

bool SIRegisterInfo::isAGPR(const MachineRegisterInfo &MRI,
                            Register Reg) const {
  const TargetRegisterClass *RC = getRegClassForReg(MRI, Reg);

  // Registers without classes are unaddressable, SGPR-like registers.
  return RC && hasAGPRs(RC);
}

bool SIRegisterInfo::shouldCoalesce(MachineInstr *MI,
                                    const TargetRegisterClass *SrcRC,
                                    unsigned SubReg,
                                    const TargetRegisterClass *DstRC,
                                    unsigned DstSubReg,
                                    const TargetRegisterClass *NewRC,
                                    LiveIntervals &LIS) const {
  unsigned SrcSize = getRegSizeInBits(*SrcRC);
  unsigned DstSize = getRegSizeInBits(*DstRC);
  unsigned NewSize = getRegSizeInBits(*NewRC);

  // Do not increase size of registers beyond dword, we would need to allocate
  // adjacent registers and constraint regalloc more than needed.

  // Always allow dword coalescing.
  if (SrcSize <= 32 || DstSize <= 32)
    return true;

  return NewSize <= DstSize || NewSize <= SrcSize;
}

unsigned SIRegisterInfo::getRegPressureLimit(const TargetRegisterClass *RC,
                                             MachineFunction &MF) const {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  unsigned Occupancy = ST.getOccupancyWithLocalMemSize(MFI->getLDSSize(),
                                                       MF.getFunction());
  switch (RC->getID()) {
  default:
    return AMDGPUGenRegisterInfo::getRegPressureLimit(RC, MF);
  case AMDGPU::VGPR_32RegClassID:
  case AMDGPU::VGPR_LO16RegClassID:
  case AMDGPU::VGPR_HI16RegClassID:
    return std::min(ST.getMaxNumVGPRs(Occupancy), ST.getMaxNumVGPRs(MF));
  case AMDGPU::SGPR_32RegClassID:
  case AMDGPU::SGPR_LO16RegClassID:
    return std::min(ST.getMaxNumSGPRs(Occupancy, true), ST.getMaxNumSGPRs(MF));
  }
}

unsigned SIRegisterInfo::getRegPressureSetLimit(const MachineFunction &MF,
                                                unsigned Idx) const {
  if (Idx == AMDGPU::RegisterPressureSets::VGPR_32 ||
      Idx == AMDGPU::RegisterPressureSets::AGPR_32)
    return getRegPressureLimit(&AMDGPU::VGPR_32RegClass,
                               const_cast<MachineFunction &>(MF));

  if (Idx == AMDGPU::RegisterPressureSets::SReg_32)
    return getRegPressureLimit(&AMDGPU::SGPR_32RegClass,
                               const_cast<MachineFunction &>(MF));

  llvm_unreachable("Unexpected register pressure set!");
}

const int *SIRegisterInfo::getRegUnitPressureSets(unsigned RegUnit) const {
  static const int Empty[] = { -1 };

  if (RegPressureIgnoredUnits[RegUnit])
    return Empty;

  return AMDGPUGenRegisterInfo::getRegUnitPressureSets(RegUnit);
}

MCRegister SIRegisterInfo::getReturnAddressReg(const MachineFunction &MF) const {
  // Not a callee saved register.
  return AMDGPU::SGPR30_SGPR31;
}

const TargetRegisterClass *
SIRegisterInfo::getRegClassForSizeOnBank(unsigned Size,
                                         const RegisterBank &RB,
                                         const MachineRegisterInfo &MRI) const {
  switch (RB.getID()) {
  case AMDGPU::VGPRRegBankID:
    return getVGPRClassForBitWidth(std::max(32u, Size));
  case AMDGPU::VCCRegBankID:
    assert(Size == 1);
    return isWave32 ? &AMDGPU::SReg_32_XM0_XEXECRegClass
                    : &AMDGPU::SReg_64_XEXECRegClass;
  case AMDGPU::SGPRRegBankID:
    return getSGPRClassForBitWidth(std::max(32u, Size));
  case AMDGPU::AGPRRegBankID:
    return getAGPRClassForBitWidth(std::max(32u, Size));
  default:
    llvm_unreachable("unknown register bank");
  }
}

const TargetRegisterClass *
SIRegisterInfo::getConstrainedRegClassForOperand(const MachineOperand &MO,
                                         const MachineRegisterInfo &MRI) const {
  const RegClassOrRegBank &RCOrRB = MRI.getRegClassOrRegBank(MO.getReg());
  if (const RegisterBank *RB = RCOrRB.dyn_cast<const RegisterBank*>())
    return getRegClassForTypeOnBank(MRI.getType(MO.getReg()), *RB, MRI);

  const TargetRegisterClass *RC = RCOrRB.get<const TargetRegisterClass*>();
  return getAllocatableClass(RC);
}

MCRegister SIRegisterInfo::getVCC() const {
  return isWave32 ? AMDGPU::VCC_LO : AMDGPU::VCC;
}

const TargetRegisterClass *
SIRegisterInfo::getRegClass(unsigned RCID) const {
  switch ((int)RCID) {
  case AMDGPU::SReg_1RegClassID:
    return getBoolRC();
  case AMDGPU::SReg_1_XEXECRegClassID:
    return isWave32 ? &AMDGPU::SReg_32_XM0_XEXECRegClass
      : &AMDGPU::SReg_64_XEXECRegClass;
  case -1:
    return nullptr;
  default:
    return AMDGPUGenRegisterInfo::getRegClass(RCID);
  }
}

// Find reaching register definition
MachineInstr *SIRegisterInfo::findReachingDef(Register Reg, unsigned SubReg,
                                              MachineInstr &Use,
                                              MachineRegisterInfo &MRI,
                                              LiveIntervals *LIS) const {
  auto &MDT = LIS->getAnalysis<MachineDominatorTree>();
  SlotIndex UseIdx = LIS->getInstructionIndex(Use);
  SlotIndex DefIdx;

  if (Reg.isVirtual()) {
    if (!LIS->hasInterval(Reg))
      return nullptr;
    LiveInterval &LI = LIS->getInterval(Reg);
    LaneBitmask SubLanes = SubReg ? getSubRegIndexLaneMask(SubReg)
                                  : MRI.getMaxLaneMaskForVReg(Reg);
    VNInfo *V = nullptr;
    if (LI.hasSubRanges()) {
      for (auto &S : LI.subranges()) {
        if ((S.LaneMask & SubLanes) == SubLanes) {
          V = S.getVNInfoAt(UseIdx);
          break;
        }
      }
    } else {
      V = LI.getVNInfoAt(UseIdx);
    }
    if (!V)
      return nullptr;
    DefIdx = V->def;
  } else {
    // Find last def.
    for (MCRegUnitIterator Units(Reg, this); Units.isValid(); ++Units) {
      LiveRange &LR = LIS->getRegUnit(*Units);
      if (VNInfo *V = LR.getVNInfoAt(UseIdx)) {
        if (!DefIdx.isValid() ||
            MDT.dominates(LIS->getInstructionFromIndex(DefIdx),
                          LIS->getInstructionFromIndex(V->def)))
          DefIdx = V->def;
      } else {
        return nullptr;
      }
    }
  }

  MachineInstr *Def = LIS->getInstructionFromIndex(DefIdx);

  if (!Def || !MDT.dominates(Def, &Use))
    return nullptr;

  assert(Def->modifiesRegister(Reg, this));

  return Def;
}

MCPhysReg SIRegisterInfo::get32BitRegister(MCPhysReg Reg) const {
  assert(getRegSizeInBits(*getPhysRegClass(Reg)) <= 32);

  for (const TargetRegisterClass &RC : { AMDGPU::VGPR_32RegClass,
                                         AMDGPU::SReg_32RegClass,
                                         AMDGPU::AGPR_32RegClass } ) {
    if (MCPhysReg Super = getMatchingSuperReg(Reg, AMDGPU::lo16, &RC))
      return Super;
  }
  if (MCPhysReg Super = getMatchingSuperReg(Reg, AMDGPU::hi16,
                                            &AMDGPU::VGPR_32RegClass)) {
      return Super;
  }

  return AMDGPU::NoRegister;
}

bool SIRegisterInfo::isConstantPhysReg(MCRegister PhysReg) const {
  switch (PhysReg) {
  case AMDGPU::SGPR_NULL:
  case AMDGPU::SRC_SHARED_BASE:
  case AMDGPU::SRC_PRIVATE_BASE:
  case AMDGPU::SRC_SHARED_LIMIT:
  case AMDGPU::SRC_PRIVATE_LIMIT:
    return true;
  default:
    return false;
  }
}

ArrayRef<MCPhysReg>
SIRegisterInfo::getAllSGPR128(const MachineFunction &MF) const {
  return makeArrayRef(AMDGPU::SGPR_128RegClass.begin(),
                      ST.getMaxNumSGPRs(MF) / 4);
}

ArrayRef<MCPhysReg>
SIRegisterInfo::getAllSGPR32(const MachineFunction &MF) const {
  return makeArrayRef(AMDGPU::SGPR_32RegClass.begin(), ST.getMaxNumSGPRs(MF));
}
