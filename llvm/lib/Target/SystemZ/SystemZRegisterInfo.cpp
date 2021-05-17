//===-- SystemZRegisterInfo.cpp - SystemZ register information ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZRegisterInfo.h"
#include "SystemZInstrInfo.h"
#include "SystemZSubtarget.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/DebugInfoMetadata.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "SystemZGenRegisterInfo.inc"

// Given that MO is a GRX32 operand, return either GR32 or GRH32 if MO
// somehow belongs in it. Otherwise, return GRX32.
static const TargetRegisterClass *getRC32(MachineOperand &MO,
                                          const VirtRegMap *VRM,
                                          const MachineRegisterInfo *MRI) {
  const TargetRegisterClass *RC = MRI->getRegClass(MO.getReg());

  if (SystemZ::GR32BitRegClass.hasSubClassEq(RC) ||
      MO.getSubReg() == SystemZ::subreg_l32 ||
      MO.getSubReg() == SystemZ::subreg_hl32)
    return &SystemZ::GR32BitRegClass;
  if (SystemZ::GRH32BitRegClass.hasSubClassEq(RC) ||
      MO.getSubReg() == SystemZ::subreg_h32 ||
      MO.getSubReg() == SystemZ::subreg_hh32)
    return &SystemZ::GRH32BitRegClass;

  if (VRM && VRM->hasPhys(MO.getReg())) {
    Register PhysReg = VRM->getPhys(MO.getReg());
    if (SystemZ::GR32BitRegClass.contains(PhysReg))
      return &SystemZ::GR32BitRegClass;
    assert (SystemZ::GRH32BitRegClass.contains(PhysReg) &&
            "Phys reg not in GR32 or GRH32?");
    return &SystemZ::GRH32BitRegClass;
  }

  assert (RC == &SystemZ::GRX32BitRegClass);
  return RC;
}

// Pass the registers of RC as hints while making sure that if any of these
// registers are copy hints (and therefore already in Hints), hint them
// first.
static void addHints(ArrayRef<MCPhysReg> Order,
                     SmallVectorImpl<MCPhysReg> &Hints,
                     const TargetRegisterClass *RC,
                     const MachineRegisterInfo *MRI) {
  SmallSet<unsigned, 4> CopyHints;
  CopyHints.insert(Hints.begin(), Hints.end());
  Hints.clear();
  for (MCPhysReg Reg : Order)
    if (CopyHints.count(Reg) &&
        RC->contains(Reg) && !MRI->isReserved(Reg))
      Hints.push_back(Reg);
  for (MCPhysReg Reg : Order)
    if (!CopyHints.count(Reg) &&
        RC->contains(Reg) && !MRI->isReserved(Reg))
      Hints.push_back(Reg);
}

bool SystemZRegisterInfo::getRegAllocationHints(
    Register VirtReg, ArrayRef<MCPhysReg> Order,
    SmallVectorImpl<MCPhysReg> &Hints, const MachineFunction &MF,
    const VirtRegMap *VRM, const LiveRegMatrix *Matrix) const {
  const MachineRegisterInfo *MRI = &MF.getRegInfo();
  const SystemZSubtarget &Subtarget = MF.getSubtarget<SystemZSubtarget>();
  const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();

  bool BaseImplRetVal = TargetRegisterInfo::getRegAllocationHints(
      VirtReg, Order, Hints, MF, VRM, Matrix);

  if (VRM != nullptr) {
    // Add any two address hints after any copy hints.
    SmallSet<unsigned, 4> TwoAddrHints;
    for (auto &Use : MRI->reg_nodbg_instructions(VirtReg))
      if (SystemZ::getTwoOperandOpcode(Use.getOpcode()) != -1) {
        const MachineOperand *VRRegMO = nullptr;
        const MachineOperand *OtherMO = nullptr;
        const MachineOperand *CommuMO = nullptr;
        if (VirtReg == Use.getOperand(0).getReg()) {
          VRRegMO = &Use.getOperand(0);
          OtherMO = &Use.getOperand(1);
          if (Use.isCommutable())
            CommuMO = &Use.getOperand(2);
        } else if (VirtReg == Use.getOperand(1).getReg()) {
          VRRegMO = &Use.getOperand(1);
          OtherMO = &Use.getOperand(0);
        } else if (VirtReg == Use.getOperand(2).getReg() &&
                   Use.isCommutable()) {
          VRRegMO = &Use.getOperand(2);
          OtherMO = &Use.getOperand(0);
        } else
          continue;

        auto tryAddHint = [&](const MachineOperand *MO) -> void {
          Register Reg = MO->getReg();
          Register PhysReg = Register::isPhysicalRegister(Reg)
                                 ? Reg
                                 : Register(VRM->getPhys(Reg));
          if (PhysReg) {
            if (MO->getSubReg())
              PhysReg = getSubReg(PhysReg, MO->getSubReg());
            if (VRRegMO->getSubReg())
              PhysReg = getMatchingSuperReg(PhysReg, VRRegMO->getSubReg(),
                                            MRI->getRegClass(VirtReg));
            if (!MRI->isReserved(PhysReg) && !is_contained(Hints, PhysReg))
              TwoAddrHints.insert(PhysReg);
          }
        };
        tryAddHint(OtherMO);
        if (CommuMO)
          tryAddHint(CommuMO);
      }
    for (MCPhysReg OrderReg : Order)
      if (TwoAddrHints.count(OrderReg))
        Hints.push_back(OrderReg);
  }

  if (MRI->getRegClass(VirtReg) == &SystemZ::GRX32BitRegClass) {
    SmallVector<Register, 8> Worklist;
    SmallSet<Register, 4> DoneRegs;
    Worklist.push_back(VirtReg);
    while (Worklist.size()) {
      Register Reg = Worklist.pop_back_val();
      if (!DoneRegs.insert(Reg).second)
        continue;

      for (auto &Use : MRI->reg_instructions(Reg)) {
        // For LOCRMux, see if the other operand is already a high or low
        // register, and in that case give the corresponding hints for
        // VirtReg. LOCR instructions need both operands in either high or
        // low parts. Same handling for SELRMux.
        if (Use.getOpcode() == SystemZ::LOCRMux ||
            Use.getOpcode() == SystemZ::SELRMux) {
          MachineOperand &TrueMO = Use.getOperand(1);
          MachineOperand &FalseMO = Use.getOperand(2);
          const TargetRegisterClass *RC =
            TRI->getCommonSubClass(getRC32(FalseMO, VRM, MRI),
                                   getRC32(TrueMO, VRM, MRI));
          if (Use.getOpcode() == SystemZ::SELRMux)
            RC = TRI->getCommonSubClass(RC,
                                        getRC32(Use.getOperand(0), VRM, MRI));
          if (RC && RC != &SystemZ::GRX32BitRegClass) {
            addHints(Order, Hints, RC, MRI);
            // Return true to make these hints the only regs available to
            // RA. This may mean extra spilling but since the alternative is
            // a jump sequence expansion of the LOCRMux, it is preferred.
            return true;
          }

          // Add the other operand of the LOCRMux to the worklist.
          Register OtherReg =
              (TrueMO.getReg() == Reg ? FalseMO.getReg() : TrueMO.getReg());
          if (MRI->getRegClass(OtherReg) == &SystemZ::GRX32BitRegClass)
            Worklist.push_back(OtherReg);
        } // end LOCRMux
        else if (Use.getOpcode() == SystemZ::CHIMux ||
                 Use.getOpcode() == SystemZ::CFIMux) {
          if (Use.getOperand(1).getImm() == 0) {
            bool OnlyLMuxes = true;
            for (MachineInstr &DefMI : MRI->def_instructions(VirtReg))
              if (DefMI.getOpcode() != SystemZ::LMux)
                OnlyLMuxes = false;
            if (OnlyLMuxes) {
              addHints(Order, Hints, &SystemZ::GR32BitRegClass, MRI);
              // Return false to make these hints preferred but not obligatory.
              return false;
            }
          }
        } // end CHIMux / CFIMux
      }
    }
  }

  return BaseImplRetVal;
}

const MCPhysReg *
SystemZXPLINK64Registers::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_SystemZ_XPLINK64_SaveList;
}

const MCPhysReg *
SystemZELFRegisters::getCalleeSavedRegs(const MachineFunction *MF) const {
  const SystemZSubtarget &Subtarget = MF->getSubtarget<SystemZSubtarget>();
  if (MF->getFunction().getCallingConv() == CallingConv::GHC)
    return CSR_SystemZ_NoRegs_SaveList;
  if (MF->getFunction().getCallingConv() == CallingConv::AnyReg)
    return Subtarget.hasVector()? CSR_SystemZ_AllRegs_Vector_SaveList
                                : CSR_SystemZ_AllRegs_SaveList;
  if (MF->getSubtarget().getTargetLowering()->supportSwiftError() &&
      MF->getFunction().getAttributes().hasAttrSomewhere(
          Attribute::SwiftError))
    return CSR_SystemZ_SwiftError_SaveList;
  return CSR_SystemZ_ELF_SaveList;
}

const uint32_t *
SystemZXPLINK64Registers::getCallPreservedMask(const MachineFunction &MF,
                                               CallingConv::ID CC) const {
  return CSR_SystemZ_XPLINK64_RegMask;
}

const uint32_t *
SystemZELFRegisters::getCallPreservedMask(const MachineFunction &MF,
                                          CallingConv::ID CC) const {
  const SystemZSubtarget &Subtarget = MF.getSubtarget<SystemZSubtarget>();
  if (CC == CallingConv::GHC)
    return CSR_SystemZ_NoRegs_RegMask;
  if (CC == CallingConv::AnyReg)
    return Subtarget.hasVector()? CSR_SystemZ_AllRegs_Vector_RegMask
                                : CSR_SystemZ_AllRegs_RegMask;
  if (MF.getSubtarget().getTargetLowering()->supportSwiftError() &&
      MF.getFunction().getAttributes().hasAttrSomewhere(
          Attribute::SwiftError))
    return CSR_SystemZ_SwiftError_RegMask;
  return CSR_SystemZ_ELF_RegMask;
}

SystemZRegisterInfo::SystemZRegisterInfo(unsigned int RA)
    : SystemZGenRegisterInfo(RA) {}

const MCPhysReg *
SystemZRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {

  const SystemZSubtarget *Subtarget = &MF->getSubtarget<SystemZSubtarget>();
  SystemZCallingConventionRegisters *Regs = Subtarget->getSpecialRegisters();

  return Regs->getCalleeSavedRegs(MF);
}

const uint32_t *
SystemZRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                          CallingConv::ID CC) const {

  const SystemZSubtarget *Subtarget = &MF.getSubtarget<SystemZSubtarget>();
  SystemZCallingConventionRegisters *Regs = Subtarget->getSpecialRegisters();
  return Regs->getCallPreservedMask(MF, CC);
}

BitVector
SystemZRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  const SystemZFrameLowering *TFI = getFrameLowering(MF);
  const SystemZSubtarget *Subtarget = &MF.getSubtarget<SystemZSubtarget>();
  SystemZCallingConventionRegisters *Regs = Subtarget->getSpecialRegisters();
  if (TFI->hasFP(MF))
    // The frame pointer. Reserve all aliases.
    for (MCRegAliasIterator AI(Regs->getFramePointerRegister(), this, true);
         AI.isValid(); ++AI)
      Reserved.set(*AI);

  // Reserve all aliases for the stack pointer.
  for (MCRegAliasIterator AI(Regs->getStackPointerRegister(), this, true);
       AI.isValid(); ++AI)
    Reserved.set(*AI);

  // A0 and A1 hold the thread pointer.
  Reserved.set(SystemZ::A0);
  Reserved.set(SystemZ::A1);

  // FPC is the floating-point control register.
  Reserved.set(SystemZ::FPC);

  return Reserved;
}

void
SystemZRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                         int SPAdj, unsigned FIOperandNum,
                                         RegScavenger *RS) const {
  assert(SPAdj == 0 && "Outgoing arguments should be part of the frame");

  MachineBasicBlock &MBB = *MI->getParent();
  MachineFunction &MF = *MBB.getParent();
  auto *TII =
      static_cast<const SystemZInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const SystemZFrameLowering *TFI = getFrameLowering(MF);
  DebugLoc DL = MI->getDebugLoc();

  // Decompose the frame index into a base and offset.
  int FrameIndex = MI->getOperand(FIOperandNum).getIndex();
  Register BasePtr;
  int64_t Offset =
      (TFI->getFrameIndexReference(MF, FrameIndex, BasePtr).getFixed() +
       MI->getOperand(FIOperandNum + 1).getImm());

  // Special handling of dbg_value instructions.
  if (MI->isDebugValue()) {
    MI->getOperand(FIOperandNum).ChangeToRegister(BasePtr, /*isDef*/ false);
    if (MI->isNonListDebugValue()) {
      MI->getDebugOffset().ChangeToImmediate(Offset);
    } else {
      unsigned OpIdx = MI->getDebugOperandIndex(&MI->getOperand(FIOperandNum));
      SmallVector<uint64_t, 3> Ops;
      DIExpression::appendOffset(
          Ops, TFI->getFrameIndexReference(MF, FrameIndex, BasePtr).getFixed());
      MI->getDebugExpressionOp().setMetadata(
          DIExpression::appendOpsToArg(MI->getDebugExpression(), Ops, OpIdx));
    }
    return;
  }

  // See if the offset is in range, or if an equivalent instruction that
  // accepts the offset exists.
  unsigned Opcode = MI->getOpcode();
  unsigned OpcodeForOffset = TII->getOpcodeForOffset(Opcode, Offset);
  if (OpcodeForOffset) {
    if (OpcodeForOffset == SystemZ::LE &&
        MF.getSubtarget<SystemZSubtarget>().hasVector()) {
      // If LE is ok for offset, use LDE instead on z13.
      OpcodeForOffset = SystemZ::LDE32;
    }
    MI->getOperand(FIOperandNum).ChangeToRegister(BasePtr, false);
  }
  else {
    // Create an anchor point that is in range.  Start at 0xffff so that
    // can use LLILH to load the immediate.
    int64_t OldOffset = Offset;
    int64_t Mask = 0xffff;
    do {
      Offset = OldOffset & Mask;
      OpcodeForOffset = TII->getOpcodeForOffset(Opcode, Offset);
      Mask >>= 1;
      assert(Mask && "One offset must be OK");
    } while (!OpcodeForOffset);

    Register ScratchReg =
        MF.getRegInfo().createVirtualRegister(&SystemZ::ADDR64BitRegClass);
    int64_t HighOffset = OldOffset - Offset;

    if (MI->getDesc().TSFlags & SystemZII::HasIndex
        && MI->getOperand(FIOperandNum + 2).getReg() == 0) {
      // Load the offset into the scratch register and use it as an index.
      // The scratch register then dies here.
      TII->loadImmediate(MBB, MI, ScratchReg, HighOffset);
      MI->getOperand(FIOperandNum).ChangeToRegister(BasePtr, false);
      MI->getOperand(FIOperandNum + 2).ChangeToRegister(ScratchReg,
                                                        false, false, true);
    } else {
      // Load the anchor address into a scratch register.
      unsigned LAOpcode = TII->getOpcodeForOffset(SystemZ::LA, HighOffset);
      if (LAOpcode)
        BuildMI(MBB, MI, DL, TII->get(LAOpcode),ScratchReg)
          .addReg(BasePtr).addImm(HighOffset).addReg(0);
      else {
        // Load the high offset into the scratch register and use it as
        // an index.
        TII->loadImmediate(MBB, MI, ScratchReg, HighOffset);
        BuildMI(MBB, MI, DL, TII->get(SystemZ::LA), ScratchReg)
          .addReg(BasePtr, RegState::Kill).addImm(0).addReg(ScratchReg);
      }

      // Use the scratch register as the base.  It then dies here.
      MI->getOperand(FIOperandNum).ChangeToRegister(ScratchReg,
                                                    false, false, true);
    }
  }
  MI->setDesc(TII->get(OpcodeForOffset));
  MI->getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
}

bool SystemZRegisterInfo::shouldCoalesce(MachineInstr *MI,
                                  const TargetRegisterClass *SrcRC,
                                  unsigned SubReg,
                                  const TargetRegisterClass *DstRC,
                                  unsigned DstSubReg,
                                  const TargetRegisterClass *NewRC,
                                  LiveIntervals &LIS) const {
  assert (MI->isCopy() && "Only expecting COPY instructions");

  // Coalesce anything which is not a COPY involving a subreg to/from GR128.
  if (!(NewRC->hasSuperClassEq(&SystemZ::GR128BitRegClass) &&
        (getRegSizeInBits(*SrcRC) <= 64 || getRegSizeInBits(*DstRC) <= 64)))
    return true;

  // Allow coalescing of a GR128 subreg COPY only if the live ranges are small
  // and local to one MBB with not too much interferring registers. Otherwise
  // regalloc may run out of registers.

  unsigned WideOpNo = (getRegSizeInBits(*SrcRC) == 128 ? 1 : 0);
  Register GR128Reg = MI->getOperand(WideOpNo).getReg();
  Register GRNarReg = MI->getOperand((WideOpNo == 1) ? 0 : 1).getReg();
  LiveInterval &IntGR128 = LIS.getInterval(GR128Reg);
  LiveInterval &IntGRNar = LIS.getInterval(GRNarReg);

  // Check that the two virtual registers are local to MBB.
  MachineBasicBlock *MBB = MI->getParent();
  MachineInstr *FirstMI_GR128 =
    LIS.getInstructionFromIndex(IntGR128.beginIndex());
  MachineInstr *FirstMI_GRNar =
    LIS.getInstructionFromIndex(IntGRNar.beginIndex());
  MachineInstr *LastMI_GR128 = LIS.getInstructionFromIndex(IntGR128.endIndex());
  MachineInstr *LastMI_GRNar = LIS.getInstructionFromIndex(IntGRNar.endIndex());
  if ((!FirstMI_GR128 || FirstMI_GR128->getParent() != MBB) ||
      (!FirstMI_GRNar || FirstMI_GRNar->getParent() != MBB) ||
      (!LastMI_GR128 || LastMI_GR128->getParent() != MBB) ||
      (!LastMI_GRNar || LastMI_GRNar->getParent() != MBB))
    return false;

  MachineBasicBlock::iterator MII = nullptr, MEE = nullptr;
  if (WideOpNo == 1) {
    MII = FirstMI_GR128;
    MEE = LastMI_GRNar;
  } else {
    MII = FirstMI_GRNar;
    MEE = LastMI_GR128;
  }

  // Check if coalescing seems safe by finding the set of clobbered physreg
  // pairs in the region.
  BitVector PhysClobbered(getNumRegs());
  MEE++;
  for (; MII != MEE; ++MII) {
    for (const MachineOperand &MO : MII->operands())
      if (MO.isReg() && Register::isPhysicalRegister(MO.getReg())) {
        for (MCSuperRegIterator SI(MO.getReg(), this, true/*IncludeSelf*/);
             SI.isValid(); ++SI)
          if (NewRC->contains(*SI)) {
            PhysClobbered.set(*SI);
            break;
          }
      }
  }

  // Demand an arbitrary margin of free regs.
  unsigned const DemandedFreeGR128 = 3;
  if (PhysClobbered.count() > (NewRC->getNumRegs() - DemandedFreeGR128))
    return false;

  return true;
}

Register
SystemZRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const SystemZFrameLowering *TFI = getFrameLowering(MF);
  const SystemZSubtarget *Subtarget = &MF.getSubtarget<SystemZSubtarget>();
  SystemZCallingConventionRegisters *Regs = Subtarget->getSpecialRegisters();

  return TFI->hasFP(MF) ? Regs->getFramePointerRegister()
                        : Regs->getStackPointerRegister();
}

const TargetRegisterClass *
SystemZRegisterInfo::getCrossCopyRegClass(const TargetRegisterClass *RC) const {
  if (RC == &SystemZ::CCRRegClass)
    return &SystemZ::GR32BitRegClass;
  return RC;
}

