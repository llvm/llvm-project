//===-- RISCVRegisterInfo.cpp - RISCV Register Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the RISCV implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "RISCVRegisterInfo.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_REGINFO_TARGET_DESC
#include "RISCVGenRegisterInfo.inc"

using namespace llvm;

static cl::opt<bool>
    DisableRegAllocHints("riscv-disable-regalloc-hints", cl::Hidden,
                         cl::init(false),
                         cl::desc("Disable two address hints for register "
                                  "allocation"));

static_assert(RISCV::X1 == RISCV::X0 + 1, "Register list not consecutive");
static_assert(RISCV::X31 == RISCV::X0 + 31, "Register list not consecutive");
static_assert(RISCV::F1_H == RISCV::F0_H + 1, "Register list not consecutive");
static_assert(RISCV::F31_H == RISCV::F0_H + 31,
              "Register list not consecutive");
static_assert(RISCV::F1_F == RISCV::F0_F + 1, "Register list not consecutive");
static_assert(RISCV::F31_F == RISCV::F0_F + 31,
              "Register list not consecutive");
static_assert(RISCV::F1_D == RISCV::F0_D + 1, "Register list not consecutive");
static_assert(RISCV::F31_D == RISCV::F0_D + 31,
              "Register list not consecutive");
static_assert(RISCV::V1 == RISCV::V0 + 1, "Register list not consecutive");
static_assert(RISCV::V31 == RISCV::V0 + 31, "Register list not consecutive");

RISCVRegisterInfo::RISCVRegisterInfo(unsigned HwMode)
    : RISCVGenRegisterInfo(RISCV::X1, /*DwarfFlavour*/0, /*EHFlavor*/0,
                           /*PC*/0, HwMode) {}

const MCPhysReg *
RISCVRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  auto &Subtarget = MF->getSubtarget<RISCVSubtarget>();
  if (MF->getFunction().getCallingConv() == CallingConv::GHC)
    return CSR_NoRegs_SaveList;
  if (MF->getFunction().hasFnAttribute("interrupt")) {
    if (Subtarget.hasStdExtD())
      return CSR_XLEN_F64_Interrupt_SaveList;
    if (Subtarget.hasStdExtF())
      return CSR_XLEN_F32_Interrupt_SaveList;
    return CSR_Interrupt_SaveList;
  }

  switch (Subtarget.getTargetABI()) {
  default:
    llvm_unreachable("Unrecognized ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    return CSR_ILP32_LP64_SaveList;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    return CSR_ILP32F_LP64F_SaveList;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    return CSR_ILP32D_LP64D_SaveList;
  }
}

BitVector RISCVRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  const RISCVFrameLowering *TFI = getFrameLowering(MF);
  BitVector Reserved(getNumRegs());

  // Mark any registers requested to be reserved as such
  for (size_t Reg = 0; Reg < getNumRegs(); Reg++) {
    if (MF.getSubtarget<RISCVSubtarget>().isRegisterReservedByUser(Reg))
      markSuperRegs(Reserved, Reg);
  }

  // Use markSuperRegs to ensure any register aliases are also reserved
  markSuperRegs(Reserved, RISCV::X0); // zero
  markSuperRegs(Reserved, RISCV::X2); // sp
  markSuperRegs(Reserved, RISCV::X3); // gp
  markSuperRegs(Reserved, RISCV::X4); // tp
  if (TFI->hasFP(MF))
    markSuperRegs(Reserved, RISCV::X8); // fp
  // Reserve the base register if we need to realign the stack and allocate
  // variable-sized objects at runtime.
  if (TFI->hasBP(MF))
    markSuperRegs(Reserved, RISCVABI::getBPReg()); // bp

  // V registers for code generation. We handle them manually.
  markSuperRegs(Reserved, RISCV::VL);
  markSuperRegs(Reserved, RISCV::VTYPE);
  markSuperRegs(Reserved, RISCV::VXSAT);
  markSuperRegs(Reserved, RISCV::VXRM);
  markSuperRegs(Reserved, RISCV::VLENB); // vlenb (constant)

  // Floating point environment registers.
  markSuperRegs(Reserved, RISCV::FRM);
  markSuperRegs(Reserved, RISCV::FFLAGS);

  assert(checkAllSuperRegsMarked(Reserved));
  return Reserved;
}

bool RISCVRegisterInfo::isAsmClobberable(const MachineFunction &MF,
                                         MCRegister PhysReg) const {
  return !MF.getSubtarget<RISCVSubtarget>().isRegisterReservedByUser(PhysReg);
}

const uint32_t *RISCVRegisterInfo::getNoPreservedMask() const {
  return CSR_NoRegs_RegMask;
}

// Frame indexes representing locations of CSRs which are given a fixed location
// by save/restore libcalls.
static const std::pair<unsigned, int> FixedCSRFIMap[] = {
  {/*ra*/  RISCV::X1,   -1},
  {/*s0*/  RISCV::X8,   -2},
  {/*s1*/  RISCV::X9,   -3},
  {/*s2*/  RISCV::X18,  -4},
  {/*s3*/  RISCV::X19,  -5},
  {/*s4*/  RISCV::X20,  -6},
  {/*s5*/  RISCV::X21,  -7},
  {/*s6*/  RISCV::X22,  -8},
  {/*s7*/  RISCV::X23,  -9},
  {/*s8*/  RISCV::X24,  -10},
  {/*s9*/  RISCV::X25,  -11},
  {/*s10*/ RISCV::X26,  -12},
  {/*s11*/ RISCV::X27,  -13}
};

bool RISCVRegisterInfo::hasReservedSpillSlot(const MachineFunction &MF,
                                             Register Reg,
                                             int &FrameIdx) const {
  const auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  if (!RVFI->useSaveRestoreLibCalls(MF))
    return false;

  const auto *FII =
      llvm::find_if(FixedCSRFIMap, [&](auto P) { return P.first == Reg; });
  if (FII == std::end(FixedCSRFIMap))
    return false;

  FrameIdx = FII->second;
  return true;
}

bool RISCVRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, unsigned FIOperandNum,
                                            RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected non-zero SPAdj value");

  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo *TII = ST.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  Register FrameReg;
  StackOffset Offset =
      getFrameLowering(MF)->getFrameIndexReference(MF, FrameIndex, FrameReg);
  bool IsRVVSpill = RISCV::isRVVSpill(MI);
  if (!IsRVVSpill)
    Offset += StackOffset::getFixed(MI.getOperand(FIOperandNum + 1).getImm());

  if (Offset.getScalable() &&
      ST.getRealMinVLen() == ST.getRealMaxVLen()) {
    // For an exact VLEN value, scalable offsets become constant and thus
    // can be converted entirely into fixed offsets.
    int64_t FixedValue = Offset.getFixed();
    int64_t ScalableValue = Offset.getScalable();
    assert(ScalableValue % 8 == 0 &&
           "Scalable offset is not a multiple of a single vector size.");
    int64_t NumOfVReg = ScalableValue / 8;
    int64_t VLENB = ST.getRealMinVLen() / 8;
    Offset = StackOffset::getFixed(FixedValue + NumOfVReg * VLENB);
  }

  if (!isInt<32>(Offset.getFixed())) {
    report_fatal_error(
        "Frame offsets outside of the signed 32-bit range not supported");
  }

  MachineBasicBlock &MBB = *MI.getParent();
  bool FrameRegIsKill = false;

  // If the instruction is an ADDI, we can use it's destination as a scratch
  // register. Load instructions might have an FP or vector destination and
  // stores don't have a destination register.
  Register DestReg;
  if (MI.getOpcode() == RISCV::ADDI)
    DestReg = MI.getOperand(0).getReg();

  // If required, pre-compute the scalable factor amount which will be used in
  // later offset computation. Since this sequence requires up to two scratch
  // registers -- after which one is made free -- this grants us better
  // scavenging of scratch registers as only up to two are live at one time,
  // rather than three.
  unsigned ScalableAdjOpc = RISCV::ADD;
  if (Offset.getScalable()) {
    int64_t ScalableValue = Offset.getScalable();
    if (ScalableValue < 0) {
      ScalableValue = -ScalableValue;
      ScalableAdjOpc = RISCV::SUB;
    }
    // Use DestReg if it exists, otherwise create a new register.
    if (!DestReg)
      DestReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    // Get vlenb and multiply vlen with the number of vector registers.
    TII->getVLENFactoredAmount(MF, MBB, II, DL, DestReg, ScalableValue);
  }

  if (!isInt<12>(Offset.getFixed())) {
    // The offset won't fit in an immediate, so use a scratch register instead
    // Modify Offset and FrameReg appropriately.

    // Reuse destination register if it exists and is not holding a scalable
    // offset.
    Register ScratchReg = DestReg;
    if (!DestReg || Offset.getScalable()) {
      ScratchReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
      // Also save to DestReg if it doesn't exist.
      if (!DestReg)
        DestReg = ScratchReg;
    }

    TII->movImm(MBB, II, DL, ScratchReg, Offset.getFixed());
    BuildMI(MBB, II, DL, TII->get(RISCV::ADD), ScratchReg)
        .addReg(FrameReg, getKillRegState(FrameRegIsKill))
        .addReg(ScratchReg, RegState::Kill);
    // If this was an ADDI and there is no scalable offset, we can remove it.
    if (MI.getOpcode() == RISCV::ADDI && !Offset.getScalable()) {
      assert(MI.getOperand(0).getReg() == ScratchReg &&
             "Expected to have written ADDI destination register");
      MI.eraseFromParent();
      return true;
    }

    Offset = StackOffset::get(0, Offset.getScalable());
    FrameReg = ScratchReg;
    FrameRegIsKill = true;
  }

  // Add in the scalable offset which has already been computed in DestReg.
  if (Offset.getScalable()) {
    assert(DestReg && "DestReg should be valid");
    BuildMI(MBB, II, DL, TII->get(ScalableAdjOpc), DestReg)
        .addReg(FrameReg, getKillRegState(FrameRegIsKill))
        .addReg(DestReg, RegState::Kill);
    // If this was an ADDI and there is no fixed offset, we can remove it.
    if (MI.getOpcode() == RISCV::ADDI && !Offset.getFixed()) {
      assert(MI.getOperand(0).getReg() == DestReg &&
             "Expected to have written ADDI destination register");
      MI.eraseFromParent();
      return true;
    }
    FrameReg = DestReg;
    FrameRegIsKill = true;
  }

  // Handle the fixed offset which might be zero.
  if (IsRVVSpill) {
    // RVVSpills don't have an immediate. Add an ADDI if the fixed offset is
    // needed.
    if (Offset.getFixed()) {
      // Reuse DestReg if it exists, otherwise create a new register.
      if (!DestReg)
        DestReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
      BuildMI(MBB, II, DL, TII->get(RISCV::ADDI), DestReg)
        .addReg(FrameReg, getKillRegState(FrameRegIsKill))
        .addImm(Offset.getFixed());
      FrameReg = DestReg;
      FrameRegIsKill = true;
    }
  } else {
    // Otherwise we can replace the original immediate.
    MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset.getFixed());
  }

  // Finally, replace the frame index operand.
  MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, false, false,
                                               FrameRegIsKill);

  auto ZvlssegInfo = RISCV::isRVVSpillForZvlsseg(MI.getOpcode());
  if (ZvlssegInfo) {
    Register VL = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    BuildMI(MBB, II, DL, TII->get(RISCV::PseudoReadVLENB), VL);
    uint32_t ShiftAmount = Log2_32(ZvlssegInfo->second);
    if (ShiftAmount != 0)
      BuildMI(MBB, II, DL, TII->get(RISCV::SLLI), VL)
          .addReg(VL)
          .addImm(ShiftAmount);
    // The last argument of pseudo spilling opcode for zvlsseg is the length of
    // one element of zvlsseg types. For example, for vint32m2x2_t, it will be
    // the length of vint32m2_t.
    MI.getOperand(FIOperandNum + 1).ChangeToRegister(VL, /*isDef=*/false);
  }
  return false;
}

Register RISCVRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = getFrameLowering(MF);
  return TFI->hasFP(MF) ? RISCV::X8 : RISCV::X2;
}

const uint32_t *
RISCVRegisterInfo::getCallPreservedMask(const MachineFunction & MF,
                                        CallingConv::ID CC) const {
  auto &Subtarget = MF.getSubtarget<RISCVSubtarget>();

  if (CC == CallingConv::GHC)
    return CSR_NoRegs_RegMask;
  switch (Subtarget.getTargetABI()) {
  default:
    llvm_unreachable("Unrecognized ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    return CSR_ILP32_LP64_RegMask;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    return CSR_ILP32F_LP64F_RegMask;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    return CSR_ILP32D_LP64D_RegMask;
  }
}

const TargetRegisterClass *
RISCVRegisterInfo::getLargestLegalSuperClass(const TargetRegisterClass *RC,
                                             const MachineFunction &) const {
  if (RC == &RISCV::VMV0RegClass)
    return &RISCV::VRRegClass;
  return RC;
}

void RISCVRegisterInfo::getOffsetOpcodes(const StackOffset &Offset,
                                         SmallVectorImpl<uint64_t> &Ops) const {
  // VLENB is the length of a vector register in bytes. We use <vscale x 8 x i8>
  // to represent one vector register. The dwarf offset is
  // VLENB * scalable_offset / 8.
  assert(Offset.getScalable() % 8 == 0 && "Invalid frame offset");

  // Add fixed-sized offset using existing DIExpression interface.
  DIExpression::appendOffset(Ops, Offset.getFixed());

  unsigned VLENB = getDwarfRegNum(RISCV::VLENB, true);
  int64_t VLENBSized = Offset.getScalable() / 8;
  if (VLENBSized > 0) {
    Ops.push_back(dwarf::DW_OP_constu);
    Ops.push_back(VLENBSized);
    Ops.append({dwarf::DW_OP_bregx, VLENB, 0ULL});
    Ops.push_back(dwarf::DW_OP_mul);
    Ops.push_back(dwarf::DW_OP_plus);
  } else if (VLENBSized < 0) {
    Ops.push_back(dwarf::DW_OP_constu);
    Ops.push_back(-VLENBSized);
    Ops.append({dwarf::DW_OP_bregx, VLENB, 0ULL});
    Ops.push_back(dwarf::DW_OP_mul);
    Ops.push_back(dwarf::DW_OP_minus);
  }
}

unsigned
RISCVRegisterInfo::getRegisterCostTableIndex(const MachineFunction &MF) const {
  return MF.getSubtarget<RISCVSubtarget>().hasStdExtC() ? 1 : 0;
}

// Add two address hints to improve chances of being able to use a compressed
// instruction.
bool RISCVRegisterInfo::getRegAllocationHints(
    Register VirtReg, ArrayRef<MCPhysReg> Order,
    SmallVectorImpl<MCPhysReg> &Hints, const MachineFunction &MF,
    const VirtRegMap *VRM, const LiveRegMatrix *Matrix) const {
  const MachineRegisterInfo *MRI = &MF.getRegInfo();

  bool BaseImplRetVal = TargetRegisterInfo::getRegAllocationHints(
      VirtReg, Order, Hints, MF, VRM, Matrix);

  if (!VRM || DisableRegAllocHints)
    return BaseImplRetVal;

  // Add any two address hints after any copy hints.
  SmallSet<unsigned, 4> TwoAddrHints;

  auto tryAddHint = [&](const MachineOperand &VRRegMO,
                        const MachineOperand &MO) -> void {
    Register Reg = MO.getReg();
    Register PhysReg =
        Register::isPhysicalRegister(Reg) ? Reg : Register(VRM->getPhys(Reg));
    if (PhysReg) {
      assert(!MO.getSubReg() && !VRRegMO.getSubReg() && "Unexpected subreg!");
      if (!MRI->isReserved(PhysReg) && !is_contained(Hints, PhysReg))
        TwoAddrHints.insert(PhysReg);
    }
  };

  // For now we support the compressible instructions which can encode all
  // registers and have a single register source.
  // TODO: Add more compressed instructions.
  auto isCompressible = [](const MachineInstr &MI) {
    switch (MI.getOpcode()) {
    default:
      return false;
    case RISCV::ADD:
    case RISCV::SLLI:
      return true;
    case RISCV::ADDI:
    case RISCV::ADDIW:
      return MI.getOperand(2).isImm() && isInt<6>(MI.getOperand(2).getImm());
    }
  };

  for (auto &MO : MRI->reg_nodbg_operands(VirtReg)) {
    const MachineInstr &MI = *MO.getParent();
    if (isCompressible(MI)) {
      unsigned OpIdx = MI.getOperandNo(&MO);
      if (OpIdx == 0 && MI.getOperand(1).isReg()) {
        tryAddHint(MO, MI.getOperand(1));
        if (MI.isCommutable() && MI.getOperand(2).isReg())
          tryAddHint(MO, MI.getOperand(2));
      } else if (OpIdx == 1) {
        tryAddHint(MO, MI.getOperand(0));
      } else if (MI.isCommutable() && OpIdx == 2) {
        tryAddHint(MO, MI.getOperand(0));
      }
    }
  }

  for (MCPhysReg OrderReg : Order)
    if (TwoAddrHints.count(OrderReg))
      Hints.push_back(OrderReg);

  return BaseImplRetVal;
}
