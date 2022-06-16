//===----- RISCVMergeBaseOffset.cpp - Optimise address calculations  ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Merge the offset of address calculation into the offset field
// of instructions in a global address lowering sequence. This pass transforms:
//   lui  vreg1, %hi(s)
//   addi vreg2, vreg1, %lo(s)
//   addi vreg3, verg2, Offset
//
//   Into:
//   lui  vreg1, %hi(s+Offset)
//   addi vreg2, vreg1, %lo(s+Offset)
//
// The transformation is carried out under certain conditions:
// 1) The offset field in the base of global address lowering sequence is zero.
// 2) The lowered global address has only one use.
//
// The offset field can be in a different form. This pass handles all of them.
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetOptions.h"
#include <set>
using namespace llvm;

#define DEBUG_TYPE "riscv-merge-base-offset"
#define RISCV_MERGE_BASE_OFFSET_NAME "RISCV Merge Base Offset"
namespace {

struct RISCVMergeBaseOffsetOpt : public MachineFunctionPass {
private:
  const RISCVSubtarget *ST = nullptr;

public:
  static char ID;
  bool runOnMachineFunction(MachineFunction &Fn) override;
  bool detectLuiAddiGlobal(MachineInstr &LUI, MachineInstr *&ADDI);

  bool detectAndFoldOffset(MachineInstr &HiLUI, MachineInstr &LoADDI);
  void foldOffset(MachineInstr &HiLUI, MachineInstr &LoADDI, MachineInstr &Tail,
                  int64_t Offset);
  bool matchLargeOffset(MachineInstr &TailAdd, Register GSReg, int64_t &Offset);
  bool matchShiftedOffset(MachineInstr &TailShXAdd, Register GSReg,
                          int64_t &Offset);

  RISCVMergeBaseOffsetOpt() : MachineFunctionPass(ID) {}

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return RISCV_MERGE_BASE_OFFSET_NAME;
  }

private:
  MachineRegisterInfo *MRI;
  std::set<MachineInstr *> DeadInstrs;
};
} // end anonymous namespace

char RISCVMergeBaseOffsetOpt::ID = 0;
INITIALIZE_PASS(RISCVMergeBaseOffsetOpt, DEBUG_TYPE,
                RISCV_MERGE_BASE_OFFSET_NAME, false, false)

// Detect the pattern:
//   lui   vreg1, %hi(s)
//   addi  vreg2, vreg1, %lo(s)
//
//   Pattern only accepted if:
//     1) ADDI has only one use.
//     2) LUI has only one use; which is the ADDI.
//     3) Both ADDI and LUI have GlobalAddress type which indicates that these
//        are generated from global address lowering.
//     4) Offset value in the Global Address is 0.
bool RISCVMergeBaseOffsetOpt::detectLuiAddiGlobal(MachineInstr &HiLUI,
                                                  MachineInstr *&LoADDI) {
  if (HiLUI.getOpcode() != RISCV::LUI ||
      HiLUI.getOperand(1).getTargetFlags() != RISCVII::MO_HI ||
      HiLUI.getOperand(1).getType() != MachineOperand::MO_GlobalAddress ||
      HiLUI.getOperand(1).getOffset() != 0 ||
      !MRI->hasOneUse(HiLUI.getOperand(0).getReg()))
    return false;
  Register HiLuiDestReg = HiLUI.getOperand(0).getReg();
  LoADDI = &*MRI->use_instr_begin(HiLuiDestReg);
  if (LoADDI->getOpcode() != RISCV::ADDI ||
      LoADDI->getOperand(2).getTargetFlags() != RISCVII::MO_LO ||
      LoADDI->getOperand(2).getType() != MachineOperand::MO_GlobalAddress ||
      LoADDI->getOperand(2).getOffset() != 0 ||
      !MRI->hasOneUse(LoADDI->getOperand(0).getReg()))
    return false;
  return true;
}

// Update the offset in HiLUI and LoADDI instructions.
// Delete the tail instruction and update all the uses to use the
// output from LoADDI.
void RISCVMergeBaseOffsetOpt::foldOffset(MachineInstr &HiLUI,
                                         MachineInstr &LoADDI,
                                         MachineInstr &Tail, int64_t Offset) {
  assert(isInt<32>(Offset) && "Unexpected offset");
  // Put the offset back in HiLUI and the LoADDI
  HiLUI.getOperand(1).setOffset(Offset);
  LoADDI.getOperand(2).setOffset(Offset);
  // Delete the tail instruction.
  DeadInstrs.insert(&Tail);
  MRI->replaceRegWith(Tail.getOperand(0).getReg(),
                      LoADDI.getOperand(0).getReg());
  LLVM_DEBUG(dbgs() << "  Merged offset " << Offset << " into base.\n"
                    << "     " << HiLUI << "     " << LoADDI;);
}

// Detect patterns for large offsets that are passed into an ADD instruction.
//
//                     Base address lowering is of the form:
//                        HiLUI:  lui   vreg1, %hi(s)
//                       LoADDI:  addi  vreg2, vreg1, %lo(s)
//                       /                                  \
//                      /                                    \
//                     /                                      \
//                    /  The large offset can be of two forms: \
//  1) Offset that has non zero bits in lower      2) Offset that has non zero
//     12 bits and upper 20 bits                      bits in upper 20 bits only
//   OffseLUI: lui   vreg3, 4
// OffsetTail: addi  voff, vreg3, 188                OffsetTail: lui  voff, 128
//                    \                                        /
//                     \                                      /
//                      \                                    /
//                       \                                  /
//                         TailAdd: add  vreg4, vreg2, voff
bool RISCVMergeBaseOffsetOpt::matchLargeOffset(MachineInstr &TailAdd,
                                               Register GAReg,
                                               int64_t &Offset) {
  assert((TailAdd.getOpcode() == RISCV::ADD) && "Expected ADD instruction!");
  Register Rs = TailAdd.getOperand(1).getReg();
  Register Rt = TailAdd.getOperand(2).getReg();
  Register Reg = Rs == GAReg ? Rt : Rs;

  // Can't fold if the register has more than one use.
  if (!MRI->hasOneUse(Reg))
    return false;
  // This can point to an ADDI or a LUI:
  MachineInstr &OffsetTail = *MRI->getVRegDef(Reg);
  if (OffsetTail.getOpcode() == RISCV::ADDI ||
      OffsetTail.getOpcode() == RISCV::ADDIW) {
    // The offset value has non zero bits in both %hi and %lo parts.
    // Detect an ADDI that feeds from a LUI instruction.
    MachineOperand &AddiImmOp = OffsetTail.getOperand(2);
    if (AddiImmOp.getTargetFlags() != RISCVII::MO_None)
      return false;
    int64_t OffLo = AddiImmOp.getImm();
    MachineInstr &OffsetLui =
        *MRI->getVRegDef(OffsetTail.getOperand(1).getReg());
    MachineOperand &LuiImmOp = OffsetLui.getOperand(1);
    if (OffsetLui.getOpcode() != RISCV::LUI ||
        LuiImmOp.getTargetFlags() != RISCVII::MO_None ||
        !MRI->hasOneUse(OffsetLui.getOperand(0).getReg()))
      return false;
    Offset = SignExtend64<32>(LuiImmOp.getImm() << 12);
    Offset += OffLo;
    // RV32 ignores the upper 32 bits. ADDIW sign extends the result.
    if (!ST->is64Bit() || OffsetTail.getOpcode() == RISCV::ADDIW)
       Offset = SignExtend64<32>(Offset);
    // We can only fold simm32 offsets.
    if (!isInt<32>(Offset))
      return false;
    LLVM_DEBUG(dbgs() << "  Offset Instrs: " << OffsetTail
                      << "                 " << OffsetLui);
    DeadInstrs.insert(&OffsetTail);
    DeadInstrs.insert(&OffsetLui);
    return true;
  } else if (OffsetTail.getOpcode() == RISCV::LUI) {
    // The offset value has all zero bits in the lower 12 bits. Only LUI
    // exists.
    LLVM_DEBUG(dbgs() << "  Offset Instr: " << OffsetTail);
    Offset = SignExtend64<32>(OffsetTail.getOperand(1).getImm() << 12);
    DeadInstrs.insert(&OffsetTail);
    return true;
  }
  return false;
}

// Detect patterns for offsets that are passed into a SHXADD instruction.
// The offset has 1,2, or 3 trailing zeros and fits in simm13, simm14, simm15.
// The constant is created with addi    voff, x0, C, and shXadd is used to
// fill insert the trailing zeros and do the addition.
//
// HiLUI:      lui     vreg1, %hi(s)
// LoADDI:     addi    vreg2, vreg1, %lo(s)
// OffsetTail: addi    voff, x0, C
// TailAdd:    shXadd  vreg4, voff, vreg2
bool RISCVMergeBaseOffsetOpt::matchShiftedOffset(MachineInstr &TailShXAdd,
                                                 Register GAReg,
                                                 int64_t &Offset) {
  assert((TailShXAdd.getOpcode() == RISCV::SH1ADD ||
          TailShXAdd.getOpcode() == RISCV::SH2ADD ||
          TailShXAdd.getOpcode() == RISCV::SH3ADD) &&
         "Expected SHXADD instruction!");

  // The first source is the shifted operand.
  Register Rs1 = TailShXAdd.getOperand(1).getReg();

  if (GAReg != TailShXAdd.getOperand(2).getReg())
    return false;

  // Can't fold if the register has more than one use.
  if (!MRI->hasOneUse(Rs1))
    return false;
  // This can point to an ADDI X0, C.
  MachineInstr &OffsetTail = *MRI->getVRegDef(Rs1);
  if (OffsetTail.getOpcode() != RISCV::ADDI)
    return false;
  if (!OffsetTail.getOperand(1).isReg() ||
      OffsetTail.getOperand(1).getReg() != RISCV::X0 ||
      !OffsetTail.getOperand(2).isImm())
    return false;

  Offset = OffsetTail.getOperand(2).getImm();
  assert(isInt<12>(Offset) && "Unexpected offset");

  unsigned ShAmt;
  switch (TailShXAdd.getOpcode()) {
  default: llvm_unreachable("Unexpected opcode");
  case RISCV::SH1ADD: ShAmt = 1; break;
  case RISCV::SH2ADD: ShAmt = 2; break;
  case RISCV::SH3ADD: ShAmt = 3; break;
  }

  Offset = (uint64_t)Offset << ShAmt;

  LLVM_DEBUG(dbgs() << "  Offset Instr: " << OffsetTail);
  DeadInstrs.insert(&OffsetTail);
  return true;
}

bool RISCVMergeBaseOffsetOpt::detectAndFoldOffset(MachineInstr &HiLUI,
                                                  MachineInstr &LoADDI) {
  Register DestReg = LoADDI.getOperand(0).getReg();
  assert(MRI->hasOneUse(DestReg) && "expected one use for LoADDI");
  // LoADDI has only one use.
  MachineInstr &Tail = *MRI->use_instr_begin(DestReg);
  switch (Tail.getOpcode()) {
  default:
    LLVM_DEBUG(dbgs() << "Don't know how to get offset from this instr:"
                      << Tail);
    return false;
  case RISCV::ADDI: {
    // Offset is simply an immediate operand.
    int64_t Offset = Tail.getOperand(2).getImm();

    // We might have two ADDIs in a row.
    Register TailDestReg = Tail.getOperand(0).getReg();
    if (MRI->hasOneUse(TailDestReg)) {
      MachineInstr &TailTail = *MRI->use_instr_begin(TailDestReg);
      if (TailTail.getOpcode() == RISCV::ADDI) {
        Offset += TailTail.getOperand(2).getImm();
        LLVM_DEBUG(dbgs() << "  Offset Instrs: " << Tail << TailTail);
        DeadInstrs.insert(&Tail);
        foldOffset(HiLUI, LoADDI, TailTail, Offset);
        return true;
      }
    }

    LLVM_DEBUG(dbgs() << "  Offset Instr: " << Tail);
    foldOffset(HiLUI, LoADDI, Tail, Offset);
    return true;
  }
  case RISCV::ADD: {
    // The offset is too large to fit in the immediate field of ADDI.
    // This can be in two forms:
    // 1) LUI hi_Offset followed by:
    //    ADDI lo_offset
    //    This happens in case the offset has non zero bits in
    //    both hi 20 and lo 12 bits.
    // 2) LUI (offset20)
    //    This happens in case the lower 12 bits of the offset are zeros.
    int64_t Offset;
    if (!matchLargeOffset(Tail, DestReg, Offset))
      return false;
    foldOffset(HiLUI, LoADDI, Tail, Offset);
    return true;
  }
  case RISCV::SH1ADD:
  case RISCV::SH2ADD:
  case RISCV::SH3ADD: {
    // The offset is too large to fit in the immediate field of ADDI.
    // It may be encoded as (SH2ADD (ADDI X0, C), DestReg) or
    // (SH3ADD (ADDI X0, C), DestReg).
    int64_t Offset;
    if (!matchShiftedOffset(Tail, DestReg, Offset))
      return false;
    foldOffset(HiLUI, LoADDI, Tail, Offset);
    return true;
  }
  case RISCV::LB:
  case RISCV::LH:
  case RISCV::LW:
  case RISCV::LBU:
  case RISCV::LHU:
  case RISCV::LWU:
  case RISCV::LD:
  case RISCV::FLH:
  case RISCV::FLW:
  case RISCV::FLD:
  case RISCV::SB:
  case RISCV::SH:
  case RISCV::SW:
  case RISCV::SD:
  case RISCV::FSH:
  case RISCV::FSW:
  case RISCV::FSD: {
    // Transforms the sequence:            Into:
    // HiLUI:  lui vreg1, %hi(foo)          --->  lui vreg1, %hi(foo+8)
    // LoADDI: addi vreg2, vreg1, %lo(foo)  --->  lw vreg3, lo(foo+8)(vreg1)
    // Tail:   lw vreg3, 8(vreg2)
    if (Tail.getOperand(1).isFI())
      return false;
    // Register defined by LoADDI should be used in the base part of the
    // load\store instruction. Otherwise, no folding possible.
    Register BaseAddrReg = Tail.getOperand(1).getReg();
    if (DestReg != BaseAddrReg)
      return false;
    MachineOperand &TailImmOp = Tail.getOperand(2);
    int64_t Offset = TailImmOp.getImm();
    // Update the offsets in global address lowering.
    HiLUI.getOperand(1).setOffset(Offset);
    // Update the immediate in the Tail instruction to add the offset.
    Tail.removeOperand(2);
    MachineOperand &ImmOp = LoADDI.getOperand(2);
    ImmOp.setOffset(Offset);
    Tail.addOperand(ImmOp);
    // Update the base reg in the Tail instruction to feed from LUI.
    // Output of HiLUI is only used in LoADDI, no need to use
    // MRI->replaceRegWith().
    Tail.getOperand(1).setReg(HiLUI.getOperand(0).getReg());
    DeadInstrs.insert(&LoADDI);
    return true;
  }
  }
  return false;
}

bool RISCVMergeBaseOffsetOpt::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  ST = &Fn.getSubtarget<RISCVSubtarget>();

  bool MadeChange = false;
  DeadInstrs.clear();
  MRI = &Fn.getRegInfo();
  for (MachineBasicBlock &MBB : Fn) {
    LLVM_DEBUG(dbgs() << "MBB: " << MBB.getName() << "\n");
    for (MachineInstr &HiLUI : MBB) {
      MachineInstr *LoADDI = nullptr;
      if (!detectLuiAddiGlobal(HiLUI, LoADDI))
        continue;
      LLVM_DEBUG(dbgs() << "  Found lowered global address with one use: "
                        << *LoADDI->getOperand(2).getGlobal() << "\n");
      // If the use count is only one, merge the offset
      MadeChange |= detectAndFoldOffset(HiLUI, *LoADDI);
    }
  }
  // Delete dead instructions.
  for (auto *MI : DeadInstrs)
    MI->eraseFromParent();
  return MadeChange;
}

/// Returns an instance of the Merge Base Offset Optimization pass.
FunctionPass *llvm::createRISCVMergeBaseOffsetOptPass() {
  return new RISCVMergeBaseOffsetOpt();
}
