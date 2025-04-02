//===---- LoongArchMergeBaseOffset.cpp - Optimise address calculations ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Merge the offset of address calculation into the offset field
// of instructions in a global address lowering sequence.
//
//===----------------------------------------------------------------------===//

#include "LoongArch.h"
#include "LoongArchTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetOptions.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "loongarch-merge-base-offset"
#define LoongArch_MERGE_BASE_OFFSET_NAME "LoongArch Merge Base Offset"

namespace {

class LoongArchMergeBaseOffsetOpt : public MachineFunctionPass {
  const LoongArchSubtarget *ST = nullptr;
  MachineRegisterInfo *MRI;

public:
  static char ID;
  bool runOnMachineFunction(MachineFunction &Fn) override;
  bool detectFoldable(MachineInstr &Hi20, MachineInstr *&Lo12,
                      MachineInstr *&Lo20, MachineInstr *&Hi12,
                      MachineInstr *&Last);

  bool detectAndFoldOffset(MachineInstr &Hi20, MachineInstr &Lo12,
                           MachineInstr *&Lo20, MachineInstr *&Hi12,
                           MachineInstr *&Last);
  void foldOffset(MachineInstr &Hi20, MachineInstr &Lo12, MachineInstr *&Lo20,
                  MachineInstr *&Hi12, MachineInstr *&Last, MachineInstr &Tail,
                  int64_t Offset);
  bool foldLargeOffset(MachineInstr &Hi20, MachineInstr &Lo12,
                       MachineInstr *&Lo20, MachineInstr *&Hi12,
                       MachineInstr *&Last, MachineInstr &TailAdd,
                       Register GAReg);

  bool foldIntoMemoryOps(MachineInstr &Hi20, MachineInstr &Lo12,
                         MachineInstr *&Lo20, MachineInstr *&Hi12,
                         MachineInstr *&Last);

  LoongArchMergeBaseOffsetOpt() : MachineFunctionPass(ID) {}

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return LoongArch_MERGE_BASE_OFFSET_NAME;
  }
};
} // end anonymous namespace

char LoongArchMergeBaseOffsetOpt::ID = 0;
INITIALIZE_PASS(LoongArchMergeBaseOffsetOpt, DEBUG_TYPE,
                LoongArch_MERGE_BASE_OFFSET_NAME, false, false)

// Detect either of the patterns:
//
// 1. (small/medium):
//   pcalau12i vreg1, %pc_hi20(s)
//   addi.d    vreg2, vreg1, %pc_lo12(s)
//
// 2. (large):
//   pcalau12i vreg1, %pc_hi20(s)
//   addi.d    vreg2, $zero, %pc_lo12(s)
//   lu32i.d   vreg3, vreg2, %pc64_lo20(s)
//   lu52i.d   vreg4, vreg3, %pc64_hi12(s)
//   add.d     vreg5, vreg4, vreg1

// The pattern is only accepted if:
//    1) For small and medium pattern, the first instruction has only one use,
//       which is the ADDI.
//    2) For large pattern, the first four instructions each have only one use,
//       and the user of the fourth instruction is ADD.
//    3) The address operands have the appropriate type, reflecting the
//       lowering of a global address or constant pool using the pattern.
//    4) The offset value in the Global Address or Constant Pool is 0.
bool LoongArchMergeBaseOffsetOpt::detectFoldable(MachineInstr &Hi20,
                                                 MachineInstr *&Lo12,
                                                 MachineInstr *&Lo20,
                                                 MachineInstr *&Hi12,
                                                 MachineInstr *&Last) {
  if (Hi20.getOpcode() != LoongArch::PCALAU12I)
    return false;

  const MachineOperand &Hi20Op1 = Hi20.getOperand(1);
  if (LoongArchII::getDirectFlags(Hi20Op1) != LoongArchII::MO_PCREL_HI)
    return false;

  auto isGlobalOrCPIOrBlockAddress = [](const MachineOperand &Op) {
    return Op.isGlobal() || Op.isCPI() || Op.isBlockAddress();
  };

  if (!isGlobalOrCPIOrBlockAddress(Hi20Op1) || Hi20Op1.getOffset() != 0)
    return false;

  Register HiDestReg = Hi20.getOperand(0).getReg();
  if (!MRI->hasOneUse(HiDestReg))
    return false;

  MachineInstr *UseInst = &*MRI->use_instr_begin(HiDestReg);
  if (UseInst->getOpcode() != LoongArch::ADD_D) {
    Lo12 = UseInst;
    if ((ST->is64Bit() && Lo12->getOpcode() != LoongArch::ADDI_D) ||
        (!ST->is64Bit() && Lo12->getOpcode() != LoongArch::ADDI_W))
      return false;
  } else {
    assert(ST->is64Bit());
    Last = UseInst;

    Register LastOp1Reg = Last->getOperand(1).getReg();
    if (!LastOp1Reg.isVirtual())
      return false;
    Hi12 = MRI->getVRegDef(LastOp1Reg);
    const MachineOperand &Hi12Op2 = Hi12->getOperand(2);
    if (Hi12Op2.getTargetFlags() != LoongArchII::MO_PCREL64_HI)
      return false;
    if (!isGlobalOrCPIOrBlockAddress(Hi12Op2) || Hi12Op2.getOffset() != 0)
      return false;
    if (!MRI->hasOneUse(Hi12->getOperand(0).getReg()))
      return false;

    Lo20 = MRI->getVRegDef(Hi12->getOperand(1).getReg());
    const MachineOperand &Lo20Op2 = Lo20->getOperand(2);
    if (Lo20Op2.getTargetFlags() != LoongArchII::MO_PCREL64_LO)
      return false;
    if (!isGlobalOrCPIOrBlockAddress(Lo20Op2) || Lo20Op2.getOffset() != 0)
      return false;
    if (!MRI->hasOneUse(Lo20->getOperand(0).getReg()))
      return false;

    Lo12 = MRI->getVRegDef(Lo20->getOperand(1).getReg());
    if (!MRI->hasOneUse(Lo12->getOperand(0).getReg()))
      return false;
  }

  const MachineOperand &Lo12Op2 = Lo12->getOperand(2);
  assert(Hi20.getOpcode() == LoongArch::PCALAU12I);
  if (LoongArchII::getDirectFlags(Lo12Op2) != LoongArchII::MO_PCREL_LO ||
      !(isGlobalOrCPIOrBlockAddress(Lo12Op2) || Lo12Op2.isMCSymbol()) ||
      Lo12Op2.getOffset() != 0)
    return false;

  if (Hi20Op1.isGlobal()) {
    LLVM_DEBUG(dbgs() << "  Found lowered global address: "
                      << *Hi20Op1.getGlobal() << "\n");
  } else if (Hi20Op1.isBlockAddress()) {
    LLVM_DEBUG(dbgs() << "  Found lowered basic address: "
                      << *Hi20Op1.getBlockAddress() << "\n");
  } else if (Hi20Op1.isCPI()) {
    LLVM_DEBUG(dbgs() << "  Found lowered constant pool: " << Hi20Op1.getIndex()
                      << "\n");
  }

  return true;
}

// Update the offset in Hi20, Lo12, Lo20 and Hi12 instructions.
// Delete the tail instruction and update all the uses to use the
// output from Last.
void LoongArchMergeBaseOffsetOpt::foldOffset(
    MachineInstr &Hi20, MachineInstr &Lo12, MachineInstr *&Lo20,
    MachineInstr *&Hi12, MachineInstr *&Last, MachineInstr &Tail,
    int64_t Offset) {
  // Put the offset back in Hi and the Lo
  Hi20.getOperand(1).setOffset(Offset);
  Lo12.getOperand(2).setOffset(Offset);
  if (Lo20 && Hi12) {
    Lo20->getOperand(2).setOffset(Offset);
    Hi12->getOperand(2).setOffset(Offset);
  }
  // Delete the tail instruction.
  MachineInstr *Def = Last ? Last : &Lo12;
  MRI->constrainRegClass(Def->getOperand(0).getReg(),
                         MRI->getRegClass(Tail.getOperand(0).getReg()));
  MRI->replaceRegWith(Tail.getOperand(0).getReg(), Def->getOperand(0).getReg());
  Tail.eraseFromParent();
  LLVM_DEBUG(dbgs() << "  Merged offset " << Offset << " into base.\n"
                    << "     " << Hi20 << "     " << Lo12;);
  if (Lo20 && Hi12) {
    LLVM_DEBUG(dbgs() << "     " << *Lo20 << "     " << *Hi12;);
  }
}

// Detect patterns for large offsets that are passed into an ADD instruction.
// If the pattern is found, updates the offset in Hi20, Lo12, Lo20 and Hi12
// instructions and deletes TailAdd and the instructions that produced the
// offset.
//
//   (The instructions marked with "!" are not necessarily present)
//
//        Base address lowering is of the form:
//           Hi20:  pcalau12i vreg1, %pc_hi20(s)
//        +- Lo12:  addi.d vreg2, vreg1, %pc_lo12(s)
//        |  Lo20:  lu32i.d vreg2, %pc64_lo20(s) !
//        +- Hi12:  lu52i.d vreg2, vreg2, %pc64_hi12(s) !
//        |
//        | The large offset can be one of the forms:
//        |
//        +-> 1) Offset that has non zero bits in Hi20 and Lo12 bits:
//        |     OffsetHi20: lu12i.w vreg3, 4
//        |     OffsetLo12: ori voff, vreg3, 188    ------------------+
//        |                                                           |
//        +-> 2) Offset that has non zero bits in Hi20 bits only:     |
//        |     OffsetHi20: lu12i.w voff, 128       ------------------+
//        |                                                           |
//        +-> 3) Offset that has non zero bits in Lo20 bits:          |
//        |     OffsetHi20: lu12i.w vreg3, 121 !                      |
//        |     OffsetLo12: ori voff, vreg3, 122 !                    |
//        |     OffsetLo20: lu32i.d voff, 123       ------------------+
//        +-> 4) Offset that has non zero bits in Hi12 bits:          |
//              OffsetHi20: lu12i.w vreg3, 121 !                      |
//              OffsetLo12: ori voff, vreg3, 122 !                    |
//              OffsetLo20: lu32i.d vreg3, 123 !                      |
//              OffsetHi12: lu52i.d voff, vrg3, 124 ------------------+
//                                                                    |
//        TailAdd: add.d  vreg4, vreg2, voff       <------------------+
//
bool LoongArchMergeBaseOffsetOpt::foldLargeOffset(
    MachineInstr &Hi20, MachineInstr &Lo12, MachineInstr *&Lo20,
    MachineInstr *&Hi12, MachineInstr *&Last, MachineInstr &TailAdd,
    Register GAReg) {
  assert((TailAdd.getOpcode() == LoongArch::ADD_W ||
          TailAdd.getOpcode() == LoongArch::ADD_D) &&
         "Expected ADD instruction!");
  Register Rs = TailAdd.getOperand(1).getReg();
  Register Rt = TailAdd.getOperand(2).getReg();
  Register Reg = Rs == GAReg ? Rt : Rs;
  SmallVector<MachineInstr *, 4> Instrs;
  int64_t Offset = 0;
  int64_t Mask = -1;

  // This can point to one of [ORI, LU12I.W, LU32I.D, LU52I.D]:
  for (int i = 0; i < 4; i++) {
    // Handle Reg is R0.
    if (Reg == LoongArch::R0)
      break;

    // Can't fold if the register has more than one use.
    if (!Reg.isVirtual() || !MRI->hasOneUse(Reg))
      return false;

    MachineInstr *Curr = MRI->getVRegDef(Reg);
    if (!Curr)
      break;

    switch (Curr->getOpcode()) {
    default:
      // Can't fold if the instruction opcode is unexpected.
      return false;
    case LoongArch::ORI: {
      MachineOperand ImmOp = Curr->getOperand(2);
      if (ImmOp.getTargetFlags() != LoongArchII::MO_None)
        return false;
      Offset += ImmOp.getImm();
      Reg = Curr->getOperand(1).getReg();
      Instrs.push_back(Curr);
      break;
    }
    case LoongArch::LU12I_W: {
      MachineOperand ImmOp = Curr->getOperand(1);
      if (ImmOp.getTargetFlags() != LoongArchII::MO_None)
        return false;
      Offset += SignExtend64<32>(ImmOp.getImm() << 12) & Mask;
      Reg = LoongArch::R0;
      Instrs.push_back(Curr);
      break;
    }
    case LoongArch::LU32I_D: {
      MachineOperand ImmOp = Curr->getOperand(2);
      if (ImmOp.getTargetFlags() != LoongArchII::MO_None || !Lo20)
        return false;
      Offset += SignExtend64<52>(ImmOp.getImm() << 32) & Mask;
      Mask ^= 0x000FFFFF00000000ULL;
      Reg = Curr->getOperand(1).getReg();
      Instrs.push_back(Curr);
      break;
    }
    case LoongArch::LU52I_D: {
      MachineOperand ImmOp = Curr->getOperand(2);
      if (ImmOp.getTargetFlags() != LoongArchII::MO_None || !Hi12)
        return false;
      Offset += ImmOp.getImm() << 52;
      Mask ^= 0xFFF0000000000000ULL;
      Reg = Curr->getOperand(1).getReg();
      Instrs.push_back(Curr);
      break;
    }
    }
  }

  // Can't fold if the offset is not extracted.
  if (!Offset)
    return false;

  foldOffset(Hi20, Lo12, Lo20, Hi12, Last, TailAdd, Offset);
  LLVM_DEBUG(dbgs() << "  Offset Instrs:\n");
  for (auto I : Instrs) {
    LLVM_DEBUG(dbgs() << "                 " << *I);
    I->eraseFromParent();
  }

  return true;
}

bool LoongArchMergeBaseOffsetOpt::detectAndFoldOffset(MachineInstr &Hi20,
                                                      MachineInstr &Lo12,
                                                      MachineInstr *&Lo20,
                                                      MachineInstr *&Hi12,
                                                      MachineInstr *&Last) {
  Register DestReg =
      Last ? Last->getOperand(0).getReg() : Lo12.getOperand(0).getReg();

  // Look for arithmetic instructions we can get an offset from.
  // We might be able to remove the arithmetic instructions by folding the
  // offset into the PCALAU12I+(ADDI/ADDI+LU32I+LU52I).
  if (!MRI->hasOneUse(DestReg))
    return false;

  // DestReg has only one use.
  MachineInstr &Tail = *MRI->use_instr_begin(DestReg);
  switch (Tail.getOpcode()) {
  default:
    LLVM_DEBUG(dbgs() << "Don't know how to get offset from this instr:"
                      << Tail);
    break;
  case LoongArch::ADDI_W:
    if (ST->is64Bit())
      return false;
    [[fallthrough]];
  case LoongArch::ADDI_D:
  case LoongArch::ADDU16I_D: {
    // Offset is simply an immediate operand.
    int64_t Offset = Tail.getOperand(2).getImm();
    if (Tail.getOpcode() == LoongArch::ADDU16I_D)
      Offset = SignExtend64<32>(Offset << 16);

    // We might have two ADDIs in a row.
    Register TailDestReg = Tail.getOperand(0).getReg();
    if (MRI->hasOneUse(TailDestReg)) {
      MachineInstr &TailTail = *MRI->use_instr_begin(TailDestReg);
      if (ST->is64Bit() && TailTail.getOpcode() == LoongArch::ADDI_W)
        return false;
      if (TailTail.getOpcode() == LoongArch::ADDI_W ||
          TailTail.getOpcode() == LoongArch::ADDI_D) {
        Offset += TailTail.getOperand(2).getImm();
        LLVM_DEBUG(dbgs() << "  Offset Instrs: " << Tail << TailTail);
        foldOffset(Hi20, Lo12, Lo20, Hi12, Last, TailTail, Offset);
        Tail.eraseFromParent();
        return true;
      }
    }

    LLVM_DEBUG(dbgs() << "  Offset Instr: " << Tail);
    foldOffset(Hi20, Lo12, Lo20, Hi12, Last, Tail, Offset);
    return true;
  }
  case LoongArch::ADD_W:
    if (ST->is64Bit())
      return false;
    [[fallthrough]];
  case LoongArch::ADD_D:
    // The offset is too large to fit in the immediate field of ADDI.
    return foldLargeOffset(Hi20, Lo12, Lo20, Hi12, Last, Tail, DestReg);
    break;
  }

  return false;
}

// Memory access opcode mapping for transforms.
static unsigned getNewOpc(unsigned Op, bool isLarge) {
  switch (Op) {
  case LoongArch::LD_B:
    return isLarge ? LoongArch::LDX_B : LoongArch::LD_B;
  case LoongArch::LD_H:
    return isLarge ? LoongArch::LDX_H : LoongArch::LD_H;
  case LoongArch::LD_W:
  case LoongArch::LDPTR_W:
    return isLarge ? LoongArch::LDX_W : LoongArch::LD_W;
  case LoongArch::LD_D:
  case LoongArch::LDPTR_D:
    return isLarge ? LoongArch::LDX_D : LoongArch::LD_D;
  case LoongArch::LD_BU:
    return isLarge ? LoongArch::LDX_BU : LoongArch::LD_BU;
  case LoongArch::LD_HU:
    return isLarge ? LoongArch::LDX_HU : LoongArch::LD_HU;
  case LoongArch::LD_WU:
    return isLarge ? LoongArch::LDX_WU : LoongArch::LD_WU;
  case LoongArch::FLD_S:
    return isLarge ? LoongArch::FLDX_S : LoongArch::FLD_S;
  case LoongArch::FLD_D:
    return isLarge ? LoongArch::FLDX_D : LoongArch::FLD_D;
  case LoongArch::VLD:
    return isLarge ? LoongArch::VLDX : LoongArch::VLD;
  case LoongArch::XVLD:
    return isLarge ? LoongArch::XVLDX : LoongArch::XVLD;
  case LoongArch::VLDREPL_B:
    return LoongArch::VLDREPL_B;
  case LoongArch::XVLDREPL_B:
    return LoongArch::XVLDREPL_B;
  case LoongArch::ST_B:
    return isLarge ? LoongArch::STX_B : LoongArch::ST_B;
  case LoongArch::ST_H:
    return isLarge ? LoongArch::STX_H : LoongArch::ST_H;
  case LoongArch::ST_W:
  case LoongArch::STPTR_W:
    return isLarge ? LoongArch::STX_W : LoongArch::ST_W;
  case LoongArch::ST_D:
  case LoongArch::STPTR_D:
    return isLarge ? LoongArch::STX_D : LoongArch::ST_D;
  case LoongArch::FST_S:
    return isLarge ? LoongArch::FSTX_S : LoongArch::FST_S;
  case LoongArch::FST_D:
    return isLarge ? LoongArch::FSTX_D : LoongArch::FST_D;
  case LoongArch::VST:
    return isLarge ? LoongArch::VSTX : LoongArch::VST;
  case LoongArch::XVST:
    return isLarge ? LoongArch::XVSTX : LoongArch::XVST;
  default:
    llvm_unreachable("Unexpected opcode for replacement");
  }
}

bool LoongArchMergeBaseOffsetOpt::foldIntoMemoryOps(MachineInstr &Hi20,
                                                    MachineInstr &Lo12,
                                                    MachineInstr *&Lo20,
                                                    MachineInstr *&Hi12,
                                                    MachineInstr *&Last) {
  Register DestReg =
      Last ? Last->getOperand(0).getReg() : Lo12.getOperand(0).getReg();

  // If all the uses are memory ops with the same offset, we can transform:
  //
  // 1. (small/medium):
  //   pcalau12i vreg1, %pc_hi20(s)
  //   addi.d    vreg2, vreg1, %pc_lo12(s)
  //   ld.w      vreg3, 8(vreg2)
  //
  //   =>
  //
  //   pcalau12i vreg1, %pc_hi20(s+8)
  //   ld.w      vreg3, vreg1, %pc_lo12(s+8)(vreg1)
  //
  // 2. (large):
  //   pcalau12i vreg1, %pc_hi20(s)
  //   addi.d    vreg2, $zero, %pc_lo12(s)
  //   lu32i.d   vreg3, vreg2, %pc64_lo20(s)
  //   lu52i.d   vreg4, vreg3, %pc64_hi12(s)
  //   add.d     vreg5, vreg4, vreg1
  //   ld.w      vreg6, 8(vreg5)
  //
  //   =>
  //
  //   pcalau12i vreg1, %pc_hi20(s+8)
  //   addi.d    vreg2, $zero, %pc_lo12(s+8)
  //   lu32i.d   vreg3, vreg2, %pc64_lo20(s+8)
  //   lu52i.d   vreg4, vreg3, %pc64_hi12(s+8)
  //   ldx.w     vreg6, vreg4, vreg1

  std::optional<int64_t> CommonOffset;
  DenseMap<const MachineInstr *, SmallVector<unsigned>>
      InlineAsmMemoryOpIndexesMap;
  for (const MachineInstr &UseMI : MRI->use_instructions(DestReg)) {
    switch (UseMI.getOpcode()) {
    default:
      LLVM_DEBUG(dbgs() << "Not a load or store instruction: " << UseMI);
      return false;
    case LoongArch::VLDREPL_B:
    case LoongArch::XVLDREPL_B:
      // We can't do this for large pattern.
      if (Last)
        return false;
      [[fallthrough]];
    case LoongArch::LD_B:
    case LoongArch::LD_H:
    case LoongArch::LD_W:
    case LoongArch::LD_D:
    case LoongArch::LD_BU:
    case LoongArch::LD_HU:
    case LoongArch::LD_WU:
    case LoongArch::LDPTR_W:
    case LoongArch::LDPTR_D:
    case LoongArch::FLD_S:
    case LoongArch::FLD_D:
    case LoongArch::VLD:
    case LoongArch::XVLD:
    case LoongArch::ST_B:
    case LoongArch::ST_H:
    case LoongArch::ST_W:
    case LoongArch::ST_D:
    case LoongArch::STPTR_W:
    case LoongArch::STPTR_D:
    case LoongArch::FST_S:
    case LoongArch::FST_D:
    case LoongArch::VST:
    case LoongArch::XVST: {
      if (UseMI.getOperand(1).isFI())
        return false;
      // Register defined by Lo should not be the value register.
      if (DestReg == UseMI.getOperand(0).getReg())
        return false;
      assert(DestReg == UseMI.getOperand(1).getReg() &&
             "Expected base address use");
      // All load/store instructions must use the same offset.
      int64_t Offset = UseMI.getOperand(2).getImm();
      if (CommonOffset && Offset != CommonOffset)
        return false;
      CommonOffset = Offset;
      break;
    }
    case LoongArch::INLINEASM:
    case LoongArch::INLINEASM_BR: {
      // We can't do this for large pattern.
      if (Last)
        return false;
      SmallVector<unsigned> InlineAsmMemoryOpIndexes;
      unsigned NumOps = 0;
      for (unsigned I = InlineAsm::MIOp_FirstOperand;
           I < UseMI.getNumOperands(); I += 1 + NumOps) {
        const MachineOperand &FlagsMO = UseMI.getOperand(I);
        // Should be an imm.
        if (!FlagsMO.isImm())
          continue;

        const InlineAsm::Flag Flags(FlagsMO.getImm());
        NumOps = Flags.getNumOperandRegisters();

        // Memory constraints have two operands.
        if (NumOps != 2 || !Flags.isMemKind()) {
          // If the register is used by something other than a memory contraint,
          // we should not fold.
          for (unsigned J = 0; J < NumOps; ++J) {
            const MachineOperand &MO = UseMI.getOperand(I + 1 + J);
            if (MO.isReg() && MO.getReg() == DestReg)
              return false;
          }
          continue;
        }

        // We can only do this for constraint m.
        if (Flags.getMemoryConstraintID() != InlineAsm::ConstraintCode::m)
          return false;

        const MachineOperand &AddrMO = UseMI.getOperand(I + 1);
        if (!AddrMO.isReg() || AddrMO.getReg() != DestReg)
          continue;

        const MachineOperand &OffsetMO = UseMI.getOperand(I + 2);
        if (!OffsetMO.isImm())
          continue;

        // All inline asm memory operands must use the same offset.
        int64_t Offset = OffsetMO.getImm();
        if (CommonOffset && Offset != CommonOffset)
          return false;
        CommonOffset = Offset;
        InlineAsmMemoryOpIndexes.push_back(I + 1);
      }
      InlineAsmMemoryOpIndexesMap.insert(
          std::make_pair(&UseMI, InlineAsmMemoryOpIndexes));
      break;
    }
    }
  }

  // We found a common offset.
  // Update the offsets in global address lowering.
  // We may have already folded some arithmetic so we need to add to any
  // existing offset.
  int64_t NewOffset = Hi20.getOperand(1).getOffset() + *CommonOffset;
  // LA32 ignores the upper 32 bits.
  if (!ST->is64Bit())
    NewOffset = SignExtend64<32>(NewOffset);
  // We can only fold simm32 offsets.
  if (!isInt<32>(NewOffset))
    return false;

  // If optimized by this pass successfully, MO_RELAX bitmask target-flag should
  // be removed from the code sequence.
  //
  // For example:
  //   pcalau12i $a0, %pc_hi20(symbol)
  //   addi.d $a0, $a0, %pc_lo12(symbol)
  //   ld.w $a0, $a0, 0
  //
  //   =>
  //
  //   pcalau12i $a0, %pc_hi20(symbol)
  //   ld.w $a0, $a0, %pc_lo12(symbol)
  //
  // Code sequence optimized before can be relax by linker. But after being
  // optimized, it cannot be relaxed any more. So MO_RELAX flag should not be
  // carried by them.
  Hi20.getOperand(1).setOffset(NewOffset);
  Hi20.getOperand(1).setTargetFlags(
      LoongArchII::getDirectFlags(Hi20.getOperand(1)));
  MachineOperand &ImmOp = Lo12.getOperand(2);
  ImmOp.setOffset(NewOffset);
  ImmOp.setTargetFlags(LoongArchII::getDirectFlags(ImmOp));
  if (Lo20 && Hi12) {
    Lo20->getOperand(2).setOffset(NewOffset);
    Hi12->getOperand(2).setOffset(NewOffset);
  }

  // Update the immediate in the load/store instructions to add the offset.
  const LoongArchInstrInfo &TII = *ST->getInstrInfo();
  for (MachineInstr &UseMI :
       llvm::make_early_inc_range(MRI->use_instructions(DestReg))) {
    if (UseMI.getOpcode() == LoongArch::INLINEASM ||
        UseMI.getOpcode() == LoongArch::INLINEASM_BR) {
      auto &InlineAsmMemoryOpIndexes = InlineAsmMemoryOpIndexesMap[&UseMI];
      for (unsigned I : InlineAsmMemoryOpIndexes) {
        MachineOperand &MO = UseMI.getOperand(I + 1);
        switch (ImmOp.getType()) {
        case MachineOperand::MO_GlobalAddress:
          MO.ChangeToGA(ImmOp.getGlobal(), ImmOp.getOffset(),
                        LoongArchII::getDirectFlags(ImmOp));
          break;
        case MachineOperand::MO_MCSymbol:
          MO.ChangeToMCSymbol(ImmOp.getMCSymbol(),
                              LoongArchII::getDirectFlags(ImmOp));
          MO.setOffset(ImmOp.getOffset());
          break;
        case MachineOperand::MO_BlockAddress:
          MO.ChangeToBA(ImmOp.getBlockAddress(), ImmOp.getOffset(),
                        LoongArchII::getDirectFlags(ImmOp));
          break;
        default:
          report_fatal_error("unsupported machine operand type");
          break;
        }
      }
    } else {
      UseMI.setDesc(TII.get(getNewOpc(UseMI.getOpcode(), Last)));
      if (Last) {
        UseMI.removeOperand(2);
        UseMI.removeOperand(1);
        UseMI.addOperand(Last->getOperand(1));
        UseMI.addOperand(Last->getOperand(2));
        UseMI.getOperand(1).setIsKill(false);
        UseMI.getOperand(2).setIsKill(false);
      } else {
        UseMI.removeOperand(2);
        UseMI.addOperand(ImmOp);
      }
    }
  }

  if (Last) {
    Last->eraseFromParent();
    return true;
  }

  MRI->replaceRegWith(Lo12.getOperand(0).getReg(), Hi20.getOperand(0).getReg());
  Lo12.eraseFromParent();
  return true;
}

bool LoongArchMergeBaseOffsetOpt::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  ST = &Fn.getSubtarget<LoongArchSubtarget>();

  bool MadeChange = false;
  MRI = &Fn.getRegInfo();
  for (MachineBasicBlock &MBB : Fn) {
    LLVM_DEBUG(dbgs() << "MBB: " << MBB.getName() << "\n");
    for (MachineInstr &Hi20 : MBB) {
      MachineInstr *Lo12 = nullptr;
      MachineInstr *Lo20 = nullptr;
      MachineInstr *Hi12 = nullptr;
      MachineInstr *Last = nullptr;
      if (!detectFoldable(Hi20, Lo12, Lo20, Hi12, Last))
        continue;
      MadeChange |= detectAndFoldOffset(Hi20, *Lo12, Lo20, Hi12, Last);
      MadeChange |= foldIntoMemoryOps(Hi20, *Lo12, Lo20, Hi12, Last);
    }
  }

  return MadeChange;
}

/// Returns an instance of the Merge Base Offset Optimization pass.
FunctionPass *llvm::createLoongArchMergeBaseOffsetOptPass() {
  return new LoongArchMergeBaseOffsetOpt();
}
