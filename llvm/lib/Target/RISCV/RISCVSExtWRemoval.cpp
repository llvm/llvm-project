//===-------------- RISCVSExtWRemoval.cpp - MI sext.w Removal -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass removes unneeded sext.w instructions at the MI level. Either
// because the sign extended bits aren't consumed or because the input was
// already sign extended by an earlier instruction.
//
//===---------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-sextw-removal"

STATISTIC(NumRemovedSExtW, "Number of removed sign-extensions");
STATISTIC(NumTransformedToWInstrs,
          "Number of instructions transformed to W-ops");

static cl::opt<bool> DisableSExtWRemoval("riscv-disable-sextw-removal",
                                         cl::desc("Disable removal of sext.w"),
                                         cl::init(false), cl::Hidden);
namespace {

class RISCVSExtWRemoval : public MachineFunctionPass {
public:
  static char ID;

  RISCVSExtWRemoval() : MachineFunctionPass(ID) {
    initializeRISCVSExtWRemovalPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "RISCV sext.w Removal"; }
};

} // end anonymous namespace

char RISCVSExtWRemoval::ID = 0;
INITIALIZE_PASS(RISCVSExtWRemoval, DEBUG_TYPE, "RISCV sext.w Removal", false,
                false)

FunctionPass *llvm::createRISCVSExtWRemovalPass() {
  return new RISCVSExtWRemoval();
}

// This function returns true if the machine instruction always outputs a value
// where bits 63:32 match bit 31.
static bool isSignExtendingOpW(const MachineInstr &MI,
                               const MachineRegisterInfo &MRI) {
  uint64_t TSFlags = MI.getDesc().TSFlags;

  // Instructions that can be determined from opcode are marked in tablegen.
  if (TSFlags & RISCVII::IsSignExtendingOpWMask)
    return true;

  // Special cases that require checking operands.
  switch (MI.getOpcode()) {
  // shifting right sufficiently makes the value 32-bit sign-extended
  case RISCV::SRAI:
    return MI.getOperand(2).getImm() >= 32;
  case RISCV::SRLI:
    return MI.getOperand(2).getImm() > 32;
  // The LI pattern ADDI rd, X0, imm is sign extended.
  case RISCV::ADDI:
    return MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == RISCV::X0;
  // An ANDI with an 11 bit immediate will zero bits 63:11.
  case RISCV::ANDI:
    return isUInt<11>(MI.getOperand(2).getImm());
  // An ORI with an >11 bit immediate (negative 12-bit) will set bits 63:11.
  case RISCV::ORI:
    return !isUInt<11>(MI.getOperand(2).getImm());
  // Copying from X0 produces zero.
  case RISCV::COPY:
    return MI.getOperand(1).getReg() == RISCV::X0;
  }

  return false;
}

static bool isSignExtendedW(Register SrcReg, const MachineRegisterInfo &MRI,
                            const RISCVInstrInfo &TII,
                            SmallPtrSetImpl<MachineInstr *> &FixableDef) {

  SmallPtrSet<const MachineInstr *, 4> Visited;
  SmallVector<MachineInstr *, 4> Worklist;

  auto AddRegDefToWorkList = [&](Register SrcReg) {
    if (!SrcReg.isVirtual())
      return false;
    MachineInstr *SrcMI = MRI.getVRegDef(SrcReg);
    if (!SrcMI)
      return false;
    // Add SrcMI to the worklist.
    Worklist.push_back(SrcMI);
    return true;
  };

  if (!AddRegDefToWorkList(SrcReg))
    return false;

  while (!Worklist.empty()) {
    MachineInstr *MI = Worklist.pop_back_val();

    // If we already visited this instruction, we don't need to check it again.
    if (!Visited.insert(MI).second)
      continue;

    // If this is a sign extending operation we don't need to look any further.
    if (isSignExtendingOpW(*MI, MRI))
      continue;

    // Is this an instruction that propagates sign extend?
    switch (MI->getOpcode()) {
    default:
      // Unknown opcode, give up.
      return false;
    case RISCV::COPY: {
      const MachineFunction *MF = MI->getMF();
      const RISCVMachineFunctionInfo *RVFI =
          MF->getInfo<RISCVMachineFunctionInfo>();

      // If this is the entry block and the register is livein, see if we know
      // it is sign extended.
      if (MI->getParent() == &MF->front()) {
        Register VReg = MI->getOperand(0).getReg();
        if (MF->getRegInfo().isLiveIn(VReg) && RVFI->isSExt32Register(VReg))
          continue;
      }

      Register CopySrcReg = MI->getOperand(1).getReg();
      if (CopySrcReg == RISCV::X10) {
        // For a method return value, we check the ZExt/SExt flags in attribute.
        // We assume the following code sequence for method call.
        // PseudoCALL @bar, ...
        // ADJCALLSTACKUP 0, 0, implicit-def dead $x2, implicit $x2
        // %0:gpr = COPY $x10
        //
        // We use the PseudoCall to look up the IR function being called to find
        // its return attributes.
        const MachineBasicBlock *MBB = MI->getParent();
        auto II = MI->getIterator();
        if (II == MBB->instr_begin() ||
            (--II)->getOpcode() != RISCV::ADJCALLSTACKUP)
          return false;

        const MachineInstr &CallMI = *(--II);
        if (!CallMI.isCall() || !CallMI.getOperand(0).isGlobal())
          return false;

        auto *CalleeFn =
            dyn_cast_if_present<Function>(CallMI.getOperand(0).getGlobal());
        if (!CalleeFn)
          return false;

        auto *IntTy = dyn_cast<IntegerType>(CalleeFn->getReturnType());
        if (!IntTy)
          return false;

        const AttributeSet &Attrs = CalleeFn->getAttributes().getRetAttrs();
        unsigned BitWidth = IntTy->getBitWidth();
        if ((BitWidth <= 32 && Attrs.hasAttribute(Attribute::SExt)) ||
            (BitWidth < 32 && Attrs.hasAttribute(Attribute::ZExt)))
          continue;
      }

      if (!AddRegDefToWorkList(CopySrcReg))
        return false;

      break;
    }

    // For these, we just need to check if the 1st operand is sign extended.
    case RISCV::BCLRI:
    case RISCV::BINVI:
    case RISCV::BSETI:
      if (MI->getOperand(2).getImm() >= 31)
        return false;
      [[fallthrough]];
    case RISCV::REM:
    case RISCV::ANDI:
    case RISCV::ORI:
    case RISCV::XORI:
      // |Remainder| is always <= |Dividend|. If D is 32-bit, then so is R.
      // DIV doesn't work because of the edge case 0xf..f 8000 0000 / (long)-1
      // Logical operations use a sign extended 12-bit immediate.
      if (!AddRegDefToWorkList(MI->getOperand(1).getReg()))
        return false;

      break;
    case RISCV::PseudoCCADDW:
    case RISCV::PseudoCCSUBW:
      // Returns operand 4 or an ADDW/SUBW of operands 5 and 6. We only need to
      // check if operand 4 is sign extended.
      if (!AddRegDefToWorkList(MI->getOperand(4).getReg()))
        return false;
      break;
    case RISCV::REMU:
    case RISCV::AND:
    case RISCV::OR:
    case RISCV::XOR:
    case RISCV::ANDN:
    case RISCV::ORN:
    case RISCV::XNOR:
    case RISCV::MAX:
    case RISCV::MAXU:
    case RISCV::MIN:
    case RISCV::MINU:
    case RISCV::PseudoCCMOVGPR:
    case RISCV::PseudoCCAND:
    case RISCV::PseudoCCOR:
    case RISCV::PseudoCCXOR:
    case RISCV::PHI: {
      // If all incoming values are sign-extended, the output of AND, OR, XOR,
      // MIN, MAX, or PHI is also sign-extended.

      // The input registers for PHI are operand 1, 3, ...
      // The input registers for PseudoCCMOVGPR are 4 and 5.
      // The input registers for PseudoCCAND/OR/XOR are 4, 5, and 6.
      // The input registers for others are operand 1 and 2.
      unsigned B = 1, E = 3, D = 1;
      switch (MI->getOpcode()) {
      case RISCV::PHI:
        E = MI->getNumOperands();
        D = 2;
        break;
      case RISCV::PseudoCCMOVGPR:
        B = 4;
        E = 6;
        break;
      case RISCV::PseudoCCAND:
      case RISCV::PseudoCCOR:
      case RISCV::PseudoCCXOR:
        B = 4;
        E = 7;
        break;
       }

      for (unsigned I = B; I != E; I += D) {
        if (!MI->getOperand(I).isReg())
          return false;

        if (!AddRegDefToWorkList(MI->getOperand(I).getReg()))
          return false;
      }

      break;
    }

    case RISCV::VT_MASKC:
    case RISCV::VT_MASKCN:
      // Instructions return zero or operand 1. Result is sign extended if
      // operand 1 is sign extended.
      if (!AddRegDefToWorkList(MI->getOperand(1).getReg()))
        return false;
      break;

    // With these opcode, we can "fix" them with the W-version
    // if we know all users of the result only rely on bits 31:0
    case RISCV::SLLI:
      // SLLIW reads the lowest 5 bits, while SLLI reads lowest 6 bits
      if (MI->getOperand(2).getImm() >= 32)
        return false;
      [[fallthrough]];
    case RISCV::ADDI:
    case RISCV::ADD:
    case RISCV::LD:
    case RISCV::LWU:
    case RISCV::MUL:
    case RISCV::SUB:
      if (TII.hasAllWUsers(*MI, MRI)) {
        FixableDef.insert(MI);
        break;
      }
      return false;
    }
  }

  // If we get here, then every node we visited produces a sign extended value
  // or propagated sign extended values. So the result must be sign extended.
  return true;
}

static unsigned getWOp(unsigned Opcode) {
  switch (Opcode) {
  case RISCV::ADDI:
    return RISCV::ADDIW;
  case RISCV::ADD:
    return RISCV::ADDW;
  case RISCV::LD:
  case RISCV::LWU:
    return RISCV::LW;
  case RISCV::MUL:
    return RISCV::MULW;
  case RISCV::SLLI:
    return RISCV::SLLIW;
  case RISCV::SUB:
    return RISCV::SUBW;
  default:
    llvm_unreachable("Unexpected opcode for replacement with W variant");
  }
}

bool RISCVSExtWRemoval::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()) || DisableSExtWRemoval)
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo &TII = *ST.getInstrInfo();

  if (!ST.is64Bit())
    return false;

  bool MadeChange = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), IE = MBB.end(); I != IE;) {
      MachineInstr *MI = &*I++;

      // We're looking for the sext.w pattern ADDIW rd, rs1, 0.
      if (!RISCV::isSEXT_W(*MI))
        continue;

      Register SrcReg = MI->getOperand(1).getReg();

      SmallPtrSet<MachineInstr *, 4> FixableDefs;

      // If all users only use the lower bits, this sext.w is redundant.
      // Or if all definitions reaching MI sign-extend their output,
      // then sext.w is redundant.
      if (!TII.hasAllWUsers(*MI, MRI) &&
          !isSignExtendedW(SrcReg, MRI, TII, FixableDefs))
        continue;

      Register DstReg = MI->getOperand(0).getReg();
      if (!MRI.constrainRegClass(SrcReg, MRI.getRegClass(DstReg)))
        continue;

      // Convert Fixable instructions to their W versions.
      for (MachineInstr *Fixable : FixableDefs) {
        LLVM_DEBUG(dbgs() << "Replacing " << *Fixable);
        Fixable->setDesc(TII.get(getWOp(Fixable->getOpcode())));
        Fixable->clearFlag(MachineInstr::MIFlag::NoSWrap);
        Fixable->clearFlag(MachineInstr::MIFlag::NoUWrap);
        Fixable->clearFlag(MachineInstr::MIFlag::IsExact);
        LLVM_DEBUG(dbgs() << "     with " << *Fixable);
        ++NumTransformedToWInstrs;
      }

      LLVM_DEBUG(dbgs() << "Removing redundant sign-extension\n");
      MRI.replaceRegWith(DstReg, SrcReg);
      MRI.clearKillFlags(SrcReg);
      MI->eraseFromParent();
      ++NumRemovedSExtW;
      MadeChange = true;
    }
  }

  return MadeChange;
}
