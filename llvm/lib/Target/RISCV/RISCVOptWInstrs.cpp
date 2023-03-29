//===- RISCVOptWInstrs.cpp - MI W instruction optimizations ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass does some optimizations for *W instructions at the MI level.
//
// First it removes unneeded sext.w instructions. Either because the sign
// extended bits aren't consumed or because the input was already sign extended
// by an earlier instruction.
//
// Then it removes the -w suffix from each addiw and slliw instructions
// whenever all users are dependent only on the lower word of the result of the
// instruction. We do this only for addiw, slliw, and mulw because the -w forms
// are less compressible.
//
//===---------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-opt-w-instrs"
#define RISCV_OPT_W_INSTRS_NAME "RISC-V Optimize W Instructions"

STATISTIC(NumRemovedSExtW, "Number of removed sign-extensions");
STATISTIC(NumTransformedToWInstrs,
          "Number of instructions transformed to W-ops");

static cl::opt<bool> DisableSExtWRemoval("riscv-disable-sextw-removal",
                                         cl::desc("Disable removal of sext.w"),
                                         cl::init(false), cl::Hidden);
static cl::opt<bool> DisableStripWSuffix("riscv-disable-strip-w-suffix",
                                         cl::desc("Disable strip W suffix"),
                                         cl::init(false), cl::Hidden);

namespace {

class RISCVOptWInstrs : public MachineFunctionPass {
public:
  static char ID;

  RISCVOptWInstrs() : MachineFunctionPass(ID) {
    initializeRISCVOptWInstrsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  bool removeSExtWInstrs(MachineFunction &MF, const RISCVInstrInfo &TII,
                         const RISCVSubtarget &ST, MachineRegisterInfo &MRI);
  bool stripWSuffixes(MachineFunction &MF, const RISCVInstrInfo &TII,
                      const RISCVSubtarget &ST, MachineRegisterInfo &MRI);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_OPT_W_INSTRS_NAME; }
};

} // end anonymous namespace

char RISCVOptWInstrs::ID = 0;
INITIALIZE_PASS(RISCVOptWInstrs, DEBUG_TYPE, RISCV_OPT_W_INSTRS_NAME, false,
                false)

FunctionPass *llvm::createRISCVOptWInstrsPass() {
  return new RISCVOptWInstrs();
}

// Checks if all users only demand the lower \p OrigBits of the original
// instruction's result.
// TODO: handle multiple interdependent transformations
static bool hasAllNBitUsers(const MachineInstr &OrigMI,
                            const RISCVSubtarget &ST,
                            const MachineRegisterInfo &MRI, unsigned OrigBits) {

  SmallSet<std::pair<const MachineInstr *, unsigned>, 4> Visited;
  SmallVector<std::pair<const MachineInstr *, unsigned>, 4> Worklist;

  Worklist.push_back(std::make_pair(&OrigMI, OrigBits));

  while (!Worklist.empty()) {
    auto P = Worklist.pop_back_val();
    const MachineInstr *MI = P.first;
    unsigned Bits = P.second;

    if (!Visited.insert(P).second)
      continue;

    // Only handle instructions with one def.
    if (MI->getNumExplicitDefs() != 1)
      return false;

    for (auto &UserOp : MRI.use_operands(MI->getOperand(0).getReg())) {
      const MachineInstr *UserMI = UserOp.getParent();
      unsigned OpIdx = UserOp.getOperandNo();

      switch (UserMI->getOpcode()) {
      default:
        return false;

      case RISCV::ADDIW:
      case RISCV::ADDW:
      case RISCV::DIVUW:
      case RISCV::DIVW:
      case RISCV::MULW:
      case RISCV::REMUW:
      case RISCV::REMW:
      case RISCV::SLLIW:
      case RISCV::SLLW:
      case RISCV::SRAIW:
      case RISCV::SRAW:
      case RISCV::SRLIW:
      case RISCV::SRLW:
      case RISCV::SUBW:
      case RISCV::ROLW:
      case RISCV::RORW:
      case RISCV::RORIW:
      case RISCV::CLZW:
      case RISCV::CTZW:
      case RISCV::CPOPW:
      case RISCV::SLLI_UW:
      case RISCV::FMV_W_X:
      case RISCV::FCVT_H_W:
      case RISCV::FCVT_H_WU:
      case RISCV::FCVT_S_W:
      case RISCV::FCVT_S_WU:
      case RISCV::FCVT_D_W:
      case RISCV::FCVT_D_WU:
        if (Bits >= 32)
          break;
        return false;
      case RISCV::SEXT_B:
      case RISCV::PACKH:
        if (Bits >= 8)
          break;
        return false;
      case RISCV::SEXT_H:
      case RISCV::FMV_H_X:
      case RISCV::ZEXT_H_RV32:
      case RISCV::ZEXT_H_RV64:
      case RISCV::PACKW:
        if (Bits >= 16)
          break;
        return false;

      case RISCV::PACK:
        if (Bits >= (ST.getXLen() / 2))
          break;
        return false;

      case RISCV::SRLI: {
        // If we are shifting right by less than Bits, and users don't demand
        // any bits that were shifted into [Bits-1:0], then we can consider this
        // as an N-Bit user.
        unsigned ShAmt = UserMI->getOperand(2).getImm();
        if (Bits > ShAmt) {
          Worklist.push_back(std::make_pair(UserMI, Bits - ShAmt));
          break;
        }
        return false;
      }

      // these overwrite higher input bits, otherwise the lower word of output
      // depends only on the lower word of input. So check their uses read W.
      case RISCV::SLLI:
        if (Bits >= (ST.getXLen() - UserMI->getOperand(2).getImm()))
          break;
        Worklist.push_back(std::make_pair(UserMI, Bits));
        break;
      case RISCV::ANDI: {
        uint64_t Imm = UserMI->getOperand(2).getImm();
        if (Bits >= (unsigned)llvm::bit_width(Imm))
          break;
        Worklist.push_back(std::make_pair(UserMI, Bits));
        break;
      }
      case RISCV::ORI: {
        uint64_t Imm = UserMI->getOperand(2).getImm();
        if (Bits >= (unsigned)llvm::bit_width<uint64_t>(~Imm))
          break;
        Worklist.push_back(std::make_pair(UserMI, Bits));
        break;
      }

      case RISCV::SLL:
      case RISCV::BSET:
      case RISCV::BCLR:
      case RISCV::BINV:
        // Operand 2 is the shift amount which uses log2(xlen) bits.
        if (OpIdx == 2) {
          if (Bits >= Log2_32(ST.getXLen()))
            break;
          return false;
        }
        Worklist.push_back(std::make_pair(UserMI, Bits));
        break;

      case RISCV::SRA:
      case RISCV::SRL:
      case RISCV::ROL:
      case RISCV::ROR:
        // Operand 2 is the shift amount which uses 6 bits.
        if (OpIdx == 2 && Bits >= Log2_32(ST.getXLen()))
          break;
        return false;

      case RISCV::ADD_UW:
      case RISCV::SH1ADD_UW:
      case RISCV::SH2ADD_UW:
      case RISCV::SH3ADD_UW:
        // Operand 1 is implicitly zero extended.
        if (OpIdx == 1 && Bits >= 32)
          break;
        Worklist.push_back(std::make_pair(UserMI, Bits));
        break;

      case RISCV::BEXTI:
        if (UserMI->getOperand(2).getImm() >= Bits)
          return false;
        break;

      case RISCV::SB:
        // The first argument is the value to store.
        if (OpIdx == 0 && Bits >= 8)
          break;
        return false;
      case RISCV::SH:
        // The first argument is the value to store.
        if (OpIdx == 0 && Bits >= 16)
          break;
        return false;
      case RISCV::SW:
        // The first argument is the value to store.
        if (OpIdx == 0 && Bits >= 32)
          break;
        return false;

      // For these, lower word of output in these operations, depends only on
      // the lower word of input. So, we check all uses only read lower word.
      case RISCV::COPY:
      case RISCV::PHI:

      case RISCV::ADD:
      case RISCV::ADDI:
      case RISCV::AND:
      case RISCV::MUL:
      case RISCV::OR:
      case RISCV::SUB:
      case RISCV::XOR:
      case RISCV::XORI:

      case RISCV::ANDN:
      case RISCV::BREV8:
      case RISCV::CLMUL:
      case RISCV::ORC_B:
      case RISCV::ORN:
      case RISCV::SH1ADD:
      case RISCV::SH2ADD:
      case RISCV::SH3ADD:
      case RISCV::XNOR:
      case RISCV::BSETI:
      case RISCV::BCLRI:
      case RISCV::BINVI:
        Worklist.push_back(std::make_pair(UserMI, Bits));
        break;

      case RISCV::PseudoCCMOVGPR:
        // Either operand 4 or operand 5 is returned by this instruction. If
        // only the lower word of the result is used, then only the lower word
        // of operand 4 and 5 is used.
        if (OpIdx != 4 && OpIdx != 5)
          return false;
        Worklist.push_back(std::make_pair(UserMI, Bits));
        break;

      case RISCV::VT_MASKC:
      case RISCV::VT_MASKCN:
        if (OpIdx != 1)
          return false;
        Worklist.push_back(std::make_pair(UserMI, Bits));
        break;
      }
    }
  }

  return true;
}

static bool hasAllWUsers(const MachineInstr &OrigMI, const RISCVSubtarget &ST,
                         const MachineRegisterInfo &MRI) {
  return hasAllNBitUsers(OrigMI, ST, MRI, 32);
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

static bool isSignExtendedW(Register SrcReg, const RISCVSubtarget &ST,
                            const MachineRegisterInfo &MRI,
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
      if (hasAllWUsers(*MI, ST, MRI)) {
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

bool RISCVOptWInstrs::removeSExtWInstrs(MachineFunction &MF,
                                        const RISCVInstrInfo &TII,
                                        const RISCVSubtarget &ST,
                                        MachineRegisterInfo &MRI) {
  if (DisableSExtWRemoval)
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
      if (!hasAllWUsers(*MI, ST, MRI) &&
          !isSignExtendedW(SrcReg, ST, MRI, FixableDefs))
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

bool RISCVOptWInstrs::stripWSuffixes(MachineFunction &MF,
                                     const RISCVInstrInfo &TII,
                                     const RISCVSubtarget &ST,
                                     MachineRegisterInfo &MRI) {
  if (DisableStripWSuffix)
    return false;

  bool MadeChange = false;
  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), IE = MBB.end(); I != IE; ++I) {
      MachineInstr &MI = *I;

      unsigned Opc;
      switch (MI.getOpcode()) {
      default:
        continue;
      case RISCV::ADDW:  Opc = RISCV::ADD;  break;
      case RISCV::MULW:  Opc = RISCV::MUL;  break;
      case RISCV::SLLIW: Opc = RISCV::SLLI; break;
      }

      if (hasAllWUsers(MI, ST, MRI)) {
        MI.setDesc(TII.get(Opc));
        MadeChange = true;
      }
    }
  }

  return MadeChange;
}

bool RISCVOptWInstrs::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo &TII = *ST.getInstrInfo();

  if (!ST.is64Bit())
    return false;

  bool MadeChange = false;
  MadeChange |= removeSExtWInstrs(MF, TII, ST, MRI);
  MadeChange |= stripWSuffixes(MF, TII, ST, MRI);

  return MadeChange;
}
