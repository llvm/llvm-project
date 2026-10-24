//=- AArch64RedundantCondBranch.cpp - Remove redundant conditional branches -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Late in the pipeline, especially with zero phi operands propagated after tail
// duplications, we can end up with CBZ/CBNZ/TBZ/TBNZ with a zero register. This
// simple pass looks at the terminators to a block, removing the redundant
// instructions where necessary.
//
// The same tail duplication exposes other constant phi operands: the block that
// supplied the constant now holds the duplicated user right after the MOV that
// materializes it. Where the user could have taken the constant as an
// immediate, rewrite it to the immediate form and drop the MOV once it is dead.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64ExpandImm.h"
#include "AArch64InstrInfo.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-redundantcondbranch"

namespace {

class AArch64RedundantCondBranchLegacy : public MachineFunctionPass {
public:
  static char ID;
  AArch64RedundantCondBranchLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AArch64 Redundant Conditional Branch Elimination";
  }

protected:
  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }
};
char AArch64RedundantCondBranchLegacy::ID = 0;
} // namespace

INITIALIZE_PASS(AArch64RedundantCondBranchLegacy, "aarch64-redundantcondbranch",
                "AArch64 Redundant Conditional Branch Elimination pass", false,
                false)

// If \p MI materializes a constant in one instruction, return the value it
// leaves in the 64-bit register.
static std::optional<uint64_t> getSingleInstrConstant(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return std::nullopt;
  case AArch64::MOVZWi:
  case AArch64::MOVZXi:
  case AArch64::MOVNWi:
  case AArch64::MOVNXi: {
    // The source may be a symbol with a :abs_gN: specifier (large code model,
    // TLS, block addresses) rather than a constant.
    if (!MI.getOperand(1).isImm())
      return std::nullopt;
    uint64_t Imm = (uint64_t)MI.getOperand(1).getImm()
                   << MI.getOperand(2).getImm();
    switch (MI.getOpcode()) {
    case AArch64::MOVZWi:
    case AArch64::MOVZXi:
      return Imm;
    case AArch64::MOVNWi:
      return ~Imm & 0xffffffffULL;
    default:
      return ~Imm;
    }
  }
  case AArch64::ORRWri:
    if (MI.getOperand(1).getReg() != AArch64::WZR)
      return std::nullopt;
    return AArch64_AM::decodeLogicalImmediate(MI.getOperand(2).getImm(), 32);
  case AArch64::ORRXri:
    if (MI.getOperand(1).getReg() != AArch64::XZR)
      return std::nullopt;
    return AArch64_AM::decodeLogicalImmediate(MI.getOperand(2).getImm(), 64);
  }
}

// Rewrite \p MI, a copy of the constant \p Value or an add/sub or logical
// instruction taking it in register \p Reg, to materialize or take the constant
// as an immediate. Returns false if it is not such an instruction or the
// constant does not encode.
static bool foldImmediateIntoUser(MachineInstr &MI, Register Reg,
                                  uint64_t Value, const TargetInstrInfo &TII,
                                  const TargetRegisterInfo &TRI) {
  unsigned RIOpc, NegRIOpc = 0;
  bool Is64Bit = false, IsCommutable = true, IsLogical = false,
       SetsFlags = false;
  switch (MI.getOpcode()) {
  default:
    return false;
  case AArch64::ADDXrs:
    Is64Bit = true;
    [[fallthrough]];
  case AArch64::ADDWrs:
    RIOpc = Is64Bit ? AArch64::ADDXri : AArch64::ADDWri;
    NegRIOpc = Is64Bit ? AArch64::SUBXri : AArch64::SUBWri;
    break;
  case AArch64::SUBXrs:
    Is64Bit = true;
    [[fallthrough]];
  case AArch64::SUBWrs:
    RIOpc = Is64Bit ? AArch64::SUBXri : AArch64::SUBWri;
    NegRIOpc = Is64Bit ? AArch64::ADDXri : AArch64::ADDWri;
    IsCommutable = false;
    break;
  case AArch64::ADDSXrs:
    Is64Bit = true;
    [[fallthrough]];
  case AArch64::ADDSWrs:
    RIOpc = Is64Bit ? AArch64::ADDSXri : AArch64::ADDSWri;
    NegRIOpc = Is64Bit ? AArch64::SUBSXri : AArch64::SUBSWri;
    SetsFlags = true;
    break;
  case AArch64::SUBSXrs:
    Is64Bit = true;
    [[fallthrough]];
  case AArch64::SUBSWrs:
    RIOpc = Is64Bit ? AArch64::SUBSXri : AArch64::SUBSWri;
    NegRIOpc = Is64Bit ? AArch64::ADDSXri : AArch64::ADDSWri;
    IsCommutable = false;
    SetsFlags = true;
    break;
  case AArch64::ANDXrs:
    Is64Bit = true;
    [[fallthrough]];
  case AArch64::ANDWrs:
    RIOpc = Is64Bit ? AArch64::ANDXri : AArch64::ANDWri;
    IsLogical = true;
    break;
  case AArch64::ANDSXrs:
    Is64Bit = true;
    [[fallthrough]];
  case AArch64::ANDSWrs:
    RIOpc = Is64Bit ? AArch64::ANDSXri : AArch64::ANDSWri;
    IsLogical = SetsFlags = true;
    break;
  case AArch64::EORXrs:
    Is64Bit = true;
    [[fallthrough]];
  case AArch64::EORWrs:
    RIOpc = Is64Bit ? AArch64::EORXri : AArch64::EORWri;
    IsLogical = true;
    break;
  case AArch64::ORRXrs:
    Is64Bit = true;
    [[fallthrough]];
  case AArch64::ORRWrs:
    RIOpc = Is64Bit ? AArch64::ORRXri : AArch64::ORRWri;
    IsLogical = true;
    break;
  }
  // Only the unshifted register operand holds the constant itself.
  if (MI.getOperand(3).getImm() != 0)
    return false;

  unsigned ImmIdx;
  if (TRI.isSuperOrSubRegisterEq(Reg, MI.getOperand(2).getReg()))
    ImmIdx = 2;
  else if (IsCommutable &&
           TRI.isSuperOrSubRegisterEq(Reg, MI.getOperand(1).getReg()))
    ImmIdx = 1;
  else
    return false;
  const MachineOperand &Dst = MI.getOperand(0);
  const MachineOperand &Src = MI.getOperand(ImmIdx == 2 ? 1 : 2);

  // Register 31 is SP in Rn of the add/sub immediate forms and in Rd of the
  // forms that do not set flags, where the register forms read ZR.
  bool DstIsZR = Dst.getReg() == AArch64::WZR || Dst.getReg() == AArch64::XZR;
  bool SrcIsZR = Src.getReg() == AArch64::WZR || Src.getReg() == AArch64::XZR;
  if ((DstIsZR && !SetsFlags) || (SrcIsZR && !IsLogical))
    return false;

  unsigned RegSize = Is64Bit ? 64 : 32;
  uint64_t Mask = Is64Bit ? ~0ULL : 0xffffffffULL;
  uint64_t Imm = Value & Mask;

  // A copy of the constant, ORR Rd, ZR, Rm, materializes it in Rd instead when
  // that takes one instruction.
  if (SrcIsZR && (MI.getOpcode() == AArch64::ORRWrs ||
                  MI.getOpcode() == AArch64::ORRXrs)) {
    SmallVector<AArch64_IMM::ImmInsnModel, 4> Insn;
    AArch64_IMM::expandMOVImm(Imm, RegSize, Insn);
    if (Insn.size() != 1)
      return false;
    MachineInstrBuilder MIB =
        BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), TII.get(Insn[0].Opcode))
            .add(Dst);
    if (Insn[0].Opcode == AArch64::ORRWri || Insn[0].Opcode == AArch64::ORRXri)
      MIB.addReg(Is64Bit ? AArch64::XZR : AArch64::WZR).addImm(*Insn[0].Op2);
    else
      MIB.addImm(*Insn[0].Op1).addImm(*Insn[0].Op2);
    LLVM_DEBUG(dbgs() << "Rematerializing: " << MI
                      << "            into: " << *MIB);
    MI.eraseFromParent();
    return true;
  }

  unsigned NewOpc = RIOpc;
  uint64_t Enc;
  unsigned Shift = 0;
  if (IsLogical) {
    if (!AArch64_AM::isLogicalImmediate(Imm, RegSize))
      return false;
    Enc = AArch64_AM::encodeLogicalImmediate(Imm, RegSize);
  } else {
    // A constant that does not encode may do so negated under the opposite
    // operation, which leaves the flags unchanged, as cmn does for cmp.
    if (AArch64_AM::isLegalArithImmed(Imm)) {
      Enc = Imm;
    } else if (AArch64_AM::isLegalArithImmed(-Imm & Mask)) {
      Enc = -Imm & Mask;
      NewOpc = NegRIOpc;
    } else {
      return false;
    }
    Shift = AArch64_AM::getArithImmedShift(Enc);
    Enc >>= Shift;
  }

  MachineInstrBuilder MIB =
      BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), TII.get(NewOpc))
          .add(Dst)
          .add(Src)
          .addImm(Enc);
  if (!IsLogical)
    MIB.addImm(Shift);
  if (SetsFlags) {
    MachineOperand *OldNZCV = MI.findRegisterDefOperand(AArch64::NZCV, &TRI);
    MachineOperand *NewNZCV = MIB->findRegisterDefOperand(AArch64::NZCV, &TRI);
    if (OldNZCV && NewNZCV)
      NewNZCV->setIsDead(OldNZCV->isDead());
  }
  LLVM_DEBUG(dbgs() << "Folding immediate: " << MI << "        into: " << *MIB);
  MI.eraseFromParent();
  return true;
}

// Fold the constants that single MOVs in \p MBB materialize into their first
// user when it can take them as an immediate.
static bool foldImmediates(MachineBasicBlock &MBB, const TargetInstrInfo &TII,
                           const TargetRegisterInfo &TRI) {
  bool Changed = false;
  for (MachineBasicBlock::iterator It = MBB.begin(); It != MBB.end();) {
    MachineInstr &MovMI = *It++;
    std::optional<uint64_t> Value = getSingleInstrConstant(MovMI);
    if (!Value || MovMI.isBundled())
      continue;
    Register Reg = MovMI.getOperand(0).getReg();
    // Look at the 64-bit register so that both widths of the constant and of
    // its user are covered.
    if (AArch64::GPR32RegClass.contains(Reg))
      Reg = TRI.getMatchingSuperReg(Reg, AArch64::sub_32,
                                    &AArch64::GPR64RegClass);
    if (!Reg)
      continue;

    // Find the first instruction touching the register.
    MachineBasicBlock::iterator User = std::next(MovMI.getIterator());
    for (; User != MBB.end(); ++User)
      if (User->readsRegister(Reg, &TRI) || User->modifiesRegister(Reg, &TRI))
        break;
    if (User == MBB.end() || !User->readsRegister(Reg, &TRI) ||
        User->isBundled())
      continue;

    // The MOV is dead once its register is killed or overwritten by the user,
    // or is not read again afterwards.
    bool Dead =
        User->killsRegister(Reg, &TRI) || User->modifiesRegister(Reg, &TRI);
    MachineBasicBlock::iterator Next = std::next(User);
    if (!foldImmediateIntoUser(*User, Reg, *Value, TII, TRI))
      continue;
    Changed = true;
    // The user, which may have been the next instruction, is replaced.
    It = std::next(MovMI.getIterator());
    if (Dead || MBB.computeRegisterLiveness(&TRI, Reg, Next, 16) ==
                    MachineBasicBlock::LQR_Dead)
      MovMI.eraseFromParent();
  }
  return Changed;
}

static bool runAArch64RedundantCondBranch(MachineFunction &MF) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    Changed |= foldImmediates(MBB, TII, TRI);
    Changed |= optimizeTerminators(&MBB, TII);
  }
  return Changed;
}

bool AArch64RedundantCondBranchLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  return runAArch64RedundantCondBranch(MF);
}

PreservedAnalyses
AArch64RedundantCondBranchPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &) {
  if (runAArch64RedundantCondBranch(MF)) {
    return getMachineFunctionPassPreservedAnalyses();
  }
  return PreservedAnalyses::all();
}

FunctionPass *llvm::createAArch64RedundantCondBranchPass() {
  return new AArch64RedundantCondBranchLegacy();
}
