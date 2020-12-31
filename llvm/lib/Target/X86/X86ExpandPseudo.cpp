//===------- X86ExpandPseudo.cpp - Expand pseudo instructions -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions to allow proper scheduling, if-conversion, other late
// optimizations, or simply the encoding of the instructions.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86FrameLowering.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h" // For IDs of passes that are preserved.
#include "llvm/IR/GlobalValue.h"
using namespace llvm;

#define DEBUG_TYPE "x86-pseudo"
#define X86_EXPAND_PSEUDO_NAME "X86 pseudo instruction expansion pass"

namespace {
class X86ExpandPseudo : public MachineFunctionPass {
public:
  static char ID;
  X86ExpandPseudo() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreservedID(MachineLoopInfoID);
    AU.addPreservedID(MachineDominatorsID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  const X86Subtarget *STI = nullptr;
  const X86InstrInfo *TII = nullptr;
  const X86RegisterInfo *TRI = nullptr;
  const X86MachineFunctionInfo *X86FI = nullptr;
  const X86FrameLowering *X86FL = nullptr;

  bool runOnMachineFunction(MachineFunction &Fn) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  StringRef getPassName() const override {
    return "X86 pseudo instruction expansion pass";
  }

private:
  void ExpandICallBranchFunnel(MachineBasicBlock *MBB,
                               MachineBasicBlock::iterator MBBI);

  bool ExpandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
  bool ExpandMBB(MachineBasicBlock &MBB);
};
char X86ExpandPseudo::ID = 0;

} // End anonymous namespace.

INITIALIZE_PASS(X86ExpandPseudo, DEBUG_TYPE, X86_EXPAND_PSEUDO_NAME, false,
                false)

void X86ExpandPseudo::ExpandICallBranchFunnel(
    MachineBasicBlock *MBB, MachineBasicBlock::iterator MBBI) {
  MachineBasicBlock *JTMBB = MBB;
  MachineInstr *JTInst = &*MBBI;
  MachineFunction *MF = MBB->getParent();
  const BasicBlock *BB = MBB->getBasicBlock();
  auto InsPt = MachineFunction::iterator(MBB);
  ++InsPt;

  std::vector<std::pair<MachineBasicBlock *, unsigned>> TargetMBBs;
  DebugLoc DL = JTInst->getDebugLoc();
  MachineOperand Selector = JTInst->getOperand(0);
  const GlobalValue *CombinedGlobal = JTInst->getOperand(1).getGlobal();

  auto CmpTarget = [&](unsigned Target) {
    if (Selector.isReg())
      MBB->addLiveIn(Selector.getReg());
    BuildMI(*MBB, MBBI, DL, TII->get(X86::LEA64r), X86::R11)
        .addReg(X86::RIP)
        .addImm(1)
        .addReg(0)
        .addGlobalAddress(CombinedGlobal,
                          JTInst->getOperand(2 + 2 * Target).getImm())
        .addReg(0);
    BuildMI(*MBB, MBBI, DL, TII->get(X86::CMP64rr))
        .add(Selector)
        .addReg(X86::R11);
  };

  auto CreateMBB = [&]() {
    auto *NewMBB = MF->CreateMachineBasicBlock(BB);
    MBB->addSuccessor(NewMBB);
    if (!MBB->isLiveIn(X86::EFLAGS))
      MBB->addLiveIn(X86::EFLAGS);
    return NewMBB;
  };

  auto EmitCondJump = [&](unsigned CC, MachineBasicBlock *ThenMBB) {
    BuildMI(*MBB, MBBI, DL, TII->get(X86::JCC_1)).addMBB(ThenMBB).addImm(CC);

    auto *ElseMBB = CreateMBB();
    MF->insert(InsPt, ElseMBB);
    MBB = ElseMBB;
    MBBI = MBB->end();
  };

  auto EmitCondJumpTarget = [&](unsigned CC, unsigned Target) {
    auto *ThenMBB = CreateMBB();
    TargetMBBs.push_back({ThenMBB, Target});
    EmitCondJump(CC, ThenMBB);
  };

  auto EmitTailCall = [&](unsigned Target) {
    BuildMI(*MBB, MBBI, DL, TII->get(X86::TAILJMPd64))
        .add(JTInst->getOperand(3 + 2 * Target));
  };

  std::function<void(unsigned, unsigned)> EmitBranchFunnel =
      [&](unsigned FirstTarget, unsigned NumTargets) {
    if (NumTargets == 1) {
      EmitTailCall(FirstTarget);
      return;
    }

    if (NumTargets == 2) {
      CmpTarget(FirstTarget + 1);
      EmitCondJumpTarget(X86::COND_B, FirstTarget);
      EmitTailCall(FirstTarget + 1);
      return;
    }

    if (NumTargets < 6) {
      CmpTarget(FirstTarget + 1);
      EmitCondJumpTarget(X86::COND_B, FirstTarget);
      EmitCondJumpTarget(X86::COND_E, FirstTarget + 1);
      EmitBranchFunnel(FirstTarget + 2, NumTargets - 2);
      return;
    }

    auto *ThenMBB = CreateMBB();
    CmpTarget(FirstTarget + (NumTargets / 2));
    EmitCondJump(X86::COND_B, ThenMBB);
    EmitCondJumpTarget(X86::COND_E, FirstTarget + (NumTargets / 2));
    EmitBranchFunnel(FirstTarget + (NumTargets / 2) + 1,
                  NumTargets - (NumTargets / 2) - 1);

    MF->insert(InsPt, ThenMBB);
    MBB = ThenMBB;
    MBBI = MBB->end();
    EmitBranchFunnel(FirstTarget, NumTargets / 2);
  };

  EmitBranchFunnel(0, (JTInst->getNumOperands() - 2) / 2);
  for (auto P : TargetMBBs) {
    MF->insert(InsPt, P.first);
    BuildMI(P.first, DL, TII->get(X86::TAILJMPd64))
        .add(JTInst->getOperand(3 + 2 * P.second));
  }
  JTMBB->erase(JTInst);
}

/// If \p MBBI is a pseudo instruction, this method expands
/// it to the corresponding (sequence of) actual instruction(s).
/// \returns true if \p MBBI has been expanded.
bool X86ExpandPseudo::ExpandMI(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI) {
  MachineInstr &MI = *MBBI;
  unsigned Opcode = MI.getOpcode();
  DebugLoc DL = MBBI->getDebugLoc();
  switch (Opcode) {
  default:
    return false;
  case X86::TCRETURNdi:
  case X86::TCRETURNdicc:
  case X86::TCRETURNri:
  case X86::TCRETURNmi:
  case X86::TCRETURNdi64:
  case X86::TCRETURNdi64cc:
  case X86::TCRETURNri64:
  case X86::TCRETURNmi64: {
    bool isMem = Opcode == X86::TCRETURNmi || Opcode == X86::TCRETURNmi64;
    MachineOperand &JumpTarget = MBBI->getOperand(0);
    MachineOperand &StackAdjust = MBBI->getOperand(isMem ? X86::AddrNumOperands
                                                         : 1);
    assert(StackAdjust.isImm() && "Expecting immediate value.");

    // Adjust stack pointer.
    int StackAdj = StackAdjust.getImm();
    int MaxTCDelta = X86FI->getTCReturnAddrDelta();
    int Offset = 0;
    assert(MaxTCDelta <= 0 && "MaxTCDelta should never be positive");

    // Incoporate the retaddr area.
    Offset = StackAdj - MaxTCDelta;
    assert(Offset >= 0 && "Offset should never be negative");

    if (Opcode == X86::TCRETURNdicc || Opcode == X86::TCRETURNdi64cc) {
      assert(Offset == 0 && "Conditional tail call cannot adjust the stack.");
    }

    if (Offset) {
      // Check for possible merge with preceding ADD instruction.
      Offset += X86FL->mergeSPUpdates(MBB, MBBI, true);
      X86FL->emitSPUpdate(MBB, MBBI, DL, Offset, /*InEpilogue=*/true);
    }

    // Jump to label or value in register.
    bool IsWin64 = STI->isTargetWin64();
    if (Opcode == X86::TCRETURNdi || Opcode == X86::TCRETURNdicc ||
        Opcode == X86::TCRETURNdi64 || Opcode == X86::TCRETURNdi64cc) {
      unsigned Op;
      switch (Opcode) {
      case X86::TCRETURNdi:
        Op = X86::TAILJMPd;
        break;
      case X86::TCRETURNdicc:
        Op = X86::TAILJMPd_CC;
        break;
      case X86::TCRETURNdi64cc:
        assert(!MBB.getParent()->hasWinCFI() &&
               "Conditional tail calls confuse "
               "the Win64 unwinder.");
        Op = X86::TAILJMPd64_CC;
        break;
      default:
        // Note: Win64 uses REX prefixes indirect jumps out of functions, but
        // not direct ones.
        Op = X86::TAILJMPd64;
        break;
      }
      MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(Op));
      if (JumpTarget.isGlobal()) {
        MIB.addGlobalAddress(JumpTarget.getGlobal(), JumpTarget.getOffset(),
                             JumpTarget.getTargetFlags());
      } else {
        assert(JumpTarget.isSymbol());
        MIB.addExternalSymbol(JumpTarget.getSymbolName(),
                              JumpTarget.getTargetFlags());
      }
      if (Op == X86::TAILJMPd_CC || Op == X86::TAILJMPd64_CC) {
        MIB.addImm(MBBI->getOperand(2).getImm());
      }

    } else if (Opcode == X86::TCRETURNmi || Opcode == X86::TCRETURNmi64) {
      unsigned Op = (Opcode == X86::TCRETURNmi)
                        ? X86::TAILJMPm
                        : (IsWin64 ? X86::TAILJMPm64_REX : X86::TAILJMPm64);
      MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(Op));
      for (unsigned i = 0; i != X86::AddrNumOperands; ++i)
        MIB.add(MBBI->getOperand(i));
    } else if (Opcode == X86::TCRETURNri64) {
      JumpTarget.setIsKill();
      BuildMI(MBB, MBBI, DL,
              TII->get(IsWin64 ? X86::TAILJMPr64_REX : X86::TAILJMPr64))
          .add(JumpTarget);
    } else {
      JumpTarget.setIsKill();
      BuildMI(MBB, MBBI, DL, TII->get(X86::TAILJMPr))
          .add(JumpTarget);
    }

    MachineInstr &NewMI = *std::prev(MBBI);
    NewMI.copyImplicitOps(*MBBI->getParent()->getParent(), *MBBI);

    // Update the call site info.
    if (MBBI->isCandidateForCallSiteEntry())
      MBB.getParent()->moveCallSiteInfo(&*MBBI, &NewMI);

    // Delete the pseudo instruction TCRETURN.
    MBB.erase(MBBI);

    return true;
  }
  case X86::EH_RETURN:
  case X86::EH_RETURN64: {
    MachineOperand &DestAddr = MBBI->getOperand(0);
    assert(DestAddr.isReg() && "Offset should be in register!");
    const bool Uses64BitFramePtr =
        STI->isTarget64BitLP64() || STI->isTargetNaCl64();
    Register StackPtr = TRI->getStackRegister();
    BuildMI(MBB, MBBI, DL,
            TII->get(Uses64BitFramePtr ? X86::MOV64rr : X86::MOV32rr), StackPtr)
        .addReg(DestAddr.getReg());
    // The EH_RETURN pseudo is really removed during the MC Lowering.
    return true;
  }
  case X86::IRET: {
    // Adjust stack to erase error code
    int64_t StackAdj = MBBI->getOperand(0).getImm();
    X86FL->emitSPUpdate(MBB, MBBI, DL, StackAdj, true);
    // Replace pseudo with machine iret
    BuildMI(MBB, MBBI, DL,
            TII->get(STI->is64Bit() ? X86::IRET64 : X86::IRET32));
    MBB.erase(MBBI);
    return true;
  }
  case X86::RET: {
    // Adjust stack to erase error code
    int64_t StackAdj = MBBI->getOperand(0).getImm();
    MachineInstrBuilder MIB;
    if (StackAdj == 0) {
      MIB = BuildMI(MBB, MBBI, DL,
                    TII->get(STI->is64Bit() ? X86::RETQ : X86::RETL));
    } else if (isUInt<16>(StackAdj)) {
      MIB = BuildMI(MBB, MBBI, DL,
                    TII->get(STI->is64Bit() ? X86::RETIQ : X86::RETIL))
                .addImm(StackAdj);
    } else {
      assert(!STI->is64Bit() &&
             "shouldn't need to do this for x86_64 targets!");
      // A ret can only handle immediates as big as 2**16-1.  If we need to pop
      // off bytes before the return address, we must do it manually.
      BuildMI(MBB, MBBI, DL, TII->get(X86::POP32r)).addReg(X86::ECX, RegState::Define);
      X86FL->emitSPUpdate(MBB, MBBI, DL, StackAdj, /*InEpilogue=*/true);
      BuildMI(MBB, MBBI, DL, TII->get(X86::PUSH32r)).addReg(X86::ECX);
      MIB = BuildMI(MBB, MBBI, DL, TII->get(X86::RETL));
    }
    for (unsigned I = 1, E = MBBI->getNumOperands(); I != E; ++I)
      MIB.add(MBBI->getOperand(I));
    MBB.erase(MBBI);
    return true;
  }
  case X86::LCMPXCHG16B_SAVE_RBX: {
    // Perform the following transformation.
    // SaveRbx = pseudocmpxchg Addr, <4 opds for the address>, InArg, SaveRbx
    // =>
    // RBX = InArg
    // actualcmpxchg Addr
    // RBX = SaveRbx
    const MachineOperand &InArg = MBBI->getOperand(6);
    Register SaveRbx = MBBI->getOperand(7).getReg();

    // Copy the input argument of the pseudo into the argument of the
    // actual instruction.
    // NOTE: We don't copy the kill flag since the input might be the same reg
    // as one of the other operands of LCMPXCHG16B.
    TII->copyPhysReg(MBB, MBBI, DL, X86::RBX, InArg.getReg(), false);
    // Create the actual instruction.
    MachineInstr *NewInstr = BuildMI(MBB, MBBI, DL, TII->get(X86::LCMPXCHG16B));
    // Copy the operands related to the address.
    for (unsigned Idx = 1; Idx < 6; ++Idx)
      NewInstr->addOperand(MBBI->getOperand(Idx));
    // Finally, restore the value of RBX.
    TII->copyPhysReg(MBB, MBBI, DL, X86::RBX, SaveRbx,
                     /*SrcIsKill*/ true);

    // Delete the pseudo.
    MBBI->eraseFromParent();
    return true;
  }
  // Loading/storing mask pairs requires two kmov operations. The second one of
  // these needs a 2 byte displacement relative to the specified address (with
  // 32 bit spill size). The pairs of 1bit masks up to 16 bit masks all use the
  // same spill size, they all are stored using MASKPAIR16STORE, loaded using
  // MASKPAIR16LOAD.
  //
  // The displacement value might wrap around in theory, thus the asserts in
  // both cases.
  case X86::MASKPAIR16LOAD: {
    int64_t Disp = MBBI->getOperand(1 + X86::AddrDisp).getImm();
    assert(Disp >= 0 && Disp <= INT32_MAX - 2 && "Unexpected displacement");
    Register Reg = MBBI->getOperand(0).getReg();
    bool DstIsDead = MBBI->getOperand(0).isDead();
    Register Reg0 = TRI->getSubReg(Reg, X86::sub_mask_0);
    Register Reg1 = TRI->getSubReg(Reg, X86::sub_mask_1);

    auto MIBLo = BuildMI(MBB, MBBI, DL, TII->get(X86::KMOVWkm))
      .addReg(Reg0, RegState::Define | getDeadRegState(DstIsDead));
    auto MIBHi = BuildMI(MBB, MBBI, DL, TII->get(X86::KMOVWkm))
      .addReg(Reg1, RegState::Define | getDeadRegState(DstIsDead));

    for (int i = 0; i < X86::AddrNumOperands; ++i) {
      MIBLo.add(MBBI->getOperand(1 + i));
      if (i == X86::AddrDisp)
        MIBHi.addImm(Disp + 2);
      else
        MIBHi.add(MBBI->getOperand(1 + i));
    }

    // Split the memory operand, adjusting the offset and size for the halves.
    MachineMemOperand *OldMMO = MBBI->memoperands().front();
    MachineFunction *MF = MBB.getParent();
    MachineMemOperand *MMOLo = MF->getMachineMemOperand(OldMMO, 0, 2);
    MachineMemOperand *MMOHi = MF->getMachineMemOperand(OldMMO, 2, 2);

    MIBLo.setMemRefs(MMOLo);
    MIBHi.setMemRefs(MMOHi);

    // Delete the pseudo.
    MBB.erase(MBBI);
    return true;
  }
  case X86::MASKPAIR16STORE: {
    int64_t Disp = MBBI->getOperand(X86::AddrDisp).getImm();
    assert(Disp >= 0 && Disp <= INT32_MAX - 2 && "Unexpected displacement");
    Register Reg = MBBI->getOperand(X86::AddrNumOperands).getReg();
    bool SrcIsKill = MBBI->getOperand(X86::AddrNumOperands).isKill();
    Register Reg0 = TRI->getSubReg(Reg, X86::sub_mask_0);
    Register Reg1 = TRI->getSubReg(Reg, X86::sub_mask_1);

    auto MIBLo = BuildMI(MBB, MBBI, DL, TII->get(X86::KMOVWmk));
    auto MIBHi = BuildMI(MBB, MBBI, DL, TII->get(X86::KMOVWmk));

    for (int i = 0; i < X86::AddrNumOperands; ++i) {
      MIBLo.add(MBBI->getOperand(i));
      if (i == X86::AddrDisp)
        MIBHi.addImm(Disp + 2);
      else
        MIBHi.add(MBBI->getOperand(i));
    }
    MIBLo.addReg(Reg0, getKillRegState(SrcIsKill));
    MIBHi.addReg(Reg1, getKillRegState(SrcIsKill));

    // Split the memory operand, adjusting the offset and size for the halves.
    MachineMemOperand *OldMMO = MBBI->memoperands().front();
    MachineFunction *MF = MBB.getParent();
    MachineMemOperand *MMOLo = MF->getMachineMemOperand(OldMMO, 0, 2);
    MachineMemOperand *MMOHi = MF->getMachineMemOperand(OldMMO, 2, 2);

    MIBLo.setMemRefs(MMOLo);
    MIBHi.setMemRefs(MMOHi);

    // Delete the pseudo.
    MBB.erase(MBBI);
    return true;
  }
  case X86::MWAITX_SAVE_RBX: {
    // Perform the following transformation.
    // SaveRbx = pseudomwaitx InArg, SaveRbx
    // =>
    // [E|R]BX = InArg
    // actualmwaitx
    // [E|R]BX = SaveRbx
    const MachineOperand &InArg = MBBI->getOperand(1);
    // Copy the input argument of the pseudo into the argument of the
    // actual instruction.
    TII->copyPhysReg(MBB, MBBI, DL, X86::EBX, InArg.getReg(), InArg.isKill());
    // Create the actual instruction.
    BuildMI(MBB, MBBI, DL, TII->get(X86::MWAITXrrr));
    // Finally, restore the value of RBX.
    Register SaveRbx = MBBI->getOperand(2).getReg();
    TII->copyPhysReg(MBB, MBBI, DL, X86::RBX, SaveRbx, /*SrcIsKill*/ true);
    // Delete the pseudo.
    MBBI->eraseFromParent();
    return true;
  }
  case TargetOpcode::ICALL_BRANCH_FUNNEL:
    ExpandICallBranchFunnel(&MBB, MBBI);
    return true;
  case X86::PLDTILECFG: {
    MI.RemoveOperand(0);
    MI.setDesc(TII->get(X86::LDTILECFG));
    return true;
  }
  case X86::PSTTILECFG: {
    MI.RemoveOperand(MI.getNumOperands() - 1); // Remove $tmmcfg
    MI.setDesc(TII->get(X86::STTILECFG));
    return true;
  }
  case X86::PTILELOADDV: {
    MI.RemoveOperand(8); // Remove $tmmcfg
    for (unsigned i = 2; i > 0; --i)
      MI.RemoveOperand(i);
    MI.setDesc(TII->get(X86::TILELOADD));
    return true;
  }
  case X86::PTDPBSSDV: {
    MI.RemoveOperand(7); // Remove $tmmcfg
    MI.untieRegOperand(4);
    for (unsigned i = 3; i > 0; --i)
      MI.RemoveOperand(i);
    MI.setDesc(TII->get(X86::TDPBSSD));
    MI.tieOperands(0, 1);
    return true;
  }
  case X86::PTILESTOREDV: {
    MI.RemoveOperand(8); // Remove $tmmcfg
    for (int i = 1; i >= 0; --i)
      MI.RemoveOperand(i);
    MI.setDesc(TII->get(X86::TILESTORED));
    return true;
  }
  case X86::PTILEZEROV: {
    for (int i = 3; i > 0; --i) // Remove row, col, $tmmcfg
      MI.RemoveOperand(i);
    MI.setDesc(TII->get(X86::TILEZERO));
    return true;
  }
  }
  llvm_unreachable("Previous switch has a fallthrough?");
}

/// Expand all pseudo instructions contained in \p MBB.
/// \returns true if any expansion occurred for \p MBB.
bool X86ExpandPseudo::ExpandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  // MBBI may be invalidated by the expansion.
  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= ExpandMI(MBB, MBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool X86ExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  STI = &static_cast<const X86Subtarget &>(MF.getSubtarget());
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  X86FI = MF.getInfo<X86MachineFunctionInfo>();
  X86FL = STI->getFrameLowering();

  bool Modified = false;
  for (MachineBasicBlock &MBB : MF)
    Modified |= ExpandMBB(MBB);
  return Modified;
}

/// Returns an instance of the pseudo instruction expansion pass.
FunctionPass *llvm::createX86ExpandPseudoPass() {
  return new X86ExpandPseudo();
}
