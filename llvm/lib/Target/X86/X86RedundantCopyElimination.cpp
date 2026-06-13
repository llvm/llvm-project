
//=- X86RedundantCopyElimination.cpp - Remove useless copy for X86 ----------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "x86-copyelim"

STATISTIC(NumCopiesRemoved, "Number of copies removed.");

namespace llvm {
cl::opt<bool> EnableRedundantCopyElimination(
    "x86-enable-copyelim",
    cl::desc("Enable the redundant copy elimination pass"), cl::init(true),
    cl::Hidden);
} // namespace llvm

namespace {

struct KnownValue {
  bool IsReg;
  union {
    MCPhysReg Reg;
    int64_t Imm;
  } U;

  KnownValue() : IsReg(false) { U.Imm = 0; }
  KnownValue(MCPhysReg R) : IsReg(true) { U.Reg = R; }
  KnownValue(int64_t I) : IsReg(false) { U.Imm = I; }
};

static bool isRedundantZeroDefinition(const MachineInstr &MI) {
  if (MI.getOpcode() == X86::MOV32r0)
    return true;
  if (MI.isMoveImmediate())
    return MI.getOperand(1).isImm() && MI.getOperand(1).getImm() == 0;
  switch (MI.getOpcode()) {
  case X86::XOR8rr:
  case X86::XOR16rr:
  case X86::XOR32rr:
  case X86::XOR64rr:
    return MI.getOperand(1).getReg() == MI.getOperand(2).getReg();
  default:
    return false;
  }
}

class X86RedundantCopyEliminationImpl {
public:
  X86RedundantCopyEliminationImpl() {}
  bool run(MachineFunction &MF);

private:
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;

  bool optimizeBlock(MachineBasicBlock &MBB);
};

class X86RedundantCopyEliminationLegacy : public MachineFunctionPass {
public:
  static char ID;
  X86RedundantCopyEliminationLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }
  StringRef getPassName() const override {
    return "X86 Redundant Copy Elimination";
  }
};

bool X86RedundantCopyEliminationImpl::optimizeBlock(MachineBasicBlock &MBB) {
  if (MBB.pred_size() != 1)
    return false;

  MachineBasicBlock *BranchMBB = *MBB.pred_begin();
  SmallVector<MachineBasicBlock *, 4> Path;

  // Trace back along single-predecessor edges.
  while (BranchMBB->succ_size() != 2) {
    if (BranchMBB->pred_size() != 1)
      return false;
    Path.push_back(BranchMBB);
    BranchMBB = *BranchMBB->pred_begin();
  }

  MachineBasicBlock *TBB = nullptr;
  MachineBasicBlock *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (TII->analyzeBranch(*BranchMBB, TBB, FBB, Cond, /*AllowModify=*/false) ||
      Cond.size() != 1)
    return false;

  X86::CondCode CC = static_cast<X86::CondCode>(Cond[0].getImm());

  MachineBasicBlock *NextOnPath = Path.empty() ? &MBB : Path.back();
  bool IsTruePath = (NextOnPath == TBB);
  if (!IsTruePath) {
    if (FBB) {
      if (NextOnPath != FBB)
        return false;
    } else if (NextOnPath != BranchMBB->getNextNode()) {
      return false;
    }
  }

  MachineInstr *CondBr = nullptr;
  for (MachineInstr &Term : BranchMBB->terminators()) {
    if (X86::isJCC(Term.getOpcode())) {
      CondBr = &Term;
      break;
    }
  }
  if (!CondBr)
    return false;

  // Find the flag-setting instruction
  MachineInstr *FlagSetter = nullptr;
  MachineBasicBlock::reverse_iterator RIt = CondBr->getReverseIterator();
  for (MachineInstr &PredI : make_range(std::next(RIt), BranchMBB->rend())) {
    if (PredI.modifiesRegister(X86::EFLAGS, TRI)) {
      FlagSetter = &PredI;
      break;
    }
  }
  if (!FlagSetter)
    return false;

  SmallVector<std::pair<MCPhysReg, KnownValue>, 4> KnownRegs;

  // Helper: is the equality/zero path taken?
  bool IsEqualPath =
      (CC == X86::COND_E && IsTruePath) || (CC == X86::COND_NE && !IsTruePath);

  // Analyze the flag setter.
  switch (FlagSetter->getOpcode()) {
  default:
    break;

  // TEST is BinOpRR_F: (outs), (ins $src1, $src2) — no destination register.
  // Operand 0 = $src1, Operand 1 = $src2.
  case X86::TEST8rr:
  case X86::TEST16rr:
  case X86::TEST32rr:
  case X86::TEST64rr:
    if (FlagSetter->getOperand(0).getReg() ==
        FlagSetter->getOperand(1).getReg()) {
      if (IsEqualPath)
        KnownRegs.push_back(
            {FlagSetter->getOperand(0).getReg(), KnownValue((int64_t)0)});
    }
    break;

  // AND/OR are BinOpRR_RF: (outs $dst), (ins $src1, $src2).
  // Operand 0 = $dst (tied to $src1), Operand 1 = $src1, Operand 2 = $src2.
  // Self-AND/OR (src1 == src2) sets ZF iff the value is zero.
  case X86::AND8rr:
  case X86::AND16rr:
  case X86::AND32rr:
  case X86::AND64rr:
  case X86::OR8rr:
  case X86::OR16rr:
  case X86::OR32rr:
  case X86::OR64rr:
    if (FlagSetter->getOperand(1).getReg() ==
        FlagSetter->getOperand(2).getReg()) {
      if (IsEqualPath)
        KnownRegs.push_back(
            {FlagSetter->getOperand(0).getReg(), KnownValue((int64_t)0)});
    }
    break;

  // CMP reg, reg is BinOpRR_F: (outs), (ins $src1, $src2).
  // If equal, $src1 == $src2.
  case X86::CMP8rr:
  case X86::CMP16rr:
  case X86::CMP32rr:
  case X86::CMP64rr:
    if (IsEqualPath) {
      KnownRegs.push_back(
          {FlagSetter->getOperand(0).getReg(),
           KnownValue((MCPhysReg)FlagSetter->getOperand(1).getReg())});
      KnownRegs.push_back(
          {FlagSetter->getOperand(1).getReg(),
           KnownValue((MCPhysReg)FlagSetter->getOperand(0).getReg())});
    }
    break;

  // CMP reg, imm: (outs), (ins $src1, $imm).
  // If equal, $src1 == $imm.
  case X86::CMP8ri:
  case X86::CMP16ri:
  case X86::CMP16ri8:
  case X86::CMP32ri:
  case X86::CMP32ri8:
  case X86::CMP64ri8:
  case X86::CMP64ri32:
    if (IsEqualPath && FlagSetter->getOperand(1).isImm()) {
      int64_t Imm = FlagSetter->getOperand(1).getImm();
      KnownRegs.push_back(
          {FlagSetter->getOperand(0).getReg(), KnownValue(Imm)});
    }
    break;

  // SUB reg, reg is BinOpRR_RF: (outs $dst), (ins $src1, $src2).
  // If ZF set, $dst is zero. (Note: src1 and src2 were equal before the
  // instruction, but $dst is tied to $src1, so $src1 is overwritten with zero
  // and no longer equals $src2).
  case X86::SUB8rr:
  case X86::SUB16rr:
  case X86::SUB32rr:
  case X86::SUB64rr:
    if (IsEqualPath) {
      KnownRegs.push_back(
          {FlagSetter->getOperand(0).getReg(), KnownValue((int64_t)0)});
    }
    break;
  }

  if (KnownRegs.empty())
    return false;

  // Check if any known register was clobbered in the path.
  auto isClobbered = [&](MCPhysReg Reg) {
    // Check BranchMBB from FlagSetter to end
    for (MachineInstr &MI :
         make_range(std::next(MachineBasicBlock::iterator(FlagSetter)),
                    BranchMBB->end())) {
      if (MI.modifiesRegister(Reg, TRI))
        return true;
    }
    // Check Path blocks
    for (auto *PMBB : llvm::reverse(Path)) {
      for (MachineInstr &MI : *PMBB) {
        if (MI.modifiesRegister(Reg, TRI))
          return true;
      }
    }
    return false;
  };

  for (unsigned i = 0; i < KnownRegs.size();) {
    if (isClobbered(KnownRegs[i].first) ||
        (KnownRegs[i].second.IsReg && isClobbered(KnownRegs[i].second.U.Reg))) {
      KnownRegs[i] = KnownRegs.back();
      KnownRegs.pop_back();
    } else {
      ++i;
    }
  }

  if (KnownRegs.empty())
    return false;

  bool Changed = false;
  MachineBasicBlock::iterator LastChange = MBB.begin();

  SmallVector<MCPhysReg, 4> OptimizedRegs;

  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;) {
    MachineInstr *MI = &*I;
    ++I;

    // Optimization 1: Remove redundant immediate definitions.
    // This covers zero-idioms (XOR reg, reg; MOV32r0; MOV reg, 0) as well as
    // move-immediates matching a known constant from CMP reg, imm.
    if (isRedundantZeroDefinition(*MI)) {
      Register DefReg = MI->getOperand(0).getReg();
      auto It = llvm::find_if(KnownRegs,
                              [&](const std::pair<MCPhysReg, KnownValue> &K) {
                                return K.first == DefReg;
                              });
      if (It != KnownRegs.end() && !It->second.IsReg && It->second.U.Imm == 0) {
        LLVM_DEBUG(dbgs() << "Remove redundant zero definition: " << *MI);
        MI->eraseFromParent();
        Changed = true;
        LastChange = I;
        ++NumCopiesRemoved;
        OptimizedRegs.push_back(DefReg);
        continue;
      }
    } else if (MI->isMoveImmediate() && MI->getOperand(1).isImm()) {
      Register DefReg = MI->getOperand(0).getReg();
      int64_t MIImm = MI->getOperand(1).getImm();
      auto It = llvm::find_if(KnownRegs,
                              [&](const std::pair<MCPhysReg, KnownValue> &K) {
                                return K.first == DefReg;
                              });
      if (It != KnownRegs.end() && !It->second.IsReg &&
          It->second.U.Imm == MIImm) {
        LLVM_DEBUG(dbgs() << "Remove redundant immediate definition: " << *MI);
        MI->eraseFromParent();
        Changed = true;
        LastChange = I;
        ++NumCopiesRemoved;
        OptimizedRegs.push_back(DefReg);
        continue;
      }
    }

    // Optimization 2: Remove redundant register copies/moves.
    if (MI->isCopy() || MI->isMoveReg()) {
      Register DefReg = MI->getOperand(0).getReg();
      Register SrcReg = MI->getOperand(1).getReg();
      auto It = llvm::find_if(KnownRegs,
                              [&](const std::pair<MCPhysReg, KnownValue> &K) {
                                return K.first == DefReg;
                              });
      if (It != KnownRegs.end() && It->second.IsReg &&
          It->second.U.Reg == SrcReg) {
        LLVM_DEBUG(dbgs() << "Remove redundant register copy: " << *MI);
        MI->eraseFromParent();
        Changed = true;
        LastChange = I;
        ++NumCopiesRemoved;
        OptimizedRegs.push_back(DefReg);
        continue;
      }
    }

    // Remove clobbered registers from KnownRegs.
    for (unsigned i = 0; i < KnownRegs.size();) {
      if (MI->modifiesRegister(KnownRegs[i].first, TRI) ||
          (KnownRegs[i].second.IsReg &&
           MI->modifiesRegister(KnownRegs[i].second.U.Reg, TRI))) {
        KnownRegs[i] = KnownRegs.back();
        KnownRegs.pop_back();
      } else {
        ++i;
      }
    }
    if (KnownRegs.empty())
      break;
  }

  if (!Changed)
    return false;

  // Liveness fixups: clear kill flags along the entire extended live range.
  for (MCPhysReg Reg : OptimizedRegs) {
    // Clear kills in BranchMBB from FlagSetter to end (covers CondBr too).
    for (MachineInstr &MMI :
         make_range(MachineBasicBlock::iterator(FlagSetter), BranchMBB->end()))
      MMI.clearRegisterKills(Reg, TRI);

    if (!MBB.isLiveIn(Reg))
      MBB.addLiveIn(Reg);

    // Clear kills on Path blocks and add live-ins.
    for (auto *PMBB : Path) {
      if (!PMBB->isLiveIn(Reg))
        PMBB->addLiveIn(Reg);
      for (MachineInstr &MMI : *PMBB)
        MMI.clearRegisterKills(Reg, TRI);
    }

    // Clear kills in MBB up to the last change.
    for (MachineInstr &MMI : make_range(MBB.begin(), LastChange))
      MMI.clearRegisterKills(Reg, TRI);
  }

  return true;
}

bool X86RedundantCopyEliminationImpl::run(MachineFunction &MF) {
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= optimizeBlock(MBB);
  return Changed;
}

bool X86RedundantCopyEliminationLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  return X86RedundantCopyEliminationImpl().run(MF);
}

} // end anonymous namespace

char X86RedundantCopyEliminationLegacy::ID = 0;

INITIALIZE_PASS(X86RedundantCopyEliminationLegacy, DEBUG_TYPE,
                "X86 redundant copy elimination pass", false, false)

PreservedAnalyses
X86RedundantCopyEliminationPass::run(MachineFunction &MF,
                                     MachineFunctionAnalysisManager &MFAM) {
  const bool Changed = X86RedundantCopyEliminationImpl().run(MF);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

FunctionPass *llvm::createX86RedundantCopyEliminationLegacyPass() {
  return new X86RedundantCopyEliminationLegacy();
}
