//===-- RISCVLateBranchOpt.cpp - Late Stage Branch Optimization -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This pass folds immediate operands into conditional branches after register
/// allocation and simplifies branches whose conditions can be statically
/// evaluated.
///
/// Run late to preserve constant sharing and expose path-specific constants.
///
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Analysis.h"

using namespace llvm;

#define RISCV_LATE_BRANCH_OPT_NAME "RISC-V Late Branch Optimisation Pass"

namespace {

struct RISCVLateBranchOptLegacy : public MachineFunctionPass {
  static char ID;

  RISCVLateBranchOptLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return RISCV_LATE_BRANCH_OPT_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;
};
} // namespace

char RISCVLateBranchOptLegacy::ID = 0;
INITIALIZE_PASS(RISCVLateBranchOptLegacy, "riscv-late-branch-opt",
                RISCV_LATE_BRANCH_OPT_NAME, false, false)

// Match encodable constants; zero comparisons already use x0.
static bool getBranchImmediate(const MachineInstr &MI, Register Reg,
                               int64_t &Imm) {
  if (MI.getOpcode() != RISCV::ADDI || MI.getOperand(0).getReg() != Reg ||
      !MI.getOperand(1).isReg() || MI.getOperand(1).getReg() != RISCV::X0 ||
      !MI.getOperand(2).isImm())
    return false;
  Imm = MI.getOperand(2).getImm();
  return Imm == -1 || (Imm >= 1 && Imm <= 31);
}

// Stop at the nearest write, including aliases and call clobbers.
static MachineInstr *findBranchImmediate(MachineInstr &Branch, Register Reg,
                                         const TargetRegisterInfo *TRI,
                                         int64_t &Imm) {
  if (Reg == RISCV::X0)
    return nullptr;
  for (auto I = Branch.getIterator(), E = Branch.getParent()->instr_begin();
       I != E;) {
    MachineInstr &MI = *--I;
    if (MI.modifiesRegister(Reg, TRI))
      return getBranchImmediate(MI, Reg, Imm) ? &MI : nullptr;
  }
  return nullptr;
}

// Require one constant across all incoming paths. Cycles alone do not prove
// a value, and function live-ins remain unknown.
static bool
findBranchImmediateAcrossBlocks(MachineInstr &Branch, Register Reg,
                                const RISCVInstrInfo &TII, int64_t &Imm,
                                SmallVectorImpl<MachineInstr *> &Definitions) {
  const TargetRegisterInfo *TRI = &TII.getRegisterInfo();
  // Copies may change the incoming register, so visit block/register pairs.
  using State = std::pair<MachineBasicBlock *, Register>;
  SmallVector<State, 16> Worklist;
  SmallDenseSet<State, 32> Visited;
  std::optional<int64_t> Known;
  // Bound traversal cost; abandon the fold if the budget runs out. Both
  // scanned instructions and queued predecessors are charged.
  unsigned Budget = 4096;

  auto Scan = [&](MachineBasicBlock &MBB, MachineBasicBlock::instr_iterator End,
                  Register Reg) {
    for (auto I = End; I != MBB.instr_begin();) {
      MachineInstr &MI = *--I;
      if (!Budget--)
        return false;
      if (!MI.modifiesRegister(Reg, TRI))
        continue;
      int64_t Value;
      if (getBranchImmediate(MI, Reg, Value)) {
        if (Known && *Known != Value)
          return false;
        Known = Value;
        Definitions.push_back(&MI);
        return true;
      }
      // Only defined whole-register copies preserve the proven value.
      auto Copy = TII.isCopyInstr(MI);
      if (!Copy || Copy->Destination->getReg() != Reg ||
          Copy->Destination->getSubReg() || Copy->Source->getSubReg() ||
          Copy->Source->isUndef() || !Copy->Source->getReg().isPhysical() ||
          Copy->Source->getReg() == RISCV::X0)
        return false;
      Definitions.push_back(&MI);
      Reg = Copy->Source->getReg();
    }
    // An entry without a local definition may carry an unknown value.
    if (&MBB == &MBB.getParent()->front() || MBB.isEHPad() ||
        MBB.isEHFuncletEntry() || MBB.pred_empty())
      return false;
    for (MachineBasicBlock *Pred : MBB.predecessors()) {
      if (!Budget--)
        return false;
      Worklist.emplace_back(Pred, Reg);
    }
    return true;
  };

  if (!Scan(*Branch.getParent(), Branch.getIterator(), Reg))
    return false;
  while (!Worklist.empty()) {
    auto [MBB, IncomingReg] = Worklist.pop_back_val();
    if (!Visited.insert({MBB, IncomingReg}).second)
      continue;
    auto End = MBB->getFirstTerminator();
    // Writes after a conditional branch may not reach every outgoing edge.
    for (const MachineInstr &MI : make_range(End, MBB->end()))
      if (MI.modifiesRegister(IncomingReg, TRI))
        return false;
    if (!Scan(*MBB, End.getInstrIterator(), IncomingReg))
      return false;
  }
  if (!Known)
    return false;
  Imm = *Known;
  return true;
}

static bool
tryFoldBranchImmediate(MachineInstr &Branch, const RISCVInstrInfo &TII,
                       SmallVectorImpl<MachineInstr *> &Definitions) {
  if (Branch.getOpcode() != RISCV::BEQ && Branch.getOpcode() != RISCV::BNE)
    return false;
  if (!Branch.getOperand(0).getReg().isPhysical() ||
      !Branch.getOperand(1).getReg().isPhysical())
    return false;
  const TargetRegisterInfo *TRI = &TII.getRegisterInfo();
  int64_t Imm;
  // Preserve the local-first, right-operand-first choice.
  for (bool AcrossBlocks : {false, true}) {
    for (unsigned Operand : {1u, 0u}) {
      Register Reg = Branch.getOperand(Operand).getReg();
      if (Reg == RISCV::X0)
        continue;
      SmallVector<MachineInstr *, 8> ReachingDefs;
      if (AcrossBlocks) {
        if (!findBranchImmediateAcrossBlocks(Branch, Reg, TII, Imm,
                                             ReachingDefs))
          continue;
      } else if (MachineInstr *LI =
                     findBranchImmediate(Branch, Reg, TRI, Imm)) {
        ReachingDefs.push_back(LI);
      } else {
        continue;
      }
      if (Operand == 0) {
        // Preserve the remaining register's use flags when swapping operands.
        Branch.getOperand(0).setReg(Branch.getOperand(1).getReg());
        Branch.getOperand(0).setIsKill(Branch.getOperand(1).isKill());
        Branch.getOperand(0).setIsUndef(Branch.getOperand(1).isUndef());
        Branch.getOperand(0).setIsRenamable(Branch.getOperand(1).isRenamable());
      }
      Branch.setDesc(TII.get(Branch.getOpcode() == RISCV::BEQ ? RISCV::BEQI
                                                              : RISCV::BNEI));
      Branch.getOperand(1).ChangeToImmediate(Imm);
      llvm::append_range(Definitions, ReachingDefs);
      return true;
    }
  }
  return false;
}

// Stores and other branches may still need the constant. Delete definitions
// only after updating liveness.
static bool removeDeadBranchImmediateDefs(MachineFunction &MF,
                                          ArrayRef<MachineInstr *> Definitions,
                                          const TargetRegisterInfo *TRI) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  // Deleting physical definitions requires valid post-RA liveness.
  if (Definitions.empty() || !MRI.tracksLiveness() || MRI.getNumVirtRegs())
    return false;
  SmallPtrSet<MachineInstr *, 16> Candidates;
  SmallDenseSet<Register, 16> Registers;
  for (MachineInstr *MI : Definitions) {
    Register Reg = MI->getOperand(0).getReg();
    if (MRI.isReserved(Reg) || !RISCV::GPRRegClass.contains(Reg))
      continue;
    Registers.insert(Reg);
    Candidates.insert(MI);
  }
  // Preserve unrelated live-ins and lane masks. An overlapping entry prevents
  // rebuilding the affected GPR as a full-register live-in.
  LiveRegUnits UnchangedLiveIns(*TRI);
  for (const MachineBasicBlock &MBB : MF)
    for (const auto &LI : MBB.liveins())
      if (!Registers.contains(LI.PhysReg))
        UnchangedLiveIns.addReg(LI.PhysReg);
  Registers.remove_if([&](Register Reg) {
    // Debug uses are excluded from backward liveness, so retain their values.
    for (MCRegAliasIterator R(Reg, TRI, true); R.isValid(); ++R)
      if (any_of(MRI.use_operands(*R), [](const MachineOperand &MO) {
            return MO.getParent()->isDebugInstr();
          }))
        return true;
    return !UnchangedLiveIns.available(Reg);
  });
  // Protected instructions must remain, but their live-ins may still shrink.
  Candidates.remove_if([&](MachineInstr *MI) {
    return !Registers.contains(MI->getOperand(0).getReg()) || MI->isBundled() ||
           MI->peekDebugInstrNum() || MI->getFlag(MachineInstr::FrameSetup) ||
           MI->getFlag(MachineInstr::FrameDestroy) ||
           any_of(MI->operands(), [&](const MachineOperand &MO) {
             return MO.isRegMask() ||
                    (MO.isReg() && MO.isDef() &&
                     MO.getReg() != MI->getOperand(0).getReg());
           });
  });
  if (Registers.empty())
    return false;
  bool Changed = false;
  bool Removed;
  do {
    // Clear affected live-ins together so stale liveness cannot survive a loop.
    // Recompute only these GPRs, preserving unrelated lane masks.
    SmallSetVector<MachineBasicBlock *, 16> Worklist;
    for (MachineBasicBlock &MBB : MF) {
      for (Register Reg : Registers)
        MBB.removeLiveIn(Reg);
      Worklist.insert(&MBB);
    }
    // Propagate live-in changes only to predecessors.
    while (!Worklist.empty()) {
      MachineBasicBlock &MBB = *Worklist.pop_back_val();
      LivePhysRegs Live;
      computeLiveIns(Live, MBB);
      for (Register Reg : Registers)
        if (Live.contains(Reg) && !MBB.isLiveIn(Reg)) {
          MBB.addLiveIn(Reg);
          Worklist.insert_range(MBB.predecessors());
        }
    }

    Removed = false;
    // Skip erased instructions when stepping backward so dead copy chains
    // can collapse within this scan.
    for (MachineBasicBlock &MBB : MF) {
      LiveRegUnits Live(*TRI);
      Live.addLiveOuts(MBB);
      for (MachineInstr &MI : make_early_inc_range(reverse(MBB)))
        if (Candidates.contains(&MI) && MI.isDead(MRI, &Live)) {
          Candidates.erase(&MI);
          MI.eraseFromParent();
          Removed = Changed = true;
        } else {
          Live.stepBackward(MI);
        }
    }
    // Deletions may expose dead definitions in predecessor blocks.
  } while (Removed);
  for (MachineBasicBlock &MBB : MF)
    MBB.sortUniqueLiveIns();
  return Changed;
}

static bool runLateBranchOptOnMBB(MachineBasicBlock &MBB,
                                  const RISCVInstrInfo *RII) {
  MachineBasicBlock *TBB, *FBB;
  SmallVector<MachineOperand, 4> Cond;
  if (RII->analyzeBranch(MBB, TBB, FBB, Cond, /*AllowModify=*/false))
    return false;

  if (!TBB || Cond.size() != 3)
    return false;

  RISCVCC::CondCode CC = RISCVInstrInfo::getCondFromBranchOpc(Cond[0].getImm());
  assert(CC != RISCVCC::COND_INVALID);

  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  // Try and convert a conditional branch that can be evaluated statically
  // into an unconditional branch.
  int64_t C0, C1;
  if (!RISCVInstrInfo::isFromLoadImm(MRI, Cond[1], C0) ||
      !RISCVInstrInfo::isFromLoadImm(MRI, Cond[2], C1))
    return false;

  MachineBasicBlock *Folded =
      RISCVInstrInfo::evaluateCondBranch(CC, C0, C1) ? TBB : FBB;

  // At this point, its legal to optimize.
  RII->removeBranch(MBB);

  // Only need to insert a branch if we're not falling through.
  if (Folded) {
    DebugLoc DL = MBB.findBranchDebugLoc();
    RII->insertBranch(MBB, Folded, nullptr, {}, DL);
  }

  // Update the successors. Remove them all and add back the correct one.
  while (!MBB.succ_empty())
    MBB.removeSuccessor(MBB.succ_end() - 1);

  // If it's a fallthrough, we need to figure out where MBB is going.
  if (!Folded) {
    MachineFunction::iterator Fallthrough = ++MBB.getIterator();
    if (Fallthrough != MBB.getParent()->end())
      MBB.addSuccessor(&*Fallthrough);
  } else
    MBB.addSuccessor(Folded);

  return true;
}

static bool runLateBranchOptOnMF(MachineFunction &Fn) {
  auto &ST = Fn.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo *RII = ST.getInstrInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : Fn)
    Changed |= runLateBranchOptOnMBB(MBB, RII);
  if (ST.hasStdExtZibi()) {
    // Finish all proofs before removing definitions needed by later branches.
    SmallVector<MachineInstr *, 16> Definitions;
    for (MachineBasicBlock &MBB : Fn)
      for (MachineInstr &MI : MBB.terminators())
        Changed |= tryFoldBranchImmediate(MI, *RII, Definitions);
    Changed |=
        removeDeadBranchImmediateDefs(Fn, Definitions, ST.getRegisterInfo());
  }
  return Changed;
}

FunctionPass *llvm::createRISCVLateBranchOptLegacyPass() {
  return new RISCVLateBranchOptLegacy();
}

bool RISCVLateBranchOptLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  return runLateBranchOptOnMF(MF);
}

PreservedAnalyses
RISCVLateBranchOptPass::run(MachineFunction &MF,
                            MachineFunctionAnalysisManager &MFAM) {
  return runLateBranchOptOnMF(MF) ? getMachineFunctionPassPreservedAnalyses()
                                  : PreservedAnalyses::all();
}
