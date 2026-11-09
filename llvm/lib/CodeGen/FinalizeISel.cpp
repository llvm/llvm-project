//===-- llvm/CodeGen/FinalizeISel.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This pass expands Pseudo-instructions produced by ISel, fixes register
/// reservations, tries to remove FREEZEs, and may do machine frame information
/// adjustments. The pseudo instructions are used to allow the expansion to
/// contain control flow, such as a conditional move implemented with a
/// conditional branch and a phi, or an atomic operation implemented with a
/// loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/FinalizeISel.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "finalize-isel"

namespace {
  class FinalizeISel : public MachineFunctionPass {
  public:
    static char ID; // Pass identification, replacement for typeid
    FinalizeISel() : MachineFunctionPass(ID) {}

  private:
    bool runOnMachineFunction(MachineFunction &MF) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
} // end anonymous namespace

/// Return true if \p MI copies its register inputs into its result. This
/// includes target instructions that PeepholeOptimizer looks through like
/// copies, such as bitcasts.
static bool copiesInputs(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case TargetOpcode::PHI:
  case TargetOpcode::COPY:
  case TargetOpcode::INSERT_SUBREG:
  case TargetOpcode::SUBREG_TO_REG:
  case TargetOpcode::REG_SEQUENCE:
    return true;
  default:
    return MI.isBitcast() || MI.isRegSequenceLike() ||
           MI.isInsertSubregLike() || MI.isExtractSubregLike();
  }
}

/// Return true if an undefined input to \p MI can lead to uses reading
/// different values. A copy passes the input through, and a rematerializable
/// instruction may be recomputed at each read.
static bool passesUndefThrough(const MachineInstr &MI,
                               const TargetInstrInfo &TII) {
  return copiesInputs(MI) || TII.isReMaterializable(MI);
}

/// Return true if \p MO has the undef flag or reads a virtual register that
/// either has multiple definitions or none.
static bool mayReadUndef(const MachineOperand &MO,
                         const MachineRegisterInfo &MRI) {
  if (MO.isUndef())
    return true;
  Register Reg = MO.getReg();
  return Reg.isVirtual() && !MRI.hasOneDef(Reg);
}

/// Return true if the result of \p MI may be undefined. This is the case if
/// \p MI is an IMPLICIT_DEF, leaves part of its result unwritten, or passes an
/// undefined input through.
static bool definesUndef(const MachineInstr &MI, const MachineRegisterInfo &MRI,
                         const TargetInstrInfo &TII) {
  if (MI.isImplicitDef())
    return true;
  for (const MachineOperand &Def : MI.all_defs())
    if (Def.getSubReg())
      return true;
  if (MI.isRegSequence()) {
    LaneBitmask Covered;
    for (unsigned I = 2, E = MI.getNumOperands(); I < E; I += 2)
      Covered |= MRI.getTargetRegisterInfo()->getSubRegIndexLaneMask(
          MI.getOperand(I).getImm());
    if ((MRI.getMaxLaneMaskForVReg(MI.getOperand(0).getReg()) & ~Covered).any())
      return true;
  }
  return any_of(MI.all_uses(),
                [&](const MachineOperand &Use) {
                  return mayReadUndef(Use, MRI);
                }) &&
         passesUndefThrough(MI, TII);
}

/// Return true if \p MI has any poison-generating flags.
static bool hasPoisonFlags(const MachineInstr &MI) {
  return MI.getFlags() & MachineInstr::getPoisonGeneratingFlags();
}

/// Clear the poison-generating flags of every instruction whose result
/// \p Freeze reads through copies of virtual registers. Otherwise, a later pass
/// that looks through the copies would trust the flags for the frozen value. If
/// a flag is behind a physical register, the flags and the FREEZE are kept.
static bool dropCopiedPoisonFlags(const MachineInstr &Freeze,
                                  MachineRegisterInfo &MRI,
                                  SmallPtrSetImpl<MachineInstr *> &Visited) {
  bool Changed = false;
  SmallVector<Register, 8> Worklist;
  Worklist.push_back(Freeze.getOperand(1).getReg());
  while (!Worklist.empty()) {
    Register Reg = Worklist.pop_back_val();
    if (!Reg.isVirtual())
      continue;
    for (MachineInstr &Def : MRI.def_instructions(Reg)) {
      if (!Visited.insert(&Def).second)
        continue;
      if (hasPoisonFlags(Def)) {
        Def.clearFlags(MachineInstr::getPoisonGeneratingFlags());
        Changed = true;
      }
      if (copiesInputs(Def))
        for (const MachineOperand &Use : Def.all_uses())
          Worklist.push_back(Use.getReg());
    }
  }
  return Changed;
}

/// Push instructions that read the physical register \p Reg onto \p Worklist.
/// Check the instructions after \p MI until \p Reg is fully redefined. If
/// \p Reg is still live at the end of a block, continue into the successors it
/// is live into.
static void pushPhysRegUsers(MachineInstr &MI, MCRegister Reg,
                             const TargetRegisterInfo &TRI,
                             SmallVectorImpl<MachineInstr *> &Worklist) {
  using BlockRange =
      std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>;
  SmallVector<BlockRange, 4> Ranges;
  SmallPtrSet<MachineBasicBlock *, 4> Visited;
  Ranges.push_back(
      {MI.getParent(), std::next(MachineBasicBlock::iterator(MI))});
  while (!Ranges.empty()) {
    auto [MBB, Begin] = Ranges.pop_back_val();
    bool LiveOut = true;
    for (MachineInstr &UseMI : make_range(Begin, MBB->end())) {
      if (UseMI.isDebugInstr())
        continue;
      if (UseMI.readsRegister(Reg, &TRI))
        Worklist.push_back(&UseMI);
      if (UseMI.definesRegister(Reg, &TRI)) {
        LiveOut = false;
        break;
      }
    }
    if (!LiveOut)
      continue;
    for (MachineBasicBlock *Succ : MBB->successors())
      if (Succ->isLiveInOverlappedWith(Reg, TRI) && Visited.insert(Succ).second)
        Ranges.push_back({Succ, Succ->begin()});
  }
}

/// Push every instruction that may read a register defined by \p MI onto
/// \p Worklist.
static void pushUsers(MachineInstr &MI, const MachineRegisterInfo &MRI,
                      SmallVectorImpl<MachineInstr *> &Worklist) {
  for (const MachineOperand &Def : MI.all_defs()) {
    Register Reg = Def.getReg();
    if (Reg.isVirtual()) {
      for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg))
        Worklist.push_back(&UseMI);
    } else if (!Def.isDead() && !MRI.isConstantPhysReg(Reg)) {
      pushPhysRegUsers(MI, Reg.asMCReg(), *MRI.getTargetRegisterInfo(),
                       Worklist);
    }
  }
}

/// Turn each FREEZE into a COPY if the input cannot be undefined/poison.
static bool lowerDefinedFreezes(MachineFunction &MF) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<MachineInstr *, 8> Freezes;
  SmallVector<MachineInstr *, 8> Flagged; // Instrs with poison flags (e.g. nsw)
  SmallVector<MachineInstr *, 16> Worklist;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      if (MI.getOpcode() == TargetOpcode::FREEZE)
        Freezes.push_back(&MI);
      else if (definesUndef(MI, MRI, TII))
        pushUsers(MI, MRI, Worklist);
      if (hasPoisonFlags(MI))
        Flagged.push_back(&MI);
    }
  }

  // Find every instruction that may read an undefined value.
  SmallPtrSet<MachineInstr *, 16> ReadsUndef;
  while (!Worklist.empty()) {
    MachineInstr *MI = Worklist.pop_back_val();
    if (ReadsUndef.insert(MI).second && passesUndefThrough(*MI, TII))
      pushUsers(*MI, MRI, Worklist);
  }

  bool Changed = false;
  SmallPtrSet<MachineInstr *, 8> Visited;
  for (const MachineInstr *MI : Freezes)
    Changed |= dropCopiedPoisonFlags(*MI, MRI, Visited);

  // Find every instruction that may read poison because of a flag.
  for (MachineInstr *MI : Flagged)
    if (hasPoisonFlags(*MI))
      pushUsers(*MI, MRI, Worklist);
  SmallPtrSet<MachineInstr *, 16> ReadsPoison;
  while (!Worklist.empty()) {
    MachineInstr *MI = Worklist.pop_back_val();
    if (ReadsPoison.insert(MI).second && copiesInputs(*MI))
      pushUsers(*MI, MRI, Worklist);
  }

  for (MachineInstr *MI : Freezes) {
    if (mayReadUndef(MI->getOperand(1), MRI) || ReadsUndef.contains(MI) ||
        ReadsPoison.contains(MI))
      continue;
    LLVM_DEBUG(dbgs() << "Converting to COPY: " << *MI);
    MI->setDesc(TII.get(TargetOpcode::COPY));
    Changed = true;
  }
  return Changed;
}

static std::pair<bool, bool> runImpl(MachineFunction &MF) {
  bool Changed = false;
  bool PreserveCFG = true;
  bool SawFreeze = false;
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const TargetLowering *TLI = MF.getSubtarget().getTargetLowering();

  // GlobalISel finalizes lowering in InstructionSelect.
  if (!MF.getProperties().hasSelected())
    TLI->finalizeLowering(MF);

  // Iterate through each instruction in the function, looking for pseudos.
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;
    for (MachineBasicBlock::iterator MBBI = MBB->begin(), MBBE = MBB->end();
         MBBI != MBBE; ) {
      MachineInstr &MI = *MBBI++;

      // Note whether there is a FREEZE to lower below.
      if (MI.getOpcode() == TargetOpcode::FREEZE)
        SawFreeze = true;

      // Set AdjustsStack to true if the instruction selector emits a stack
      // frame setup instruction or a stack aligning inlineasm.
      if (TII->isFrameInstr(MI) || MI.isStackAligningInlineAsm())
        MF.getFrameInfo().setAdjustsStack(true);

      // If MI is a pseudo, expand it.
      if (MI.usesCustomInsertionHook()) {
        Changed = true;
        MachineBasicBlock *NewMBB = TLI->EmitInstrWithCustomInserter(MI, MBB);
        // The expansion may involve new basic blocks.
        if (NewMBB != MBB) {
          PreserveCFG = false;
          MBB = NewMBB;
          I = NewMBB->getIterator();
          MBBI = NewMBB->begin();
          MBBE = NewMBB->end();
        }
      }
    }
  }

  if (SawFreeze)
    Changed |= lowerDefinedFreezes(MF);
  return {Changed, PreserveCFG};
}

char FinalizeISel::ID = 0;
char &llvm::FinalizeISelID = FinalizeISel::ID;
INITIALIZE_PASS(FinalizeISel, DEBUG_TYPE,
                "Finalize ISel and expand pseudo-instructions", false, false)

bool FinalizeISel::runOnMachineFunction(MachineFunction &MF) {
  return runImpl(MF).first;
}

PreservedAnalyses FinalizeISelPass::run(MachineFunction &MF,
                                        MachineFunctionAnalysisManager &) {
  auto [Changed, PreserveCFG] = runImpl(MF);
  if (!Changed)
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  if (PreserveCFG)
    PA.preserveSet<CFGAnalyses>();
  return PA;
}
