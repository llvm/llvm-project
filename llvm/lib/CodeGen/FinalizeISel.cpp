//===-- llvm/CodeGen/FinalizeISel.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This pass expands Pseudo-instructions produced by ISel, fixes register
/// reservations and may do machine frame information adjustments.
/// The pseudo instructions are used to allow the expansion to contain control
/// flow, such as a conditional move implemented with a conditional branch and a
/// phi, or an atomic operation implemented with a loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/FinalizeISel.h"
#include "llvm/ADT/BitVector.h"
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

/// Return true if \p MI copies its inputs into its result, or may be
/// rematerialized. Either way, an undefined input can make different uses of
/// the result observe different values.
static bool passesUndefThrough(const MachineInstr &MI,
                               const TargetInstrInfo &TII) {
  switch (MI.getOpcode()) {
  case TargetOpcode::PHI:
  case TargetOpcode::COPY:
  case TargetOpcode::INSERT_SUBREG:
  case TargetOpcode::SUBREG_TO_REG:
  case TargetOpcode::REG_SEQUENCE:
    return true;
  default:
    return TII.isReMaterializable(MI);
  }
}

/// Return true if the register use \p MO may read an undefined value. Either it
/// is marked undef, or it reads a virtual register without a single definition.
/// A live-in virtual register without a definition holds the incoming physical
/// register value.
static bool mayReadUndef(const MachineOperand &MO,
                         const MachineRegisterInfo &MRI) {
  if (MO.isUndef())
    return true;
  Register Reg = MO.getReg();
  return Reg.isVirtual() && !MRI.hasOneDef(Reg) && !MRI.getLiveInPhysReg(Reg);
}

/// Return true if \p MI may define an undefined register, i.e. an IMPLICIT_DEF,
/// a partial register definition, or an instruction that copies or may
/// rematerialize an undefined operand.
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
  return passesUndefThrough(MI, TII) &&
         any_of(MI.all_uses(), [&](const MachineOperand &Use) {
           return mayReadUndef(Use, MRI);
         });
}

/// Add the virtual registers defined by \p MI to \p MayBeUndef.
static void markDefsUndef(const MachineInstr &MI, BitVector &MayBeUndef,
                          SmallVectorImpl<Register> &Worklist) {
  for (const MachineOperand &Def : MI.all_defs()) {
    Register Reg = Def.getReg();
    if (Reg.isVirtual() && !MayBeUndef.test(Reg.virtRegIndex())) {
      MayBeUndef.set(Reg.virtRegIndex());
      Worklist.push_back(Reg);
    }
  }
}

/// A FREEZE is only needed if its input may be undefined. Turn every other
/// FREEZE into a COPY right after instruction selection.
static bool lowerDefinedFreezes(MachineFunction &MF) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<MachineInstr *, 8> Freezes;
  BitVector MayBeUndef(MRI.getNumVirtRegs());
  SmallVector<Register, 16> Worklist;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      if (MI.getOpcode() == TargetOpcode::FREEZE)
        Freezes.push_back(&MI);
      else if (definesUndef(MI, MRI, TII))
        markDefsUndef(MI, MayBeUndef, Worklist);
    }
  }

  while (!Worklist.empty()) {
    Register Reg = Worklist.pop_back_val();
    for (const MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg))
      if (passesUndefThrough(UseMI, TII))
        markDefsUndef(UseMI, MayBeUndef, Worklist);
  }

  bool Changed = false;
  for (MachineInstr *MI : Freezes) {
    const MachineOperand &Src = MI->getOperand(1);
    if (mayReadUndef(Src, MRI) ||
        (Src.getReg().isVirtual() &&
         MayBeUndef.test(Src.getReg().virtRegIndex())))
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

      // A FREEZE emitted by a custom inserter before MBBI is missed, and stays.
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
