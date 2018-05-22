//===-- SIFixWWMLiveness.cpp - Fix WWM live intervals ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Computations in WWM can overwrite values in inactive channels for
/// variables that the register allocator thinks are dead. This pass adds fake
/// uses of those variables to WWM instructions to make sure that they aren't
/// overwritten.
///
/// As an example, consider this snippet:
/// %vgpr0 = V_MOV_B32_e32 0.0
/// if (...) {
///   %vgpr1 = ...
///   %vgpr2 = WWM killed %vgpr1
///   ... = killed %vgpr2
///   %vgpr0 = V_MOV_B32_e32 1.0
/// }
/// ... = %vgpr0
///
/// The live intervals of %vgpr0 don't overlap with those of %vgpr1. Normally,
/// we can safely allocate %vgpr0 and %vgpr1 in the same register, since
/// writing %vgpr1 would only write to channels that would be clobbered by the
/// second write to %vgpr0 anyways. But if %vgpr1 is written with WWM enabled,
/// it would clobber even the inactive channels for which the if-condition is
/// false, for which %vgpr0 is supposed to be 0. This pass adds an implicit use
/// of %vgpr0 to the WWM instruction to make sure they aren't allocated to the
/// same register.
///
/// In general, we need to figure out what registers might have their inactive
/// channels which are eventually used accidentally clobbered by a WWM
/// instruction. We approximate this using two conditions:
///
/// 1. A definition of the variable reaches the WWM instruction.
/// 2. The variable would be live at the WWM instruction if all its defs were
/// partial defs (i.e. considered as a use), ignoring normal uses.
///
/// If a register matches both conditions, then we add an implicit use of it to
/// the WWM instruction. Condition #2 is the heart of the matter: every
/// definition is really a partial definition, since every VALU instruction is
/// implicitly predicated.  We can usually ignore this, but WWM forces us not
/// to. Condition #1 prevents false positives if the variable is undefined at
/// the WWM instruction anyways. This is overly conservative in certain cases,
/// especially in uniform control flow, but this is a workaround anyways until
/// LLVM gains the notion of predicated uses and definitions of variables.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-wwm-liveness"

namespace {

class SIFixWWMLiveness : public MachineFunctionPass {
private:
  LiveIntervals *LIS = nullptr;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;

public:
  static char ID;

  SIFixWWMLiveness() : MachineFunctionPass(ID) {
    initializeSIFixWWMLivenessPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  bool runOnWWMInstruction(MachineInstr &MI);

  void addDefs(const MachineInstr &MI, SparseBitVector<> &set);

  StringRef getPassName() const override { return "SI Fix WWM Liveness"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // Should preserve the same set that TwoAddressInstructions does.
    AU.addPreserved<SlotIndexes>();
    AU.addPreserved<LiveIntervals>();
    AU.addPreservedID(LiveVariablesID);
    AU.addPreservedID(MachineLoopInfoID);
    AU.addPreservedID(MachineDominatorsID);
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS(SIFixWWMLiveness, DEBUG_TYPE,
                "SI fix WWM liveness", false, false)

char SIFixWWMLiveness::ID = 0;

char &llvm::SIFixWWMLivenessID = SIFixWWMLiveness::ID;

FunctionPass *llvm::createSIFixWWMLivenessPass() {
  return new SIFixWWMLiveness();
}

void SIFixWWMLiveness::addDefs(const MachineInstr &MI, SparseBitVector<> &Regs)
{
  for (const MachineOperand &Op : MI.defs()) {
    if (Op.isReg()) {
      unsigned Reg = Op.getReg();
      if (TRI->isVGPR(*MRI, Reg))
        Regs.set(Reg);
    }
  }
}

bool SIFixWWMLiveness::runOnWWMInstruction(MachineInstr &WWM) {
  MachineBasicBlock *MBB = WWM.getParent();

  // Compute the registers that are live out of MI by figuring out which defs
  // are reachable from MI.
  SparseBitVector<> LiveOut;

  for (auto II = MachineBasicBlock::iterator(WWM), IE =
       MBB->end(); II != IE; ++II) {
    addDefs(*II, LiveOut);
  }

  for (df_iterator<MachineBasicBlock *> I = ++df_begin(MBB),
                                        E = df_end(MBB);
       I != E; ++I) {
    for (const MachineInstr &MI : **I) {
      addDefs(MI, LiveOut);
    }
  }

  // Compute the registers that reach MI.
  SparseBitVector<> Reachable;

  for (auto II = ++MachineBasicBlock::reverse_iterator(WWM), IE =
       MBB->rend(); II != IE; ++II) {
    addDefs(*II, Reachable);
  }

  for (idf_iterator<MachineBasicBlock *> I = ++idf_begin(MBB),
                                         E = idf_end(MBB);
       I != E; ++I) {
    for (const MachineInstr &MI : **I) {
      addDefs(MI, Reachable);
    }
  }

  // find the intersection, and add implicit uses.
  LiveOut &= Reachable;

  bool Modified = false;
  for (unsigned Reg : LiveOut) {
    WWM.addOperand(MachineOperand::CreateReg(Reg, false, /*isImp=*/true));
    if (LIS) {
      // FIXME: is there a better way to update the live interval?
      LIS->removeInterval(Reg);
      LIS->createAndComputeVirtRegInterval(Reg);
    }
    Modified = true;
  }

  return Modified;
}

bool SIFixWWMLiveness::runOnMachineFunction(MachineFunction &MF) {
  bool Modified = false;

  // This doesn't actually need LiveIntervals, but we can preserve them.
  LIS = getAnalysisIfAvailable<LiveIntervals>();

  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();

  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == AMDGPU::EXIT_WWM) {
        Modified |= runOnWWMInstruction(MI);
      }
    }
  }

  return Modified;
}
