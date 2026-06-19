//===-- AMDGPUPrivateObjectVGPRs.cpp - Lower VGPR-as-memory accesses ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lowers the SI_VGPR_FRAME_{LOAD,STORE} pseudos produced for "VGPR as memory"
/// objects (allocas in AMDGPUAS::VGPR) into register copies into/out of a
/// virtual VGPR tuple that backs the per-function VGPR file. Each pseudo
/// carries a constant byte offset, which selects the dword (subregister) to
/// copy.
///
/// This runs once the function is out of SSA form (so the single backing tuple
/// can be defined by several subregister copies) and while LiveIntervals is
/// available. The backing tuple has lane-divergent liveness (its subregisters
/// are written and read independently), which the whole-register LiveVariables
/// analysis cannot represent; the pass therefore updates the subregister-aware
/// LiveIntervals directly.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-private-object-vgprs"

namespace {

class AMDGPUPrivateObjectVGPRs : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPrivateObjectVGPRs() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Private Object VGPRs";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS(AMDGPUPrivateObjectVGPRs, DEBUG_TYPE,
                "AMDGPU Private Object VGPRs", false, false)

char AMDGPUPrivateObjectVGPRs::ID = 0;

char &llvm::AMDGPUPrivateObjectVGPRsID = AMDGPUPrivateObjectVGPRs::ID;

bool AMDGPUPrivateObjectVGPRs::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Collect the pseudos and determine how many dwords the backing tuple needs.
  SmallVector<MachineInstr *, 8> Worklist;
  unsigned NumDwords = 0;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opc = MI.getOpcode();
      if (Opc != AMDGPU::SI_VGPR_FRAME_LOAD &&
          Opc != AMDGPU::SI_VGPR_FRAME_STORE)
        continue;
      unsigned ByteOffset = MI.getOperand(1).getImm();
      NumDwords = std::max(NumDwords, ByteOffset / 4 + 1);
      Worklist.push_back(&MI);
    }
  }

  if (Worklist.empty())
    return false;

  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();

  const TargetRegisterClass *RC = TRI->getVGPRClassForBitWidth(NumDwords * 32);
  assert(RC && "no VGPR register class for VGPR-as-memory object");
  Register Storage = MRI.createVirtualRegister(RC);

  // Define the whole tuple up front so partial (subregister) writes and reads
  // of uninitialized lanes are well formed.
  MachineBasicBlock &Entry = MF.front();
  MachineInstr *ImpDef = BuildMI(Entry, Entry.begin(), DebugLoc(),
                                 TII->get(TargetOpcode::IMPLICIT_DEF), Storage);
  LIS->InsertMachineInstrInMaps(*ImpDef);

  for (MachineInstr *MI : Worklist) {
    MachineBasicBlock &MBB = *MI->getParent();
    const DebugLoc &DL = MI->getDebugLoc();
    unsigned Dword = MI->getOperand(1).getImm() / 4;
    unsigned SubReg = NumDwords == 1
                          ? AMDGPU::NoSubRegister
                          : SIRegisterInfo::getSubRegFromChannel(Dword);

    MachineInstr *Copy;
    if (MI->getOpcode() == AMDGPU::SI_VGPR_FRAME_LOAD) {
      Register Dst = MI->getOperand(0).getReg();
      Copy = BuildMI(MBB, *MI, DL, TII->get(TargetOpcode::COPY), Dst)
                 .addReg(Storage, {}, SubReg);
    } else {
      Register Src = MI->getOperand(0).getReg();
      Copy = BuildMI(MBB, *MI, DL, TII->get(TargetOpcode::COPY))
                 .addReg(Storage, RegState::Define, SubReg)
                 .addReg(Src);
    }
    // The copy takes the pseudo's slot, so the intervals of the copied
    // load/store operand stay valid.
    LIS->ReplaceMachineInstrInMaps(*MI, *Copy);
    MI->eraseFromParent();
  }

  // The backing tuple is brand new; compute its (subregister) live interval.
  LiveInterval &LI = LIS->createAndComputeVirtRegInterval(Storage);

  // Independent dwords (and the entry IMPLICIT_DEF for never-written lanes)
  // form disconnected value-number components within the single tuple, which an
  // individual live interval must not contain. Split them into separate
  // virtual registers, exactly as the register coalescer does for the intervals
  // it leaves behind.
  SmallVector<LiveInterval *, 4> SplitLIs;
  LIS->splitSeparateComponents(LI, SplitLIs);

  return true;
}
