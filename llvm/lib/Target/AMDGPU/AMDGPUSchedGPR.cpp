//===-- AMDGPUSchedGPR.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lower the V_SCHEDULE_VGPR_B*, S_SCHEDULE_SGPR_B*, and A_SCHEDULE_AGPR_B*
/// pseudos produced from llvm.amdgcn.sched.{vgpr,sgpr,agpr}. For each pseudo
/// this pass:
///   1. reads the requested register number and strict flag from the immediate
///      operands,
///   2. attaches an AMDGPURI::{VGPR,SGPR,AGPR}RegHint register allocation hint
///      to the destination (and source) vreg so the allocator prefers that
///      exact physical register,
///   3. propagates the hint through COPY / REG_SEQUENCE / INSERT_SUBREG /
///      SUBREG_TO_REG so it survives coalescing, and
///   4. replaces the pseudo with a plain COPY.
///
/// The hint is advisory unless strict: SIRegisterInfo::getRegAllocationHints
/// only reorders the allocation candidates, so a non-strict hint falls back to
/// the default order and the pass never forces a spill.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSchedGPR.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-sched-gpr"

enum ScheduleGPRKind { SGK_None, SGK_VGPR, SGK_SGPR, SGK_AGPR };

static ScheduleGPRKind getScheduleGPRKind(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::V_SCHEDULE_VGPR_B32:
  case AMDGPU::V_SCHEDULE_VGPR_B64:
  case AMDGPU::V_SCHEDULE_VGPR_B128:
  case AMDGPU::V_SCHEDULE_VGPR_B256:
  case AMDGPU::V_SCHEDULE_VGPR_B512:
    return SGK_VGPR;
  case AMDGPU::S_SCHEDULE_SGPR_B32:
  case AMDGPU::S_SCHEDULE_SGPR_B64:
  case AMDGPU::S_SCHEDULE_SGPR_B128:
  case AMDGPU::S_SCHEDULE_SGPR_B256:
  case AMDGPU::S_SCHEDULE_SGPR_B512:
    return SGK_SGPR;
  case AMDGPU::A_SCHEDULE_AGPR_B32:
  case AMDGPU::A_SCHEDULE_AGPR_B64:
  case AMDGPU::A_SCHEDULE_AGPR_B128:
  case AMDGPU::A_SCHEDULE_AGPR_B256:
  case AMDGPU::A_SCHEDULE_AGPR_B512:
    return SGK_AGPR;
  default:
    return SGK_None;
  }
}

/// Walk the SSA use chain of \p Reg and propagate the hint through
/// value-preserving pseudos so the preference reaches the vregs that survive
/// coalescing.
static void propagateHint(MachineRegisterInfo &MRI, Register Reg,
                          unsigned HintValue, unsigned HintKind,
                          SmallDenseSet<unsigned, 32> &Visited) {
  if (!Reg.isVirtual() || !Visited.insert(Reg.id()).second)
    return;

  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
    Register DefReg;
    switch (UseMI.getOpcode()) {
    case TargetOpcode::COPY:
    case TargetOpcode::REG_SEQUENCE:
    case TargetOpcode::INSERT_SUBREG:
    case TargetOpcode::SUBREG_TO_REG:
      DefReg = UseMI.getOperand(0).getReg();
      break;
    default:
      continue;
    }
    if (!DefReg.isVirtual())
      continue;

    // Do not clobber an existing, conflicting hint.
    std::pair<unsigned, Register> Existing = MRI.getRegAllocationHint(DefReg);
    if (Existing.first != 0 &&
        (Existing.first != HintKind || Existing.second != HintValue))
      continue;

    MRI.setRegAllocationHint(DefReg, HintKind, HintValue);
    propagateHint(MRI, DefReg, HintValue, HintKind, Visited);
  }
}

static bool runSchedGPR(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  SmallVector<MachineInstr *, 16> ToErase;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      ScheduleGPRKind Kind = getScheduleGPRKind(MI.getOpcode());
      if (Kind == SGK_None)
        continue;

      Register DstReg = MI.getOperand(0).getReg();
      const MachineOperand &SrcMO = MI.getOperand(1);
      Register SrcReg = SrcMO.getReg();
      unsigned RegNum = MI.getOperand(2).getImm();
      bool Strict = MI.getOperand(3).getImm() != 0;

      unsigned HintKind;
      switch (Kind) {
      case SGK_VGPR:
        HintKind = Strict ? AMDGPURI::StrictVGPRRegHint : AMDGPURI::VGPRRegHint;
        break;
      case SGK_SGPR:
        HintKind = Strict ? AMDGPURI::StrictSGPRRegHint : AMDGPURI::SGPRRegHint;
        break;
      case SGK_AGPR:
        HintKind = Strict ? AMDGPURI::StrictAGPRRegHint : AMDGPURI::AGPRRegHint;
        break;
      default:
        llvm_unreachable("unexpected schedule GPR kind");
      }

      LLVM_DEBUG(dbgs() << "  sched."
                        << (Kind == SGK_VGPR   ? "vgpr"
                            : Kind == SGK_SGPR ? "sgpr"
                                               : "agpr")
                        << ": " << printReg(DstReg) << " <- "
                        << printReg(SrcReg) << " reg " << RegNum << " strict "
                        << Strict << '\n');

      // Hint both the destination and the source so the preference survives
      // whichever side coalescing keeps.
      if (DstReg.isVirtual())
        MRI.setRegAllocationHint(DstReg, HintKind, RegNum);
      if (SrcReg.isVirtual())
        MRI.setRegAllocationHint(SrcReg, HintKind, RegNum);

      SmallDenseSet<unsigned, 32> Visited;
      if (DstReg.isVirtual())
        propagateHint(MRI, DstReg, RegNum, HintKind, Visited);

      // Replace the pseudo with a plain COPY.
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(TargetOpcode::COPY), DstReg)
          .addReg(SrcReg, getRegState(SrcMO), SrcMO.getSubReg());

      ToErase.push_back(&MI);
      Changed = true;
    }
  }

  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();

  return Changed;
}

namespace {

class AMDGPUSchedGPR : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUSchedGPR() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return runSchedGPR(MF);
  }

  StringRef getPassName() const override { return "AMDGPU Sched GPR"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS(AMDGPUSchedGPR, DEBUG_TYPE, "AMDGPU Sched GPR", false, false)

char AMDGPUSchedGPR::ID = 0;

char &llvm::AMDGPUSchedGPRID = AMDGPUSchedGPR::ID;

PreservedAnalyses
AMDGPUSchedGPRPass::run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM) {
  if (!runSchedGPR(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
