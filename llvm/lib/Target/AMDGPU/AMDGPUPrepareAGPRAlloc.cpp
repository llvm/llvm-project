//===-- AMDGPUPrepareAGPRAlloc.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Make simple transformations to relax register constraints for cases which can
// allocate to AGPRs or VGPRs. Replace materialize of inline immediates into
// AGPR or VGPR with a pseudo with an AV_* class register constraint. This
// allows later passes to inflate the register class if necessary. The register
// allocator does not know to replace instructions to relax constraints.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUPrepareAGPRAlloc.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-prepare-agpr-alloc"

namespace {

class AMDGPUPrepareAGPRAllocImpl {
private:
  const SIInstrInfo &TII;
  MachineRegisterInfo &MRI;

  bool isAV64Imm(const MachineOperand &MO) const;
  bool constrainMFMACD(MachineInstr &MI);

public:
  AMDGPUPrepareAGPRAllocImpl(const GCNSubtarget &ST, MachineRegisterInfo &MRI)
      : TII(*ST.getInstrInfo()), MRI(MRI) {}
  bool run(MachineFunction &MF, bool Optimize);
};

class AMDGPUPrepareAGPRAllocLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPrepareAGPRAllocLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AMDGPU Prepare AGPR Alloc"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUPrepareAGPRAllocLegacy, DEBUG_TYPE,
                "AMDGPU Prepare AGPR Alloc", false, false)

char AMDGPUPrepareAGPRAllocLegacy::ID = 0;

char &llvm::AMDGPUPrepareAGPRAllocLegacyID = AMDGPUPrepareAGPRAllocLegacy::ID;

bool AMDGPUPrepareAGPRAllocLegacy::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  return AMDGPUPrepareAGPRAllocImpl(ST, MF.getRegInfo())
      .run(MF, !skipFunction(MF.getFunction()));
}

PreservedAnalyses
AMDGPUPrepareAGPRAllocPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!AMDGPUPrepareAGPRAllocImpl(ST, MF.getRegInfo())
           .run(MF, !MF.getFunction().hasOptNone() &&
                        !shouldSkipOptimizationForOptBisect(MF.getFunction())))
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}

bool AMDGPUPrepareAGPRAllocImpl::isAV64Imm(const MachineOperand &MO) const {
  return MO.isImm() && TII.isLegalAV64PseudoImm(MO.getImm());
}

bool AMDGPUPrepareAGPRAllocImpl::constrainMFMACD(MachineInstr &MI) {
  auto Policy = TII.getMFMACDRegClass(MI);
  if (!Policy)
    return false;
  MachineFunction &MF = *MI.getMF();
  bool UseAGPR = *Policy == AMDGPU::MFMACDRegClass::AGPR;
  if ((UseAGPR && MRI.isReserved(AMDGPU::AGPR0)) ||
      (!UseAGPR && !MF.getSubtarget<GCNSubtarget>().hasGFX90AInsts())) {
    MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported(
        MF.getFunction(),
        UseAGPR ? "amdgpu-mfma-cd-regclass: AGPRs are unavailable"
                : "amdgpu-mfma-cd-regclass: VGPR C/D is unavailable",
        MI.getDebugLoc()));
    return false;
  }
  int Opcode = UseAGPR ? AMDGPU::getAGPRFormOp(MI.getOpcode())
                       : AMDGPU::getVGPRFormOp(MI.getOpcode());
  if (Opcode == -1)
    return false; // Already uses the requested bank.

  const MCInstrDesc &Desc = TII.get(Opcode);
  const TargetRegisterClass *RC = TII.getRegClass(Desc, 0);
  MachineBasicBlock &MBB = *MI.getParent();
  MachineOperand &Src2 = *TII.getNamedOperand(MI, AMDGPU::OpName::src2);
  // Keep other uses and definitions in their original bank. Coalescing can
  // remove these copies when adjacent instructions have compatible policies.
  if (Src2.isReg()) {
    Register NewSrc = MRI.createVirtualRegister(RC);
    BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(TargetOpcode::COPY), NewSrc)
        .add(Src2);
    Src2.setReg(NewSrc);
    Src2.setSubReg(0);
    Src2.setIsUndef(false);
    Src2.setIsKill(true);
  }
  MachineOperand &Dst = MI.getOperand(0);
  Register OldDst = Dst.getReg();
  Register NewDst = MRI.createVirtualRegister(RC);
  BuildMI(MBB, std::next(MI.getIterator()), MI.getDebugLoc(),
          TII.get(TargetOpcode::COPY))
      .addReg(OldDst, RegState::Define | getDeadRegState(Dst.isDead()),
              Dst.getSubReg())
      .addReg(NewDst, RegState::Kill);
  Dst.setReg(NewDst);
  Dst.setSubReg(0);
  Dst.setIsDead(false);
  MI.setDesc(Desc);
  return true;
}

bool AMDGPUPrepareAGPRAllocImpl::run(MachineFunction &MF, bool Optimize) {
  // Hard requests are required even at -O0/optnone. Only the pre-existing
  // immediate-materialization optimization remains optional.
  Optimize &= MF.getTarget().getOptLevel() != CodeGenOptLevel::None &&
              !MRI.isReserved(AMDGPU::AGPR0);

  const MCInstrDesc &AVImmPseudo32 = TII.get(AMDGPU::AV_MOV_B32_IMM_PSEUDO);
  const MCInstrDesc &AVImmPseudo64 = TII.get(AMDGPU::AV_MOV_B64_IMM_PSEUDO);

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      Changed |= constrainMFMACD(MI);
      if (!Optimize)
        continue;
      if ((MI.getOpcode() == AMDGPU::V_MOV_B32_e32 &&
           TII.isInlineConstant(MI, 1)) ||
          (MI.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64 &&
           MI.getOperand(1).isImm())) {
        MI.setDesc(AVImmPseudo32);
        Changed = true;
        continue;
      }

      // TODO: If only half of the value is rewritable, is it worth splitting it
      // up?
      if ((MI.getOpcode() == AMDGPU::V_MOV_B64_e64 ||
           MI.getOpcode() == AMDGPU::V_MOV_B64_PSEUDO) &&
          isAV64Imm(MI.getOperand(1))) {
        MI.setDesc(AVImmPseudo64);
        Changed = true;
        continue;
      }
    }
  }

  return Changed;
}
