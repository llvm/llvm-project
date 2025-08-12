//===-- AMDGPURewriteAGPRCopyMFMA.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file \brief Try to replace MFMA instructions using VGPRs with MFMA
/// instructions using AGPRs. We expect MFMAs to be selected using VGPRs, and
/// only use AGPRs if it helps avoid spilling. In this case, the MFMA will have
/// copies between AGPRs and VGPRs and the AGPR variant of an MFMA pseudo. This
/// pass will attempt to delete the cross register bank copy and replace the
/// MFMA opcode.
///
/// TODO:
///  - Handle non-tied dst+src2 cases. We need to try to find a copy from an
///    AGPR from src2, or reassign src2 to an available AGPR (which should work
///    in the common case of a load).
///
///  - Handle multiple MFMA uses of the same register. e.g. chained MFMAs that
///    can be rewritten as a set
///
///  - Update LiveIntervals incrementally instead of recomputing from scratch
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-rewrite-agpr-copy-mfma"

namespace {

class AMDGPURewriteAGPRCopyMFMAImpl {
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  MachineRegisterInfo &MRI;
  VirtRegMap &VRM;
  LiveRegMatrix &LRM;
  LiveIntervals &LIS;

public:
  AMDGPURewriteAGPRCopyMFMAImpl(MachineFunction &MF, VirtRegMap &VRM,
                                LiveRegMatrix &LRM, LiveIntervals &LIS)
      : ST(MF.getSubtarget<GCNSubtarget>()), TII(*ST.getInstrInfo()),
        TRI(*ST.getRegisterInfo()), MRI(MF.getRegInfo()), VRM(VRM), LRM(LRM),
        LIS(LIS) {}

  // TODO: Remove this restriction
  bool mfmaHasSameSrc2AndDstReg(const MachineInstr &MI) const {
    const MachineOperand *Src2 = TII.getNamedOperand(MI, AMDGPU::OpName::src2);
    const MachineOperand *Dst = TII.getNamedOperand(MI, AMDGPU::OpName::vdst);
    return Src2->getReg() == Dst->getReg() &&
           Src2->getSubReg() == Dst->getSubReg();
  }

  bool isRewriteCandidate(const MachineInstr &MI) const {
    return TII.isMAI(MI) &&
           AMDGPU::getMFMASrcCVDstAGPROp(MI.getOpcode()) != -1 &&
           mfmaHasSameSrc2AndDstReg(MI);
  }

  /// Compute the register class constraints based on the uses of \p Reg,
  /// excluding MFMA uses from which can be rewritten to change the register
  /// class constraint. This should be nearly identical to
  /// MachineRegisterInfo::recomputeRegClass.
  const TargetRegisterClass *
  recomputeRegClassExceptRewritable(Register Reg,
                                    const TargetRegisterClass *OldRC,
                                    const TargetRegisterClass *NewRC) const;

  bool run(MachineFunction &MF) const;
};

const TargetRegisterClass *
AMDGPURewriteAGPRCopyMFMAImpl::recomputeRegClassExceptRewritable(
    Register Reg, const TargetRegisterClass *OldRC,
    const TargetRegisterClass *NewRC) const {

  // Accumulate constraints from all uses.
  for (MachineOperand &MO : MRI.reg_nodbg_operands(Reg)) {
    // Apply the effect of the given operand to NewRC.
    MachineInstr *MI = MO.getParent();

    // We can swap the classes of dst + src2 as a pair to AGPR, so ignore the
    // effects of rewrite candidates. It just so happens that we can use either
    // AGPR or VGPR in src0/src1, so don't bother checking the constraint
    // effects of the individual operands.
    if (isRewriteCandidate(*MI))
      continue;

    unsigned OpNo = &MO - &MI->getOperand(0);
    NewRC = MI->getRegClassConstraintEffect(OpNo, NewRC, &TII, &TRI);
    if (!NewRC || NewRC == OldRC)
      return nullptr;
  }

  return NewRC;
}

bool AMDGPURewriteAGPRCopyMFMAImpl::run(MachineFunction &MF) const {
  // This only applies on subtargets that have a configurable AGPR vs. VGPR
  // allocation.
  if (!ST.hasGFX90AInsts())
    return false;

  // Early exit if no AGPRs were assigned.
  if (!LRM.isPhysRegUsed(AMDGPU::AGPR0)) {
    LLVM_DEBUG(dbgs() << "skipping function that did not allocate AGPRs\n");
    return false;
  }

  bool MadeChange = false;

  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register VReg = Register::index2VirtReg(I);
    Register PhysReg = VRM.getPhys(VReg);
    if (!PhysReg)
      continue;

    // Find AV_* registers assigned to AGPRs.
    const TargetRegisterClass *VirtRegRC = MRI.getRegClass(VReg);
    if (!TRI.hasAGPRs(VirtRegRC))
      continue;

    const TargetRegisterClass *AssignedRC = VirtRegRC;
    if (TRI.hasVGPRs(VirtRegRC)) {
      // If this is an AV register, we have to check if the actual assignment is
      // to an AGPR
      AssignedRC = TRI.getPhysRegBaseClass(PhysReg);
      if (!TRI.isAGPRClass(AssignedRC))
        continue;
    }

    LiveInterval &LI = LIS.getInterval(VReg);

    // TODO: Test multiple uses
    for (VNInfo *VNI : LI.vnis()) {
      MachineInstr *DefMI = LIS.getInstructionFromIndex(VNI->def);

      // TODO: Handle SplitKit produced copy bundles for partially defined
      // registers.
      if (!DefMI || !DefMI->isFullCopy())
        continue;

      Register CopySrcReg = DefMI->getOperand(1).getReg();
      if (!CopySrcReg.isVirtual())
        continue;

      LiveInterval &CopySrcLI = LIS.getInterval(CopySrcReg);
      LiveQueryResult LRQ = CopySrcLI.Query(VNI->def.getRegSlot());
      MachineInstr *CopySrcMI = LIS.getInstructionFromIndex(LRQ.valueIn()->def);
      if (!CopySrcMI)
        continue;

      int AGPROp = AMDGPU::getMFMASrcCVDstAGPROp(CopySrcMI->getOpcode());
      if (AGPROp == -1)
        continue;

      MachineOperand *Src2 =
          TII.getNamedOperand(*CopySrcMI, AMDGPU::OpName::src2);

      // FIXME: getMinimalPhysRegClass returns a nonsense AV_* subclass instead
      // of an AGPR or VGPR subclass, so we can't simply use the result on the
      // assignment.

      LLVM_DEBUG({
        Register Src2PhysReg = VRM.getPhys(Src2->getReg());
        dbgs() << "Attempting to replace VGPR MFMA with AGPR version:"
               << " Dst=[" << printReg(VReg) << " => "
               << printReg(PhysReg, &TRI) << "], Src2=["
               << printReg(Src2->getReg(), &TRI) << " => "
               << printReg(Src2PhysReg, &TRI) << "]: " << *CopySrcMI;
      });

      // If the inputs are tied and the same register, we can shortcut and
      // directly replace the register.
      if (!Src2->isReg() || Src2->getReg() != CopySrcReg ||
          Src2->getSubReg() != DefMI->getOperand(1).getSubReg()) {
        LLVM_DEBUG(
            dbgs()
            << "Replacing untied VGPR MFMAs with AGPR form not yet handled\n");
        // TODO: Only handles the tied case for now. If the input operand is a
        // different register, we need to also reassign it (either by looking
        // for a compatible copy-from-AGPR, or by seeing if an available AGPR is
        // compatible with all other uses.

        // If we can't reassign it, we'd need to introduce a different copy
        // which is likely worse than the copy we'd be saving.
        continue;
      }

      const TargetRegisterClass *Src2VirtRegRC =
          MRI.getRegClass(Src2->getReg());

      // We've found av = COPY (MFMA), and need to verify that we can trivially
      // rewrite src2 to use the new AGPR. If we can't trivially replace it,
      // we're going to induce as many copies as we would have emitted in the
      // first place, as well as need to assign another register, and need to
      // figure out where to put them. The live range splitting is smarter than
      // anything we're doing here, so trust it did something reasonable.
      const TargetRegisterClass *Src2ExceptRC =
          recomputeRegClassExceptRewritable(Src2->getReg(), Src2VirtRegRC,
                                            VirtRegRC);
      if (!Src2ExceptRC) {
        LLVM_DEBUG(dbgs() << "Could not recompute the regclass\n");
        continue;
      }

      const TargetRegisterClass *NewSrc2ConstraintRC =
          TII.getRegClass(TII.get(AGPROp), Src2->getOperandNo(), &TRI, MF);

      // Try to constrain src2 to the replacement instruction candidate's
      // register class.
      const TargetRegisterClass *NewSrc2RC =
          TRI.getCommonSubClass(Src2ExceptRC, NewSrc2ConstraintRC);
      if (!NewSrc2RC) {
        LLVM_DEBUG(dbgs() << "Other uses of " << printReg(Src2->getReg(), &TRI)
                          << " are incompatible with replacement class\n");
        continue;
      }

      MRI.setRegClass(VReg, AssignedRC);
      MRI.setRegClass(Src2->getReg(), NewSrc2RC);

      CopySrcMI->setDesc(TII.get(AGPROp));

      // Perform replacement of the register, rewriting the rewritable uses.
      for (MachineInstr &UseMI :
           make_early_inc_range(MRI.reg_instructions(CopySrcReg))) {
        if (TII.isMAI(UseMI)) {
          // Note the register we need to rewrite may still appear in src0/src1,
          // but that's fine since those can use A or V anyway.
          int ReplacementOp = AMDGPU::getMFMASrcCVDstAGPROp(UseMI.getOpcode());
          if (ReplacementOp != -1)
            UseMI.setDesc(TII.get(ReplacementOp));
        }

        UseMI.substituteRegister(CopySrcReg, VReg, AMDGPU::NoSubRegister, TRI);
      }

      LLVM_DEBUG(dbgs() << "Replaced VGPR MFMA with AGPR: " << *CopySrcMI);

      // We left behind an identity copy, so delete it.
      LIS.RemoveMachineInstrFromMaps(*DefMI);
      DefMI->eraseFromParent();

      LRM.unassign(CopySrcLI);

      // We don't need the liveness information anymore, so don't bother
      // updating the intervals. Just delete the stale information.
      // TODO: Is it worth preserving these?
      LIS.removeInterval(CopySrcReg);
      LIS.removeInterval(VReg);
      LIS.createAndComputeVirtRegInterval(VReg);

      MadeChange = true;
    }
  }

  return MadeChange;
}

class AMDGPURewriteAGPRCopyMFMALegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPURewriteAGPRCopyMFMALegacy() : MachineFunctionPass(ID) {
    initializeAMDGPURewriteAGPRCopyMFMALegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Rewrite AGPR-Copy-MFMA";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>();

    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<VirtRegMapWrapperLegacy>();
    AU.addPreserved<LiveRegMatrixWrapperLegacy>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPURewriteAGPRCopyMFMALegacy, DEBUG_TYPE,
                      "AMDGPU Rewrite AGPR-Copy-MFMA", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_END(AMDGPURewriteAGPRCopyMFMALegacy, DEBUG_TYPE,
                    "AMDGPU Rewrite AGPR-Copy-MFMA", false, false)

char AMDGPURewriteAGPRCopyMFMALegacy::ID = 0;

char &llvm::AMDGPURewriteAGPRCopyMFMALegacyID =
    AMDGPURewriteAGPRCopyMFMALegacy::ID;

bool AMDGPURewriteAGPRCopyMFMALegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  auto &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  auto &LRM = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();

  AMDGPURewriteAGPRCopyMFMAImpl Impl(MF, VRM, LRM, LIS);
  return Impl.run(MF);
}

PreservedAnalyses
AMDGPURewriteAGPRCopyMFMAPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  VirtRegMap &VRM = MFAM.getResult<VirtRegMapAnalysis>(MF);
  LiveRegMatrix &LRM = MFAM.getResult<LiveRegMatrixAnalysis>(MF);
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

  AMDGPURewriteAGPRCopyMFMAImpl Impl(MF, VRM, LRM, LIS);
  if (!Impl.run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
