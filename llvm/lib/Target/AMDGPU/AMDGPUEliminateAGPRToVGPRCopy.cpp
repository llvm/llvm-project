//===-- AMDGPUEliminateAGPRToVGPRCopy.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file \brief TODO
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-eliminate-agpr-to-vgpr-copy"

STATISTIC(NumEliminated, "Number of copies eliminated");

namespace {

class AMDGPUEliminateAGPRToVGPRCopyImpl {
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  MachineRegisterInfo &MRI;
  VirtRegMap &VRM;
  LiveRegMatrix &LRM;
  LiveIntervals &LIS;

public:
  AMDGPUEliminateAGPRToVGPRCopyImpl(MachineFunction &MF, VirtRegMap &VRM,
                                    LiveRegMatrix &LRM, LiveIntervals &LIS)
      : ST(MF.getSubtarget<GCNSubtarget>()), TII(*ST.getInstrInfo()),
        TRI(*ST.getRegisterInfo()), MRI(MF.getRegInfo()), VRM(VRM), LRM(LRM),
        LIS(LIS) {}

  bool areAllUsesCompatible(Register Reg) const;

  bool run(MachineFunction &MF) const;
};

bool AMDGPUEliminateAGPRToVGPRCopyImpl::areAllUsesCompatible(
    Register Reg) const {
  return all_of(MRI.use_operands(Reg), [&](const MachineOperand &MO) {
    const MachineInstr &ParentMI = *MO.getParent();
    if (!SIInstrInfo::isMFMA(ParentMI))
      return false;
    return &MO == TII.getNamedOperand(ParentMI, AMDGPU::OpName::src0) ||
           &MO == TII.getNamedOperand(ParentMI, AMDGPU::OpName::src1);
  });
}

bool AMDGPUEliminateAGPRToVGPRCopyImpl::run(MachineFunction &MF) const {
  // This only applies on subtargets that have a configurable AGPR vs. VGPR
  // allocation.
  if (!ST.hasGFX90AInsts())
    return false;

  // Early exit if no AGPRs were assigned.
  if (!LRM.isPhysRegUsed(AMDGPU::AGPR0))
    return false;

  bool MadeChange = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &CopyMI : make_early_inc_range(MBB)) {
      // Find full copies...
      if (!CopyMI.isFullCopy())
        continue;

      // ... whose destination was mapped to a VGPR or AGPR...
      Register DstReg = CopyMI.getOperand(0).getReg();
      if (!DstReg.isVirtual())
        continue;
      Register DstPhysReg = VRM.getPhys(DstReg);
      if (!DstPhysReg)
        continue;
      const TargetRegisterClass *DstRC = TRI.getPhysRegBaseClass(DstPhysReg);
      if (!TRI.hasVectorRegisters(DstRC) || TRI.hasSGPRs(DstRC))
        continue;

      // ... and whose source was mapped to an AGPR.
      Register SrcReg = CopyMI.getOperand(1).getReg();
      if (!SrcReg.isVirtual() || SrcReg == DstReg)
        continue;
      Register SrcPhysReg = VRM.getPhys(SrcReg);
      if (!SrcPhysReg)
        continue;
      const TargetRegisterClass *SrcRC = TRI.getPhysRegBaseClass(SrcPhysReg);
      if (!TRI.isAGPRClass(SrcRC))
        continue;

      bool DstIsAGPR = TRI.hasAGPRs(DstRC);

      LLVM_DEBUG({
        dbgs() << "AGPR->AVGPR copy: " << CopyMI;
        dbgs() << "                  "
               << printReg(DstReg, &TRI, CopyMI.getOperand(0).getSubReg(), &MRI)
               << " <-> " << printReg(DstPhysReg, &TRI, 0, &MRI) << "\n";
        dbgs() << "                  "
               << printReg(SrcReg, &TRI, CopyMI.getOperand(1).getSubReg(), &MRI)
               << " <-> " << printReg(SrcPhysReg, &TRI, 0, &MRI) << "\n";
      });

      LiveInterval &SrcLI = LIS.getInterval(SrcReg);
      const VNInfo *SrcVNI = SrcLI.getVNInfoAt(LIS.getInstructionIndex(CopyMI));
      assert(SrcVNI && "VNI must exist");

      bool AllUsesCompatible =
          all_of(MRI.use_operands(DstReg), [&](const MachineOperand &MO) {
            // Destination's use must be src0/src1 operands of an MFMA or
            // another copy.
            const MachineInstr &UseMI = *MO.getParent();
            if (!DstIsAGPR) {
              if (SIInstrInfo::isMFMA(UseMI)) {
                if (&MO != TII.getNamedOperand(UseMI, AMDGPU::OpName::src0) &&
                    &MO != TII.getNamedOperand(UseMI, AMDGPU::OpName::src1)) {
                  LLVM_DEBUG(dbgs()
                             << "  Incompatible MFMA operand: " << UseMI);
                  return false;
                }
              } else if (!UseMI.isFullCopy()) {
                LLVM_DEBUG(dbgs() << "  Incompatible user: " << UseMI);
                return false;
              }
            } else {
              LLVM_DEBUG(dbgs() << " Skipping user check (dst is AGPR)\n");
            }

            // Source must be available at use point.
            const VNInfo *UseVNI =
                SrcLI.getVNInfoAt(LIS.getInstructionIndex(UseMI));
            if (SrcVNI != UseVNI) {
              LLVM_DEBUG(dbgs() << "  AGPR no longer available at " << UseMI);
            }
            return true;
          });
      if (!AllUsesCompatible)
        continue;

      LLVM_DEBUG(dbgs() << "  -> Eliminated\n");
      ++NumEliminated;

      // Remove the copy's destination register.
      MRI.replaceRegWith(DstReg, SrcReg);
      LRM.unassign(LIS.getInterval(DstReg));
      LIS.removeInterval(DstReg);

      // Delete the copy instruction.
      LIS.RemoveMachineInstrFromMaps(CopyMI);
      CopyMI.eraseFromParent();

      // Recompute the source register's interval.
      // TODO: necessary? It is already live at all uses by construction.
      LIS.removeInterval(SrcReg);
      LIS.createAndComputeVirtRegInterval(SrcReg);
      MadeChange = true;
    }
  }

  return MadeChange;
}

class AMDGPUEliminateAGPRToVGPRCopyLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUEliminateAGPRToVGPRCopyLegacy() : MachineFunctionPass(ID) {
    initializeAMDGPUEliminateAGPRToVGPRCopyLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Eliminate AGPR-to-VGPR Copy";
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

INITIALIZE_PASS_BEGIN(AMDGPUEliminateAGPRToVGPRCopyLegacy, DEBUG_TYPE,
                      "AMDGPU Eliminate AGPR-to-VGPR Copy", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_END(AMDGPUEliminateAGPRToVGPRCopyLegacy, DEBUG_TYPE,
                    "AMDGPU Eliminate AGPR-to-VGPR Copy", false, false)

char AMDGPUEliminateAGPRToVGPRCopyLegacy::ID = 0;

char &llvm::AMDGPUEliminateAGPRToVGPRCopyLegacyID =
    AMDGPUEliminateAGPRToVGPRCopyLegacy::ID;

bool AMDGPUEliminateAGPRToVGPRCopyLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  auto &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  auto &LRM = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();

  AMDGPUEliminateAGPRToVGPRCopyImpl Impl(MF, VRM, LRM, LIS);
  return Impl.run(MF);
}

PreservedAnalyses
AMDGPUEliminateAGPRToVGPRCopyPass::run(MachineFunction &MF,
                                       MachineFunctionAnalysisManager &MFAM) {
  VirtRegMap &VRM = MFAM.getResult<VirtRegMapAnalysis>(MF);
  LiveRegMatrix &LRM = MFAM.getResult<LiveRegMatrixAnalysis>(MF);
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

  AMDGPUEliminateAGPRToVGPRCopyImpl Impl(MF, VRM, LRM, LIS);
  if (!Impl.run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
