//===-- SIMergeVGPRCopies.cpp - Merge adjacent VGPR copies ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Merge adjacent 32-bit copy-to-VGPR pairs into a single 64-bit copy
/// (V_PK_MOV_B32 or V_MOV_B64).
///
/// Both sources and destinations must be consecutive and even-aligned, but the
/// source and destination pairs can be in either order. The sources can be
/// SGPRs if the target supports V_MOV_B64, but not if the merged copy needs to
/// use V_PK_MOV_B32 (e.g. if the source pairs are inversed relative to the
/// destinations).
///
/// This runs after register allocation, when all registers are physical and
/// before ExpandPostRAPseudos lowers each copy individually.
///
//===----------------------------------------------------------------------===//

#include "SIMergeVGPRCopies.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "si-merge-vgpr-copies"

namespace {

class SIMergeVGPRCopiesLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIMergeVGPRCopiesLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI Merge VGPR copies"; }
};

class SIMergeVGPRCopies {
public:
  bool run(MachineFunction &MF);
};

} // End anonymous namespace.

INITIALIZE_PASS(SIMergeVGPRCopiesLegacy, DEBUG_TYPE, "SI Merge VGPR copies",
                false, false)

char SIMergeVGPRCopiesLegacy::ID = 0;

char &llvm::SIMergeVGPRCopiesID = SIMergeVGPRCopiesLegacy::ID;

PreservedAnalyses SIMergeVGPRCopiesPass::run(MachineFunction &MF,
                                             MachineFunctionAnalysisManager &) {
  SIMergeVGPRCopies().run(MF);
  return PreservedAnalyses::all();
}

bool SIMergeVGPRCopiesLegacy::runOnMachineFunction(MachineFunction &MF) {
  return SIMergeVGPRCopies().run(MF);
}

static bool getBaseIfAlignedAndConsecutive(const Register &Reg1,
                                           const Register &Reg2,
                                           const SIRegisterInfo *TRI,
                                           Register &Base) {
  long Dist =
      static_cast<long>(TRI->getHWRegIndex(Reg2)) - TRI->getHWRegIndex(Reg1);
  if (std::abs(Dist) != 1)
    return false;
  Base = (Dist == 1) ? Reg1 : Reg2;
  return TRI->getHWRegIndex(Base) % 2 == 0;
}

bool SIMergeVGPRCopies::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  // Only merge when the target has a 64-bit move instruction. Skip targets
  // with VOPD: GCNCreateVOPD runs later and can pair 32-bit moves with
  // unrelated instructions, which is potentially better than v_mov_b64.
  if ((!ST.hasPkMovB32() && !ST.hasMovB64()) || AMDGPU::hasVOPD(ST))
    return false;

  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const TargetRegisterClass *VGPR64RC = TRI->getVGPR64Class();
  const TargetRegisterClass *SGPR64RC = &AMDGPU::SReg_64RegClass;
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &MI = *I;

      I = next_nodbg(I, E);
      if (I == E || I->getOpcode() != AMDGPU::COPY ||
          MI.getOpcode() != AMDGPU::COPY)
        continue;
      MachineInstr &NextMI = *I;

      {
        // Both destinations must be VGPRs.
        Register MIDstReg = MI.getOperand(0).getReg();
        Register NextMIDstReg = NextMI.getOperand(0).getReg();
        if (!AMDGPU::VGPR_32RegClass.contains(MIDstReg) ||
            !AMDGPU::VGPR_32RegClass.contains(NextMIDstReg))
          continue;
      }

      bool UsesSGPRSources = false;
      {
        // Sources may always be VGPRs, but can also be SGPRs if the target
        // supports V_MOV_B64. Either way, the sources must be the same class.
        Register MISrcReg = MI.getOperand(1).getReg();
        Register NextMISrcReg = NextMI.getOperand(1).getReg();
        UsesSGPRSources = ST.hasMovB64() &&
                          AMDGPU::SGPR_32RegClass.contains(MISrcReg) &&
                          AMDGPU::SGPR_32RegClass.contains(NextMISrcReg);
        if (!UsesSGPRSources &&
            !(AMDGPU::VGPR_32RegClass.contains(MISrcReg) &&
              AMDGPU::VGPR_32RegClass.contains(NextMISrcReg)))
          continue;
      }

      // Ensure the first copy is the one with the lower-numbered destination
      // register. This simplifies checking for consecutive sources and ensures
      // the merged copy reads the same registers as the original pair.
      Register DstBase;
      if (!getBaseIfAlignedAndConsecutive(MI.getOperand(0).getReg(),
                                          NextMI.getOperand(0).getReg(), TRI,
                                          DstBase))
        continue;

      Register SrcBase;
      if (!getBaseIfAlignedAndConsecutive(MI.getOperand(1).getReg(),
                                          NextMI.getOperand(1).getReg(), TRI,
                                          SrcBase))
        continue;

      // The sources must also be consecutive and even-aligned, but they can be
      // in either order. If they are inversed, the merged copy needs to use
      // V_PK_MOV_B32 which in turn cannot use SGPR sources, otherwise it can
      // use V_MOV_B64.
      bool SrcIsInversed = (DstBase == NextMI.getOperand(0).getReg()) !=
                           (SrcBase == NextMI.getOperand(1).getReg());
      if (SrcIsInversed && (UsesSGPRSources || !ST.hasPkMovB32()))
        continue;

      Register Dst64 =
          TRI->getMatchingSuperReg(DstBase, AMDGPU::sub0, VGPR64RC);
      Register Src64 = TRI->getMatchingSuperReg(
          SrcBase, AMDGPU::sub0, UsesSGPRSources ? SGPR64RC : VGPR64RC);
      if (!Dst64 || !Src64)
        continue;

      bool KillSrc = MI.getOperand(1).isKill() && NextMI.getOperand(1).isKill();
      if (SrcIsInversed) {
        BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(AMDGPU::V_PK_MOV_B32),
                Dst64)
            .addImm(SISrcMods::OP_SEL_0 | SISrcMods::OP_SEL_1)
            .addReg(Src64)
            .addImm(SISrcMods::OP_SEL_1)
            .addReg(Src64)
            .addImm(0) // op_sel_lo
            .addImm(0) // op_sel_hi
            .addImm(0) // neg_lo
            .addImm(0) // neg_hi
            .addImm(0) // clamp
            .addReg(Src64, getKillRegState(KillSrc) | RegState::Implicit);
      } else {
        BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(AMDGPU::COPY), Dst64)
            .addReg(Src64, getKillRegState(KillSrc))
            .addReg(AMDGPU::EXEC, RegState::Implicit);
      }

      LLVM_DEBUG(dbgs() << "Merged VGPR32 copy pair into 64-bit copy from "
                        << MI << " and " << NextMI << '\n');
      I = next_nodbg(I, E); // Advance past NextMI before erasing it.
      MI.eraseFromParent();
      NextMI.eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}
