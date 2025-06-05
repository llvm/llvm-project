//===------- AMDGPUFinalizeISelWaveTransform.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass runs at the end of DAG-ISel in WaveTransformCF mode.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "SILowerI1Copies.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-finalize-isel-wave-transform"

namespace {

class Vreg1WideningHelper : public PhiLoweringHelper {
public:
  Vreg1WideningHelper(MachineFunction *MF);

private:
  DenseSet<Register> ConstrainRegs;

public:
  void markAsLaneMask(Register DstReg) const override {}
  void getCandidatesForLowering(
      SmallVectorImpl<MachineInstr *> &Vreg1Phis) const override {}
  void collectIncomingValuesFromPhi(
      const MachineInstr *MI,
      SmallVectorImpl<Incoming> &Incomings) const override {}
  void replaceDstReg(Register NewReg, Register OldReg,
                     MachineBasicBlock *MBB) override {}
  void buildMergeLaneMasks(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, const DebugLoc &DL,
                           Register DstReg, Register PrevReg,
                           Register CurReg) override {}
  void constrainAsLaneMask(Incoming &In) override {}

  bool widenVreg1s();

  bool cleanConstrainRegs(bool Changed) {
    assert(Changed || ConstrainRegs.empty());
    for (Register Reg : ConstrainRegs)
      MRI->constrainRegClass(Reg, IsWave32 ? &AMDGPU::SReg_32_XM0_XEXECRegClass
                                           : &AMDGPU::SReg_64_XEXECRegClass);
    ConstrainRegs.clear();
    return Changed;
  }
  bool isVreg1(Register Reg) const {
    return Reg.isVirtual() && MRI->getRegClass(Reg) == &AMDGPU::VReg_1RegClass;
  }
  bool isVreg32(Register Reg) const {
    return Reg.isVirtual() && MRI->getRegClass(Reg) == &AMDGPU::VGPR_32RegClass;
  }
};

Vreg1WideningHelper::Vreg1WideningHelper(MachineFunction *MF)
    : PhiLoweringHelper(MF, nullptr, nullptr) {}

// When WaveTransform happens later and CFG is not structurized,
// We need to apply a different algorithm for lowering vreg_1
// PhiNodes. Plus maybe some other lowering work needed?
class AMDGPUFinalizeISelWaveTransform : public MachineFunctionPass {
public:
  static char ID;

public:
  AMDGPUFinalizeISelWaveTransform() : MachineFunctionPass(ID) {
    initializeAMDGPUFinalizeISelWaveTransformPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Finalize ISel for Wave Transform";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUFinalizeISelWaveTransform, DEBUG_TYPE,
                      "AMDGPU Finalize ISel Wave Transform", false, false)
INITIALIZE_PASS_END(AMDGPUFinalizeISelWaveTransform, DEBUG_TYPE,
                    "AMDGPU Finalize ISel Wave Transform", false, false)

char AMDGPUFinalizeISelWaveTransform::ID = 0;
char &llvm::AMDGPUFinalizeISelWaveTransformID =
    AMDGPUFinalizeISelWaveTransform::ID;

FunctionPass *llvm::createAMDGPUFinalizeISelWaveTransformPass() {
  return new AMDGPUFinalizeISelWaveTransform();
}

//===----------------------------------------------------------------------===//
//
// This pass lowers all occurrences of i1 values (with a vreg_1 register class)
// to vreg_32 (32-bit vgpr per lane). The pass assumes machine SSA
// form and a per-thread control flow graph.
//
// Before this pass, values that are semantically i1 and are defined and used
// within the same basic block are already represented as lane masks in scalar
// registers. However, values that cross basic blocks are always transferred
// between basic blocks in vreg_1 virtual registers and are lowered by this
// pass.
//
// The only instructions that use or define vreg_1 virtual registers are COPY,
// PHI, and IMPLICIT_DEF.
//
//===----------------------------------------------------------------------===//
bool Vreg1WideningHelper::widenVreg1s() {
  bool Changed = false;
  SmallVector<MachineInstr *, 8> DeadCopies;
  SmallVector<MachineInstr *, 8> CopiesFromVreg1;
  DenseSet<Register> Vreg32Set;

  // Round#1, create the replacing instruction per Vreg1 definition.
  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      if (!MI.isPHI() && MI.getOpcode() != AMDGPU::COPY &&
          MI.getOpcode() != AMDGPU::IMPLICIT_DEF)
        continue;

      // Collect all the copies with a Vreg1 source.
      if (MI.getOpcode() == AMDGPU::COPY) {
        auto SrcReg = MI.getOperand(1).getReg();
        // If SrcReg has been renamed, it is in Vreg32Set.
        if (isVreg1(SrcReg) || Vreg32Set.count(SrcReg))
          CopiesFromVreg1.push_back(&MI);
      }

      Register DstReg = MI.getOperand(0).getReg();
      if (!isVreg1(DstReg))
        continue;

      Changed = true;
      LLVM_DEBUG(dbgs() << "create vreg32 def that replaces vreg1 def: " << MI);
      DebugLoc DL = MI.getDebugLoc();

      assert(!MI.getOperand(0).getSubReg());

      Register DefReg32b = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
      Vreg32Set.insert(DefReg32b);

      if (MI.getOpcode() == AMDGPU::IMPLICIT_DEF || MI.isPHI()) {
        // Simply replace the register with on existing instructions.
        MRI->replaceRegWith(DstReg, DefReg32b);
      } else if (MI.getOpcode() == AMDGPU::COPY) {
        Register SrcReg = MI.getOperand(1).getReg();
        assert(!MI.getOperand(1).getSubReg());
        if (isLaneMaskReg(SrcReg)) {
          ConstrainRegs.insert(SrcReg);
          MachineInstr *NewMI =
              BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_CNDMASK_B32_e64),
                      DefReg32b)
                  .addImm(0)
                  .addImm(0)
                  .addImm(0)
                  .addImm(-1)
                  .addReg(SrcReg);
          DeadCopies.push_back(&MI);
        } else {
          assert(isVreg1(SrcReg) || Vreg32Set.count(SrcReg));
        }

        MRI->replaceRegWith(DstReg, DefReg32b);
      }
    } // For MI.
  } // For MBB.

  // Round#2, replace copies from a VReg1.
  for (auto *MI : CopiesFromVreg1) {
    auto SrcReg = MI->getOperand(1).getReg();
    auto DstReg = MI->getOperand(0).getReg();
    // Should have been renamed.
    assert(isVreg32(SrcReg));
    DebugLoc DL = MI->getDebugLoc();
    if (isLaneMaskReg(DstReg)) {
      BuildMI(*MI->getParent(), MI, DL, TII->get(AMDGPU::V_CMP_NE_U32_e64),
              DstReg)
          .addReg(SrcReg)
          .addImm(0);
      DeadCopies.push_back(MI);
    } else
      assert(isVreg32(DstReg));
  }

  for (MachineInstr *MI : DeadCopies)
    MI->eraseFromParent();
  DeadCopies.clear();

  return Changed;
}

bool AMDGPUFinalizeISelWaveTransform::runOnMachineFunction(
    MachineFunction &MF) {
  // Only need to run this in SelectionDAG path.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::Selected))
    return false;

  Vreg1WideningHelper Helper(&MF);
  bool Changed = Helper.widenVreg1s();
  return Helper.cleanConstrainRegs(Changed);
}
