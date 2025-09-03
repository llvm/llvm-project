//===- SIPreAllocateWWMRegs.cpp - WWM Register Pre-allocation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Pass to pre-allocated WWM registers
//
//===----------------------------------------------------------------------===//

#include "SIPreAllocateWWMRegs.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-pre-allocate-wwm-regs"

static cl::opt<bool>
    EnablePreallocateSGPRSpillVGPRs("amdgpu-prealloc-sgpr-spill-vgprs",
                                    cl::init(false), cl::Hidden);

namespace {

class SIPreAllocateWWMRegs {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  LiveRegMatrix *Matrix;
  VirtRegMap *VRM;
  RegisterClassInfo RegClassInfo;

  std::vector<unsigned> RegsToRewrite;
#ifndef NDEBUG
  void printWWMInfo(const MachineInstr &MI);
#endif
  bool processDef(MachineOperand &MO);
  void rewriteRegs(MachineFunction &MF);

public:
  SIPreAllocateWWMRegs(LiveIntervals *LIS, LiveRegMatrix *Matrix,
                       VirtRegMap *VRM)
      : LIS(LIS), Matrix(Matrix), VRM(VRM) {}
  bool run(MachineFunction &MF);
};

class SIPreAllocateWWMRegsLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIPreAllocateWWMRegsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIPreAllocateWWMRegsLegacy, DEBUG_TYPE,
                      "SI Pre-allocate WWM Registers", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_END(SIPreAllocateWWMRegsLegacy, DEBUG_TYPE,
                    "SI Pre-allocate WWM Registers", false, false)

char SIPreAllocateWWMRegsLegacy::ID = 0;

char &llvm::SIPreAllocateWWMRegsLegacyID = SIPreAllocateWWMRegsLegacy::ID;

FunctionPass *llvm::createSIPreAllocateWWMRegsLegacyPass() {
  return new SIPreAllocateWWMRegsLegacy();
}

bool SIPreAllocateWWMRegs::processDef(MachineOperand &MO) {
  Register Reg = MO.getReg();
  if (Reg.isPhysical())
    return false;

  if (!TRI->isVGPR(*MRI, Reg))
    return false;

  if (VRM->hasPhys(Reg))
    return false;

  LiveInterval &LI = LIS->getInterval(Reg);

  for (MCRegister PhysReg : RegClassInfo.getOrder(MRI->getRegClass(Reg))) {
    if (!MRI->isPhysRegUsed(PhysReg, /*SkipRegMaskTest=*/true) &&
        Matrix->checkInterference(LI, PhysReg) == LiveRegMatrix::IK_Free) {
      Matrix->assign(LI, PhysReg);
      assert(PhysReg != 0);
      RegsToRewrite.push_back(Reg);
      return true;
    }
  }

  llvm_unreachable("physreg not found for WWM expression");
}

void SIPreAllocateWWMRegs::rewriteRegs(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;

        const Register VirtReg = MO.getReg();
        if (VirtReg.isPhysical())
          continue;

        if (!VirtReg.isValid())
          continue;

        if (!VRM->hasPhys(VirtReg))
          continue;

        Register PhysReg = VRM->getPhys(VirtReg);
        const unsigned SubReg = MO.getSubReg();
        if (SubReg != 0) {
          PhysReg = TRI->getSubReg(PhysReg, SubReg);
          MO.setSubReg(0);
        }

        MO.setReg(PhysReg);
        MO.setIsRenamable(false);
      }
    }
  }

  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  for (unsigned Reg : RegsToRewrite) {
    LIS->removeInterval(Reg);

    const Register PhysReg = VRM->getPhys(Reg);
    assert(PhysReg != 0);

    MFI->reserveWWMRegister(PhysReg);
  }

  RegsToRewrite.clear();

  // Update the set of reserved registers to include WWM ones.
  MRI->freezeReservedRegs();
}

#ifndef NDEBUG
LLVM_DUMP_METHOD void
SIPreAllocateWWMRegs::printWWMInfo(const MachineInstr &MI) {

  unsigned Opc = MI.getOpcode();

  if (Opc == AMDGPU::ENTER_STRICT_WWM || Opc == AMDGPU::ENTER_STRICT_WQM) {
    dbgs() << "Entering ";
  } else {
    assert(Opc == AMDGPU::EXIT_STRICT_WWM || Opc == AMDGPU::EXIT_STRICT_WQM);
    dbgs() << "Exiting ";
  }

  if (Opc == AMDGPU::ENTER_STRICT_WWM || Opc == AMDGPU::EXIT_STRICT_WWM) {
    dbgs() << "Strict WWM ";
  } else {
    assert(Opc == AMDGPU::ENTER_STRICT_WQM || Opc == AMDGPU::EXIT_STRICT_WQM);
    dbgs() << "Strict WQM ";
  }

  dbgs() << "region: " << MI;
}

#endif

bool SIPreAllocateWWMRegsLegacy::runOnMachineFunction(MachineFunction &MF) {
  auto *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  auto *Matrix = &getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  auto *VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  return SIPreAllocateWWMRegs(LIS, Matrix, VRM).run(MF);
}

bool SIPreAllocateWWMRegs::run(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "SIPreAllocateWWMRegs: function " << MF.getName() << "\n");

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();

  RegClassInfo.runOnMachineFunction(MF);

  bool PreallocateSGPRSpillVGPRs =
      EnablePreallocateSGPRSpillVGPRs ||
      MF.getFunction().hasFnAttribute("amdgpu-prealloc-sgpr-spill-vgprs");

  bool RegsAssigned = false;

  // We use a reverse post-order traversal of the control-flow graph to
  // guarantee that we visit definitions in dominance order. Since WWM
  // expressions are guaranteed to never involve phi nodes, and we can only
  // escape WWM through the special WWM instruction, this means that this is a
  // perfect elimination order, so we can never do any better.
  ReversePostOrderTraversal<MachineFunction*> RPOT(&MF);

  for (MachineBasicBlock *MBB : RPOT) {
    bool InWWM = false;
    for (MachineInstr &MI : *MBB) {
      if (MI.getOpcode() == AMDGPU::SI_SPILL_S32_TO_VGPR) {
        if (PreallocateSGPRSpillVGPRs)
          RegsAssigned |= processDef(MI.getOperand(0));
        continue;
      }

      if (MI.getOpcode() == AMDGPU::ENTER_STRICT_WWM ||
          MI.getOpcode() == AMDGPU::ENTER_STRICT_WQM) {
        LLVM_DEBUG(printWWMInfo(MI));
        InWWM = true;
        continue;
      }

      if (MI.getOpcode() == AMDGPU::EXIT_STRICT_WWM ||
          MI.getOpcode() == AMDGPU::EXIT_STRICT_WQM) {
        LLVM_DEBUG(printWWMInfo(MI));
        InWWM = false;
      }

      if (!InWWM)
        continue;

      LLVM_DEBUG(dbgs() << "Processing " << MI);

      for (MachineOperand &DefOpnd : MI.defs()) {
        RegsAssigned |= processDef(DefOpnd);
      }
    }
  }

  if (!RegsAssigned)
    return false;

  rewriteRegs(MF);
  return true;
}

PreservedAnalyses
SIPreAllocateWWMRegsPass::run(MachineFunction &MF,
                              MachineFunctionAnalysisManager &MFAM) {
  auto *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  auto *Matrix = &MFAM.getResult<LiveRegMatrixAnalysis>(MF);
  auto *VRM = &MFAM.getResult<VirtRegMapAnalysis>(MF);
  SIPreAllocateWWMRegs(LIS, Matrix, VRM).run(MF);
  return PreservedAnalyses::all();
}
