//===- AMDGPULowerIdxOps.cpp - Expand sub-dword VGPR-memory accesses -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Expand the sub-dword VGPR "as memory" (address space 13) pseudos into a
/// whole-dword indexed access plus a bit-field extract or insert:
///
///   V_LOAD_IDX_BITS  -> V_LOAD_IDX_B32 + V_BFE_{U,I}32
///   V_STORE_IDX_BITS -> V_LOAD_IDX_B32 + V_BFI_B32 + V_STORE_IDX_B32
///
/// A sub-dword store is therefore a read-modify-write of the containing dword.
/// This runs before AMDGPUAssignIdxToM0, so the whole-dword accesses created
/// here take part in the usual M0 setup, and before register allocation
/// because it introduces new virtual registers.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMachineInstrs.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-idx-ops"

namespace {

class LowerIdxOps {
public:
  LowerIdxOps(MachineFunction &MF)
      : TII(MF.getSubtarget<GCNSubtarget>().getInstrInfo()),
        MRI(&MF.getRegInfo()) {}

  bool run(MachineFunction &MF);

private:
  void lowerLoadIdxBits(MachineInstr &MI);
  void lowerStoreIdxBits(MachineInstr &MI);

  const SIInstrInfo *TII;
  MachineRegisterInfo *MRI;
};

void LowerIdxOps::lowerLoadIdxBits(MachineInstr &MI) {
  MachineBasicBlock *MBB = MI.getParent();
  auto &LoadIdx = cast<AMDGPUMI::VLoadIdxInst>(MI);

  const bool IsSigned = MI.getOperand(5).getImm() != 0;
  const MCInstrDesc &II =
      TII->get(IsSigned ? AMDGPU::V_BFE_I32_e64 : AMDGPU::V_BFE_U32_e64);

  Register SrcAReg = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);

  // Read the containing dword.
  auto LoadMIB =
      BuildMI(*MBB, MI, MI.getDebugLoc(),
              TII->get(AMDGPUMI::VLoadIdxInst::getOpcodeForBitWidth(32)),
              SrcAReg)
          .add(LoadIdx.getIdxOp())
          .add(LoadIdx.getOffsetOp());
  LoadMIB.addMemOperand(*MI.memoperands_begin());
  // Match what instruction selection does for a whole-dword access with a
  // register index: record that the M0 write implied by the eventual movrel
  // clobbers M0 (see AMDGPUAssignIdxToM0).
  if (LoadIdx.getIdxOp().isReg())
    LoadMIB.addReg(AMDGPU::M0, RegState::ImplicitDefine);

  // Extract the accessed bits out of it.
  MachineOperand BitOffset = MI.getOperand(4);
  if (BitOffset.isImm())
    BitOffset.setImm(BitOffset.getImm() & 31);

  BuildMI(*MBB, MI, MI.getDebugLoc(), II, LoadIdx.getDataOp().getReg())
      .addReg(SrcAReg)
      .add(BitOffset)
      .add(MI.getOperand(3)); // bitsize

  LLVM_DEBUG(dbgs() << " *** Expanded pseudo: "; MI.print(dbgs()));

  MI.eraseFromParent();
}

void LowerIdxOps::lowerStoreIdxBits(MachineInstr &MI) {
  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction *MF = MBB->getParent();
  auto &StoreIdx = cast<AMDGPUMI::VStoreIdxInst>(MI);

  Register SrcAReg = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register DstAReg = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);

  // Read the containing dword.
  auto LoadMIB = BuildMI(*MBB, MI, MI.getDebugLoc(),
                         TII->get(AMDGPU::V_LOAD_IDX_B32), SrcAReg)
                     .add(StoreIdx.getIdxOp())
                     .add(StoreIdx.getOffsetOp());
  // The index is read again by the store below, so it does not die here.
  LoadMIB->getOperand(1).setIsKill(false);
  auto *StoreMMO = *MI.memoperands_begin();
  // Synthesize a load MMO from the store's.
  auto NewFlags = MachineMemOperand::MOLoad;
  NewFlags |= StoreMMO->getFlags() & ~MachineMemOperand::MOStore;
  LoadMIB.addMemOperand(MF->getMachineMemOperand(StoreMMO, NewFlags));
  if (StoreIdx.getIdxOp().isReg())
    LoadMIB.addReg(AMDGPU::M0, RegState::ImplicitDefine);

  // Insert the stored bits into it.
  BuildMI(*MBB, MI, MI.getDebugLoc(), TII->get(AMDGPU::V_BFI_B32_e64), DstAReg)
      .add(StoreIdx.getOperand(3)) // mask
      .addReg(StoreIdx.getDataOp().getReg())
      .addReg(SrcAReg);

  // Write the dword back.
  auto StoreMIB =
      BuildMI(*MBB, MI, MI.getDebugLoc(), TII->get(AMDGPU::V_STORE_IDX_B32))
          .addReg(DstAReg)
          .add(StoreIdx.getIdxOp())
          .add(StoreIdx.getOffsetOp());
  StoreMIB.addMemOperand(StoreMMO);
  if (StoreIdx.getIdxOp().isReg())
    StoreMIB.addReg(AMDGPU::M0, RegState::ImplicitDefine);

  LLVM_DEBUG(dbgs() << " *** Expanded pseudo: "; MI.print(dbgs()));

  MI.eraseFromParent();
}

bool LowerIdxOps::run(MachineFunction &MF) {
  bool Changed = false;

  LLVM_DEBUG(dbgs() << "\nLowerIdxOps on function: " << MF.getName() << "\n");

  for (MachineBasicBlock &MBB : MF) {
    for (auto MII = MBB.begin(), E = MBB.end(); MII != E;) {
      MachineInstr &MI = *MII++;
      switch (MI.getOpcode()) {
      case AMDGPU::V_LOAD_IDX_BITS:
        lowerLoadIdxBits(MI);
        Changed = true;
        break;
      case AMDGPU::V_STORE_IDX_BITS:
        lowerStoreIdxBits(MI);
        Changed = true;
        break;
      default:
        break;
      }
    }
  }

  return Changed;
}

class AMDGPULowerIdxOpsLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPULowerIdxOpsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    // This is required lowering, not an optimization: nothing else expands the
    // sub-dword pseudos, and AMDGPULowerVGPREncoding cannot lower them. It
    // therefore must not be skipped for optnone functions.
    return LowerIdxOps(MF).run(MF);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "AMDGPU Lower Idx Ops"; }
};

} // end anonymous namespace

PreservedAnalyses
AMDGPULowerIdxOpsPass::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &MFAM) {
  if (!LowerIdxOps(MF).run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

char AMDGPULowerIdxOpsLegacy::ID = 0;

char &llvm::AMDGPULowerIdxOpsID = AMDGPULowerIdxOpsLegacy::ID;

INITIALIZE_PASS(AMDGPULowerIdxOpsLegacy, DEBUG_TYPE, "AMDGPU Lower Idx Ops",
                false, false)
