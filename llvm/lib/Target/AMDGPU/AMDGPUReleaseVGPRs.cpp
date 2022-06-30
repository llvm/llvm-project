//===- AMDGPUReleaseVGPRs.cpp - Automatically release vgprs on GFX11+ -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Insert S_SENDMSG instructions to release vgprs on GFX11+.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineOperand.h"
using namespace llvm;

#define DEBUG_TYPE "release-vgprs"

namespace {

class AMDGPUReleaseVGPRs : public MachineFunctionPass {
public:
  static char ID;

  const SIInstrInfo *SII;
  const SIRegisterInfo *TRI;

  AMDGPUReleaseVGPRs() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // Used to cache the result of isLastInstructionVMEMStore for each block
  using BlockVMEMStoreType = DenseMap<MachineBasicBlock *, bool>;
  BlockVMEMStoreType BlockVMEMStore;

  // Return true if the last instruction referencing a vgpr in this MBB
  // is a VMEM store, otherwise return false.
  // Visit previous basic blocks to find this last instruction if needed.
  // Because this pass is late in the pipeline, it is expected that the
  // last vgpr use will likely be one of vmem store, ds, exp.
  // Loads and others vgpr operations would have been
  // deleted by this point, except for complex control flow involving loops.
  // This is why we are just testing the type of instructions rather
  // than the operands.
  bool isLastVGPRUseVMEMStore(MachineBasicBlock &MBB) {
    // Use the cache to break infinite loop and save some time. Initialize to
    // false in case we have a cycle.
    BlockVMEMStoreType::iterator It;
    bool Inserted;
    std::tie(It, Inserted) = BlockVMEMStore.insert({&MBB, false});
    bool &CacheEntry = It->second;
    if (!Inserted)
      return CacheEntry;

    for (auto &MI : reverse(MBB.instrs())) {
      // If it's a VMEM store, a vgpr will be used, return true.
      if ((SIInstrInfo::isVMEM(MI) || SIInstrInfo::isFLAT(MI)) && MI.mayStore())
        return CacheEntry = true;

      // If it's referencing a VGPR but is not a VMEM store, return false.
      if (SIInstrInfo::isDS(MI) || SIInstrInfo::isEXP(MI) ||
          SIInstrInfo::isVMEM(MI) || SIInstrInfo::isFLAT(MI) ||
          SIInstrInfo::isVALU(MI))
        return CacheEntry = false;
    }

    // Recursive call into parent blocks. Look into predecessors if there is no
    // vgpr used in this block.
    return CacheEntry = llvm::any_of(MBB.predecessors(),
                                     [this](MachineBasicBlock *Parent) {
                                       return isLastVGPRUseVMEMStore(*Parent);
                                     });
  }

  bool runOnMachineBasicBlock(MachineBasicBlock &MBB) {

    bool Changed = false;

    for (auto &MI : MBB.terminators()) {
      // Look for S_ENDPGM instructions
      if (MI.getOpcode() == AMDGPU::S_ENDPGM ||
          MI.getOpcode() == AMDGPU::S_ENDPGM_SAVED) {
        // If the last instruction using a VGPR in the block is a VMEM store,
        // release VGPRs. The VGPRs release will be placed just before ending
        // the program
        if (isLastVGPRUseVMEMStore(MBB)) {
          BuildMI(MBB, MI, DebugLoc(), SII->get(AMDGPU::S_SENDMSG))
              .addImm(AMDGPU::SendMsg::ID_DEALLOC_VGPRS_GFX11Plus);
          Changed = true;
        }
      }
    }

    return Changed;
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    Function &F = MF.getFunction();
    if (skipFunction(F) || !AMDGPU::isEntryFunctionCC(F.getCallingConv()))
      return false;

    // This pass only runs on GFX11+
    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
    if (ST.getGeneration() < AMDGPUSubtarget::GFX11)
      return false;

    LLVM_DEBUG(dbgs() << "AMDGPUReleaseVGPRs running on " << MF.getName()
                      << "\n");

    SII = ST.getInstrInfo();
    TRI = ST.getRegisterInfo();

    bool Changed = false;
    for (auto &MBB : MF) {
      Changed |= runOnMachineBasicBlock(MBB);
    }

    BlockVMEMStore.clear();

    return Changed;
  }
};

} // namespace

char AMDGPUReleaseVGPRs::ID = 0;

char &llvm::AMDGPUReleaseVGPRsID = AMDGPUReleaseVGPRs::ID;

INITIALIZE_PASS(AMDGPUReleaseVGPRs, DEBUG_TYPE, "Release VGPRs", false, false)
