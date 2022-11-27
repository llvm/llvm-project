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
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineOperand.h"
#include <optional>
using namespace llvm;

#define DEBUG_TYPE "release-vgprs"

namespace {

class AMDGPUReleaseVGPRs : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUReleaseVGPRs() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // Track if the last instruction referencing a vgpr in a MBB is a VMEM
  // store. Because this pass is late in the pipeline, it is expected that the
  // last vgpr use will likely be one of vmem store, ds, exp.
  // Loads and others vgpr operations would have been
  // deleted by this point, except for complex control flow involving loops.
  // This is why we are just testing the type of instructions rather
  // than the operands.
  class LastVGPRUseIsVMEMStore {
    BitVector BlockVMEMStore;

    static std::optional<bool>
    lastVGPRUseIsStore(const MachineBasicBlock &MBB) {
      for (auto &MI : reverse(MBB.instrs())) {
        // If it's a VMEM store, a VGPR will be used, return true.
        if ((SIInstrInfo::isVMEM(MI) || SIInstrInfo::isFLAT(MI)) &&
            MI.mayStore())
          return true;

        // If it's referencing a VGPR but is not a VMEM store, return false.
        if (SIInstrInfo::isDS(MI) || SIInstrInfo::isEXP(MI) ||
            SIInstrInfo::isVMEM(MI) || SIInstrInfo::isFLAT(MI) ||
            SIInstrInfo::isVALU(MI))
          return false;
      }
      // Wait until the values are propagated from the predecessors
      return None;
    }

  public:
    LastVGPRUseIsVMEMStore(const MachineFunction &MF)
        : BlockVMEMStore(MF.getNumBlockIDs()) {

      df_iterator_default_set<const MachineBasicBlock *> Visited;
      SmallVector<const MachineBasicBlock *> EndWithVMEMStoreBlocks;

      for (const auto &MBB : MF) {
        auto LastUseIsStore = lastVGPRUseIsStore(MBB);
        if (!LastUseIsStore.has_value())
          continue;

        if (*LastUseIsStore) {
          EndWithVMEMStoreBlocks.push_back(&MBB);
        } else {
          Visited.insert(&MBB);
        }
      }

      for (const auto *MBB : EndWithVMEMStoreBlocks) {
        for (const auto *Succ : depth_first_ext(MBB, Visited)) {
          BlockVMEMStore[Succ->getNumber()] = true;
        }
      }
    }

    // Return true if the last instruction referencing a vgpr in this MBB
    // is a VMEM store, otherwise return false.
    bool isLastVGPRUseVMEMStore(const MachineBasicBlock &MBB) const {
      return BlockVMEMStore[MBB.getNumber()];
    }
  };

  static bool
  runOnMachineBasicBlock(MachineBasicBlock &MBB, const SIInstrInfo *SII,
                         const LastVGPRUseIsVMEMStore &BlockVMEMStore) {

    bool Changed = false;

    for (auto &MI : MBB.terminators()) {
      // Look for S_ENDPGM instructions
      if (MI.getOpcode() == AMDGPU::S_ENDPGM ||
          MI.getOpcode() == AMDGPU::S_ENDPGM_SAVED) {
        // If the last instruction using a VGPR in the block is a VMEM store,
        // release VGPRs. The VGPRs release will be placed just before ending
        // the program
        if (BlockVMEMStore.isLastVGPRUseVMEMStore(MBB)) {
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

    const SIInstrInfo *SII = ST.getInstrInfo();
    LastVGPRUseIsVMEMStore BlockVMEMStore(MF);

    bool Changed = false;
    for (auto &MBB : MF) {
      Changed |= runOnMachineBasicBlock(MBB, SII, BlockVMEMStore);
    }

    return Changed;
  }
};

} // namespace

char AMDGPUReleaseVGPRs::ID = 0;

char &llvm::AMDGPUReleaseVGPRsID = AMDGPUReleaseVGPRs::ID;

INITIALIZE_PASS(AMDGPUReleaseVGPRs, DEBUG_TYPE, "Release VGPRs", false, false)
