//===----- AMDGPULoopAlign.cpp - Generate loop alignment directives  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Inspect a basic block and if certain conditions are met then align to 32
// bytes.
//===----------------------------------------------------------------------===//

#include "AMDGPULoopAlign.h"
#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
using namespace llvm;

#define DEBUG_TYPE "amdgpu-loop-align"

static cl::opt<bool>
    DisableLoopAlign("disable-amdgpu-loop-align", cl::init(false), cl::Hidden,
                     cl::desc("Disable AMDGPU loop alignment pass"));

namespace {

class AMDGPULoopAlign {
private:
  MachineLoopInfo &MLI;

public:
  AMDGPULoopAlign(MachineLoopInfo &MLI) : MLI(MLI) {}

  struct BasicBlockInfo {
    // Offset - Distance from the beginning of the function to the beginning
    // of this basic block.
    uint64_t Offset = 0;
    // Size - Size of the basic block in bytes
    uint64_t Size = 0;
  };

  void generateBlockInfo(MachineFunction &MF,
                         SmallVectorImpl<BasicBlockInfo> &BlockInfo) {
    BlockInfo.clear();
    BlockInfo.resize(MF.getNumBlockIDs());
    const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
    for (const MachineBasicBlock &MBB : MF) {
      BlockInfo[MBB.getNumber()].Size = 0;
      for (const MachineInstr &MI : MBB) {
        BlockInfo[MBB.getNumber()].Size += TII->getInstSizeInBytes(MI);
      }
    }
    uint64_t PrevNum = (&MF)->begin()->getNumber();
    for (auto &MBB :
         make_range(std::next(MachineFunction::iterator((&MF)->begin())),
                    (&MF)->end())) {
      uint64_t Num = MBB.getNumber();
      BlockInfo[Num].Offset =
          BlockInfo[PrevNum].Offset + BlockInfo[PrevNum].Size;
      unsigned blockAlignment = MBB.getAlignment().value();
      unsigned ParentAlign = MBB.getParent()->getAlignment().value();
      if (blockAlignment <= ParentAlign)
        BlockInfo[Num].Offset = alignTo(BlockInfo[Num].Offset, blockAlignment);
      else
        BlockInfo[Num].Offset = alignTo(BlockInfo[Num].Offset, blockAlignment) +
                                blockAlignment - ParentAlign;
      PrevNum = Num;
    }
  }

  bool run(MachineFunction &MF) {
    if (DisableLoopAlign)
      return false;

    // The starting address of all shader programs must be 256 bytes aligned.
    // Regular functions just need the basic required instruction alignment.
    const AMDGPUMachineFunction *MFI = MF.getInfo<AMDGPUMachineFunction>();
    MF.setAlignment(MFI->isEntryFunction() ? Align(256) : Align(4));
    if (MF.getAlignment().value() < 32)
      return false;

    const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
    SmallVector<BasicBlockInfo, 16> BlockInfo;
    generateBlockInfo(MF, BlockInfo);

    bool Changed = false;
    for (MachineLoop *ML : MLI.getLoopsInPreorder()) {
      // Check if loop is innermost
      if (!ML->isInnermost())
        continue;
      MachineBasicBlock *Header = ML->getHeader();
      // Check if loop is already evaluated for prefetch & aligned
      if (Header->getAlignment().value() == 64 ||
          ML->getTopBlock()->getAlignment().value() == 64)
        continue;

      // If loop is < 8-dwords, align aggressively to 0 mod 8 dword boundary.
      // else align to 0 mod 8 dword boundary only if less than 4 dwords of
      // instructions are available
      unsigned loopSizeInBytes = 0;
      for (MachineBasicBlock *MBB : ML->getBlocks())
        for (MachineInstr &MI : *MBB)
          loopSizeInBytes += TII->getInstSizeInBytes(MI);

      if (loopSizeInBytes < 32) {
        Header->setAlignment(llvm::Align(32));
        generateBlockInfo(MF, BlockInfo);
        Changed = true;
      } else if (BlockInfo[Header->getNumber()].Offset % 32 > 16) {
        Header->setAlignment(llvm::Align(32));
        generateBlockInfo(MF, BlockInfo);
        Changed = true;
      }
    }
    return Changed;
  }
};

class AMDGPULoopAlignLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPULoopAlignLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return AMDGPULoopAlign(getAnalysis<MachineLoopInfoWrapperPass>().getLI())
        .run(MF);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AMDGPULoopAlignLegacy::ID = 0;

char &llvm::AMDGPULoopAlignLegacyID = AMDGPULoopAlignLegacy::ID;

INITIALIZE_PASS(AMDGPULoopAlignLegacy, DEBUG_TYPE, "AMDGPU Loop Align", false,
                false)

PreservedAnalyses
AMDGPULoopAlignPass::run(MachineFunction &MF,
                         MachineFunctionAnalysisManager &MFAM) {
  auto &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  if (AMDGPULoopAlign(MLI).run(MF))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
