//===-- SISinkAsyncDMA.cpp - Sink async DMA out of execz then-blocks ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// LLVM lowers a divergent branch around global_load_async_to_lds /
/// global_store_async_from_lds with an S_CBRANCH_EXECZ, so fully-masked waves
/// skip the DMA entirely and the ASYNCcnt observed at the join depends on
/// whether the wave took the branch. Software-pipelined kernels then have to
/// use a conservative async waitcnt.
///
/// This pass sinks each such DMA into the join, immediately before SI_ELSE or
/// SI_END_CF:
///
///        MBB                     MBB        SI_IF sets EXEC to the then-block
///       /   \                   /   \       mask before the branch, so both
///   ThenBB   |               ThenBB  |      edges carry it and per-lane
///    [DMA]   |      ==>          \  /       behavior is unchanged. But every
///       \   /                   JoinBB      wave now issues the DMA, so
///      JoinBB                    [DMA]      ASYNCcnt at the join no longer
///    [SI_END_CF]                   |        depends on the branch.
///                             [SI_END_CF]
///
/// The join is split so that SI_END_CF starts a block of its own, because
/// SILowerControlFlow emits the EXEC restore at the top of the block holding
/// it, which would otherwise place it above the sunk DMAs.

//
//===----------------------------------------------------------------------===//

#include "SISinkAsyncDMA.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"

using namespace llvm;

#define DEBUG_TYPE "si-sink-async-dma"

namespace {

class SISinkAsyncDMA {
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  bool sinkFromBlock(MachineBasicBlock &MBB);

public:
  bool run(MachineFunction &MF);
};

class SISinkAsyncDMALegacy : public MachineFunctionPass {
public:
  static char ID;

  SISinkAsyncDMALegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI sink async DMA out of execz then-blocks";
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().setNoPHIs();
  }
};

} // namespace

char SISinkAsyncDMALegacy::ID = 0;

INITIALIZE_PASS(SISinkAsyncDMALegacy, DEBUG_TYPE,
                "SI sink async DMA out of execz then-blocks", false, false)

char &llvm::SISinkAsyncDMALegacyID = SISinkAsyncDMALegacy::ID;

static bool isAsyncDMA(const MachineInstr &MI) {
  return SIInstrInfo::isLDSDMA(MI) && SIInstrInfo::usesASYNC_CNT(MI);
}

static bool isAsyncMarker(const MachineInstr &MI) {
  return MI.getOpcode() == AMDGPU::ASYNCMARK ||
         MI.getOpcode() == AMDGPU::WAIT_ASYNCMARK;
}

bool SISinkAsyncDMA::sinkFromBlock(MachineBasicBlock &MBB) {
  if (MBB.succ_size() != 2)
    return false;

  // A region head ends in SI_IF or SI_ELSE ($dst, $cond, $target), which define
  // the mask the region restores in $dst and the join block in $target.
  auto ControlMI = MBB.getFirstTerminator();
  if (ControlMI == MBB.end() || (ControlMI->getOpcode() != AMDGPU::SI_IF &&
                                 ControlMI->getOpcode() != AMDGPU::SI_ELSE))
    return false;

  Register SavedExec = ControlMI->getOperand(0).getReg();
  MachineBasicBlock *JoinBB = ControlMI->getOperand(2).getMBB();

  if (!MBB.isSuccessor(JoinBB) || JoinBB->pred_size() != 2)
    return false;

  auto ThenIt = find_if(MBB.successors(),
                        [JoinBB](MachineBasicBlock *S) { return S != JoinBB; });
  if (ThenIt == MBB.succ_end() || (*ThenIt)->getSingleSuccessor() != JoinBB)
    return false;
  MachineBasicBlock *ThenBB = *ThenIt;

  auto Boundary = JoinBB->getFirstNonPHI();
  while (Boundary != JoinBB->end() && Boundary->isMetaInstruction())
    ++Boundary;
  if (Boundary == JoinBB->end())
    return false;

  // The boundary must consume the mask this region saved ($saved for SI_END_CF,
  // $src for a chained SI_ELSE), otherwise it closes a different region.
  bool IsEndCF = Boundary->getOpcode() == AMDGPU::SI_END_CF &&
                 Boundary->getOperand(0).getReg() == SavedExec;
  bool IsElse = ControlMI->getOpcode() == AMDGPU::SI_IF &&
                Boundary->getOpcode() == AMDGPU::SI_ELSE &&
                Boundary->getOperand(1).getReg() == SavedExec;
  if (!IsEndCF && !IsElse)
    return false;

  // Scan bottom-up so everything a DMA moves across is already accumulated when
  // the DMA is reached. Instructions above the topmost DMA are never crossed,
  // so an unsafe one only matters once a DMA turns up above it.
  SmallVector<MachineInstr *, 4> ToSink;
  bool CrossedUnsafe = false;
  bool CrossedM0Write = false;
  bool DontMoveAcrossStore = true;

  for (MachineInstr &MI : reverse(*ThenBB)) {
    if (isAsyncMarker(MI))
      return false;
    if (MI.isMetaInstruction() || MI.isUnconditionalBranch())
      continue;

    if (isAsyncDMA(MI)) {
      // A cluster load takes its mask from M0.
      if (CrossedUnsafe ||
          (CrossedM0Write && MI.readsRegister(AMDGPU::M0, TRI)))
        return false;
      ToSink.push_back(&MI);
      continue;
    }

    CrossedUnsafe |= TII->hasUnwantedEffectsWhenEXECEmpty(MI) ||
                     MI.modifiesRegister(AMDGPU::EXEC, TRI) ||
                     !MI.isSafeToMove(DontMoveAcrossStore);
    CrossedM0Write |= MI.modifiesRegister(AMDGPU::M0, TRI);
  }
  if (ToSink.empty())
    return false;

  MachineSSAUpdater Updater(*MBB.getParent());
  SmallDenseMap<Register, Register, 4> MergedRegs;
  for (MachineInstr *DmaMI : reverse(ToSink)) {
    for (MachineOperand &MO : DmaMI->uses()) {
      if (!MO.isReg() || !MO.readsReg() || !MO.getReg().isVirtual())
        continue;
      Register Reg = MO.getReg();
      if (MRI->getDefBlock(Reg) != ThenBB)
        continue;
      Register &Merged = MergedRegs[Reg];
      if (!Merged) {
        Updater.Initialize(Reg);
        Updater.AddAvailableValue(ThenBB, Reg);
        Merged = Updater.GetValueInMiddleOfBlock(JoinBB);
      }
      MO.setReg(Merged);
    }

    LLVM_DEBUG(dbgs() << "Sinking async DMA out of execz then-block: "
                      << *DmaMI);
    DmaMI->moveBefore(&*Boundary);
  }

  JoinBB->splitAt(*ToSink.front(), /*UpdateLiveIns=*/true);

  return true;
}

bool SISinkAsyncDMA::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasAsynccnt())
    return false;

  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= sinkFromBlock(MBB);

  return Changed;
}

bool SISinkAsyncDMALegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  return SISinkAsyncDMA().run(MF);
}

PreservedAnalyses SISinkAsyncDMAPass::run(MachineFunction &MF,
                                          MachineFunctionAnalysisManager &) {
  MFPropsModifier _(*this, MF);

  return SISinkAsyncDMA().run(MF) ? getMachineFunctionPassPreservedAnalyses()
                                  : PreservedAnalyses::all();
}
