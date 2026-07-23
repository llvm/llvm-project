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
/// global_store_async_from_lds with an S_CBRANCH_EXECZ. Fully-masked waves
/// therefore skip the DMA entirely, which makes the ASYNCcnt observed at the
/// join point depend on whether the wave took the branch. Software-pipelined
/// kernels must then use conservative async waitcnts.
///
/// This pass sinks each async DMA out of such a then-block into the join
/// block, immediately before the EXEC-mask restore (the S_OR that ends the
/// control-flow region). EXEC at the sunk slot is the same masked value that
/// guarded the then-block, so per-lane behavior is unchanged, but every wave
/// now issues the DMA and ASYNCcnt becomes deterministic.
///
/// This runs right after SILowerControlFlow so the EXEC restore anchor exists,
/// and before waitcnt insertion so the improved counts can be used.
///
/// This is determinism-only: it never relaxes or rewrites a wait (an
/// s_wait_asynccnt 0 stays 0). Correctness never depends on the transform
/// firing; with the pass disabled the code is still correct, just conservative.
//
//===----------------------------------------------------------------------===//

#include "SISinkAsyncDMA.h"
#include "AMDGPU.h"
#include "AMDGPULaneMaskUtils.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "si-sink-async-dma"

namespace {

class SISinkAsyncDMA {
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  LiveVariables *LV = nullptr;
  const AMDGPU::LaneMaskConstants &LMC;

  bool sinkFromBlock(MachineBasicBlock &MBB);

public:
  SISinkAsyncDMA(const GCNSubtarget *ST, LiveVariables *LV)
      : TII(ST->getInstrInfo()), TRI(&TII->getRegisterInfo()), LV(LV),
        LMC(AMDGPU::LaneMaskConstants::get(*ST)) {}

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

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addUsedIfAvailable<LiveVariablesWrapperPass>();
    AU.addPreserved<LiveVariablesWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char SISinkAsyncDMALegacy::ID = 0;

INITIALIZE_PASS(SISinkAsyncDMALegacy, DEBUG_TYPE,
                "SI sink async DMA out of execz then-blocks", false, false)

char &llvm::SISinkAsyncDMALegacyID = SISinkAsyncDMALegacy::ID;

/// Return the S_OR EXEC restore at the top of \p JoinBB, if present.
static MachineInstr *findExecRestore(MachineBasicBlock &JoinBB,
                                     const AMDGPU::LaneMaskConstants &LMC) {
  auto I = JoinBB.getFirstNonDebugInstr();
  return I != JoinBB.end() && I->getOpcode() == LMC.OrOpc &&
                 I->getOperand(0).getReg() == LMC.ExecReg &&
                 I->getOperand(1).getReg() == LMC.ExecReg
             ? &*I
             : nullptr;
}

static bool isAsyncDMA(const MachineInstr &MI) {
  return SIInstrInfo::isLDSDMA(MI) && SIInstrInfo::usesASYNC_CNT(MI);
}

bool SISinkAsyncDMA::sinkFromBlock(MachineBasicBlock &MBB) {
  if (MBB.succ_size() != 2)
    return false;

  MachineBasicBlock *TBB = nullptr;
  MachineBasicBlock *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (TII->analyzeBranch(MBB, TBB, FBB, Cond) || !TBB || Cond.empty())
    return false;

  auto CondBr = find_if(MBB.terminators(), [](const MachineInstr &MI) {
    return MI.isConditionalBranch();
  });
  if (CondBr == MBB.terminators().end() ||
      CondBr->getOpcode() != AMDGPU::S_CBRANCH_EXECZ)
    return false;

  MachineBasicBlock *JoinBB = TBB;
  MachineBasicBlock *S0 = *MBB.succ_begin();
  MachineBasicBlock *S1 = *std::next(MBB.succ_begin());
  MachineBasicBlock *ThenBB = (S0 == JoinBB) ? S1 : S0;
  if (ThenBB == JoinBB || ThenBB->succ_size() != 1 ||
      *ThenBB->succ_begin() != JoinBB)
    return false;

  // A third incoming edge could reach the DMA under an unrelated EXEC.
  if (JoinBB->pred_size() != 2)
    return false;

  MachineInstr *ExecRestore = findExecRestore(*JoinBB, LMC);
  if (!ExecRestore)
    return false;

  SmallVector<MachineInstr *, 4> ToSink;
  for (MachineInstr &TMI : *ThenBB) {
    if (TMI.isMetaInstruction() || TMI.isTerminator())
      continue;
    if (TII->hasUnwantedEffectsWhenEXECEmpty(TMI))
      return false;
    if (isAsyncDMA(TMI)) {
      ToSink.push_back(&TMI);
      continue;
    }
    if (TMI.modifiesRegister(AMDGPU::EXEC, TRI) ||
        TMI.hasUnmodeledSideEffects())
      return false;

    if (ToSink.empty())
      continue;
    if (TMI.mayStore())
      return false;
    if (TMI.mayLoad() && !TMI.isDereferenceableInvariantLoad())
      return false;

    // Reject dependencies with earlier DMAs that would cross this instruction.
    for (const MachineOperand &MO : TMI.operands()) {
      if (!MO.isReg() || !MO.getReg() || (!MO.isDef() && !MO.readsReg()) ||
          TRI->regsOverlap(MO.getReg(), AMDGPU::EXEC))
        continue;
      if (any_of(ToSink, [&](const MachineInstr *DmaMI) {
            return DmaMI->readsRegister(MO.getReg(), TRI) ||
                   DmaMI->modifiesRegister(MO.getReg(), TRI);
          }))
        return false;
    }
  }
  if (ToSink.empty())
    return false;

  SmallSet<Register, 4> NeedsImpDef;
  SmallSet<Register, 8> LiveThroughThen;
  for (const MachineInstr *DmaMI : ToSink) {
    for (const MachineOperand &MO : DmaMI->uses()) {
      if (!MO.isReg() || !MO.readsReg() || !MO.getReg().isVirtual())
        continue;
      Register Reg = MO.getReg();
      MachineInstr *Def = MRI->getUniqueVRegDef(Reg);
      if (!Def)
        return false;
      if (Def->getParent() == ThenBB) {
        if (!isAsyncDMA(*Def))
          NeedsImpDef.insert(Reg);
      } else {
        LiveThroughThen.insert(Reg);
      }
    }
  }

  auto FirstTerm = MBB.getFirstTerminator();
  for (Register Reg : NeedsImpDef)
    BuildMI(MBB, FirstTerm, FirstTerm->getDebugLoc(),
            TII->get(TargetOpcode::IMPLICIT_DEF), Reg);

  for (MachineInstr *DmaMI : ToSink) {
    LLVM_DEBUG(dbgs() << "Sinking async DMA out of execz then-block: "
                      << *DmaMI);
    DmaMI->moveBefore(ExecRestore);
    for (const MachineOperand &MO : DmaMI->uses()) {
      if (!MO.isReg() || !MO.readsReg())
        continue;
      Register Reg = MO.getReg();
      if (Reg.isPhysical() && Reg != LMC.ExecReg)
        JoinBB->addLiveIn(Reg);
    }
  }

  if (LV)
    for (Register Reg : LiveThroughThen)
      LV->getVarInfo(Reg).AliveBlocks.set(ThenBB->getNumber());

  JoinBB->sortUniqueLiveIns();
  return true;
}

bool SISinkAsyncDMA::run(MachineFunction &MF) {
  MRI = &MF.getRegInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= sinkFromBlock(MBB);

  return Changed;
}

bool SISinkAsyncDMALegacy::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  auto *LVWrapper = getAnalysisIfAvailable<LiveVariablesWrapperPass>();
  LiveVariables *LV = LVWrapper ? &LVWrapper->getLV() : nullptr;
  return SISinkAsyncDMA(ST, LV).run(MF);
}

PreservedAnalyses
SISinkAsyncDMAPass::run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM) {
  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  LiveVariables *LV = MFAM.getCachedResult<LiveVariablesAnalysis>(MF);

  bool Changed = SISinkAsyncDMA(ST, LV).run(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<LiveVariablesAnalysis>();
  return PA;
}
