//===- SIInsertWaitcnts.cpp - Insert Wait Instructions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Insert wait instructions for memory reads and writes.
///
/// Memory reads and writes are issued asynchronously, so we need to insert
/// S_WAITCNT instructions when we want to access any of their results or
/// overwrite any register that's used asynchronously.
///
/// TODO: This pass currently keeps one timeline per hardware counter. A more
/// finely-grained approach that keeps one timeline per event type could
/// sometimes get away with generating weaker s_waitcnt instructions. For
/// example, when both SMEM and LDS are in flight and we need to wait for
/// the i-th-last LDS instruction, then an lgkmcnt(i) is actually sufficient,
/// but the pass will currently generate a conservative lgkmcnt(0) because
/// multiple event types are in flight.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDGPUWaitCountUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/TargetParser/TargetParser.h"
using namespace llvm;
using namespace llvm::AMDGPU;

#define DEBUG_TYPE "si-insert-waitcnts"

DEBUG_COUNTER(ForceExpCounter, DEBUG_TYPE"-forceexp",
              "Force emit s_waitcnt expcnt(0) instrs");
DEBUG_COUNTER(ForceLgkmCounter, DEBUG_TYPE"-forcelgkm",
              "Force emit s_waitcnt lgkmcnt(0) instrs");
DEBUG_COUNTER(ForceVMCounter, DEBUG_TYPE"-forcevm",
              "Force emit s_waitcnt vmcnt(0) instrs");

static cl::opt<bool> ForceEmitZeroFlag(
  "amdgpu-waitcnt-forcezero",
  cl::desc("Force all waitcnt instrs to be emitted as s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"),
  cl::init(false), cl::Hidden);

//===----------------------------------------------------------------------===//
// SIWaitCntsInserter helper class interface.
//===----------------------------------------------------------------------===//

class SIWaitCntsInserter : public AMDGPUWaitCntInserter {
public:
  SIWaitCntsInserter() {}
  SIWaitCntsInserter(const GCNSubtarget *ST, const MachineRegisterInfo *MRI,
                     WaitCntGenerator *WCG, InstCounterType MC, bool FEZWC,
                     MachineLoopInfo *MLI, MachinePostDominatorTree *PDT,
                     AliasAnalysis *AA)
      : AMDGPUWaitCntInserter(ST, MRI, WCG, MC), MLI(MLI), PDT(PDT), AA(AA),
        ForceEmitZeroWaitcnts(FEZWC) {
    for (auto T : inst_counter_types())
      ForceEmitWaitcnt[T] = false;
  }
  bool generateWaitcntInstBefore(MachineInstr &MI,
                                 WaitcntBrackets &ScoreBrackets,
                                 MachineInstr *OldWaitcntInstr, bool FlushVmCnt,
                                 VGPRInstsSet *VGPRInsts) override;
  bool insertWaitcntInBlock(MachineFunction &MF, MachineBasicBlock &Block,
                            WaitcntBrackets &ScoreBrackets,
                            VGPRInstsSet *VGPRInsts = nullptr) override;
  void updateEventWaitcntAfter(MachineInstr &Inst,
                               WaitcntBrackets *ScoreBrackets) override;

private:
  MachineLoopInfo *MLI;
  MachinePostDominatorTree *PDT;
  AliasAnalysis *AA = nullptr;

  bool mayAccessLDSThroughFlat(const MachineInstr &MI) const;
  bool isPreheaderToFlush(MachineBasicBlock &MBB,
                          WaitcntBrackets &ScoreBrackets);
  bool shouldFlushVmCnt(MachineLoop *ML, WaitcntBrackets &Brackets) const;
  WaitEventType getVmemWaitEventType(const MachineInstr &Inst) const;
  void setForceEmitWaitcnt();

  DenseMap<const Value *, MachineBasicBlock *> SLoadAddresses;
  DenseMap<MachineBasicBlock *, bool> PreheadersToFlush;

  // ForceEmitZeroWaitcnts: force all waitcnts insts to be s_waitcnt 0
  // because of amdgpu-waitcnt-forcezero flag
  bool ForceEmitZeroWaitcnts;
  bool ForceEmitWaitcnt[NUM_INST_CNTS];
};

// This is a flat memory operation. Check to see if it has memory tokens for
// either LDS or FLAT.
bool SIWaitCntsInserter::mayAccessLDSThroughFlat(const MachineInstr &MI) const {
  assert(TII->isFLAT(MI));

  // Flat instruction such as SCRATCH and GLOBAL do not use the lgkm counter.
  if (!TII->usesLGKM_CNT(MI))
    return false;

  // If in tgsplit mode then there can be no use of LDS.
  if (ST->isTgSplitEnabled())
    return false;

  // If there are no memory operands then conservatively assume the flat
  // operation may access LDS.
  if (MI.memoperands_empty())
    return true;

  // See if any memory operand specifies an address space that involves LDS.
  for (const MachineMemOperand *Memop : MI.memoperands()) {
    unsigned AS = Memop->getAddrSpace();
    if (AS == AMDGPUAS::LOCAL_ADDRESS || AS == AMDGPUAS::FLAT_ADDRESS)
      return true;
  }

  return false;
}

// Return true if the given machine basic block is a preheader of a loop in
// which we want to flush the vmcnt counter, and false otherwise.
bool SIWaitCntsInserter::isPreheaderToFlush(MachineBasicBlock &MBB,
                                            WaitcntBrackets &ScoreBrackets) {
  auto [Iterator, IsInserted] = PreheadersToFlush.try_emplace(&MBB, false);
  if (!IsInserted)
    return Iterator->second;

  MachineBasicBlock *Succ = MBB.getSingleSuccessor();
  if (!Succ)
    return false;

  MachineLoop *Loop = MLI->getLoopFor(Succ);
  if (!Loop)
    return false;

  if (Loop->getLoopPreheader() == &MBB &&
      shouldFlushVmCnt(Loop, ScoreBrackets)) {
    Iterator->second = true;
    return true;
  }

  return false;
}

// Return true if it is better to flush the vmcnt counter in the preheader of
// the given loop. We currently decide to flush in two situations:
// 1. The loop contains vmem store(s), no vmem load and at least one use of a
//    vgpr containing a value that is loaded outside of the loop. (Only on
//    targets with no vscnt counter).
// 2. The loop contains vmem load(s), but the loaded values are not used in the
//    loop, and at least one use of a vgpr containing a value that is loaded
//    outside of the loop.
bool SIWaitCntsInserter::shouldFlushVmCnt(MachineLoop *ML,
                                          WaitcntBrackets &Brackets) const {
  bool HasVMemLoad = false;
  bool HasVMemStore = false;
  bool UsesVgprLoadedOutside = false;
  DenseSet<Register> VgprUse;
  DenseSet<Register> VgprDef;

  for (MachineBasicBlock *MBB : ML->blocks()) {
    for (MachineInstr &MI : *MBB) {
      if (isVMEMOrFlatVMEM(MI)) {
        if (MI.mayLoad())
          HasVMemLoad = true;
        if (MI.mayStore())
          HasVMemStore = true;
      }
      for (unsigned I = 0; I < MI.getNumOperands(); I++) {
        MachineOperand &Op = MI.getOperand(I);
        if (!Op.isReg() || !TRI->isVectorRegister(*MRI, Op.getReg()))
          continue;
        auto [RegLow, RegHigh] = Brackets.getRegInterval(&MI, MRI, TRI, I);
        // Vgpr use
        if (Op.isUse()) {
          for (int RegNo = RegLow; RegNo < RegHigh; ++RegNo) {
            // If we find a register that is loaded inside the loop, 1. and 2.
            // are invalidated and we can exit.
            if (VgprDef.contains(RegNo))
              return false;
            VgprUse.insert(RegNo);
            // If at least one of Op's registers is in the score brackets, the
            // value is likely loaded outside of the loop.
            if (Brackets.getRegScore(RegNo, LOAD_CNT) >
                    Brackets.getScoreLB(LOAD_CNT) ||
                Brackets.getRegScore(RegNo, SAMPLE_CNT) >
                    Brackets.getScoreLB(SAMPLE_CNT) ||
                Brackets.getRegScore(RegNo, BVH_CNT) >
                    Brackets.getScoreLB(BVH_CNT)) {
              UsesVgprLoadedOutside = true;
              break;
            }
          }
        }
        // VMem load vgpr def
        else if (isVMEMOrFlatVMEM(MI) && MI.mayLoad() && Op.isDef())
          for (int RegNo = RegLow; RegNo < RegHigh; ++RegNo) {
            // If we find a register that is loaded inside the loop, 1. and 2.
            // are invalidated and we can exit.
            if (VgprUse.contains(RegNo))
              return false;
            VgprDef.insert(RegNo);
          }
      }
    }
  }
  if (!ST->hasVscnt() && HasVMemStore && !HasVMemLoad && UsesVgprLoadedOutside)
    return true;
  return HasVMemLoad && UsesVgprLoadedOutside;
}

// Return the appropriate VMEM_*_ACCESS type for Inst, which must be a VMEM or
// FLAT instruction.
WaitEventType
SIWaitCntsInserter::getVmemWaitEventType(const MachineInstr &Inst) const {
  // Maps VMEM access types to their corresponding WaitEventType.
  static const WaitEventType VmemReadMapping[NUM_VMEM_TYPES] = {
      VMEM_READ_ACCESS, VMEM_SAMPLER_READ_ACCESS, VMEM_BVH_READ_ACCESS};

  assert(SIInstrInfo::isVMEM(Inst) || SIInstrInfo::isFLAT(Inst));
  // LDS DMA loads are also stores, but on the LDS side. On the VMEM side
  // these should use VM_CNT.
  if (!ST->hasVscnt() || SIInstrInfo::mayWriteLDSThroughDMA(Inst))
    return VMEM_ACCESS;
  if (Inst.mayStore() && !SIInstrInfo::isAtomicRet(Inst)) {
    // FLAT and SCRATCH instructions may access scratch. Other VMEM
    // instructions do not.
    if (SIInstrInfo::isFLAT(Inst) && mayAccessScratchThroughFlat(Inst))
      return SCRATCH_WRITE_ACCESS;
    return VMEM_WRITE_ACCESS;
  }
  if (!ST->hasExtendedWaitCounts() || SIInstrInfo::isFLAT(Inst))
    return VMEM_READ_ACCESS;
  return VmemReadMapping[getVmemType(Inst)];
}

void SIWaitCntsInserter::setForceEmitWaitcnt() {
// For non-debug builds, ForceEmitWaitcnt has been initialized to false;
// For debug builds, get the debug counter info and adjust if need be
#ifndef NDEBUG
  if (DebugCounter::isCounterSet(ForceExpCounter) &&
      DebugCounter::shouldExecute(ForceExpCounter)) {
    ForceEmitWaitcnt[EXP_CNT] = true;
  } else {
    ForceEmitWaitcnt[EXP_CNT] = false;
  }

  if (DebugCounter::isCounterSet(ForceLgkmCounter) &&
      DebugCounter::shouldExecute(ForceLgkmCounter)) {
    ForceEmitWaitcnt[DS_CNT] = true;
    ForceEmitWaitcnt[KM_CNT] = true;
  } else {
    ForceEmitWaitcnt[DS_CNT] = false;
    ForceEmitWaitcnt[KM_CNT] = false;
  }

  if (DebugCounter::isCounterSet(ForceVMCounter) &&
      DebugCounter::shouldExecute(ForceVMCounter)) {
    ForceEmitWaitcnt[LOAD_CNT] = true;
    ForceEmitWaitcnt[SAMPLE_CNT] = true;
    ForceEmitWaitcnt[BVH_CNT] = true;
  } else {
    ForceEmitWaitcnt[LOAD_CNT] = false;
    ForceEmitWaitcnt[SAMPLE_CNT] = false;
    ForceEmitWaitcnt[BVH_CNT] = false;
  }
#endif // NDEBUG
}

///  Generate s_waitcnt instruction to be placed before cur_Inst.
///  Instructions of a given type are returned in order,
///  but instructions of different types can complete out of order.
///  We rely on this in-order completion
///  and simply assign a score to the memory access instructions.
///  We keep track of the active "score bracket" to determine
///  if an access of a memory read requires an s_waitcnt
///  and if so what the value of each counter is.
///  The "score bracket" is bound by the lower bound and upper bound
///  scores (*_score_LB and *_score_ub respectively).
///  If FlushVmCnt is true, that means that we want to generate a s_waitcnt to
///  flush the vmcnt counter here.
bool SIWaitCntsInserter::generateWaitcntInstBefore(
    MachineInstr &MI, WaitcntBrackets &ScoreBrackets,
    MachineInstr *OldWaitcntInstr, bool FlushVmCnt, VGPRInstsSet *VGPRInsts) {
  setForceEmitWaitcnt();

  if (MI.isMetaInstruction())
    return false;

  AMDGPU::Waitcnt Wait;

  // FIXME: This should have already been handled by the memory legalizer.
  // Removing this currently doesn't affect any lit tests, but we need to
  // verify that nothing was relying on this. The number of buffer invalidates
  // being handled here should not be expanded.
  if (MI.getOpcode() == AMDGPU::BUFFER_WBINVL1 ||
      MI.getOpcode() == AMDGPU::BUFFER_WBINVL1_SC ||
      MI.getOpcode() == AMDGPU::BUFFER_WBINVL1_VOL ||
      MI.getOpcode() == AMDGPU::BUFFER_GL0_INV ||
      MI.getOpcode() == AMDGPU::BUFFER_GL1_INV) {
    Wait.LoadCnt = 0;
  }

  MachineFunction *MF = MI.getParent()->getParent();
  bool OptNone = MF->getFunction().hasOptNone() ||
                 MF->getTarget().getOptLevel() == CodeGenOptLevel::None;
  InstCounterType SmemAccessCounter =
      eventCounter(WCG->getWaitEventMask(), SMEM_ACCESS);

  // All waits must be resolved at call return.
  // NOTE: this could be improved with knowledge of all call sites or
  //   with knowledge of the called routines.
  if (MI.getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG ||
      MI.getOpcode() == AMDGPU::SI_RETURN ||
      MI.getOpcode() == AMDGPU::S_SETPC_B64_return ||
      (MI.isReturn() && MI.isCall() && !callWaitsOnFunctionEntry(MI))) {
    Wait = Wait.combined(WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/false));
  }
  // Identify S_ENDPGM instructions which may have to wait for outstanding VMEM
  // stores. In this case it can be useful to send a message to explicitly
  // release all VGPRs before the stores have completed, but it is only safe to
  // do this if:
  // * there are no outstanding scratch stores
  // * we are not in Dynamic VGPR mode
  else if (MI.getOpcode() == AMDGPU::S_ENDPGM ||
           MI.getOpcode() == AMDGPU::S_ENDPGM_SAVED) {
    if (ST->getGeneration() >= AMDGPUSubtarget::GFX11 && !OptNone &&
        ScoreBrackets.getScoreRange(STORE_CNT) != 0 &&
        !ScoreBrackets.hasPendingEvent(SCRATCH_WRITE_ACCESS))
      VGPRInsts->insert(&MI);
  }
  // Resolve vm waits before gs-done.
  else if ((MI.getOpcode() == AMDGPU::S_SENDMSG ||
            MI.getOpcode() == AMDGPU::S_SENDMSGHALT) &&
           ST->hasLegacyGeometry() &&
           ((MI.getOperand(0).getImm() & AMDGPU::SendMsg::ID_MASK_PreGFX11_) ==
            AMDGPU::SendMsg::ID_GS_DONE_PreGFX11)) {
    Wait.LoadCnt = 0;
  }
#if 0 // TODO: the following blocks of logic when we have fence.
  else if (MI.getOpcode() == SC_FENCE) {
    const unsigned int group_size =
      context->shader_info->GetMaxThreadGroupSize();
    // group_size == 0 means thread group size is unknown at compile time
    const bool group_is_multi_wave =
      (group_size == 0 || group_size > target_info->GetWaveFrontSize());
    const bool fence_is_global = !((SCInstInternalMisc*)Inst)->IsGroupFence();

    for (unsigned int i = 0; i < Inst->NumSrcOperands(); i++) {
      SCRegType src_type = Inst->GetSrcType(i);
      switch (src_type) {
        case SCMEM_LDS:
          if (group_is_multi_wave ||
            context->OptFlagIsOn(OPT_R1100_LDSMEM_FENCE_CHICKEN_BIT)) {
            EmitWaitcnt |= ScoreBrackets->updateByWait(DS_CNT,
                               ScoreBrackets->getScoreUB(DS_CNT));
            // LDS may have to wait for VMcnt after buffer load to LDS
            if (target_info->HasBufferLoadToLDS()) {
              EmitWaitcnt |= ScoreBrackets->updateByWait(LOAD_CNT,
                                 ScoreBrackets->getScoreUB(LOAD_CNT));
            }
          }
          break;

        case SCMEM_GDS:
          if (group_is_multi_wave || fence_is_global) {
            EmitWaitcnt |= ScoreBrackets->updateByWait(EXP_CNT,
              ScoreBrackets->getScoreUB(EXP_CNT));
            EmitWaitcnt |= ScoreBrackets->updateByWait(DS_CNT,
              ScoreBrackets->getScoreUB(DS_CNT));
          }
          break;

        case SCMEM_UAV:
        case SCMEM_TFBUF:
        case SCMEM_RING:
        case SCMEM_SCATTER:
          if (group_is_multi_wave || fence_is_global) {
            EmitWaitcnt |= ScoreBrackets->updateByWait(EXP_CNT,
              ScoreBrackets->getScoreUB(EXP_CNT));
            EmitWaitcnt |= ScoreBrackets->updateByWait(LOAD_CNT,
              ScoreBrackets->getScoreUB(LOAD_CNT));
          }
          break;

        case SCMEM_SCRATCH:
        default:
          break;
      }
    }
  }
#endif

  // Export & GDS instructions do not read the EXEC mask until after the export
  // is granted (which can occur well after the instruction is issued).
  // The shader program must flush all EXP operations on the export-count
  // before overwriting the EXEC mask.
  else {
    if (MI.modifiesRegister(AMDGPU::EXEC, TRI)) {
      // Export and GDS are tracked individually, either may trigger a waitcnt
      // for EXEC.
      if (ScoreBrackets.hasPendingEvent(EXP_GPR_LOCK) ||
          ScoreBrackets.hasPendingEvent(EXP_PARAM_ACCESS) ||
          ScoreBrackets.hasPendingEvent(EXP_POS_ACCESS) ||
          ScoreBrackets.hasPendingEvent(GDS_GPR_LOCK)) {
        Wait.ExpCnt = 0;
      }
    }

    if (MI.isCall() && callWaitsOnFunctionEntry(MI)) {
      // The function is going to insert a wait on everything in its prolog.
      // This still needs to be careful if the call target is a load (e.g. a GOT
      // load). We also need to check WAW dependency with saved PC.
      Wait = AMDGPU::Waitcnt();

      int CallAddrOpIdx =
          AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::src0);

      if (MI.getOperand(CallAddrOpIdx).isReg()) {
        auto [CallAddrOpLow, CallAddrOpHigh] =
            ScoreBrackets.getRegInterval(&MI, MRI, TRI, CallAddrOpIdx);

        for (int RegNo = CallAddrOpLow; RegNo < CallAddrOpHigh; ++RegNo)
          ScoreBrackets.determineWait(SmemAccessCounter, RegNo, Wait);

        int RtnAddrOpIdx =
          AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::dst);
        if (RtnAddrOpIdx != -1) {
          auto [RtnAddrOpLow, RtnAddrOpHigh] =
              ScoreBrackets.getRegInterval(&MI, MRI, TRI, RtnAddrOpIdx);

          for (int RegNo = RtnAddrOpLow; RegNo < RtnAddrOpHigh; ++RegNo)
            ScoreBrackets.determineWait(SmemAccessCounter, RegNo, Wait);
        }
      }
    } else {
      // FIXME: Should not be relying on memoperands.
      // Look at the source operands of every instruction to see if
      // any of them results from a previous memory operation that affects
      // its current usage. If so, an s_waitcnt instruction needs to be
      // emitted.
      // If the source operand was defined by a load, add the s_waitcnt
      // instruction.
      //
      // Two cases are handled for destination operands:
      // 1) If the destination operand was defined by a load, add the s_waitcnt
      // instruction to guarantee the right WAW order.
      // 2) If a destination operand that was used by a recent export/store ins,
      // add s_waitcnt on exp_cnt to guarantee the WAR order.

      for (const MachineMemOperand *Memop : MI.memoperands()) {
        const Value *Ptr = Memop->getValue();
        if (Memop->isStore() && SLoadAddresses.count(Ptr)) {
          addWait(Wait, SmemAccessCounter, 0);
          if (PDT->dominates(MI.getParent(), SLoadAddresses.find(Ptr)->second))
            SLoadAddresses.erase(Ptr);
        }
        unsigned AS = Memop->getAddrSpace();
        if (AS != AMDGPUAS::LOCAL_ADDRESS && AS != AMDGPUAS::FLAT_ADDRESS)
          continue;
        // No need to wait before load from VMEM to LDS.
        if (TII->mayWriteLDSThroughDMA(MI))
          continue;

        // LOAD_CNT is only relevant to vgpr or LDS.
        unsigned RegNo = SQ_MAX_PGM_VGPRS + EXTRA_VGPR_LDS;
        bool FoundAliasingStore = false;
        // Only objects with alias scope info were added to LDSDMAScopes array.
        // In the absense of the scope info we will not be able to disambiguate
        // aliasing here. There is no need to try searching for a corresponding
        // store slot. This is conservatively correct because in that case we
        // will produce a wait using the first (general) LDS DMA wait slot which
        // will wait on all of them anyway.
        if (Ptr && Memop->getAAInfo() && Memop->getAAInfo().Scope) {
          const auto &LDSDMAStores = ScoreBrackets.getLDSDMAStores();
          for (unsigned I = 0, E = LDSDMAStores.size(); I != E; ++I) {
            if (MI.mayAlias(AA, *LDSDMAStores[I], true)) {
              FoundAliasingStore = true;
              ScoreBrackets.determineWait(LOAD_CNT, RegNo + I + 1, Wait);
            }
          }
        }
        if (!FoundAliasingStore)
          ScoreBrackets.determineWait(LOAD_CNT, RegNo, Wait);
        if (Memop->isStore()) {
          ScoreBrackets.determineWait(EXP_CNT, RegNo, Wait);
        }
      }

      // Loop over use and def operands.
      for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
        MachineOperand &Op = MI.getOperand(I);
        if (!Op.isReg())
          continue;

        // If the instruction does not read tied source, skip the operand.
        if (Op.isTied() && Op.isUse() && TII->doesNotReadTiedSource(MI))
          continue;

        auto [RegLow, RegHigh] = ScoreBrackets.getRegInterval(&MI, MRI, TRI, I);

        const bool IsVGPR = TRI->isVectorRegister(*MRI, Op.getReg());
        for (int RegNo = RegLow; RegNo < RegHigh; ++RegNo) {
          if (IsVGPR) {
            // RAW always needs an s_waitcnt. WAW needs an s_waitcnt unless the
            // previous write and this write are the same type of VMEM
            // instruction, in which case they're guaranteed to write their
            // results in order anyway.
            if (Op.isUse() || !updateVMCntOnly(MI) ||
                ScoreBrackets.hasOtherPendingVmemTypes(RegNo,
                                                       getVmemType(MI))) {
              ScoreBrackets.determineWait(LOAD_CNT, RegNo, Wait);
              ScoreBrackets.determineWait(SAMPLE_CNT, RegNo, Wait);
              ScoreBrackets.determineWait(BVH_CNT, RegNo, Wait);
              ScoreBrackets.clearVgprVmemTypes(RegNo);
            }
            if (Op.isDef() || ScoreBrackets.hasPendingEvent(EXP_LDS_ACCESS)) {
              ScoreBrackets.determineWait(EXP_CNT, RegNo, Wait);
            }
            ScoreBrackets.determineWait(DS_CNT, RegNo, Wait);
          } else {
            ScoreBrackets.determineWait(SmemAccessCounter, RegNo, Wait);
          }
        }
      }
    }
  }

  // The subtarget may have an implicit S_WAITCNT 0 before barriers. If it does
  // not, we need to ensure the subtarget is capable of backing off barrier
  // instructions in case there are any outstanding memory operations that may
  // cause an exception. Otherwise, insert an explicit S_WAITCNT 0 here.
  if (MI.getOpcode() == AMDGPU::S_BARRIER &&
      !ST->hasAutoWaitcntBeforeBarrier() && !ST->supportsBackOffBarrier()) {
    Wait = Wait.combined(WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/true));
  }

  // TODO: Remove this work-around, enable the assert for Bug 457939
  //       after fixing the scheduler. Also, the Shader Compiler code is
  //       independent of target.
  if (readsVCCZ(MI) && ST->hasReadVCCZBug()) {
    if (ScoreBrackets.hasPendingEvent(SMEM_ACCESS)) {
      Wait.DsCnt = 0;
    }
  }

  // Verify that the wait is actually needed.
  ScoreBrackets.simplifyWaitcnt(Wait);

  if (ForceEmitZeroWaitcnts)
    Wait = WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/false);

  if (ForceEmitWaitcnt[LOAD_CNT])
    Wait.LoadCnt = 0;
  if (ForceEmitWaitcnt[EXP_CNT])
    Wait.ExpCnt = 0;
  if (ForceEmitWaitcnt[DS_CNT])
    Wait.DsCnt = 0;
  if (ForceEmitWaitcnt[SAMPLE_CNT])
    Wait.SampleCnt = 0;
  if (ForceEmitWaitcnt[BVH_CNT])
    Wait.BvhCnt = 0;
  if (ForceEmitWaitcnt[KM_CNT])
    Wait.KmCnt = 0;

  if (FlushVmCnt) {
    if (ScoreBrackets.hasPendingEvent(LOAD_CNT))
      Wait.LoadCnt = 0;
    if (ScoreBrackets.hasPendingEvent(SAMPLE_CNT))
      Wait.SampleCnt = 0;
    if (ScoreBrackets.hasPendingEvent(BVH_CNT))
      Wait.BvhCnt = 0;
  }

  return generateWaitcnt(Wait, MI.getIterator(), *MI.getParent(), ScoreBrackets,
                         OldWaitcntInstr);
}

// Generate s_waitcnt instructions where needed.
bool SIWaitCntsInserter::insertWaitcntInBlock(MachineFunction &MF,
                                              MachineBasicBlock &Block,
                                              WaitcntBrackets &ScoreBrackets,
                                              VGPRInstsSet *VGPRInsts) {
  bool Modified = false;

  LLVM_DEBUG({
    dbgs() << "*** Block" << Block.getNumber() << " ***";
    ScoreBrackets.dump();
  });

  // Track the correctness of vccz through this basic block. There are two
  // reasons why it might be incorrect; see ST->hasReadVCCZBug() and
  // ST->partialVCCWritesUpdateVCCZ().
  bool VCCZCorrect = true;
  if (ST->hasReadVCCZBug()) {
    // vccz could be incorrect at a basic block boundary if a predecessor wrote
    // to vcc and then issued an smem load.
    VCCZCorrect = false;
  } else if (!ST->partialVCCWritesUpdateVCCZ()) {
    // vccz could be incorrect at a basic block boundary if a predecessor wrote
    // to vcc_lo or vcc_hi.
    VCCZCorrect = false;
  }

  // Walk over the instructions.
  MachineInstr *OldWaitcntInstr = nullptr;

  for (MachineBasicBlock::instr_iterator Iter = Block.instr_begin(),
                                         E = Block.instr_end();
       Iter != E;) {
    MachineInstr &Inst = *Iter;

    // Track pre-existing waitcnts that were added in earlier iterations or by
    // the memory legalizer.
    if (isWaitInstr(Inst)) {
      if (!OldWaitcntInstr)
        OldWaitcntInstr = &Inst;
      ++Iter;
      continue;
    }

    bool FlushVmCnt = Block.getFirstTerminator() == Inst &&
                      isPreheaderToFlush(Block, ScoreBrackets);

    // Generate an s_waitcnt instruction to be placed before Inst, if needed.
    Modified |= generateWaitcntInstBefore(Inst, ScoreBrackets, OldWaitcntInstr,
                                          FlushVmCnt, VGPRInsts);
    OldWaitcntInstr = nullptr;

    // Restore vccz if it's not known to be correct already.
    bool RestoreVCCZ = !VCCZCorrect && readsVCCZ(Inst);

    // Don't examine operands unless we need to track vccz correctness.
    if (ST->hasReadVCCZBug() || !ST->partialVCCWritesUpdateVCCZ()) {
      if (Inst.definesRegister(AMDGPU::VCC_LO) ||
          Inst.definesRegister(AMDGPU::VCC_HI)) {
        // Up to gfx9, writes to vcc_lo and vcc_hi don't update vccz.
        if (!ST->partialVCCWritesUpdateVCCZ())
          VCCZCorrect = false;
      } else if (Inst.definesRegister(AMDGPU::VCC)) {
        // There is a hardware bug on CI/SI where SMRD instruction may corrupt
        // vccz bit, so when we detect that an instruction may read from a
        // corrupt vccz bit, we need to:
        // 1. Insert s_waitcnt lgkm(0) to wait for all outstanding SMRD
        //    operations to complete.
        // 2. Restore the correct value of vccz by writing the current value
        //    of vcc back to vcc.
        if (ST->hasReadVCCZBug() &&
            ScoreBrackets.hasPendingEvent(SMEM_ACCESS)) {
          // Writes to vcc while there's an outstanding smem read may get
          // clobbered as soon as any read completes.
          VCCZCorrect = false;
        } else {
          // Writes to vcc will fix any incorrect value in vccz.
          VCCZCorrect = true;
        }
      }
    }

    if (TII->isSMRD(Inst)) {
      for (const MachineMemOperand *Memop : Inst.memoperands()) {
        // No need to handle invariant loads when avoiding WAR conflicts, as
        // there cannot be a vector store to the same memory location.
        if (!Memop->isInvariant()) {
          const Value *Ptr = Memop->getValue();
          SLoadAddresses.insert(std::pair(Ptr, Inst.getParent()));
        }
      }
      if (ST->hasReadVCCZBug()) {
        // This smem read could complete and clobber vccz at any time.
        VCCZCorrect = false;
      }
    }

    updateEventWaitcntAfter(Inst, &ScoreBrackets);

#if 0 // TODO: implement resource type check controlled by options with ub = LB.
    // If this instruction generates a S_SETVSKIP because it is an
    // indexed resource, and we are on Tahiti, then it will also force
    // an S_WAITCNT vmcnt(0)
    if (RequireCheckResourceType(Inst, context)) {
      // Force the score to as if an S_WAITCNT vmcnt(0) is emitted.
      ScoreBrackets->setScoreLB(LOAD_CNT,
      ScoreBrackets->getScoreUB(LOAD_CNT));
    }
#endif

    LLVM_DEBUG({
      Inst.print(dbgs());
      ScoreBrackets.dump();
    });

    // TODO: Remove this work-around after fixing the scheduler and enable the
    // assert above.
    if (RestoreVCCZ) {
      // Restore the vccz bit.  Any time a value is written to vcc, the vcc
      // bit is updated, so we can restore the bit by reading the value of
      // vcc and then writing it back to the register.
      BuildMI(Block, Inst, Inst.getDebugLoc(),
              TII->get(ST->isWave32() ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64),
              TRI->getVCC())
          .addReg(TRI->getVCC());
      VCCZCorrect = true;
      Modified = true;
    }

    ++Iter;
  }

  if (Block.getFirstTerminator() == Block.end() &&
      isPreheaderToFlush(Block, ScoreBrackets))
    Modified |= generateWaitcntBlockEnd(Block, ScoreBrackets, OldWaitcntInstr);

  return Modified;
}

void SIWaitCntsInserter::updateEventWaitcntAfter(
    MachineInstr &Inst, WaitcntBrackets *ScoreBrackets) {
  // Now look at the instruction opcode. If it is a memory access
  // instruction, update the upper-bound of the appropriate counter's
  // bracket and the destination operand scores.
  // TODO: Use the (TSFlags & SIInstrFlags::DS_CNT) property everywhere.

  if (TII->isDS(Inst) && TII->usesLGKM_CNT(Inst)) {
    if (TII->isAlwaysGDS(Inst.getOpcode()) ||
        TII->hasModifiersSet(Inst, AMDGPU::OpName::gds)) {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, GDS_ACCESS, Inst);
      ScoreBrackets->updateByEvent(TII, TRI, MRI, GDS_GPR_LOCK, Inst);
    } else {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, LDS_ACCESS, Inst);
    }
  } else if (TII->isFLAT(Inst)) {
    // TODO: Track this properly.
    if (isCacheInvOrWBInst(Inst))
      return;

    assert(Inst.mayLoadOrStore());

    int FlatASCount = 0;

    if (mayAccessVMEMThroughFlat(Inst)) {
      ++FlatASCount;
      ScoreBrackets->updateByEvent(TII, TRI, MRI, getVmemWaitEventType(Inst),
                                   Inst);
    }

    if (mayAccessLDSThroughFlat(Inst)) {
      ++FlatASCount;
      ScoreBrackets->updateByEvent(TII, TRI, MRI, LDS_ACCESS, Inst);
    }

    // A Flat memory operation must access at least one address space.
    assert(FlatASCount);

    // This is a flat memory operation that access both VMEM and LDS, so note it
    // - it will require that both the VM and LGKM be flushed to zero if it is
    // pending when a VM or LGKM dependency occurs.
    if (FlatASCount > 1)
      ScoreBrackets->setPendingFlat();
  } else if (SIInstrInfo::isVMEM(Inst) &&
             !llvm::AMDGPU::getMUBUFIsBufferInv(Inst.getOpcode())) {
    ScoreBrackets->updateByEvent(TII, TRI, MRI, getVmemWaitEventType(Inst),
                                 Inst);

    if (ST->vmemWriteNeedsExpWaitcnt() &&
        (Inst.mayStore() || SIInstrInfo::isAtomicRet(Inst))) {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, VMW_GPR_LOCK, Inst);
    }
  } else if (TII->isSMRD(Inst)) {
    ScoreBrackets->updateByEvent(TII, TRI, MRI, SMEM_ACCESS, Inst);
  } else if (Inst.isCall()) {
    if (callWaitsOnFunctionReturn(Inst)) {
      // Act as a wait on everything
      ScoreBrackets->applyWaitcnt(
          WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/false));
      ScoreBrackets->setStateOnFunctionEntryOrReturn();
    } else {
      // May need to way wait for anything.
      ScoreBrackets->applyWaitcnt(AMDGPU::Waitcnt());
    }
  } else if (SIInstrInfo::isLDSDIR(Inst)) {
    ScoreBrackets->updateByEvent(TII, TRI, MRI, EXP_LDS_ACCESS, Inst);
  } else if (TII->isVINTERP(Inst)) {
    int64_t Imm = TII->getNamedOperand(Inst, AMDGPU::OpName::waitexp)->getImm();
    ScoreBrackets->applyWaitcnt(EXP_CNT, Imm);
  } else if (SIInstrInfo::isEXP(Inst)) {
    unsigned Imm = TII->getNamedOperand(Inst, AMDGPU::OpName::tgt)->getImm();
    if (Imm >= AMDGPU::Exp::ET_PARAM0 && Imm <= AMDGPU::Exp::ET_PARAM31)
      ScoreBrackets->updateByEvent(TII, TRI, MRI, EXP_PARAM_ACCESS, Inst);
    else if (Imm >= AMDGPU::Exp::ET_POS0 && Imm <= AMDGPU::Exp::ET_POS_LAST)
      ScoreBrackets->updateByEvent(TII, TRI, MRI, EXP_POS_ACCESS, Inst);
    else
      ScoreBrackets->updateByEvent(TII, TRI, MRI, EXP_GPR_LOCK, Inst);
  } else {
    switch (Inst.getOpcode()) {
    case AMDGPU::S_SENDMSG:
    case AMDGPU::S_SENDMSG_RTN_B32:
    case AMDGPU::S_SENDMSG_RTN_B64:
    case AMDGPU::S_SENDMSGHALT:
      ScoreBrackets->updateByEvent(TII, TRI, MRI, SQ_MESSAGE, Inst);
      break;
    case AMDGPU::S_MEMTIME:
    case AMDGPU::S_MEMREALTIME:
    case AMDGPU::S_BARRIER_SIGNAL_ISFIRST_M0:
    case AMDGPU::S_BARRIER_SIGNAL_ISFIRST_IMM:
    case AMDGPU::S_BARRIER_LEAVE:
    case AMDGPU::S_GET_BARRIER_STATE_M0:
    case AMDGPU::S_GET_BARRIER_STATE_IMM:
      ScoreBrackets->updateByEvent(TII, TRI, MRI, SMEM_ACCESS, Inst);
      break;
    }
  }
}

class SIInsertWaitcnts : public MachineFunctionPass {
public:
  static char ID;

  SIInsertWaitcnts() : MachineFunctionPass(ID) {
    (void)ForceExpCounter;
    (void)ForceLgkmCounter;
    (void)ForceVMCounter;
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI Insert Wait Instructions";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfo>();
    AU.addRequired<MachinePostDominatorTree>();
    AU.addUsedIfAvailable<AAResultsWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

INITIALIZE_PASS_BEGIN(SIInsertWaitcnts, DEBUG_TYPE,
                      "SI Insert Wait Instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_END(SIInsertWaitcnts, DEBUG_TYPE, "SI Insert Wait Instructions",
                    false, false)

char SIInsertWaitcnts::ID = 0;

char &llvm::SIInsertWaitcntsID = SIInsertWaitcnts::ID;

FunctionPass *llvm::createSIInsertWaitcntsPass() {
  return new SIInsertWaitcnts();
}

bool SIInsertWaitcnts::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  MachineLoopInfo *MLI = &getAnalysis<MachineLoopInfo>();
  MachinePostDominatorTree *PDT = &getAnalysis<MachinePostDominatorTree>();
  AliasAnalysis *AA = nullptr;
  if (auto AAR = getAnalysisIfAvailable<AAResultsWrapperPass>())
    AA = &AAR->getAAResults();

  WaitCntGeneratorPreGFX12 WCGPreGFX12;
  WaitCntGeneratorGFX12Plus WCGGFX12Plus;
  InstCounterType MaxCounter;
  WaitCntGenerator *WCG =
      getWaitCntGenerator(MF, WCGPreGFX12, WCGGFX12Plus, MaxCounter);

  SIWaitCntsInserter WCountsInserter = SIWaitCntsInserter(
      ST, &MF.getRegInfo(), WCG, MaxCounter, ForceEmitZeroFlag, MLI, PDT, AA);

  // S_ENDPGM instructions before which we should insert a DEALLOC_VGPRS
  // message.
  DenseSet<MachineInstr *> ReleaseVGPRInsts;

  return WCountsInserter.insertWaitCntsInFunction(MF, &ReleaseVGPRInsts);
}
