//===- AMDGPUWaitcntTracking.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUWaitcntTracking.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "si-insert-waitcnts"

namespace llvm {
namespace AMDGPU {

namespace {
static bool isAsync(const MachineInstr &MI, const SIInstrInfo &TII) {
  if (!SIInstrInfo::isLDSDMA(MI))
    return false;
  if (SIInstrInfo::usesASYNC_CNT(MI))
    return true;
  const MachineOperand *Async = TII.getNamedOperand(MI, OpName::IsAsync);
  return Async && (Async->getImm());
}

static bool isNonAsyncLdsDmaWrite(const MachineInstr &MI,
                                  const SIInstrInfo &TII) {
  return SIInstrInfo::mayWriteLDSThroughDMA(MI) && !isAsync(MI, TII);
}

static bool isAsyncLdsDmaWrite(const MachineInstr &MI, const SIInstrInfo &TII) {
  return SIInstrInfo::mayWriteLDSThroughDMA(MI) && isAsync(MI, TII);
}

static bool shouldUpdateAsyncMark(const MachineInstr &MI, InstCounterType T,
                                  const SIInstrInfo &TII) {
  if (SIInstrInfo::usesTENSOR_CNT(MI))
    return T == TENSOR_CNT;
  if (!isAsyncLdsDmaWrite(MI, TII))
    return false;
  if (SIInstrInfo::usesASYNC_CNT(MI))
    return T == ASYNC_CNT;
  return T == LOAD_CNT;
}
} // namespace

//===----------------------------------------------------------------------===//
// WaitcntBrackets
//===----------------------------------------------------------------------===//

WaitcntBrackets::WaitcntBrackets(const WaitcntBracketsContext &Ctx)
    : Ctx(&Ctx) {
  assert(Ctx.TRI.getNumRegUnits() < REGUNITS_END);
}

#ifndef NDEBUG
WaitcntBrackets::~WaitcntBrackets() {
  unsigned NumUnusedVmem = 0, NumUnusedSGPRs = 0;
  for (auto &[ID, Val] : VMem) {
    if (Val.empty())
      ++NumUnusedVmem;
  }
  for (auto &[ID, Val] : SGPRs) {
    if (Val.empty())
      ++NumUnusedSGPRs;
  }

  if (NumUnusedVmem || NumUnusedSGPRs) {
    errs() << "WaitcntBracket had unused entries at destruction time: "
           << NumUnusedVmem << " VMem and " << NumUnusedSGPRs
           << " SGPR unused entries\n";
    std::abort();
  }
}
#endif

void WaitcntBrackets::verify() {
  // These track the same underlying hardware counter so the score ranges
  // should be identical.
  assert(getScoreLB(AMDGPU::VA_VDST_RD) == getScoreLB(AMDGPU::VA_VDST_WR));
  assert(getScoreUB(AMDGPU::VA_VDST_RD) == getScoreUB(AMDGPU::VA_VDST_WR));
}

void WaitcntBrackets::setScoreByOperand(const MachineOperand &Op,
                                        InstCounterType CntTy, unsigned Score) {
  setRegScore(Op.getReg().asMCReg(), CntTy, Score);
}

unsigned WaitcntBrackets::SGPRInfo::get(InstCounterType T) const {
  assert((T == DS_CNT || T == KM_CNT || T == X_CNT) && "Invalid counter");
  return T == X_CNT ? ScoreXCnt : ScoreDsKmCnt;
}
unsigned &WaitcntBrackets::SGPRInfo::get(InstCounterType T) {
  assert((T == DS_CNT || T == KM_CNT || T == X_CNT) && "Invalid counter");
  return T == X_CNT ? ScoreXCnt : ScoreDsKmCnt;
}

bool WaitcntBrackets::hasPendingFlat() const {
  return (
      (LastFlatDsCnt > ScoreLBs[DS_CNT] && LastFlatDsCnt <= ScoreUBs[DS_CNT]) ||
      (LastFlatLoadCnt > ScoreLBs[LOAD_CNT] &&
       LastFlatLoadCnt <= ScoreUBs[LOAD_CNT]));
}

void WaitcntBrackets::setPendingFlat() {
  LastFlatLoadCnt = ScoreUBs[LOAD_CNT];
  LastFlatDsCnt = ScoreUBs[DS_CNT];
}

bool WaitcntBrackets::hasPendingGDS() const {
  return LastGDS > ScoreLBs[DS_CNT] && LastGDS <= ScoreUBs[DS_CNT];
}

unsigned WaitcntBrackets::getPendingGDSWait() const {
  return std::min(getScoreUB(DS_CNT) - LastGDS, getLimit(DS_CNT) - 1);
}

bool WaitcntBrackets::hasDifferentVGPRPendingEvents(MCPhysReg Reg,
                                                    HWEvents E) const {
  for (MCRegUnit RU : regunits(Reg)) {
    auto It = VMem.find(toVMEMID(RU));
    if (It != VMem.end() && (It->second.VGPRPendingEvents & ~E).any())
      return true;
  }
  return false;
}

void WaitcntBrackets::clearVGPRPendingEvents(MCPhysReg Reg) {
  for (MCRegUnit RU : regunits(Reg)) {
    if (auto It = VMem.find(toVMEMID(RU)); It != VMem.end()) {
      It->second.VGPRPendingEvents = HWEvents::NONE;
      if (It->second.empty())
        VMem.erase(It);
    }
  }
}

void WaitcntBrackets::setStateOnFunctionEntryOrReturn() {
  setScoreUB(STORE_CNT, getScoreUB(STORE_CNT) + getLimit(STORE_CNT));
  PendingEvents |= getWaitEvents(STORE_CNT);
}

bool WaitcntBrackets::hasPendingEvent(InstCounterType T) const {
  bool HasPending = (PendingEvents & getWaitEvents(T)).any();
  assert(HasPending == !empty(T) &&
         "Expected pending events iff scoreboard is not empty");
  return HasPending;
}

bool WaitcntBrackets::hasMixedPendingEvents(InstCounterType T) const {
  HWEvents Events = PendingEvents & getWaitEvents(T);
  // Return true if more than one bit is set in Events.
  return Events.size() > 1;
}

// Return true if the subtarget is one that enables Point Sample Acceleration
// and the MachineInstr passed in is one to which it might be applied (the
// hardware makes this decision based on several factors, but we can't determine
// this at compile time, so we have to assume it might be applied if the
// instruction supports it).
bool WaitcntBrackets::hasPointSampleAccel(const MachineInstr &MI) const {
  if (!Ctx->ST.hasPointSampleAccel() || !SIInstrInfo::isMIMG(MI))
    return false;

  const MIMGInfo *Info = getMIMGInfo(MI.getOpcode());
  const MIMGBaseOpcodeInfo *BaseInfo = getMIMGBaseOpcodeInfo(Info->BaseOpcode);
  return BaseInfo->PointSampleAccel;
}

// Return true if the subtarget enables Point Sample Acceleration, the supplied
// MachineInstr is one to which it might be applied and the supplied interval is
// one that has outstanding writes to vmem-types different than VMEM_NOSAMPLER
// (this is the type that a point sample accelerated instruction effectively
// becomes)
bool WaitcntBrackets::hasPointSamplePendingVmemTypes(const MachineInstr &MI,
                                                     MCPhysReg Reg) const {
  if (!hasPointSampleAccel(MI))
    return false;

  return hasDifferentVGPRPendingEvents(Reg, HWEvents::VMEM_READ_ACCESS);
}

void WaitcntBrackets::updateByEvent(HWEvents E, MachineInstr &Inst) {
  assert(E.size() == 1 && "Expected singular event!");
  InstCounterType T = getCounterFromEvent(E);
  assert(T < Ctx->MaxCounter);

  const GCNSubtarget &ST = Ctx->ST;
  const SIInstrInfo &TII = Ctx->TII;
  const SIRegisterInfo &TRI = Ctx->TRI;
  const MachineRegisterInfo &MRI = Ctx->MRI;

  unsigned UB = getScoreUB(T);
  unsigned Increment = 1;
  if ((T == AMDGPU::VA_VDST_RD || T == AMDGPU::VA_VDST_WR) &&
      getHasMatrixScale(Inst.getOpcode()) &&
      ST.hasVOP3PX2IncrementsVaVdstTwice()) {
    // V_WMMA_SCALE instructions use VOP3PX2 encoding. Hardware treats this as
    // two VOP3P instructions and increments VA_VDST twice.
    Increment = 2;
  }
  unsigned CurrScore = UB + Increment;
  if (CurrScore == 0)
    report_fatal_error("InsertWaitcnt score wraparound");
  // PendingEvents and ScoreUB need to be update regardless if this event
  // changes the score of a register or not.
  // Examples including vm_cnt when buffer-store or lgkm_cnt when send-message.
  PendingEvents |= E;
  setScoreUB(T, CurrScore);

  if (T == EXP_CNT) {
    // Put score on the source vgprs. If this is a store, just use those
    // specific register(s).
    if (TII.isDS(Inst) && Inst.mayLoadOrStore()) {
      // All GDS operations must protect their address register (same as
      // export.)
      if (const auto *AddrOp = TII.getNamedOperand(Inst, OpName::addr))
        setScoreByOperand(*AddrOp, EXP_CNT, CurrScore);

      if (Inst.mayStore()) {
        if (const auto *Data0 = TII.getNamedOperand(Inst, OpName::data0))
          setScoreByOperand(*Data0, EXP_CNT, CurrScore);
        if (const auto *Data1 = TII.getNamedOperand(Inst, OpName::data1))
          setScoreByOperand(*Data1, EXP_CNT, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst) && !SIInstrInfo::isGWS(Inst) &&
                 Inst.getOpcode() != DS_APPEND &&
                 Inst.getOpcode() != DS_CONSUME &&
                 Inst.getOpcode() != DS_ORDERED_COUNT) {
        for (const MachineOperand &Op : Inst.all_uses()) {
          if (TRI.isVectorRegister(MRI, Op.getReg()))
            setScoreByOperand(Op, EXP_CNT, CurrScore);
        }
      }
    } else if (TII.isFLAT(Inst)) {
      if (Inst.mayStore()) {
        setScoreByOperand(*TII.getNamedOperand(Inst, OpName::data), EXP_CNT,
                          CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setScoreByOperand(*TII.getNamedOperand(Inst, OpName::data), EXP_CNT,
                          CurrScore);
      }
    } else if (TII.isMIMG(Inst)) {
      if (Inst.mayStore()) {
        setScoreByOperand(Inst.getOperand(0), EXP_CNT, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setScoreByOperand(*TII.getNamedOperand(Inst, OpName::data), EXP_CNT,
                          CurrScore);
      }
    } else if (TII.isMTBUF(Inst)) {
      if (Inst.mayStore())
        setScoreByOperand(Inst.getOperand(0), EXP_CNT, CurrScore);
    } else if (TII.isMUBUF(Inst)) {
      if (Inst.mayStore()) {
        setScoreByOperand(Inst.getOperand(0), EXP_CNT, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setScoreByOperand(*TII.getNamedOperand(Inst, OpName::data), EXP_CNT,
                          CurrScore);
      }
    } else if (TII.isLDSDIR(Inst)) {
      // LDSDIR instructions attach the score to the destination.
      setScoreByOperand(*TII.getNamedOperand(Inst, OpName::vdst), EXP_CNT,
                        CurrScore);
    } else {
      if (TII.isEXP(Inst)) {
        // For export the destination registers are really temps that
        // can be used as the actual source after export patching, so
        // we need to treat them like sources and set the EXP_CNT
        // score.
        for (MachineOperand &DefMO : Inst.all_defs()) {
          if (TRI.isVGPR(MRI, DefMO.getReg())) {
            setScoreByOperand(DefMO, EXP_CNT, CurrScore);
          }
        }
      }
      for (const MachineOperand &Op : Inst.all_uses()) {
        if (TRI.isVectorRegister(MRI, Op.getReg()))
          setScoreByOperand(Op, EXP_CNT, CurrScore);
      }
    }
  } else if (T == X_CNT) {
    HWEvents OtherEvent =
        E == HWEvents::SMEM_GROUP ? HWEvents::VMEM_GROUP : HWEvents::SMEM_GROUP;
    if (PendingEvents.contains(OtherEvent)) {
      // Hardware inserts an implicit xcnt between interleaved
      // SMEM and VMEM operations. So there will never be
      // outstanding address translations for both SMEM and
      // VMEM at the same time.
      setScoreLB(T, getScoreUB(T) - 1);
      PendingEvents -= OtherEvent;
    }
    for (const MachineOperand &Op : Inst.all_uses())
      setScoreByOperand(Op, T, CurrScore);
  } else if (T == AMDGPU::VA_VDST_RD || T == AMDGPU::VA_VDST_WR ||
             T == AMDGPU::VM_VSRC) {
    // Match the score to the VGPR destination or source registers as
    // appropriate
    for (const MachineOperand &Op : Inst.operands()) {
      if (!Op.isReg())
        continue;

      // Skip based on counter type and operand type
      if (T == AMDGPU::VA_VDST_RD && Op.isDef())
        continue; // RD tracks reads only
      if (T == AMDGPU::VA_VDST_WR && Op.isUse())
        continue; // WR tracks writes only
      if (T == AMDGPU::VM_VSRC && Op.isDef())
        continue;

      if (TRI.isVectorRegister(Ctx->MRI, Op.getReg()))
        setScoreByOperand(Op, T, CurrScore);
    }
  } else /* LGKM_CNT || EXP_CNT || VS_CNT || NUM_INST_CNTS */ {
    // Match the score to the destination registers.
    //
    // Check only explicit operands. Stores, especially spill stores, include
    // implicit uses and defs of their super registers which would create an
    // artificial dependency, while these are there only for register liveness
    // accounting purposes.
    //
    // Special cases where implicit register defs exists, such as M0 or VCC,
    // but none with memory instructions.
    for (const MachineOperand &Op : Inst.defs()) {
      if (T == LOAD_CNT || T == SAMPLE_CNT || T == BVH_CNT) {
        if (!TRI.isVectorRegister(MRI, Op.getReg()))
          continue;
        if (SIInstrInfo::updateVMCntOnly(Inst)) {
          // updateVMCntOnly should only leave us with VGPRs
          // MUBUF, MTBUF, MIMG, FlatGlobal, and FlatScratch only have VGPR/AGPR
          // defs.
          assert(TRI.isVectorRegister(MRI, Op.getReg()));
          HWEvents VGPRContext = getSimplifiedVMEMEventsFor(Inst, TII);
          // If instruction can have Point Sample Accel applied, we have to flag
          // this with another potential dependency
          if (hasPointSampleAccel(Inst))
            VGPRContext |= HWEvents::VMEM_READ_ACCESS;
          for (MCRegUnit RU : regunits(Op.getReg().asMCReg()))
            VMem[toVMEMID(RU)].VGPRPendingEvents |= VGPRContext;
        }
      }
      setScoreByOperand(Op, T, CurrScore);
    }
    if (Inst.mayStore() &&
        (TII.isDS(Inst) || isNonAsyncLdsDmaWrite(Inst, TII))) {
      // MUBUF and FLAT LDS DMA operations need a wait on vmcnt before LDS
      // written can be accessed. A load from LDS to VMEM does not need a wait.
      //
      // The "Slot" is the offset from LDSDMA_BEGIN. If it's non-zero, then
      // there is a MachineInstr in LDSDMAStores used to track this LDSDMA
      // store. The "Slot" is the index into LDSDMAStores + 1.
      unsigned Slot = 0;
      for (const auto *MemOp : Inst.memoperands()) {
        if (!MemOp->isStore() ||
            MemOp->getAddrSpace() != AMDGPUAS::LOCAL_ADDRESS)
          continue;
        // Comparing just AA info does not guarantee memoperands are equal
        // in general, but this is so for LDS DMA in practice.
        auto AAI = MemOp->getAAInfo();
        // Alias scope information gives a way to definitely identify an
        // original memory object and practically produced in the module LDS
        // lowering pass. If there is no scope available we will not be able
        // to disambiguate LDS aliasing as after the module lowering all LDS
        // is squashed into a single big object.
        if (!AAI || !AAI.Scope)
          break;
        for (unsigned I = 0, E = LDSDMAStores.size(); I != E && !Slot; ++I) {
          for (const auto *MemOp : LDSDMAStores[I]->memoperands()) {
            if (MemOp->isStore() && AAI == MemOp->getAAInfo()) {
              Slot = I + 1;
              break;
            }
          }
        }
        if (Slot)
          break;
        // The slot may not be valid because it can be >= NUM_LDSDMA which
        // means the scoreboard cannot track it. We still want to preserve the
        // MI in order to check alias information, though.
        LDSDMAStores.push_back(&Inst);
        Slot = LDSDMAStores.size();
        break;
      }
      setVMemScore(LDSDMA_BEGIN, T, CurrScore);
      if (Slot && Slot < NUM_LDSDMA)
        setVMemScore(LDSDMA_BEGIN + Slot, T, CurrScore);
    }

    if (shouldUpdateAsyncMark(Inst, T, TII)) {
      AsyncScore[T] = CurrScore;
    }

    if (SIInstrInfo::isSBarrierSCCWrite(Inst.getOpcode())) {
      setRegScore(SCC, T, CurrScore);
      PendingSCCWrite = &Inst;
    }
  }
}

void WaitcntBrackets::recordAsyncMark(MachineInstr &Inst) {
  // In the absence of loops, AsyncMarks can grow linearly with the program
  // until we encounter an ASYNCMARK_WAIT. We could drop the oldest mark above a
  // limit every time we push a new mark, but that seems like unnecessary work
  // in practical cases. We do separately truncate the array when processing a
  // loop, which should be sufficient.
  AsyncMarks.push_back(AsyncScore);
  AsyncScore = {};
  LLVM_DEBUG({
    dbgs() << "recordAsyncMark:\n" << Inst;
    for (const auto &Mark : AsyncMarks) {
      llvm::interleaveComma(Mark, dbgs());
      dbgs() << '\n';
    }
  });
}

void WaitcntBrackets::print(raw_ostream &OS) const {
  const GCNSubtarget &ST = Ctx->ST;
  const SIRegisterInfo &TRI = Ctx->TRI;

  for (auto T : inst_counter_types(Ctx->MaxCounter)) {
    unsigned SR = getScoreRange(T);
    switch (T) {
    case LOAD_CNT:
      OS << "    " << (ST.hasExtendedWaitCounts() ? "LOAD" : "VM") << "_CNT("
         << SR << "):";
      break;
    case DS_CNT:
      OS << "    " << (ST.hasExtendedWaitCounts() ? "DS" : "LGKM") << "_CNT("
         << SR << "):";
      break;
    case EXP_CNT:
      OS << "    EXP_CNT(" << SR << "):";
      break;
    case STORE_CNT:
      OS << "    " << (ST.hasExtendedWaitCounts() ? "STORE" : "VS") << "_CNT("
         << SR << "):";
      break;
    case SAMPLE_CNT:
      OS << "    SAMPLE_CNT(" << SR << "):";
      break;
    case BVH_CNT:
      OS << "    BVH_CNT(" << SR << "):";
      break;
    case KM_CNT:
      OS << "    KM_CNT(" << SR << "):";
      break;
    case X_CNT:
      OS << "    X_CNT(" << SR << "):";
      break;
    case ASYNC_CNT:
      OS << "    ASYNC_CNT(" << SR << "):";
      break;
    case AMDGPU::VA_VDST_RD:
      OS << "    VA_VDST_RD(" << SR << "): ";
      break;
    case AMDGPU::VA_VDST_WR:
      OS << "    VA_VDST_WR(" << SR << "): ";
      break;
    case VM_VSRC:
      OS << "    VM_VSRC(" << SR << "): ";
      break;
    default:
      OS << "    UNKNOWN(" << SR << "):";
      break;
    }

    if (SR != 0) {
      // Print vgpr scores.
      unsigned LB = getScoreLB(T);

      SmallVector<VMEMID> SortedVMEMIDs(VMem.keys());
      sort(SortedVMEMIDs);

      for (auto ID : SortedVMEMIDs) {
        unsigned RegScore = VMem.at(ID).Scores[T];
        if (RegScore <= LB)
          continue;
        unsigned RelScore = RegScore - LB - 1;
        if (ID < REGUNITS_END) {
          OS << ' ' << RelScore << ":"
             << printRegUnit(static_cast<MCRegUnit>(ID), &TRI);
        } else {
          assert(ID >= LDSDMA_BEGIN && ID < LDSDMA_END &&
                 "Unhandled/unexpected ID value!");
          OS << ' ' << RelScore << ":LDSDMA" << ID;
        }
      }

      // Also need to print sgpr scores for lgkm_cnt or xcnt.
      if (isSmemAccessCounter(T) || T == X_CNT) {
        SmallVector<MCRegUnit> SortedSMEMIDs(SGPRs.keys());
        sort(SortedSMEMIDs);
        for (auto ID : SortedSMEMIDs) {
          unsigned RegScore = SGPRs.at(ID).get(T);
          if (RegScore <= LB)
            continue;
          unsigned RelScore = RegScore - LB - 1;
          OS << ' ' << RelScore << ":"
             << printRegUnit(static_cast<MCRegUnit>(ID), &TRI);
        }
      }

      if (T == KM_CNT && SCCScore > 0)
        OS << ' ' << SCCScore << ":scc";
    }
    OS << '\n';
  }

  OS << "Pending Events: ";
  if (hasPendingEvent()) {
    OS << getPendingEvents();
  } else {
    OS << "none";
  }
  OS << '\n';

  OS << "Async score: ";
  if (AsyncScore.empty())
    OS << "none";
  else
    llvm::interleaveComma(AsyncScore, OS);
  OS << '\n';

  OS << "Async marks: " << AsyncMarks.size() << '\n';

  for (const auto &Mark : AsyncMarks) {
    for (auto T : inst_counter_types()) {
      unsigned MarkedScore = Mark[T];
      switch (T) {
      case LOAD_CNT:
        OS << "  " << (ST.hasExtendedWaitCounts() ? "LOAD" : "VM")
           << "_CNT: " << MarkedScore;
        break;
      case DS_CNT:
        OS << "  " << (ST.hasExtendedWaitCounts() ? "DS" : "LGKM")
           << "_CNT: " << MarkedScore;
        break;
      case EXP_CNT:
        OS << "  EXP_CNT: " << MarkedScore;
        break;
      case STORE_CNT:
        OS << "  " << (ST.hasExtendedWaitCounts() ? "STORE" : "VS")
           << "_CNT: " << MarkedScore;
        break;
      case SAMPLE_CNT:
        OS << "  SAMPLE_CNT: " << MarkedScore;
        break;
      case BVH_CNT:
        OS << "  BVH_CNT: " << MarkedScore;
        break;
      case KM_CNT:
        OS << "  KM_CNT: " << MarkedScore;
        break;
      case X_CNT:
        OS << "  X_CNT: " << MarkedScore;
        break;
      case ASYNC_CNT:
        OS << "  ASYNC_CNT: " << MarkedScore;
        break;
      default:
        OS << "  UNKNOWN: " << MarkedScore;
        break;
      }
    }
    OS << '\n';
  }
  OS << '\n';
}

void WaitcntBrackets::simplifyWaitcnt(Waitcnt &Wait) const {
  simplifyWaitcnt(Wait, Wait);
}

/// Simplify \p UpdateWait by removing waits that are redundant based on the
/// current WaitcntBrackets and any other waits specified in \p CheckWait.
void WaitcntBrackets::simplifyWaitcnt(const Waitcnt &CheckWait,
                                      Waitcnt &UpdateWait) const {
  simplifyWaitcnt(UpdateWait, LOAD_CNT);
  simplifyWaitcnt(UpdateWait, EXP_CNT);
  simplifyWaitcnt(UpdateWait, DS_CNT);
  simplifyWaitcnt(UpdateWait, STORE_CNT);
  simplifyWaitcnt(UpdateWait, SAMPLE_CNT);
  simplifyWaitcnt(UpdateWait, BVH_CNT);
  simplifyWaitcnt(UpdateWait, KM_CNT);
  simplifyXcnt(CheckWait, UpdateWait);
  simplifyWaitcnt(UpdateWait, VA_VDST_RD);
  simplifyWaitcnt(UpdateWait, VA_VDST_WR);
  simplifyVmVsrc(CheckWait, UpdateWait);
  simplifyWaitcnt(UpdateWait, ASYNC_CNT);
}

void WaitcntBrackets::simplifyWaitcnt(InstCounterType T,
                                      unsigned &Count) const {
  // The number of outstanding events for this type, T, can be calculated
  // as (UB - LB). If the current Count is greater than or equal to the number
  // of outstanding events, then the wait for this counter is redundant.
  if (Count >= getScoreRange(T))
    Count = ~0u;
}

void WaitcntBrackets::simplifyWaitcnt(Waitcnt &Wait, InstCounterType T) const {
  unsigned Cnt = Wait.get(T);
  simplifyWaitcnt(T, Cnt);
  Wait.set(T, Cnt);
}

void WaitcntBrackets::simplifyXcnt(const Waitcnt &CheckWait,
                                   Waitcnt &UpdateWait) const {
  // Try to simplify xcnt further by checking for joint kmcnt and loadcnt
  // optimizations. On entry to a block with multiple predescessors, there may
  // be pending SMEM and VMEM events active at the same time.
  // In such cases, only clear one active event at a time.
  // TODO: Revisit xcnt optimizations for gfx1250.
  // Wait on XCNT is redundant if we are already waiting for a load to complete.
  // SMEM can return out of order, so only omit XCNT wait if we are waiting till
  // zero.
  if (CheckWait.get(KM_CNT) == 0 && hasPendingEvent(HWEvents::SMEM_GROUP))
    UpdateWait.set(X_CNT, ~0u);
  // If we have pending store we cannot optimize XCnt because we do not wait for
  // stores. VMEM loads retun in order, so if we only have loads XCnt is
  // decremented to the same number as LOADCnt.
  if (CheckWait.get(LOAD_CNT) != ~0u && hasPendingEvent(HWEvents::VMEM_GROUP) &&
      !hasPendingEvent(STORE_CNT) &&
      CheckWait.get(X_CNT) >= CheckWait.get(LOAD_CNT))
    UpdateWait.set(X_CNT, ~0u);
  simplifyWaitcnt(UpdateWait, X_CNT);
}

void WaitcntBrackets::simplifyVmVsrc(const Waitcnt &CheckWait,
                                     Waitcnt &UpdateWait) const {
  // Waiting for a VMEM counter (like LOAD_CNT) implies an equivalent wait for
  // VM_VSRC, because if the VMEM operation has completed then it must surely
  // have read its VGPR sources, but only if there are no other outstanding VMEM
  // operations that use a different counter (like SAMPLE_CNT).
  static constexpr AMDGPU::InstCounterType VmemCounters[] = {
      AMDGPU::LOAD_CNT, AMDGPU::STORE_CNT, AMDGPU::SAMPLE_CNT, AMDGPU::BVH_CNT,
      AMDGPU::DS_CNT};
  HWEvents VmemEvents = llvm::accumulate(
      VmemCounters, HWEvents(), [&](HWEvents Acc, AMDGPU::InstCounterType T) {
        return Acc | getWaitEvents(T);
      });
  HWEvents PendingVmemEvents = PendingEvents & VmemEvents;
  for (AMDGPU::InstCounterType T : VmemCounters) {
    unsigned CheckCount = CheckWait.get(T);
    if (UpdateWait.get(AMDGPU::VM_VSRC) >= CheckCount &&
        (CheckCount == 0 || !counterOutOfOrder(T)) &&
        (PendingVmemEvents & ~getWaitEvents(T)) == 0)
      UpdateWait.set(AMDGPU::VM_VSRC, ~0u);
  }
  simplifyWaitcnt(UpdateWait, VM_VSRC);
}

unsigned WaitcntBrackets::getScoreLB(InstCounterType T) const {
  assert(T < NUM_INST_CNTS);
  return ScoreLBs[T];
}

unsigned WaitcntBrackets::getScoreUB(InstCounterType T) const {
  assert(T < NUM_INST_CNTS);
  return ScoreUBs[T];
}

unsigned WaitcntBrackets::getScoreRange(InstCounterType T) const {
  return getScoreUB(T) - getScoreLB(T);
}

unsigned WaitcntBrackets::getSGPRScore(MCRegUnit RU, InstCounterType T) const {
  auto It = SGPRs.find(RU);
  return It != SGPRs.end() ? It->second.get(T) : 0;
}

unsigned WaitcntBrackets::getVMemScore(VMEMID TID, InstCounterType T) const {
  auto It = VMem.find(TID);
  return It != VMem.end() ? It->second.Scores[T] : 0;
}

unsigned WaitcntBrackets::getLimit(InstCounterType T) const {
  return Ctx->Limits.get(T);
}

void WaitcntBrackets::purgeEmptyTrackingData() {
  VMem.remove_if([](const auto &P) { return P.second.empty(); });
  SGPRs.remove_if([](const auto &P) { return P.second.empty(); });
}

void WaitcntBrackets::determineWaitForScore(InstCounterType T,
                                            unsigned ScoreToWait,
                                            Waitcnt &Wait) const {
  const unsigned LB = getScoreLB(T);
  const unsigned UB = getScoreUB(T);

  // If the score falls within the bracket, we need a waitcnt.
  if ((UB >= ScoreToWait) && (ScoreToWait > LB)) {
    if ((T == LOAD_CNT || T == DS_CNT) && hasPendingFlat() &&
        !Ctx->ST.hasFlatLgkmVMemCountInOrder()) {
      // If there is a pending FLAT operation, and this is a VMem or LGKM
      // waitcnt and the target can report early completion, then we need
      // to force a waitcnt 0.
      Wait.add(T, 0);
    } else if (counterOutOfOrder(T)) {
      // Counter can get decremented out-of-order when there
      // are multiple types event in the bracket. Also emit an s_wait counter
      // with a conservative value of 0 for the counter.
      Wait.add(T, 0);
    } else {
      // If a counter has been maxed out avoid overflow by waiting for
      // MAX(CounterType) - 1 instead.
      unsigned NeededWait = std::min(UB - ScoreToWait, getLimit(T) - 1);
      Wait.add(T, NeededWait);
    }
  }
}

Waitcnt WaitcntBrackets::determineAsyncWait(unsigned N) {
  LLVM_DEBUG({
    dbgs() << "Need " << N << " async marks. Found " << AsyncMarks.size()
           << ":\n";
    for (const auto &Mark : AsyncMarks) {
      llvm::interleaveComma(Mark, dbgs());
      dbgs() << '\n';
    }
  });

  if (AsyncMarks.size() == MaxAsyncMarks) {
    // Enforcing MaxAsyncMarks here is unnecessary work because the size of
    // MaxAsyncMarks is linear when traversing straightline code. But we do
    // need to check if truncation may have occured at a merge, and adjust N
    // to ensure that a wait is generated.
    LLVM_DEBUG(dbgs() << "Possible truncation. Ensuring a non-trivial wait.\n");
    N = std::min(N, (unsigned)MaxAsyncMarks - 1);
  }

  Waitcnt Wait;
  if (AsyncMarks.size() <= N) {
    LLVM_DEBUG(dbgs() << "No additional wait for async mark.\n");
    return Wait;
  }

  size_t MarkIndex = AsyncMarks.size() - N - 1;
  const auto &RequiredMark = AsyncMarks[MarkIndex];
  for (InstCounterType T : inst_counter_types())
    determineWaitForScore(T, RequiredMark[T], Wait);

  // Immediately remove the waited mark and all older ones
  // This happens BEFORE the wait is actually inserted, which is fine
  // because we've already extracted the wait requirements
  LLVM_DEBUG({
    dbgs() << "Removing " << (MarkIndex + 1)
           << " async marks after determining wait\n";
  });
  AsyncMarks.erase(AsyncMarks.begin(), AsyncMarks.begin() + MarkIndex + 1);

  LLVM_DEBUG(dbgs() << "Waits to add: " << Wait);
  return Wait;
}

// With D16Write32BitVgpr, D16 inst might be clobbered by events running on the
// other half 16bit.
//
// Replace VGPR16 to VGPR32 for wait check if:
// 1. MI is a VALU, and there is a wait event on the other half
// 2. MI is a LdSt, and there is a wait event on the other half from different
// order group
MCPhysReg WaitcntBrackets::determineVGPR16Dependency(const MachineInstr &MI,
                                                     InstCounterType T,
                                                     MCPhysReg Reg) const {
  const SIRegisterInfo &TRI = Ctx->TRI;
  const GCNSubtarget &ST = Ctx->ST;
  const SIInstrInfo &TII = Ctx->TII;

  const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Reg);
  unsigned Size = TRI.getRegSizeInBits(*RC);

  if (Size != 16 || !ST.hasD16Writes32BitVgpr())
    return Reg;

  // With D16Writes32BitVgpr, D16 Inst might clobber the whole vgpr32
  // check dependency on the other half
  Register Reg32 = TRI.get32BitRegister(Reg);
  Register OtherHalf = TRI.getSubReg(Reg32, isHi16Reg(Reg, TRI) ? lo16 : hi16);

  Waitcnt Wait;
  for (MCRegUnit RU : regunits(OtherHalf))
    determineWaitForScore(T, getVMemScore(toVMEMID(RU), T), Wait);

  // No wait on otherhalf
  if (!Wait.hasWait())
    return Reg;

  if (TII.isVALU(MI, /*AllowLDSDMA=*/true))
    return Reg32;

  // If hi/lo16 mixed events
  HWEvents MIEvents = getEventsFor(MI, ST, Ctx->IsExpertMode, Ctx->IsTgSplit);
  HWEvents OtherHalfEvents = getWaitEvents(T);
  HWEvents Events = MIEvents & OtherHalfEvents;
  if (Events.size() > 1)
    return Reg32;
  return Reg;
}

void WaitcntBrackets::determineWaitForPhysReg(InstCounterType T, MCPhysReg Reg,
                                              Waitcnt &Wait,
                                              const MachineInstr &MI) const {
  if (Reg == SCC) {
    determineWaitForScore(T, SCCScore, Wait);
  } else {
    bool IsVGPR = Ctx->TRI.isVectorRegister(Ctx->MRI, Reg);
    if (IsVGPR)
      Reg = determineVGPR16Dependency(MI, T, Reg);
    for (MCRegUnit RU : regunits(Reg))
      determineWaitForScore(
          T, IsVGPR ? getVMemScore(toVMEMID(RU), T) : getSGPRScore(RU, T),
          Wait);
  }
}

void WaitcntBrackets::determineWaitForLDSDMA(InstCounterType T, VMEMID TID,
                                             Waitcnt &Wait) const {
  assert(TID >= LDSDMA_BEGIN && TID < LDSDMA_END);
  determineWaitForScore(T, getVMemScore(TID, T), Wait);
}

void WaitcntBrackets::tryClearSCCWriteEvent(MachineInstr *Inst) {
  // S_BARRIER_WAIT on the same barrier guarantees that the pending write to
  // SCC has landed
  if (PendingSCCWrite &&
      PendingSCCWrite->getOpcode() == S_BARRIER_SIGNAL_ISFIRST_IMM &&
      PendingSCCWrite->getOperand(0).getImm() == Inst->getOperand(0).getImm()) {
    HWEvents SCCWRITEPendingEvent = HWEvents::SCC_WRITE;
    // If this SCC_WRITE is the only pending KM_CNT event, clear counter.
    if ((PendingEvents & getWaitEvents(KM_CNT)) == SCCWRITEPendingEvent) {
      setScoreLB(KM_CNT, getScoreUB(KM_CNT));
    }

    PendingEvents -= SCCWRITEPendingEvent;
    PendingSCCWrite = nullptr;
  }
}

void WaitcntBrackets::applyWaitcnt(const Waitcnt &Wait) {
  for (AMDGPU::InstCounterType T : AMDGPU::inst_counter_types()) {
    unsigned Cnt;
    if (T == AMDGPU::VA_VDST_RD || T == AMDGPU::VA_VDST_WR) {
      Cnt =
          std::min(Wait.get(AMDGPU::VA_VDST_RD), Wait.get(AMDGPU::VA_VDST_WR));
    } else {
      Cnt = Wait.get(T);
    }
    applyWaitcnt(T, Cnt);
  }
}

void WaitcntBrackets::applyWaitcnt(InstCounterType T, unsigned Count) {
  const unsigned UB = getScoreUB(T);
  if (Count >= UB)
    return;
  if (Count != 0) {
    if (counterOutOfOrder(T))
      return;
    setScoreLB(T, std::max(getScoreLB(T), UB - Count));
  } else {
    setScoreLB(T, UB);
    PendingEvents -= getWaitEvents(T);
  }

  if (T == KM_CNT && Count == 0 && hasPendingEvent(HWEvents::SMEM_GROUP)) {
    if (!hasMixedPendingEvents(X_CNT))
      applyWaitcnt(X_CNT, 0);
    else
      PendingEvents -= HWEvents::SMEM_GROUP;
  }
  if (T == LOAD_CNT && hasPendingEvent(HWEvents::VMEM_GROUP) &&
      !hasPendingEvent(STORE_CNT)) {
    if (!hasMixedPendingEvents(X_CNT))
      applyWaitcnt(X_CNT, Count);
    else if (Count == 0)
      PendingEvents -= HWEvents::VMEM_GROUP;
  }
}

void WaitcntBrackets::applyWaitcnt(const Waitcnt &Wait, InstCounterType T) {
  unsigned Cnt = Wait.get(T);
  applyWaitcnt(T, Cnt);
}

// Where there are multiple types of event in the bracket of a counter,
// the decrement may go out of order.
bool WaitcntBrackets::counterOutOfOrder(InstCounterType T) const {
  // Scalar memory read always can go out of order.
  if ((isSmemAccessCounter(T) && hasPendingEvent(HWEvents::SMEM_ACCESS)) ||
      (T == X_CNT && hasPendingEvent(HWEvents::SMEM_GROUP)))
    return true;

  if (T == LOAD_CNT) {

    // On targets without VScnt, LOAD_CNT includes all of STORE_CNT as well.
    // All these events use one counter and do not go out of order with respect
    // to each other.
    if (!Ctx->ST.hasVscnt())
      return false;

    HWEvents Events = PendingEvents & getWaitEvents(T);

    // If the target does not have extended counters, VMEM_BVH/SAMPLE_READ
    // events are equivalent to VMEM_READ_ACCESS. We do not go out of order in
    // such cases.
    static constexpr HWEvents ExtendedImageEvents =
        HWEvents::VMEM_SAMPLER_READ_ACCESS | HWEvents::VMEM_BVH_READ_ACCESS;
    if (!Ctx->ST.hasExtendedWaitCounts() &&
        (Events & ExtendedImageEvents).any()) {
      Events -= ExtendedImageEvents;
      Events |= HWEvents::VMEM_READ_ACCESS;
    }

    // GLOBAL_INV completes in-order with other LOAD_CNT events,
    // so having GLOBAL_INV_ACCESS mixed with other LOAD_CNT
    // events doesn't cause out-of-order completion.
    Events -= HWEvents::GLOBAL_INV_ACCESS;

    // Return true only if there are still multiple event types after removing
    // GLOBAL_INV
    return Events.size() > 1;
  }

  return hasMixedPendingEvents(T);
}

bool WaitcntBrackets::mergeScore(const MergeInfo &M, unsigned &Score,
                                 unsigned OtherScore) {
  unsigned MyShifted = Score <= M.OldLB ? 0 : Score + M.MyShift;
  unsigned OtherShifted =
      OtherScore <= M.OtherLB ? 0 : OtherScore + M.OtherShift;
  Score = std::max(MyShifted, OtherShifted);
  return OtherShifted > MyShifted;
}

bool WaitcntBrackets::mergeAsyncMarks(ArrayRef<MergeInfo> MergeInfos,
                                      ArrayRef<CounterValueArray> OtherMarks) {
  bool StrictDom = false;

  LLVM_DEBUG(dbgs() << "Merging async marks ...");
  // Early exit: nothing to merge when both sides are empty.
  if (AsyncMarks.empty() && OtherMarks.empty()) {
    LLVM_DEBUG(dbgs() << " nothing to merge\n");
    return false;
  }
  LLVM_DEBUG(dbgs() << '\n');

  // Determine maximum length needed after merging
  auto MaxSize = (unsigned)std::max(AsyncMarks.size(), OtherMarks.size());
  MaxSize = std::min(MaxSize, MaxAsyncMarks);

  // Keep only the most recent marks within our limit.
  if (AsyncMarks.size() > MaxSize)
    AsyncMarks.erase(AsyncMarks.begin(),
                     AsyncMarks.begin() + (AsyncMarks.size() - MaxSize));

  // Pad with zero-filled marks if our list is shorter. Zero represents "no
  // pending async operations at this checkpoint" and acts as the identity
  // element for max() during merging. We pad at the beginning since the marks
  // need to be aligned in most-recent order.
  constexpr CounterValueArray ZeroMark{};
  AsyncMarks.insert(AsyncMarks.begin(), MaxSize - AsyncMarks.size(), ZeroMark);

  LLVM_DEBUG({
    dbgs() << "Before merge:\n";
    for (const auto &Mark : AsyncMarks) {
      llvm::interleaveComma(Mark, dbgs());
      dbgs() << '\n';
    }
    dbgs() << "Other marks:\n";
    for (const auto &Mark : OtherMarks) {
      llvm::interleaveComma(Mark, dbgs());
      dbgs() << '\n';
    }
  });

  // Merge element-wise using the existing mergeScore function and the
  // appropriate MergeInfo for each counter type. Iterate only while we have
  // elements in both vectors.
  unsigned OtherSize = OtherMarks.size();
  unsigned OurSize = AsyncMarks.size();
  unsigned MergeCount = std::min(OtherSize, OurSize);
  // OtherMarks is empty -> OtherSize == 0 -> MergeCount == 0.
  // Our existing marks are the conservative result; return early to avoid
  // passing MergeCount == 0 to seq_inclusive which asserts Begin <= End.
  if (MergeCount == 0)
    return StrictDom;
  for (auto Idx : seq_inclusive<unsigned>(1, MergeCount)) {
    for (auto T : inst_counter_types(Ctx->MaxCounter)) {
      StrictDom |= mergeScore(MergeInfos[T], AsyncMarks[OurSize - Idx][T],
                              OtherMarks[OtherSize - Idx][T]);
    }
  }

  LLVM_DEBUG({
    dbgs() << "After merge:\n";
    for (const auto &Mark : AsyncMarks) {
      llvm::interleaveComma(Mark, dbgs());
      dbgs() << '\n';
    }
  });

  return StrictDom;
}

iterator_range<MCRegUnitIterator>
WaitcntBrackets::regunits(MCPhysReg Reg) const {
  assert(Reg != SCC && "Shouldn't be used on SCC");
  if (!Ctx->TRI.isInAllocatableClass(Reg))
    return {{}, {}};
  return Ctx->TRI.regunits(Reg);
}

void WaitcntBrackets::setScoreLB(InstCounterType T, unsigned Val) {
  assert(T < NUM_INST_CNTS);
  ScoreLBs[T] = Val;
}

void WaitcntBrackets::setScoreUB(InstCounterType T, unsigned Val) {
  assert(T < NUM_INST_CNTS);
  ScoreUBs[T] = Val;

  if (T != EXP_CNT)
    return;

  if (getScoreRange(EXP_CNT) > getLimit(EXP_CNT))
    ScoreLBs[EXP_CNT] = ScoreUBs[EXP_CNT] - getLimit(EXP_CNT);
}

void WaitcntBrackets::setRegScore(MCPhysReg Reg, InstCounterType T,
                                  unsigned Val) {
  if (Reg == SCC) {
    SCCScore = Val;
  } else if (Ctx->TRI.isVectorRegister(Ctx->MRI, Reg)) {
    for (MCRegUnit RU : regunits(Reg))
      VMem[toVMEMID(RU)].Scores[T] = Val;
  } else if (Ctx->TRI.isSGPRReg(Ctx->MRI, Reg)) {
    for (MCRegUnit RU : regunits(Reg))
      SGPRs[RU].get(T) = Val;
  } else {
    llvm_unreachable("Register cannot be tracked/unknown register!");
  }
}

/// Merge the pending events and associater score brackets of \p Other into
/// this brackets status.
///
/// Returns whether the merge resulted in a change that requires tighter waits
/// (i.e. the merged brackets strictly dominate the original brackets).
bool WaitcntBrackets::merge(const WaitcntBrackets &Other) {
  bool StrictDom = false;

  // Check if "other" has keys we don't have, and create default entries for
  // those. If they remain empty after merging, we will clean it up after.
  for (auto K : Other.VMem.keys())
    VMem.try_emplace(K);
  for (auto K : Other.SGPRs.keys())
    SGPRs.try_emplace(K);

  // Array to store MergeInfo for each counter type
  MergeInfo MergeInfos[NUM_INST_CNTS];

  for (auto T : inst_counter_types(Ctx->MaxCounter)) {
    // Merge event flags for this counter
    const HWEvents &EventsForT = getWaitEvents(T);
    const HWEvents OldEvents = PendingEvents & EventsForT;
    const HWEvents OtherEvents = Other.PendingEvents & EventsForT;
    if (!OldEvents.contains(OtherEvents))
      StrictDom = true;
    PendingEvents |= OtherEvents;

    // Merge scores for this counter
    const unsigned MyPending = ScoreUBs[T] - ScoreLBs[T];
    const unsigned OtherPending = Other.ScoreUBs[T] - Other.ScoreLBs[T];
    const unsigned NewUB = ScoreLBs[T] + std::max(MyPending, OtherPending);
    if (NewUB < ScoreLBs[T])
      report_fatal_error("waitcnt score overflow");

    MergeInfo &M = MergeInfos[T];
    M.OldLB = ScoreLBs[T];
    M.OtherLB = Other.ScoreLBs[T];
    M.MyShift = NewUB - ScoreUBs[T];
    M.OtherShift = NewUB - Other.ScoreUBs[T];

    ScoreUBs[T] = NewUB;

    if (T == LOAD_CNT)
      StrictDom |= mergeScore(M, LastFlatLoadCnt, Other.LastFlatLoadCnt);

    if (T == DS_CNT) {
      StrictDom |= mergeScore(M, LastFlatDsCnt, Other.LastFlatDsCnt);
      StrictDom |= mergeScore(M, LastGDS, Other.LastGDS);
    }

    if (T == KM_CNT) {
      StrictDom |= mergeScore(M, SCCScore, Other.SCCScore);
      if (Other.hasPendingEvent(HWEvents::SCC_WRITE)) {
        if (!(OldEvents & HWEvents::SCC_WRITE)) {
          PendingSCCWrite = Other.PendingSCCWrite;
        } else if (PendingSCCWrite != Other.PendingSCCWrite) {
          PendingSCCWrite = nullptr;
        }
      }
    }

    for (auto &[RegID, Info] : VMem)
      StrictDom |= mergeScore(M, Info.Scores[T], Other.getVMemScore(RegID, T));

    if (isSmemAccessCounter(T) || T == X_CNT) {
      for (auto &[RegID, Info] : SGPRs) {
        auto It = Other.SGPRs.find(RegID);
        unsigned OtherScore = (It != Other.SGPRs.end()) ? It->second.get(T) : 0;
        StrictDom |= mergeScore(M, Info.get(T), OtherScore);
      }
    }
  }

  for (auto &[TID, Info] : VMem) {
    if (auto It = Other.VMem.find(TID); It != Other.VMem.end()) {
      HWEvents NewVGPRContext =
          Info.VGPRPendingEvents | It->second.VGPRPendingEvents;
      StrictDom |= NewVGPRContext != Info.VGPRPendingEvents;
      Info.VGPRPendingEvents = NewVGPRContext;
    }
  }

  StrictDom |= mergeAsyncMarks(MergeInfos, Other.AsyncMarks);
  for (auto T : inst_counter_types(Ctx->MaxCounter))
    StrictDom |= mergeScore(MergeInfos[T], AsyncScore[T], Other.AsyncScore[T]);

  purgeEmptyTrackingData();
  return StrictDom;
}

} // namespace AMDGPU
} // namespace llvm
