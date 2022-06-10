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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/TargetParser.h"
using namespace llvm;

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

namespace {
// Class of object that encapsulates latest instruction counter score
// associated with the operand.  Used for determining whether
// s_waitcnt instruction needs to be emitted.

#define CNT_MASK(t) (1u << (t))

enum InstCounterType { VM_CNT = 0, LGKM_CNT, EXP_CNT, VS_CNT, NUM_INST_CNTS };
} // namespace

namespace llvm {
template <> struct enum_iteration_traits<InstCounterType> {
  static constexpr bool is_iterable = true;
};
} // namespace llvm

namespace {
auto inst_counter_types() { return enum_seq(VM_CNT, NUM_INST_CNTS); }

using RegInterval = std::pair<int, int>;

struct HardwareLimits {
  unsigned VmcntMax;
  unsigned ExpcntMax;
  unsigned LgkmcntMax;
  unsigned VscntMax;
};

struct RegisterEncoding {
  unsigned VGPR0;
  unsigned VGPRL;
  unsigned SGPR0;
  unsigned SGPRL;
};

enum WaitEventType {
  VMEM_ACCESS,      // vector-memory read & write
  VMEM_READ_ACCESS, // vector-memory read
  VMEM_WRITE_ACCESS,// vector-memory write
  LDS_ACCESS,       // lds read & write
  GDS_ACCESS,       // gds read & write
  SQ_MESSAGE,       // send message
  SMEM_ACCESS,      // scalar-memory read & write
  EXP_GPR_LOCK,     // export holding on its data src
  GDS_GPR_LOCK,     // GDS holding on its data and addr src
  EXP_POS_ACCESS,   // write to export position
  EXP_PARAM_ACCESS, // write to export parameter
  VMW_GPR_LOCK,     // vector-memory write holding on its data src
  NUM_WAIT_EVENTS,
};

static const unsigned WaitEventMaskForInst[NUM_INST_CNTS] = {
  (1 << VMEM_ACCESS) | (1 << VMEM_READ_ACCESS),
  (1 << SMEM_ACCESS) | (1 << LDS_ACCESS) | (1 << GDS_ACCESS) |
      (1 << SQ_MESSAGE),
  (1 << EXP_GPR_LOCK) | (1 << GDS_GPR_LOCK) | (1 << VMW_GPR_LOCK) |
      (1 << EXP_PARAM_ACCESS) | (1 << EXP_POS_ACCESS),
  (1 << VMEM_WRITE_ACCESS)
};

// The mapping is:
//  0                .. SQ_MAX_PGM_VGPRS-1               real VGPRs
//  SQ_MAX_PGM_VGPRS .. NUM_ALL_VGPRS-1                  extra VGPR-like slots
//  NUM_ALL_VGPRS    .. NUM_ALL_VGPRS+SQ_MAX_PGM_SGPRS-1 real SGPRs
// We reserve a fixed number of VGPR slots in the scoring tables for
// special tokens like SCMEM_LDS (needed for buffer load to LDS).
enum RegisterMapping {
  SQ_MAX_PGM_VGPRS = 512, // Maximum programmable VGPRs across all targets.
  AGPR_OFFSET = 256,      // Maximum programmable ArchVGPRs across all targets.
  SQ_MAX_PGM_SGPRS = 256, // Maximum programmable SGPRs across all targets.
  NUM_EXTRA_VGPRS = 1,    // A reserved slot for DS.
  EXTRA_VGPR_LDS = 0,     // An artificial register to track LDS writes.
  NUM_ALL_VGPRS = SQ_MAX_PGM_VGPRS + NUM_EXTRA_VGPRS, // Where SGPR starts.
};

// Enumerate different types of result-returning VMEM operations. Although
// s_waitcnt orders them all with a single vmcnt counter, in the absence of
// s_waitcnt only instructions of the same VmemType are guaranteed to write
// their results in order -- so there is no need to insert an s_waitcnt between
// two instructions of the same type that write the same vgpr.
enum VmemType {
  // BUF instructions and MIMG instructions without a sampler.
  VMEM_NOSAMPLER,
  // MIMG instructions with a sampler.
  VMEM_SAMPLER,
  // BVH instructions
  VMEM_BVH
};

VmemType getVmemType(const MachineInstr &Inst) {
  assert(SIInstrInfo::isVMEM(Inst));
  if (!SIInstrInfo::isMIMG(Inst))
    return VMEM_NOSAMPLER;
  const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Inst.getOpcode());
  const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
      AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
  return BaseInfo->BVH ? VMEM_BVH
                       : BaseInfo->Sampler ? VMEM_SAMPLER : VMEM_NOSAMPLER;
}

void addWait(AMDGPU::Waitcnt &Wait, InstCounterType T, unsigned Count) {
  switch (T) {
  case VM_CNT:
    Wait.VmCnt = std::min(Wait.VmCnt, Count);
    break;
  case EXP_CNT:
    Wait.ExpCnt = std::min(Wait.ExpCnt, Count);
    break;
  case LGKM_CNT:
    Wait.LgkmCnt = std::min(Wait.LgkmCnt, Count);
    break;
  case VS_CNT:
    Wait.VsCnt = std::min(Wait.VsCnt, Count);
    break;
  default:
    llvm_unreachable("bad InstCounterType");
  }
}

// This objects maintains the current score brackets of each wait counter, and
// a per-register scoreboard for each wait counter.
//
// We also maintain the latest score for every event type that can change the
// waitcnt in order to know if there are multiple types of events within
// the brackets. When multiple types of event happen in the bracket,
// wait count may get decreased out of order, therefore we need to put in
// "s_waitcnt 0" before use.
class WaitcntBrackets {
public:
  WaitcntBrackets(const GCNSubtarget *SubTarget, HardwareLimits Limits,
                  RegisterEncoding Encoding)
      : ST(SubTarget), Limits(Limits), Encoding(Encoding) {}

  unsigned getWaitCountMax(InstCounterType T) const {
    switch (T) {
    case VM_CNT:
      return Limits.VmcntMax;
    case LGKM_CNT:
      return Limits.LgkmcntMax;
    case EXP_CNT:
      return Limits.ExpcntMax;
    case VS_CNT:
      return Limits.VscntMax;
    default:
      break;
    }
    return 0;
  }

  unsigned getScoreLB(InstCounterType T) const {
    assert(T < NUM_INST_CNTS);
    return ScoreLBs[T];
  }

  unsigned getScoreUB(InstCounterType T) const {
    assert(T < NUM_INST_CNTS);
    return ScoreUBs[T];
  }

  // Mapping from event to counter.
  InstCounterType eventCounter(WaitEventType E) {
    if (WaitEventMaskForInst[VM_CNT] & (1 << E))
      return VM_CNT;
    if (WaitEventMaskForInst[LGKM_CNT] & (1 << E))
      return LGKM_CNT;
    if (WaitEventMaskForInst[VS_CNT] & (1 << E))
      return VS_CNT;
    assert(WaitEventMaskForInst[EXP_CNT] & (1 << E));
    return EXP_CNT;
  }

  unsigned getRegScore(int GprNo, InstCounterType T) {
    if (GprNo < NUM_ALL_VGPRS) {
      return VgprScores[T][GprNo];
    }
    assert(T == LGKM_CNT);
    return SgprScores[GprNo - NUM_ALL_VGPRS];
  }

  bool merge(const WaitcntBrackets &Other);

  RegInterval getRegInterval(const MachineInstr *MI, const SIInstrInfo *TII,
                             const MachineRegisterInfo *MRI,
                             const SIRegisterInfo *TRI, unsigned OpNo) const;

  bool counterOutOfOrder(InstCounterType T) const;
  void simplifyWaitcnt(AMDGPU::Waitcnt &Wait) const;
  void simplifyWaitcnt(InstCounterType T, unsigned &Count) const;
  void determineWait(InstCounterType T, unsigned ScoreToWait,
                     AMDGPU::Waitcnt &Wait) const;
  void applyWaitcnt(const AMDGPU::Waitcnt &Wait);
  void applyWaitcnt(InstCounterType T, unsigned Count);
  void updateByEvent(const SIInstrInfo *TII, const SIRegisterInfo *TRI,
                     const MachineRegisterInfo *MRI, WaitEventType E,
                     MachineInstr &MI);

  bool hasPending() const { return PendingEvents != 0; }
  bool hasPendingEvent(WaitEventType E) const {
    return PendingEvents & (1 << E);
  }

  bool hasMixedPendingEvents(InstCounterType T) const {
    unsigned Events = PendingEvents & WaitEventMaskForInst[T];
    // Return true if more than one bit is set in Events.
    return Events & (Events - 1);
  }

  bool hasPendingFlat() const {
    return ((LastFlat[LGKM_CNT] > ScoreLBs[LGKM_CNT] &&
             LastFlat[LGKM_CNT] <= ScoreUBs[LGKM_CNT]) ||
            (LastFlat[VM_CNT] > ScoreLBs[VM_CNT] &&
             LastFlat[VM_CNT] <= ScoreUBs[VM_CNT]));
  }

  void setPendingFlat() {
    LastFlat[VM_CNT] = ScoreUBs[VM_CNT];
    LastFlat[LGKM_CNT] = ScoreUBs[LGKM_CNT];
  }

  // Return true if there might be pending writes to the specified vgpr by VMEM
  // instructions with types different from V.
  bool hasOtherPendingVmemTypes(int GprNo, VmemType V) const {
    assert(GprNo < NUM_ALL_VGPRS);
    return VgprVmemTypes[GprNo] & ~(1 << V);
  }

  void clearVgprVmemTypes(int GprNo) {
    assert(GprNo < NUM_ALL_VGPRS);
    VgprVmemTypes[GprNo] = 0;
  }

  void print(raw_ostream &);
  void dump() { print(dbgs()); }

private:
  struct MergeInfo {
    unsigned OldLB;
    unsigned OtherLB;
    unsigned MyShift;
    unsigned OtherShift;
  };
  static bool mergeScore(const MergeInfo &M, unsigned &Score,
                         unsigned OtherScore);

  void setScoreLB(InstCounterType T, unsigned Val) {
    assert(T < NUM_INST_CNTS);
    ScoreLBs[T] = Val;
  }

  void setScoreUB(InstCounterType T, unsigned Val) {
    assert(T < NUM_INST_CNTS);
    ScoreUBs[T] = Val;
    if (T == EXP_CNT) {
      unsigned UB = ScoreUBs[T] - getWaitCountMax(EXP_CNT);
      if (ScoreLBs[T] < UB && UB < ScoreUBs[T])
        ScoreLBs[T] = UB;
    }
  }

  void setRegScore(int GprNo, InstCounterType T, unsigned Val) {
    if (GprNo < NUM_ALL_VGPRS) {
      VgprUB = std::max(VgprUB, GprNo);
      VgprScores[T][GprNo] = Val;
    } else {
      assert(T == LGKM_CNT);
      SgprUB = std::max(SgprUB, GprNo - NUM_ALL_VGPRS);
      SgprScores[GprNo - NUM_ALL_VGPRS] = Val;
    }
  }

  void setExpScore(const MachineInstr *MI, const SIInstrInfo *TII,
                   const SIRegisterInfo *TRI, const MachineRegisterInfo *MRI,
                   unsigned OpNo, unsigned Val);

  const GCNSubtarget *ST = nullptr;
  HardwareLimits Limits = {};
  RegisterEncoding Encoding = {};
  unsigned ScoreLBs[NUM_INST_CNTS] = {0};
  unsigned ScoreUBs[NUM_INST_CNTS] = {0};
  unsigned PendingEvents = 0;
  // Remember the last flat memory operation.
  unsigned LastFlat[NUM_INST_CNTS] = {0};
  // wait_cnt scores for every vgpr.
  // Keep track of the VgprUB and SgprUB to make merge at join efficient.
  int VgprUB = -1;
  int SgprUB = -1;
  unsigned VgprScores[NUM_INST_CNTS][NUM_ALL_VGPRS] = {{0}};
  // Wait cnt scores for every sgpr, only lgkmcnt is relevant.
  unsigned SgprScores[SQ_MAX_PGM_SGPRS] = {0};
  // Bitmask of the VmemTypes of VMEM instructions that might have a pending
  // write to each vgpr.
  unsigned char VgprVmemTypes[NUM_ALL_VGPRS] = {0};
};

class SIInsertWaitcnts : public MachineFunctionPass {
private:
  const GCNSubtarget *ST = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  AMDGPU::IsaVersion IV;

  DenseSet<MachineInstr *> TrackedWaitcntSet;
  DenseMap<const Value *, MachineBasicBlock *> SLoadAddresses;
  MachinePostDominatorTree *PDT;

  struct BlockInfo {
    MachineBasicBlock *MBB;
    std::unique_ptr<WaitcntBrackets> Incoming;
    bool Dirty = true;

    explicit BlockInfo(MachineBasicBlock *MBB) : MBB(MBB) {}
  };

  MapVector<MachineBasicBlock *, BlockInfo> BlockInfos;

  // ForceEmitZeroWaitcnts: force all waitcnts insts to be s_waitcnt 0
  // because of amdgpu-waitcnt-forcezero flag
  bool ForceEmitZeroWaitcnts;
  bool ForceEmitWaitcnt[NUM_INST_CNTS];

public:
  static char ID;

  SIInsertWaitcnts() : MachineFunctionPass(ID) {
    (void)ForceExpCounter;
    (void)ForceLgkmCounter;
    (void)ForceVMCounter;
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI insert wait instructions";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachinePostDominatorTree>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool isForceEmitWaitcnt() const {
    for (auto T : inst_counter_types())
      if (ForceEmitWaitcnt[T])
        return true;
    return false;
  }

  void setForceEmitWaitcnt() {
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
      ForceEmitWaitcnt[LGKM_CNT] = true;
    } else {
      ForceEmitWaitcnt[LGKM_CNT] = false;
    }

    if (DebugCounter::isCounterSet(ForceVMCounter) &&
        DebugCounter::shouldExecute(ForceVMCounter)) {
      ForceEmitWaitcnt[VM_CNT] = true;
    } else {
      ForceEmitWaitcnt[VM_CNT] = false;
    }
#endif // NDEBUG
  }

  bool mayAccessVMEMThroughFlat(const MachineInstr &MI) const;
  bool mayAccessLDSThroughFlat(const MachineInstr &MI) const;
  bool generateWaitcntInstBefore(MachineInstr &MI,
                                 WaitcntBrackets &ScoreBrackets,
                                 MachineInstr *OldWaitcntInstr);
  void updateEventWaitcntAfter(MachineInstr &Inst,
                               WaitcntBrackets *ScoreBrackets);
  bool insertWaitcntInBlock(MachineFunction &MF, MachineBasicBlock &Block,
                            WaitcntBrackets &ScoreBrackets);
  bool applyPreexistingWaitcnt(WaitcntBrackets &ScoreBrackets,
                               MachineInstr &OldWaitcntInstr,
                               AMDGPU::Waitcnt &Wait, const MachineInstr *MI);
};

} // end anonymous namespace

RegInterval WaitcntBrackets::getRegInterval(const MachineInstr *MI,
                                            const SIInstrInfo *TII,
                                            const MachineRegisterInfo *MRI,
                                            const SIRegisterInfo *TRI,
                                            unsigned OpNo) const {
  const MachineOperand &Op = MI->getOperand(OpNo);
  if (!TRI->isInAllocatableClass(Op.getReg()))
    return {-1, -1};

  // A use via a PW operand does not need a waitcnt.
  // A partial write is not a WAW.
  assert(!Op.getSubReg() || !Op.isUndef());

  RegInterval Result;

  unsigned Reg = TRI->getEncodingValue(AMDGPU::getMCReg(Op.getReg(), *ST));

  if (TRI->isVectorRegister(*MRI, Op.getReg())) {
    assert(Reg >= Encoding.VGPR0 && Reg <= Encoding.VGPRL);
    Result.first = Reg - Encoding.VGPR0;
    if (TRI->isAGPR(*MRI, Op.getReg()))
      Result.first += AGPR_OFFSET;
    assert(Result.first >= 0 && Result.first < SQ_MAX_PGM_VGPRS);
  } else if (TRI->isSGPRReg(*MRI, Op.getReg())) {
    assert(Reg >= Encoding.SGPR0 && Reg < SQ_MAX_PGM_SGPRS);
    Result.first = Reg - Encoding.SGPR0 + NUM_ALL_VGPRS;
    assert(Result.first >= NUM_ALL_VGPRS &&
           Result.first < SQ_MAX_PGM_SGPRS + NUM_ALL_VGPRS);
  }
  // TODO: Handle TTMP
  // else if (TRI->isTTMP(*MRI, Reg.getReg())) ...
  else
    return {-1, -1};

  const TargetRegisterClass *RC = TII->getOpRegClass(*MI, OpNo);
  unsigned Size = TRI->getRegSizeInBits(*RC);
  Result.second = Result.first + ((Size + 16) / 32);

  return Result;
}

void WaitcntBrackets::setExpScore(const MachineInstr *MI,
                                  const SIInstrInfo *TII,
                                  const SIRegisterInfo *TRI,
                                  const MachineRegisterInfo *MRI, unsigned OpNo,
                                  unsigned Val) {
  RegInterval Interval = getRegInterval(MI, TII, MRI, TRI, OpNo);
  assert(TRI->isVectorRegister(*MRI, MI->getOperand(OpNo).getReg()));
  for (int RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
    setRegScore(RegNo, EXP_CNT, Val);
  }
}

// MUBUF and FLAT LDS DMA operations need a wait on vmcnt before LDS written
// can be accessed. A load from LDS to VMEM does not need a wait.
static bool mayWriteLDSThroughDMA(const MachineInstr &MI) {
  return SIInstrInfo::isVALU(MI) &&
         (SIInstrInfo::isMUBUF(MI) || SIInstrInfo::isFLAT(MI)) &&
         MI.getOpcode() != AMDGPU::BUFFER_STORE_LDS_DWORD;
}

void WaitcntBrackets::updateByEvent(const SIInstrInfo *TII,
                                    const SIRegisterInfo *TRI,
                                    const MachineRegisterInfo *MRI,
                                    WaitEventType E, MachineInstr &Inst) {
  InstCounterType T = eventCounter(E);
  unsigned CurrScore = getScoreUB(T) + 1;
  if (CurrScore == 0)
    report_fatal_error("InsertWaitcnt score wraparound");
  // PendingEvents and ScoreUB need to be update regardless if this event
  // changes the score of a register or not.
  // Examples including vm_cnt when buffer-store or lgkm_cnt when send-message.
  PendingEvents |= 1 << E;
  setScoreUB(T, CurrScore);

  if (T == EXP_CNT) {
    // Put score on the source vgprs. If this is a store, just use those
    // specific register(s).
    if (TII->isDS(Inst) && (Inst.mayStore() || Inst.mayLoad())) {
      int AddrOpIdx =
          AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::addr);
      // All GDS operations must protect their address register (same as
      // export.)
      if (AddrOpIdx != -1) {
        setExpScore(&Inst, TII, TRI, MRI, AddrOpIdx, CurrScore);
      }

      if (Inst.mayStore()) {
        if (AMDGPU::getNamedOperandIdx(Inst.getOpcode(),
                                       AMDGPU::OpName::data0) != -1) {
          setExpScore(
              &Inst, TII, TRI, MRI,
              AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data0),
              CurrScore);
        }
        if (AMDGPU::getNamedOperandIdx(Inst.getOpcode(),
                                       AMDGPU::OpName::data1) != -1) {
          setExpScore(&Inst, TII, TRI, MRI,
                      AMDGPU::getNamedOperandIdx(Inst.getOpcode(),
                                                 AMDGPU::OpName::data1),
                      CurrScore);
        }
      } else if (SIInstrInfo::isAtomicRet(Inst) &&
                 Inst.getOpcode() != AMDGPU::DS_GWS_INIT &&
                 Inst.getOpcode() != AMDGPU::DS_GWS_SEMA_V &&
                 Inst.getOpcode() != AMDGPU::DS_GWS_SEMA_BR &&
                 Inst.getOpcode() != AMDGPU::DS_GWS_SEMA_P &&
                 Inst.getOpcode() != AMDGPU::DS_GWS_BARRIER &&
                 Inst.getOpcode() != AMDGPU::DS_APPEND &&
                 Inst.getOpcode() != AMDGPU::DS_CONSUME &&
                 Inst.getOpcode() != AMDGPU::DS_ORDERED_COUNT) {
        for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
          const MachineOperand &Op = Inst.getOperand(I);
          if (Op.isReg() && !Op.isDef() &&
              TRI->isVectorRegister(*MRI, Op.getReg())) {
            setExpScore(&Inst, TII, TRI, MRI, I, CurrScore);
          }
        }
      }
    } else if (TII->isFLAT(Inst)) {
      if (Inst.mayStore()) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data),
            CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data),
            CurrScore);
      }
    } else if (TII->isMIMG(Inst)) {
      if (Inst.mayStore()) {
        setExpScore(&Inst, TII, TRI, MRI, 0, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data),
            CurrScore);
      }
    } else if (TII->isMTBUF(Inst)) {
      if (Inst.mayStore()) {
        setExpScore(&Inst, TII, TRI, MRI, 0, CurrScore);
      }
    } else if (TII->isMUBUF(Inst)) {
      if (Inst.mayStore()) {
        setExpScore(&Inst, TII, TRI, MRI, 0, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data),
            CurrScore);
      }
    } else {
      if (TII->isEXP(Inst)) {
        // For export the destination registers are really temps that
        // can be used as the actual source after export patching, so
        // we need to treat them like sources and set the EXP_CNT
        // score.
        for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
          MachineOperand &DefMO = Inst.getOperand(I);
          if (DefMO.isReg() && DefMO.isDef() &&
              TRI->isVGPR(*MRI, DefMO.getReg())) {
            setRegScore(
                TRI->getEncodingValue(AMDGPU::getMCReg(DefMO.getReg(), *ST)),
                EXP_CNT, CurrScore);
          }
        }
      }
      for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
        MachineOperand &MO = Inst.getOperand(I);
        if (MO.isReg() && !MO.isDef() &&
            TRI->isVectorRegister(*MRI, MO.getReg())) {
          setExpScore(&Inst, TII, TRI, MRI, I, CurrScore);
        }
      }
    }
#if 0 // TODO: check if this is handled by MUBUF code above.
  } else if (Inst.getOpcode() == AMDGPU::BUFFER_STORE_DWORD ||
       Inst.getOpcode() == AMDGPU::BUFFER_STORE_DWORDX2 ||
       Inst.getOpcode() == AMDGPU::BUFFER_STORE_DWORDX4) {
    MachineOperand *MO = TII->getNamedOperand(Inst, AMDGPU::OpName::data);
    unsigned OpNo;//TODO: find the OpNo for this operand;
    RegInterval Interval = getRegInterval(&Inst, TII, MRI, TRI, OpNo);
    for (int RegNo = Interval.first; RegNo < Interval.second;
    ++RegNo) {
      setRegScore(RegNo + NUM_ALL_VGPRS, t, CurrScore);
    }
#endif
  } else {
    // Match the score to the destination registers.
    for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
      auto &Op = Inst.getOperand(I);
      if (!Op.isReg() || !Op.isDef())
        continue;
      RegInterval Interval = getRegInterval(&Inst, TII, MRI, TRI, I);
      if (T == VM_CNT) {
        if (Interval.first >= NUM_ALL_VGPRS)
          continue;
        if (SIInstrInfo::isVMEM(Inst)) {
          VmemType V = getVmemType(Inst);
          for (int RegNo = Interval.first; RegNo < Interval.second; ++RegNo)
            VgprVmemTypes[RegNo] |= 1 << V;
        }
      }
      for (int RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
        setRegScore(RegNo, T, CurrScore);
      }
    }
    if (Inst.mayStore() && (TII->isDS(Inst) || mayWriteLDSThroughDMA(Inst))) {
      setRegScore(SQ_MAX_PGM_VGPRS + EXTRA_VGPR_LDS, T, CurrScore);
    }
  }
}

void WaitcntBrackets::print(raw_ostream &OS) {
  OS << '\n';
  for (auto T : inst_counter_types()) {
    unsigned LB = getScoreLB(T);
    unsigned UB = getScoreUB(T);

    switch (T) {
    case VM_CNT:
      OS << "    VM_CNT(" << UB - LB << "): ";
      break;
    case LGKM_CNT:
      OS << "    LGKM_CNT(" << UB - LB << "): ";
      break;
    case EXP_CNT:
      OS << "    EXP_CNT(" << UB - LB << "): ";
      break;
    case VS_CNT:
      OS << "    VS_CNT(" << UB - LB << "): ";
      break;
    default:
      OS << "    UNKNOWN(" << UB - LB << "): ";
      break;
    }

    if (LB < UB) {
      // Print vgpr scores.
      for (int J = 0; J <= VgprUB; J++) {
        unsigned RegScore = getRegScore(J, T);
        if (RegScore <= LB)
          continue;
        unsigned RelScore = RegScore - LB - 1;
        if (J < SQ_MAX_PGM_VGPRS + EXTRA_VGPR_LDS) {
          OS << RelScore << ":v" << J << " ";
        } else {
          OS << RelScore << ":ds ";
        }
      }
      // Also need to print sgpr scores for lgkm_cnt.
      if (T == LGKM_CNT) {
        for (int J = 0; J <= SgprUB; J++) {
          unsigned RegScore = getRegScore(J + NUM_ALL_VGPRS, LGKM_CNT);
          if (RegScore <= LB)
            continue;
          unsigned RelScore = RegScore - LB - 1;
          OS << RelScore << ":s" << J << " ";
        }
      }
    }
    OS << '\n';
  }
  OS << '\n';
}

/// Simplify the waitcnt, in the sense of removing redundant counts, and return
/// whether a waitcnt instruction is needed at all.
void WaitcntBrackets::simplifyWaitcnt(AMDGPU::Waitcnt &Wait) const {
  simplifyWaitcnt(VM_CNT, Wait.VmCnt);
  simplifyWaitcnt(EXP_CNT, Wait.ExpCnt);
  simplifyWaitcnt(LGKM_CNT, Wait.LgkmCnt);
  simplifyWaitcnt(VS_CNT, Wait.VsCnt);
}

void WaitcntBrackets::simplifyWaitcnt(InstCounterType T,
                                      unsigned &Count) const {
  const unsigned LB = getScoreLB(T);
  const unsigned UB = getScoreUB(T);

  // The number of outstanding events for this type, T, can be calculated
  // as (UB - LB). If the current Count is greater than or equal to the number
  // of outstanding events, then the wait for this counter is redundant.
  if (Count >= UB - LB)
    Count = ~0u;
}

void WaitcntBrackets::determineWait(InstCounterType T, unsigned ScoreToWait,
                                    AMDGPU::Waitcnt &Wait) const {
  // If the score of src_operand falls within the bracket, we need an
  // s_waitcnt instruction.
  const unsigned LB = getScoreLB(T);
  const unsigned UB = getScoreUB(T);
  if ((UB >= ScoreToWait) && (ScoreToWait > LB)) {
    if ((T == VM_CNT || T == LGKM_CNT) &&
        hasPendingFlat() &&
        !ST->hasFlatLgkmVMemCountInOrder()) {
      // If there is a pending FLAT operation, and this is a VMem or LGKM
      // waitcnt and the target can report early completion, then we need
      // to force a waitcnt 0.
      addWait(Wait, T, 0);
    } else if (counterOutOfOrder(T)) {
      // Counter can get decremented out-of-order when there
      // are multiple types event in the bracket. Also emit an s_wait counter
      // with a conservative value of 0 for the counter.
      addWait(Wait, T, 0);
    } else {
      // If a counter has been maxed out avoid overflow by waiting for
      // MAX(CounterType) - 1 instead.
      unsigned NeededWait = std::min(UB - ScoreToWait, getWaitCountMax(T) - 1);
      addWait(Wait, T, NeededWait);
    }
  }
}

void WaitcntBrackets::applyWaitcnt(const AMDGPU::Waitcnt &Wait) {
  applyWaitcnt(VM_CNT, Wait.VmCnt);
  applyWaitcnt(EXP_CNT, Wait.ExpCnt);
  applyWaitcnt(LGKM_CNT, Wait.LgkmCnt);
  applyWaitcnt(VS_CNT, Wait.VsCnt);
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
    PendingEvents &= ~WaitEventMaskForInst[T];
  }
}

// Where there are multiple types of event in the bracket of a counter,
// the decrement may go out of order.
bool WaitcntBrackets::counterOutOfOrder(InstCounterType T) const {
  // Scalar memory read always can go out of order.
  if (T == LGKM_CNT && hasPendingEvent(SMEM_ACCESS))
    return true;
  return hasMixedPendingEvents(T);
}

INITIALIZE_PASS_BEGIN(SIInsertWaitcnts, DEBUG_TYPE, "SI Insert Waitcnts", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_END(SIInsertWaitcnts, DEBUG_TYPE, "SI Insert Waitcnts", false,
                    false)

char SIInsertWaitcnts::ID = 0;

char &llvm::SIInsertWaitcntsID = SIInsertWaitcnts::ID;

FunctionPass *llvm::createSIInsertWaitcntsPass() {
  return new SIInsertWaitcnts();
}

/// Combine consecutive waitcnt instructions that precede \p MI and follow
/// \p OldWaitcntInstr and apply any extra wait from waitcnt that were added
/// by previous passes. Currently this pass conservatively assumes that these
/// preexisting waitcnt are required for correctness.
bool SIInsertWaitcnts::applyPreexistingWaitcnt(WaitcntBrackets &ScoreBrackets,
                                               MachineInstr &OldWaitcntInstr,
                                               AMDGPU::Waitcnt &Wait,
                                               const MachineInstr *MI) {
  bool Modified = false;
  MachineInstr *WaitcntInstr = nullptr;
  MachineInstr *WaitcntVsCntInstr = nullptr;
  for (auto II = OldWaitcntInstr.getIterator(), NextI = std::next(II);
       &*II != MI; II = NextI, ++NextI) {
    if (II->isMetaInstruction())
      continue;

    if (II->getOpcode() == AMDGPU::S_WAITCNT) {
      // Conservatively update required wait if this waitcnt was added in an
      // earlier pass. In this case it will not exist in the tracked waitcnt
      // set.
      if (!TrackedWaitcntSet.count(&*II)) {
        unsigned IEnc = II->getOperand(0).getImm();
        AMDGPU::Waitcnt OldWait = AMDGPU::decodeWaitcnt(IV, IEnc);
        Wait = Wait.combined(OldWait);
      }

      // Merge consecutive waitcnt of the same type by erasing multiples.
      if (!WaitcntInstr) {
        WaitcntInstr = &*II;
      } else {
        II->eraseFromParent();
        Modified = true;
      }

    } else {
      assert(II->getOpcode() == AMDGPU::S_WAITCNT_VSCNT);
      assert(II->getOperand(0).getReg() == AMDGPU::SGPR_NULL);
      if (!TrackedWaitcntSet.count(&*II)) {
        unsigned OldVSCnt =
            TII->getNamedOperand(*II, AMDGPU::OpName::simm16)->getImm();
        Wait.VsCnt = std::min(Wait.VsCnt, OldVSCnt);
      }

      if (!WaitcntVsCntInstr) {
        WaitcntVsCntInstr = &*II;
      } else {
        II->eraseFromParent();
        Modified = true;
      }
    }
  }

  // Updated encoding of merged waitcnt with the required wait.
  if (WaitcntInstr) {
    if (Wait.hasWaitExceptVsCnt()) {
      unsigned NewEnc = AMDGPU::encodeWaitcnt(IV, Wait);
      unsigned OldEnc = WaitcntInstr->getOperand(0).getImm();
      if (OldEnc != NewEnc) {
        WaitcntInstr->getOperand(0).setImm(NewEnc);
        Modified = true;
      }
      ScoreBrackets.applyWaitcnt(Wait);
      Wait.VmCnt = ~0u;
      Wait.LgkmCnt = ~0u;
      Wait.ExpCnt = ~0u;

      LLVM_DEBUG(dbgs() << "generateWaitcntInstBefore\n"
                        << "Old Instr: " << *MI << "New Instr: " << *WaitcntInstr
                        << '\n');
    } else {
      WaitcntInstr->eraseFromParent();
      Modified = true;
    }
  }

  if (WaitcntVsCntInstr) {
    if (Wait.hasWaitVsCnt()) {
      assert(ST->hasVscnt());
      unsigned OldVSCnt =
          TII->getNamedOperand(*WaitcntVsCntInstr, AMDGPU::OpName::simm16)
              ->getImm();
      if (Wait.VsCnt != OldVSCnt) {
        TII->getNamedOperand(*WaitcntVsCntInstr, AMDGPU::OpName::simm16)
            ->setImm(Wait.VsCnt);
        Modified = true;
      }
      ScoreBrackets.applyWaitcnt(Wait);
      Wait.VsCnt = ~0u;

      LLVM_DEBUG(dbgs() << "generateWaitcntInstBefore\n"
                        << "Old Instr: " << *MI
                        << "New Instr: " << *WaitcntVsCntInstr << '\n');
    } else {
      WaitcntVsCntInstr->eraseFromParent();
      Modified = true;
    }
  }

  return Modified;
}

static bool readsVCCZ(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return (Opc == AMDGPU::S_CBRANCH_VCCNZ || Opc == AMDGPU::S_CBRANCH_VCCZ) &&
         !MI.getOperand(1).isUndef();
}

/// \returns true if the callee inserts an s_waitcnt 0 on function entry.
static bool callWaitsOnFunctionEntry(const MachineInstr &MI) {
  // Currently all conventions wait, but this may not always be the case.
  //
  // TODO: If IPRA is enabled, and the callee is isSafeForNoCSROpt, it may make
  // senses to omit the wait and do it in the caller.
  return true;
}

/// \returns true if the callee is expected to wait for any outstanding waits
/// before returning.
static bool callWaitsOnFunctionReturn(const MachineInstr &MI) {
  return true;
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
bool SIInsertWaitcnts::generateWaitcntInstBefore(
    MachineInstr &MI, WaitcntBrackets &ScoreBrackets,
    MachineInstr *OldWaitcntInstr) {
  setForceEmitWaitcnt();

  if (MI.isMetaInstruction())
    return false;

  AMDGPU::Waitcnt Wait;
  bool Modified = false;

  // FIXME: This should have already been handled by the memory legalizer.
  // Removing this currently doesn't affect any lit tests, but we need to
  // verify that nothing was relying on this. The number of buffer invalidates
  // being handled here should not be expanded.
  if (MI.getOpcode() == AMDGPU::BUFFER_WBINVL1 ||
      MI.getOpcode() == AMDGPU::BUFFER_WBINVL1_SC ||
      MI.getOpcode() == AMDGPU::BUFFER_WBINVL1_VOL ||
      MI.getOpcode() == AMDGPU::BUFFER_GL0_INV ||
      MI.getOpcode() == AMDGPU::BUFFER_GL1_INV) {
    Wait.VmCnt = 0;
  }

  // All waits must be resolved at call return.
  // NOTE: this could be improved with knowledge of all call sites or
  //   with knowledge of the called routines.
  if (MI.getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG ||
      MI.getOpcode() == AMDGPU::SI_RETURN ||
      MI.getOpcode() == AMDGPU::S_SETPC_B64_return ||
      (MI.isReturn() && MI.isCall() && !callWaitsOnFunctionEntry(MI))) {
    Wait = Wait.combined(AMDGPU::Waitcnt::allZero(ST->hasVscnt()));
  }
  // Resolve vm waits before gs-done.
  else if ((MI.getOpcode() == AMDGPU::S_SENDMSG ||
            MI.getOpcode() == AMDGPU::S_SENDMSGHALT) &&
           ST->hasLegacyGeometry() &&
           ((MI.getOperand(0).getImm() & AMDGPU::SendMsg::ID_MASK_PreGFX11_) ==
            AMDGPU::SendMsg::ID_GS_DONE_PreGFX11)) {
    Wait.VmCnt = 0;
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
            EmitWaitcnt |= ScoreBrackets->updateByWait(LGKM_CNT,
                               ScoreBrackets->getScoreUB(LGKM_CNT));
            // LDS may have to wait for VM_CNT after buffer load to LDS
            if (target_info->HasBufferLoadToLDS()) {
              EmitWaitcnt |= ScoreBrackets->updateByWait(VM_CNT,
                                 ScoreBrackets->getScoreUB(VM_CNT));
            }
          }
          break;

        case SCMEM_GDS:
          if (group_is_multi_wave || fence_is_global) {
            EmitWaitcnt |= ScoreBrackets->updateByWait(EXP_CNT,
              ScoreBrackets->getScoreUB(EXP_CNT));
            EmitWaitcnt |= ScoreBrackets->updateByWait(LGKM_CNT,
              ScoreBrackets->getScoreUB(LGKM_CNT));
          }
          break;

        case SCMEM_UAV:
        case SCMEM_TFBUF:
        case SCMEM_RING:
        case SCMEM_SCATTER:
          if (group_is_multi_wave || fence_is_global) {
            EmitWaitcnt |= ScoreBrackets->updateByWait(EXP_CNT,
              ScoreBrackets->getScoreUB(EXP_CNT));
            EmitWaitcnt |= ScoreBrackets->updateByWait(VM_CNT,
              ScoreBrackets->getScoreUB(VM_CNT));
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
        RegInterval CallAddrOpInterval =
          ScoreBrackets.getRegInterval(&MI, TII, MRI, TRI, CallAddrOpIdx);

        for (int RegNo = CallAddrOpInterval.first;
             RegNo < CallAddrOpInterval.second; ++RegNo)
          ScoreBrackets.determineWait(
            LGKM_CNT, ScoreBrackets.getRegScore(RegNo, LGKM_CNT), Wait);

        int RtnAddrOpIdx =
          AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::dst);
        if (RtnAddrOpIdx != -1) {
          RegInterval RtnAddrOpInterval =
            ScoreBrackets.getRegInterval(&MI, TII, MRI, TRI, RtnAddrOpIdx);

          for (int RegNo = RtnAddrOpInterval.first;
               RegNo < RtnAddrOpInterval.second; ++RegNo)
            ScoreBrackets.determineWait(
              LGKM_CNT, ScoreBrackets.getRegScore(RegNo, LGKM_CNT), Wait);
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
          addWait(Wait, LGKM_CNT, 0);
          if (PDT->dominates(MI.getParent(), SLoadAddresses.find(Ptr)->second))
            SLoadAddresses.erase(Ptr);
        }
        unsigned AS = Memop->getAddrSpace();
        if (AS != AMDGPUAS::LOCAL_ADDRESS && AS != AMDGPUAS::FLAT_ADDRESS)
          continue;
        // No need to wait before load from VMEM to LDS.
        if (mayWriteLDSThroughDMA(MI))
          continue;
        unsigned RegNo = SQ_MAX_PGM_VGPRS + EXTRA_VGPR_LDS;
        // VM_CNT is only relevant to vgpr or LDS.
        ScoreBrackets.determineWait(
            VM_CNT, ScoreBrackets.getRegScore(RegNo, VM_CNT), Wait);
        if (Memop->isStore()) {
          ScoreBrackets.determineWait(
              EXP_CNT, ScoreBrackets.getRegScore(RegNo, EXP_CNT), Wait);
        }
      }

      // Loop over use and def operands.
      for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
        MachineOperand &Op = MI.getOperand(I);
        if (!Op.isReg())
          continue;
        RegInterval Interval =
            ScoreBrackets.getRegInterval(&MI, TII, MRI, TRI, I);

        const bool IsVGPR = TRI->isVectorRegister(*MRI, Op.getReg());
        for (int RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
          if (IsVGPR) {
            // RAW always needs an s_waitcnt. WAW needs an s_waitcnt unless the
            // previous write and this write are the same type of VMEM
            // instruction, in which case they're guaranteed to write their
            // results in order anyway.
            if (Op.isUse() || !SIInstrInfo::isVMEM(MI) ||
                ScoreBrackets.hasOtherPendingVmemTypes(RegNo,
                                                       getVmemType(MI))) {
              ScoreBrackets.determineWait(
                  VM_CNT, ScoreBrackets.getRegScore(RegNo, VM_CNT), Wait);
              ScoreBrackets.clearVgprVmemTypes(RegNo);
            }
            if (Op.isDef()) {
              ScoreBrackets.determineWait(
                  EXP_CNT, ScoreBrackets.getRegScore(RegNo, EXP_CNT), Wait);
            }
          }
          ScoreBrackets.determineWait(
              LGKM_CNT, ScoreBrackets.getRegScore(RegNo, LGKM_CNT), Wait);
        }
      }
    }
  }

  // Check to see if this is an S_BARRIER, and if an implicit S_WAITCNT 0
  // occurs before the instruction. Doing it here prevents any additional
  // S_WAITCNTs from being emitted if the instruction was marked as
  // requiring a WAITCNT beforehand.
  if (MI.getOpcode() == AMDGPU::S_BARRIER &&
      !ST->hasAutoWaitcntBeforeBarrier()) {
    Wait = Wait.combined(AMDGPU::Waitcnt::allZero(ST->hasVscnt()));
  }

  // TODO: Remove this work-around, enable the assert for Bug 457939
  //       after fixing the scheduler. Also, the Shader Compiler code is
  //       independent of target.
  if (readsVCCZ(MI) && ST->hasReadVCCZBug()) {
    if (ScoreBrackets.getScoreLB(LGKM_CNT) <
            ScoreBrackets.getScoreUB(LGKM_CNT) &&
        ScoreBrackets.hasPendingEvent(SMEM_ACCESS)) {
      Wait.LgkmCnt = 0;
    }
  }

  // Verify that the wait is actually needed.
  ScoreBrackets.simplifyWaitcnt(Wait);

  if (ForceEmitZeroWaitcnts)
    Wait = AMDGPU::Waitcnt::allZero(ST->hasVscnt());

  if (ForceEmitWaitcnt[VM_CNT])
    Wait.VmCnt = 0;
  if (ForceEmitWaitcnt[EXP_CNT])
    Wait.ExpCnt = 0;
  if (ForceEmitWaitcnt[LGKM_CNT])
    Wait.LgkmCnt = 0;
  if (ForceEmitWaitcnt[VS_CNT])
    Wait.VsCnt = 0;

  if (OldWaitcntInstr) {
    // Try to merge the required wait with preexisting waitcnt instructions.
    // Also erase redundant waitcnt.
    Modified =
        applyPreexistingWaitcnt(ScoreBrackets, *OldWaitcntInstr, Wait, &MI);
  } else {
    // Update waitcnt brackets after determining the required wait.
    ScoreBrackets.applyWaitcnt(Wait);
  }

  // Build new waitcnt instructions unless no wait is needed or the old waitcnt
  // instruction was modified to handle the required wait.
  if (Wait.hasWaitExceptVsCnt()) {
    unsigned Enc = AMDGPU::encodeWaitcnt(IV, Wait);
    auto SWaitInst = BuildMI(*MI.getParent(), MI.getIterator(),
                             MI.getDebugLoc(), TII->get(AMDGPU::S_WAITCNT))
                         .addImm(Enc);
    TrackedWaitcntSet.insert(SWaitInst);
    Modified = true;

    LLVM_DEBUG(dbgs() << "generateWaitcntInstBefore\n"
                      << "Old Instr: " << MI
                      << "New Instr: " << *SWaitInst << '\n');
  }

  if (Wait.hasWaitVsCnt()) {
    assert(ST->hasVscnt());

    auto SWaitInst =
        BuildMI(*MI.getParent(), MI.getIterator(), MI.getDebugLoc(),
                TII->get(AMDGPU::S_WAITCNT_VSCNT))
            .addReg(AMDGPU::SGPR_NULL, RegState::Undef)
            .addImm(Wait.VsCnt);
    TrackedWaitcntSet.insert(SWaitInst);
    Modified = true;

    LLVM_DEBUG(dbgs() << "generateWaitcntInstBefore\n"
                      << "Old Instr: " << MI
                      << "New Instr: " << *SWaitInst << '\n');
  }

  return Modified;
}

// This is a flat memory operation. Check to see if it has memory tokens other
// than LDS. Other address spaces supported by flat memory operations involve
// global memory.
bool SIInsertWaitcnts::mayAccessVMEMThroughFlat(const MachineInstr &MI) const {
  assert(TII->isFLAT(MI));

  // All flat instructions use the VMEM counter.
  assert(TII->usesVM_CNT(MI));

  // If there are no memory operands then conservatively assume the flat
  // operation may access VMEM.
  if (MI.memoperands_empty())
    return true;

  // See if any memory operand specifies an address space that involves VMEM.
  // Flat operations only supported FLAT, LOCAL (LDS), or address spaces
  // involving VMEM such as GLOBAL, CONSTANT, PRIVATE (SCRATCH), etc. The REGION
  // (GDS) address space is not supported by flat operations. Therefore, simply
  // return true unless only the LDS address space is found.
  for (const MachineMemOperand *Memop : MI.memoperands()) {
    unsigned AS = Memop->getAddrSpace();
    assert(AS != AMDGPUAS::REGION_ADDRESS);
    if (AS != AMDGPUAS::LOCAL_ADDRESS)
      return true;
  }

  return false;
}

// This is a flat memory operation. Check to see if it has memory tokens for
// either LDS or FLAT.
bool SIInsertWaitcnts::mayAccessLDSThroughFlat(const MachineInstr &MI) const {
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

void SIInsertWaitcnts::updateEventWaitcntAfter(MachineInstr &Inst,
                                               WaitcntBrackets *ScoreBrackets) {
  // Now look at the instruction opcode. If it is a memory access
  // instruction, update the upper-bound of the appropriate counter's
  // bracket and the destination operand scores.
  // TODO: Use the (TSFlags & SIInstrFlags::LGKM_CNT) property everywhere.
  if (TII->isDS(Inst) && TII->usesLGKM_CNT(Inst)) {
    if (TII->isAlwaysGDS(Inst.getOpcode()) ||
        TII->hasModifiersSet(Inst, AMDGPU::OpName::gds)) {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, GDS_ACCESS, Inst);
      ScoreBrackets->updateByEvent(TII, TRI, MRI, GDS_GPR_LOCK, Inst);
    } else {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, LDS_ACCESS, Inst);
    }
  } else if (TII->isFLAT(Inst)) {
    assert(Inst.mayLoadOrStore());

    int FlatASCount = 0;

    if (mayAccessVMEMThroughFlat(Inst)) {
      ++FlatASCount;
      if (!ST->hasVscnt())
        ScoreBrackets->updateByEvent(TII, TRI, MRI, VMEM_ACCESS, Inst);
      else if (Inst.mayLoad() && !SIInstrInfo::isAtomicNoRet(Inst))
        ScoreBrackets->updateByEvent(TII, TRI, MRI, VMEM_READ_ACCESS, Inst);
      else
        ScoreBrackets->updateByEvent(TII, TRI, MRI, VMEM_WRITE_ACCESS, Inst);
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
    if (!ST->hasVscnt())
      ScoreBrackets->updateByEvent(TII, TRI, MRI, VMEM_ACCESS, Inst);
    else if ((Inst.mayLoad() && !SIInstrInfo::isAtomicNoRet(Inst)) ||
             /* IMAGE_GET_RESINFO / IMAGE_GET_LOD */
             (TII->isMIMG(Inst) && !Inst.mayLoad() && !Inst.mayStore()))
      ScoreBrackets->updateByEvent(TII, TRI, MRI, VMEM_READ_ACCESS, Inst);
    else if (Inst.mayStore())
      ScoreBrackets->updateByEvent(TII, TRI, MRI, VMEM_WRITE_ACCESS, Inst);

    if (ST->vmemWriteNeedsExpWaitcnt() &&
        (Inst.mayStore() || SIInstrInfo::isAtomicRet(Inst))) {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, VMW_GPR_LOCK, Inst);
    }
  } else if (TII->isSMRD(Inst)) {
    ScoreBrackets->updateByEvent(TII, TRI, MRI, SMEM_ACCESS, Inst);
  } else if (Inst.isCall()) {
    if (callWaitsOnFunctionReturn(Inst)) {
      // Act as a wait on everything
      ScoreBrackets->applyWaitcnt(AMDGPU::Waitcnt::allZero(ST->hasVscnt()));
    } else {
      // May need to way wait for anything.
      ScoreBrackets->applyWaitcnt(AMDGPU::Waitcnt());
    }
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
      ScoreBrackets->updateByEvent(TII, TRI, MRI, SMEM_ACCESS, Inst);
      break;
    }
  }
}

bool WaitcntBrackets::mergeScore(const MergeInfo &M, unsigned &Score,
                                 unsigned OtherScore) {
  unsigned MyShifted = Score <= M.OldLB ? 0 : Score + M.MyShift;
  unsigned OtherShifted =
      OtherScore <= M.OtherLB ? 0 : OtherScore + M.OtherShift;
  Score = std::max(MyShifted, OtherShifted);
  return OtherShifted > MyShifted;
}

/// Merge the pending events and associater score brackets of \p Other into
/// this brackets status.
///
/// Returns whether the merge resulted in a change that requires tighter waits
/// (i.e. the merged brackets strictly dominate the original brackets).
bool WaitcntBrackets::merge(const WaitcntBrackets &Other) {
  bool StrictDom = false;

  VgprUB = std::max(VgprUB, Other.VgprUB);
  SgprUB = std::max(SgprUB, Other.SgprUB);

  for (auto T : inst_counter_types()) {
    // Merge event flags for this counter
    const unsigned OldEvents = PendingEvents & WaitEventMaskForInst[T];
    const unsigned OtherEvents = Other.PendingEvents & WaitEventMaskForInst[T];
    if (OtherEvents & ~OldEvents)
      StrictDom = true;
    PendingEvents |= OtherEvents;

    // Merge scores for this counter
    const unsigned MyPending = ScoreUBs[T] - ScoreLBs[T];
    const unsigned OtherPending = Other.ScoreUBs[T] - Other.ScoreLBs[T];
    const unsigned NewUB = ScoreLBs[T] + std::max(MyPending, OtherPending);
    if (NewUB < ScoreLBs[T])
      report_fatal_error("waitcnt score overflow");

    MergeInfo M;
    M.OldLB = ScoreLBs[T];
    M.OtherLB = Other.ScoreLBs[T];
    M.MyShift = NewUB - ScoreUBs[T];
    M.OtherShift = NewUB - Other.ScoreUBs[T];

    ScoreUBs[T] = NewUB;

    StrictDom |= mergeScore(M, LastFlat[T], Other.LastFlat[T]);

    bool RegStrictDom = false;
    for (int J = 0; J <= VgprUB; J++) {
      RegStrictDom |= mergeScore(M, VgprScores[T][J], Other.VgprScores[T][J]);
    }

    if (T == VM_CNT) {
      for (int J = 0; J <= VgprUB; J++) {
        unsigned char NewVmemTypes = VgprVmemTypes[J] | Other.VgprVmemTypes[J];
        RegStrictDom |= NewVmemTypes != VgprVmemTypes[J];
        VgprVmemTypes[J] = NewVmemTypes;
      }
    }

    if (T == LGKM_CNT) {
      for (int J = 0; J <= SgprUB; J++) {
        RegStrictDom |= mergeScore(M, SgprScores[J], Other.SgprScores[J]);
      }
    }

    if (RegStrictDom)
      StrictDom = true;
  }

  return StrictDom;
}

// Generate s_waitcnt instructions where needed.
bool SIInsertWaitcnts::insertWaitcntInBlock(MachineFunction &MF,
                                            MachineBasicBlock &Block,
                                            WaitcntBrackets &ScoreBrackets) {
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
    if (Inst.getOpcode() == AMDGPU::S_WAITCNT ||
        (Inst.getOpcode() == AMDGPU::S_WAITCNT_VSCNT &&
         Inst.getOperand(0).isReg() &&
         Inst.getOperand(0).getReg() == AMDGPU::SGPR_NULL)) {
      if (!OldWaitcntInstr)
        OldWaitcntInstr = &Inst;
      ++Iter;
      continue;
    }

    // Generate an s_waitcnt instruction to be placed before Inst, if needed.
    Modified |= generateWaitcntInstBefore(Inst, ScoreBrackets, OldWaitcntInstr);
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
            ScoreBrackets.getScoreLB(LGKM_CNT) <
                ScoreBrackets.getScoreUB(LGKM_CNT) &&
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
          SLoadAddresses.insert(std::make_pair(Ptr, Inst.getParent()));
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
      ScoreBrackets->setScoreLB(VM_CNT,
      ScoreBrackets->getScoreUB(VM_CNT));
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

  return Modified;
}

bool SIInsertWaitcnts::runOnMachineFunction(MachineFunction &MF) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  TII = ST->getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();
  IV = AMDGPU::getIsaVersion(ST->getCPU());
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  PDT = &getAnalysis<MachinePostDominatorTree>();

  ForceEmitZeroWaitcnts = ForceEmitZeroFlag;
  for (auto T : inst_counter_types())
    ForceEmitWaitcnt[T] = false;

  HardwareLimits Limits = {};
  Limits.VmcntMax = AMDGPU::getVmcntBitMask(IV);
  Limits.ExpcntMax = AMDGPU::getExpcntBitMask(IV);
  Limits.LgkmcntMax = AMDGPU::getLgkmcntBitMask(IV);
  Limits.VscntMax = ST->hasVscnt() ? 63 : 0;

  unsigned NumVGPRsMax = ST->getAddressableNumVGPRs();
  unsigned NumSGPRsMax = ST->getAddressableNumSGPRs();
  assert(NumVGPRsMax <= SQ_MAX_PGM_VGPRS);
  assert(NumSGPRsMax <= SQ_MAX_PGM_SGPRS);

  RegisterEncoding Encoding = {};
  Encoding.VGPR0 = TRI->getEncodingValue(AMDGPU::VGPR0);
  Encoding.VGPRL = Encoding.VGPR0 + NumVGPRsMax - 1;
  Encoding.SGPR0 = TRI->getEncodingValue(AMDGPU::SGPR0);
  Encoding.SGPRL = Encoding.SGPR0 + NumSGPRsMax - 1;

  TrackedWaitcntSet.clear();
  BlockInfos.clear();
  bool Modified = false;

  if (!MFI->isEntryFunction()) {
    // Wait for any outstanding memory operations that the input registers may
    // depend on. We can't track them and it's better to do the wait after the
    // costly call sequence.

    // TODO: Could insert earlier and schedule more liberally with operations
    // that only use caller preserved registers.
    MachineBasicBlock &EntryBB = MF.front();
    MachineBasicBlock::iterator I = EntryBB.begin();
    for (MachineBasicBlock::iterator E = EntryBB.end();
         I != E && (I->isPHI() || I->isMetaInstruction()); ++I)
      ;
    BuildMI(EntryBB, I, DebugLoc(), TII->get(AMDGPU::S_WAITCNT)).addImm(0);
    if (ST->hasVscnt())
      BuildMI(EntryBB, I, DebugLoc(), TII->get(AMDGPU::S_WAITCNT_VSCNT))
          .addReg(AMDGPU::SGPR_NULL, RegState::Undef)
          .addImm(0);

    Modified = true;
  }

  // Keep iterating over the blocks in reverse post order, inserting and
  // updating s_waitcnt where needed, until a fix point is reached.
  for (auto *MBB : ReversePostOrderTraversal<MachineFunction *>(&MF))
    BlockInfos.insert({MBB, BlockInfo(MBB)});

  std::unique_ptr<WaitcntBrackets> Brackets;
  bool Repeat;
  do {
    Repeat = false;

    for (auto BII = BlockInfos.begin(), BIE = BlockInfos.end(); BII != BIE;
         ++BII) {
      BlockInfo &BI = BII->second;
      if (!BI.Dirty)
        continue;

      if (BI.Incoming) {
        if (!Brackets)
          Brackets = std::make_unique<WaitcntBrackets>(*BI.Incoming);
        else
          *Brackets = *BI.Incoming;
      } else {
        if (!Brackets)
          Brackets = std::make_unique<WaitcntBrackets>(ST, Limits, Encoding);
        else
          *Brackets = WaitcntBrackets(ST, Limits, Encoding);
      }

      Modified |= insertWaitcntInBlock(MF, *BI.MBB, *Brackets);
      BI.Dirty = false;

      if (Brackets->hasPending()) {
        BlockInfo *MoveBracketsToSucc = nullptr;
        for (MachineBasicBlock *Succ : BI.MBB->successors()) {
          auto SuccBII = BlockInfos.find(Succ);
          BlockInfo &SuccBI = SuccBII->second;
          if (!SuccBI.Incoming) {
            SuccBI.Dirty = true;
            if (SuccBII <= BII)
              Repeat = true;
            if (!MoveBracketsToSucc) {
              MoveBracketsToSucc = &SuccBI;
            } else {
              SuccBI.Incoming = std::make_unique<WaitcntBrackets>(*Brackets);
            }
          } else if (SuccBI.Incoming->merge(*Brackets)) {
            SuccBI.Dirty = true;
            if (SuccBII <= BII)
              Repeat = true;
          }
        }
        if (MoveBracketsToSucc)
          MoveBracketsToSucc->Incoming = std::move(Brackets);
      }
    }
  } while (Repeat);

  if (ST->hasScalarStores()) {
    SmallVector<MachineBasicBlock *, 4> EndPgmBlocks;
    bool HaveScalarStores = false;

    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (!HaveScalarStores && TII->isScalarStore(MI))
          HaveScalarStores = true;

        if (MI.getOpcode() == AMDGPU::S_ENDPGM ||
            MI.getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG)
          EndPgmBlocks.push_back(&MBB);
      }
    }

    if (HaveScalarStores) {
      // If scalar writes are used, the cache must be flushed or else the next
      // wave to reuse the same scratch memory can be clobbered.
      //
      // Insert s_dcache_wb at wave termination points if there were any scalar
      // stores, and only if the cache hasn't already been flushed. This could
      // be improved by looking across blocks for flushes in postdominating
      // blocks from the stores but an explicitly requested flush is probably
      // very rare.
      for (MachineBasicBlock *MBB : EndPgmBlocks) {
        bool SeenDCacheWB = false;

        for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
             I != E; ++I) {
          if (I->getOpcode() == AMDGPU::S_DCACHE_WB)
            SeenDCacheWB = true;
          else if (TII->isScalarStore(*I))
            SeenDCacheWB = false;

          // FIXME: It would be better to insert this before a waitcnt if any.
          if ((I->getOpcode() == AMDGPU::S_ENDPGM ||
               I->getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG) &&
              !SeenDCacheWB) {
            Modified = true;
            BuildMI(*MBB, I, I->getDebugLoc(), TII->get(AMDGPU::S_DCACHE_WB));
          }
        }
      }
    }
  }

  return Modified;
}
