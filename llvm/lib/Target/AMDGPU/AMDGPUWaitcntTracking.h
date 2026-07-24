//===- AMDGPUWaitcntTracking.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
///   Utilities to track HWEvents for the purposes of inserting Waitcnt
///   instructions. While the main client of this file is the InsertWaitCnts
///   pass, this is treated like a reusable ADT in order to enforce a
///   separation of concerns.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUWAITCNTTRACKING_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUWAITCNTTRACKING_H

#include "AMDGPUHWEvents.h"
#include "AMDGPUWaitcntUtils.h"
#include "SIRegisterInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace llvm {

class MachineInstr;

namespace AMDGPU {

/// Integer IDs used to track vector memory locations we may have to wait on.
/// Encoded as u16 chunks:
///
///   [0,               REGUNITS_END ): MCRegUnit
///   [LDSDMA_BEGIN,    LDSDMA_END  ) : LDS DMA IDs
///
/// NOTE: The choice of encoding these as "u16 chunks" is arbitrary.
/// It gives (2 << 16) - 1 entries per category which is more than enough
/// for all register units. MCPhysReg is u16 so we don't even support >u16
/// physical register numbers at this time, let alone >u16 register units.
/// In any case, an assertion in "WaitcntBrackets" ensures REGUNITS_END
/// is enough for all register units.
using VMEMID = uint32_t;

enum : VMEMID {
  TRACKINGID_RANGE_LEN = (1 << 16),

  // Important: MCRegUnits must always be tracked starting from 0, as we
  // need to be able to convert between a MCRegUnit and a VMEMID freely.
  REGUNITS_BEGIN = 0,
  REGUNITS_END = REGUNITS_BEGIN + TRACKINGID_RANGE_LEN,

  // Note for LDSDMA: LDSDMA_BEGIN corresponds to the "common"
  // entry, which is updated for all LDS DMA operations encountered.
  // Specific LDS DMA IDs start at LDSDMA_BEGIN + 1.
  NUM_LDSDMA = TRACKINGID_RANGE_LEN,
  LDSDMA_BEGIN = REGUNITS_END,
  LDSDMA_END = LDSDMA_BEGIN + NUM_LDSDMA,
};

/// Convert a MCRegUnit to a VMEMID.
static constexpr VMEMID toVMEMID(MCRegUnit RU) {
  return static_cast<unsigned>(RU);
}

/// Small info struct to provide necessary context for waitcnt tracking.
struct WaitcntBracketsContext {
  using GetWaitEventsFn = std::function<HWEvents(InstCounterType)>;
  using GetCounterFromEventFn = std::function<InstCounterType(HWEvents)>;

  WaitcntBracketsContext(const GCNSubtarget &ST, const MachineRegisterInfo &MRI,
                         const SIInstrInfo &TII, const SIRegisterInfo &TRI,
                         bool IsTgSplit, bool IsExpertMode,
                         InstCounterType MaxCounter, HardwareLimits Limits,
                         GetWaitEventsFn GetWaitEvents,
                         GetCounterFromEventFn GetCounterFromEvent)
      : ST(ST), MRI(MRI), TII(TII), TRI(TRI), IsTgSplit(IsTgSplit),
        IsExpertMode(IsExpertMode), MaxCounter(MaxCounter), Limits(Limits),
        GetWaitEvents(GetWaitEvents), GetCounterFromEvent(GetCounterFromEvent) {
  }

  const GCNSubtarget &ST;
  const MachineRegisterInfo &MRI;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;

  bool IsTgSplit;
  bool IsExpertMode;

  InstCounterType MaxCounter;
  HardwareLimits Limits;

  // TODO: Should we eventually precompute everything into an array and just
  // store an ArrayRef here instead ?
  GetWaitEventsFn GetWaitEvents;
  GetCounterFromEventFn GetCounterFromEvent;
};

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
  /// \param Ctx Context to track waitcnts. This is not copied and is expected
  /// to outlive the class, as the typical usage pattern for this class implies
  /// creating one instance for each basic block.
  WaitcntBrackets(const WaitcntBracketsContext &Ctx);

#ifndef NDEBUG
  ~WaitcntBrackets();
#endif

  // Do some internal consistency checks.
  void verify();

  unsigned getOutstanding(InstCounterType T) const {
    return ScoreUBs[T] - ScoreLBs[T];
  }

  bool hasPendingVMEM(VMEMID ID, InstCounterType T) const {
    return getVMemScore(ID, T) > getScoreLB(T);
  }

  bool empty(InstCounterType T) const { return getScoreRange(T) == 0; }

  bool counterOutOfOrder(InstCounterType T) const;

  bool merge(const WaitcntBrackets &Other);

  void simplifyWaitcnt(Waitcnt &Wait) const;
  void simplifyWaitcnt(const Waitcnt &CheckWait, Waitcnt &UpdateWait) const;
  void simplifyWaitcnt(InstCounterType T, unsigned &Count) const;
  void simplifyWaitcnt(Waitcnt &Wait, InstCounterType T) const;

  void determineWaitForPhysReg(InstCounterType T, MCPhysReg Reg, Waitcnt &Wait,
                               const MachineInstr &MI) const;
  void determineWaitForLDSDMA(InstCounterType T, VMEMID TID,
                              Waitcnt &Wait) const;
  Waitcnt determineAsyncWait(unsigned N);
  void tryClearSCCWriteEvent(MachineInstr *Inst);

  void applyWaitcnt(const Waitcnt &Wait);
  void applyWaitcnt(InstCounterType T, unsigned Count);
  void applyWaitcnt(const Waitcnt &Wait, InstCounterType T);
  void updateByEvent(HWEvents E, MachineInstr &MI);
  void recordAsyncMark(MachineInstr &MI);

  HWEvents getPendingEvents() const { return PendingEvents; }
  bool hasPendingEvent() const { return PendingEvents.any(); }
  bool hasPendingEvent(HWEvents E) const { return PendingEvents.contains(E); }
  bool hasPendingEvent(InstCounterType T) const;
  bool hasMixedPendingEvents(InstCounterType T) const;

  bool hasPendingFlat() const;
  void setPendingFlat();

  bool hasPendingGDS() const;
  unsigned getPendingGDSWait() const;
  void setPendingGDS() { LastGDS = ScoreUBs[DS_CNT]; }

  /// \return true if there might be pending writes to the vgpr-interval by VMEM
  /// instructions where the HWEvents in VGPRContext are not contained in E.
  bool hasDifferentVGPRPendingEvents(MCPhysReg Reg, HWEvents E) const;
  void clearVGPRPendingEvents(MCPhysReg Reg);

  void setStateOnFunctionEntryOrReturn();

  ArrayRef<const MachineInstr *> getLDSDMAStores() const {
    return LDSDMAStores;
  }

  bool hasPointSampleAccel(const MachineInstr &MI) const;
  bool hasPointSamplePendingVmemTypes(const MachineInstr &MI,
                                      MCPhysReg RU) const;

  void print(raw_ostream &) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const { print(dbgs()); }
#endif

private:
  HWEvents getWaitEvents(InstCounterType T) const {
    return Ctx->GetWaitEvents(T);
  }

  InstCounterType getCounterFromEvent(HWEvents E) const {
    return Ctx->GetCounterFromEvent(E);
  }

  bool isSmemAccessCounter(InstCounterType T) const {
    return T == getCounterFromEvent(HWEvents::SMEM_ACCESS);
  }

  MCPhysReg determineVGPR16Dependency(const MachineInstr &MI, InstCounterType T,
                                      MCPhysReg Reg) const;

  void simplifyXcnt(const Waitcnt &CheckWait, Waitcnt &UpdateWait) const;
  void simplifyVmVsrc(const Waitcnt &CheckWait, Waitcnt &UpdateWait) const;

  unsigned getScoreLB(InstCounterType T) const;
  unsigned getScoreUB(InstCounterType T) const;
  unsigned getScoreRange(InstCounterType T) const;
  unsigned getSGPRScore(MCRegUnit RU, InstCounterType T) const;
  unsigned getVMemScore(VMEMID TID, InstCounterType T) const;

  unsigned getLimit(InstCounterType T) const;

  // Free up memory by removing empty entries from the DenseMap that track event
  // scores.
  void purgeEmptyTrackingData();

  struct MergeInfo {
    unsigned OldLB;
    unsigned OtherLB;
    unsigned MyShift;
    unsigned OtherShift;
  };

  using CounterValueArray = std::array<unsigned, NUM_INST_CNTS>;

  void determineWaitForScore(InstCounterType T, unsigned Score,
                             Waitcnt &Wait) const;

  static bool mergeScore(const MergeInfo &M, unsigned &Score,
                         unsigned OtherScore);
  bool mergeAsyncMarks(ArrayRef<MergeInfo> MergeInfos,
                       ArrayRef<CounterValueArray> OtherMarks);

  iterator_range<MCRegUnitIterator> regunits(MCPhysReg Reg) const;

  void setScoreLB(InstCounterType T, unsigned Val);
  void setScoreUB(InstCounterType T, unsigned Val);

  void setRegScore(MCPhysReg Reg, InstCounterType T, unsigned Val);

  void setVMemScore(VMEMID TID, InstCounterType T, unsigned Val) {
    VMem[TID].Scores[T] = Val;
  }

  void setScoreByOperand(const MachineOperand &Op, InstCounterType CntTy,
                         unsigned Val);

  const WaitcntBracketsContext *Ctx = nullptr;

  unsigned ScoreLBs[NUM_INST_CNTS] = {0};
  unsigned ScoreUBs[NUM_INST_CNTS] = {0};
  HWEvents PendingEvents;

  // Remember the last flat memory operation.
  unsigned LastFlatDsCnt = 0;
  unsigned LastFlatLoadCnt = 0;
  // Remember the last GDS operation.
  unsigned LastGDS = 0;

  // The score tracking logic is fragmented as follows:
  // - VMem: VGPR RegUnits and LDS DMA IDs, see the VMEMID encoding.
  // - SGPRs: SGPR RegUnits
  // - SCC: Non-allocatable and not general purpose: not a SGPR.
  //
  // For the VMem case, if the key is within the range of LDS DMA IDs,
  // then the corresponding index into the `LDSDMAStores` vector below is:
  //   Key - LDSDMA_BEGIN - 1
  // This is because LDSDMA_BEGIN is a generic entry and does not have an
  // associated MachineInstr.
  //
  // TODO: Could we track SCC alongside SGPRs so it's not longer a special case?

  struct VMEMInfo {
    // Scores for all instruction counters. Zero-initialized.
    CounterValueArray Scores{};
    // For VGPRs, we need to track an additional fine-grained set of pending
    // events.
    HWEvents VGPRPendingEvents;

    bool empty() const {
      return all_of(Scores, equal_to(0)) && !VGPRPendingEvents;
    }
  };

  /// Wait cnt scores for every sgpr, the DS_CNT (corresponding to LGKMcnt
  /// pre-gfx12) or KM_CNT (gfx12+ only), and X_CNT (gfx1250) are relevant.
  class SGPRInfo {
    /// Either DS_CNT or KM_CNT score.
    unsigned ScoreDsKmCnt = 0;
    unsigned ScoreXCnt = 0;

  public:
    unsigned get(InstCounterType T) const;
    unsigned &get(InstCounterType T);
    bool empty() const { return !ScoreDsKmCnt && !ScoreXCnt; }
  };

  DenseMap<VMEMID, VMEMInfo> VMem; // VGPR + LDS DMA
  DenseMap<MCRegUnit, SGPRInfo> SGPRs;

  // Reg score for SCC.
  unsigned SCCScore = 0;
  // The unique instruction that has an SCC write pending, if there is one.
  const MachineInstr *PendingSCCWrite = nullptr;

  // Store representative LDS DMA operations. The only useful info here is
  // alias info. One store is kept per unique AAInfo.
  SmallVector<const MachineInstr *> LDSDMAStores;

  // State of all counters at each async mark encountered so far.
  SmallVector<CounterValueArray> AsyncMarks;

  // But in the rare pathological case, a nest of loops that pushes marks
  // without waiting on any mark can cause AsyncMarks to grow very large. We cap
  // it to a reasonable limit. We can tune this later or potentially introduce a
  // user option to control the value.
  static constexpr unsigned MaxAsyncMarks = 16;

  // Track the upper bound score for async operations that are not part of a
  // mark yet. Initialized to all zeros.
  CounterValueArray AsyncScore{};
};

} // namespace AMDGPU

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUWAITCNTTRACKING_H
