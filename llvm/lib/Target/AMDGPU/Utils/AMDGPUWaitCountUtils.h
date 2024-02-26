//===- AMDGPUWaitCountUtils.h - Wait count insertion interface -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUWAITCOUNTUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUWAITCOUNTUTILS_H

#include "GCNSubtarget.h"

namespace llvm {

namespace AMDGPU {

// Class of object that encapsulates latest instruction counter score
// associated with the operand.  Used for determining whether
// s_waitcnt instruction needs to be emitted.
enum InstCounterType : uint8_t {
  CT_START = 0,
  LOAD_CNT = CT_START, // VMcnt prior to gfx12.
  DS_CNT,              // LKGMcnt prior to gfx12.
  EXP_CNT,             //
  STORE_CNT,           // VScnt in gfx10/gfx11.
  NUM_NORMAL_INST_CNTS,
  SAMPLE_CNT = NUM_NORMAL_INST_CNTS, // gfx12+ only.
  BVH_CNT,                           // gfx12+ only.
  KM_CNT,                            // gfx12+ only.
  NUM_EXTENDED_INST_CNTS,
  NUM_INST_CNTS = NUM_EXTENDED_INST_CNTS
};

} // namespace AMDGPU

using AMDGPU::InstCounterType;

template <> struct enum_iteration_traits<InstCounterType> {
  static constexpr bool is_iterable = true;
};

namespace AMDGPU {

// Return an iterator over all counters between the first counter and \c
// MaxCounter (exclusive, default value yields an enumeration over all
// counters).
inline auto inst_counter_types(InstCounterType MaxCounter = NUM_INST_CNTS) {
  return enum_seq(CT_START, MaxCounter);
}

enum WaitEventType : uint8_t {
  VMEM_ACCESS,              // vector-memory read & write
  VMEM_READ_ACCESS,         // vector-memory read
  VMEM_SAMPLER_READ_ACCESS, // vector-memory SAMPLER read (gfx12+ only)
  VMEM_BVH_READ_ACCESS,     // vector-memory BVH read (gfx12+ only)
  VMEM_WRITE_ACCESS,        // vector-memory write that is not scratch
  SCRATCH_WRITE_ACCESS,     // vector-memory write that may be scratch
  LDS_ACCESS,               // lds read & write
  GDS_ACCESS,               // gds read & write
  SQ_MESSAGE,               // send message
  SMEM_ACCESS,              // scalar-memory read & write
  EXP_GPR_LOCK,             // export holding on its data src
  GDS_GPR_LOCK,             // GDS holding on its data and addr src
  EXP_POS_ACCESS,           // write to export position
  EXP_PARAM_ACCESS,         // write to export parameter
  VMW_GPR_LOCK,             // vector-memory write holding on its data src
  EXP_LDS_ACCESS,           // read by ldsdir counting as export
  NUM_WAIT_EVENTS
};
using AMDGPU::WaitEventType;

using RegInterval = std::pair<int, int>;

struct RegisterEncoding {
  unsigned VGPR0;
  unsigned VGPRL;
  unsigned SGPR0;
  unsigned SGPRL;
};

// The mapping is:
//  0                .. SQ_MAX_PGM_VGPRS-1               real VGPRs
//  SQ_MAX_PGM_VGPRS .. NUM_ALL_VGPRS-1                  extra VGPR-like slots
//  NUM_ALL_VGPRS    .. NUM_ALL_VGPRS+SQ_MAX_PGM_SGPRS-1 real SGPRs
// We reserve a fixed number of VGPR slots in the scoring tables for
// special tokens like SCMEM_LDS (needed for buffer load to LDS).
enum RegisterMapping {
  SQ_MAX_PGM_VGPRS = 512, // Maximum programmable VGPRs across all targets.
  AGPR_OFFSET = 256,      // Maximum programmable AccVGPRs across all targets.
  SQ_MAX_PGM_SGPRS = 256, // Maximum programmable SGPRs across all targets.
  NUM_EXTRA_VGPRS = 9,    // Reserved slot for DS.
  // Artificial register slots to track LDS writes into specific LDS locations
  // if a location is known. When slots are exhausted or location is
  // unknown use the first slot. The first slot is also always updated in
  // addition to known location's slot to properly generate waits if dependent
  // instruction's location is unknown.
  EXTRA_VGPR_LDS = 0,
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
  VMEM_BVH,
  NUM_VMEM_TYPES
};

// Maps values of InstCounterType to the instruction that waits on that
// counter. Only used if GCNSubtarget::hasExtendedWaitCounts()
// returns true.
static const unsigned instrsForExtendedCounterTypes[NUM_EXTENDED_INST_CNTS] = {
    AMDGPU::S_WAIT_LOADCNT,  AMDGPU::S_WAIT_DSCNT,     AMDGPU::S_WAIT_EXPCNT,
    AMDGPU::S_WAIT_STORECNT, AMDGPU::S_WAIT_SAMPLECNT, AMDGPU::S_WAIT_BVHCNT,
    AMDGPU::S_WAIT_KMCNT};

using WaitCntBitMaskFn = std::function<unsigned(const IsaVersion &)>;
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
  WaitcntBrackets(const GCNSubtarget *SubTarget, InstCounterType MaxCounter,
                  RegisterEncoding Encoding,
                  const unsigned *WaitEventMaskForInst,
                  InstCounterType SmemAccessCounter)
      : ST(SubTarget), MaxCounter(MaxCounter), Encoding(Encoding),
        WaitEventMaskForInst(WaitEventMaskForInst),
        SmemAccessCounter(SmemAccessCounter) {
    AMDGPU::IsaVersion IV = AMDGPU::getIsaVersion(ST->getCPU());
    for (auto T : inst_counter_types()) {
      auto Fn = getWaitCntBitMaskFn(T);
      HardwareLimits[T] = Fn(IV);
    }
  }

  unsigned getWaitCountMax(InstCounterType T) const {
    return HardwareLimits[T];
  }

  bool isSmemCounter(InstCounterType T) const { return T == SmemAccessCounter; }

  unsigned getScoreLB(InstCounterType T) const {
    assert(T < NUM_INST_CNTS);
    return ScoreLBs[T];
  }

  unsigned getScoreUB(InstCounterType T) const {
    assert(T < NUM_INST_CNTS);
    return ScoreUBs[T];
  }

  unsigned getScoreRange(InstCounterType T) const {
    return getScoreUB(T) - getScoreLB(T);
  }

  unsigned getRegScore(int GprNo, InstCounterType T) const {
    if (GprNo < NUM_ALL_VGPRS) {
      return VgprScores[T][GprNo];
    }
    assert(isSmemCounter(T));
    return SgprScores[GprNo - NUM_ALL_VGPRS];
  }

  bool merge(const WaitcntBrackets &Other);

  RegInterval getRegInterval(const MachineInstr *MI,
                             const MachineRegisterInfo *MRI,
                             const SIRegisterInfo *TRI, unsigned OpNo) const;

  bool counterOutOfOrder(InstCounterType T) const;
  void simplifyWaitcnt(AMDGPU::Waitcnt &Wait) const;
  void simplifyWaitcnt(InstCounterType T, unsigned &Count) const;
  void determineWait(InstCounterType T, int RegNo, AMDGPU::Waitcnt &Wait) const;
  void applyWaitcnt(const AMDGPU::Waitcnt &Wait);
  void applyWaitcnt(InstCounterType T, unsigned Count);
  void updateByEvent(const SIInstrInfo *TII, const SIRegisterInfo *TRI,
                     const MachineRegisterInfo *MRI, WaitEventType E,
                     MachineInstr &MI);

  unsigned hasPendingEvent() const { return PendingEvents; }
  unsigned hasPendingEvent(WaitEventType E) const {
    return PendingEvents & (1 << E);
  }
  unsigned hasPendingEvent(InstCounterType T) const {
    unsigned HasPending = PendingEvents & WaitEventMaskForInst[T];
    assert((HasPending != 0) == (getScoreRange(T) != 0));
    return HasPending;
  }

  bool hasMixedPendingEvents(InstCounterType T) const {
    unsigned Events = hasPendingEvent(T);
    // Return true if more than one bit is set in Events.
    return Events & (Events - 1);
  }

  bool hasPendingFlat() const {
    return ((LastFlat[DS_CNT] > ScoreLBs[DS_CNT] &&
             LastFlat[DS_CNT] <= ScoreUBs[DS_CNT]) ||
            (LastFlat[LOAD_CNT] > ScoreLBs[LOAD_CNT] &&
             LastFlat[LOAD_CNT] <= ScoreUBs[LOAD_CNT]));
  }

  void setPendingFlat() {
    LastFlat[LOAD_CNT] = ScoreUBs[LOAD_CNT];
    LastFlat[DS_CNT] = ScoreUBs[DS_CNT];
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

  void setStateOnFunctionEntryOrReturn() {
    setScoreUB(STORE_CNT, getScoreUB(STORE_CNT) + getWaitCountMax(STORE_CNT));
    PendingEvents |= WaitEventMaskForInst[STORE_CNT];
  }

  ArrayRef<const MachineInstr *> getLDSDMAStores() const {
    return LDSDMAStores;
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

  WaitCntBitMaskFn getWaitCntBitMaskFn(InstCounterType T);
  static bool mergeScore(const MergeInfo &M, unsigned &Score,
                         unsigned OtherScore);

  void setScoreLB(InstCounterType T, unsigned Val) {
    assert(T < NUM_INST_CNTS);
    ScoreLBs[T] = Val;
  }

  void setScoreUB(InstCounterType T, unsigned Val) {
    assert(T < NUM_INST_CNTS);
    ScoreUBs[T] = Val;

    if (T != EXP_CNT)
      return;

    if (getScoreRange(EXP_CNT) > getWaitCountMax(EXP_CNT))
      ScoreLBs[EXP_CNT] = ScoreUBs[EXP_CNT] - getWaitCountMax(EXP_CNT);
  }

  void setRegScore(int GprNo, InstCounterType T, unsigned Val) {
    if (GprNo < NUM_ALL_VGPRS) {
      VgprUB = std::max(VgprUB, GprNo);
      VgprScores[T][GprNo] = Val;
    } else {
      assert(isSmemCounter(T));
      SgprUB = std::max(SgprUB, GprNo - NUM_ALL_VGPRS);
      SgprScores[GprNo - NUM_ALL_VGPRS] = Val;
    }
  }

  void setExpScore(const MachineInstr *MI, const SIInstrInfo *TII,
                   const SIRegisterInfo *TRI, const MachineRegisterInfo *MRI,
                   unsigned OpNo, unsigned Val);

  const GCNSubtarget *ST = nullptr;
  InstCounterType MaxCounter = NUM_EXTENDED_INST_CNTS;
  unsigned HardwareLimits[NUM_INST_CNTS] = {0};
  RegisterEncoding Encoding = {};
  const unsigned *WaitEventMaskForInst;
  InstCounterType SmemAccessCounter;
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
  // Wait cnt scores for every sgpr, the DS_CNT (corresponding to LGKMcnt
  // pre-gfx12) or KM_CNT (gfx12+ only) are relevant.
  unsigned SgprScores[SQ_MAX_PGM_SGPRS] = {0};
  // Bitmask of the VmemTypes of VMEM instructions that might have a pending
  // write to each vgpr.
  unsigned char VgprVmemTypes[NUM_ALL_VGPRS] = {0};
  // Store representative LDS DMA operations. The only useful info here is
  // alias info. One store is kept per unique AAInfo.
  SmallVector<const MachineInstr *, NUM_EXTRA_VGPRS - 1> LDSDMAStores;
};

struct BlockInfo {
  std::unique_ptr<WaitcntBrackets> Incoming;
  bool Dirty = true;
};

// This abstracts the logic for generating and updating S_WAIT* instructions
// away from the analysis that determines where they are needed. This was
// done because the set of counters and instructions for waiting on them
// underwent a major shift with gfx12, sufficiently so that having this
// abstraction allows the main analysis logic to be simpler than it would
// otherwise have had to become.
class WaitCntGenerator {
protected:
  const GCNSubtarget *ST = nullptr;
  const SIInstrInfo *TII = nullptr;
  AMDGPU::IsaVersion IV;
  InstCounterType MaxCounter;

public:
  WaitCntGenerator() {}
  WaitCntGenerator(const GCNSubtarget *ST, InstCounterType MaxCounter)
      : ST(ST), TII(ST->getInstrInfo()),
        IV(AMDGPU::getIsaVersion(ST->getCPU())), MaxCounter(MaxCounter) {}

  // Edits an existing sequence of wait count instructions according
  // to an incoming Waitcnt value, which is itself updated to reflect
  // any new wait count instructions which may need to be generated by
  // WaitCntGenerator::createNewWaitcnt(). It will return true if any edits
  // were made.
  //
  // This editing will usually be merely updated operands, but it may also
  // delete instructions if the incoming Wait value indicates they are not
  // needed. It may also remove existing instructions for which a wait
  // is needed if it can be determined that it is better to generate new
  // instructions later, as can happen on gfx12.
  virtual bool
  applyPreexistingWaitcnt(WaitcntBrackets &ScoreBrackets,
                          MachineInstr &OldWaitcntInstr, AMDGPU::Waitcnt &Wait,
                          MachineBasicBlock::instr_iterator It) const = 0;

  // Generates new wait count instructions according to the  value of
  // Wait, returning true if any new instructions were created.
  virtual bool createNewWaitcnt(MachineBasicBlock &Block,
                                MachineBasicBlock::instr_iterator It,
                                AMDGPU::Waitcnt Wait) = 0;

  // Returns an array of bit masks which can be used to map values in
  // WaitEventType to corresponding counter values in InstCounterType.
  virtual const unsigned *getWaitEventMask() const = 0;

  // Returns a new waitcnt with all counters except VScnt set to 0. If
  // IncludeVSCnt is true, VScnt is set to 0, otherwise it is set to ~0u.
  virtual AMDGPU::Waitcnt getAllZeroWaitcnt(bool IncludeVSCnt) const = 0;

  virtual ~WaitCntGenerator() = default;

  // Transform a soft waitcnt into a normal one.
  bool promoteSoftWaitCnt(MachineInstr *Waitcnt) const {
    unsigned Opcode =
        SIInstrInfo::getNonSoftWaitcntOpcode(Waitcnt->getOpcode());
    if (Opcode == Waitcnt->getOpcode())
      return false;

    Waitcnt->setDesc(TII->get(Opcode));
    return true;
  }

  // Create a mask value from the initializer list of wait event types.
  unsigned eventMask(std::initializer_list<WaitEventType> Events) const {
    unsigned Mask = 0;
    for (auto &E : Events)
      Mask |= 1 << E;

    return Mask;
  }
};

class WaitCntGeneratorPreGFX12 : public WaitCntGenerator {
public:
  WaitCntGeneratorPreGFX12() {}
  WaitCntGeneratorPreGFX12(const GCNSubtarget *ST)
      : WaitCntGenerator(ST, NUM_NORMAL_INST_CNTS) {}

  virtual AMDGPU::Waitcnt getAllZeroWaitcnt(bool IncludeVSCnt) const override;
  bool
  applyPreexistingWaitcnt(WaitcntBrackets &ScoreBrackets,
                          MachineInstr &OldWaitcntInstr, AMDGPU::Waitcnt &Wait,
                          MachineBasicBlock::instr_iterator It) const override;
  bool createNewWaitcnt(MachineBasicBlock &Block,
                        MachineBasicBlock::instr_iterator It,
                        AMDGPU::Waitcnt Wait) override;

  const unsigned *getWaitEventMask() const override {
    assert(ST);

    static const unsigned WaitEventMaskForInstPreGFX12[NUM_INST_CNTS] = {
        eventMask({VMEM_ACCESS, VMEM_READ_ACCESS, VMEM_SAMPLER_READ_ACCESS,
                   VMEM_BVH_READ_ACCESS}),
        eventMask({SMEM_ACCESS, LDS_ACCESS, GDS_ACCESS, SQ_MESSAGE}),
        eventMask({EXP_GPR_LOCK, GDS_GPR_LOCK, VMW_GPR_LOCK, EXP_PARAM_ACCESS,
                   EXP_POS_ACCESS, EXP_LDS_ACCESS}),
        eventMask({VMEM_WRITE_ACCESS, SCRATCH_WRITE_ACCESS}),
        0,
        0,
        0};

    return WaitEventMaskForInstPreGFX12;
  }
};

class WaitCntGeneratorGFX12Plus : public WaitCntGenerator {
public:
  WaitCntGeneratorGFX12Plus() {}
  WaitCntGeneratorGFX12Plus(const GCNSubtarget *ST, InstCounterType MaxCounter)
      : WaitCntGenerator(ST, MaxCounter) {}

  virtual AMDGPU::Waitcnt getAllZeroWaitcnt(bool IncludeVSCnt) const override;
  bool
  applyPreexistingWaitcnt(WaitcntBrackets &ScoreBrackets,
                          MachineInstr &OldWaitcntInstr, AMDGPU::Waitcnt &Wait,
                          MachineBasicBlock::instr_iterator It) const override;
  bool createNewWaitcnt(MachineBasicBlock &Block,
                        MachineBasicBlock::instr_iterator It,
                        AMDGPU::Waitcnt Wait) override;

  const unsigned *getWaitEventMask() const override {
    assert(ST);

    static const unsigned WaitEventMaskForInstGFX12Plus[NUM_INST_CNTS] = {
        eventMask({VMEM_ACCESS, VMEM_READ_ACCESS}),
        eventMask({LDS_ACCESS, GDS_ACCESS}),
        eventMask({EXP_GPR_LOCK, GDS_GPR_LOCK, VMW_GPR_LOCK, EXP_PARAM_ACCESS,
                   EXP_POS_ACCESS, EXP_LDS_ACCESS}),
        eventMask({VMEM_WRITE_ACCESS, SCRATCH_WRITE_ACCESS}),
        eventMask({VMEM_SAMPLER_READ_ACCESS}),
        eventMask({VMEM_BVH_READ_ACCESS}),
        eventMask({SMEM_ACCESS, SQ_MESSAGE})};

    return WaitEventMaskForInstGFX12Plus;
  }
};

using VGPRInstsSet = DenseSet<MachineInstr *>;

/// This class provides the abstraction for the wait count insertions in a
/// function. Virtual methods are provided to handle the waitcnt insertion in a
/// baisc block for various memory operations as per subtarget requirements.
class AMDGPUWaitCntInserter {
public:
  AMDGPUWaitCntInserter() {}
  AMDGPUWaitCntInserter(const GCNSubtarget *ST, const MachineRegisterInfo *MRI,
                        WaitCntGenerator *WCG, InstCounterType MC)
      : ST(ST), TII(ST->getInstrInfo()), TRI(ST->getRegisterInfo()), MRI(MRI),
        WCG(WCG), MaxCounter(MC) {}
  virtual ~AMDGPUWaitCntInserter() = default;

  InstCounterType getMaxCounter() const { return MaxCounter; }

  bool mayAccessScratchThroughFlat(const MachineInstr &MI) const;
  bool generateWaitcnt(AMDGPU::Waitcnt Wait,
                       MachineBasicBlock::instr_iterator It,
                       MachineBasicBlock &Block, WaitcntBrackets &ScoreBrackets,
                       MachineInstr *OldWaitcntInstr);
  bool generateWaitcntBlockEnd(MachineBasicBlock &Block,
                               WaitcntBrackets &ScoreBrackets,
                               MachineInstr *OldWaitcntInstr);
  bool insertWaitCntsInFunction(MachineFunction &MF, VGPRInstsSet *VGPRInsts);

  virtual bool generateWaitcntInstBefore(MachineInstr &MI,
                                         WaitcntBrackets &ScoreBrackets,
                                         MachineInstr *OldWaitcntInstr,
                                         bool FlushVmCnt,
                                         VGPRInstsSet *VGPRInsts) = 0;

  virtual bool insertWaitcntInBlock(MachineFunction &MF,
                                    MachineBasicBlock &Block,
                                    WaitcntBrackets &ScoreBrackets,
                                    VGPRInstsSet *VGPRInsts) = 0;

  virtual void updateEventWaitcntAfter(MachineInstr &Inst,
                                       WaitcntBrackets *ScoreBrackets) = 0;

protected:
  bool isVMEMOrFlatVMEM(const MachineInstr &MI) const;
  bool mayAccessVMEMThroughFlat(const MachineInstr &MI) const;

  const GCNSubtarget *ST = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;

  // WCG will point to one of the generator objects of its derived classes,
  // which must have been re-initialised before use from a value made using a
  // subtarget constructor.
  WaitCntGenerator *WCG = nullptr;
  InstCounterType MaxCounter;
};

bool isWaitInstr(MachineInstr &Inst);
VmemType getVmemType(const MachineInstr &Inst);
bool callWaitsOnFunctionEntry(const MachineInstr &MI);
bool callWaitsOnFunctionReturn(const MachineInstr &MI);
InstCounterType eventCounter(const unsigned *masks, WaitEventType E);
bool readsVCCZ(const MachineInstr &MI);
bool isCacheInvOrWBInst(MachineInstr &Inst);
bool updateVMCntOnly(const MachineInstr &Inst);
void addWait(AMDGPU::Waitcnt &Wait, InstCounterType T, unsigned Count);
WaitCntGenerator *getWaitCntGenerator(MachineFunction &MF,
                                      WaitCntGeneratorPreGFX12 &WCGPreGFX12,
                                      WaitCntGeneratorGFX12Plus &WCGGFX12Plus,
                                      InstCounterType &MaxCounter);
} // end namespace AMDGPU

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUWAITCOUNTUTILS_H
