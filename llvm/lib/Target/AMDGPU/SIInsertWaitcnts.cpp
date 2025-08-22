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
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace llvm;

#define DEBUG_TYPE "si-insert-waitcnts"

DEBUG_COUNTER(ForceExpCounter, DEBUG_TYPE "-forceexp",
              "Force emit s_waitcnt expcnt(0) instrs");
DEBUG_COUNTER(ForceLgkmCounter, DEBUG_TYPE "-forcelgkm",
              "Force emit s_waitcnt lgkmcnt(0) instrs");
DEBUG_COUNTER(ForceVMCounter, DEBUG_TYPE "-forcevm",
              "Force emit s_waitcnt vmcnt(0) instrs");

static cl::opt<bool>
    ForceEmitZeroFlag("amdgpu-waitcnt-forcezero",
                      cl::desc("Force all waitcnt instrs to be emitted as "
                               "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"),
                      cl::init(false), cl::Hidden);

static cl::opt<bool> ForceEmitZeroLoadFlag(
    "amdgpu-waitcnt-load-forcezero",
    cl::desc("Force all waitcnt load counters to wait until 0"),
    cl::init(false), cl::Hidden);

namespace {
// Class of object that encapsulates latest instruction counter score
// associated with the operand.  Used for determining whether
// s_waitcnt instruction needs to be emitted.

enum InstCounterType {
  LOAD_CNT = 0, // VMcnt prior to gfx12.
  DS_CNT,       // LKGMcnt prior to gfx12.
  EXP_CNT,      //
  STORE_CNT,    // VScnt in gfx10/gfx11.
  NUM_NORMAL_INST_CNTS,
  SAMPLE_CNT = NUM_NORMAL_INST_CNTS, // gfx12+ only.
  BVH_CNT,                           // gfx12+ only.
  KM_CNT,                            // gfx12+ only.
  X_CNT,                             // gfx1250.
  NUM_EXTENDED_INST_CNTS,
  NUM_INST_CNTS = NUM_EXTENDED_INST_CNTS
};
} // namespace

namespace llvm {
template <> struct enum_iteration_traits<InstCounterType> {
  static constexpr bool is_iterable = true;
};
} // namespace llvm

namespace {
// Return an iterator over all counters between LOAD_CNT (the first counter)
// and \c MaxCounter (exclusive, default value yields an enumeration over
// all counters).
auto inst_counter_types(InstCounterType MaxCounter = NUM_INST_CNTS) {
  return enum_seq(LOAD_CNT, MaxCounter);
}

using RegInterval = std::pair<int, int>;

struct HardwareLimits {
  unsigned LoadcntMax; // Corresponds to VMcnt prior to gfx12.
  unsigned ExpcntMax;
  unsigned DscntMax;     // Corresponds to LGKMcnt prior to gfx12.
  unsigned StorecntMax;  // Corresponds to VScnt in gfx10/gfx11.
  unsigned SamplecntMax; // gfx12+ only.
  unsigned BvhcntMax;    // gfx12+ only.
  unsigned KmcntMax;     // gfx12+ only.
  unsigned XcntMax;      // gfx1250.
};

#define AMDGPU_DECLARE_WAIT_EVENTS(DECL)                                       \
  DECL(VMEM_ACCESS)              /* vmem read & write */                       \
  DECL(VMEM_READ_ACCESS)         /* vmem read */                               \
  DECL(VMEM_SAMPLER_READ_ACCESS) /* vmem SAMPLER read (gfx12+ only) */         \
  DECL(VMEM_BVH_READ_ACCESS)     /* vmem BVH read (gfx12+ only) */             \
  DECL(VMEM_WRITE_ACCESS)        /* vmem write that is not scratch */          \
  DECL(SCRATCH_WRITE_ACCESS)     /* vmem write that may be scratch */          \
  DECL(VMEM_GROUP)               /* vmem group */                              \
  DECL(LDS_ACCESS)               /* lds read & write */                        \
  DECL(GDS_ACCESS)               /* gds read & write */                        \
  DECL(SQ_MESSAGE)               /* send message */                            \
  DECL(SMEM_ACCESS)              /* scalar-memory read & write */              \
  DECL(SMEM_GROUP)               /* scalar-memory group */                     \
  DECL(EXP_GPR_LOCK)             /* export holding on its data src */          \
  DECL(GDS_GPR_LOCK)             /* GDS holding on its data and addr src */    \
  DECL(EXP_POS_ACCESS)           /* write to export position */                \
  DECL(EXP_PARAM_ACCESS)         /* write to export parameter */               \
  DECL(VMW_GPR_LOCK)             /* vmem write holding on its data src */      \
  DECL(EXP_LDS_ACCESS)           /* read by ldsdir counting as export */

// clang-format off
#define AMDGPU_EVENT_ENUM(Name) Name,
enum WaitEventType {
  AMDGPU_DECLARE_WAIT_EVENTS(AMDGPU_EVENT_ENUM)
  NUM_WAIT_EVENTS
};
#undef AMDGPU_EVENT_ENUM

#define AMDGPU_EVENT_NAME(Name) #Name,
static constexpr StringLiteral WaitEventTypeName[] = {
  AMDGPU_DECLARE_WAIT_EVENTS(AMDGPU_EVENT_NAME)
};
#undef AMDGPU_EVENT_NAME
// clang-format on

// The mapping is:
//  0                .. SQ_MAX_PGM_VGPRS-1               real VGPRs
//  SQ_MAX_PGM_VGPRS .. NUM_ALL_VGPRS-1                  extra VGPR-like slots
//  NUM_ALL_VGPRS    .. NUM_ALL_VGPRS+SQ_MAX_PGM_SGPRS-1 real SGPRs
// We reserve a fixed number of VGPR slots in the scoring tables for
// special tokens like SCMEM_LDS (needed for buffer load to LDS).
enum RegisterMapping {
  SQ_MAX_PGM_VGPRS = 1024, // Maximum programmable VGPRs across all targets.
  AGPR_OFFSET = 512,       // Maximum programmable ArchVGPRs across all targets.
  SQ_MAX_PGM_SGPRS = 128,  // Maximum programmable SGPRs across all targets.
  // Artificial register slots to track LDS writes into specific LDS locations
  // if a location is known. When slots are exhausted or location is
  // unknown use the first slot. The first slot is also always updated in
  // addition to known location's slot to properly generate waits if dependent
  // instruction's location is unknown.
  FIRST_LDS_VGPR = SQ_MAX_PGM_VGPRS, // Extra slots for LDS stores.
  NUM_LDS_VGPRS = 9,                 // One more than the stores we track.
  NUM_ALL_VGPRS = SQ_MAX_PGM_VGPRS + NUM_LDS_VGPRS, // Where SGPRs start.
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
    AMDGPU::S_WAIT_KMCNT,    AMDGPU::S_WAIT_XCNT};

static bool updateVMCntOnly(const MachineInstr &Inst) {
  return (SIInstrInfo::isVMEM(Inst) && !SIInstrInfo::isFLAT(Inst)) ||
         SIInstrInfo::isFLATGlobal(Inst) || SIInstrInfo::isFLATScratch(Inst);
}

#ifndef NDEBUG
static bool isNormalMode(InstCounterType MaxCounter) {
  return MaxCounter == NUM_NORMAL_INST_CNTS;
}
#endif // NDEBUG

VmemType getVmemType(const MachineInstr &Inst) {
  assert(updateVMCntOnly(Inst));
  if (!SIInstrInfo::isImage(Inst))
    return VMEM_NOSAMPLER;
  const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Inst.getOpcode());
  const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
      AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);

  if (BaseInfo->BVH)
    return VMEM_BVH;

  // We have to make an additional check for isVSAMPLE here since some
  // instructions don't have a sampler, but are still classified as sampler
  // instructions for the purposes of e.g. waitcnt.
  if (BaseInfo->Sampler || BaseInfo->MSAA || SIInstrInfo::isVSAMPLE(Inst))
    return VMEM_SAMPLER;

  return VMEM_NOSAMPLER;
}

unsigned &getCounterRef(AMDGPU::Waitcnt &Wait, InstCounterType T) {
  switch (T) {
  case LOAD_CNT:
    return Wait.LoadCnt;
  case EXP_CNT:
    return Wait.ExpCnt;
  case DS_CNT:
    return Wait.DsCnt;
  case STORE_CNT:
    return Wait.StoreCnt;
  case SAMPLE_CNT:
    return Wait.SampleCnt;
  case BVH_CNT:
    return Wait.BvhCnt;
  case KM_CNT:
    return Wait.KmCnt;
  case X_CNT:
    return Wait.XCnt;
  default:
    llvm_unreachable("bad InstCounterType");
  }
}

void addWait(AMDGPU::Waitcnt &Wait, InstCounterType T, unsigned Count) {
  unsigned &WC = getCounterRef(Wait, T);
  WC = std::min(WC, Count);
}

void setNoWait(AMDGPU::Waitcnt &Wait, InstCounterType T) {
  getCounterRef(Wait, T) = ~0u;
}

unsigned getWait(AMDGPU::Waitcnt &Wait, InstCounterType T) {
  return getCounterRef(Wait, T);
}

// Mapping from event to counter according to the table masks.
InstCounterType eventCounter(const unsigned *masks, WaitEventType E) {
  for (auto T : inst_counter_types()) {
    if (masks[T] & (1 << E))
      return T;
  }
  llvm_unreachable("event type has no associated counter");
}

class WaitcntBrackets;

// This abstracts the logic for generating and updating S_WAIT* instructions
// away from the analysis that determines where they are needed. This was
// done because the set of counters and instructions for waiting on them
// underwent a major shift with gfx12, sufficiently so that having this
// abstraction allows the main analysis logic to be simpler than it would
// otherwise have had to become.
class WaitcntGenerator {
protected:
  const GCNSubtarget *ST = nullptr;
  const SIInstrInfo *TII = nullptr;
  AMDGPU::IsaVersion IV;
  InstCounterType MaxCounter;
  bool OptNone;

public:
  WaitcntGenerator() = default;
  WaitcntGenerator(const MachineFunction &MF, InstCounterType MaxCounter)
      : ST(&MF.getSubtarget<GCNSubtarget>()), TII(ST->getInstrInfo()),
        IV(AMDGPU::getIsaVersion(ST->getCPU())), MaxCounter(MaxCounter),
        OptNone(MF.getFunction().hasOptNone() ||
                MF.getTarget().getOptLevel() == CodeGenOptLevel::None) {}

  // Return true if the current function should be compiled with no
  // optimization.
  bool isOptNone() const { return OptNone; }

  // Edits an existing sequence of wait count instructions according
  // to an incoming Waitcnt value, which is itself updated to reflect
  // any new wait count instructions which may need to be generated by
  // WaitcntGenerator::createNewWaitcnt(). It will return true if any edits
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

  // Transform a soft waitcnt into a normal one.
  bool promoteSoftWaitCnt(MachineInstr *Waitcnt) const;

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

  virtual ~WaitcntGenerator() = default;

  // Create a mask value from the initializer list of wait event types.
  static constexpr unsigned
  eventMask(std::initializer_list<WaitEventType> Events) {
    unsigned Mask = 0;
    for (auto &E : Events)
      Mask |= 1 << E;

    return Mask;
  }
};

class WaitcntGeneratorPreGFX12 : public WaitcntGenerator {
public:
  WaitcntGeneratorPreGFX12() = default;
  WaitcntGeneratorPreGFX12(const MachineFunction &MF)
      : WaitcntGenerator(MF, NUM_NORMAL_INST_CNTS) {}

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
        0,
        0};

    return WaitEventMaskForInstPreGFX12;
  }

  AMDGPU::Waitcnt getAllZeroWaitcnt(bool IncludeVSCnt) const override;
};

class WaitcntGeneratorGFX12Plus : public WaitcntGenerator {
public:
  WaitcntGeneratorGFX12Plus() = default;
  WaitcntGeneratorGFX12Plus(const MachineFunction &MF,
                            InstCounterType MaxCounter)
      : WaitcntGenerator(MF, MaxCounter) {}

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
        eventMask({SMEM_ACCESS, SQ_MESSAGE}),
        eventMask({VMEM_GROUP, SMEM_GROUP})};

    return WaitEventMaskForInstGFX12Plus;
  }

  AMDGPU::Waitcnt getAllZeroWaitcnt(bool IncludeVSCnt) const override;
};

class SIInsertWaitcnts {
public:
  const GCNSubtarget *ST;
  InstCounterType SmemAccessCounter;
  InstCounterType MaxCounter;
  const unsigned *WaitEventMaskForInst;

private:
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;

  DenseMap<const Value *, MachineBasicBlock *> SLoadAddresses;
  DenseMap<MachineBasicBlock *, bool> PreheadersToFlush;
  MachineLoopInfo *MLI;
  MachinePostDominatorTree *PDT;
  AliasAnalysis *AA = nullptr;

  struct BlockInfo {
    std::unique_ptr<WaitcntBrackets> Incoming;
    bool Dirty = true;
  };

  MapVector<MachineBasicBlock *, BlockInfo> BlockInfos;

  bool ForceEmitWaitcnt[NUM_INST_CNTS];

  // In any given run of this pass, WCG will point to one of these two
  // generator objects, which must have been re-initialised before use
  // from a value made using a subtarget constructor.
  WaitcntGeneratorPreGFX12 WCGPreGFX12;
  WaitcntGeneratorGFX12Plus WCGGFX12Plus;

  WaitcntGenerator *WCG = nullptr;

  // S_ENDPGM instructions before which we should insert a DEALLOC_VGPRS
  // message.
  DenseSet<MachineInstr *> ReleaseVGPRInsts;

  HardwareLimits Limits;

public:
  SIInsertWaitcnts(MachineLoopInfo *MLI, MachinePostDominatorTree *PDT,
                   AliasAnalysis *AA)
      : MLI(MLI), PDT(PDT), AA(AA) {
    (void)ForceExpCounter;
    (void)ForceLgkmCounter;
    (void)ForceVMCounter;
  }

  unsigned getWaitCountMax(InstCounterType T) const {
    switch (T) {
    case LOAD_CNT:
      return Limits.LoadcntMax;
    case DS_CNT:
      return Limits.DscntMax;
    case EXP_CNT:
      return Limits.ExpcntMax;
    case STORE_CNT:
      return Limits.StorecntMax;
    case SAMPLE_CNT:
      return Limits.SamplecntMax;
    case BVH_CNT:
      return Limits.BvhcntMax;
    case KM_CNT:
      return Limits.KmcntMax;
    case X_CNT:
      return Limits.XcntMax;
    default:
      break;
    }
    return 0;
  }

  bool shouldFlushVmCnt(MachineLoop *ML, const WaitcntBrackets &Brackets);
  bool isPreheaderToFlush(MachineBasicBlock &MBB,
                          const WaitcntBrackets &ScoreBrackets);
  bool isVMEMOrFlatVMEM(const MachineInstr &MI) const;
  bool run(MachineFunction &MF);

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

  // Return the appropriate VMEM_*_ACCESS type for Inst, which must be a VMEM
  // instruction.
  WaitEventType getVmemWaitEventType(const MachineInstr &Inst) const {
    switch (Inst.getOpcode()) {
    case AMDGPU::GLOBAL_INV:
      return VMEM_READ_ACCESS; // tracked using loadcnt
    case AMDGPU::GLOBAL_WB:
    case AMDGPU::GLOBAL_WBINV:
      return VMEM_WRITE_ACCESS; // tracked using storecnt
    default:
      break;
    }

    // Maps VMEM access types to their corresponding WaitEventType.
    static const WaitEventType VmemReadMapping[NUM_VMEM_TYPES] = {
        VMEM_READ_ACCESS, VMEM_SAMPLER_READ_ACCESS, VMEM_BVH_READ_ACCESS};

    assert(SIInstrInfo::isVMEM(Inst));
    // LDS DMA loads are also stores, but on the LDS side. On the VMEM side
    // these should use VM_CNT.
    if (!ST->hasVscnt() || SIInstrInfo::mayWriteLDSThroughDMA(Inst))
      return VMEM_ACCESS;
    if (Inst.mayStore() &&
        (!Inst.mayLoad() || SIInstrInfo::isAtomicNoRet(Inst))) {
      // FLAT and SCRATCH instructions may access scratch. Other VMEM
      // instructions do not.
      if (TII->mayAccessScratchThroughFlat(Inst))
        return SCRATCH_WRITE_ACCESS;
      return VMEM_WRITE_ACCESS;
    }
    if (!ST->hasExtendedWaitCounts() || SIInstrInfo::isFLAT(Inst))
      return VMEM_READ_ACCESS;
    return VmemReadMapping[getVmemType(Inst)];
  }

  bool hasXcnt() const { return ST->hasWaitXCnt(); }

  bool mayAccessVMEMThroughFlat(const MachineInstr &MI) const;
  bool mayAccessLDSThroughFlat(const MachineInstr &MI) const;
  bool isVmemAccess(const MachineInstr &MI) const;
  bool generateWaitcntInstBefore(MachineInstr &MI,
                                 WaitcntBrackets &ScoreBrackets,
                                 MachineInstr *OldWaitcntInstr,
                                 bool FlushVmCnt);
  bool generateWaitcnt(AMDGPU::Waitcnt Wait,
                       MachineBasicBlock::instr_iterator It,
                       MachineBasicBlock &Block, WaitcntBrackets &ScoreBrackets,
                       MachineInstr *OldWaitcntInstr);
  void updateEventWaitcntAfter(MachineInstr &Inst,
                               WaitcntBrackets *ScoreBrackets);
  bool isNextENDPGM(MachineBasicBlock::instr_iterator It,
                    MachineBasicBlock *Block) const;
  bool insertForcedWaitAfter(MachineInstr &Inst, MachineBasicBlock &Block,
                             WaitcntBrackets &ScoreBrackets);
  bool insertWaitcntInBlock(MachineFunction &MF, MachineBasicBlock &Block,
                            WaitcntBrackets &ScoreBrackets);
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
  WaitcntBrackets(const SIInsertWaitcnts *Context) : Context(Context) {}

  bool isSmemCounter(InstCounterType T) const {
    return T == Context->SmemAccessCounter || T == X_CNT;
  }

  unsigned getSgprScoresIdx(InstCounterType T) const {
    assert(isSmemCounter(T) && "Invalid SMEM counter");
    return T == X_CNT ? 1 : 0;
  }

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
    if (GprNo < NUM_ALL_VGPRS)
      return VgprScores[T][GprNo];
    return SgprScores[getSgprScoresIdx(T)][GprNo - NUM_ALL_VGPRS];
  }

  bool merge(const WaitcntBrackets &Other);

  RegInterval getRegInterval(const MachineInstr *MI,
                             const MachineRegisterInfo *MRI,
                             const SIRegisterInfo *TRI,
                             const MachineOperand &Op) const;

  bool counterOutOfOrder(InstCounterType T) const;
  void simplifyWaitcnt(AMDGPU::Waitcnt &Wait) const;
  void simplifyWaitcnt(InstCounterType T, unsigned &Count) const;

  void determineWait(InstCounterType T, RegInterval Interval,
                     AMDGPU::Waitcnt &Wait) const;
  void determineWait(InstCounterType T, int RegNo,
                     AMDGPU::Waitcnt &Wait) const {
    determineWait(T, {RegNo, RegNo + 1}, Wait);
  }

  void applyWaitcnt(const AMDGPU::Waitcnt &Wait);
  void applyWaitcnt(InstCounterType T, unsigned Count);
  void applyXcnt(const AMDGPU::Waitcnt &Wait);
  void updateByEvent(const SIInstrInfo *TII, const SIRegisterInfo *TRI,
                     const MachineRegisterInfo *MRI, WaitEventType E,
                     MachineInstr &MI);

  unsigned hasPendingEvent() const { return PendingEvents; }
  unsigned hasPendingEvent(WaitEventType E) const {
    return PendingEvents & (1 << E);
  }
  unsigned hasPendingEvent(InstCounterType T) const {
    unsigned HasPending = PendingEvents & Context->WaitEventMaskForInst[T];
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

  bool hasPendingGDS() const {
    return LastGDS > ScoreLBs[DS_CNT] && LastGDS <= ScoreUBs[DS_CNT];
  }

  unsigned getPendingGDSWait() const {
    return std::min(getScoreUB(DS_CNT) - LastGDS,
                    Context->getWaitCountMax(DS_CNT) - 1);
  }

  void setPendingGDS() { LastGDS = ScoreUBs[DS_CNT]; }

  // Return true if there might be pending writes to the vgpr-interval by VMEM
  // instructions with types different from V.
  bool hasOtherPendingVmemTypes(RegInterval Interval, VmemType V) const {
    for (int RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
      assert(RegNo < NUM_ALL_VGPRS);
      if (VgprVmemTypes[RegNo] & ~(1 << V))
        return true;
    }
    return false;
  }

  void clearVgprVmemTypes(RegInterval Interval) {
    for (int RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
      assert(RegNo < NUM_ALL_VGPRS);
      VgprVmemTypes[RegNo] = 0;
    }
  }

  void setStateOnFunctionEntryOrReturn() {
    setScoreUB(STORE_CNT,
               getScoreUB(STORE_CNT) + Context->getWaitCountMax(STORE_CNT));
    PendingEvents |= Context->WaitEventMaskForInst[STORE_CNT];
  }

  ArrayRef<const MachineInstr *> getLDSDMAStores() const {
    return LDSDMAStores;
  }

  bool hasPointSampleAccel(const MachineInstr &MI) const;
  bool hasPointSamplePendingVmemTypes(const MachineInstr &MI,
                                      RegInterval Interval) const;

  void print(raw_ostream &) const;
  void dump() const { print(dbgs()); }

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

    if (T != EXP_CNT)
      return;

    if (getScoreRange(EXP_CNT) > Context->getWaitCountMax(EXP_CNT))
      ScoreLBs[EXP_CNT] = ScoreUBs[EXP_CNT] - Context->getWaitCountMax(EXP_CNT);
  }

  void setRegScore(int GprNo, InstCounterType T, unsigned Val) {
    setScoreByInterval({GprNo, GprNo + 1}, T, Val);
  }

  void setScoreByInterval(RegInterval Interval, InstCounterType CntTy,
                          unsigned Score);

  void setScoreByOperand(const MachineInstr *MI, const SIRegisterInfo *TRI,
                         const MachineRegisterInfo *MRI,
                         const MachineOperand &Op, InstCounterType CntTy,
                         unsigned Val);

  const SIInsertWaitcnts *Context;

  unsigned ScoreLBs[NUM_INST_CNTS] = {0};
  unsigned ScoreUBs[NUM_INST_CNTS] = {0};
  unsigned PendingEvents = 0;
  // Remember the last flat memory operation.
  unsigned LastFlat[NUM_INST_CNTS] = {0};
  // Remember the last GDS operation.
  unsigned LastGDS = 0;
  // wait_cnt scores for every vgpr.
  // Keep track of the VgprUB and SgprUB to make merge at join efficient.
  int VgprUB = -1;
  int SgprUB = -1;
  unsigned VgprScores[NUM_INST_CNTS][NUM_ALL_VGPRS] = {{0}};
  // Wait cnt scores for every sgpr, the DS_CNT (corresponding to LGKMcnt
  // pre-gfx12) or KM_CNT (gfx12+ only), and X_CNT (gfx1250) are relevant.
  // Row 0 represents the score for either DS_CNT or KM_CNT and row 1 keeps the
  // X_CNT score.
  unsigned SgprScores[2][SQ_MAX_PGM_SGPRS] = {{0}};
  // Bitmask of the VmemTypes of VMEM instructions that might have a pending
  // write to each vgpr.
  unsigned char VgprVmemTypes[NUM_ALL_VGPRS] = {0};
  // Store representative LDS DMA operations. The only useful info here is
  // alias info. One store is kept per unique AAInfo.
  SmallVector<const MachineInstr *, NUM_LDS_VGPRS - 1> LDSDMAStores;
};

class SIInsertWaitcntsLegacy : public MachineFunctionPass {
public:
  static char ID;
  SIInsertWaitcntsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI insert wait instructions";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    AU.addUsedIfAvailable<AAResultsWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

RegInterval WaitcntBrackets::getRegInterval(const MachineInstr *MI,
                                            const MachineRegisterInfo *MRI,
                                            const SIRegisterInfo *TRI,
                                            const MachineOperand &Op) const {
  if (!TRI->isInAllocatableClass(Op.getReg()))
    return {-1, -1};

  // A use via a PW operand does not need a waitcnt.
  // A partial write is not a WAW.
  assert(!Op.getSubReg() || !Op.isUndef());

  RegInterval Result;

  MCRegister MCReg = AMDGPU::getMCReg(Op.getReg(), *Context->ST);
  unsigned RegIdx = TRI->getHWRegIndex(MCReg);
  assert(isUInt<8>(RegIdx));

  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(Op.getReg());
  unsigned Size = TRI->getRegSizeInBits(*RC);

  // AGPRs/VGPRs are tracked every 16 bits, SGPRs by 32 bits
  if (TRI->isVectorRegister(*MRI, Op.getReg())) {
    unsigned Reg = RegIdx << 1 | (AMDGPU::isHi16Reg(MCReg, *TRI) ? 1 : 0);
    assert(Reg < AGPR_OFFSET);
    Result.first = Reg;
    if (TRI->isAGPR(*MRI, Op.getReg()))
      Result.first += AGPR_OFFSET;
    assert(Result.first >= 0 && Result.first < SQ_MAX_PGM_VGPRS);
    assert(Size % 16 == 0);
    Result.second = Result.first + (Size / 16);
  } else if (TRI->isSGPRReg(*MRI, Op.getReg()) && RegIdx < SQ_MAX_PGM_SGPRS) {
    // SGPRs including VCC, TTMPs and EXEC but excluding read-only scalar
    // sources like SRC_PRIVATE_BASE.
    Result.first = RegIdx + NUM_ALL_VGPRS;
    Result.second = Result.first + divideCeil(Size, 32);
  } else {
    return {-1, -1};
  }

  return Result;
}

void WaitcntBrackets::setScoreByInterval(RegInterval Interval,
                                         InstCounterType CntTy,
                                         unsigned Score) {
  for (int RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
    if (RegNo < NUM_ALL_VGPRS) {
      VgprUB = std::max(VgprUB, RegNo);
      VgprScores[CntTy][RegNo] = Score;
    } else {
      SgprUB = std::max(SgprUB, RegNo - NUM_ALL_VGPRS);
      SgprScores[getSgprScoresIdx(CntTy)][RegNo - NUM_ALL_VGPRS] = Score;
    }
  }
}

void WaitcntBrackets::setScoreByOperand(const MachineInstr *MI,
                                        const SIRegisterInfo *TRI,
                                        const MachineRegisterInfo *MRI,
                                        const MachineOperand &Op,
                                        InstCounterType CntTy, unsigned Score) {
  RegInterval Interval = getRegInterval(MI, MRI, TRI, Op);
  setScoreByInterval(Interval, CntTy, Score);
}

// Return true if the subtarget is one that enables Point Sample Acceleration
// and the MachineInstr passed in is one to which it might be applied (the
// hardware makes this decision based on several factors, but we can't determine
// this at compile time, so we have to assume it might be applied if the
// instruction supports it).
bool WaitcntBrackets::hasPointSampleAccel(const MachineInstr &MI) const {
  if (!Context->ST->hasPointSampleAccel() || !SIInstrInfo::isMIMG(MI))
    return false;

  const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(MI.getOpcode());
  const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
      AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
  return BaseInfo->PointSampleAccel;
}

// Return true if the subtarget enables Point Sample Acceleration, the supplied
// MachineInstr is one to which it might be applied and the supplied interval is
// one that has outstanding writes to vmem-types different than VMEM_NOSAMPLER
// (this is the type that a point sample accelerated instruction effectively
// becomes)
bool WaitcntBrackets::hasPointSamplePendingVmemTypes(
    const MachineInstr &MI, RegInterval Interval) const {
  if (!hasPointSampleAccel(MI))
    return false;

  return hasOtherPendingVmemTypes(Interval, VMEM_NOSAMPLER);
}

void WaitcntBrackets::updateByEvent(const SIInstrInfo *TII,
                                    const SIRegisterInfo *TRI,
                                    const MachineRegisterInfo *MRI,
                                    WaitEventType E, MachineInstr &Inst) {
  InstCounterType T = eventCounter(Context->WaitEventMaskForInst, E);

  unsigned UB = getScoreUB(T);
  unsigned CurrScore = UB + 1;
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
    if (TII->isDS(Inst) && Inst.mayLoadOrStore()) {
      // All GDS operations must protect their address register (same as
      // export.)
      if (const auto *AddrOp = TII->getNamedOperand(Inst, AMDGPU::OpName::addr))
        setScoreByOperand(&Inst, TRI, MRI, *AddrOp, EXP_CNT, CurrScore);

      if (Inst.mayStore()) {
        if (const auto *Data0 =
                TII->getNamedOperand(Inst, AMDGPU::OpName::data0))
          setScoreByOperand(&Inst, TRI, MRI, *Data0, EXP_CNT, CurrScore);
        if (const auto *Data1 =
                TII->getNamedOperand(Inst, AMDGPU::OpName::data1))
          setScoreByOperand(&Inst, TRI, MRI, *Data1, EXP_CNT, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst) && !SIInstrInfo::isGWS(Inst) &&
                 Inst.getOpcode() != AMDGPU::DS_APPEND &&
                 Inst.getOpcode() != AMDGPU::DS_CONSUME &&
                 Inst.getOpcode() != AMDGPU::DS_ORDERED_COUNT) {
        for (const MachineOperand &Op : Inst.all_uses()) {
          if (TRI->isVectorRegister(*MRI, Op.getReg()))
            setScoreByOperand(&Inst, TRI, MRI, Op, EXP_CNT, CurrScore);
        }
      }
    } else if (TII->isFLAT(Inst)) {
      if (Inst.mayStore()) {
        setScoreByOperand(&Inst, TRI, MRI,
                          *TII->getNamedOperand(Inst, AMDGPU::OpName::data),
                          EXP_CNT, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setScoreByOperand(&Inst, TRI, MRI,
                          *TII->getNamedOperand(Inst, AMDGPU::OpName::data),
                          EXP_CNT, CurrScore);
      }
    } else if (TII->isMIMG(Inst)) {
      if (Inst.mayStore()) {
        setScoreByOperand(&Inst, TRI, MRI, Inst.getOperand(0), EXP_CNT,
                          CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setScoreByOperand(&Inst, TRI, MRI,
                          *TII->getNamedOperand(Inst, AMDGPU::OpName::data),
                          EXP_CNT, CurrScore);
      }
    } else if (TII->isMTBUF(Inst)) {
      if (Inst.mayStore())
        setScoreByOperand(&Inst, TRI, MRI, Inst.getOperand(0), EXP_CNT,
                          CurrScore);
    } else if (TII->isMUBUF(Inst)) {
      if (Inst.mayStore()) {
        setScoreByOperand(&Inst, TRI, MRI, Inst.getOperand(0), EXP_CNT,
                          CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setScoreByOperand(&Inst, TRI, MRI,
                          *TII->getNamedOperand(Inst, AMDGPU::OpName::data),
                          EXP_CNT, CurrScore);
      }
    } else if (TII->isLDSDIR(Inst)) {
      // LDSDIR instructions attach the score to the destination.
      setScoreByOperand(&Inst, TRI, MRI,
                        *TII->getNamedOperand(Inst, AMDGPU::OpName::vdst),
                        EXP_CNT, CurrScore);
    } else {
      if (TII->isEXP(Inst)) {
        // For export the destination registers are really temps that
        // can be used as the actual source after export patching, so
        // we need to treat them like sources and set the EXP_CNT
        // score.
        for (MachineOperand &DefMO : Inst.all_defs()) {
          if (TRI->isVGPR(*MRI, DefMO.getReg())) {
            setScoreByOperand(&Inst, TRI, MRI, DefMO, EXP_CNT, CurrScore);
          }
        }
      }
      for (const MachineOperand &Op : Inst.all_uses()) {
        if (TRI->isVectorRegister(*MRI, Op.getReg()))
          setScoreByOperand(&Inst, TRI, MRI, Op, EXP_CNT, CurrScore);
      }
    }
  } else if (T == X_CNT) {
    for (const MachineOperand &Op : Inst.all_uses())
      setScoreByOperand(&Inst, TRI, MRI, Op, T, CurrScore);
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
      RegInterval Interval = getRegInterval(&Inst, MRI, TRI, Op);
      if (T == LOAD_CNT || T == SAMPLE_CNT || T == BVH_CNT) {
        if (Interval.first >= NUM_ALL_VGPRS)
          continue;
        if (updateVMCntOnly(Inst)) {
          // updateVMCntOnly should only leave us with VGPRs
          // MUBUF, MTBUF, MIMG, FlatGlobal, and FlatScratch only have VGPR/AGPR
          // defs. That's required for a sane index into `VgprMemTypes` below
          assert(TRI->isVectorRegister(*MRI, Op.getReg()));
          VmemType V = getVmemType(Inst);
          unsigned char TypesMask = 1 << V;
          // If instruction can have Point Sample Accel applied, we have to flag
          // this with another potential dependency
          if (hasPointSampleAccel(Inst))
            TypesMask |= 1 << VMEM_NOSAMPLER;
          for (int RegNo = Interval.first; RegNo < Interval.second; ++RegNo)
            VgprVmemTypes[RegNo] |= TypesMask;
        }
      }
      setScoreByInterval(Interval, T, CurrScore);
    }
    if (Inst.mayStore() &&
        (TII->isDS(Inst) || TII->mayWriteLDSThroughDMA(Inst))) {
      // MUBUF and FLAT LDS DMA operations need a wait on vmcnt before LDS
      // written can be accessed. A load from LDS to VMEM does not need a wait.
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
        // is squashed into a single big object. Do not attempt to use one of
        // the limited LDSDMAStores for something we will not be able to use
        // anyway.
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
        if (Slot || LDSDMAStores.size() == NUM_LDS_VGPRS - 1)
          break;
        LDSDMAStores.push_back(&Inst);
        Slot = LDSDMAStores.size();
        break;
      }
      setRegScore(FIRST_LDS_VGPR + Slot, T, CurrScore);
      if (Slot)
        setRegScore(FIRST_LDS_VGPR, T, CurrScore);
    }
  }
}

void WaitcntBrackets::print(raw_ostream &OS) const {
  const GCNSubtarget *ST = Context->ST;

  OS << '\n';
  for (auto T : inst_counter_types(Context->MaxCounter)) {
    unsigned SR = getScoreRange(T);

    switch (T) {
    case LOAD_CNT:
      OS << "    " << (ST->hasExtendedWaitCounts() ? "LOAD" : "VM") << "_CNT("
         << SR << "): ";
      break;
    case DS_CNT:
      OS << "    " << (ST->hasExtendedWaitCounts() ? "DS" : "LGKM") << "_CNT("
         << SR << "): ";
      break;
    case EXP_CNT:
      OS << "    EXP_CNT(" << SR << "): ";
      break;
    case STORE_CNT:
      OS << "    " << (ST->hasExtendedWaitCounts() ? "STORE" : "VS") << "_CNT("
         << SR << "): ";
      break;
    case SAMPLE_CNT:
      OS << "    SAMPLE_CNT(" << SR << "): ";
      break;
    case BVH_CNT:
      OS << "    BVH_CNT(" << SR << "): ";
      break;
    case KM_CNT:
      OS << "    KM_CNT(" << SR << "): ";
      break;
    case X_CNT:
      OS << "    X_CNT(" << SR << "): ";
      break;
    default:
      OS << "    UNKNOWN(" << SR << "): ";
      break;
    }

    if (SR != 0) {
      // Print vgpr scores.
      unsigned LB = getScoreLB(T);

      for (int J = 0; J <= VgprUB; J++) {
        unsigned RegScore = getRegScore(J, T);
        if (RegScore <= LB)
          continue;
        unsigned RelScore = RegScore - LB - 1;
        if (J < FIRST_LDS_VGPR) {
          OS << RelScore << ":v" << J << " ";
        } else {
          OS << RelScore << ":ds ";
        }
      }
      // Also need to print sgpr scores for lgkm_cnt or xcnt.
      if (isSmemCounter(T)) {
        for (int J = 0; J <= SgprUB; J++) {
          unsigned RegScore = getRegScore(J + NUM_ALL_VGPRS, T);
          if (RegScore <= LB)
            continue;
          unsigned RelScore = RegScore - LB - 1;
          OS << RelScore << ":s" << J << " ";
        }
      }
    }
    OS << '\n';
  }

  OS << "Pending Events: ";
  if (hasPendingEvent()) {
    ListSeparator LS;
    for (unsigned I = 0; I != NUM_WAIT_EVENTS; ++I) {
      if (hasPendingEvent((WaitEventType)I)) {
        OS << LS << WaitEventTypeName[I];
      }
    }
  } else {
    OS << "none";
  }
  OS << '\n';

  OS << '\n';
}

/// Simplify the waitcnt, in the sense of removing redundant counts, and return
/// whether a waitcnt instruction is needed at all.
void WaitcntBrackets::simplifyWaitcnt(AMDGPU::Waitcnt &Wait) const {
  simplifyWaitcnt(LOAD_CNT, Wait.LoadCnt);
  simplifyWaitcnt(EXP_CNT, Wait.ExpCnt);
  simplifyWaitcnt(DS_CNT, Wait.DsCnt);
  simplifyWaitcnt(STORE_CNT, Wait.StoreCnt);
  simplifyWaitcnt(SAMPLE_CNT, Wait.SampleCnt);
  simplifyWaitcnt(BVH_CNT, Wait.BvhCnt);
  simplifyWaitcnt(KM_CNT, Wait.KmCnt);
  simplifyWaitcnt(X_CNT, Wait.XCnt);
}

void WaitcntBrackets::simplifyWaitcnt(InstCounterType T,
                                      unsigned &Count) const {
  // The number of outstanding events for this type, T, can be calculated
  // as (UB - LB). If the current Count is greater than or equal to the number
  // of outstanding events, then the wait for this counter is redundant.
  if (Count >= getScoreRange(T))
    Count = ~0u;
}

void WaitcntBrackets::determineWait(InstCounterType T, RegInterval Interval,
                                    AMDGPU::Waitcnt &Wait) const {
  const unsigned LB = getScoreLB(T);
  const unsigned UB = getScoreUB(T);
  for (int RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
    unsigned ScoreToWait = getRegScore(RegNo, T);

    // If the score of src_operand falls within the bracket, we need an
    // s_waitcnt instruction.
    if ((UB >= ScoreToWait) && (ScoreToWait > LB)) {
      if ((T == LOAD_CNT || T == DS_CNT) && hasPendingFlat() &&
          !Context->ST->hasFlatLgkmVMemCountInOrder()) {
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
        unsigned NeededWait =
            std::min(UB - ScoreToWait, Context->getWaitCountMax(T) - 1);
        addWait(Wait, T, NeededWait);
      }
    }
  }
}

void WaitcntBrackets::applyWaitcnt(const AMDGPU::Waitcnt &Wait) {
  applyWaitcnt(LOAD_CNT, Wait.LoadCnt);
  applyWaitcnt(EXP_CNT, Wait.ExpCnt);
  applyWaitcnt(DS_CNT, Wait.DsCnt);
  applyWaitcnt(STORE_CNT, Wait.StoreCnt);
  applyWaitcnt(SAMPLE_CNT, Wait.SampleCnt);
  applyWaitcnt(BVH_CNT, Wait.BvhCnt);
  applyWaitcnt(KM_CNT, Wait.KmCnt);
  applyXcnt(Wait);
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
    PendingEvents &= ~Context->WaitEventMaskForInst[T];
  }
}

void WaitcntBrackets::applyXcnt(const AMDGPU::Waitcnt &Wait) {
  // Wait on XCNT is redundant if we are already waiting for a load to complete.
  // SMEM can return out of order, so only omit XCNT wait if we are waiting till
  // zero.
  if (Wait.KmCnt == 0 && hasPendingEvent(SMEM_GROUP))
    return applyWaitcnt(X_CNT, 0);

  // If we have pending store we cannot optimize XCnt because we do not wait for
  // stores. VMEM loads retun in order, so if we only have loads XCnt is
  // decremented to the same number as LOADCnt.
  if (Wait.LoadCnt != ~0u && hasPendingEvent(VMEM_GROUP) &&
      !hasPendingEvent(STORE_CNT))
    return applyWaitcnt(X_CNT, std::min(Wait.XCnt, Wait.LoadCnt));

  applyWaitcnt(X_CNT, Wait.XCnt);
}

// Where there are multiple types of event in the bracket of a counter,
// the decrement may go out of order.
bool WaitcntBrackets::counterOutOfOrder(InstCounterType T) const {
  // Scalar memory read always can go out of order.
  if ((T == Context->SmemAccessCounter && hasPendingEvent(SMEM_ACCESS)) ||
      (T == X_CNT && hasPendingEvent(SMEM_GROUP)))
    return true;
  return hasMixedPendingEvents(T);
}

INITIALIZE_PASS_BEGIN(SIInsertWaitcntsLegacy, DEBUG_TYPE, "SI Insert Waitcnts",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(SIInsertWaitcntsLegacy, DEBUG_TYPE, "SI Insert Waitcnts",
                    false, false)

char SIInsertWaitcntsLegacy::ID = 0;

char &llvm::SIInsertWaitcntsID = SIInsertWaitcntsLegacy::ID;

FunctionPass *llvm::createSIInsertWaitcntsPass() {
  return new SIInsertWaitcntsLegacy();
}

static bool updateOperandIfDifferent(MachineInstr &MI, AMDGPU::OpName OpName,
                                     unsigned NewEnc) {
  int OpIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), OpName);
  assert(OpIdx >= 0);

  MachineOperand &MO = MI.getOperand(OpIdx);

  if (NewEnc == MO.getImm())
    return false;

  MO.setImm(NewEnc);
  return true;
}

/// Determine if \p MI is a gfx12+ single-counter S_WAIT_*CNT instruction,
/// and if so, which counter it is waiting on.
static std::optional<InstCounterType> counterTypeForInstr(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::S_WAIT_LOADCNT:
    return LOAD_CNT;
  case AMDGPU::S_WAIT_EXPCNT:
    return EXP_CNT;
  case AMDGPU::S_WAIT_STORECNT:
    return STORE_CNT;
  case AMDGPU::S_WAIT_SAMPLECNT:
    return SAMPLE_CNT;
  case AMDGPU::S_WAIT_BVHCNT:
    return BVH_CNT;
  case AMDGPU::S_WAIT_DSCNT:
    return DS_CNT;
  case AMDGPU::S_WAIT_KMCNT:
    return KM_CNT;
  case AMDGPU::S_WAIT_XCNT:
    return X_CNT;
  default:
    return {};
  }
}

bool WaitcntGenerator::promoteSoftWaitCnt(MachineInstr *Waitcnt) const {
  unsigned Opcode = SIInstrInfo::getNonSoftWaitcntOpcode(Waitcnt->getOpcode());
  if (Opcode == Waitcnt->getOpcode())
    return false;

  Waitcnt->setDesc(TII->get(Opcode));
  return true;
}

/// Combine consecutive S_WAITCNT and S_WAITCNT_VSCNT instructions that
/// precede \p It and follow \p OldWaitcntInstr and apply any extra waits
/// from \p Wait that were added by previous passes. Currently this pass
/// conservatively assumes that these preexisting waits are required for
/// correctness.
bool WaitcntGeneratorPreGFX12::applyPreexistingWaitcnt(
    WaitcntBrackets &ScoreBrackets, MachineInstr &OldWaitcntInstr,
    AMDGPU::Waitcnt &Wait, MachineBasicBlock::instr_iterator It) const {
  assert(ST);
  assert(isNormalMode(MaxCounter));

  bool Modified = false;
  MachineInstr *WaitcntInstr = nullptr;
  MachineInstr *WaitcntVsCntInstr = nullptr;

  LLVM_DEBUG({
    dbgs() << "PreGFX12::applyPreexistingWaitcnt at: ";
    if (It == OldWaitcntInstr.getParent()->instr_end())
      dbgs() << "end of block\n";
    else
      dbgs() << *It;
  });

  for (auto &II :
       make_early_inc_range(make_range(OldWaitcntInstr.getIterator(), It))) {
    LLVM_DEBUG(dbgs() << "pre-existing iter: " << II);
    if (II.isMetaInstruction()) {
      LLVM_DEBUG(dbgs() << "skipped meta instruction\n");
      continue;
    }

    unsigned Opcode = SIInstrInfo::getNonSoftWaitcntOpcode(II.getOpcode());
    bool TrySimplify = Opcode != II.getOpcode() && !OptNone;

    // Update required wait count. If this is a soft waitcnt (= it was added
    // by an earlier pass), it may be entirely removed.
    if (Opcode == AMDGPU::S_WAITCNT) {
      unsigned IEnc = II.getOperand(0).getImm();
      AMDGPU::Waitcnt OldWait = AMDGPU::decodeWaitcnt(IV, IEnc);
      if (TrySimplify)
        ScoreBrackets.simplifyWaitcnt(OldWait);
      Wait = Wait.combined(OldWait);

      // Merge consecutive waitcnt of the same type by erasing multiples.
      if (WaitcntInstr || (!Wait.hasWaitExceptStoreCnt() && TrySimplify)) {
        II.eraseFromParent();
        Modified = true;
      } else
        WaitcntInstr = &II;
    } else if (Opcode == AMDGPU::S_WAITCNT_lds_direct) {
      assert(ST->hasVMemToLDSLoad());
      LLVM_DEBUG(dbgs() << "Processing S_WAITCNT_lds_direct: " << II
                        << "Before: " << Wait.LoadCnt << '\n';);
      ScoreBrackets.determineWait(LOAD_CNT, FIRST_LDS_VGPR, Wait);
      LLVM_DEBUG(dbgs() << "After: " << Wait.LoadCnt << '\n';);

      // It is possible (but unlikely) that this is the only wait instruction,
      // in which case, we exit this loop without a WaitcntInstr to consume
      // `Wait`. But that works because `Wait` was passed in by reference, and
      // the callee eventually calls createNewWaitcnt on it. We test this
      // possibility in an articial MIR test since such a situation cannot be
      // recreated by running the memory legalizer.
      II.eraseFromParent();
    } else {
      assert(Opcode == AMDGPU::S_WAITCNT_VSCNT);
      assert(II.getOperand(0).getReg() == AMDGPU::SGPR_NULL);

      unsigned OldVSCnt =
          TII->getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      if (TrySimplify)
        ScoreBrackets.simplifyWaitcnt(InstCounterType::STORE_CNT, OldVSCnt);
      Wait.StoreCnt = std::min(Wait.StoreCnt, OldVSCnt);

      if (WaitcntVsCntInstr || (!Wait.hasWaitStoreCnt() && TrySimplify)) {
        II.eraseFromParent();
        Modified = true;
      } else
        WaitcntVsCntInstr = &II;
    }
  }

  if (WaitcntInstr) {
    Modified |= updateOperandIfDifferent(*WaitcntInstr, AMDGPU::OpName::simm16,
                                         AMDGPU::encodeWaitcnt(IV, Wait));
    Modified |= promoteSoftWaitCnt(WaitcntInstr);

    ScoreBrackets.applyWaitcnt(LOAD_CNT, Wait.LoadCnt);
    ScoreBrackets.applyWaitcnt(EXP_CNT, Wait.ExpCnt);
    ScoreBrackets.applyWaitcnt(DS_CNT, Wait.DsCnt);
    Wait.LoadCnt = ~0u;
    Wait.ExpCnt = ~0u;
    Wait.DsCnt = ~0u;

    LLVM_DEBUG(It == WaitcntInstr->getParent()->end()
                   ? dbgs()
                         << "applied pre-existing waitcnt\n"
                         << "New Instr at block end: " << *WaitcntInstr << '\n'
                   : dbgs() << "applied pre-existing waitcnt\n"
                            << "Old Instr: " << *It
                            << "New Instr: " << *WaitcntInstr << '\n');
  }

  if (WaitcntVsCntInstr) {
    Modified |= updateOperandIfDifferent(*WaitcntVsCntInstr,
                                         AMDGPU::OpName::simm16, Wait.StoreCnt);
    Modified |= promoteSoftWaitCnt(WaitcntVsCntInstr);

    ScoreBrackets.applyWaitcnt(STORE_CNT, Wait.StoreCnt);
    Wait.StoreCnt = ~0u;

    LLVM_DEBUG(It == WaitcntVsCntInstr->getParent()->end()
                   ? dbgs() << "applied pre-existing waitcnt\n"
                            << "New Instr at block end: " << *WaitcntVsCntInstr
                            << '\n'
                   : dbgs() << "applied pre-existing waitcnt\n"
                            << "Old Instr: " << *It
                            << "New Instr: " << *WaitcntVsCntInstr << '\n');
  }

  return Modified;
}

/// Generate S_WAITCNT and/or S_WAITCNT_VSCNT instructions for any
/// required counters in \p Wait
bool WaitcntGeneratorPreGFX12::createNewWaitcnt(
    MachineBasicBlock &Block, MachineBasicBlock::instr_iterator It,
    AMDGPU::Waitcnt Wait) {
  assert(ST);
  assert(isNormalMode(MaxCounter));

  bool Modified = false;
  const DebugLoc &DL = Block.findDebugLoc(It);

  // Waits for VMcnt, LKGMcnt and/or EXPcnt are encoded together into a
  // single instruction while VScnt has its own instruction.
  if (Wait.hasWaitExceptStoreCnt()) {
    unsigned Enc = AMDGPU::encodeWaitcnt(IV, Wait);
    [[maybe_unused]] auto SWaitInst =
        BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAITCNT)).addImm(Enc);
    Modified = true;

    LLVM_DEBUG(dbgs() << "generateWaitcnt\n";
               if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
               dbgs() << "New Instr: " << *SWaitInst << '\n');
  }

  if (Wait.hasWaitStoreCnt()) {
    assert(ST->hasVscnt());

    [[maybe_unused]] auto SWaitInst =
        BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAITCNT_VSCNT))
            .addReg(AMDGPU::SGPR_NULL, RegState::Undef)
            .addImm(Wait.StoreCnt);
    Modified = true;

    LLVM_DEBUG(dbgs() << "generateWaitcnt\n";
               if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
               dbgs() << "New Instr: " << *SWaitInst << '\n');
  }

  return Modified;
}

AMDGPU::Waitcnt
WaitcntGeneratorPreGFX12::getAllZeroWaitcnt(bool IncludeVSCnt) const {
  return AMDGPU::Waitcnt(0, 0, 0, IncludeVSCnt && ST->hasVscnt() ? 0 : ~0u);
}

AMDGPU::Waitcnt
WaitcntGeneratorGFX12Plus::getAllZeroWaitcnt(bool IncludeVSCnt) const {
  return AMDGPU::Waitcnt(0, 0, 0, IncludeVSCnt ? 0 : ~0u, 0, 0, 0,
                         ~0u /* XCNT */);
}

/// Combine consecutive S_WAIT_*CNT instructions that precede \p It and
/// follow \p OldWaitcntInstr and apply any extra waits from \p Wait that
/// were added by previous passes. Currently this pass conservatively
/// assumes that these preexisting waits are required for correctness.
bool WaitcntGeneratorGFX12Plus::applyPreexistingWaitcnt(
    WaitcntBrackets &ScoreBrackets, MachineInstr &OldWaitcntInstr,
    AMDGPU::Waitcnt &Wait, MachineBasicBlock::instr_iterator It) const {
  assert(ST);
  assert(!isNormalMode(MaxCounter));

  bool Modified = false;
  MachineInstr *CombinedLoadDsCntInstr = nullptr;
  MachineInstr *CombinedStoreDsCntInstr = nullptr;
  MachineInstr *WaitInstrs[NUM_EXTENDED_INST_CNTS] = {};

  LLVM_DEBUG({
    dbgs() << "GFX12Plus::applyPreexistingWaitcnt at: ";
    if (It == OldWaitcntInstr.getParent()->instr_end())
      dbgs() << "end of block\n";
    else
      dbgs() << *It;
  });

  for (auto &II :
       make_early_inc_range(make_range(OldWaitcntInstr.getIterator(), It))) {
    LLVM_DEBUG(dbgs() << "pre-existing iter: " << II);
    if (II.isMetaInstruction()) {
      LLVM_DEBUG(dbgs() << "skipped meta instruction\n");
      continue;
    }

    MachineInstr **UpdatableInstr;

    // Update required wait count. If this is a soft waitcnt (= it was added
    // by an earlier pass), it may be entirely removed.

    unsigned Opcode = SIInstrInfo::getNonSoftWaitcntOpcode(II.getOpcode());
    bool TrySimplify = Opcode != II.getOpcode() && !OptNone;

    // Don't crash if the programmer used legacy waitcnt intrinsics, but don't
    // attempt to do more than that either.
    if (Opcode == AMDGPU::S_WAITCNT)
      continue;

    if (Opcode == AMDGPU::S_WAIT_LOADCNT_DSCNT) {
      unsigned OldEnc =
          TII->getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      AMDGPU::Waitcnt OldWait = AMDGPU::decodeLoadcntDscnt(IV, OldEnc);
      if (TrySimplify)
        ScoreBrackets.simplifyWaitcnt(OldWait);
      Wait = Wait.combined(OldWait);
      UpdatableInstr = &CombinedLoadDsCntInstr;
    } else if (Opcode == AMDGPU::S_WAIT_STORECNT_DSCNT) {
      unsigned OldEnc =
          TII->getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      AMDGPU::Waitcnt OldWait = AMDGPU::decodeStorecntDscnt(IV, OldEnc);
      if (TrySimplify)
        ScoreBrackets.simplifyWaitcnt(OldWait);
      Wait = Wait.combined(OldWait);
      UpdatableInstr = &CombinedStoreDsCntInstr;
    } else if (Opcode == AMDGPU::S_WAITCNT_lds_direct) {
      // Architectures higher than GFX10 do not have direct loads to
      // LDS, so no work required here yet.
      II.eraseFromParent();
      continue;
    } else {
      std::optional<InstCounterType> CT = counterTypeForInstr(Opcode);
      assert(CT.has_value());
      unsigned OldCnt =
          TII->getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      if (TrySimplify)
        ScoreBrackets.simplifyWaitcnt(CT.value(), OldCnt);
      addWait(Wait, CT.value(), OldCnt);
      UpdatableInstr = &WaitInstrs[CT.value()];
    }

    // Merge consecutive waitcnt of the same type by erasing multiples.
    if (!*UpdatableInstr) {
      *UpdatableInstr = &II;
    } else {
      II.eraseFromParent();
      Modified = true;
    }
  }

  if (CombinedLoadDsCntInstr) {
    // Only keep an S_WAIT_LOADCNT_DSCNT if both counters actually need
    // to be waited for. Otherwise, let the instruction be deleted so
    // the appropriate single counter wait instruction can be inserted
    // instead, when new S_WAIT_*CNT instructions are inserted by
    // createNewWaitcnt(). As a side effect, resetting the wait counts will
    // cause any redundant S_WAIT_LOADCNT or S_WAIT_DSCNT to be removed by
    // the loop below that deals with single counter instructions.
    if (Wait.LoadCnt != ~0u && Wait.DsCnt != ~0u) {
      unsigned NewEnc = AMDGPU::encodeLoadcntDscnt(IV, Wait);
      Modified |= updateOperandIfDifferent(*CombinedLoadDsCntInstr,
                                           AMDGPU::OpName::simm16, NewEnc);
      Modified |= promoteSoftWaitCnt(CombinedLoadDsCntInstr);
      ScoreBrackets.applyWaitcnt(LOAD_CNT, Wait.LoadCnt);
      ScoreBrackets.applyWaitcnt(DS_CNT, Wait.DsCnt);
      Wait.LoadCnt = ~0u;
      Wait.DsCnt = ~0u;

      LLVM_DEBUG(It == OldWaitcntInstr.getParent()->end()
                     ? dbgs() << "applied pre-existing waitcnt\n"
                              << "New Instr at block end: "
                              << *CombinedLoadDsCntInstr << '\n'
                     : dbgs() << "applied pre-existing waitcnt\n"
                              << "Old Instr: " << *It << "New Instr: "
                              << *CombinedLoadDsCntInstr << '\n');
    } else {
      CombinedLoadDsCntInstr->eraseFromParent();
      Modified = true;
    }
  }

  if (CombinedStoreDsCntInstr) {
    // Similarly for S_WAIT_STORECNT_DSCNT.
    if (Wait.StoreCnt != ~0u && Wait.DsCnt != ~0u) {
      unsigned NewEnc = AMDGPU::encodeStorecntDscnt(IV, Wait);
      Modified |= updateOperandIfDifferent(*CombinedStoreDsCntInstr,
                                           AMDGPU::OpName::simm16, NewEnc);
      Modified |= promoteSoftWaitCnt(CombinedStoreDsCntInstr);
      ScoreBrackets.applyWaitcnt(STORE_CNT, Wait.StoreCnt);
      ScoreBrackets.applyWaitcnt(DS_CNT, Wait.DsCnt);
      Wait.StoreCnt = ~0u;
      Wait.DsCnt = ~0u;

      LLVM_DEBUG(It == OldWaitcntInstr.getParent()->end()
                     ? dbgs() << "applied pre-existing waitcnt\n"
                              << "New Instr at block end: "
                              << *CombinedStoreDsCntInstr << '\n'
                     : dbgs() << "applied pre-existing waitcnt\n"
                              << "Old Instr: " << *It << "New Instr: "
                              << *CombinedStoreDsCntInstr << '\n');
    } else {
      CombinedStoreDsCntInstr->eraseFromParent();
      Modified = true;
    }
  }

  // Look for an opportunity to convert existing S_WAIT_LOADCNT,
  // S_WAIT_STORECNT and S_WAIT_DSCNT into new S_WAIT_LOADCNT_DSCNT
  // or S_WAIT_STORECNT_DSCNT. This is achieved by selectively removing
  // instructions so that createNewWaitcnt() will create new combined
  // instructions to replace them.

  if (Wait.DsCnt != ~0u) {
    // This is a vector of addresses in WaitInstrs pointing to instructions
    // that should be removed if they are present.
    SmallVector<MachineInstr **, 2> WaitsToErase;

    // If it's known that both DScnt and either LOADcnt or STOREcnt (but not
    // both) need to be waited for, ensure that there are no existing
    // individual wait count instructions for these.

    if (Wait.LoadCnt != ~0u) {
      WaitsToErase.push_back(&WaitInstrs[LOAD_CNT]);
      WaitsToErase.push_back(&WaitInstrs[DS_CNT]);
    } else if (Wait.StoreCnt != ~0u) {
      WaitsToErase.push_back(&WaitInstrs[STORE_CNT]);
      WaitsToErase.push_back(&WaitInstrs[DS_CNT]);
    }

    for (MachineInstr **WI : WaitsToErase) {
      if (!*WI)
        continue;

      (*WI)->eraseFromParent();
      *WI = nullptr;
      Modified = true;
    }
  }

  for (auto CT : inst_counter_types(NUM_EXTENDED_INST_CNTS)) {
    if (!WaitInstrs[CT])
      continue;

    unsigned NewCnt = getWait(Wait, CT);
    if (NewCnt != ~0u) {
      Modified |= updateOperandIfDifferent(*WaitInstrs[CT],
                                           AMDGPU::OpName::simm16, NewCnt);
      Modified |= promoteSoftWaitCnt(WaitInstrs[CT]);

      ScoreBrackets.applyWaitcnt(CT, NewCnt);
      setNoWait(Wait, CT);

      LLVM_DEBUG(It == OldWaitcntInstr.getParent()->end()
                     ? dbgs() << "applied pre-existing waitcnt\n"
                              << "New Instr at block end: " << *WaitInstrs[CT]
                              << '\n'
                     : dbgs() << "applied pre-existing waitcnt\n"
                              << "Old Instr: " << *It
                              << "New Instr: " << *WaitInstrs[CT] << '\n');
    } else {
      WaitInstrs[CT]->eraseFromParent();
      Modified = true;
    }
  }

  return Modified;
}

/// Generate S_WAIT_*CNT instructions for any required counters in \p Wait
bool WaitcntGeneratorGFX12Plus::createNewWaitcnt(
    MachineBasicBlock &Block, MachineBasicBlock::instr_iterator It,
    AMDGPU::Waitcnt Wait) {
  assert(ST);
  assert(!isNormalMode(MaxCounter));

  bool Modified = false;
  const DebugLoc &DL = Block.findDebugLoc(It);

  // Check for opportunities to use combined wait instructions.
  if (Wait.DsCnt != ~0u) {
    MachineInstr *SWaitInst = nullptr;

    if (Wait.LoadCnt != ~0u) {
      unsigned Enc = AMDGPU::encodeLoadcntDscnt(IV, Wait);

      SWaitInst = BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAIT_LOADCNT_DSCNT))
                      .addImm(Enc);

      Wait.LoadCnt = ~0u;
      Wait.DsCnt = ~0u;
    } else if (Wait.StoreCnt != ~0u) {
      unsigned Enc = AMDGPU::encodeStorecntDscnt(IV, Wait);

      SWaitInst =
          BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAIT_STORECNT_DSCNT))
              .addImm(Enc);

      Wait.StoreCnt = ~0u;
      Wait.DsCnt = ~0u;
    }

    if (SWaitInst) {
      Modified = true;

      LLVM_DEBUG(dbgs() << "generateWaitcnt\n";
                 if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
                 dbgs() << "New Instr: " << *SWaitInst << '\n');
    }
  }

  // Generate an instruction for any remaining counter that needs
  // waiting for.

  for (auto CT : inst_counter_types(NUM_EXTENDED_INST_CNTS)) {
    unsigned Count = getWait(Wait, CT);
    if (Count == ~0u)
      continue;

    [[maybe_unused]] auto SWaitInst =
        BuildMI(Block, It, DL, TII->get(instrsForExtendedCounterTypes[CT]))
            .addImm(Count);

    Modified = true;

    LLVM_DEBUG(dbgs() << "generateWaitcnt\n";
               if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
               dbgs() << "New Instr: " << *SWaitInst << '\n');
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
static bool callWaitsOnFunctionReturn(const MachineInstr &MI) { return true; }

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
bool SIInsertWaitcnts::generateWaitcntInstBefore(MachineInstr &MI,
                                                 WaitcntBrackets &ScoreBrackets,
                                                 MachineInstr *OldWaitcntInstr,
                                                 bool FlushVmCnt) {
  setForceEmitWaitcnt();

  assert(!MI.isMetaInstruction());

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

  // All waits must be resolved at call return.
  // NOTE: this could be improved with knowledge of all call sites or
  //   with knowledge of the called routines.
  if (MI.getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG ||
      MI.getOpcode() == AMDGPU::SI_RETURN ||
      MI.getOpcode() == AMDGPU::SI_WHOLE_WAVE_FUNC_RETURN ||
      MI.getOpcode() == AMDGPU::S_SETPC_B64_return ||
      (MI.isReturn() && MI.isCall() && !callWaitsOnFunctionEntry(MI))) {
    Wait = Wait.combined(WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/false));
  }
  // In dynamic VGPR mode, we want to release the VGPRs before the wave exits.
  // Technically the hardware will do this on its own if we don't, but that
  // might cost extra cycles compared to doing it explicitly.
  // When not in dynamic VGPR mode, identify S_ENDPGM instructions which may
  // have to wait for outstanding VMEM stores. In this case it can be useful to
  // send a message to explicitly release all VGPRs before the stores have
  // completed, but it is only safe to do this if there are no outstanding
  // scratch stores.
  else if (MI.getOpcode() == AMDGPU::S_ENDPGM ||
           MI.getOpcode() == AMDGPU::S_ENDPGM_SAVED) {
    if (!WCG->isOptNone() &&
        (MI.getMF()->getInfo<SIMachineFunctionInfo>()->isDynamicVGPREnabled() ||
         (ST->getGeneration() >= AMDGPUSubtarget::GFX11 &&
          ScoreBrackets.getScoreRange(STORE_CNT) != 0 &&
          !ScoreBrackets.hasPendingEvent(SCRATCH_WRITE_ACCESS))))
      ReleaseVGPRInsts.insert(&MI);
  }
  // Resolve vm waits before gs-done.
  else if ((MI.getOpcode() == AMDGPU::S_SENDMSG ||
            MI.getOpcode() == AMDGPU::S_SENDMSGHALT) &&
           ST->hasLegacyGeometry() &&
           ((MI.getOperand(0).getImm() & AMDGPU::SendMsg::ID_MASK_PreGFX11_) ==
            AMDGPU::SendMsg::ID_GS_DONE_PreGFX11)) {
    Wait.LoadCnt = 0;
  }

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

    // Wait for any pending GDS instruction to complete before any
    // "Always GDS" instruction.
    if (TII->isAlwaysGDS(MI.getOpcode()) && ScoreBrackets.hasPendingGDS())
      addWait(Wait, DS_CNT, ScoreBrackets.getPendingGDSWait());

    if (MI.isCall() && callWaitsOnFunctionEntry(MI)) {
      // The function is going to insert a wait on everything in its prolog.
      // This still needs to be careful if the call target is a load (e.g. a GOT
      // load). We also need to check WAW dependency with saved PC.
      Wait = AMDGPU::Waitcnt();

      const auto &CallAddrOp = *TII->getNamedOperand(MI, AMDGPU::OpName::src0);
      if (CallAddrOp.isReg()) {
        RegInterval CallAddrOpInterval =
            ScoreBrackets.getRegInterval(&MI, MRI, TRI, CallAddrOp);

        ScoreBrackets.determineWait(SmemAccessCounter, CallAddrOpInterval,
                                    Wait);

        if (const auto *RtnAddrOp =
                TII->getNamedOperand(MI, AMDGPU::OpName::dst)) {
          RegInterval RtnAddrOpInterval =
              ScoreBrackets.getRegInterval(&MI, MRI, TRI, *RtnAddrOp);

          ScoreBrackets.determineWait(SmemAccessCounter, RtnAddrOpInterval,
                                      Wait);
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
        if (Memop->isStore()) {
          if (auto It = SLoadAddresses.find(Ptr); It != SLoadAddresses.end()) {
            addWait(Wait, SmemAccessCounter, 0);
            if (PDT->dominates(MI.getParent(), It->second))
              SLoadAddresses.erase(It);
          }
        }
        unsigned AS = Memop->getAddrSpace();
        if (AS != AMDGPUAS::LOCAL_ADDRESS && AS != AMDGPUAS::FLAT_ADDRESS)
          continue;
        // No need to wait before load from VMEM to LDS.
        if (TII->mayWriteLDSThroughDMA(MI))
          continue;

        // LOAD_CNT is only relevant to vgpr or LDS.
        unsigned RegNo = FIRST_LDS_VGPR;
        // Only objects with alias scope info were added to LDSDMAScopes array.
        // In the absense of the scope info we will not be able to disambiguate
        // aliasing here. There is no need to try searching for a corresponding
        // store slot. This is conservatively correct because in that case we
        // will produce a wait using the first (general) LDS DMA wait slot which
        // will wait on all of them anyway.
        if (Ptr && Memop->getAAInfo() && Memop->getAAInfo().Scope) {
          const auto &LDSDMAStores = ScoreBrackets.getLDSDMAStores();
          for (unsigned I = 0, E = LDSDMAStores.size(); I != E; ++I) {
            if (MI.mayAlias(AA, *LDSDMAStores[I], true))
              ScoreBrackets.determineWait(LOAD_CNT, RegNo + I + 1, Wait);
          }
        } else {
          ScoreBrackets.determineWait(LOAD_CNT, RegNo, Wait);
        }
        if (Memop->isStore()) {
          ScoreBrackets.determineWait(EXP_CNT, RegNo, Wait);
        }
      }

      // Loop over use and def operands.
      for (const MachineOperand &Op : MI.operands()) {
        if (!Op.isReg())
          continue;

        // If the instruction does not read tied source, skip the operand.
        if (Op.isTied() && Op.isUse() && TII->doesNotReadTiedSource(MI))
          continue;

        RegInterval Interval = ScoreBrackets.getRegInterval(&MI, MRI, TRI, Op);

        const bool IsVGPR = TRI->isVectorRegister(*MRI, Op.getReg());
        if (IsVGPR) {
          // Implicit VGPR defs and uses are never a part of the memory
          // instructions description and usually present to account for
          // super-register liveness.
          // TODO: Most of the other instructions also have implicit uses
          // for the liveness accounting only.
          if (Op.isImplicit() && MI.mayLoadOrStore())
            continue;

          // RAW always needs an s_waitcnt. WAW needs an s_waitcnt unless the
          // previous write and this write are the same type of VMEM
          // instruction, in which case they are (in some architectures)
          // guaranteed to write their results in order anyway.
          // Additionally check instructions where Point Sample Acceleration
          // might be applied.
          if (Op.isUse() || !updateVMCntOnly(MI) ||
              ScoreBrackets.hasOtherPendingVmemTypes(Interval,
                                                     getVmemType(MI)) ||
              ScoreBrackets.hasPointSamplePendingVmemTypes(MI, Interval) ||
              !ST->hasVmemWriteVgprInOrder()) {
            ScoreBrackets.determineWait(LOAD_CNT, Interval, Wait);
            ScoreBrackets.determineWait(SAMPLE_CNT, Interval, Wait);
            ScoreBrackets.determineWait(BVH_CNT, Interval, Wait);
            ScoreBrackets.clearVgprVmemTypes(Interval);
          }

          if (Op.isDef() || ScoreBrackets.hasPendingEvent(EXP_LDS_ACCESS)) {
            ScoreBrackets.determineWait(EXP_CNT, Interval, Wait);
          }
          ScoreBrackets.determineWait(DS_CNT, Interval, Wait);
        } else {
          ScoreBrackets.determineWait(SmemAccessCounter, Interval, Wait);
        }

        if (hasXcnt() && Op.isDef())
          ScoreBrackets.determineWait(X_CNT, Interval, Wait);
      }
    }
  }

  // The subtarget may have an implicit S_WAITCNT 0 before barriers. If it does
  // not, we need to ensure the subtarget is capable of backing off barrier
  // instructions in case there are any outstanding memory operations that may
  // cause an exception. Otherwise, insert an explicit S_WAITCNT 0 here.
  if (TII->isBarrierStart(MI.getOpcode()) &&
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

  // When forcing emit, we need to skip terminators because that would break the
  // terminators of the MBB if we emit a waitcnt between terminators.
  if (ForceEmitZeroFlag && !MI.isTerminator())
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
  if (ForceEmitWaitcnt[X_CNT])
    Wait.XCnt = 0;

  if (FlushVmCnt) {
    if (ScoreBrackets.hasPendingEvent(LOAD_CNT))
      Wait.LoadCnt = 0;
    if (ScoreBrackets.hasPendingEvent(SAMPLE_CNT))
      Wait.SampleCnt = 0;
    if (ScoreBrackets.hasPendingEvent(BVH_CNT))
      Wait.BvhCnt = 0;
  }

  if (ForceEmitZeroLoadFlag && Wait.LoadCnt != ~0u)
    Wait.LoadCnt = 0;

  return generateWaitcnt(Wait, MI.getIterator(), *MI.getParent(), ScoreBrackets,
                         OldWaitcntInstr);
}

bool SIInsertWaitcnts::generateWaitcnt(AMDGPU::Waitcnt Wait,
                                       MachineBasicBlock::instr_iterator It,
                                       MachineBasicBlock &Block,
                                       WaitcntBrackets &ScoreBrackets,
                                       MachineInstr *OldWaitcntInstr) {
  bool Modified = false;

  if (OldWaitcntInstr)
    // Try to merge the required wait with preexisting waitcnt instructions.
    // Also erase redundant waitcnt.
    Modified =
        WCG->applyPreexistingWaitcnt(ScoreBrackets, *OldWaitcntInstr, Wait, It);

  // Any counts that could have been applied to any existing waitcnt
  // instructions will have been done so, now deal with any remaining.
  ScoreBrackets.applyWaitcnt(Wait);

  // ExpCnt can be merged into VINTERP.
  if (Wait.ExpCnt != ~0u && It != Block.instr_end() &&
      SIInstrInfo::isVINTERP(*It)) {
    MachineOperand *WaitExp =
        TII->getNamedOperand(*It, AMDGPU::OpName::waitexp);
    if (Wait.ExpCnt < WaitExp->getImm()) {
      WaitExp->setImm(Wait.ExpCnt);
      Modified = true;
    }
    Wait.ExpCnt = ~0u;

    LLVM_DEBUG(dbgs() << "generateWaitcnt\n"
                      << "Update Instr: " << *It);
  }

  // XCnt may be already consumed by a load wait.
  if (Wait.KmCnt == 0 && Wait.XCnt != ~0u &&
      !ScoreBrackets.hasPendingEvent(SMEM_GROUP))
    Wait.XCnt = ~0u;

  if (Wait.LoadCnt == 0 && Wait.XCnt != ~0u &&
      !ScoreBrackets.hasPendingEvent(VMEM_GROUP))
    Wait.XCnt = ~0u;

  // Since the translation for VMEM addresses occur in-order, we can skip the
  // XCnt if the current instruction is of VMEM type and has a memory dependency
  // with another VMEM instruction in flight.
  if (Wait.XCnt != ~0u && isVmemAccess(*It))
    Wait.XCnt = ~0u;

  if (WCG->createNewWaitcnt(Block, It, Wait))
    Modified = true;

  return Modified;
}

// This is a flat memory operation. Check to see if it has memory tokens other
// than LDS. Other address spaces supported by flat memory operations involve
// global memory.
bool SIInsertWaitcnts::mayAccessVMEMThroughFlat(const MachineInstr &MI) const {
  assert(TII->isFLAT(MI));

  // All flat instructions use the VMEM counter except prefetch.
  if (!TII->usesVM_CNT(MI))
    return false;

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

bool SIInsertWaitcnts::isVmemAccess(const MachineInstr &MI) const {
  return (TII->isFLAT(MI) && mayAccessVMEMThroughFlat(MI)) ||
         (TII->isVMEM(MI) && !AMDGPU::getMUBUFIsBufferInv(MI.getOpcode()));
}

static bool isGFX12CacheInvOrWBInst(MachineInstr &Inst) {
  auto Opc = Inst.getOpcode();
  return Opc == AMDGPU::GLOBAL_INV || Opc == AMDGPU::GLOBAL_WB ||
         Opc == AMDGPU::GLOBAL_WBINV;
}

// Return true if the next instruction is S_ENDPGM, following fallthrough
// blocks if necessary.
bool SIInsertWaitcnts::isNextENDPGM(MachineBasicBlock::instr_iterator It,
                                    MachineBasicBlock *Block) const {
  auto BlockEnd = Block->getParent()->end();
  auto BlockIter = Block->getIterator();

  while (true) {
    if (It.isEnd()) {
      if (++BlockIter != BlockEnd) {
        It = BlockIter->instr_begin();
        continue;
      }

      return false;
    }

    if (!It->isMetaInstruction())
      break;

    It++;
  }

  assert(!It.isEnd());

  return It->getOpcode() == AMDGPU::S_ENDPGM;
}

// Add a wait after an instruction if architecture requirements mandate one.
bool SIInsertWaitcnts::insertForcedWaitAfter(MachineInstr &Inst,
                                             MachineBasicBlock &Block,
                                             WaitcntBrackets &ScoreBrackets) {
  AMDGPU::Waitcnt Wait;
  bool NeedsEndPGMCheck = false;

  if (ST->isPreciseMemoryEnabled() && Inst.mayLoadOrStore())
    Wait = WCG->getAllZeroWaitcnt(Inst.mayStore() &&
                                  !SIInstrInfo::isAtomicRet(Inst));

  if (TII->isAlwaysGDS(Inst.getOpcode())) {
    Wait.DsCnt = 0;
    NeedsEndPGMCheck = true;
  }

  ScoreBrackets.simplifyWaitcnt(Wait);

  auto SuccessorIt = std::next(Inst.getIterator());
  bool Result = generateWaitcnt(Wait, SuccessorIt, Block, ScoreBrackets,
                                /*OldWaitcntInstr=*/nullptr);

  if (Result && NeedsEndPGMCheck && isNextENDPGM(SuccessorIt, &Block)) {
    BuildMI(Block, SuccessorIt, Inst.getDebugLoc(), TII->get(AMDGPU::S_NOP))
        .addImm(0);
  }

  return Result;
}

void SIInsertWaitcnts::updateEventWaitcntAfter(MachineInstr &Inst,
                                               WaitcntBrackets *ScoreBrackets) {
  // Now look at the instruction opcode. If it is a memory access
  // instruction, update the upper-bound of the appropriate counter's
  // bracket and the destination operand scores.
  // TODO: Use the (TSFlags & SIInstrFlags::DS_CNT) property everywhere.

  bool IsVMEMAccess = false;
  bool IsSMEMAccess = false;
  if (TII->isDS(Inst) && TII->usesLGKM_CNT(Inst)) {
    if (TII->isAlwaysGDS(Inst.getOpcode()) ||
        TII->hasModifiersSet(Inst, AMDGPU::OpName::gds)) {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, GDS_ACCESS, Inst);
      ScoreBrackets->updateByEvent(TII, TRI, MRI, GDS_GPR_LOCK, Inst);
      ScoreBrackets->setPendingGDS();
    } else {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, LDS_ACCESS, Inst);
    }
  } else if (TII->isFLAT(Inst)) {
    if (isGFX12CacheInvOrWBInst(Inst)) {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, getVmemWaitEventType(Inst),
                                   Inst);
      return;
    }

    assert(Inst.mayLoadOrStore());

    int FlatASCount = 0;

    if (mayAccessVMEMThroughFlat(Inst)) {
      ++FlatASCount;
      IsVMEMAccess = true;
      ScoreBrackets->updateByEvent(TII, TRI, MRI, getVmemWaitEventType(Inst),
                                   Inst);
    }

    if (mayAccessLDSThroughFlat(Inst)) {
      ++FlatASCount;
      ScoreBrackets->updateByEvent(TII, TRI, MRI, LDS_ACCESS, Inst);
    }

    // This is a flat memory operation that access both VMEM and LDS, so note it
    // - it will require that both the VM and LGKM be flushed to zero if it is
    // pending when a VM or LGKM dependency occurs.
    if (FlatASCount > 1)
      ScoreBrackets->setPendingFlat();
  } else if (SIInstrInfo::isVMEM(Inst) &&
             !llvm::AMDGPU::getMUBUFIsBufferInv(Inst.getOpcode())) {
    IsVMEMAccess = true;
    ScoreBrackets->updateByEvent(TII, TRI, MRI, getVmemWaitEventType(Inst),
                                 Inst);

    if (ST->vmemWriteNeedsExpWaitcnt() &&
        (Inst.mayStore() || SIInstrInfo::isAtomicRet(Inst))) {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, VMW_GPR_LOCK, Inst);
    }
  } else if (TII->isSMRD(Inst)) {
    IsSMEMAccess = true;
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

  if (!hasXcnt())
    return;

  if (IsVMEMAccess)
    ScoreBrackets->updateByEvent(TII, TRI, MRI, VMEM_GROUP, Inst);

  if (IsSMEMAccess)
    ScoreBrackets->updateByEvent(TII, TRI, MRI, SMEM_GROUP, Inst);
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

  for (auto T : inst_counter_types(Context->MaxCounter)) {
    // Merge event flags for this counter
    const unsigned *WaitEventMaskForInst = Context->WaitEventMaskForInst;
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

    if (T == DS_CNT)
      StrictDom |= mergeScore(M, LastGDS, Other.LastGDS);

    for (int J = 0; J <= VgprUB; J++)
      StrictDom |= mergeScore(M, VgprScores[T][J], Other.VgprScores[T][J]);

    if (isSmemCounter(T)) {
      unsigned Idx = getSgprScoresIdx(T);
      for (int J = 0; J <= SgprUB; J++)
        StrictDom |=
            mergeScore(M, SgprScores[Idx][J], Other.SgprScores[Idx][J]);
    }
  }

  for (int J = 0; J <= VgprUB; J++) {
    unsigned char NewVmemTypes = VgprVmemTypes[J] | Other.VgprVmemTypes[J];
    StrictDom |= NewVmemTypes != VgprVmemTypes[J];
    VgprVmemTypes[J] = NewVmemTypes;
  }

  return StrictDom;
}

static bool isWaitInstr(MachineInstr &Inst) {
  unsigned Opcode = SIInstrInfo::getNonSoftWaitcntOpcode(Inst.getOpcode());
  return Opcode == AMDGPU::S_WAITCNT ||
         (Opcode == AMDGPU::S_WAITCNT_VSCNT && Inst.getOperand(0).isReg() &&
          Inst.getOperand(0).getReg() == AMDGPU::SGPR_NULL) ||
         Opcode == AMDGPU::S_WAIT_LOADCNT_DSCNT ||
         Opcode == AMDGPU::S_WAIT_STORECNT_DSCNT ||
         Opcode == AMDGPU::S_WAITCNT_lds_direct ||
         counterTypeForInstr(Opcode).has_value();
}

// Generate s_waitcnt instructions where needed.
bool SIInsertWaitcnts::insertWaitcntInBlock(MachineFunction &MF,
                                            MachineBasicBlock &Block,
                                            WaitcntBrackets &ScoreBrackets) {
  bool Modified = false;

  LLVM_DEBUG({
    dbgs() << "*** Begin Block: ";
    Block.printName(dbgs());
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
    if (Inst.isMetaInstruction()) {
      ++Iter;
      continue;
    }

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
                                          FlushVmCnt);
    OldWaitcntInstr = nullptr;

    // Restore vccz if it's not known to be correct already.
    bool RestoreVCCZ = !VCCZCorrect && readsVCCZ(Inst);

    // Don't examine operands unless we need to track vccz correctness.
    if (ST->hasReadVCCZBug() || !ST->partialVCCWritesUpdateVCCZ()) {
      if (Inst.definesRegister(AMDGPU::VCC_LO, /*TRI=*/nullptr) ||
          Inst.definesRegister(AMDGPU::VCC_HI, /*TRI=*/nullptr)) {
        // Up to gfx9, writes to vcc_lo and vcc_hi don't update vccz.
        if (!ST->partialVCCWritesUpdateVCCZ())
          VCCZCorrect = false;
      } else if (Inst.definesRegister(AMDGPU::VCC, /*TRI=*/nullptr)) {
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

    Modified |= insertForcedWaitAfter(Inst, Block, ScoreBrackets);

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

  // Flush the LOADcnt, SAMPLEcnt and BVHcnt counters at the end of the block if
  // needed.
  AMDGPU::Waitcnt Wait;
  if (Block.getFirstTerminator() == Block.end() &&
      isPreheaderToFlush(Block, ScoreBrackets)) {
    if (ScoreBrackets.hasPendingEvent(LOAD_CNT))
      Wait.LoadCnt = 0;
    if (ScoreBrackets.hasPendingEvent(SAMPLE_CNT))
      Wait.SampleCnt = 0;
    if (ScoreBrackets.hasPendingEvent(BVH_CNT))
      Wait.BvhCnt = 0;
  }

  // Combine or remove any redundant waitcnts at the end of the block.
  Modified |= generateWaitcnt(Wait, Block.instr_end(), Block, ScoreBrackets,
                              OldWaitcntInstr);

  LLVM_DEBUG({
    dbgs() << "*** End Block: ";
    Block.printName(dbgs());
    ScoreBrackets.dump();
  });

  return Modified;
}

// Return true if the given machine basic block is a preheader of a loop in
// which we want to flush the vmcnt counter, and false otherwise.
bool SIInsertWaitcnts::isPreheaderToFlush(
    MachineBasicBlock &MBB, const WaitcntBrackets &ScoreBrackets) {
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

bool SIInsertWaitcnts::isVMEMOrFlatVMEM(const MachineInstr &MI) const {
  if (SIInstrInfo::isFLAT(MI))
    return mayAccessVMEMThroughFlat(MI);
  return SIInstrInfo::isVMEM(MI);
}

// Return true if it is better to flush the vmcnt counter in the preheader of
// the given loop. We currently decide to flush in two situations:
// 1. The loop contains vmem store(s), no vmem load and at least one use of a
//    vgpr containing a value that is loaded outside of the loop. (Only on
//    targets with no vscnt counter).
// 2. The loop contains vmem load(s), but the loaded values are not used in the
//    loop, and at least one use of a vgpr containing a value that is loaded
//    outside of the loop.
bool SIInsertWaitcnts::shouldFlushVmCnt(MachineLoop *ML,
                                        const WaitcntBrackets &Brackets) {
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
      for (const MachineOperand &Op : MI.all_uses()) {
        if (!TRI->isVectorRegister(*MRI, Op.getReg()))
          continue;
        RegInterval Interval = Brackets.getRegInterval(&MI, MRI, TRI, Op);
        // Vgpr use
        for (int RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
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
      if (isVMEMOrFlatVMEM(MI) && MI.mayLoad()) {
        for (const MachineOperand &Op : MI.all_defs()) {
          RegInterval Interval = Brackets.getRegInterval(&MI, MRI, TRI, Op);
          for (int RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
            // If we find a register that is loaded inside the loop, 1. and 2.
            // are invalidated and we can exit.
            if (VgprUse.contains(RegNo))
              return false;
            VgprDef.insert(RegNo);
          }
        }
      }
    }
  }
  if (!ST->hasVscnt() && HasVMemStore && !HasVMemLoad && UsesVgprLoadedOutside)
    return true;
  return HasVMemLoad && UsesVgprLoadedOutside && ST->hasVmemWriteVgprInOrder();
}

bool SIInsertWaitcntsLegacy::runOnMachineFunction(MachineFunction &MF) {
  auto *MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  auto *PDT =
      &getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
  AliasAnalysis *AA = nullptr;
  if (auto *AAR = getAnalysisIfAvailable<AAResultsWrapperPass>())
    AA = &AAR->getAAResults();

  return SIInsertWaitcnts(MLI, PDT, AA).run(MF);
}

PreservedAnalyses
SIInsertWaitcntsPass::run(MachineFunction &MF,
                          MachineFunctionAnalysisManager &MFAM) {
  auto *MLI = &MFAM.getResult<MachineLoopAnalysis>(MF);
  auto *PDT = &MFAM.getResult<MachinePostDominatorTreeAnalysis>(MF);
  auto *AA = MFAM.getResult<FunctionAnalysisManagerMachineFunctionProxy>(MF)
                 .getManager()
                 .getCachedResult<AAManager>(MF.getFunction());

  if (!SIInsertWaitcnts(MLI, PDT, AA).run(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses()
      .preserveSet<CFGAnalyses>()
      .preserve<AAManager>();
}

bool SIInsertWaitcnts::run(MachineFunction &MF) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  TII = ST->getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  AMDGPU::IsaVersion IV = AMDGPU::getIsaVersion(ST->getCPU());

  if (ST->hasExtendedWaitCounts()) {
    MaxCounter = NUM_EXTENDED_INST_CNTS;
    WCGGFX12Plus = WaitcntGeneratorGFX12Plus(MF, MaxCounter);
    WCG = &WCGGFX12Plus;
  } else {
    MaxCounter = NUM_NORMAL_INST_CNTS;
    WCGPreGFX12 = WaitcntGeneratorPreGFX12(MF);
    WCG = &WCGPreGFX12;
  }

  for (auto T : inst_counter_types())
    ForceEmitWaitcnt[T] = false;

  WaitEventMaskForInst = WCG->getWaitEventMask();

  SmemAccessCounter = eventCounter(WaitEventMaskForInst, SMEM_ACCESS);

  if (ST->hasExtendedWaitCounts()) {
    Limits.LoadcntMax = AMDGPU::getLoadcntBitMask(IV);
    Limits.DscntMax = AMDGPU::getDscntBitMask(IV);
  } else {
    Limits.LoadcntMax = AMDGPU::getVmcntBitMask(IV);
    Limits.DscntMax = AMDGPU::getLgkmcntBitMask(IV);
  }
  Limits.ExpcntMax = AMDGPU::getExpcntBitMask(IV);
  Limits.StorecntMax = AMDGPU::getStorecntBitMask(IV);
  Limits.SamplecntMax = AMDGPU::getSamplecntBitMask(IV);
  Limits.BvhcntMax = AMDGPU::getBvhcntBitMask(IV);
  Limits.KmcntMax = AMDGPU::getKmcntBitMask(IV);
  Limits.XcntMax = AMDGPU::getXcntBitMask(IV);

  [[maybe_unused]] unsigned NumVGPRsMax =
      ST->getAddressableNumVGPRs(MFI->getDynamicVGPRBlockSize());
  [[maybe_unused]] unsigned NumSGPRsMax = ST->getAddressableNumSGPRs();
  assert(NumVGPRsMax <= SQ_MAX_PGM_VGPRS);
  assert(NumSGPRsMax <= SQ_MAX_PGM_SGPRS);

  BlockInfos.clear();
  bool Modified = false;

  MachineBasicBlock &EntryBB = MF.front();
  MachineBasicBlock::iterator I = EntryBB.begin();

  if (!MFI->isEntryFunction()) {
    // Wait for any outstanding memory operations that the input registers may
    // depend on. We can't track them and it's better to do the wait after the
    // costly call sequence.

    // TODO: Could insert earlier and schedule more liberally with operations
    // that only use caller preserved registers.
    for (MachineBasicBlock::iterator E = EntryBB.end();
         I != E && (I->isPHI() || I->isMetaInstruction()); ++I)
      ;

    if (ST->hasExtendedWaitCounts()) {
      BuildMI(EntryBB, I, DebugLoc(), TII->get(AMDGPU::S_WAIT_LOADCNT_DSCNT))
          .addImm(0);
      for (auto CT : inst_counter_types(NUM_EXTENDED_INST_CNTS)) {
        if (CT == LOAD_CNT || CT == DS_CNT || CT == STORE_CNT || CT == X_CNT)
          continue;

        if (!ST->hasImageInsts() &&
            (CT == EXP_CNT || CT == SAMPLE_CNT || CT == BVH_CNT))
          continue;

        BuildMI(EntryBB, I, DebugLoc(),
                TII->get(instrsForExtendedCounterTypes[CT]))
            .addImm(0);
      }
    } else {
      BuildMI(EntryBB, I, DebugLoc(), TII->get(AMDGPU::S_WAITCNT)).addImm(0);
    }

    auto NonKernelInitialState = std::make_unique<WaitcntBrackets>(this);
    NonKernelInitialState->setStateOnFunctionEntryOrReturn();
    BlockInfos[&EntryBB].Incoming = std::move(NonKernelInitialState);

    Modified = true;
  }

  // Keep iterating over the blocks in reverse post order, inserting and
  // updating s_waitcnt where needed, until a fix point is reached.
  for (auto *MBB : ReversePostOrderTraversal<MachineFunction *>(&MF))
    BlockInfos.try_emplace(MBB);

  std::unique_ptr<WaitcntBrackets> Brackets;
  bool Repeat;
  do {
    Repeat = false;

    for (auto BII = BlockInfos.begin(), BIE = BlockInfos.end(); BII != BIE;
         ++BII) {
      MachineBasicBlock *MBB = BII->first;
      BlockInfo &BI = BII->second;
      if (!BI.Dirty)
        continue;

      if (BI.Incoming) {
        if (!Brackets)
          Brackets = std::make_unique<WaitcntBrackets>(*BI.Incoming);
        else
          *Brackets = *BI.Incoming;
      } else {
        if (!Brackets) {
          Brackets = std::make_unique<WaitcntBrackets>(this);
        } else {
          // Reinitialize in-place. N.B. do not do this by assigning from a
          // temporary because the WaitcntBrackets class is large and it could
          // cause this function to use an unreasonable amount of stack space.
          Brackets->~WaitcntBrackets();
          new (Brackets.get()) WaitcntBrackets(this);
        }
      }

      Modified |= insertWaitcntInBlock(MF, *MBB, *Brackets);
      BI.Dirty = false;

      if (Brackets->hasPendingEvent()) {
        BlockInfo *MoveBracketsToSucc = nullptr;
        for (MachineBasicBlock *Succ : MBB->successors()) {
          auto *SuccBII = BlockInfos.find(Succ);
          BlockInfo &SuccBI = SuccBII->second;
          if (!SuccBI.Incoming) {
            SuccBI.Dirty = true;
            if (SuccBII <= BII) {
              LLVM_DEBUG(dbgs() << "repeat on backedge\n");
              Repeat = true;
            }
            if (!MoveBracketsToSucc) {
              MoveBracketsToSucc = &SuccBI;
            } else {
              SuccBI.Incoming = std::make_unique<WaitcntBrackets>(*Brackets);
            }
          } else if (SuccBI.Incoming->merge(*Brackets)) {
            SuccBI.Dirty = true;
            if (SuccBII <= BII) {
              LLVM_DEBUG(dbgs() << "repeat on backedge\n");
              Repeat = true;
            }
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

  // Deallocate the VGPRs before previously identified S_ENDPGM instructions.
  // This is done in different ways depending on how the VGPRs were allocated
  // (i.e. whether we're in dynamic VGPR mode or not).
  // Skip deallocation if kernel is waveslot limited vs VGPR limited. A short
  // waveslot limited kernel runs slower with the deallocation.
  if (MFI->isDynamicVGPREnabled()) {
    for (MachineInstr *MI : ReleaseVGPRInsts) {
      BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
              TII->get(AMDGPU::S_ALLOC_VGPR))
          .addImm(0);
      Modified = true;
    }
  } else {
    if (!ReleaseVGPRInsts.empty() &&
        (MF.getFrameInfo().hasCalls() ||
         ST->getOccupancyWithNumVGPRs(
             TRI->getNumUsedPhysRegs(*MRI, AMDGPU::VGPR_32RegClass),
             /*IsDynamicVGPR=*/false) <
             AMDGPU::IsaInfo::getMaxWavesPerEU(ST))) {
      for (MachineInstr *MI : ReleaseVGPRInsts) {
        if (ST->requiresNopBeforeDeallocVGPRs()) {
          BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
                  TII->get(AMDGPU::S_NOP))
              .addImm(0);
        }
        BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
                TII->get(AMDGPU::S_SENDMSG))
            .addImm(AMDGPU::SendMsg::ID_DEALLOC_VGPRS_GFX11Plus);
        Modified = true;
      }
    }
  }
  ReleaseVGPRInsts.clear();
  PreheadersToFlush.clear();
  SLoadAddresses.clear();

  return Modified;
}
