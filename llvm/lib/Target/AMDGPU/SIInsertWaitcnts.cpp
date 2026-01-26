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

static cl::opt<bool> ExpertSchedulingModeFlag(
    "amdgpu-expert-scheduling-mode",
    cl::desc("Enable expert scheduling mode 2 for all functions (GFX12+ only)"),
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
  VA_VDST = NUM_EXTENDED_INST_CNTS, // gfx12+ expert mode only.
  VM_VSRC,                          // gfx12+ expert mode only.
  NUM_EXPERT_INST_CNTS,
  NUM_INST_CNTS = NUM_EXPERT_INST_CNTS
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

// Get the maximum wait count value for a given counter type.
static unsigned getWaitCountMax(const AMDGPU::HardwareLimits &Limits,
                                InstCounterType T) {
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
  case VA_VDST:
    return Limits.VaVdstMax;
  case VM_VSRC:
    return Limits.VmVsrcMax;
  default:
    return 0;
  }
}

static bool isSoftXcnt(MachineInstr &MI) {
  return MI.getOpcode() == AMDGPU::S_WAIT_XCNT_soft;
}

static bool isAtomicRMW(MachineInstr &MI) {
  return (MI.getDesc().TSFlags & SIInstrFlags::maybeAtomic) && MI.mayLoad() &&
         MI.mayStore();
}

enum class AtomicRMWState {
  NewBlock,    // Start of a new atomic RMW block
  InsideBlock, // Middle of an existing block
  NotInBlock   // Not in an atomic RMW block
};

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

#define AMDGPU_DECLARE_WAIT_EVENTS(DECL)                                       \
  DECL(VMEM_ACCESS) /* vmem read & write (pre-gfx10), vmem read (gfx10+) */    \
  DECL(VMEM_SAMPLER_READ_ACCESS) /* vmem SAMPLER read (gfx12+ only) */         \
  DECL(VMEM_BVH_READ_ACCESS)     /* vmem BVH read (gfx12+ only) */             \
  DECL(GLOBAL_INV_ACCESS)        /* GLOBAL_INV (gfx12+ only) */                \
  DECL(VMEM_WRITE_ACCESS)        /* vmem write that is not scratch */          \
  DECL(SCRATCH_WRITE_ACCESS)     /* vmem write that may be scratch */          \
  DECL(VMEM_GROUP)               /* vmem group */                              \
  DECL(LDS_ACCESS)               /* lds read & write */                        \
  DECL(GDS_ACCESS)               /* gds read & write */                        \
  DECL(SQ_MESSAGE)               /* send message */                            \
  DECL(SCC_WRITE)                /* write to SCC from barrier */               \
  DECL(SMEM_ACCESS)              /* scalar-memory read & write */              \
  DECL(SMEM_GROUP)               /* scalar-memory group */                     \
  DECL(EXP_GPR_LOCK)             /* export holding on its data src */          \
  DECL(GDS_GPR_LOCK)             /* GDS holding on its data and addr src */    \
  DECL(EXP_POS_ACCESS)           /* write to export position */                \
  DECL(EXP_PARAM_ACCESS)         /* write to export parameter */               \
  DECL(VMW_GPR_LOCK)             /* vmem write holding on its data src */      \
  DECL(EXP_LDS_ACCESS)           /* read by ldsdir counting as export */       \
  DECL(VGPR_CSMACC_WRITE)        /* write VGPR dest in Core/Side-MACC VALU */  \
  DECL(VGPR_DPMACC_WRITE)        /* write VGPR dest in DPMACC VALU */          \
  DECL(VGPR_TRANS_WRITE)         /* write VGPR dest in TRANS VALU */           \
  DECL(VGPR_XDL_WRITE)           /* write VGPR dest in XDL VALU */             \
  DECL(VGPR_LDS_READ)            /* read VGPR source in LDS */                 \
  DECL(VGPR_FLAT_READ)           /* read VGPR source in FLAT */                \
  DECL(VGPR_VMEM_READ)           /* read VGPR source in other VMEM */

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
// returns true, and does not cover VA_VDST or VM_VSRC.
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
  case VA_VDST:
    return Wait.VaVdst;
  case VM_VSRC:
    return Wait.VmVsrc;
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
  bool ExpandWaitcntProfiling = false;
  const AMDGPU::HardwareLimits *Limits = nullptr;

public:
  WaitcntGenerator() = delete;
  WaitcntGenerator(const WaitcntGenerator &) = delete;
  WaitcntGenerator(const MachineFunction &MF, InstCounterType MaxCounter,
                   const AMDGPU::HardwareLimits *Limits)
      : ST(&MF.getSubtarget<GCNSubtarget>()), TII(ST->getInstrInfo()),
        IV(AMDGPU::getIsaVersion(ST->getCPU())), MaxCounter(MaxCounter),
        OptNone(MF.getFunction().hasOptNone() ||
                MF.getTarget().getOptLevel() == CodeGenOptLevel::None),
        ExpandWaitcntProfiling(
            MF.getFunction().hasFnAttribute("amdgpu-expand-waitcnt-profiling")),
        Limits(Limits) {}

  // Return true if the current function should be compiled with no
  // optimization.
  bool isOptNone() const { return OptNone; }

  const AMDGPU::HardwareLimits &getLimits() const { return *Limits; }

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

  // Generates new wait count instructions according to the value of
  // Wait, returning true if any new instructions were created.
  // ScoreBrackets is used for profiling expansion.
  virtual bool createNewWaitcnt(MachineBasicBlock &Block,
                                MachineBasicBlock::instr_iterator It,
                                AMDGPU::Waitcnt Wait,
                                const WaitcntBrackets &ScoreBrackets) = 0;

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

class WaitcntGeneratorPreGFX12 final : public WaitcntGenerator {
  static constexpr const unsigned WaitEventMaskForInstPreGFX12[NUM_INST_CNTS] =
      {eventMask({VMEM_ACCESS, VMEM_SAMPLER_READ_ACCESS, VMEM_BVH_READ_ACCESS}),
       eventMask({SMEM_ACCESS, LDS_ACCESS, GDS_ACCESS, SQ_MESSAGE}),
       eventMask({EXP_GPR_LOCK, GDS_GPR_LOCK, VMW_GPR_LOCK, EXP_PARAM_ACCESS,
                  EXP_POS_ACCESS, EXP_LDS_ACCESS}),
       eventMask({VMEM_WRITE_ACCESS, SCRATCH_WRITE_ACCESS}),
       0,
       0,
       0,
       0,
       0,
       0};

public:
  using WaitcntGenerator::WaitcntGenerator;
  bool
  applyPreexistingWaitcnt(WaitcntBrackets &ScoreBrackets,
                          MachineInstr &OldWaitcntInstr, AMDGPU::Waitcnt &Wait,
                          MachineBasicBlock::instr_iterator It) const override;

  bool createNewWaitcnt(MachineBasicBlock &Block,
                        MachineBasicBlock::instr_iterator It,
                        AMDGPU::Waitcnt Wait,
                        const WaitcntBrackets &ScoreBrackets) override;

  const unsigned *getWaitEventMask() const override {
    assert(ST);
    return WaitEventMaskForInstPreGFX12;
  }

  AMDGPU::Waitcnt getAllZeroWaitcnt(bool IncludeVSCnt) const override;
};

class WaitcntGeneratorGFX12Plus final : public WaitcntGenerator {
protected:
  bool IsExpertMode;
  static constexpr const unsigned WaitEventMaskForInstGFX12Plus[NUM_INST_CNTS] =
      {eventMask({VMEM_ACCESS, GLOBAL_INV_ACCESS}),
       eventMask({LDS_ACCESS, GDS_ACCESS}),
       eventMask({EXP_GPR_LOCK, GDS_GPR_LOCK, VMW_GPR_LOCK, EXP_PARAM_ACCESS,
                  EXP_POS_ACCESS, EXP_LDS_ACCESS}),
       eventMask({VMEM_WRITE_ACCESS, SCRATCH_WRITE_ACCESS}),
       eventMask({VMEM_SAMPLER_READ_ACCESS}),
       eventMask({VMEM_BVH_READ_ACCESS}),
       eventMask({SMEM_ACCESS, SQ_MESSAGE, SCC_WRITE}),
       eventMask({VMEM_GROUP, SMEM_GROUP}),
       eventMask({VGPR_CSMACC_WRITE, VGPR_DPMACC_WRITE, VGPR_TRANS_WRITE,
                  VGPR_XDL_WRITE}),
       eventMask({VGPR_LDS_READ, VGPR_FLAT_READ, VGPR_VMEM_READ})};

public:
  WaitcntGeneratorGFX12Plus() = delete;
  WaitcntGeneratorGFX12Plus(const MachineFunction &MF,
                            InstCounterType MaxCounter,
                            const AMDGPU::HardwareLimits *Limits,
                            bool IsExpertMode)
      : WaitcntGenerator(MF, MaxCounter, Limits), IsExpertMode(IsExpertMode) {}

  bool
  applyPreexistingWaitcnt(WaitcntBrackets &ScoreBrackets,
                          MachineInstr &OldWaitcntInstr, AMDGPU::Waitcnt &Wait,
                          MachineBasicBlock::instr_iterator It) const override;

  bool createNewWaitcnt(MachineBasicBlock &Block,
                        MachineBasicBlock::instr_iterator It,
                        AMDGPU::Waitcnt Wait,
                        const WaitcntBrackets &ScoreBrackets) override;

  const unsigned *getWaitEventMask() const override {
    assert(ST);
    return WaitEventMaskForInstGFX12Plus;
  }

  AMDGPU::Waitcnt getAllZeroWaitcnt(bool IncludeVSCnt) const override;
};

// Flags indicating which counters should be flushed in a loop preheader.
struct PreheaderFlushFlags {
  bool FlushVmCnt = false;
  bool FlushDsCnt = false;
};

class SIInsertWaitcnts {
public:
  const GCNSubtarget *ST;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  InstCounterType SmemAccessCounter;
  InstCounterType MaxCounter;
  bool IsExpertMode = false;
  const unsigned *WaitEventMaskForInst;

private:
  DenseMap<const Value *, MachineBasicBlock *> SLoadAddresses;
  DenseMap<MachineBasicBlock *, PreheaderFlushFlags> PreheadersToFlush;
  MachineLoopInfo *MLI;
  MachinePostDominatorTree *PDT;
  AliasAnalysis *AA = nullptr;

  struct BlockInfo {
    std::unique_ptr<WaitcntBrackets> Incoming;
    bool Dirty = true;
  };

  MapVector<MachineBasicBlock *, BlockInfo> BlockInfos;

  bool ForceEmitWaitcnt[NUM_INST_CNTS];

  std::unique_ptr<WaitcntGenerator> WCG;

  // Remember call and return instructions in the function.
  DenseSet<MachineInstr *> CallInsts;
  DenseSet<MachineInstr *> ReturnInsts;

  // S_ENDPGM instructions before which we should insert a DEALLOC_VGPRS
  // message.
  DenseSet<MachineInstr *> ReleaseVGPRInsts;

  AMDGPU::HardwareLimits Limits;

public:
  SIInsertWaitcnts(MachineLoopInfo *MLI, MachinePostDominatorTree *PDT,
                   AliasAnalysis *AA)
      : MLI(MLI), PDT(PDT), AA(AA) {
    (void)ForceExpCounter;
    (void)ForceLgkmCounter;
    (void)ForceVMCounter;
  }

  const AMDGPU::HardwareLimits &getLimits() const { return Limits; }

  PreheaderFlushFlags getPreheaderFlushFlags(MachineLoop *ML,
                                             const WaitcntBrackets &Brackets);
  PreheaderFlushFlags isPreheaderToFlush(MachineBasicBlock &MBB,
                                         const WaitcntBrackets &ScoreBrackets);
  bool isVMEMOrFlatVMEM(const MachineInstr &MI) const;
  bool isDSRead(const MachineInstr &MI) const;
  bool mayStoreIncrementingDSCNT(const MachineInstr &MI) const;
  bool run(MachineFunction &MF);

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

    ForceEmitWaitcnt[VA_VDST] = false;
    ForceEmitWaitcnt[VM_VSRC] = false;
#endif // NDEBUG
  }

  // Return the appropriate VMEM_*_ACCESS type for Inst, which must be a VMEM
  // instruction.
  WaitEventType getVmemWaitEventType(const MachineInstr &Inst) const {
    switch (Inst.getOpcode()) {
    // FIXME: GLOBAL_INV needs to be tracked with xcnt too.
    case AMDGPU::GLOBAL_INV:
      return GLOBAL_INV_ACCESS; // tracked using loadcnt, but doesn't write
                                // VGPRs
    case AMDGPU::GLOBAL_WB:
    case AMDGPU::GLOBAL_WBINV:
      return VMEM_WRITE_ACCESS; // tracked using storecnt
    default:
      break;
    }

    // Maps VMEM access types to their corresponding WaitEventType.
    static const WaitEventType VmemReadMapping[NUM_VMEM_TYPES] = {
        VMEM_ACCESS, VMEM_SAMPLER_READ_ACCESS, VMEM_BVH_READ_ACCESS};

    assert(SIInstrInfo::isVMEM(Inst));
    // LDS DMA loads are also stores, but on the LDS side. On the VMEM side
    // these should use VM_CNT.
    if (!ST->hasVscnt() || SIInstrInfo::mayWriteLDSThroughDMA(Inst))
      return VMEM_ACCESS;
    if (Inst.mayStore() &&
        (!Inst.mayLoad() || SIInstrInfo::isAtomicNoRet(Inst))) {
      if (TII->mayAccessScratch(Inst))
        return SCRATCH_WRITE_ACCESS;
      return VMEM_WRITE_ACCESS;
    }
    if (!ST->hasExtendedWaitCounts() || SIInstrInfo::isFLAT(Inst))
      return VMEM_ACCESS;
    return VmemReadMapping[getVmemType(Inst)];
  }

  std::optional<WaitEventType>
  getExpertSchedulingEventType(const MachineInstr &Inst) const;

  bool isVmemAccess(const MachineInstr &MI) const;
  bool generateWaitcntInstBefore(MachineInstr &MI,
                                 WaitcntBrackets &ScoreBrackets,
                                 MachineInstr *OldWaitcntInstr,
                                 PreheaderFlushFlags FlushFlags);
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
  void setSchedulingMode(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                         bool ExpertMode) const;
  AtomicRMWState getAtomicRMWState(MachineInstr &MI,
                                   AtomicRMWState PrevState) const;
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
  WaitcntBrackets(const SIInsertWaitcnts *Context) : Context(Context) {
    assert(Context->TRI->getNumRegUnits() < REGUNITS_END);
  }

#ifndef NDEBUG
  ~WaitcntBrackets() {
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

  unsigned getSGPRScore(MCRegUnit RU, InstCounterType T) const {
    auto It = SGPRs.find(RU);
    return It != SGPRs.end() ? It->second.Scores[getSgprScoresIdx(T)] : 0;
  }

  unsigned getVMemScore(VMEMID TID, InstCounterType T) const {
    auto It = VMem.find(TID);
    return It != VMem.end() ? It->second.Scores[T] : 0;
  }

  bool merge(const WaitcntBrackets &Other);

  bool counterOutOfOrder(InstCounterType T) const;
  void simplifyWaitcnt(AMDGPU::Waitcnt &Wait) const {
    simplifyWaitcnt(Wait, Wait);
  }
  void simplifyWaitcnt(const AMDGPU::Waitcnt &CheckWait,
                       AMDGPU::Waitcnt &UpdateWait) const;
  void simplifyWaitcnt(InstCounterType T, unsigned &Count) const;
  void simplifyXcnt(const AMDGPU::Waitcnt &CheckWait,
                    AMDGPU::Waitcnt &UpdateWait) const;
  void simplifyVmVsrc(const AMDGPU::Waitcnt &CheckWait,
                      AMDGPU::Waitcnt &UpdateWait) const;

  void determineWaitForPhysReg(InstCounterType T, MCPhysReg Reg,
                               AMDGPU::Waitcnt &Wait) const;
  void determineWaitForLDSDMA(InstCounterType T, VMEMID TID,
                              AMDGPU::Waitcnt &Wait) const;
  void tryClearSCCWriteEvent(MachineInstr *Inst);

  void applyWaitcnt(const AMDGPU::Waitcnt &Wait);
  void applyWaitcnt(InstCounterType T, unsigned Count);
  void updateByEvent(WaitEventType E, MachineInstr &MI);

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
                    getWaitCountMax(Context->getLimits(), DS_CNT) - 1);
  }

  void setPendingGDS() { LastGDS = ScoreUBs[DS_CNT]; }

  // Return true if there might be pending writes to the vgpr-interval by VMEM
  // instructions with types different from V.
  bool hasOtherPendingVmemTypes(MCPhysReg Reg, VmemType V) const {
    for (MCRegUnit RU : regunits(Reg)) {
      auto It = VMem.find(toVMEMID(RU));
      if (It != VMem.end() && (It->second.VMEMTypes & ~(1 << V)))
        return true;
    }
    return false;
  }

  void clearVgprVmemTypes(MCPhysReg Reg) {
    for (MCRegUnit RU : regunits(Reg)) {
      if (auto It = VMem.find(toVMEMID(RU)); It != VMem.end()) {
        It->second.VMEMTypes = 0;
        if (It->second.empty())
          VMem.erase(It);
      }
    }
  }

  void setStateOnFunctionEntryOrReturn() {
    setScoreUB(STORE_CNT, getScoreUB(STORE_CNT) +
                              getWaitCountMax(Context->getLimits(), STORE_CNT));
    PendingEvents |= Context->WaitEventMaskForInst[STORE_CNT];
  }

  ArrayRef<const MachineInstr *> getLDSDMAStores() const {
    return LDSDMAStores;
  }

  bool hasPointSampleAccel(const MachineInstr &MI) const;
  bool hasPointSamplePendingVmemTypes(const MachineInstr &MI,
                                      MCPhysReg RU) const;

  void print(raw_ostream &) const;
  void dump() const { print(dbgs()); }

  // Free up memory by removing empty entries from the DenseMap that track event
  // scores.
  void purgeEmptyTrackingData();

private:
  struct MergeInfo {
    unsigned OldLB;
    unsigned OtherLB;
    unsigned MyShift;
    unsigned OtherShift;
  };

  void determineWaitForScore(InstCounterType T, unsigned Score,
                             AMDGPU::Waitcnt &Wait) const;

  static bool mergeScore(const MergeInfo &M, unsigned &Score,
                         unsigned OtherScore);

  iterator_range<MCRegUnitIterator> regunits(MCPhysReg Reg) const {
    assert(Reg != AMDGPU::SCC && "Shouldn't be used on SCC");
    if (!Context->TRI->isInAllocatableClass(Reg))
      return {{}, {}};
    const TargetRegisterClass *RC = Context->TRI->getPhysRegBaseClass(Reg);
    unsigned Size = Context->TRI->getRegSizeInBits(*RC);
    if (Size == 16 && Context->ST->hasD16Writes32BitVgpr())
      Reg = Context->TRI->get32BitRegister(Reg);
    return Context->TRI->regunits(Reg);
  }

  void setScoreLB(InstCounterType T, unsigned Val) {
    assert(T < NUM_INST_CNTS);
    ScoreLBs[T] = Val;
  }

  void setScoreUB(InstCounterType T, unsigned Val) {
    assert(T < NUM_INST_CNTS);
    ScoreUBs[T] = Val;

    if (T != EXP_CNT)
      return;

    if (getScoreRange(EXP_CNT) > getWaitCountMax(Context->getLimits(), EXP_CNT))
      ScoreLBs[EXP_CNT] =
          ScoreUBs[EXP_CNT] - getWaitCountMax(Context->getLimits(), EXP_CNT);
  }

  void setRegScore(MCPhysReg Reg, InstCounterType T, unsigned Val) {
    const SIRegisterInfo *TRI = Context->TRI;
    if (Reg == AMDGPU::SCC) {
      SCCScore = Val;
    } else if (TRI->isVectorRegister(*Context->MRI, Reg)) {
      for (MCRegUnit RU : regunits(Reg))
        VMem[toVMEMID(RU)].Scores[T] = Val;
    } else if (TRI->isSGPRReg(*Context->MRI, Reg)) {
      auto STy = getSgprScoresIdx(T);
      for (MCRegUnit RU : regunits(Reg))
        SGPRs[RU].Scores[STy] = Val;
    } else {
      llvm_unreachable("Register cannot be tracked/unknown register!");
    }
  }

  void setVMemScore(VMEMID TID, InstCounterType T, unsigned Val) {
    VMem[TID].Scores[T] = Val;
  }

  void setScoreByOperand(const MachineOperand &Op, InstCounterType CntTy,
                         unsigned Val);

  const SIInsertWaitcnts *Context;

  unsigned ScoreLBs[NUM_INST_CNTS] = {0};
  unsigned ScoreUBs[NUM_INST_CNTS] = {0};
  unsigned PendingEvents = 0;
  // Remember the last flat memory operation.
  unsigned LastFlat[NUM_INST_CNTS] = {0};
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
    // Scores for all instruction counters.
    std::array<unsigned, NUM_INST_CNTS> Scores = {0};
    // Bitmask of the VmemTypes of VMEM instructions for this VGPR.
    unsigned VMEMTypes = 0;

    bool empty() const { return all_of(Scores, equal_to(0)) && !VMEMTypes; }
  };

  struct SGPRInfo {
    // Wait cnt scores for every sgpr, the DS_CNT (corresponding to LGKMcnt
    // pre-gfx12) or KM_CNT (gfx12+ only), and X_CNT (gfx1250) are relevant.
    // Row 0 represents the score for either DS_CNT or KM_CNT and row 1 keeps
    // the X_CNT score.
    std::array<unsigned, 2> Scores = {0};

    bool empty() const { return !Scores[0] && !Scores[1]; }
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

void WaitcntBrackets::setScoreByOperand(const MachineOperand &Op,
                                        InstCounterType CntTy, unsigned Score) {
  setRegScore(Op.getReg().asMCReg(), CntTy, Score);
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
bool WaitcntBrackets::hasPointSamplePendingVmemTypes(const MachineInstr &MI,
                                                     MCPhysReg Reg) const {
  if (!hasPointSampleAccel(MI))
    return false;

  return hasOtherPendingVmemTypes(Reg, VMEM_NOSAMPLER);
}

void WaitcntBrackets::updateByEvent(WaitEventType E, MachineInstr &Inst) {
  InstCounterType T = eventCounter(Context->WaitEventMaskForInst, E);
  assert(T < Context->MaxCounter);

  unsigned UB = getScoreUB(T);
  unsigned CurrScore = UB + 1;
  if (CurrScore == 0)
    report_fatal_error("InsertWaitcnt score wraparound");
  // PendingEvents and ScoreUB need to be update regardless if this event
  // changes the score of a register or not.
  // Examples including vm_cnt when buffer-store or lgkm_cnt when send-message.
  PendingEvents |= 1 << E;
  setScoreUB(T, CurrScore);

  const SIRegisterInfo *TRI = Context->TRI;
  const MachineRegisterInfo *MRI = Context->MRI;
  const SIInstrInfo *TII = Context->TII;

  if (T == EXP_CNT) {
    // Put score on the source vgprs. If this is a store, just use those
    // specific register(s).
    if (TII->isDS(Inst) && Inst.mayLoadOrStore()) {
      // All GDS operations must protect their address register (same as
      // export.)
      if (const auto *AddrOp = TII->getNamedOperand(Inst, AMDGPU::OpName::addr))
        setScoreByOperand(*AddrOp, EXP_CNT, CurrScore);

      if (Inst.mayStore()) {
        if (const auto *Data0 =
                TII->getNamedOperand(Inst, AMDGPU::OpName::data0))
          setScoreByOperand(*Data0, EXP_CNT, CurrScore);
        if (const auto *Data1 =
                TII->getNamedOperand(Inst, AMDGPU::OpName::data1))
          setScoreByOperand(*Data1, EXP_CNT, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst) && !SIInstrInfo::isGWS(Inst) &&
                 Inst.getOpcode() != AMDGPU::DS_APPEND &&
                 Inst.getOpcode() != AMDGPU::DS_CONSUME &&
                 Inst.getOpcode() != AMDGPU::DS_ORDERED_COUNT) {
        for (const MachineOperand &Op : Inst.all_uses()) {
          if (TRI->isVectorRegister(*MRI, Op.getReg()))
            setScoreByOperand(Op, EXP_CNT, CurrScore);
        }
      }
    } else if (TII->isFLAT(Inst)) {
      if (Inst.mayStore()) {
        setScoreByOperand(*TII->getNamedOperand(Inst, AMDGPU::OpName::data),
                          EXP_CNT, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setScoreByOperand(*TII->getNamedOperand(Inst, AMDGPU::OpName::data),
                          EXP_CNT, CurrScore);
      }
    } else if (TII->isMIMG(Inst)) {
      if (Inst.mayStore()) {
        setScoreByOperand(Inst.getOperand(0), EXP_CNT, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setScoreByOperand(*TII->getNamedOperand(Inst, AMDGPU::OpName::data),
                          EXP_CNT, CurrScore);
      }
    } else if (TII->isMTBUF(Inst)) {
      if (Inst.mayStore())
        setScoreByOperand(Inst.getOperand(0), EXP_CNT, CurrScore);
    } else if (TII->isMUBUF(Inst)) {
      if (Inst.mayStore()) {
        setScoreByOperand(Inst.getOperand(0), EXP_CNT, CurrScore);
      } else if (SIInstrInfo::isAtomicRet(Inst)) {
        setScoreByOperand(*TII->getNamedOperand(Inst, AMDGPU::OpName::data),
                          EXP_CNT, CurrScore);
      }
    } else if (TII->isLDSDIR(Inst)) {
      // LDSDIR instructions attach the score to the destination.
      setScoreByOperand(*TII->getNamedOperand(Inst, AMDGPU::OpName::vdst),
                        EXP_CNT, CurrScore);
    } else {
      if (TII->isEXP(Inst)) {
        // For export the destination registers are really temps that
        // can be used as the actual source after export patching, so
        // we need to treat them like sources and set the EXP_CNT
        // score.
        for (MachineOperand &DefMO : Inst.all_defs()) {
          if (TRI->isVGPR(*MRI, DefMO.getReg())) {
            setScoreByOperand(DefMO, EXP_CNT, CurrScore);
          }
        }
      }
      for (const MachineOperand &Op : Inst.all_uses()) {
        if (TRI->isVectorRegister(*MRI, Op.getReg()))
          setScoreByOperand(Op, EXP_CNT, CurrScore);
      }
    }
  } else if (T == X_CNT) {
    WaitEventType OtherEvent = E == SMEM_GROUP ? VMEM_GROUP : SMEM_GROUP;
    if (PendingEvents & (1 << OtherEvent)) {
      // Hardware inserts an implicit xcnt between interleaved
      // SMEM and VMEM operations. So there will never be
      // outstanding address translations for both SMEM and
      // VMEM at the same time.
      setScoreLB(T, getScoreUB(T) - 1);
      PendingEvents &= ~(1 << OtherEvent);
    }
    for (const MachineOperand &Op : Inst.all_uses())
      setScoreByOperand(Op, T, CurrScore);
  } else if (T == VA_VDST || T == VM_VSRC) {
    // Match the score to the VGPR destination or source registers as
    // appropriate
    for (const MachineOperand &Op : Inst.operands()) {
      if (!Op.isReg() || (T == VA_VDST && Op.isUse()) ||
          (T == VM_VSRC && Op.isDef()))
        continue;
      if (TRI->isVectorRegister(*Context->MRI, Op.getReg()))
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
        if (!TRI->isVectorRegister(*MRI, Op.getReg())) // TODO: add wrapper
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
          for (MCRegUnit RU : regunits(Op.getReg().asMCReg()))
            VMem[toVMEMID(RU)].VMEMTypes |= TypesMask;
        }
      }
      setScoreByOperand(Op, T, CurrScore);
    }
    if (Inst.mayStore() &&
        (TII->isDS(Inst) || TII->mayWriteLDSThroughDMA(Inst))) {
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

    if (SIInstrInfo::isSBarrierSCCWrite(Inst.getOpcode())) {
      setRegScore(AMDGPU::SCC, T, CurrScore);
      PendingSCCWrite = &Inst;
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
         << SR << "):";
      break;
    case DS_CNT:
      OS << "    " << (ST->hasExtendedWaitCounts() ? "DS" : "LGKM") << "_CNT("
         << SR << "):";
      break;
    case EXP_CNT:
      OS << "    EXP_CNT(" << SR << "):";
      break;
    case STORE_CNT:
      OS << "    " << (ST->hasExtendedWaitCounts() ? "STORE" : "VS") << "_CNT("
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
    case VA_VDST:
      OS << "    VA_VDST(" << SR << "): ";
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
          OS << ' ' << RelScore << ":vRU" << ID;
        } else {
          assert(ID >= LDSDMA_BEGIN && ID < LDSDMA_END &&
                 "Unhandled/unexpected ID value!");
          OS << ' ' << RelScore << ":LDSDMA" << ID;
        }
      }

      // Also need to print sgpr scores for lgkm_cnt or xcnt.
      if (isSmemCounter(T)) {
        SmallVector<MCRegUnit> SortedSMEMIDs(SGPRs.keys());
        sort(SortedSMEMIDs);
        for (auto ID : SortedSMEMIDs) {
          unsigned RegScore = SGPRs.at(ID).Scores[getSgprScoresIdx(T)];
          if (RegScore <= LB)
            continue;
          unsigned RelScore = RegScore - LB - 1;
          OS << ' ' << RelScore << ":sRU" << static_cast<unsigned>(ID);
        }
      }

      if (T == KM_CNT && SCCScore > 0)
        OS << ' ' << SCCScore << ":scc";
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

/// Simplify \p UpdateWait by removing waits that are redundant based on the
/// current WaitcntBrackets and any other waits specified in \p CheckWait.
void WaitcntBrackets::simplifyWaitcnt(const AMDGPU::Waitcnt &CheckWait,
                                      AMDGPU::Waitcnt &UpdateWait) const {
  simplifyWaitcnt(LOAD_CNT, UpdateWait.LoadCnt);
  simplifyWaitcnt(EXP_CNT, UpdateWait.ExpCnt);
  simplifyWaitcnt(DS_CNT, UpdateWait.DsCnt);
  simplifyWaitcnt(STORE_CNT, UpdateWait.StoreCnt);
  simplifyWaitcnt(SAMPLE_CNT, UpdateWait.SampleCnt);
  simplifyWaitcnt(BVH_CNT, UpdateWait.BvhCnt);
  simplifyWaitcnt(KM_CNT, UpdateWait.KmCnt);
  simplifyXcnt(CheckWait, UpdateWait);
  simplifyWaitcnt(VA_VDST, UpdateWait.VaVdst);
  simplifyVmVsrc(CheckWait, UpdateWait);
}

void WaitcntBrackets::simplifyWaitcnt(InstCounterType T,
                                      unsigned &Count) const {
  // The number of outstanding events for this type, T, can be calculated
  // as (UB - LB). If the current Count is greater than or equal to the number
  // of outstanding events, then the wait for this counter is redundant.
  if (Count >= getScoreRange(T))
    Count = ~0u;
}

void WaitcntBrackets::simplifyVmVsrc(const AMDGPU::Waitcnt &CheckWait,
                                     AMDGPU::Waitcnt &UpdateWait) const {
  // Waiting for some counters implies waiting for VM_VSRC, since an
  // instruction that decrements a counter on completion would have
  // decremented VM_VSRC once its VGPR operands had been read.
  if (CheckWait.VmVsrc >=
      std::min({CheckWait.LoadCnt, CheckWait.StoreCnt, CheckWait.SampleCnt,
                CheckWait.BvhCnt, CheckWait.DsCnt}))
    UpdateWait.VmVsrc = ~0u;
  simplifyWaitcnt(VM_VSRC, UpdateWait.VmVsrc);
}

void WaitcntBrackets::purgeEmptyTrackingData() {
  for (auto &[K, V] : make_early_inc_range(VMem)) {
    if (V.empty())
      VMem.erase(K);
  }
  for (auto &[K, V] : make_early_inc_range(SGPRs)) {
    if (V.empty())
      SGPRs.erase(K);
  }
}

void WaitcntBrackets::determineWaitForScore(InstCounterType T,
                                            unsigned ScoreToWait,
                                            AMDGPU::Waitcnt &Wait) const {
  const unsigned LB = getScoreLB(T);
  const unsigned UB = getScoreUB(T);

  // If the score falls within the bracket, we need a waitcnt.
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
      unsigned NeededWait = std::min(
          UB - ScoreToWait, getWaitCountMax(Context->getLimits(), T) - 1);
      addWait(Wait, T, NeededWait);
    }
  }
}

void WaitcntBrackets::determineWaitForPhysReg(InstCounterType T, MCPhysReg Reg,
                                              AMDGPU::Waitcnt &Wait) const {
  if (Reg == AMDGPU::SCC) {
    determineWaitForScore(T, SCCScore, Wait);
  } else {
    bool IsVGPR = Context->TRI->isVectorRegister(*Context->MRI, Reg);
    for (MCRegUnit RU : regunits(Reg))
      determineWaitForScore(
          T, IsVGPR ? getVMemScore(toVMEMID(RU), T) : getSGPRScore(RU, T),
          Wait);
  }
}

void WaitcntBrackets::determineWaitForLDSDMA(InstCounterType T, VMEMID TID,
                                             AMDGPU::Waitcnt &Wait) const {
  assert(TID >= LDSDMA_BEGIN && TID < LDSDMA_END);
  determineWaitForScore(T, getVMemScore(TID, T), Wait);
}

void WaitcntBrackets::tryClearSCCWriteEvent(MachineInstr *Inst) {
  // S_BARRIER_WAIT on the same barrier guarantees that the pending write to
  // SCC has landed
  if (PendingSCCWrite &&
      PendingSCCWrite->getOpcode() == AMDGPU::S_BARRIER_SIGNAL_ISFIRST_IMM &&
      PendingSCCWrite->getOperand(0).getImm() == Inst->getOperand(0).getImm()) {
    unsigned SCC_WRITE_PendingEvent = 1 << SCC_WRITE;
    // If this SCC_WRITE is the only pending KM_CNT event, clear counter.
    if ((PendingEvents & Context->WaitEventMaskForInst[KM_CNT]) ==
        SCC_WRITE_PendingEvent) {
      setScoreLB(KM_CNT, getScoreUB(KM_CNT));
    }

    PendingEvents &= ~SCC_WRITE_PendingEvent;
    PendingSCCWrite = nullptr;
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
  applyWaitcnt(X_CNT, Wait.XCnt);
  applyWaitcnt(VA_VDST, Wait.VaVdst);
  applyWaitcnt(VM_VSRC, Wait.VmVsrc);
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

  if (T == KM_CNT && Count == 0 && hasPendingEvent(SMEM_GROUP)) {
    if (!hasMixedPendingEvents(X_CNT))
      applyWaitcnt(X_CNT, 0);
    else
      PendingEvents &= ~(1 << SMEM_GROUP);
  }
  if (T == LOAD_CNT && hasPendingEvent(VMEM_GROUP) &&
      !hasPendingEvent(STORE_CNT)) {
    if (!hasMixedPendingEvents(X_CNT))
      applyWaitcnt(X_CNT, Count);
    else if (Count == 0)
      PendingEvents &= ~(1 << VMEM_GROUP);
  }
}

void WaitcntBrackets::simplifyXcnt(const AMDGPU::Waitcnt &CheckWait,
                                   AMDGPU::Waitcnt &UpdateWait) const {
  // Try to simplify xcnt further by checking for joint kmcnt and loadcnt
  // optimizations. On entry to a block with multiple predescessors, there may
  // be pending SMEM and VMEM events active at the same time.
  // In such cases, only clear one active event at a time.
  // TODO: Revisit xcnt optimizations for gfx1250.
  // Wait on XCNT is redundant if we are already waiting for a load to complete.
  // SMEM can return out of order, so only omit XCNT wait if we are waiting till
  // zero.
  if (CheckWait.KmCnt == 0 && hasPendingEvent(SMEM_GROUP))
    UpdateWait.XCnt = ~0u;
  // If we have pending store we cannot optimize XCnt because we do not wait for
  // stores. VMEM loads retun in order, so if we only have loads XCnt is
  // decremented to the same number as LOADCnt.
  if (CheckWait.LoadCnt != ~0u && hasPendingEvent(VMEM_GROUP) &&
      !hasPendingEvent(STORE_CNT) && CheckWait.XCnt >= CheckWait.LoadCnt)
    UpdateWait.XCnt = ~0u;
  simplifyWaitcnt(X_CNT, UpdateWait.XCnt);
}

// Where there are multiple types of event in the bracket of a counter,
// the decrement may go out of order.
bool WaitcntBrackets::counterOutOfOrder(InstCounterType T) const {
  // Scalar memory read always can go out of order.
  if ((T == Context->SmemAccessCounter && hasPendingEvent(SMEM_ACCESS)) ||
      (T == X_CNT && hasPendingEvent(SMEM_GROUP)))
    return true;

  // GLOBAL_INV completes in-order with other LOAD_CNT events (VMEM_ACCESS),
  // so having GLOBAL_INV_ACCESS mixed with other LOAD_CNT events doesn't cause
  // out-of-order completion.
  if (T == LOAD_CNT) {
    unsigned Events = hasPendingEvent(T);
    // Remove GLOBAL_INV_ACCESS from the event mask before checking for mixed
    // events
    Events &= ~(1 << GLOBAL_INV_ACCESS);
    // Return true only if there are still multiple event types after removing
    // GLOBAL_INV
    return Events & (Events - 1);
  }

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
    if (It.isEnd())
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
                        << "Before: " << Wait << '\n';);
      ScoreBrackets.determineWaitForLDSDMA(LOAD_CNT, LDSDMA_BEGIN, Wait);
      LLVM_DEBUG(dbgs() << "After: " << Wait << '\n';);

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

    LLVM_DEBUG(It.isEnd() ? dbgs() << "applied pre-existing waitcnt\n"
                                   << "New Instr at block end: "
                                   << *WaitcntInstr << '\n'
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

    LLVM_DEBUG(It.isEnd()
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
    AMDGPU::Waitcnt Wait, const WaitcntBrackets &ScoreBrackets) {
  assert(ST);
  assert(isNormalMode(MaxCounter));

  bool Modified = false;
  const DebugLoc &DL = Block.findDebugLoc(It);

  // Helper to emit expanded waitcnt sequence for profiling.
  // Emits waitcnts from (Outstanding-1) down to Target.
  // The EmitWaitcnt callback emits a single waitcnt.
  auto EmitExpandedWaitcnt = [&](unsigned Outstanding, unsigned Target,
                                 auto EmitWaitcnt) {
    do {
      EmitWaitcnt(--Outstanding);
    } while (Outstanding > Target);
    Modified = true;
  };

  // Waits for VMcnt, LKGMcnt and/or EXPcnt are encoded together into a
  // single instruction while VScnt has its own instruction.
  if (Wait.hasWaitExceptStoreCnt()) {
    // If profiling expansion is enabled, emit an expanded sequence
    if (ExpandWaitcntProfiling) {
      // Check if any of the counters to be waited on are out-of-order.
      // If so, fall back to normal (non-expanded) behavior since expansion
      // would provide misleading profiling information.
      bool AnyOutOfOrder = false;
      for (auto CT : {LOAD_CNT, DS_CNT, EXP_CNT}) {
        unsigned &WaitCnt = getCounterRef(Wait, CT);
        if (WaitCnt != ~0u && ScoreBrackets.counterOutOfOrder(CT)) {
          AnyOutOfOrder = true;
          break;
        }
      }

      if (AnyOutOfOrder) {
        // Fall back to non-expanded wait
        unsigned Enc = AMDGPU::encodeWaitcnt(IV, Wait);
        BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAITCNT)).addImm(Enc);
        Modified = true;
      } else {
        // All counters are in-order, safe to expand
        for (auto CT : {LOAD_CNT, DS_CNT, EXP_CNT}) {
          unsigned &WaitCnt = getCounterRef(Wait, CT);
          if (WaitCnt == ~0u)
            continue;

          unsigned Outstanding = std::min(ScoreBrackets.getScoreUB(CT) -
                                              ScoreBrackets.getScoreLB(CT),
                                          getWaitCountMax(getLimits(), CT) - 1);
          EmitExpandedWaitcnt(Outstanding, WaitCnt, [&](unsigned Count) {
            AMDGPU::Waitcnt W;
            getCounterRef(W, CT) = Count;
            BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAITCNT))
                .addImm(AMDGPU::encodeWaitcnt(IV, W));
          });
        }
      }
    } else {
      // Normal behavior: emit single combined waitcnt
      unsigned Enc = AMDGPU::encodeWaitcnt(IV, Wait);
      [[maybe_unused]] auto SWaitInst =
          BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAITCNT)).addImm(Enc);
      Modified = true;

      LLVM_DEBUG(dbgs() << "PreGFX12::createNewWaitcnt\n";
                 if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
                 dbgs() << "New Instr: " << *SWaitInst << '\n');
    }
  }

  if (Wait.hasWaitStoreCnt()) {
    assert(ST->hasVscnt());

    if (ExpandWaitcntProfiling && Wait.StoreCnt != ~0u &&
        !ScoreBrackets.counterOutOfOrder(STORE_CNT)) {
      // Only expand if counter is not out-of-order
      unsigned Outstanding =
          std::min(ScoreBrackets.getScoreUB(STORE_CNT) -
                       ScoreBrackets.getScoreLB(STORE_CNT),
                   getWaitCountMax(getLimits(), STORE_CNT) - 1);
      EmitExpandedWaitcnt(Outstanding, Wait.StoreCnt, [&](unsigned Count) {
        BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAITCNT_VSCNT))
            .addReg(AMDGPU::SGPR_NULL, RegState::Undef)
            .addImm(Count);
      });
    } else {
      [[maybe_unused]] auto SWaitInst =
          BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAITCNT_VSCNT))
              .addReg(AMDGPU::SGPR_NULL, RegState::Undef)
              .addImm(Wait.StoreCnt);
      Modified = true;

      LLVM_DEBUG(dbgs() << "PreGFX12::createNewWaitcnt\n";
                 if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
                 dbgs() << "New Instr: " << *SWaitInst << '\n');
    }
  }

  return Modified;
}

AMDGPU::Waitcnt
WaitcntGeneratorPreGFX12::getAllZeroWaitcnt(bool IncludeVSCnt) const {
  return AMDGPU::Waitcnt(0, 0, 0, IncludeVSCnt && ST->hasVscnt() ? 0 : ~0u);
}

AMDGPU::Waitcnt
WaitcntGeneratorGFX12Plus::getAllZeroWaitcnt(bool IncludeVSCnt) const {
  unsigned ExpertVal = IsExpertMode ? 0 : ~0u;
  return AMDGPU::Waitcnt(0, 0, 0, IncludeVSCnt ? 0 : ~0u, 0, 0, 0,
                         ~0u /* XCNT */, ExpertVal, ExpertVal);
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
  MachineInstr *WaitcntDepctrInstr = nullptr;
  MachineInstr *WaitInstrs[NUM_EXTENDED_INST_CNTS] = {};

  LLVM_DEBUG({
    dbgs() << "GFX12Plus::applyPreexistingWaitcnt at: ";
    if (It.isEnd())
      dbgs() << "end of block\n";
    else
      dbgs() << *It;
  });

  // Accumulate waits that should not be simplified.
  AMDGPU::Waitcnt RequiredWait;

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
        Wait = Wait.combined(OldWait);
      else
        RequiredWait = RequiredWait.combined(OldWait);
      UpdatableInstr = &CombinedLoadDsCntInstr;
    } else if (Opcode == AMDGPU::S_WAIT_STORECNT_DSCNT) {
      unsigned OldEnc =
          TII->getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      AMDGPU::Waitcnt OldWait = AMDGPU::decodeStorecntDscnt(IV, OldEnc);
      if (TrySimplify)
        Wait = Wait.combined(OldWait);
      else
        RequiredWait = RequiredWait.combined(OldWait);
      UpdatableInstr = &CombinedStoreDsCntInstr;
    } else if (Opcode == AMDGPU::S_WAITCNT_DEPCTR) {
      unsigned OldEnc =
          TII->getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      AMDGPU::Waitcnt OldWait;
      OldWait.VaVdst = AMDGPU::DepCtr::decodeFieldVaVdst(OldEnc);
      OldWait.VmVsrc = AMDGPU::DepCtr::decodeFieldVmVsrc(OldEnc);
      if (TrySimplify)
        ScoreBrackets.simplifyWaitcnt(OldWait);
      Wait = Wait.combined(OldWait);
      UpdatableInstr = &WaitcntDepctrInstr;
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
        addWait(Wait, CT.value(), OldCnt);
      else
        addWait(RequiredWait, CT.value(), OldCnt);
      UpdatableInstr = &WaitInstrs[CT.value()];
    }

    // Merge consecutive waitcnt of the same type by erasing multiples.
    if (!*UpdatableInstr) {
      *UpdatableInstr = &II;
    } else if (Opcode == AMDGPU::S_WAITCNT_DEPCTR) {
      // S_WAITCNT_DEPCTR requires special care. Don't remove a
      // duplicate if it is waiting on things other than VA_VDST or
      // VM_VSRC. If that is the case, just make sure the VA_VDST and
      // VM_VSRC subfields of the operand are set to the "no wait"
      // values.

      unsigned Enc = TII->getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      Enc = AMDGPU::DepCtr::encodeFieldVmVsrc(Enc, ~0u);
      Enc = AMDGPU::DepCtr::encodeFieldVaVdst(Enc, ~0u);

      if (Enc != (unsigned)AMDGPU::DepCtr::getDefaultDepCtrEncoding(*ST)) {
        Modified |= updateOperandIfDifferent(II, AMDGPU::OpName::simm16, Enc);
        Modified |= promoteSoftWaitCnt(&II);
      } else {
        II.eraseFromParent();
        Modified = true;
      }
    } else {
      II.eraseFromParent();
      Modified = true;
    }
  }

  ScoreBrackets.simplifyWaitcnt(Wait.combined(RequiredWait), Wait);
  Wait = Wait.combined(RequiredWait);

  if (CombinedLoadDsCntInstr) {
    // Only keep an S_WAIT_LOADCNT_DSCNT if both counters actually need
    // to be waited for. Otherwise, let the instruction be deleted so
    // the appropriate single counter wait instruction can be inserted
    // instead, when new S_WAIT_*CNT instructions are inserted by
    // createNewWaitcnt(). As a side effect, resetting the wait counts will
    // cause any redundant S_WAIT_LOADCNT or S_WAIT_DSCNT to be removed by
    // the loop below that deals with single counter instructions.
    //
    // A wait for LOAD_CNT or DS_CNT implies a wait for VM_VSRC, since
    // instructions that have decremented LOAD_CNT or DS_CNT on completion
    // will have needed to wait for their register sources to be available
    // first.
    if (Wait.LoadCnt != ~0u && Wait.DsCnt != ~0u) {
      unsigned NewEnc = AMDGPU::encodeLoadcntDscnt(IV, Wait);
      Modified |= updateOperandIfDifferent(*CombinedLoadDsCntInstr,
                                           AMDGPU::OpName::simm16, NewEnc);
      Modified |= promoteSoftWaitCnt(CombinedLoadDsCntInstr);
      ScoreBrackets.applyWaitcnt(LOAD_CNT, Wait.LoadCnt);
      ScoreBrackets.applyWaitcnt(DS_CNT, Wait.DsCnt);
      Wait.LoadCnt = ~0u;
      Wait.DsCnt = ~0u;

      LLVM_DEBUG(It.isEnd() ? dbgs() << "applied pre-existing waitcnt\n"
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

      LLVM_DEBUG(It.isEnd() ? dbgs() << "applied pre-existing waitcnt\n"
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

      LLVM_DEBUG(It.isEnd()
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

  if (WaitcntDepctrInstr) {
    // Get the encoded Depctr immediate and override the VA_VDST and VM_VSRC
    // subfields with the new required values.
    unsigned Enc =
        TII->getNamedOperand(*WaitcntDepctrInstr, AMDGPU::OpName::simm16)
            ->getImm();
    Enc = AMDGPU::DepCtr::encodeFieldVmVsrc(Enc, Wait.VmVsrc);
    Enc = AMDGPU::DepCtr::encodeFieldVaVdst(Enc, Wait.VaVdst);

    ScoreBrackets.applyWaitcnt(VA_VDST, Wait.VaVdst);
    ScoreBrackets.applyWaitcnt(VM_VSRC, Wait.VmVsrc);
    Wait.VaVdst = ~0u;
    Wait.VmVsrc = ~0u;

    // If that new encoded Depctr immediate would actually still wait
    // for anything, update the instruction's operand. Otherwise it can
    // just be deleted.
    if (Enc != (unsigned)AMDGPU::DepCtr::getDefaultDepCtrEncoding(*ST)) {
      Modified |= updateOperandIfDifferent(*WaitcntDepctrInstr,
                                           AMDGPU::OpName::simm16, Enc);
      LLVM_DEBUG(It.isEnd() ? dbgs() << "applyPreexistingWaitcnt\n"
                                     << "New Instr at block end: "
                                     << *WaitcntDepctrInstr << '\n'
                            : dbgs() << "applyPreexistingWaitcnt\n"
                                     << "Old Instr: " << *It << "New Instr: "
                                     << *WaitcntDepctrInstr << '\n');
    } else {
      WaitcntDepctrInstr->eraseFromParent();
      Modified = true;
    }
  }

  return Modified;
}

/// Generate S_WAIT_*CNT instructions for any required counters in \p Wait
bool WaitcntGeneratorGFX12Plus::createNewWaitcnt(
    MachineBasicBlock &Block, MachineBasicBlock::instr_iterator It,
    AMDGPU::Waitcnt Wait, const WaitcntBrackets &ScoreBrackets) {
  assert(ST);
  assert(!isNormalMode(MaxCounter));

  bool Modified = false;
  const DebugLoc &DL = Block.findDebugLoc(It);

  // Helper to emit expanded waitcnt sequence for profiling.
  auto EmitExpandedWaitcnt = [&](unsigned Outstanding, unsigned Target,
                                 auto EmitWaitcnt) {
    for (unsigned I = Outstanding - 1; I > Target && I != ~0u; --I)
      EmitWaitcnt(I);
    EmitWaitcnt(Target);
    Modified = true;
  };

  // For GFX12+, we use separate wait instructions, which makes expansion
  // simpler
  if (ExpandWaitcntProfiling) {
    for (auto CT : inst_counter_types(NUM_EXTENDED_INST_CNTS)) {
      unsigned Count = getWait(Wait, CT);
      if (Count == ~0u)
        continue;

      // Skip expansion for out-of-order counters - emit normal wait instead
      if (ScoreBrackets.counterOutOfOrder(CT)) {
        BuildMI(Block, It, DL, TII->get(instrsForExtendedCounterTypes[CT]))
            .addImm(Count);
        Modified = true;
        continue;
      }

      unsigned Outstanding =
          std::min(ScoreBrackets.getScoreUB(CT) - ScoreBrackets.getScoreLB(CT),
                   getWaitCountMax(getLimits(), CT) - 1);
      EmitExpandedWaitcnt(Outstanding, Count, [&](unsigned Val) {
        BuildMI(Block, It, DL, TII->get(instrsForExtendedCounterTypes[CT]))
            .addImm(Val);
      });
    }
    return Modified;
  }

  // Normal behavior (no expansion)
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

      LLVM_DEBUG(dbgs() << "GFX12Plus::createNewWaitcnt\n";
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

    LLVM_DEBUG(dbgs() << "GFX12Plus::createNewWaitcnt\n";
               if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
               dbgs() << "New Instr: " << *SWaitInst << '\n');
  }

  if (Wait.hasWaitDepctr()) {
    assert(IsExpertMode);
    unsigned Enc = AMDGPU::DepCtr::encodeFieldVmVsrc(Wait.VmVsrc, *ST);
    Enc = AMDGPU::DepCtr::encodeFieldVaVdst(Enc, Wait.VaVdst);

    [[maybe_unused]] auto SWaitInst =
        BuildMI(Block, It, DL, TII->get(AMDGPU::S_WAITCNT_DEPCTR)).addImm(Enc);

    Modified = true;

    LLVM_DEBUG(dbgs() << "generateWaitcnt\n";
               if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
               dbgs() << "New Instr: " << *SWaitInst << '\n');
  }

  return Modified;
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
///  If FlushFlags.FlushVmCnt is true, we want to flush the vmcnt counter here.
///  If FlushFlags.FlushDsCnt is true, we want to flush the dscnt counter here
///  (GFX12+ only, where DS_CNT is a separate counter).
bool SIInsertWaitcnts::generateWaitcntInstBefore(
    MachineInstr &MI, WaitcntBrackets &ScoreBrackets,
    MachineInstr *OldWaitcntInstr, PreheaderFlushFlags FlushFlags) {
  setForceEmitWaitcnt();

  assert(!MI.isMetaInstruction());

  AMDGPU::Waitcnt Wait;
  const unsigned Opc = MI.getOpcode();

  // FIXME: This should have already been handled by the memory legalizer.
  // Removing this currently doesn't affect any lit tests, but we need to
  // verify that nothing was relying on this. The number of buffer invalidates
  // being handled here should not be expanded.
  if (Opc == AMDGPU::BUFFER_WBINVL1 || Opc == AMDGPU::BUFFER_WBINVL1_SC ||
      Opc == AMDGPU::BUFFER_WBINVL1_VOL || Opc == AMDGPU::BUFFER_GL0_INV ||
      Opc == AMDGPU::BUFFER_GL1_INV) {
    Wait.LoadCnt = 0;
  }

  // All waits must be resolved at call return.
  // NOTE: this could be improved with knowledge of all call sites or
  //   with knowledge of the called routines.
  if (Opc == AMDGPU::SI_RETURN_TO_EPILOG || Opc == AMDGPU::SI_RETURN ||
      Opc == AMDGPU::SI_WHOLE_WAVE_FUNC_RETURN ||
      Opc == AMDGPU::S_SETPC_B64_return) {
    ReturnInsts.insert(&MI);
    AMDGPU::Waitcnt AllZeroWait =
        WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/false);
    // On GFX12+, if LOAD_CNT is pending but no VGPRs are waiting for loads
    // (e.g., only GLOBAL_INV is pending), we can skip waiting on loadcnt.
    // GLOBAL_INV increments loadcnt but doesn't write to VGPRs, so there's
    // no need to wait for it at function boundaries.
    if (ST->hasExtendedWaitCounts() &&
        !ScoreBrackets.hasPendingEvent(VMEM_ACCESS))
      AllZeroWait.LoadCnt = ~0u;
    Wait = Wait.combined(AllZeroWait);
  }
  // In dynamic VGPR mode, we want to release the VGPRs before the wave exits.
  // Technically the hardware will do this on its own if we don't, but that
  // might cost extra cycles compared to doing it explicitly.
  // When not in dynamic VGPR mode, identify S_ENDPGM instructions which may
  // have to wait for outstanding VMEM stores. In this case it can be useful to
  // send a message to explicitly release all VGPRs before the stores have
  // completed, but it is only safe to do this if there are no outstanding
  // scratch stores.
  else if (Opc == AMDGPU::S_ENDPGM || Opc == AMDGPU::S_ENDPGM_SAVED) {
    if (!WCG->isOptNone() &&
        (MI.getMF()->getInfo<SIMachineFunctionInfo>()->isDynamicVGPREnabled() ||
         (ST->getGeneration() >= AMDGPUSubtarget::GFX11 &&
          ScoreBrackets.getScoreRange(STORE_CNT) != 0 &&
          !ScoreBrackets.hasPendingEvent(SCRATCH_WRITE_ACCESS))))
      ReleaseVGPRInsts.insert(&MI);
  }
  // Resolve vm waits before gs-done.
  else if ((Opc == AMDGPU::S_SENDMSG || Opc == AMDGPU::S_SENDMSGHALT) &&
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
    if (TII->isAlwaysGDS(Opc) && ScoreBrackets.hasPendingGDS())
      addWait(Wait, DS_CNT, ScoreBrackets.getPendingGDSWait());

    if (MI.isCall()) {
      // The function is going to insert a wait on everything in its prolog.
      // This still needs to be careful if the call target is a load (e.g. a GOT
      // load). We also need to check WAW dependency with saved PC.
      CallInsts.insert(&MI);
      Wait = AMDGPU::Waitcnt();

      const MachineOperand &CallAddrOp = TII->getCalleeOperand(MI);
      if (CallAddrOp.isReg()) {
        ScoreBrackets.determineWaitForPhysReg(
            SmemAccessCounter, CallAddrOp.getReg().asMCReg(), Wait);

        if (const auto *RtnAddrOp =
                TII->getNamedOperand(MI, AMDGPU::OpName::dst)) {
          ScoreBrackets.determineWaitForPhysReg(
              SmemAccessCounter, RtnAddrOp->getReg().asMCReg(), Wait);
        }
      }
    } else if (Opc == AMDGPU::S_BARRIER_WAIT) {
      ScoreBrackets.tryClearSCCWriteEvent(&MI);
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
        unsigned TID = LDSDMA_BEGIN;
        if (Ptr && Memop->getAAInfo()) {
          const auto &LDSDMAStores = ScoreBrackets.getLDSDMAStores();
          for (unsigned I = 0, E = LDSDMAStores.size(); I != E; ++I) {
            if (MI.mayAlias(AA, *LDSDMAStores[I], true)) {
              if ((I + 1) >= NUM_LDSDMA) {
                // We didn't have enough slot to track this LDS DMA store, it
                // has been tracked using the common RegNo (FIRST_LDS_VGPR).
                ScoreBrackets.determineWaitForLDSDMA(LOAD_CNT, TID, Wait);
                break;
              }

              ScoreBrackets.determineWaitForLDSDMA(LOAD_CNT, TID + I + 1, Wait);
            }
          }
        } else {
          ScoreBrackets.determineWaitForLDSDMA(LOAD_CNT, TID, Wait);
        }
        if (Memop->isStore()) {
          ScoreBrackets.determineWaitForLDSDMA(EXP_CNT, TID, Wait);
        }
      }

      // Loop over use and def operands.
      for (const MachineOperand &Op : MI.operands()) {
        if (!Op.isReg())
          continue;

        // If the instruction does not read tied source, skip the operand.
        if (Op.isTied() && Op.isUse() && TII->doesNotReadTiedSource(MI))
          continue;

        MCPhysReg Reg = Op.getReg().asMCReg();

        const bool IsVGPR = TRI->isVectorRegister(*MRI, Op.getReg());
        if (IsVGPR) {
          // Implicit VGPR defs and uses are never a part of the memory
          // instructions description and usually present to account for
          // super-register liveness.
          // TODO: Most of the other instructions also have implicit uses
          // for the liveness accounting only.
          if (Op.isImplicit() && MI.mayLoadOrStore())
            continue;

          ScoreBrackets.determineWaitForPhysReg(VA_VDST, Reg, Wait);
          if (Op.isDef())
            ScoreBrackets.determineWaitForPhysReg(VM_VSRC, Reg, Wait);
          // RAW always needs an s_waitcnt. WAW needs an s_waitcnt unless the
          // previous write and this write are the same type of VMEM
          // instruction, in which case they are (in some architectures)
          // guaranteed to write their results in order anyway.
          // Additionally check instructions where Point Sample Acceleration
          // might be applied.
          if (Op.isUse() || !updateVMCntOnly(MI) ||
              ScoreBrackets.hasOtherPendingVmemTypes(Reg, getVmemType(MI)) ||
              ScoreBrackets.hasPointSamplePendingVmemTypes(MI, Reg) ||
              !ST->hasVmemWriteVgprInOrder()) {
            ScoreBrackets.determineWaitForPhysReg(LOAD_CNT, Reg, Wait);
            ScoreBrackets.determineWaitForPhysReg(SAMPLE_CNT, Reg, Wait);
            ScoreBrackets.determineWaitForPhysReg(BVH_CNT, Reg, Wait);
            ScoreBrackets.clearVgprVmemTypes(Reg);
          }

          if (Op.isDef() || ScoreBrackets.hasPendingEvent(EXP_LDS_ACCESS)) {
            ScoreBrackets.determineWaitForPhysReg(EXP_CNT, Reg, Wait);
          }
          ScoreBrackets.determineWaitForPhysReg(DS_CNT, Reg, Wait);
        } else if (Op.getReg() == AMDGPU::SCC) {
          ScoreBrackets.determineWaitForPhysReg(KM_CNT, Reg, Wait);
        } else {
          ScoreBrackets.determineWaitForPhysReg(SmemAccessCounter, Reg, Wait);
        }

        if (ST->hasWaitXcnt() && Op.isDef())
          ScoreBrackets.determineWaitForPhysReg(X_CNT, Reg, Wait);
      }
    }
  }

  // Ensure safety against exceptions from outstanding memory operations while
  // waiting for a barrier:
  //
  //  * Some subtargets safely handle backing off the barrier in hardware
  //    when an exception occurs.
  //  * Some subtargets have an implicit S_WAITCNT 0 before barriers, so that
  //    there can be no outstanding memory operations during the wait.
  //  * Subtargets with split barriers don't need to back off the barrier; it
  //    is up to the trap handler to preserve the user barrier state correctly.
  //
  // In all other cases, ensure safety by ensuring that there are no outstanding
  // memory operations.
  if (Opc == AMDGPU::S_BARRIER && !ST->hasAutoWaitcntBeforeBarrier() &&
      !ST->hasBackOffBarrier()) {
    Wait = Wait.combined(WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/true));
  }

  // TODO: Remove this work-around, enable the assert for Bug 457939
  //       after fixing the scheduler. Also, the Shader Compiler code is
  //       independent of target.
  if (SIInstrInfo::isCBranchVCCZRead(MI) && ST->hasReadVCCZBug() &&
      ScoreBrackets.hasPendingEvent(SMEM_ACCESS)) {
    Wait.DsCnt = 0;
  }

  // Verify that the wait is actually needed.
  ScoreBrackets.simplifyWaitcnt(Wait);

  // It is only necessary to insert an S_WAITCNT_DEPCTR instruction that
  // waits on VA_VDST if the instruction it would precede is not a VALU
  // instruction, since hardware handles VALU->VGPR->VALU hazards in
  // expert scheduling mode.
  if (TII->isVALU(MI))
    Wait.VaVdst = ~0u;

  // Since the translation for VMEM addresses occur in-order, we can apply the
  // XCnt if the current instruction is of VMEM type and has a memory
  // dependency with another VMEM instruction in flight.
  if (Wait.XCnt != ~0u && isVmemAccess(MI)) {
    ScoreBrackets.applyWaitcnt(X_CNT, Wait.XCnt);
    Wait.XCnt = ~0u;
  }

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
  // Only force emit VA_VDST and VM_VSRC if expert mode is enabled.
  if (IsExpertMode) {
    if (ForceEmitWaitcnt[VA_VDST])
      Wait.VaVdst = 0;
    if (ForceEmitWaitcnt[VM_VSRC])
      Wait.VmVsrc = 0;
  }

  if (FlushFlags.FlushVmCnt) {
    if (ScoreBrackets.hasPendingEvent(LOAD_CNT))
      Wait.LoadCnt = 0;
    if (ScoreBrackets.hasPendingEvent(SAMPLE_CNT))
      Wait.SampleCnt = 0;
    if (ScoreBrackets.hasPendingEvent(BVH_CNT))
      Wait.BvhCnt = 0;
  }

  if (FlushFlags.FlushDsCnt && ScoreBrackets.hasPendingEvent(DS_CNT))
    Wait.DsCnt = 0;

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

  // ExpCnt can be merged into VINTERP.
  if (Wait.ExpCnt != ~0u && It != Block.instr_end() &&
      SIInstrInfo::isVINTERP(*It)) {
    MachineOperand *WaitExp =
        TII->getNamedOperand(*It, AMDGPU::OpName::waitexp);
    if (Wait.ExpCnt < WaitExp->getImm()) {
      WaitExp->setImm(Wait.ExpCnt);
      Modified = true;
    }
    // Apply ExpCnt before resetting it, so applyWaitcnt below sees all counts.
    ScoreBrackets.applyWaitcnt(EXP_CNT, Wait.ExpCnt);
    Wait.ExpCnt = ~0u;

    LLVM_DEBUG(dbgs() << "generateWaitcnt\n"
                      << "Update Instr: " << *It);
  }

  if (WCG->createNewWaitcnt(Block, It, Wait, ScoreBrackets))
    Modified = true;

  // Any counts that could have been applied to any existing waitcnt
  // instructions will have been done so, now deal with any remaining.
  ScoreBrackets.applyWaitcnt(Wait);

  return Modified;
}

std::optional<WaitEventType>
SIInsertWaitcnts::getExpertSchedulingEventType(const MachineInstr &Inst) const {
  if (TII->isVALU(Inst)) {
    // Core/Side-, DP-, XDL- and TRANS-MACC VALU instructions complete
    // out-of-order with respect to each other, so each of these classes
    // has its own event.

    if (TII->isXDL(Inst))
      return VGPR_XDL_WRITE;

    if (TII->isTRANS(Inst))
      return VGPR_TRANS_WRITE;

    if (AMDGPU::isDPMACCInstruction(Inst.getOpcode()))
      return VGPR_DPMACC_WRITE;

    return VGPR_CSMACC_WRITE;
  }

  // FLAT and LDS instructions may read their VGPR sources out-of-order
  // with respect to each other and all other VMEM instructions, so
  // each of these also has a separate event.

  if (TII->isFLAT(Inst))
    return VGPR_FLAT_READ;

  if (TII->isDS(Inst))
    return VGPR_LDS_READ;

  if (TII->isVMEM(Inst) || TII->isVIMAGE(Inst) || TII->isVSAMPLE(Inst))
    return VGPR_VMEM_READ;

  // Otherwise, no hazard.

  return {};
}

bool SIInsertWaitcnts::isVmemAccess(const MachineInstr &MI) const {
  return (TII->isFLAT(MI) && TII->mayAccessVMEMThroughFlat(MI)) ||
         (TII->isVMEM(MI) && !AMDGPU::getMUBUFIsBufferInv(MI.getOpcode()));
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
  // For architectures with X_CNT, mark the source address operands
  // with the appropriate counter values.
  // TODO: Use the (TSFlags & SIInstrFlags::DS_CNT) property everywhere.

  bool IsVMEMAccess = false;
  bool IsSMEMAccess = false;

  if (IsExpertMode) {
    if (const auto ET = getExpertSchedulingEventType(Inst))
      ScoreBrackets->updateByEvent(*ET, Inst);
  }

  if (TII->isDS(Inst) && TII->usesLGKM_CNT(Inst)) {
    if (TII->isAlwaysGDS(Inst.getOpcode()) ||
        TII->hasModifiersSet(Inst, AMDGPU::OpName::gds)) {
      ScoreBrackets->updateByEvent(GDS_ACCESS, Inst);
      ScoreBrackets->updateByEvent(GDS_GPR_LOCK, Inst);
      ScoreBrackets->setPendingGDS();
    } else {
      ScoreBrackets->updateByEvent(LDS_ACCESS, Inst);
    }
  } else if (TII->isFLAT(Inst)) {
    if (SIInstrInfo::isGFX12CacheInvOrWBInst(Inst.getOpcode())) {
      ScoreBrackets->updateByEvent(getVmemWaitEventType(Inst), Inst);
      return;
    }

    assert(Inst.mayLoadOrStore());

    int FlatASCount = 0;

    if (TII->mayAccessVMEMThroughFlat(Inst)) {
      ++FlatASCount;
      IsVMEMAccess = true;
      ScoreBrackets->updateByEvent(getVmemWaitEventType(Inst), Inst);
    }

    if (TII->mayAccessLDSThroughFlat(Inst)) {
      ++FlatASCount;
      ScoreBrackets->updateByEvent(LDS_ACCESS, Inst);
    }

    // Async/LDSDMA operations have FLAT encoding but do not actually use flat
    // pointers. They do have two operands that each access global and LDS, thus
    // making it appear at this point that they are using a flat pointer. Filter
    // them out, and for the rest, generate a dependency on flat pointers so
    // that both VM and LGKM counters are flushed.
    if (!SIInstrInfo::isLDSDMA(Inst) && FlatASCount > 1)
      ScoreBrackets->setPendingFlat();
  } else if (SIInstrInfo::isVMEM(Inst) &&
             !llvm::AMDGPU::getMUBUFIsBufferInv(Inst.getOpcode())) {
    IsVMEMAccess = true;
    ScoreBrackets->updateByEvent(getVmemWaitEventType(Inst), Inst);

    if (ST->vmemWriteNeedsExpWaitcnt() &&
        (Inst.mayStore() || SIInstrInfo::isAtomicRet(Inst))) {
      ScoreBrackets->updateByEvent(VMW_GPR_LOCK, Inst);
    }
  } else if (TII->isSMRD(Inst)) {
    IsSMEMAccess = true;
    ScoreBrackets->updateByEvent(SMEM_ACCESS, Inst);
  } else if (Inst.isCall()) {
    // Act as a wait on everything
    ScoreBrackets->applyWaitcnt(WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/false));
    ScoreBrackets->setStateOnFunctionEntryOrReturn();
  } else if (SIInstrInfo::isLDSDIR(Inst)) {
    ScoreBrackets->updateByEvent(EXP_LDS_ACCESS, Inst);
  } else if (TII->isVINTERP(Inst)) {
    int64_t Imm = TII->getNamedOperand(Inst, AMDGPU::OpName::waitexp)->getImm();
    ScoreBrackets->applyWaitcnt(EXP_CNT, Imm);
  } else if (SIInstrInfo::isEXP(Inst)) {
    unsigned Imm = TII->getNamedOperand(Inst, AMDGPU::OpName::tgt)->getImm();
    if (Imm >= AMDGPU::Exp::ET_PARAM0 && Imm <= AMDGPU::Exp::ET_PARAM31)
      ScoreBrackets->updateByEvent(EXP_PARAM_ACCESS, Inst);
    else if (Imm >= AMDGPU::Exp::ET_POS0 && Imm <= AMDGPU::Exp::ET_POS_LAST)
      ScoreBrackets->updateByEvent(EXP_POS_ACCESS, Inst);
    else
      ScoreBrackets->updateByEvent(EXP_GPR_LOCK, Inst);
  } else if (SIInstrInfo::isSBarrierSCCWrite(Inst.getOpcode())) {
    ScoreBrackets->updateByEvent(SCC_WRITE, Inst);
  } else {
    switch (Inst.getOpcode()) {
    case AMDGPU::S_SENDMSG:
    case AMDGPU::S_SENDMSG_RTN_B32:
    case AMDGPU::S_SENDMSG_RTN_B64:
    case AMDGPU::S_SENDMSGHALT:
      ScoreBrackets->updateByEvent(SQ_MESSAGE, Inst);
      break;
    case AMDGPU::S_MEMTIME:
    case AMDGPU::S_MEMREALTIME:
    case AMDGPU::S_GET_BARRIER_STATE_M0:
    case AMDGPU::S_GET_BARRIER_STATE_IMM:
      ScoreBrackets->updateByEvent(SMEM_ACCESS, Inst);
      break;
    }
  }

  if (!ST->hasWaitXcnt())
    return;

  if (IsVMEMAccess)
    ScoreBrackets->updateByEvent(VMEM_GROUP, Inst);

  if (IsSMEMAccess)
    ScoreBrackets->updateByEvent(SMEM_GROUP, Inst);
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

  // Check if "other" has keys we don't have, and create default entries for
  // those. If they remain empty after merging, we will clean it up after.
  for (auto K : Other.VMem.keys())
    VMem.try_emplace(K);
  for (auto K : Other.SGPRs.keys())
    SGPRs.try_emplace(K);

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

    if (T == KM_CNT) {
      StrictDom |= mergeScore(M, SCCScore, Other.SCCScore);
      if (Other.hasPendingEvent(SCC_WRITE)) {
        unsigned OldEventsHasSCCWrite = OldEvents & (1 << SCC_WRITE);
        if (!OldEventsHasSCCWrite) {
          PendingSCCWrite = Other.PendingSCCWrite;
        } else if (PendingSCCWrite != Other.PendingSCCWrite) {
          PendingSCCWrite = nullptr;
        }
      }
    }

    for (auto &[RegID, Info] : VMem)
      StrictDom |= mergeScore(M, Info.Scores[T], Other.getVMemScore(RegID, T));

    if (isSmemCounter(T)) {
      unsigned Idx = getSgprScoresIdx(T);
      for (auto &[RegID, Info] : SGPRs) {
        auto It = Other.SGPRs.find(RegID);
        unsigned OtherScore =
            (It != Other.SGPRs.end()) ? It->second.Scores[Idx] : 0;
        StrictDom |= mergeScore(M, Info.Scores[Idx], OtherScore);
      }
    }
  }

  for (auto &[TID, Info] : VMem) {
    if (auto It = Other.VMem.find(TID); It != Other.VMem.end()) {
      unsigned char NewVmemTypes = Info.VMEMTypes | It->second.VMEMTypes;
      StrictDom |= NewVmemTypes != Info.VMEMTypes;
      Info.VMEMTypes = NewVmemTypes;
    }
  }

  purgeEmptyTrackingData();
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

void SIInsertWaitcnts::setSchedulingMode(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator I,
                                         bool ExpertMode) const {
  const unsigned EncodedReg = AMDGPU::Hwreg::HwregEncoding::encode(
      AMDGPU::Hwreg::ID_SCHED_MODE, AMDGPU::Hwreg::HwregOffset::Default, 2);
  BuildMI(MBB, I, DebugLoc(), TII->get(AMDGPU::S_SETREG_IMM32_B32))
      .addImm(ExpertMode ? 2 : 0)
      .addImm(EncodedReg);
}

// Track back-to-back atomic RMW instructions, referred to as a block.
//
// Determines whether \p MI starts a new atomic RMW block, is inside
// an existing block, or is outside of a block. A block is broken when a
// CU-scoped memory op or an atomic store is encountered. ALU ops
// and non-memory instructions don't break a block. The function returns
// the new state after processing the current instruction based on
// \p PrevState, the previously captured state.
AtomicRMWState
SIInsertWaitcnts::getAtomicRMWState(MachineInstr &MI,
                                    AtomicRMWState PrevState) const {
  if (isAtomicRMW(MI)) {
    // Transition from NotInBlock -> NewBlock -> InsideBlock.
    if (PrevState == AtomicRMWState::NotInBlock)
      return AtomicRMWState::NewBlock;
    if (PrevState == AtomicRMWState::NewBlock)
      return AtomicRMWState::InsideBlock;

    return PrevState;
  }

  // LDS memory operations don't break the block.
  if (TII->isDS(MI) || (TII->isFLAT(MI) && TII->mayAccessLDSThroughFlat(MI)))
    return PrevState;

  // Reset the atomic RMW block state when found other VMEM and SMEM operations.
  if (MI.mayLoad() ^ MI.mayStore())
    return AtomicRMWState::NotInBlock;

  // Return the previous state otherwise.
  return PrevState;
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
  AtomicRMWState RMWState = AtomicRMWState::NotInBlock;

  for (MachineBasicBlock::instr_iterator Iter = Block.instr_begin(),
                                         E = Block.instr_end();
       Iter != E;) {
    MachineInstr &Inst = *Iter;
    if (Inst.isMetaInstruction()) {
      ++Iter;
      continue;
    }
    // Get the atomic RMW block state for current instruction.
    RMWState = getAtomicRMWState(Inst, RMWState);

    // Track pre-existing waitcnts that were added in earlier iterations or by
    // the memory legalizer.
    if (isWaitInstr(Inst) ||
        (IsExpertMode && Inst.getOpcode() == AMDGPU::S_WAITCNT_DEPCTR)) {
      ++Iter;
      bool IsSoftXcnt = isSoftXcnt(Inst);
      // The Memory Legalizer conservatively inserts a soft xcnt before each
      // atomic RMW operation. However, for sequences of back-to-back atomic
      // RMWs, only the first s_wait_xcnt insertion is necessary. Optimize away
      // the redundant soft xcnts when we're inside an atomic RMW block.
      if (Iter != E && IsSoftXcnt) {
        // Check if the next instruction can potentially change the atomic RMW
        // state.
        RMWState = getAtomicRMWState(*Iter, RMWState);
      }

      if (IsSoftXcnt && RMWState == AtomicRMWState::InsideBlock) {
        // Delete this soft xcnt.
        Inst.eraseFromParent();
        Modified = true;
      } else if (!OldWaitcntInstr) {
        OldWaitcntInstr = &Inst;
      }
      continue;
    }

    PreheaderFlushFlags FlushFlags;
    if (Block.getFirstTerminator() == Inst)
      FlushFlags = isPreheaderToFlush(Block, ScoreBrackets);

    // Generate an s_waitcnt instruction to be placed before Inst, if needed.
    Modified |= generateWaitcntInstBefore(Inst, ScoreBrackets, OldWaitcntInstr,
                                          FlushFlags);
    OldWaitcntInstr = nullptr;

    // Restore vccz if it's not known to be correct already.
    bool RestoreVCCZ = !VCCZCorrect && SIInstrInfo::isCBranchVCCZRead(Inst);

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

  // Flush counters at the end of the block if needed (for preheaders with no
  // terminator).
  AMDGPU::Waitcnt Wait;
  if (Block.getFirstTerminator() == Block.end()) {
    PreheaderFlushFlags FlushFlags = isPreheaderToFlush(Block, ScoreBrackets);
    if (FlushFlags.FlushVmCnt) {
      if (ScoreBrackets.hasPendingEvent(LOAD_CNT))
        Wait.LoadCnt = 0;
      if (ScoreBrackets.hasPendingEvent(SAMPLE_CNT))
        Wait.SampleCnt = 0;
      if (ScoreBrackets.hasPendingEvent(BVH_CNT))
        Wait.BvhCnt = 0;
    }
    if (FlushFlags.FlushDsCnt && ScoreBrackets.hasPendingEvent(DS_CNT))
      Wait.DsCnt = 0;
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

// Return flags indicating which counters should be flushed in the preheader.
PreheaderFlushFlags
SIInsertWaitcnts::isPreheaderToFlush(MachineBasicBlock &MBB,
                                     const WaitcntBrackets &ScoreBrackets) {
  auto [Iterator, IsInserted] =
      PreheadersToFlush.try_emplace(&MBB, PreheaderFlushFlags());
  if (!IsInserted)
    return Iterator->second;

  MachineBasicBlock *Succ = MBB.getSingleSuccessor();
  if (!Succ)
    return PreheaderFlushFlags();

  MachineLoop *Loop = MLI->getLoopFor(Succ);
  if (!Loop)
    return PreheaderFlushFlags();

  if (Loop->getLoopPreheader() == &MBB) {
    Iterator->second = getPreheaderFlushFlags(Loop, ScoreBrackets);
    return Iterator->second;
  }

  return PreheaderFlushFlags();
}

bool SIInsertWaitcnts::isVMEMOrFlatVMEM(const MachineInstr &MI) const {
  if (SIInstrInfo::isFLAT(MI))
    return TII->mayAccessVMEMThroughFlat(MI);
  return SIInstrInfo::isVMEM(MI);
}

bool SIInsertWaitcnts::isDSRead(const MachineInstr &MI) const {
  return SIInstrInfo::isDS(MI) && MI.mayLoad() && !MI.mayStore();
}

// Check if instruction is a store to LDS that is counted via DSCNT
// (where that counter exists).
bool SIInsertWaitcnts::mayStoreIncrementingDSCNT(const MachineInstr &MI) const {
  if (!MI.mayStore())
    return false;
  if (SIInstrInfo::isDS(MI))
    return true;
  return false;
}

// Return flags indicating which counters should be flushed in the preheader of
// the given loop. We currently decide to flush in a few situations:
// For VMEM (FlushVmCnt):
// 1. The loop contains vmem store(s), no vmem load and at least one use of a
//    vgpr containing a value that is loaded outside of the loop. (Only on
//    targets with no vscnt counter).
// 2. The loop contains vmem load(s), but the loaded values are not used in the
//    loop, and at least one use of a vgpr containing a value that is loaded
//    outside of the loop.
// For DS (FlushDsCnt, GFX12+ only):
// 3. The loop contains no DS reads, and at least one use of a vgpr containing
//    a value that is DS loaded outside of the loop.
// 4. The loop contains DS read(s), loaded values are not used in the same
//    iteration but in the next iteration (prefetch pattern), and at least one
//    use of a vgpr containing a value that is DS loaded outside of the loop.
//    Flushing in preheader reduces wait overhead if the wait requirement in
//    iteration 1 would otherwise be more strict.
PreheaderFlushFlags
SIInsertWaitcnts::getPreheaderFlushFlags(MachineLoop *ML,
                                         const WaitcntBrackets &Brackets) {
  PreheaderFlushFlags Flags;
  bool HasVMemLoad = false;
  bool HasVMemStore = false;
  bool SeenDSStoreInLoop = false;
  bool UsesVgprLoadedOutsideVMEM = false;
  bool UsesVgprLoadedOutsideDS = false;
  bool VMemInvalidated = false;
  // DS optimization only applies to GFX12+ where DS_CNT is separate.
  bool DSInvalidated = !ST->hasExtendedWaitCounts();
  DenseSet<MCRegUnit> VgprUse;
  DenseSet<MCRegUnit> VgprDefVMEM;
  DenseSet<MCRegUnit> VgprDefDS;

  for (MachineBasicBlock *MBB : ML->blocks()) {
    bool SeenDSStoreInCurrMBB = false;
    for (MachineInstr &MI : *MBB) {
      if (isVMEMOrFlatVMEM(MI)) {
        HasVMemLoad |= MI.mayLoad();
        HasVMemStore |= MI.mayStore();
      }
      if (mayStoreIncrementingDSCNT(MI))
        SeenDSStoreInCurrMBB = true;
      // Stores postdominated by a barrier will have a wait at the barrier
      // and thus no need to be waited at the loop header. Barrier found
      // later in the same MBB during in-order traversal is used here as a
      // cheaper alternative to postdomination check.
      if (MI.getOpcode() == AMDGPU::S_BARRIER)
        SeenDSStoreInCurrMBB = false;
      for (const MachineOperand &Op : MI.all_uses()) {
        if (Op.isDebug() || !TRI->isVectorRegister(*MRI, Op.getReg()))
          continue;
        // Vgpr use
        for (MCRegUnit RU : TRI->regunits(Op.getReg().asMCReg())) {
          // If we find a register that is loaded inside the loop, 1. and 2.
          // are invalidated.
          if (VgprDefVMEM.contains(RU))
            VMemInvalidated = true;

          // Check for DS loads used inside the loop
          if (VgprDefDS.contains(RU))
            DSInvalidated = true;

          // Early exit if both optimizations are invalidated
          if (VMemInvalidated && DSInvalidated)
            return Flags;

          VgprUse.insert(RU);
          // Check if this register has a pending VMEM load from outside the
          // loop (value loaded outside and used inside).
          VMEMID ID = toVMEMID(RU);
          bool HasPendingVMEM =
              Brackets.getVMemScore(ID, LOAD_CNT) >
                  Brackets.getScoreLB(LOAD_CNT) ||
              Brackets.getVMemScore(ID, SAMPLE_CNT) >
                  Brackets.getScoreLB(SAMPLE_CNT) ||
              Brackets.getVMemScore(ID, BVH_CNT) > Brackets.getScoreLB(BVH_CNT);
          if (HasPendingVMEM)
            UsesVgprLoadedOutsideVMEM = true;
          // Check if loaded outside the loop via DS (not VMEM/FLAT).
          // Only consider it a DS load if there's no pending VMEM load for
          // this register, since FLAT can set both counters.
          if (!HasPendingVMEM &&
              Brackets.getVMemScore(ID, DS_CNT) > Brackets.getScoreLB(DS_CNT))
            UsesVgprLoadedOutsideDS = true;
        }
      }

      // VMem load vgpr def
      if (isVMEMOrFlatVMEM(MI) && MI.mayLoad()) {
        for (const MachineOperand &Op : MI.all_defs()) {
          for (MCRegUnit RU : TRI->regunits(Op.getReg().asMCReg())) {
            // If we find a register that is loaded inside the loop, 1. and 2.
            // are invalidated.
            if (VgprUse.contains(RU))
              VMemInvalidated = true;
            VgprDefVMEM.insert(RU);
          }
        }
        // Early exit if both optimizations are invalidated
        if (VMemInvalidated && DSInvalidated)
          return Flags;
      }

      // DS read vgpr def
      // Note: Unlike VMEM, we DON'T invalidate when VgprUse.contains(RegNo).
      // If USE comes before DEF, it's the prefetch pattern (use value from
      // previous iteration, load for next iteration). We should still flush
      // in preheader so iteration 1 doesn't need to wait inside the loop.
      // Only invalidate when DEF comes before USE (same-iteration consumption,
      // checked above when processing uses).
      if (isDSRead(MI)) {
        for (const MachineOperand &Op : MI.all_defs()) {
          for (MCRegUnit RU : TRI->regunits(Op.getReg().asMCReg())) {
            VgprDefDS.insert(RU);
          }
        }
      }
    }
    // Accumulate unprotected DS stores from this MBB
    SeenDSStoreInLoop |= SeenDSStoreInCurrMBB;
  }

  // VMEM flush decision
  if (!VMemInvalidated && UsesVgprLoadedOutsideVMEM &&
      ((!ST->hasVscnt() && HasVMemStore && !HasVMemLoad) ||
       (HasVMemLoad && ST->hasVmemWriteVgprInOrder())))
    Flags.FlushVmCnt = true;

  // DS flush decision: flush if loop uses DS-loaded values from outside
  // and either has no DS reads in the loop, or DS reads whose results
  // are not used in the loop.
  // DSInvalidated is pre-set to true on non-GFX12+ targets where DS_CNT
  // is LGKM_CNT which also tracks FLAT/SMEM.
  if (!DSInvalidated && !SeenDSStoreInLoop && UsesVgprLoadedOutsideDS)
    Flags.FlushDsCnt = true;

  return Flags;
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

  // Initialize hardware limits first, as they're needed by the generators.
  Limits = AMDGPU::HardwareLimits(IV);

  if (ST->hasExtendedWaitCounts()) {
    IsExpertMode = ST->hasExpertSchedulingMode() &&
                   (ExpertSchedulingModeFlag.getNumOccurrences()
                        ? ExpertSchedulingModeFlag
                        : MF.getFunction()
                              .getFnAttribute("amdgpu-expert-scheduling-mode")
                              .getValueAsBool());
    MaxCounter = IsExpertMode ? NUM_EXPERT_INST_CNTS : NUM_EXTENDED_INST_CNTS;
    if (!WCG)
      WCG = std::make_unique<WaitcntGeneratorGFX12Plus>(MF, MaxCounter, &Limits,
                                                        IsExpertMode);
  } else {
    MaxCounter = NUM_NORMAL_INST_CNTS;
    if (!WCG)
      WCG = std::make_unique<WaitcntGeneratorPreGFX12>(MF, NUM_NORMAL_INST_CNTS,
                                                       &Limits);
  }

  for (auto T : inst_counter_types())
    ForceEmitWaitcnt[T] = false;

  WaitEventMaskForInst = WCG->getWaitEventMask();

  SmemAccessCounter = eventCounter(WaitEventMaskForInst, SMEM_ACCESS);

  BlockInfos.clear();
  bool Modified = false;

  MachineBasicBlock &EntryBB = MF.front();

  if (!MFI->isEntryFunction()) {
    // Wait for any outstanding memory operations that the input registers may
    // depend on. We can't track them and it's better to do the wait after the
    // costly call sequence.

    // TODO: Could insert earlier and schedule more liberally with operations
    // that only use caller preserved registers.
    MachineBasicBlock::iterator I = EntryBB.begin();
    while (I != EntryBB.end() && I->isMetaInstruction())
      ++I;

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
      if (IsExpertMode) {
        unsigned Enc = AMDGPU::DepCtr::encodeFieldVaVdst(0, *ST);
        Enc = AMDGPU::DepCtr::encodeFieldVmVsrc(Enc, 0);
        BuildMI(EntryBB, I, DebugLoc(), TII->get(AMDGPU::S_WAITCNT_DEPCTR))
            .addImm(Enc);
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

  if (IsExpertMode) {
    // Enable expert scheduling on function entry. To satisfy ABI requirements
    // and to allow calls between function with different expert scheduling
    // settings, disable it around calls and before returns.

    MachineBasicBlock::iterator I = EntryBB.begin();
    while (I != EntryBB.end() && I->isMetaInstruction())
      ++I;
    setSchedulingMode(EntryBB, I, true);

    for (MachineInstr *MI : CallInsts) {
      MachineBasicBlock &MBB = *MI->getParent();
      setSchedulingMode(MBB, MI, false);
      setSchedulingMode(MBB, std::next(MI->getIterator()), true);
    }

    for (MachineInstr *MI : ReturnInsts)
      setSchedulingMode(*MI->getParent(), MI, false);

    Modified = true;
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

  CallInsts.clear();
  ReturnInsts.clear();
  ReleaseVGPRInsts.clear();
  PreheadersToFlush.clear();
  SLoadAddresses.clear();

  return Modified;
}
