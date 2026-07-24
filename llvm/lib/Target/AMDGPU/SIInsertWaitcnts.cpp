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
#include "AMDGPUHWEvents.h"
#include "AMDGPUWaitcntTracking.h"
#include "AMDGPUWaitcntUtils.h"
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
#include "llvm/TargetParser/AMDGPUTargetParser.h"

using namespace llvm;
using HWEvents = AMDGPU::HWEvents;
using WaitcntBrackets = AMDGPU::WaitcntBrackets;

#define DEBUG_TYPE "si-insert-waitcnts"

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

template <typename EmitWaitcntFn>
static void EmitExpandedWaitcnt(unsigned Outstanding, unsigned Target,
                                EmitWaitcntFn &&EmitWaitcnt) {
  // Emit waitcnts from (Outstanding - 1) down to Target.
  for (unsigned I = Outstanding - 1; I > Target && I != ~0u; --I)
    EmitWaitcnt(I);
  EmitWaitcnt(Target);
}

} // namespace

namespace {

// Maps values of InstCounterType to the instruction that waits on that
// counter. Only used if GCNSubtarget::hasExtendedWaitCounts()
// returns true, and does not cover VA_VDST or VM_VSRC.
static const unsigned
    instrsForExtendedCounterTypes[AMDGPU::NUM_EXTENDED_INST_CNTS] = {
        AMDGPU::S_WAIT_LOADCNT,   AMDGPU::S_WAIT_DSCNT,
        AMDGPU::S_WAIT_EXPCNT,    AMDGPU::S_WAIT_STORECNT,
        AMDGPU::S_WAIT_SAMPLECNT, AMDGPU::S_WAIT_BVHCNT,
        AMDGPU::S_WAIT_KMCNT,     AMDGPU::S_WAIT_XCNT,
        AMDGPU::S_WAIT_ASYNCCNT,  AMDGPU::S_WAIT_TENSORCNT};

// ASYNCMARK and WAIT_ASYNCMARK are meta instructions that emit no hardware
// code but still need to be processed by this pass for async vmcnt tracking.
static bool isNonWaitcntMetaInst(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AMDGPU::ASYNCMARK:
  case AMDGPU::WAIT_ASYNCMARK:
    return false;
  default:
    return MI.isMetaInstruction();
  }
}

#ifndef NDEBUG
static bool isNormalMode(AMDGPU::InstCounterType MaxCounter) {
  return MaxCounter == AMDGPU::NUM_NORMAL_INST_CNTS;
}
#endif // NDEBUG

// This abstracts the logic for generating and updating S_WAIT* instructions
// away from the analysis that determines where they are needed. This was
// done because the set of counters and instructions for waiting on them
// underwent a major shift with gfx12, sufficiently so that having this
// abstraction allows the main analysis logic to be simpler than it would
// otherwise have had to become.
class WaitcntGenerator {
protected:
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  AMDGPU::IsaVersion IV;
  AMDGPU::InstCounterType MaxCounter;
  bool OptNone;
  bool ExpandWaitcntProfiling = false;
  const AMDGPU::HardwareLimits &Limits;

public:
  WaitcntGenerator() = delete;
  WaitcntGenerator(const WaitcntGenerator &) = delete;
  WaitcntGenerator(const MachineFunction &MF,
                   AMDGPU::InstCounterType MaxCounter,
                   const AMDGPU::HardwareLimits &Limits)
      : ST(MF.getSubtarget<GCNSubtarget>()), TII(*ST.getInstrInfo()),
        IV(AMDGPU::getIsaVersion(ST.getCPU())), MaxCounter(MaxCounter),
        OptNone(MF.getFunction().hasOptNone() ||
                MF.getTarget().getOptLevel() == CodeGenOptLevel::None),
        ExpandWaitcntProfiling(
            MF.getFunction().hasFnAttribute("amdgpu-expand-waitcnt-profiling")),
        Limits(Limits) {}

  // Return true if the current function should be compiled with no
  // optimization.
  bool isOptNone() const { return OptNone; }

  unsigned getLimit(AMDGPU::InstCounterType E) const { return Limits.get(E); }

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

  // Returns the set of HWEvents that corresponds to counter \p T.
  virtual HWEvents getWaitEvents(AMDGPU::InstCounterType T) const = 0;

  /// \returns the counter that corresponds to event \p E.
  AMDGPU::InstCounterType getCounterFromEvent(HWEvents E) const {
    assert(E.size() == 1 && "Cannot handle a mask of events!");
    for (auto T : AMDGPU::inst_counter_types()) {
      if (getWaitEvents(T) & E)
        return T;
    }
    llvm_unreachable("event type has no associated counter");
  }

  // Returns a new waitcnt with all counters except VScnt set to 0. If
  // IncludeVSCnt is true, VScnt is set to 0, otherwise it is set to ~0u.
  // AsyncCnt and TensorCnt always default to ~0u (don't wait for it). They
  // are only updated when a call to @llvm.amdgcn.wait.asyncmark() is
  // processed.
  virtual AMDGPU::Waitcnt getAllZeroWaitcnt(bool IncludeVSCnt) const = 0;

  virtual ~WaitcntGenerator() = default;
};

class WaitcntGeneratorPreGFX12 final : public WaitcntGenerator {
  static constexpr const HWEvents
      WaitEventMaskForInstPreGFX12[AMDGPU::NUM_INST_CNTS] = {
          HWEvents::VMEM_READ_ACCESS | HWEvents::VMEM_SAMPLER_READ_ACCESS |
              HWEvents::VMEM_BVH_READ_ACCESS,
          HWEvents::SMEM_ACCESS | HWEvents::LDS_ACCESS | HWEvents::GDS_ACCESS |
              HWEvents::SQ_MESSAGE,
          HWEvents::EXP_GPR_LOCK | HWEvents::GDS_GPR_LOCK |
              HWEvents::VMW_GPR_LOCK | HWEvents::EXP_PARAM_ACCESS |
              HWEvents::EXP_POS_ACCESS | HWEvents::EXP_LDS_ACCESS,
          HWEvents::VMEM_WRITE_ACCESS | HWEvents::SCRATCH_WRITE_ACCESS,
          HWEvents::NONE,
          HWEvents::NONE,
          HWEvents::NONE,
          HWEvents::NONE,
          HWEvents::NONE,
          HWEvents::NONE,
          HWEvents::NONE,
          HWEvents::NONE,
          HWEvents::NONE};

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

  HWEvents getWaitEvents(AMDGPU::InstCounterType T) const override {
    HWEvents EVs = WaitEventMaskForInstPreGFX12[T];
    if (T == AMDGPU::LOAD_CNT && !ST.hasVscnt())
      EVs |= WaitEventMaskForInstPreGFX12[AMDGPU::STORE_CNT];
    return EVs;
  }

  AMDGPU::Waitcnt getAllZeroWaitcnt(bool IncludeVSCnt) const override;
};

class WaitcntGeneratorGFX12Plus final : public WaitcntGenerator {
protected:
  bool IsExpertMode;
  static constexpr const HWEvents
      WaitEventMaskForInstGFX12Plus[AMDGPU::NUM_INST_CNTS] = {
          HWEvents::VMEM_READ_ACCESS | HWEvents::GLOBAL_INV_ACCESS,
          HWEvents::LDS_ACCESS | HWEvents::GDS_ACCESS,
          HWEvents::EXP_GPR_LOCK | HWEvents::GDS_GPR_LOCK |
              HWEvents::VMW_GPR_LOCK | HWEvents::EXP_PARAM_ACCESS |
              HWEvents::EXP_POS_ACCESS | HWEvents::EXP_LDS_ACCESS,

          HWEvents::VMEM_WRITE_ACCESS | HWEvents::SCRATCH_WRITE_ACCESS,
          HWEvents::VMEM_SAMPLER_READ_ACCESS,
          HWEvents::VMEM_BVH_READ_ACCESS,

          HWEvents::SMEM_ACCESS | HWEvents::SQ_MESSAGE | HWEvents::SCC_WRITE,
          HWEvents::VMEM_GROUP | HWEvents::SMEM_GROUP,
          HWEvents::ASYNC_ACCESS,
          HWEvents::TENSOR_ACCESS,
          HWEvents::VGPR_CSMACC_READ | HWEvents::VGPR_DPMACC_READ |
              HWEvents::VGPR_TRANS_READ | HWEvents::VGPR_XDL_READ,
          HWEvents::VGPR_CSMACC_WRITE | HWEvents::VGPR_DPMACC_WRITE |
              HWEvents::VGPR_TRANS_WRITE | HWEvents::VGPR_XDL_WRITE,
          HWEvents::VGPR_LDS_READ | HWEvents::VGPR_FLAT_READ |
              HWEvents::VGPR_VMEM_READ};

public:
  WaitcntGeneratorGFX12Plus() = delete;
  WaitcntGeneratorGFX12Plus(const MachineFunction &MF,
                            AMDGPU::InstCounterType MaxCounter,
                            const AMDGPU::HardwareLimits &Limits,
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

  HWEvents getWaitEvents(AMDGPU::InstCounterType T) const override {
    return WaitEventMaskForInstGFX12Plus[T];
  }

  AMDGPU::Waitcnt getAllZeroWaitcnt(bool IncludeVSCnt) const override;
};

// Flags indicating which counters should be flushed in a loop preheader.
struct PreheaderFlushFlags {
  bool FlushVmCnt = false;
  bool FlushDsCnt = false;
};

class SIInsertWaitcnts {
  DenseMap<const Value *, MachineBasicBlock *> SLoadAddresses;
  DenseMap<MachineBasicBlock *, PreheaderFlushFlags> PreheadersToFlush;
  MachineLoopInfo &MLI;
  MachinePostDominatorTree &PDT;
  AliasAnalysis *AA = nullptr;
  MachineFunction &MF;

  struct BlockInfo {
    std::unique_ptr<WaitcntBrackets> Incoming;
    bool Dirty = true;
    BlockInfo() = default;
    BlockInfo(BlockInfo &&) = default;
    BlockInfo &operator=(BlockInfo &&) = default;
    ~BlockInfo();
  };

  MapVector<MachineBasicBlock *, BlockInfo> BlockInfos;

  bool ForceEmitWaitcnt[AMDGPU::NUM_INST_CNTS] = {};

  std::unique_ptr<WaitcntGenerator> WCG;

  // Remember call and return instructions in the function.
  DenseSet<MachineInstr *> CallInsts;
  DenseSet<MachineInstr *> ReturnInsts;

  // Remember all S_ENDPGM instructions. The boolean flag is true if there might
  // be outstanding stores but definitely no outstanding scratch stores, to help
  // with insertion of DEALLOC_VGPRS messages.
  DenseMap<MachineInstr *, bool> EndPgmInsts;

  bool IsExpertMode = false;
  bool IsTgSplit = false;

  AMDGPU::InstCounterType SmemAccessCounter;
  AMDGPU::InstCounterType MaxCounter;
  AMDGPU::HardwareLimits Limits;

public:
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  const MachineRegisterInfo &MRI;

  SIInsertWaitcnts(MachineLoopInfo &MLI, MachinePostDominatorTree &PDT,
                   AliasAnalysis *AA, MachineFunction &MF)
      : MLI(MLI), PDT(PDT), AA(AA), MF(MF), ST(MF.getSubtarget<GCNSubtarget>()),
        TII(*ST.getInstrInfo()), TRI(TII.getRegisterInfo()),
        MRI(MF.getRegInfo()) {
    IsTgSplit =
        (ST.hasTgSplitSupport() && AMDGPU::isTgSplitEnabled(MF.getFunction()));
  }

  PreheaderFlushFlags getPreheaderFlushFlags(MachineLoop *ML,
                                             const WaitcntBrackets &Brackets);
  PreheaderFlushFlags isPreheaderToFlush(MachineBasicBlock &MBB,
                                         const WaitcntBrackets &ScoreBrackets);
  bool isVMEMOrFlatVMEM(const MachineInstr &MI) const;
  bool isDSRead(const MachineInstr &MI) const;
  bool mayStoreIncrementingDSCNT(const MachineInstr &MI) const;
  bool run();

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
  /// Removes redundant Soft Xcnt Waitcnts in \p Block emitted by the Memory
  /// Legalizer. Returns true if block was modified.
  bool removeRedundantSoftXcnts(MachineBasicBlock &Block);
  void setSchedulingMode(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                         bool ExpertMode) const;
};

SIInsertWaitcnts::BlockInfo::~BlockInfo() = default;

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

bool WaitcntGenerator::promoteSoftWaitCnt(MachineInstr *Waitcnt) const {
  unsigned Opcode = SIInstrInfo::getNonSoftWaitcntOpcode(Waitcnt->getOpcode());
  if (Opcode == Waitcnt->getOpcode())
    return false;

  Waitcnt->setDesc(TII.get(Opcode));
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
    if (isNonWaitcntMetaInst(II)) {
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
      assert(ST.hasVMemToLDSLoad());
      LLVM_DEBUG(dbgs() << "Processing S_WAITCNT_lds_direct: " << II
                        << "Before: " << Wait << '\n';);
      ScoreBrackets.determineWaitForLDSDMA(AMDGPU::LOAD_CNT,
                                           AMDGPU::LDSDMA_BEGIN, Wait);
      LLVM_DEBUG(dbgs() << "After: " << Wait << '\n';);

      // It is possible (but unlikely) that this is the only wait instruction,
      // in which case, we exit this loop without a WaitcntInstr to consume
      // `Wait`. But that works because `Wait` was passed in by reference, and
      // the callee eventually calls createNewWaitcnt on it. We test this
      // possibility in an articial MIR test since such a situation cannot be
      // recreated by running the memory legalizer.
      II.eraseFromParent();
    } else if (Opcode == AMDGPU::WAIT_ASYNCMARK) {
      unsigned N = II.getOperand(0).getImm();
      LLVM_DEBUG(dbgs() << "Processing WAIT_ASYNCMARK: " << II << '\n';);
      AMDGPU::Waitcnt OldWait = ScoreBrackets.determineAsyncWait(N);
      Wait = Wait.combined(OldWait);
    } else {
      assert(Opcode == AMDGPU::S_WAITCNT_VSCNT);
      assert(II.getOperand(0).getReg() == AMDGPU::SGPR_NULL);

      unsigned OldVSCnt =
          TII.getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      if (TrySimplify)
        ScoreBrackets.simplifyWaitcnt(AMDGPU::STORE_CNT, OldVSCnt);
      Wait.set(AMDGPU::STORE_CNT,
               std::min(Wait.get(AMDGPU::STORE_CNT), OldVSCnt));

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

    ScoreBrackets.applyWaitcnt(Wait, AMDGPU::LOAD_CNT);
    ScoreBrackets.applyWaitcnt(Wait, AMDGPU::EXP_CNT);
    ScoreBrackets.applyWaitcnt(Wait, AMDGPU::DS_CNT);
    Wait.set(AMDGPU::LOAD_CNT, ~0u);
    Wait.set(AMDGPU::EXP_CNT, ~0u);
    Wait.set(AMDGPU::DS_CNT, ~0u);

    LLVM_DEBUG(It.isEnd() ? dbgs() << "applied pre-existing waitcnt\n"
                                   << "New Instr at block end: "
                                   << *WaitcntInstr << '\n'
                          : dbgs() << "applied pre-existing waitcnt\n"
                                   << "Old Instr: " << *It
                                   << "New Instr: " << *WaitcntInstr << '\n');
  }

  if (WaitcntVsCntInstr) {
    Modified |=
        updateOperandIfDifferent(*WaitcntVsCntInstr, AMDGPU::OpName::simm16,
                                 Wait.get(AMDGPU::STORE_CNT));
    Modified |= promoteSoftWaitCnt(WaitcntVsCntInstr);

    ScoreBrackets.applyWaitcnt(Wait, AMDGPU::STORE_CNT);
    Wait.set(AMDGPU::STORE_CNT, ~0u);

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
  assert(isNormalMode(MaxCounter));

  bool Modified = false;
  const DebugLoc &DL = Block.findDebugLoc(It);

  // Waits for VMcnt, LKGMcnt and/or EXPcnt are encoded together into a
  // single instruction while VScnt has its own instruction.
  if (Wait.hasWaitExceptStoreCnt()) {
    // If profiling expansion is enabled, emit an expanded sequence
    if (ExpandWaitcntProfiling) {
      // Check if any of the counters to be waited on are out-of-order.
      // If so, fall back to normal (non-expanded) behavior since expansion
      // would provide misleading profiling information.
      bool AnyOutOfOrder = false;
      for (auto CT : {AMDGPU::LOAD_CNT, AMDGPU::DS_CNT, AMDGPU::EXP_CNT}) {
        unsigned WaitCnt = Wait.get(CT);
        if (WaitCnt != ~0u && ScoreBrackets.counterOutOfOrder(CT)) {
          AnyOutOfOrder = true;
          break;
        }
      }

      if (AnyOutOfOrder) {
        // Fall back to non-expanded wait
        unsigned Enc = AMDGPU::encodeWaitcnt(IV, Wait);
        BuildMI(Block, It, DL, TII.get(AMDGPU::S_WAITCNT)).addImm(Enc);
        Modified = true;
      } else {
        // All counters are in-order, safe to expand
        for (auto CT : {AMDGPU::LOAD_CNT, AMDGPU::DS_CNT, AMDGPU::EXP_CNT}) {
          unsigned WaitCnt = Wait.get(CT);
          if (WaitCnt == ~0u)
            continue;

          unsigned Outstanding =
              std::min(ScoreBrackets.getOutstanding(CT), getLimit(CT) - 1);
          EmitExpandedWaitcnt(Outstanding, WaitCnt, [&](unsigned Count) {
            AMDGPU::Waitcnt W;
            W.set(CT, Count);
            BuildMI(Block, It, DL, TII.get(AMDGPU::S_WAITCNT))
                .addImm(AMDGPU::encodeWaitcnt(IV, W));
          });
          Modified = true;
        }
      }
    } else {
      // Normal behavior: emit single combined waitcnt
      unsigned Enc = AMDGPU::encodeWaitcnt(IV, Wait);
      [[maybe_unused]] auto SWaitInst =
          BuildMI(Block, It, DL, TII.get(AMDGPU::S_WAITCNT)).addImm(Enc);
      Modified = true;

      LLVM_DEBUG(dbgs() << "PreGFX12::createNewWaitcnt\n";
                 if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
                 dbgs() << "New Instr: " << *SWaitInst << '\n');
    }
  }

  if (Wait.hasWaitStoreCnt()) {
    assert(ST.hasVscnt());

    if (ExpandWaitcntProfiling && Wait.get(AMDGPU::STORE_CNT) != ~0u &&
        !ScoreBrackets.counterOutOfOrder(AMDGPU::STORE_CNT)) {
      // Only expand if counter is not out-of-order
      unsigned Outstanding =
          std::min(ScoreBrackets.getOutstanding(AMDGPU::STORE_CNT),
                   getLimit(AMDGPU::STORE_CNT) - 1);
      EmitExpandedWaitcnt(
          Outstanding, Wait.get(AMDGPU::STORE_CNT), [&](unsigned Count) {
            BuildMI(Block, It, DL, TII.get(AMDGPU::S_WAITCNT_VSCNT))
                .addReg(AMDGPU::SGPR_NULL, RegState::Undef)
                .addImm(Count);
          });
      Modified = true;
    } else {
      [[maybe_unused]] auto SWaitInst =
          BuildMI(Block, It, DL, TII.get(AMDGPU::S_WAITCNT_VSCNT))
              .addReg(AMDGPU::SGPR_NULL, RegState::Undef)
              .addImm(Wait.get(AMDGPU::STORE_CNT));
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
  return AMDGPU::Waitcnt(0, 0, 0, IncludeVSCnt && ST.hasVscnt() ? 0 : ~0u);
}

AMDGPU::Waitcnt
WaitcntGeneratorGFX12Plus::getAllZeroWaitcnt(bool IncludeVSCnt) const {
  unsigned ExpertVal = IsExpertMode ? 0 : ~0u;
  return AMDGPU::Waitcnt(0, 0, 0, IncludeVSCnt ? 0 : ~0u, 0, 0, 0,
                         ~0u /* XCNT */, ~0u /* ASYNC_CNT */,
                         ~0u /* TENSOR_CNT */, ExpertVal, ExpertVal, ExpertVal);
}

/// Combine consecutive S_WAIT_*CNT instructions that precede \p It and
/// follow \p OldWaitcntInstr and apply any extra waits from \p Wait that
/// were added by previous passes. Currently this pass conservatively
/// assumes that these preexisting waits are required for correctness.
bool WaitcntGeneratorGFX12Plus::applyPreexistingWaitcnt(
    WaitcntBrackets &ScoreBrackets, MachineInstr &OldWaitcntInstr,
    AMDGPU::Waitcnt &Wait, MachineBasicBlock::instr_iterator It) const {
  assert(!isNormalMode(MaxCounter));

  bool Modified = false;
  MachineInstr *CombinedLoadDsCntInstr = nullptr;
  MachineInstr *CombinedStoreDsCntInstr = nullptr;
  MachineInstr *WaitcntDepctrInstr = nullptr;
  MachineInstr *WaitInstrs[AMDGPU::NUM_EXTENDED_INST_CNTS] = {};

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
    if (isNonWaitcntMetaInst(II)) {
      LLVM_DEBUG(dbgs() << "skipped meta instruction\n");
      continue;
    }

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
          TII.getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      AMDGPU::Waitcnt OldWait = AMDGPU::decodeLoadcntDscnt(IV, OldEnc);
      if (TrySimplify)
        Wait = Wait.combined(OldWait);
      else
        RequiredWait = RequiredWait.combined(OldWait);
      // Keep the first wait_loadcnt, erase the rest.
      if (CombinedLoadDsCntInstr == nullptr) {
        CombinedLoadDsCntInstr = &II;
      } else {
        II.eraseFromParent();
        Modified = true;
      }
    } else if (Opcode == AMDGPU::S_WAIT_STORECNT_DSCNT) {
      unsigned OldEnc =
          TII.getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      AMDGPU::Waitcnt OldWait = AMDGPU::decodeStorecntDscnt(IV, OldEnc);
      if (TrySimplify)
        Wait = Wait.combined(OldWait);
      else
        RequiredWait = RequiredWait.combined(OldWait);
      // Keep the first wait_storecnt, erase the rest.
      if (CombinedStoreDsCntInstr == nullptr) {
        CombinedStoreDsCntInstr = &II;
      } else {
        II.eraseFromParent();
        Modified = true;
      }
    } else if (Opcode == AMDGPU::S_WAITCNT_DEPCTR) {
      unsigned OldEnc =
          TII.getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      AMDGPU::Waitcnt OldWait;
      // Set both counters to the decoded value from the single hardware field
      unsigned VaVdst = AMDGPU::DepCtr::decodeFieldVaVdst(OldEnc);
      OldWait.set(AMDGPU::VA_VDST_RD, VaVdst);
      OldWait.set(AMDGPU::VA_VDST_WR, VaVdst);
      OldWait.set(AMDGPU::VM_VSRC, AMDGPU::DepCtr::decodeFieldVmVsrc(OldEnc));
      if (TrySimplify)
        ScoreBrackets.simplifyWaitcnt(OldWait);
      Wait = Wait.combined(OldWait);
      if (WaitcntDepctrInstr == nullptr) {
        WaitcntDepctrInstr = &II;
      } else {
        // S_WAITCNT_DEPCTR requires special care. Don't remove a
        // duplicate if it is waiting on things other than VA_VDST or
        // VM_VSRC. If that is the case, just make sure the VA_VDST and
        // VM_VSRC subfields of the operand are set to the "no wait"
        // values.

        unsigned Enc =
            TII.getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
        Enc = AMDGPU::DepCtr::encodeFieldVmVsrc(Enc, ~0u);
        // Encode min(VA_VDST_RD, VA_VDST_WR) into the single hardware field
        unsigned VaVdst = std::min(Wait.get(AMDGPU::VA_VDST_RD),
                                   Wait.get(AMDGPU::VA_VDST_WR));
        Enc = AMDGPU::DepCtr::encodeFieldVaVdst(Enc, VaVdst);

        if (Enc != (unsigned)AMDGPU::DepCtr::getDefaultDepCtrEncoding(ST)) {
          Modified |= updateOperandIfDifferent(II, AMDGPU::OpName::simm16, Enc);
          Modified |= promoteSoftWaitCnt(&II);
        } else {
          II.eraseFromParent();
          Modified = true;
        }
      }
    } else if (Opcode == AMDGPU::S_WAITCNT_lds_direct) {
      // Architectures higher than GFX10 do not have direct loads to
      // LDS, so no work required here yet.
      II.eraseFromParent();
      Modified = true;
    } else if (Opcode == AMDGPU::WAIT_ASYNCMARK) {
      // Update the Waitcnt, but don't erase the wait.asyncmark() itself. It
      // shows up in the assembly as a comment with the original parameter N.
      unsigned N = II.getOperand(0).getImm();
      AMDGPU::Waitcnt OldWait = ScoreBrackets.determineAsyncWait(N);
      Wait = Wait.combined(OldWait);
    } else {
      std::optional<AMDGPU::InstCounterType> CT =
          AMDGPU::counterTypeForInstr(Opcode);
      assert(CT.has_value());
      unsigned OldCnt =
          TII.getNamedOperand(II, AMDGPU::OpName::simm16)->getImm();
      if (TrySimplify)
        Wait.add(CT.value(), OldCnt);
      else
        RequiredWait.add(CT.value(), OldCnt);
      // Keep the first wait of its kind, erase the rest.
      if (WaitInstrs[CT.value()] == nullptr) {
        WaitInstrs[CT.value()] = &II;
      } else {
        II.eraseFromParent();
        Modified = true;
      }
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
    if (Wait.get(AMDGPU::LOAD_CNT) != ~0u && Wait.get(AMDGPU::DS_CNT) != ~0u) {
      unsigned NewEnc = AMDGPU::encodeLoadcntDscnt(IV, Wait);
      Modified |= updateOperandIfDifferent(*CombinedLoadDsCntInstr,
                                           AMDGPU::OpName::simm16, NewEnc);
      Modified |= promoteSoftWaitCnt(CombinedLoadDsCntInstr);
      ScoreBrackets.applyWaitcnt(Wait, AMDGPU::LOAD_CNT);
      ScoreBrackets.applyWaitcnt(Wait, AMDGPU::DS_CNT);
      Wait.set(AMDGPU::LOAD_CNT, ~0u);
      Wait.set(AMDGPU::DS_CNT, ~0u);

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
    if (Wait.get(AMDGPU::STORE_CNT) != ~0u && Wait.get(AMDGPU::DS_CNT) != ~0u) {
      unsigned NewEnc = AMDGPU::encodeStorecntDscnt(IV, Wait);
      Modified |= updateOperandIfDifferent(*CombinedStoreDsCntInstr,
                                           AMDGPU::OpName::simm16, NewEnc);
      Modified |= promoteSoftWaitCnt(CombinedStoreDsCntInstr);
      ScoreBrackets.applyWaitcnt(Wait, AMDGPU::STORE_CNT);
      ScoreBrackets.applyWaitcnt(Wait, AMDGPU::DS_CNT);
      Wait.set(AMDGPU::STORE_CNT, ~0u);
      Wait.set(AMDGPU::DS_CNT, ~0u);

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

  if (Wait.get(AMDGPU::DS_CNT) != ~0u) {
    // This is a vector of addresses in WaitInstrs pointing to instructions
    // that should be removed if they are present.
    SmallVector<MachineInstr **, 2> WaitsToErase;

    // If it's known that both DScnt and either LOADcnt or STOREcnt (but not
    // both) need to be waited for, ensure that there are no existing
    // individual wait count instructions for these.

    if (Wait.get(AMDGPU::LOAD_CNT) != ~0u) {
      WaitsToErase.push_back(&WaitInstrs[AMDGPU::LOAD_CNT]);
      WaitsToErase.push_back(&WaitInstrs[AMDGPU::DS_CNT]);
    } else if (Wait.get(AMDGPU::STORE_CNT) != ~0u) {
      WaitsToErase.push_back(&WaitInstrs[AMDGPU::STORE_CNT]);
      WaitsToErase.push_back(&WaitInstrs[AMDGPU::DS_CNT]);
    }

    for (MachineInstr **WI : WaitsToErase) {
      if (!*WI)
        continue;

      (*WI)->eraseFromParent();
      *WI = nullptr;
      Modified = true;
    }
  }

  for (auto CT : inst_counter_types(AMDGPU::NUM_EXTENDED_INST_CNTS)) {
    if (!WaitInstrs[CT])
      continue;

    unsigned NewCnt = Wait.get(CT);
    if (NewCnt != ~0u) {
      Modified |= updateOperandIfDifferent(*WaitInstrs[CT],
                                           AMDGPU::OpName::simm16, NewCnt);
      Modified |= promoteSoftWaitCnt(WaitInstrs[CT]);

      ScoreBrackets.applyWaitcnt(CT, NewCnt);
      Wait.clear(CT);

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
        TII.getNamedOperand(*WaitcntDepctrInstr, AMDGPU::OpName::simm16)
            ->getImm();
    Enc = AMDGPU::DepCtr::encodeFieldVmVsrc(Enc, Wait.get(AMDGPU::VM_VSRC));
    // Encode min(VA_VDST_RD, VA_VDST_WR) into the single hardware field
    unsigned VaVdst =
        std::min(Wait.get(AMDGPU::VA_VDST_RD), Wait.get(AMDGPU::VA_VDST_WR));
    Enc = AMDGPU::DepCtr::encodeFieldVaVdst(Enc, VaVdst);

    ScoreBrackets.applyWaitcnt(AMDGPU::VA_VDST_RD, VaVdst);
    ScoreBrackets.applyWaitcnt(AMDGPU::VA_VDST_WR, VaVdst);
    ScoreBrackets.applyWaitcnt(Wait, AMDGPU::VM_VSRC);
    Wait.set(AMDGPU::VA_VDST_RD, ~0u);
    Wait.set(AMDGPU::VA_VDST_WR, ~0u);
    Wait.set(AMDGPU::VM_VSRC, ~0u);

    // If that new encoded Depctr immediate would actually still wait
    // for anything, update the instruction's operand. Otherwise it can
    // just be deleted.
    if (Enc != (unsigned)AMDGPU::DepCtr::getDefaultDepCtrEncoding(ST)) {
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
  assert(!isNormalMode(MaxCounter));

  bool Modified = false;
  const DebugLoc &DL = Block.findDebugLoc(It);

  // For GFX12+, we use separate wait instructions, which makes expansion
  // simpler
  if (ExpandWaitcntProfiling) {
    for (auto CT : inst_counter_types(AMDGPU::NUM_EXTENDED_INST_CNTS)) {
      unsigned Count = Wait.get(CT);
      if (Count == ~0u)
        continue;

      // Skip expansion for out-of-order counters - emit normal wait instead
      if (ScoreBrackets.counterOutOfOrder(CT)) {
        BuildMI(Block, It, DL, TII.get(instrsForExtendedCounterTypes[CT]))
            .addImm(Count);
        Modified = true;
        continue;
      }

      unsigned Outstanding =
          std::min(ScoreBrackets.getOutstanding(CT), getLimit(CT) - 1);
      EmitExpandedWaitcnt(Outstanding, Count, [&](unsigned Val) {
        BuildMI(Block, It, DL, TII.get(instrsForExtendedCounterTypes[CT]))
            .addImm(Val);
      });
      Modified = true;
    }
    return Modified;
  }

  // Normal behavior (no expansion)
  // Check for opportunities to use combined wait instructions.
  if (Wait.get(AMDGPU::DS_CNT) != ~0u) {
    MachineInstr *SWaitInst = nullptr;

    if (Wait.get(AMDGPU::LOAD_CNT) != ~0u) {
      unsigned Enc = AMDGPU::encodeLoadcntDscnt(IV, Wait);

      SWaitInst = BuildMI(Block, It, DL, TII.get(AMDGPU::S_WAIT_LOADCNT_DSCNT))
                      .addImm(Enc);

      Wait.set(AMDGPU::LOAD_CNT, ~0u);
      Wait.set(AMDGPU::DS_CNT, ~0u);
    } else if (Wait.get(AMDGPU::STORE_CNT) != ~0u) {
      unsigned Enc = AMDGPU::encodeStorecntDscnt(IV, Wait);

      SWaitInst = BuildMI(Block, It, DL, TII.get(AMDGPU::S_WAIT_STORECNT_DSCNT))
                      .addImm(Enc);

      Wait.set(AMDGPU::STORE_CNT, ~0u);
      Wait.set(AMDGPU::DS_CNT, ~0u);
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

  for (auto CT : inst_counter_types(AMDGPU::NUM_EXTENDED_INST_CNTS)) {
    unsigned Count = Wait.get(CT);
    if (Count == ~0u)
      continue;

    [[maybe_unused]] auto SWaitInst =
        BuildMI(Block, It, DL, TII.get(instrsForExtendedCounterTypes[CT]))
            .addImm(Count);

    Modified = true;

    LLVM_DEBUG(dbgs() << "GFX12Plus::createNewWaitcnt\n";
               if (It != Block.instr_end()) dbgs() << "Old Instr: " << *It;
               dbgs() << "New Instr: " << *SWaitInst << '\n');
  }

  if (Wait.hasWaitDepctr()) {
    assert(IsExpertMode);
    unsigned Enc =
        AMDGPU::DepCtr::encodeFieldVmVsrc(Wait.get(AMDGPU::VM_VSRC), ST);
    // Encode min(VA_VDST_RD, VA_VDST_WR) into the single hardware field
    unsigned VaVdst =
        std::min(Wait.get(AMDGPU::VA_VDST_RD), Wait.get(AMDGPU::VA_VDST_WR));
    Enc = AMDGPU::DepCtr::encodeFieldVaVdst(Enc, VaVdst);

    [[maybe_unused]] auto SWaitInst =
        BuildMI(Block, It, DL, TII.get(AMDGPU::S_WAITCNT_DEPCTR)).addImm(Enc);

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
  LLVM_DEBUG(dbgs() << "\n*** GenerateWaitcntInstBefore: "; MI.print(dbgs()););

  assert(!isNonWaitcntMetaInst(MI));

  AMDGPU::Waitcnt Wait;
  const unsigned Opc = MI.getOpcode();

  switch (Opc) {
  case AMDGPU::BUFFER_WBINVL1:
  case AMDGPU::BUFFER_WBINVL1_SC:
  case AMDGPU::BUFFER_WBINVL1_VOL:
  case AMDGPU::BUFFER_GL0_INV:
  case AMDGPU::BUFFER_GL1_INV: {
    // FIXME: This should have already been handled by the memory legalizer.
    // Removing this currently doesn't affect any lit tests, but we need to
    // verify that nothing was relying on this. The number of buffer invalidates
    // being handled here should not be expanded.
    Wait.set(AMDGPU::LOAD_CNT, 0);
    break;
  }
  case AMDGPU::SI_RETURN_TO_EPILOG:
  case AMDGPU::SI_RETURN:
  case AMDGPU::SI_WHOLE_WAVE_FUNC_RETURN:
  case AMDGPU::S_SETPC_B64_return: {
    // All waits must be resolved at call return.
    // NOTE: this could be improved with knowledge of all call sites or
    //   with knowledge of the called routines.
    ReturnInsts.insert(&MI);
    AMDGPU::Waitcnt AllZeroWait =
        WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/false);
    // On GFX12+, if LOAD_CNT is pending but no VGPRs are waiting for loads
    // (e.g., only GLOBAL_INV is pending), we can skip waiting on loadcnt.
    // GLOBAL_INV increments loadcnt but doesn't write to VGPRs, so there's
    // no need to wait for it at function boundaries.
    if (ST.hasExtendedWaitCounts() &&
        !ScoreBrackets.hasPendingEvent(HWEvents::VMEM_READ_ACCESS))
      AllZeroWait.set(AMDGPU::LOAD_CNT, ~0u);
    Wait = AllZeroWait;
    break;
  }
  case AMDGPU::S_ENDPGM:
  case AMDGPU::S_ENDPGM_SAVED: {
    // In dynamic VGPR mode, we want to release the VGPRs before the wave exits.
    // Technically the hardware will do this on its own if we don't, but that
    // might cost extra cycles compared to doing it explicitly.
    // When not in dynamic VGPR mode, identify S_ENDPGM instructions which may
    // have to wait for outstanding VMEM stores. In this case it can be useful
    // to send a message to explicitly release all VGPRs before the stores have
    // completed, but it is only safe to do this if there are no outstanding
    // scratch stores.
    EndPgmInsts[&MI] =
        !ScoreBrackets.empty(AMDGPU::STORE_CNT) &&
        !ScoreBrackets.hasPendingEvent(HWEvents::SCRATCH_WRITE_ACCESS);
    break;
  }
  case AMDGPU::S_SENDMSG:
  case AMDGPU::S_SENDMSGHALT: {
    if (ST.hasLegacyGeometry() &&
        ((MI.getOperand(0).getImm() & AMDGPU::SendMsg::ID_MASK_PreGFX11_) ==
         AMDGPU::SendMsg::ID_GS_DONE_PreGFX11)) {
      // Resolve vm waits before gs-done.
      Wait.set(AMDGPU::LOAD_CNT, 0);
      break;
    }
    [[fallthrough]];
  }
  default: {

    // Export & GDS instructions do not read the EXEC mask until after the
    // export is granted (which can occur well after the instruction is issued).
    // The shader program must flush all EXP operations on the export-count
    // before overwriting the EXEC mask.
    if (MI.modifiesRegister(AMDGPU::EXEC, &TRI)) {
      // Export and GDS are tracked individually, either may trigger a waitcnt
      // for EXEC.
      if (ScoreBrackets.hasPendingEvent(HWEvents::EXP_GPR_LOCK) ||
          ScoreBrackets.hasPendingEvent(HWEvents::EXP_PARAM_ACCESS) ||
          ScoreBrackets.hasPendingEvent(HWEvents::EXP_POS_ACCESS) ||
          ScoreBrackets.hasPendingEvent(HWEvents::GDS_GPR_LOCK)) {
        Wait.set(AMDGPU::EXP_CNT, 0);
      }
    }

    // Wait for any pending GDS instruction to complete before any
    // "Always GDS" instruction.
    if (TII.isAlwaysGDS(Opc) && ScoreBrackets.hasPendingGDS())
      Wait.add(AMDGPU::DS_CNT, ScoreBrackets.getPendingGDSWait());

    if (MI.isCall()) {
      // The function is going to insert a wait on everything in its prolog.
      // This still needs to be careful if the call target is a load (e.g. a GOT
      // load). We also need to check WAW dependency with saved PC.
      CallInsts.insert(&MI);
      Wait = AMDGPU::Waitcnt();

      const MachineOperand &CallAddrOp = TII.getCalleeOperand(MI);
      if (CallAddrOp.isReg()) {
        ScoreBrackets.determineWaitForPhysReg(
            SmemAccessCounter, CallAddrOp.getReg().asMCReg(), Wait, MI);

        if (const auto *RtnAddrOp =
                TII.getNamedOperand(MI, AMDGPU::OpName::dst)) {
          ScoreBrackets.determineWaitForPhysReg(
              SmemAccessCounter, RtnAddrOp->getReg().asMCReg(), Wait, MI);
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
            Wait.add(SmemAccessCounter, 0);
            if (PDT.dominates(MI.getParent(), It->second))
              SLoadAddresses.erase(It);
          }
        }
        unsigned AS = Memop->getAddrSpace();
        if (AS != AMDGPUAS::LOCAL_ADDRESS && AS != AMDGPUAS::FLAT_ADDRESS)
          continue;
        // No need to wait before load from VMEM to LDS.
        if (TII.mayWriteLDSThroughDMA(MI))
          continue;

        // LOAD_CNT is only relevant to vgpr or LDS.
        unsigned TID = AMDGPU::LDSDMA_BEGIN;
        if (Ptr && Memop->getAAInfo()) {
          const auto &LDSDMAStores = ScoreBrackets.getLDSDMAStores();
          for (unsigned I = 0, E = LDSDMAStores.size(); I != E; ++I) {
            if (MI.mayAlias(AA, *LDSDMAStores[I], true)) {
              if ((I + 1) >= AMDGPU::NUM_LDSDMA) {
                // We didn't have enough slot to track this LDS DMA store, it
                // has been tracked using the common RegNo (FIRST_LDS_VGPR).
                ScoreBrackets.determineWaitForLDSDMA(AMDGPU::LOAD_CNT, TID,
                                                     Wait);
                break;
              }

              ScoreBrackets.determineWaitForLDSDMA(AMDGPU::LOAD_CNT,
                                                   TID + I + 1, Wait);
            }
          }
        } else {
          ScoreBrackets.determineWaitForLDSDMA(AMDGPU::LOAD_CNT, TID, Wait);
        }
        if (Memop->isStore()) {
          ScoreBrackets.determineWaitForLDSDMA(AMDGPU::EXP_CNT, TID, Wait);
        }
      }

      // Loop over use and def operands.
      for (const MachineOperand &Op : MI.operands()) {
        if (!Op.isReg())
          continue;

        // If the instruction does not read tied source, skip the operand.
        if (Op.isTied() && Op.isUse() && TII.doesNotReadTiedSource(MI))
          continue;

        MCPhysReg Reg = Op.getReg().asMCReg();

        const bool IsVGPR = TRI.isVectorRegister(MRI, Op.getReg());
        if (IsVGPR) {
          // Implicit VGPR defs and uses are never a part of the memory
          // instructions description and usually present to account for
          // super-register liveness.
          // TODO: Most of the other instructions also have implicit uses
          // for the liveness accounting only.
          if (Op.isImplicit() && MI.mayLoadOrStore())
            continue;

          ScoreBrackets.determineWaitForPhysReg(AMDGPU::VA_VDST_WR, Reg, Wait,
                                                MI);
          if (Op.isDef()) {
            ScoreBrackets.determineWaitForPhysReg(AMDGPU::VA_VDST_RD, Reg, Wait,
                                                  MI);
            ScoreBrackets.determineWaitForPhysReg(AMDGPU::VM_VSRC, Reg, Wait,
                                                  MI);
          }

          // RAW always needs an s_waitcnt. WAW needs an s_waitcnt unless the
          // previous write and this write are the same type of VMEM
          // instruction, in which case they are (in some architectures)
          // guaranteed to write their results in order anyway.
          // Additionally check instructions where Point Sample Acceleration
          // might be applied.
          if (Op.isUse() || !SIInstrInfo::updateVMCntOnly(MI) ||
              ScoreBrackets.hasDifferentVGPRPendingEvents(
                  Reg, AMDGPU::getSimplifiedVMEMEventsFor(MI, TII)) ||
              ScoreBrackets.hasPointSamplePendingVmemTypes(MI, Reg) ||
              !ST.hasVmemWriteVgprInOrder()) {
            ScoreBrackets.determineWaitForPhysReg(AMDGPU::LOAD_CNT, Reg, Wait,
                                                  MI);
            ScoreBrackets.determineWaitForPhysReg(AMDGPU::SAMPLE_CNT, Reg, Wait,
                                                  MI);
            ScoreBrackets.determineWaitForPhysReg(AMDGPU::BVH_CNT, Reg, Wait,
                                                  MI);
            ScoreBrackets.clearVGPRPendingEvents(Reg);
          }

          if (Op.isDef() ||
              ScoreBrackets.hasPendingEvent(HWEvents::EXP_LDS_ACCESS)) {
            ScoreBrackets.determineWaitForPhysReg(AMDGPU::EXP_CNT, Reg, Wait,
                                                  MI);
          }
          ScoreBrackets.determineWaitForPhysReg(AMDGPU::DS_CNT, Reg, Wait, MI);
        } else if (Op.getReg() == AMDGPU::SCC) {
          ScoreBrackets.determineWaitForPhysReg(AMDGPU::KM_CNT, Reg, Wait, MI);
        } else {
          ScoreBrackets.determineWaitForPhysReg(SmemAccessCounter, Reg, Wait,
                                                MI);
        }

        if (ST.hasWaitXcnt() && Op.isDef())
          ScoreBrackets.determineWaitForPhysReg(AMDGPU::X_CNT, Reg, Wait, MI);
      }
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
  if (Opc == AMDGPU::S_BARRIER && !ST.hasAutoWaitcntBeforeBarrier() &&
      !ST.hasBackOffBarrier()) {
    Wait = Wait.combined(WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/true));
  }

  // TODO: Remove this work-around, enable the assert for Bug 457939
  //       after fixing the scheduler. Also, the Shader Compiler code is
  //       independent of target.
  if (SIInstrInfo::isCBranchVCCZRead(MI) && ST.hasReadVCCZBug() &&
      ScoreBrackets.hasPendingEvent(HWEvents::SMEM_ACCESS)) {
    Wait.set(AMDGPU::DS_CNT, 0);
  }

  // Verify that the wait is actually needed.
  ScoreBrackets.simplifyWaitcnt(Wait);

  // It is only necessary to insert an S_WAITCNT_DEPCTR instruction that
  // waits on VA_VDST if the instruction it would precede is not a VALU
  // instruction, since hardware handles VALU->VGPR->VALU hazards in
  // expert scheduling mode.
  if (TII.isVALU(MI, /*AllowLDSDMA=*/false)) {
    Wait.set(AMDGPU::VA_VDST_RD, ~0u);
    Wait.set(AMDGPU::VA_VDST_WR, ~0u);
  }

  // Since the translation for VMEM addresses occur in-order, we can apply the
  // XCnt if the current instruction is of VMEM type and has a memory
  // dependency with another VMEM instruction in flight.
  if (Wait.get(AMDGPU::X_CNT) != ~0u && isVmemAccess(MI)) {
    ScoreBrackets.applyWaitcnt(Wait, AMDGPU::X_CNT);
    Wait.set(AMDGPU::X_CNT, ~0u);
  }

  // When forcing emit, we need to skip terminators because that would break the
  // terminators of the MBB if we emit a waitcnt between terminators.
  if (ForceEmitZeroFlag && !MI.isTerminator())
    Wait = WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/false);

  // If we force waitcnt then update Wait accordingly.
  for (AMDGPU::InstCounterType T : AMDGPU::inst_counter_types()) {
    if (!ForceEmitWaitcnt[T])
      continue;
    Wait.set(T, 0);
  }

  if (FlushFlags.FlushVmCnt) {
    for (AMDGPU::InstCounterType T :
         {AMDGPU::LOAD_CNT, AMDGPU::SAMPLE_CNT, AMDGPU::BVH_CNT})
      Wait.set(T, 0);
  }

  if (FlushFlags.FlushDsCnt && ScoreBrackets.hasPendingEvent(AMDGPU::DS_CNT))
    Wait.set(AMDGPU::DS_CNT, 0);

  if (ForceEmitZeroLoadFlag && Wait.get(AMDGPU::LOAD_CNT) != ~0u)
    Wait.set(AMDGPU::LOAD_CNT, 0);

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
  if (Wait.get(AMDGPU::EXP_CNT) != ~0u && It != Block.instr_end() &&
      SIInstrInfo::isVINTERP(*It)) {
    MachineOperand *WaitExp = TII.getNamedOperand(*It, AMDGPU::OpName::waitexp);
    if (Wait.get(AMDGPU::EXP_CNT) < WaitExp->getImm()) {
      WaitExp->setImm(Wait.get(AMDGPU::EXP_CNT));
      Modified = true;
    }
    // Apply ExpCnt before resetting it, so applyWaitcnt below sees all counts.
    ScoreBrackets.applyWaitcnt(Wait, AMDGPU::EXP_CNT);
    Wait.set(AMDGPU::EXP_CNT, ~0u);

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

bool SIInsertWaitcnts::isVmemAccess(const MachineInstr &MI) const {
  return (TII.isFLAT(MI) && TII.mayAccessVMEMThroughFlat(MI)) ||
         (TII.isVMEM(MI) && !AMDGPU::getMUBUFIsBufferInv(MI.getOpcode()));
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

  if (ST.isPreciseMemoryEnabled() && Inst.mayLoadOrStore())
    Wait = WCG->getAllZeroWaitcnt(Inst.mayStore() &&
                                  !SIInstrInfo::isAtomicRet(Inst));

  if (TII.isAlwaysGDS(Inst.getOpcode())) {
    Wait.set(AMDGPU::DS_CNT, 0);
    NeedsEndPGMCheck = true;
  }

  ScoreBrackets.simplifyWaitcnt(Wait);

  auto SuccessorIt = std::next(Inst.getIterator());
  bool Result = generateWaitcnt(Wait, SuccessorIt, Block, ScoreBrackets,
                                /*OldWaitcntInstr=*/nullptr);

  if (Result && NeedsEndPGMCheck && isNextENDPGM(SuccessorIt, &Block)) {
    BuildMI(Block, SuccessorIt, Inst.getDebugLoc(), TII.get(AMDGPU::S_NOP))
        .addImm(0);
  }

  return Result;
}

void SIInsertWaitcnts::updateEventWaitcntAfter(MachineInstr &Inst,
                                               WaitcntBrackets *ScoreBrackets) {

  HWEvents InstEvents = AMDGPU::getEventsFor(Inst, ST, IsExpertMode, IsTgSplit);
  for (HWEvents E : InstEvents)
    ScoreBrackets->updateByEvent(E, Inst);

  if (TII.isDS(Inst) && TII.usesLGKM_CNT(Inst)) {
    if (TII.isAlwaysGDS(Inst.getOpcode()) ||
        TII.hasModifiersSet(Inst, AMDGPU::OpName::gds)) {
      ScoreBrackets->setPendingGDS();
    }
  } else if (TII.isFLAT(Inst)) {
    if (Inst.mayLoadOrStore() && TII.mayAccessVMEMThroughFlat(Inst) &&
        TII.mayAccessLDSThroughFlat(Inst, IsTgSplit) &&
        !SIInstrInfo::isLDSDMA(Inst)) {
      // Async/LDSDMA operations have FLAT encoding but do not actually use flat
      // pointers. They do have two operands that each access global and LDS,
      // thus making it appear at this point that they are using a flat pointer.
      // Filter them out, and for the rest, generate a dependency on flat
      // pointers so that both VM and LGKM counters are flushed.
      ScoreBrackets->setPendingFlat();
    }
  } else if (Inst.isCall()) {
    // Act as a wait on everything, but AsyncCnt and TensorCnt are never
    // included in such blanket waits.
    ScoreBrackets->applyWaitcnt(WCG->getAllZeroWaitcnt(/*IncludeVSCnt=*/false));
    ScoreBrackets->setStateOnFunctionEntryOrReturn();
  } else if (TII.isVINTERP(Inst)) {
    int64_t Imm = TII.getNamedOperand(Inst, AMDGPU::OpName::waitexp)->getImm();
    ScoreBrackets->applyWaitcnt(AMDGPU::EXP_CNT, Imm);
  }

  // Set XCNT to zero in the bracket for instructions that implicitly drain
  // XCNT.
  if (ST.hasWaitXcnt() && SIInstrInfo::isXcntDrain(Inst))
    ScoreBrackets->applyWaitcnt(AMDGPU::X_CNT, 0);
}

static bool isWaitInstr(MachineInstr &Inst) {
  unsigned Opcode = SIInstrInfo::getNonSoftWaitcntOpcode(Inst.getOpcode());
  return Opcode == AMDGPU::S_WAITCNT ||
         (Opcode == AMDGPU::S_WAITCNT_VSCNT && Inst.getOperand(0).isReg() &&
          Inst.getOperand(0).getReg() == AMDGPU::SGPR_NULL) ||
         Opcode == AMDGPU::S_WAIT_LOADCNT_DSCNT ||
         Opcode == AMDGPU::S_WAIT_STORECNT_DSCNT ||
         Opcode == AMDGPU::S_WAITCNT_lds_direct ||
         Opcode == AMDGPU::WAIT_ASYNCMARK ||
         AMDGPU::counterTypeForInstr(Opcode).has_value();
}

void SIInsertWaitcnts::setSchedulingMode(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator I,
                                         bool ExpertMode) const {
  const unsigned EncodedReg = AMDGPU::Hwreg::HwregEncoding::encode(
      AMDGPU::Hwreg::ID_SCHED_MODE, AMDGPU::Hwreg::HwregOffset::Default, 2);
  BuildMI(MBB, I, DebugLoc(), TII.get(AMDGPU::S_SETREG_IMM32_B32))
      .addImm(ExpertMode ? 2 : 0)
      .addImm(EncodedReg);
}

namespace {
// TODO: Remove this work-around after fixing the scheduler.
// There are two reasons why vccz might be incorrect; see ST.hasReadVCCZBug()
// and ST.partialVCCWritesUpdateVCCZ().
// i. VCCZBug: There is a hardware bug on CI/SI where SMRD instruction may
//    corrupt vccz bit, so when we detect that an instruction may read from
//    a corrupt vccz bit, we need to:
//   1. Insert s_waitcnt lgkm(0) to wait for all outstanding SMRD
//      operations to complete.
//   2. Recompute the correct value of vccz by writing the current value
//      of vcc back to vcc.
// ii. Partial writes to vcc don't update vccz, so we need to recompute the
//     correct value of vccz by reading vcc and writing it back to vcc.
//     No waitcnt is needed in this case.
class VCCZWorkaround {
  const WaitcntBrackets &ScoreBrackets;
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  bool VCCZCorruptionBug = false;
  bool VCCZNotUpdatedByPartialWrites = false;
  /// vccz could be incorrect at a basic block boundary if a predecessor wrote
  /// to vcc and then issued an smem load, so initialize to true.
  bool MustRecomputeVCCZ = true;

public:
  VCCZWorkaround(const WaitcntBrackets &ScoreBrackets, const GCNSubtarget &ST,
                 const SIInstrInfo &TII, const SIRegisterInfo &TRI)
      : ScoreBrackets(ScoreBrackets), ST(ST), TII(TII), TRI(TRI) {
    VCCZCorruptionBug = ST.hasReadVCCZBug();
    VCCZNotUpdatedByPartialWrites = !ST.partialVCCWritesUpdateVCCZ();
  }
  /// If \p MI reads vccz and we must recompute it based on MustRecomputeVCCZ,
  /// then emit a vccz recompute instruction before \p MI. This needs to be
  /// called on every instruction in the basic block because it also tracks the
  /// state and updates MustRecomputeVCCZ accordingly. Returns true if it
  /// modified the IR.
  bool tryRecomputeVCCZ(MachineInstr &MI) {
    // No need to run this if neither bug is present.
    if (!VCCZCorruptionBug && !VCCZNotUpdatedByPartialWrites)
      return false;

    // If MI is an SMEM and it can corrupt vccz on this target, then we need
    // both to emit a waitcnt and to recompute vccz.
    // But we don't actually emit a waitcnt here. This is done in
    // generateWaitcntInstBefore() because it tracks all the necessary waitcnt
    // state, and can either skip emitting a waitcnt if there is already one in
    // the IR, or emit an "optimized" combined waitcnt.
    // If this is an smem read, it could complete and clobber vccz at any time.
    MustRecomputeVCCZ |= VCCZCorruptionBug && TII.isSMRD(MI);

    // If the target partial vcc writes don't update vccz, and MI is such an
    // instruction then we must recompute vccz.
    // Note: We are using PartiallyWritesToVCCOpt optional to avoid calling
    // `definesRegister()` more than needed, because it's not very cheap.
    std::optional<bool> PartiallyWritesToVCCOpt;
    auto PartiallyWritesToVCC = [](MachineInstr &MI) {
      return MI.definesRegister(AMDGPU::VCC_LO, /*TRI=*/nullptr) ||
             MI.definesRegister(AMDGPU::VCC_HI, /*TRI=*/nullptr);
    };
    if (VCCZNotUpdatedByPartialWrites) {
      PartiallyWritesToVCCOpt = PartiallyWritesToVCC(MI);
      // If this is a partial VCC write but won't update vccz, then we must
      // recompute vccz.
      MustRecomputeVCCZ |= *PartiallyWritesToVCCOpt;
    }

    // If MI is a vcc write with no pending smem, or there is a pending smem
    // but the target does not suffer from the vccz corruption bug, then we
    // don't need to recompute vccz as this write will recompute it anyway.
    if (!ScoreBrackets.hasPendingEvent(HWEvents::SMEM_ACCESS) ||
        !VCCZCorruptionBug) {
      // Compute PartiallyWritesToVCCOpt if we haven't done so already.
      if (!PartiallyWritesToVCCOpt)
        PartiallyWritesToVCCOpt = PartiallyWritesToVCC(MI);
      bool FullyWritesToVCC = !*PartiallyWritesToVCCOpt &&
                              MI.definesRegister(AMDGPU::VCC, /*TRI=*/nullptr);
      // If we write to the full vcc or we write partially and the target
      // updates vccz on partial writes, then vccz will be updated correctly.
      bool UpdatesVCCZ = FullyWritesToVCC || (!VCCZNotUpdatedByPartialWrites &&
                                              *PartiallyWritesToVCCOpt);
      if (UpdatesVCCZ)
        MustRecomputeVCCZ = false;
    }

    // If MI is a branch that reads VCCZ then emit a waitcnt and a vccz
    // restore instruction if either is needed.
    if (SIInstrInfo::isCBranchVCCZRead(MI) && MustRecomputeVCCZ) {
      // Recompute the vccz bit. Any time a value is written to vcc, the vccz
      // bit is updated, so we can restore the bit by reading the value of vcc
      // and then writing it back to the register.
      BuildMI(*MI.getParent(), MI, MI.getDebugLoc(),
              TII.get(ST.isWave32() ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64),
              TRI.getVCC())
          .addReg(TRI.getVCC());
      MustRecomputeVCCZ = false;
      return true;
    }
    return false;
  }
};

} // namespace

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
  VCCZWorkaround VCCZW(ScoreBrackets, ST, TII, TRI);

  // Walk over the instructions.
  MachineInstr *OldWaitcntInstr = nullptr;

  // NOTE: We may append instrs after Inst while iterating.
  ScoreBrackets.verify();
  for (MachineBasicBlock::instr_iterator Iter = Block.instr_begin(),
                                         E = Block.instr_end();
       Iter != E; ++Iter) {
    MachineInstr &Inst = *Iter;
    if (isNonWaitcntMetaInst(Inst))
      continue;
    // Track pre-existing waitcnts that were added in earlier iterations or by
    // the memory legalizer.
    if (isWaitInstr(Inst) ||
        (IsExpertMode && Inst.getOpcode() == AMDGPU::S_WAITCNT_DEPCTR)) {
      if (!OldWaitcntInstr)
        OldWaitcntInstr = &Inst;
      continue;
    }

    PreheaderFlushFlags FlushFlags;
    if (Block.getFirstTerminator() == Inst)
      FlushFlags = isPreheaderToFlush(Block, ScoreBrackets);

    // Generate an s_waitcnt instruction to be placed before Inst, if needed.
    Modified |= generateWaitcntInstBefore(Inst, ScoreBrackets, OldWaitcntInstr,
                                          FlushFlags);
    OldWaitcntInstr = nullptr;

    if (Inst.getOpcode() == AMDGPU::ASYNCMARK) {
      // Asyncmarks record the current wait state and so should not allow
      // waitcnts that occur after them to be merged into waitcnts that occur
      // before.
      ScoreBrackets.recordAsyncMark(Inst);
      continue;
    }

    if (TII.isSMRD(Inst)) {
      for (const MachineMemOperand *Memop : Inst.memoperands()) {
        // No need to handle invariant loads when avoiding WAR conflicts, as
        // there cannot be a vector store to the same memory location.
        if (!Memop->isInvariant()) {
          const Value *Ptr = Memop->getValue();
          SLoadAddresses.insert(std::pair(Ptr, Inst.getParent()));
        }
      }
    }

    updateEventWaitcntAfter(Inst, &ScoreBrackets);

    // Note: insertForcedWaitAfter() may add instrs after Iter that need to be
    // visited by the loop.
    Modified |= insertForcedWaitAfter(Inst, Block, ScoreBrackets);

    LLVM_DEBUG({
      Inst.print(dbgs());
      ScoreBrackets.dump();
    });

    // If the target suffers from the vccz bugs, this may emit the necessary
    // vccz recompute instruction before \p Inst if needed.
    Modified |= VCCZW.tryRecomputeVCCZ(Inst);

    ScoreBrackets.verify();
  }

  // Flush counters at the end of the block if needed (for preheaders with no
  // terminator).
  AMDGPU::Waitcnt Wait;
  if (Block.getFirstTerminator() == Block.end()) {
    PreheaderFlushFlags FlushFlags = isPreheaderToFlush(Block, ScoreBrackets);
    if (FlushFlags.FlushVmCnt) {
      if (ScoreBrackets.hasPendingEvent(AMDGPU::LOAD_CNT))
        Wait.set(AMDGPU::LOAD_CNT, 0);
      if (ScoreBrackets.hasPendingEvent(AMDGPU::SAMPLE_CNT))
        Wait.set(AMDGPU::SAMPLE_CNT, 0);
      if (ScoreBrackets.hasPendingEvent(AMDGPU::BVH_CNT))
        Wait.set(AMDGPU::BVH_CNT, 0);
    }
    if (FlushFlags.FlushDsCnt && ScoreBrackets.hasPendingEvent(AMDGPU::DS_CNT))
      Wait.set(AMDGPU::DS_CNT, 0);
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

bool SIInsertWaitcnts::removeRedundantSoftXcnts(MachineBasicBlock &Block) {
  if (Block.size() <= 1)
    return false;
  // The Memory Legalizer conservatively inserts a soft xcnt before each
  // atomic RMW operation. However, for sequences of back-to-back atomic
  // RMWs, only the first s_wait_xcnt insertion is necessary. Optimize away
  // the redundant soft xcnts.
  bool Modified = false;
  // Remember the last atomic with a soft xcnt right before it.
  MachineInstr *LastAtomicWithSoftXcnt = nullptr;

  for (MachineInstr &MI : drop_begin(Block)) {
    // Ignore last atomic if non-LDS VMEM and SMEM.
    bool IsLDS = TII.isDS(MI) ||
                 (TII.isFLAT(MI) && TII.mayAccessLDSThroughFlat(MI, IsTgSplit));
    if (!IsLDS && (MI.mayLoad() ^ MI.mayStore()))
      LastAtomicWithSoftXcnt = nullptr;

    bool IsAtomicRMW =
        SIInstrFlags::isMaybeAtomic(MI) && MI.mayLoad() && MI.mayStore();
    MachineInstr &PrevMI = *MI.getPrevNode();
    // This is an atomic with a soft xcnt.
    if (PrevMI.getOpcode() == AMDGPU::S_WAIT_XCNT_soft && IsAtomicRMW) {
      // If we have already found an atomic with a soft xcnt, remove this soft
      // xcnt as it's redundant.
      if (LastAtomicWithSoftXcnt) {
        PrevMI.eraseFromParent();
        Modified = true;
      }
      LastAtomicWithSoftXcnt = &MI;
    }
  }
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

  MachineLoop *Loop = MLI.getLoopFor(Succ);
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
    return TII.mayAccessVMEMThroughFlat(MI);
  return SIInstrInfo::isVMEM(MI);
}

bool SIInsertWaitcnts::isDSRead(const MachineInstr &MI) const {
  return SIInstrInfo::isDS(MI) && MI.mayLoad() && !MI.mayStore();
}

// Check if instruction is a store to LDS that is counted via DSCNT
// (where that counter exists).
bool SIInsertWaitcnts::mayStoreIncrementingDSCNT(const MachineInstr &MI) const {
  return MI.mayStore() && SIInstrInfo::isDS(MI);
}

// Return flags indicating which counters should be flushed in the preheader of
// the given loop. We currently decide to flush in the following situations:
// For VMEM (FlushVmCnt):
// 1. The loop contains vmem store(s), no vmem load and at least one use of a
//    vgpr containing a value that is loaded outside of the loop. (Only on
//    targets with no vscnt counter).
// 2. The loop contains vmem load(s), but the loaded values are not used in the
//    loop, and at least one use of a vgpr containing a value that is loaded
//    outside of the loop.
// For DS (FlushDsCnt, GFX12+ only):
// 3. The loop contains no DS reads, and at least one use of a vgpr containing
//    a value that is DS read outside of the loop.
// 4. The loop contains DS read(s), loaded values are not used in the same
//    iteration but in the next iteration (prefetch pattern), and at least one
//    use of a vgpr containing a value that is DS read outside of the loop.
//    Flushing in preheader reduces wait overhead if the wait requirement in
//    iteration 1 would otherwise be more strict (but unfortunately preheader
//    flush decision is taken before knowing that).
// 5. (Single-block loops only) The loop has DS prefetch reads with flush point
//    tracking. Some DS reads may be used in the same iteration (creating
//    "flush points"), but others remain unflushed at the backedge. When a DS
//    read is consumed in the same iteration, it and all prior reads are
//    "flushed" (FIFO order). No DS writes are allowed in the loop.
//    TODO: Find a way to extend to multi-block loops.
PreheaderFlushFlags
SIInsertWaitcnts::getPreheaderFlushFlags(MachineLoop *ML,
                                         const WaitcntBrackets &Brackets) {
  PreheaderFlushFlags Flags;
  bool HasVMemLoad = false;
  bool HasVMemStore = false;
  bool UsesVgprVMEMLoadedOutside = false;
  bool UsesVgprDSReadOutside = false;
  bool VMemInvalidated = false;
  // DS optimization only applies to GFX12+ where DS_CNT is separate.
  // Tracking status for "no DS read in loop" or "pure DS prefetch
  // (use only in next iteration)".
  bool TrackSimpleDSOpt = ST.hasExtendedWaitCounts();
  DenseSet<MCRegUnit> VgprUse;
  DenseSet<MCRegUnit> VgprDefVMEM;
  DenseSet<MCRegUnit> VgprDefDS;

  // Track DS reads for prefetch pattern with flush points (single-block only).
  // Keeps track of the last DS read (position counted from the top of the loop)
  // to each VGPR. Read is considered consumed (and thus needs flushing) if
  // the dest register has a use or is overwritten (by any later opertions).
  DenseMap<MCRegUnit, unsigned> LastDSReadPositionMap;
  unsigned DSReadPosition = 0;
  bool IsSingleBlock = ML->getNumBlocks() == 1;
  bool TrackDSFlushPoint = ST.hasExtendedWaitCounts() && IsSingleBlock;
  unsigned LastDSFlushPosition = 0;

  for (MachineBasicBlock *MBB : ML->blocks()) {
    for (MachineInstr &MI : *MBB) {
      if (isVMEMOrFlatVMEM(MI)) {
        HasVMemLoad |= MI.mayLoad();
        HasVMemStore |= MI.mayStore();
      }
      // TODO: Can we relax DSStore check? There may be cases where
      // these DS stores are drained prior to the end of MBB (or loop).
      if (mayStoreIncrementingDSCNT(MI)) {
        // Early exit if none of the optimizations are feasible.
        // Otherwise, set tracking status appropriately and continue.
        if (VMemInvalidated)
          return Flags;
        TrackSimpleDSOpt = false;
        TrackDSFlushPoint = false;
      }
      bool IsDSRead = isDSRead(MI);
      if (IsDSRead)
        ++DSReadPosition;

      // Helper: if RU has a pending DS read, update LastDSFlushPosition
      auto updateDSReadFlushTracking = [&](MCRegUnit RU) {
        if (!TrackDSFlushPoint)
          return;
        if (auto It = LastDSReadPositionMap.find(RU);
            It != LastDSReadPositionMap.end()) {
          // RU defined by DSRead is used or overwritten. Need to complete
          // the read, if not already implied by a later DSRead (to any RU)
          // needing to complete in FIFO order.
          LastDSFlushPosition = std::max(LastDSFlushPosition, It->second);
        }
      };

      for (const MachineOperand &Op : MI.all_uses()) {
        if (Op.isDebug() || !TRI.isVectorRegister(MRI, Op.getReg()))
          continue;
        // Vgpr use
        for (MCRegUnit RU : TRI.regunits(Op.getReg().asMCReg())) {
          // If we find a register that is loaded inside the loop, 1. and 2.
          // are invalidated.
          if (VgprDefVMEM.contains(RU))
            VMemInvalidated = true;

          // Check for DS reads used inside the loop
          if (VgprDefDS.contains(RU))
            TrackSimpleDSOpt = false;

          // Early exit if all optimizations are invalidated
          if (VMemInvalidated && !TrackSimpleDSOpt && !TrackDSFlushPoint)
            return Flags;

          // Check for flush points (DS read used in same iteration)
          updateDSReadFlushTracking(RU);

          VgprUse.insert(RU);
          // Check if this register has a pending VMEM load from outside the
          // loop (value loaded outside and used inside).
          AMDGPU::VMEMID ID = AMDGPU::toVMEMID(RU);
          if (Brackets.hasPendingVMEM(ID, AMDGPU::LOAD_CNT) ||
              Brackets.hasPendingVMEM(ID, AMDGPU::SAMPLE_CNT) ||
              Brackets.hasPendingVMEM(ID, AMDGPU::BVH_CNT))
            UsesVgprVMEMLoadedOutside = true;
          // Check if loaded outside the loop via DS (not VMEM/FLAT).
          // Only consider it a DS read if there's no pending VMEM load for
          // this register, since FLAT can set both counters.
          else if (Brackets.hasPendingVMEM(ID, AMDGPU::DS_CNT))
            UsesVgprDSReadOutside = true;
        }
      }

      // VMem load vgpr def
      if (isVMEMOrFlatVMEM(MI) && MI.mayLoad()) {
        for (const MachineOperand &Op : MI.all_defs()) {
          for (MCRegUnit RU : TRI.regunits(Op.getReg().asMCReg())) {
            // If we find a register that is loaded inside the loop, 1. and 2.
            // are invalidated.
            if (VgprUse.contains(RU))
              VMemInvalidated = true;
            VgprDefVMEM.insert(RU);
          }
        }
        // Early exit if all optimizations are invalidated
        if (VMemInvalidated && !TrackSimpleDSOpt && !TrackDSFlushPoint)
          return Flags;
      }

      // DS read vgpr def
      // Note: Unlike VMEM, we DON'T invalidate when VgprUse.contains(RegNo).
      // If USE comes before DEF, it's the prefetch pattern (use value from
      // previous iteration, read for next iteration). We should still flush
      // in preheader so iteration 1 doesn't need to wait inside the loop.
      // Only invalidate when DEF comes before USE (same-iteration consumption,
      // checked above when processing uses).
      if (IsDSRead || TrackDSFlushPoint) {
        for (const MachineOperand &Op : MI.all_defs()) {
          if (!TRI.isVectorRegister(MRI, Op.getReg()))
            continue;
          for (MCRegUnit RU : TRI.regunits(Op.getReg().asMCReg())) {
            // Check for overwrite of pending DS read (flush point) by any
            // instruction
            updateDSReadFlushTracking(RU);
            if (IsDSRead) {
              VgprDefDS.insert(RU);
              if (TrackDSFlushPoint)
                LastDSReadPositionMap[RU] = DSReadPosition;
            }
          }
        }
      }
    }
  }

  // VMEM flush decision
  if (!VMemInvalidated && UsesVgprVMEMLoadedOutside &&
      ((!ST.hasVscnt() && HasVMemStore && !HasVMemLoad) ||
       (HasVMemLoad && ST.hasVmemWriteVgprInOrder())))
    Flags.FlushVmCnt = true;

  // DS flush decision:
  // Simple DS Opt: flush if loop uses DS read values from outside
  // and either has no DS reads in the loop, or DS reads whose results
  // are not used in the loop.
  bool SimpleDSOpt = TrackSimpleDSOpt && UsesVgprDSReadOutside;
  // Prefetch with flush points: some DS reads used in same iteration,
  // but unflushed reads remain at backedge
  bool HasUnflushedDSReads = DSReadPosition > LastDSFlushPosition;
  bool DSFlushPointPrefetch =
      TrackDSFlushPoint && UsesVgprDSReadOutside && HasUnflushedDSReads;

  if (SimpleDSOpt || DSFlushPointPrefetch)
    Flags.FlushDsCnt = true;

  return Flags;
}

bool SIInsertWaitcntsLegacy::runOnMachineFunction(MachineFunction &MF) {
  auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  auto &PDT =
      getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
  AliasAnalysis *AA = nullptr;
  if (auto *AAR = getAnalysisIfAvailable<AAResultsWrapperPass>())
    AA = &AAR->getAAResults();

  return SIInsertWaitcnts(MLI, PDT, AA, MF).run();
}

PreservedAnalyses
SIInsertWaitcntsPass::run(MachineFunction &MF,
                          MachineFunctionAnalysisManager &MFAM) {
  auto &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  auto &PDT = MFAM.getResult<MachinePostDominatorTreeAnalysis>(MF);
  auto *AA = MFAM.getResult<FunctionAnalysisManagerMachineFunctionProxy>(MF)
                 .getManager()
                 .getCachedResult<AAManager>(MF.getFunction());

  if (!SIInsertWaitcnts(MLI, PDT, AA, MF).run())
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses()
      .preserveSet<CFGAnalyses>()
      .preserve<AAManager>();
}

bool SIInsertWaitcnts::run() {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  AMDGPU::IsaVersion IV = AMDGPU::getIsaVersion(ST.getCPU());

  // Initialize hardware limits first, as they're needed by the generators.
  Limits = AMDGPU::HardwareLimits(IV);

  if (ST.hasExtendedWaitCounts()) {
    IsExpertMode = ST.hasExpertSchedulingMode() &&
                   (ExpertSchedulingModeFlag.getNumOccurrences()
                        ? ExpertSchedulingModeFlag
                        : MF.getFunction()
                              .getFnAttribute("amdgpu-expert-scheduling-mode")
                              .getValueAsBool());
    MaxCounter = IsExpertMode ? AMDGPU::NUM_EXPERT_INST_CNTS
                              : AMDGPU::NUM_EXTENDED_INST_CNTS;
    // Initialize WCG per MF. It contains state that depends on MF attributes.
    WCG = std::make_unique<WaitcntGeneratorGFX12Plus>(MF, MaxCounter, Limits,
                                                      IsExpertMode);
  } else {
    MaxCounter = AMDGPU::NUM_NORMAL_INST_CNTS;
    // Initialize WCG per MF. It contains state that depends on MF attributes.
    WCG = std::make_unique<WaitcntGeneratorPreGFX12>(
        MF, AMDGPU::NUM_NORMAL_INST_CNTS, Limits);
  }

  SmemAccessCounter = WCG->getCounterFromEvent(HWEvents::SMEM_ACCESS);

  bool Modified = false;
  MachineBasicBlock &EntryBB = MF.front();

  const auto GetWaitEvents = [&](AMDGPU::InstCounterType T) {
    return WCG->getWaitEvents(T);
  };
  const auto GetCounterFromEvent = [&](HWEvents E) {
    return WCG->getCounterFromEvent(E);
  };
  AMDGPU::WaitcntBracketsContext WBC(ST, MRI, TII, TRI, IsTgSplit, IsExpertMode,
                                     MaxCounter, Limits, GetWaitEvents,
                                     GetCounterFromEvent);

  if (!MFI->isEntryFunction() &&
      !MF.getFunction().hasFnAttribute(Attribute::Naked)) {
    // Wait for any outstanding memory operations that the input registers may
    // depend on. We can't track them and it's better to do the wait after the
    // costly call sequence.

    // TODO: Could insert earlier and schedule more liberally with operations
    // that only use caller preserved registers.
    MachineBasicBlock::iterator I = EntryBB.begin();
    while (I != EntryBB.end() && I->isMetaInstruction())
      ++I;

    if (ST.hasExtendedWaitCounts()) {
      BuildMI(EntryBB, I, DebugLoc(), TII.get(AMDGPU::S_WAIT_LOADCNT_DSCNT))
          .addImm(0);
      for (auto CT : inst_counter_types(AMDGPU::NUM_EXTENDED_INST_CNTS)) {
        if (CT == AMDGPU::LOAD_CNT || CT == AMDGPU::DS_CNT ||
            CT == AMDGPU::STORE_CNT || CT == AMDGPU::X_CNT ||
            CT == AMDGPU::ASYNC_CNT || CT == AMDGPU::TENSOR_CNT)
          continue;

        if (!ST.hasImageInsts() &&
            (CT == AMDGPU::EXP_CNT || CT == AMDGPU::SAMPLE_CNT ||
             CT == AMDGPU::BVH_CNT))
          continue;

        BuildMI(EntryBB, I, DebugLoc(),
                TII.get(instrsForExtendedCounterTypes[CT]))
            .addImm(0);
      }
      if (IsExpertMode) {
        unsigned Enc = AMDGPU::DepCtr::encodeFieldVaVdst(0, ST);
        Enc = AMDGPU::DepCtr::encodeFieldVmVsrc(Enc, 0);
        BuildMI(EntryBB, I, DebugLoc(), TII.get(AMDGPU::S_WAITCNT_DEPCTR))
            .addImm(Enc);
      }
    } else {
      BuildMI(EntryBB, I, DebugLoc(), TII.get(AMDGPU::S_WAITCNT)).addImm(0);
    }

    auto NonKernelInitialState = std::make_unique<WaitcntBrackets>(WBC);
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
          Brackets = std::make_unique<WaitcntBrackets>(WBC);
        } else {
          // Reinitialize in-place. N.B. do not do this by assigning from a
          // temporary because the WaitcntBrackets class is large and it could
          // cause this function to use an unreasonable amount of stack space.
          Brackets->~WaitcntBrackets();
          new (Brackets.get()) WaitcntBrackets(WBC);
        }
      }

      if (ST.hasWaitXcnt())
        Modified |= removeRedundantSoftXcnts(*MBB);
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
              LLVM_DEBUG(dbgs() << "Repeat on backedge without merge\n");
              Repeat = true;
            }
            if (!MoveBracketsToSucc) {
              MoveBracketsToSucc = &SuccBI;
            } else {
              SuccBI.Incoming = std::make_unique<WaitcntBrackets>(*Brackets);
            }
          } else {
            LLVM_DEBUG({
              dbgs() << "Try to merge ";
              MBB->printName(dbgs());
              dbgs() << " into ";
              Succ->printName(dbgs());
              dbgs() << '\n';
            });
            if (SuccBI.Incoming->merge(*Brackets)) {
              SuccBI.Dirty = true;
              if (SuccBII <= BII) {
                LLVM_DEBUG(dbgs() << "Repeat on backedge with merge\n");
                Repeat = true;
              }
            }
          }
        }
        if (MoveBracketsToSucc)
          MoveBracketsToSucc->Incoming = std::move(Brackets);
      }
    }
  } while (Repeat);

  if (ST.hasScalarStores()) {
    SmallVector<MachineBasicBlock *, 4> EndPgmBlocks;
    bool HaveScalarStores = false;

    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (!HaveScalarStores && TII.isScalarStore(MI))
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
          else if (TII.isScalarStore(*I))
            SeenDCacheWB = false;

          // FIXME: It would be better to insert this before a waitcnt if any.
          if ((I->getOpcode() == AMDGPU::S_ENDPGM ||
               I->getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG) &&
              !SeenDCacheWB) {
            Modified = true;
            BuildMI(*MBB, I, I->getDebugLoc(), TII.get(AMDGPU::S_DCACHE_WB));
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
  if (!WCG->isOptNone() && MFI->isDynamicVGPREnabled()) {
    for (auto [MI, _] : EndPgmInsts) {
      BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
              TII.get(AMDGPU::S_ALLOC_VGPR))
          .addImm(0);
      Modified = true;
    }
  } else if (!WCG->isOptNone() &&
             ST.getGeneration() >= AMDGPUSubtarget::GFX11 &&
             (MF.getFrameInfo().hasCalls() ||
              ST.getOccupancyWithNumVGPRs(
                  TRI.getNumUsedPhysRegs(MRI, AMDGPU::VGPR_32RegClass),
                  /*IsDynamicVGPR=*/false) <
                  AMDGPU::IsaInfo::getMaxWavesPerEU(ST))) {
    for (auto [MI, Flag] : EndPgmInsts) {
      if (Flag) {
        if (ST.requiresNopBeforeDeallocVGPRs()) {
          BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
                  TII.get(AMDGPU::S_NOP))
              .addImm(0);
        }
        BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
                TII.get(AMDGPU::S_SENDMSG))
            .addImm(AMDGPU::SendMsg::ID_DEALLOC_VGPRS_GFX11Plus);
        Modified = true;
      }
    }
  }

  if (MFI->isEntryFunction() && ST.hasRequiresInitialUnclausedVmem()) {
    // Hardware entrypoints must begin with a specific sequence:
    //   GLOBAL_PREFETCH_B8 V0, S[0:1] SCOPE:SCOPE_SE
    //   V_NOP
    MachineBasicBlock::iterator I = EntryBB.begin();
    BuildMI(EntryBB, I, DebugLoc(), TII.get(AMDGPU::GLOBAL_PREFETCH_B8_SADDR))
        .addReg(AMDGPU::SGPR0_SGPR1, RegState::Undef)
        .addReg(AMDGPU::VGPR0, RegState::Undef)
        .addImm(0)
        .addImm(AMDGPU::CPol::SCOPE_SE | AMDGPU::CPol::TH_RT);
    BuildMI(EntryBB, I, DebugLoc(), TII.get(AMDGPU::V_NOP_e32));
    Modified = true;
  }

  return Modified;
}
