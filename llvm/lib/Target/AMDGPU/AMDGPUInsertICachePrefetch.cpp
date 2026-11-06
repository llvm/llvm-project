//===- AMDGPUInsertICachePrefetch.cpp - Insert ICache prefetches ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Insert instruction-cache prefetches for large AMDHSA entry functions.
//
// The prefetch instruction is fire-and-forget: it updates no wave wait counter
// and returns neither a result nor an error. It is therefore safe to insert
// after the wait-counter and hazard passes.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "SIProgramInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-insert-icache-prefetch"

static cl::opt<unsigned> ICachePrefetchInitialSize(
    "amdgpu-icache-prefetch-initial-size",
    cl::desc("Override the initial instruction prefetch size in bytes"),
    cl::init(0), cl::Hidden);

static cl::opt<unsigned> ICachePrefetchThreshold(
    "amdgpu-icache-prefetch-threshold",
    cl::desc("Override the explicit instruction prefetch threshold in bytes"),
    cl::init(0), cl::Hidden);

namespace {

struct ICachePrefetchInfo {
  unsigned DescriptorPrefetchLines;
  unsigned MaxNumPrefetchInsts;
  uint64_t InitialSize;
  uint64_t ProgramSize;
};

// Each prefetch can transfer up to 32 cache lines of 128 bytes.
constexpr unsigned CacheLinesPerPrefetch = 32;

static std::optional<ICachePrefetchInfo>
getICachePrefetchInfo(MachineFunction &MF);

class AMDGPUInsertICachePrefetch {
  MachineLoopInfo *MLI;
  MachinePostDominatorTree *PDT;

  bool isLoopFreeEntryPostDominator(const MachineBasicBlock &MBB,
                                    const MachineBasicBlock &EntryBB) const {
    return !MLI->getLoopFor(&MBB) && PDT->dominates(&MBB, &EntryBB);
  }

public:
  // These analyses describe the final machine CFG at this late insertion
  // point. CFG-based placement will use them to select loop-free blocks on
  // the entry block's post-dominator chain.
  AMDGPUInsertICachePrefetch(MachineLoopInfo *MLI,
                             MachinePostDominatorTree *PDT)
      : MLI(MLI), PDT(PDT) {}

  bool run(MachineFunction &MF, const ICachePrefetchInfo &Info);
};

class AMDGPUInsertICachePrefetchLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUInsertICachePrefetchLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addUsedIfAvailable<MachineLoopInfoWrapperPass>();
    AU.addUsedIfAvailable<MachinePostDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    std::optional<ICachePrefetchInfo> Info = getICachePrefetchInfo(MF);
    if (!Info)
      return false;

    if (MF.size() == 1)
      return AMDGPUInsertICachePrefetch(nullptr, nullptr).run(MF, *Info);

    // Try to get existing machine loop info or calculate locally if not
    // available.
    auto *MLIWrapper = getAnalysisIfAvailable<MachineLoopInfoWrapperPass>();
    MachineDominatorTree LocalMDT;
    MachineLoopInfo LocalMLI;
    if (!MLIWrapper) {
      LocalMDT.recalculate(MF);
      LocalMLI.calculate(LocalMDT);
    }
    MachineLoopInfo &MLI = MLIWrapper ? MLIWrapper->getLI() : LocalMLI;
    // Try to get existing post-dominator tree or calculate locally if not
    // available.
    auto *PDTWrapper =
        getAnalysisIfAvailable<MachinePostDominatorTreeWrapperPass>();
    MachinePostDominatorTree LocalPDT;
    if (!PDTWrapper)
      LocalPDT.recalculate(MF);
    MachinePostDominatorTree &PDT =
        PDTWrapper ? PDTWrapper->getPostDomTree() : LocalPDT;
    return AMDGPUInsertICachePrefetch(&MLI, &PDT).run(MF, *Info);
  }
};

} // end anonymous namespace

namespace {

struct ICachePrefetchConfig {
  uint64_t InitialSize;
  uint64_t Threshold;
};

static ICachePrefetchConfig getICachePrefetchConfig(const GCNSubtarget &ST) {
  assert(ST.hasInstPrefSize());

  uint32_t Mask, Shift, Width, CacheLineSize;
  ST.getInstPrefSizeArgs(Mask, Shift, Width, CacheLineSize);
  uint64_t DescriptorPrefetchCapacity =
      ((uint64_t{1} << Width) - 1) * CacheLineSize;
  uint64_t InitialSize = ICachePrefetchInitialSize.getNumOccurrences()
                             ? ICachePrefetchInitialSize
                             : ST.getInitialInstPrefSize();
  uint64_t Threshold = ICachePrefetchThreshold.getNumOccurrences()
                           ? ICachePrefetchThreshold
                           : DescriptorPrefetchCapacity;

  if (InitialSize == 0 || InitialSize % CacheLineSize != 0 ||
      InitialSize > DescriptorPrefetchCapacity)
    reportFatalUsageError(
        Twine("-amdgpu-icache-prefetch-initial-size must be a non-zero "
              "multiple of ") +
        Twine(CacheLineSize) + " bytes not exceeding " +
        Twine(DescriptorPrefetchCapacity) + " bytes");

  uint64_t ICacheSize = ST.getInstCacheSize();
  if (Threshold == 0 || Threshold % CacheLineSize != 0 ||
      Threshold > ICacheSize)
    reportFatalUsageError(
        Twine("-amdgpu-icache-prefetch-threshold must be a non-zero multiple "
              "of ") +
        Twine(CacheLineSize) + " bytes not exceeding " + Twine(ICacheSize) +
        " bytes");

  if (InitialSize > Threshold)
    reportFatalUsageError(
        Twine("-amdgpu-icache-prefetch-initial-size must not exceed "
              "-amdgpu-icache-prefetch-threshold"));

  return {InitialSize, Threshold};
}

static std::optional<ICachePrefetchInfo>
getICachePrefetchInfo(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasSmemPrefetchInsts() || !ST.hasInstPrefSize())
    return std::nullopt;

  // Kernel descriptors are emitted for AMDHSA entry functions.
  if (ST.getTargetTriple().getOS() != Triple::AMDHSA ||
      !MF.getInfo<SIMachineFunctionInfo>()->isEntryFunction())
    return std::nullopt;

  // Basic block sections may be placed independently by the linker.
  if (MF.hasBBSections() ||
      MF.getTarget().getBBSectionsType() != BasicBlockSection::None)
    return std::nullopt;

  uint64_t ICacheSize = ST.getInstCacheSize();
  if (ICacheSize == 0)
    return std::nullopt;
  const ICachePrefetchConfig Config = getICachePrefetchConfig(ST);

  SIProgramInfo PI;
  uint64_t ProgramSize = PI.getFunctionCodeSize(MF);
  if (ProgramSize <= Config.Threshold)
    return std::nullopt;

  unsigned CacheLineSize = ST.getInstCacheLineSize();
  unsigned ICacheLines = ICacheSize / CacheLineSize;
  unsigned DescriptorPrefetchLines = Config.InitialSize / CacheLineSize;
  unsigned MaxNumPrefetchInsts = llvm::divideCeil(
      ICacheLines - DescriptorPrefetchLines, CacheLinesPerPrefetch);
  if (MaxNumPrefetchInsts == 0)
    return std::nullopt;

  return ICachePrefetchInfo{DescriptorPrefetchLines, MaxNumPrefetchInsts,
                            Config.InitialSize, ProgramSize};
}

} // end anonymous namespace

static MachineBasicBlock::iterator findMBBInsertionPoint(MachineBasicBlock &MBB,
                                                         const GCNSubtarget &ST,
                                                         bool IsEntryBlock) {
  MachineBasicBlock::iterator InsertPt = MBB.begin();

  // Skip past any instructions that must remain at the very beginning:
  // - Debug values and CFI instructions
  // - In the entry block only, the gfx1250 initial unclaused-VMEM workaround
  //   and S_SETREG_IMM32_B32 instructions that set up MODE register bits
  //   (e.g., REPLAY_MODE bit 25 from SIFrameLowering)
  // In the entry block, we want the prefetches to come after all initial MODE
  // setup.
  bool SkippedInitialUnclausedVmemPrologue =
      !IsEntryBlock || !ST.hasRequiresInitialUnclausedVmem();
  while (InsertPt != MBB.end()) {
    if (InsertPt->isDebugValue() || InsertPt->isCFIInstruction() ||
        (IsEntryBlock && InsertPt->getOpcode() == AMDGPU::S_SETREG_IMM32_B32)) {
      ++InsertPt;
      continue;
    }
    if (!SkippedInitialUnclausedVmemPrologue &&
        InsertPt->getOpcode() == AMDGPU::S_MOV_B64) {
      auto Vnop = std::next(InsertPt);
      auto GlobalPrefetch = Vnop == MBB.end() ? MBB.end() : std::next(Vnop);
      if (Vnop != MBB.end() && Vnop->getOpcode() == AMDGPU::V_NOP_e32 &&
          GlobalPrefetch != MBB.end() &&
          GlobalPrefetch->getOpcode() == AMDGPU::GLOBAL_PREFETCH_B8_SADDR) {
        InsertPt = std::next(GlobalPrefetch);
        SkippedInitialUnclausedVmemPrologue = true;
        continue;
      }
    }
    break;
  }
  return InsertPt;
}

bool AMDGPUInsertICachePrefetch::run(MachineFunction &MF,
                                     const ICachePrefetchInfo &Info) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  const SIInstrInfo *TII = ST.getInstrInfo();
  MachineBasicBlock &EntryBB = MF.front();

  // Walk the post-dominator chain to get candidates in execution order. This
  // is distinct from the layout order used below to calculate code offsets.
  SmallVector<MachineBasicBlock *> Candidates = {&EntryBB};
  if (PDT) {
    for (auto *Node = PDT->getNode(&EntryBB); Node; Node = Node->getIDom()) {
      MachineBasicBlock *CandBB = Node->getBlock();
      if (!CandBB)
        break;
      if (CandBB == &EntryBB || !isLoopFreeEntryPostDominator(*CandBB, EntryBB))
        continue;
      Candidates.push_back(CandBB);
    }
  }

  // Record each candidate's current layout offset. This is the order in which
  // the assembler emits blocks, and is used to determine the code range
  // covered by a prefetch.
  DenseMap<MachineBasicBlock *, uint64_t> CandidateOffsets;
  if (Candidates.size() > 1) {
    uint64_t CodeSize = 0;
    for (MachineBasicBlock &MB : MF) {
      CodeSize = alignTo(CodeSize, MB.getAlignment());
      if (isLoopFreeEntryPostDominator(MB, EntryBB))
        CandidateOffsets[&MB] = CodeSize;
      CodeSize += SIProgramInfo::getMachineBasicBlockCodeSize(MB, *TII);
    }
  }

  DebugLoc DL;

  // Each prefetch can transfer 4KiB of instructions. Retain the existing
  // slack for growth after this late pass, such as padding and alignment. The
  // offset and sdata operands are placeholders; AMDGPUAsmPrinter fixes them
  // up using the exact emitted code size.
  constexpr uint64_t PrefetchSlack = 2 * 1024;
  constexpr uint64_t BytesPerPrefetch = 4 * 1024;
  MFI->setICachePrefetchLines(Info.DescriptorPrefetchLines);
  uint64_t ProgramPrefetchSize = Info.ProgramSize + PrefetchSlack;
  unsigned NumPrefetches = llvm::divideCeil(
      ProgramPrefetchSize - Info.InitialSize, BytesPerPrefetch);
  NumPrefetches = std::min(Info.MaxNumPrefetchInsts, NumPrefetches);

  size_t NumCandidates = Candidates.size();
  unsigned Prefetches = 0;
  // In each candidate block, prefetch as much code as necessary before control
  // flow reaches the next candidate block, plus some slack to account for
  // prefetch latency.
  for (size_t Cand = 0, NextCand = 1; Cand < NumCandidates;
       ++Cand, ++NextCand) {
    unsigned TargetPrefetchCount = NumPrefetches;
    if (NextCand < NumCandidates) {
      // To the offset of the next candidate we add:
      // - PrefetchSlack: To make sure the last prefetch has the correct number
      //   of cache lines.
      // - BytesPerPrefetch: To account for the latency of the prefetch.
      uint64_t CandidatePrefetchSize =
          CandidateOffsets.lookup(Candidates[NextCand]) + PrefetchSlack +
          BytesPerPrefetch;

      if (CandidatePrefetchSize <= Info.InitialSize)
        continue;

      unsigned PrefetchesBeforeNext = llvm::divideCeil(
          CandidatePrefetchSize - Info.InitialSize, BytesPerPrefetch);
      TargetPrefetchCount = std::min(PrefetchesBeforeNext, TargetPrefetchCount);
    }
    MachineBasicBlock *CandBB = Candidates[Cand];
    MachineBasicBlock::iterator InsertPt =
        findMBBInsertionPoint(*CandBB, ST, CandBB == &EntryBB);
    for (; Prefetches < TargetPrefetchCount; ++Prefetches) {
      BuildMI(*CandBB, InsertPt, DL, TII->get(AMDGPU::S_PREFETCH_INST_PC_REL))
          .addImm(Info.DescriptorPrefetchLines +
                  Prefetches * CacheLinesPerPrefetch)
          // Function-relative target cache-line index, fixed up later.
          .addReg(AMDGPU::SGPR_NULL) // soffset
          .addImm(0)                 // sdata (fixed up later)
          .addImm(0);                // cpol
    }
  }

  return true;
}

PreservedAnalyses llvm::AMDGPUInsertICachePrefetchPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM) {
  std::optional<ICachePrefetchInfo> Info = getICachePrefetchInfo(MF);
  if (!Info)
    return PreservedAnalyses::all();

  if (MF.size() == 1) {
    AMDGPUInsertICachePrefetch(nullptr, nullptr).run(MF, *Info);
    return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
  }

  auto &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  auto &PDT = MFAM.getResult<MachinePostDominatorTreeAnalysis>(MF);
  if (!AMDGPUInsertICachePrefetch(&MLI, &PDT).run(MF, *Info))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

char AMDGPUInsertICachePrefetchLegacy::ID = 0;
char &llvm::AMDGPUInsertICachePrefetchID = AMDGPUInsertICachePrefetchLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUInsertICachePrefetchLegacy, DEBUG_TYPE,
                      "AMDGPU Insert ICache Prefetch", false, false)
INITIALIZE_PASS_END(AMDGPUInsertICachePrefetchLegacy, DEBUG_TYPE,
                    "AMDGPU Insert ICache Prefetch", false, false)
