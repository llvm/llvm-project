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
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-insert-icache-prefetch"

static cl::opt<bool>
    EnableICachePrefetch("amdgpu-icache-prefetch",
                         cl::desc("Insert ICache prefetch instructions"),
                         cl::init(true), cl::Hidden);

namespace {

class AMDGPUInsertICachePrefetch {
  MachineLoopInfo &MLI;
  MachinePostDominatorTree &PDT;

  bool isLoopFreeEntryPostDominator(const MachineBasicBlock &MBB,
                                    const MachineBasicBlock &EntryBB) const {
    return !MLI.getLoopFor(&MBB) && PDT.dominates(&MBB, &EntryBB);
  }

public:
  // These analyses describe the final machine CFG at this late insertion
  // point. CFG-based placement will use them to select loop-free blocks on
  // the entry block's post-dominator chain.
  AMDGPUInsertICachePrefetch(MachineLoopInfo &MLI,
                             MachinePostDominatorTree &PDT)
      : MLI(MLI), PDT(PDT) {}

  bool run(MachineFunction &MF);
};

class AMDGPUInsertICachePrefetchLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUInsertICachePrefetchLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    auto &PDT =
        getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
    return AMDGPUInsertICachePrefetch(MLI, PDT).run(MF);
  }
};

} // end anonymous namespace

static MachineBasicBlock::iterator
findMBBInsertionPoint(MachineBasicBlock &MBB, const GCNSubtarget &ST,
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
        (IsEntryBlock &&
         InsertPt->getOpcode() == AMDGPU::S_SETREG_IMM32_B32)) {
      ++InsertPt;
      continue;
    }
    if (!SkippedInitialUnclausedVmemPrologue &&
        InsertPt->getOpcode() == AMDGPU::GLOBAL_PREFETCH_B8_SADDR) {
      auto Next = InsertPt;
      ++Next;
      if (Next != MBB.end() && Next->getOpcode() == AMDGPU::V_NOP_e32) {
        InsertPt = ++Next;
        SkippedInitialUnclausedVmemPrologue = true;
        continue;
      }
    }
    break;
  }
  return InsertPt;
}

bool AMDGPUInsertICachePrefetch::run(MachineFunction &MF) {
  if (!EnableICachePrefetch)
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasICachePrefetch())
    return false;

  // Only run for AMDHSA - this is where kernel descriptors are used and
  // rsrc3 INST_PREF_SIZE is relevant.
  if (ST.getTargetTriple().getOS() != Triple::AMDHSA)
    return false;

  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  if (!MFI->isEntryFunction())
    return false;

  SIProgramInfo PI;
  uint64_t ProgramSize = PI.getFunctionCodeSize(MF);
  // The kernel descriptor can specify an instruction prefetch size of up to 256
  // in INST_PREF_SIZE. At a granularity of 128B, this equals 32KiB of
  // instructions that can be prefetched without inserting explicit prefetch
  // instructions.
  constexpr uint64_t MaxKDPrefetch = 1u << 15;
  if (ProgramSize <= MaxKDPrefetch)
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();
  MachineBasicBlock &EntryBB = MF.front();

  // Walk the post-dominator chain to get candidates in execution order. This
  // is distinct from the layout order used below to calculate code offsets.
  SmallVector<MachineBasicBlock *> Candidates = {&EntryBB};
  for (auto *Node = PDT.getNode(&EntryBB); Node; Node = Node->getIDom()) {
    MachineBasicBlock *CandBB = Node->getBlock();
    if (!CandBB)
      break;
    if (CandBB == &EntryBB ||
        !isLoopFreeEntryPostDominator(*CandBB, EntryBB))
      continue;
    Candidates.push_back(CandBB);
  }

  // Record each candidate's current layout offset. This is the order in which
  // the assembler emits blocks, and is used to determine the code range
  // covered by a prefetch.
  DenseMap<MachineBasicBlock *, uint64_t> CandidateOffsets;
  uint64_t CodeSize = 0;
  for (MachineBasicBlock &MB : MF) {
    CodeSize = alignTo(CodeSize, MB.getAlignment());
    if (isLoopFreeEntryPostDominator(MB, EntryBB))
      CandidateOffsets[&MB] = CodeSize;
    CodeSize += SIProgramInfo::getMachineBasicBlockCodeSize(MB, *TII);
  }

  DebugLoc DL;

  // Each prefetch can transfer 4KiB of instructions. Retain the existing
  // slack for growth after this late pass, such as padding and alignment. The
  // offset and sdata operands are placeholders; AMDGPUAsmPrinter fixes them
  // up using the exact emitted code size.
  constexpr uint64_t PrefetchSlack = 2 * 1024;
  constexpr uint64_t BytesPerPrefetch = 4 * 1024;
  // Each prefetch can transfer up to 32 cachelines of 128 bytes = 4KiB.
  // 16 instructions cover 64KiB (the full ICache size).
  constexpr unsigned MaxNumPrefetchInsts = 16;
  unsigned NumPrefetches =
      llvm::divideCeil(ProgramSize + PrefetchSlack, BytesPerPrefetch);
  NumPrefetches = std::min(MaxNumPrefetchInsts, NumPrefetches);

  size_t NumCandidates = Candidates.size();
  unsigned Prefetches = 0;
  // In each candidate block, prefetch as much code as necessary before control
  // flow reaches the next candidate block, plus some slack to account for
  // prefetch latency.
  for (size_t Cand = 0, NextCand = 1; Cand < NumCandidates;
       ++Cand, ++NextCand) {
    unsigned PrefetchBeforeNext = NumPrefetches;
    if (NextCand < NumCandidates) {
      // To the offset of the next candidate we add:
      // - PrefetchSlack: To make sure the last prefetch has the correct number
      //   of cache lines.
      // - BytesPerPrefetch: To account for the latency of the prefetch.
      unsigned PrefetchesBeforeNext = llvm::divideCeil(
          CandidateOffsets.lookup(Candidates[NextCand]) + PrefetchSlack +
              BytesPerPrefetch,
          BytesPerPrefetch);
      PrefetchBeforeNext = std::min(PrefetchesBeforeNext, PrefetchBeforeNext);
    }
    MachineBasicBlock *CandBB = Candidates[Cand];
    MachineBasicBlock::iterator InsertPt =
        findMBBInsertionPoint(*CandBB, ST, CandBB == &EntryBB);
    for (; Prefetches < PrefetchBeforeNext; ++Prefetches) {
      BuildMI(*CandBB, InsertPt, DL, TII->get(AMDGPU::S_PREFETCH_INST_PC_REL))
          .addImm(0)                 // offset (placeholder, fixed up later)
          .addReg(AMDGPU::SGPR_NULL) // soffset
          .addImm(Prefetches);       // sdata (slot index, fixed up later)
    }
  }

  MFI->setHasICachePrefetch(true);
  return true;
}

PreservedAnalyses llvm::AMDGPUInsertICachePrefetchPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM) {
  auto &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  auto &PDT = MFAM.getResult<MachinePostDominatorTreeAnalysis>(MF);
  if (!AMDGPUInsertICachePrefetch(MLI, PDT).run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

char AMDGPUInsertICachePrefetchLegacy::ID = 0;
char &llvm::AMDGPUInsertICachePrefetchID = AMDGPUInsertICachePrefetchLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUInsertICachePrefetchLegacy, DEBUG_TYPE,
                      "AMDGPU Insert ICache Prefetch", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPUInsertICachePrefetchLegacy, DEBUG_TYPE,
                    "AMDGPU Insert ICache Prefetch", false, false)
