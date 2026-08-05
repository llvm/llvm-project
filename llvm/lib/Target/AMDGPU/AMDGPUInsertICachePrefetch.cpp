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
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "SIProgramInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
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
public:
  bool run(MachineFunction &MF);
};

class AMDGPUInsertICachePrefetchLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUInsertICachePrefetchLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    return AMDGPUInsertICachePrefetch().run(MF);
  }
};

} // end anonymous namespace

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

  MachineBasicBlock &EntryBB = MF.front();
  MachineBasicBlock::iterator InsertPt = EntryBB.begin();

  // Skip past any instructions that must remain at the very beginning:
  // - Debug values and CFI instructions
  // - The gfx1250 initial unclaused-VMEM workaround
  // - S_SETREG_IMM32_B32 instructions that set up MODE register bits
  //   (e.g., REPLAY_MODE bit 25 from SIFrameLowering)
  // We want the prefetches to come after all initial MODE setup.
  bool SkippedInitialUnclausedVmemPrologue =
      !ST.hasRequiresInitialUnclausedVmem();
  while (InsertPt != EntryBB.end()) {
    if (InsertPt->isDebugValue() || InsertPt->isCFIInstruction() ||
        InsertPt->getOpcode() == AMDGPU::S_SETREG_IMM32_B32) {
      ++InsertPt;
      continue;
    }
    if (!SkippedInitialUnclausedVmemPrologue &&
        InsertPt->getOpcode() == AMDGPU::GLOBAL_PREFETCH_B8_SADDR) {
      auto Next = InsertPt;
      ++Next;
      if (Next != EntryBB.end() &&
          Next->getOpcode() == AMDGPU::V_NOP_e32) {
        InsertPt = ++Next;
        SkippedInitialUnclausedVmemPrologue = true;
        continue;
      }
    }
    break;
  }

  const SIInstrInfo *TII = ST.getInstrInfo();
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
  for (unsigned I = 0; I < NumPrefetches; ++I) {
    BuildMI(EntryBB, InsertPt, DL, TII->get(AMDGPU::S_PREFETCH_INST_PC_REL))
        .addImm(0)                 // offset (placeholder, fixed up later)
        .addReg(AMDGPU::SGPR_NULL) // soffset
        .addImm(I);                // sdata (slot index, fixed up later)
  }

  // The instruction is fire-and-forget: it updates no wave wait counter and
  // returns neither a result nor an error. It is therefore safe to insert
  // after the wait-counter and hazard passes.
  MFI->setHasICachePrefetch(true);
  return true;
}

PreservedAnalyses
llvm::AMDGPUInsertICachePrefetchPass::run(MachineFunction &MF,
                                          MachineFunctionAnalysisManager &) {
  if (!AMDGPUInsertICachePrefetch().run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

char AMDGPUInsertICachePrefetchLegacy::ID = 0;
char &llvm::AMDGPUInsertICachePrefetchID = AMDGPUInsertICachePrefetchLegacy::ID;

INITIALIZE_PASS(AMDGPUInsertICachePrefetchLegacy, DEBUG_TYPE,
                "AMDGPU Insert ICache Prefetch", false, false)
