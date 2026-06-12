//===-- AMDGPUAlignVALURuns.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// On gfx9, the instruction fetch unit delivers 32-byte aligned windows into a
/// per-wave instruction buffer (IB). When a contiguous run of 8-byte VALU
/// instructions starts at an odd-dword (4-byte-misaligned) boundary, every 4th
/// instruction straddles a fetch-window boundary, stalling the wave for ~1 quad
/// (4 cycles) each time.
///
/// This pass detects such runs in loops and inserts a single `s_nop 0` (4
/// bytes) before the run to shift it to an even-dword boundary, eliminating all
/// straddle stalls.
///
/// Runs after BranchRelaxation so that byte offsets are final.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-align-valu-runs"

static cl::opt<unsigned> AlignThreshold(
    "amdgpu-valu-run-align-threshold",
    cl::desc("Minimum run length (in instructions) to trigger alignment"),
    cl::init(8), cl::Hidden);

static cl::opt<bool>
    DisableAlignVALURuns("amdgpu-disable-valu-run-align",
                         cl::desc("Disable the VALU run alignment pass"),
                         cl::init(false), cl::Hidden);

namespace {

class AMDGPUAlignVALURuns {
  const SIInstrInfo *TII = nullptr;

  bool isQualifyingVALU(const MachineInstr &MI) const {
    if (MI.isMetaInstruction() || MI.isDebugInstr())
      return false;
    if (!SIInstrInfo::isVALU(MI))
      return false;
    if (SIInstrInfo::isMFMA(MI))
      return false;
    return TII->getInstSizeInBytes(MI) == 8;
  }

  bool isRunBreaker(const MachineInstr &MI, bool PrevWasVALU) const {
    if (MI.isMetaInstruction() || MI.isDebugInstr())
      return false;

    unsigned Opc = MI.getOpcode();
    if (SIInstrInfo::isWaitcnt(Opc))
      return true;

    if (SIInstrInfo::isMFMA(MI))
      return true;

    if (PrevWasVALU && (SIInstrInfo::isVMEM(MI) || SIInstrInfo::isSMRD(MI) ||
                        SIInstrInfo::isDS(MI) || SIInstrInfo::isFLAT(MI)))
      return true;

    return false;
  }

public:
  bool run(MachineFunction &MF, MachineLoopInfo *MLI);
};

class AMDGPUAlignVALURunsLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUAlignVALURunsLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    return AMDGPUAlignVALURuns().run(MF, &MLI);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(AMDGPUAlignVALURunsLegacy, DEBUG_TYPE,
                      "AMDGPU Align VALU Runs", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUAlignVALURunsLegacy, DEBUG_TYPE,
                    "AMDGPU Align VALU Runs", false, false)

char AMDGPUAlignVALURunsLegacy::ID = 0;

char &llvm::AMDGPUAlignVALURunsLegacyID = AMDGPUAlignVALURunsLegacy::ID;

PreservedAnalyses
AMDGPUAlignVALURunsPass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &MFAM) {
  auto &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  if (!AMDGPUAlignVALURuns().run(MF, &MLI))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool AMDGPUAlignVALURuns::run(MachineFunction &MF, MachineLoopInfo *MLI) {
  if (DisableAlignVALURuns)
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (ST.getGeneration() != AMDGPUSubtarget::GFX9)
    return false;

  TII = ST.getInstrInfo();
  bool Changed = false;

  // Accumulate byte offset from function start as we walk MBBs in layout order.
  uint64_t Offset = 0;

  for (MachineBasicBlock &MBB : MF) {
    bool InLoop = MLI && MLI->getLoopFor(&MBB);

    // Track state for run detection within this block.
    MachineInstr *RunStart = nullptr;
    uint64_t RunStartOffset = 0;
    unsigned RunLength = 0;
    bool PrevWasVALU = false;

    auto evaluateRun = [&]() {
      if (RunLength >= AlignThreshold && (RunStartOffset % 8 != 0)) {
        assert(RunStart && "RunStart must be set when RunLength > 0");
        BuildMI(MBB, RunStart, DebugLoc(), TII->get(AMDGPU::S_NOP)).addImm(0);
        // Shift all subsequent offsets.
        Offset += 4;
        Changed = true;
      }
      RunStart = nullptr;
      RunLength = 0;
    };

    for (MachineInstr &MI : MBB) {
      if (MI.isMetaInstruction() || MI.isDebugInstr())
        continue;

      unsigned InstSize = TII->getInstSizeInBytes(MI);

      if (InLoop && isQualifyingVALU(MI)) {
        if (RunLength == 0) {
          RunStart = &MI;
          RunStartOffset = Offset;
        }
        RunLength++;
        PrevWasVALU = true;
      } else if (isRunBreaker(MI, PrevWasVALU)) {
        if (InLoop)
          evaluateRun();
        else {
          RunStart = nullptr;
          RunLength = 0;
        }
        PrevWasVALU = false;
      } else {
        // Non-qualifying, non-breaking instruction (e.g. 4-byte SALU).
        // A non-8-byte instruction changes dword parity, so end the run.
        if (InstSize != 8) {
          if (InLoop)
            evaluateRun();
          else {
            RunStart = nullptr;
            RunLength = 0;
          }
        }
        PrevWasVALU = SIInstrInfo::isVALU(MI);
      }

      Offset += InstSize;
    }

    // End-of-block: evaluate any open run.
    if (InLoop)
      evaluateRun();
  }

  if (Changed)
    MF.setAlignment(Align(8));

  return Changed;
}
