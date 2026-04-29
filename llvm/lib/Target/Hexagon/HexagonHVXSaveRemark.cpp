//===- HexagonHVXSaveRemark.cpp - Remark on HVX saves around calls --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostic pass that emits optimization remarks when HVX vector registers
// must be saved and restored around function calls.  All HVX registers are
// caller-saved (Section 5.3 of the Hexagon ABI), so every HVX value that is
// live across a call requires a save/restore pair on the stack.  Each HVX
// vector is 64 or 128 bytes (depending on the mode), making this overhead
// expensive.  The remarks help programmers identify call sites where inlining,
// hoisting, or sinking the call could reduce the save/restore cost.
//
//===----------------------------------------------------------------------===//

#include "HexagonSubtarget.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "hexagon-hvx-save"

static cl::opt<unsigned> HVXSaveThreshold(
    "hexagon-hvx-save-threshold", cl::Hidden, cl::init(1024),
    cl::desc("Minimum bytes of HVX caller-saves to trigger a remark"));

namespace {

struct HexagonHVXSaveRemark : public MachineFunctionPass {
  static char ID;

  HexagonHVXSaveRemark() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MORE = getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();
    if (!MORE.allowExtraAnalysis(DEBUG_TYPE))
      return false;

    const HexagonSubtarget &HST = MF.getSubtarget<HexagonSubtarget>();
    if (!HST.useHVXOps())
      return false;

    // Identify HVX caller-save slots by matching stack-slot sizes against
    // the HVX vector length (single vectors and vector pairs).
    const MachineFrameInfo &MFI = MF.getFrameInfo();
    unsigned HVXLen = HST.getVectorLength();
    unsigned NumVecSaves = 0;
    unsigned NumPairSaves = 0;

    for (int I = MFI.getObjectIndexBegin(), E = MFI.getObjectIndexEnd(); I < E;
         ++I) {
      if (!MFI.isSpillSlotObjectIndex(I))
        continue;
      int64_t Size = MFI.getObjectSize(I);
      if (Size == (int64_t)HVXLen)
        ++NumVecSaves;
      else if (Size == (int64_t)(2 * HVXLen))
        ++NumPairSaves;
    }

    unsigned TotalSaves = NumVecSaves + NumPairSaves;
    unsigned TotalBytes = NumVecSaves * HVXLen + NumPairSaves * 2 * HVXLen;

    if (TotalBytes < HVXSaveThreshold)
      return false;

    // Emit a remark on each call site.
    for (const MachineBasicBlock &MBB : MF) {
      for (const MachineInstr &MI : MBB) {
        if (!MI.isCall())
          continue;

        MORE.emit([&]() {
          using namespace ore;
          MachineOptimizationRemarkAnalysis R(DEBUG_TYPE, "HVXSaveAroundCall",
                                              MI.getDebugLoc(), &MBB);
          R << NV("NumSaves", TotalSaves) << " HVX caller-saved register(s) ("
            << NV("TotalBytes", TotalBytes)
            << " bytes) saved and restored around call";
          return R;
        });
      }
    }

    return false;
  }

  StringRef getPassName() const override { return "Hexagon HVX Save Remarks"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineOptimizationRemarkEmitterPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char HexagonHVXSaveRemark::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(HexagonHVXSaveRemark, DEBUG_TYPE, "Hexagon HVX Save Remarks",
                false, false)

FunctionPass *llvm::createHexagonHVXSaveRemark() {
  return new HexagonHVXSaveRemark();
}
