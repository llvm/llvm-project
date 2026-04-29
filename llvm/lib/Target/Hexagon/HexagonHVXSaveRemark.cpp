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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "hexagon-hvx-save"

static cl::opt<unsigned> HVXSaveThreshold(
    "hexagon-hvx-save-threshold", cl::Hidden, cl::init(8),
    cl::desc("Minimum number of HVX caller-saved registers live across a call "
             "to trigger a remark"));

namespace {

struct HexagonHVXSaveRemark : public MachineFunctionPass {
  static char ID;

  HexagonHVXSaveRemark() : MachineFunctionPass(ID) {}

  /// Return true if MI is an HVX vector spill to a stack slot.
  static bool isHVXSpill(const MachineInstr &MI, unsigned HVXLen) {
    if (MI.mayStore() && MI.hasOneMemOperand()) {
      const MachineMemOperand *MMO = *MI.memoperands_begin();
      if (MMO->getSize().hasValue() && MMO->getPseudoValue() &&
          isa<FixedStackPseudoSourceValue>(MMO->getPseudoValue()))
        return MMO->getSize().getValue() == HVXLen ||
               MMO->getSize().getValue() == 2 * HVXLen;
    }
    return false;
  }

  /// Count HVX spills in a bundle or single instruction that contains a call.
  static unsigned countSpillsInBundle(const MachineInstr &BundleOrMI,
                                      unsigned HVXLen) {
    unsigned Count = 0;
    if (BundleOrMI.isBundle()) {
      auto MII = BundleOrMI.getIterator();
      for (++MII; MII->isBundledWithPred(); ++MII)
        if (isHVXSpill(*MII, HVXLen))
          ++Count;
    } else {
      if (isHVXSpill(BundleOrMI, HVXLen))
        ++Count;
    }
    return Count;
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MORE = getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();
    if (!MORE.allowExtraAnalysis(DEBUG_TYPE))
      return false;

    const HexagonSubtarget &HST = MF.getSubtarget<HexagonSubtarget>();
    if (!HST.useHVXOps())
      return false;

    unsigned HVXLen = HST.getVectorLength();

    for (const MachineBasicBlock &MBB : MF) {
      for (auto I = MBB.instr_begin(), E = MBB.instr_end(); I != E; ++I) {
        const MachineInstr &MI = *I;
        if (!MI.isCall() || MI.isBundledWithPred())
          continue;

        // Count HVX spills in the bundle containing this call (spills are
        // often packetized together with the call instruction).
        unsigned NumSpills = countSpillsInBundle(MI, HVXLen);

        // Also count HVX spills in immediately preceding bundles/instructions
        // that are purely spill operations.
        auto BundleIt = MachineBasicBlock::const_iterator(MI);
        while (BundleIt != MBB.begin()) {
          --BundleIt;
          unsigned PrevSpills = countSpillsInBundle(*BundleIt, HVXLen);
          if (PrevSpills == 0)
            break;
          NumSpills += PrevSpills;
        }

        LLVM_DEBUG(dbgs() << "HVXSaveRemark: call in " << MF.getName()
                          << " has " << NumSpills << " HVX spills\n");

        if (NumSpills >= HVXSaveThreshold) {
          unsigned TotalBytes = NumSpills * HVXLen;
          MORE.emit([&]() {
            MachineOptimizationRemarkAnalysis R(DEBUG_TYPE, "HVXSaveAroundCall",
                                                MI.getDebugLoc(), &MBB);
            R << ore::NV("NumSaves", NumSpills)
              << " HVX caller-saved register(s) ("
              << ore::NV("TotalBytes", TotalBytes)
              << " bytes) live across call";
            return R;
          });
        }
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
