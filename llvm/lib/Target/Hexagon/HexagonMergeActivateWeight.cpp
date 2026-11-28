//===-- HexagonMergeActivateWeight.cpp - Pair HMX activate/weight ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "merge-hmx-activate-weight"

#include "Hexagon.h"
#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>

using namespace llvm;

namespace llvm {

// A class to generate a warning message about weight and activate
class DiagnosticInfoWeightActivate : public DiagnosticInfoResourceLimit {

public:
  DiagnosticInfoWeightActivate(const Function &Fn, unsigned S,
                               DiagnosticSeverity Severity = DS_Warning)
      : DiagnosticInfoResourceLimit(Fn, "activate weight pair", S, Severity) {}

  void print(DiagnosticPrinter &DP) const override {
    DP << "In " << getFunction() << ", "
       << "expected activate and weight pair";
  }

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_ResourceLimit;
  }
};

// Return true if \p Moved can be moved past every instruction in the range
// [Begin, End), in either direction. \p Moved is an HMX activation or weight
// instruction, which reads memory and updates the HMX state, so it must not be
// reordered with:
//  - calls, inline asm, barriers or terminators,
//  - side-effecting instructions (e.g. mxclracc, bias/cvt mxmem forms and any
//    other HMX instruction),
//  - volatile or atomic memory accesses,
//  - stores, which may write the memory read by \p Moved,
//  - instructions that redefine a register used by \p Moved, or that read or
//    redefine a register defined by \p Moved.
static bool canMoveAcross(const MachineInstr &Moved,
                          MachineBasicBlock::const_iterator Begin,
                          MachineBasicBlock::const_iterator End,
                          const TargetRegisterInfo *TRI) {
  for (const MachineInstr &MI : make_range(Begin, End)) {
    if (MI.isDebugInstr() || &MI == &Moved)
      continue;

    if (MI.isCall() || MI.isInlineAsm() || MI.isBarrier() ||
        MI.isTerminator() || MI.hasUnmodeledSideEffects() ||
        MI.hasOrderedMemoryRef() || MI.mayStore()) {
      LLVM_DEBUG(dbgs() << "Cannot move " << Moved << "  past " << MI);
      return false;
    }

    for (const MachineOperand &MO : Moved.operands()) {
      if (!MO.isReg() || !MO.getReg())
        continue;
      Register R = MO.getReg();
      if (MI.modifiesRegister(R, TRI) ||
          (MO.isDef() && MI.readsRegister(R, TRI))) {
        LLVM_DEBUG(dbgs() << "Register " << printReg(R, TRI)
                          << " prevents moving " << Moved << "  past " << MI);
        return false;
      }
    }
  }
  return true;
}

// Move each HMX activation instruction next to its matching weight instruction
// (or vice versa), so that the packetizer can put them in the same packet, as
// required by the hardware. Warn about activation or weight instructions that
// cannot be paired.
static bool runActivateWeight(MachineFunction &MF, LiveIntervals *LIS) {
  auto &HST = MF.getSubtarget<HexagonSubtarget>();
  const HexagonInstrInfo *HII = HST.getInstrInfo();
  const TargetRegisterInfo *TRI = HST.getRegisterInfo();
  const Function &F = MF.getFunction();
  bool Changed = false;

  auto Warn = [&]() {
    DiagnosticInfoWeightActivate WeightActivate(F, 1);
    F.getContext().diagnose(WeightActivate);
  };

  for (MachineBasicBlock &MBB : MF) {
    // The first instruction of a pair, waiting for its partner.
    MachineInstr *Pending = nullptr;

    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      bool IsActivate = HII->isHMXActivateInstr(MI);
      if (!IsActivate && !HII->isHMXWeightInstr(MI))
        continue;
      if (!Pending) {
        Pending = &MI;
        continue;
      }
      if (IsActivate == HII->isHMXActivateInstr(*Pending)) {
        // Two activations or two weights in a row.
        Warn();
        Pending = &MI;
        continue;
      }

      // Pending and MI form a pair. Sink the first one down to the second,
      // or, if that is not possible, hoist the second one up to the first.
      MachineBasicBlock::iterator Between =
          std::next(MachineBasicBlock::iterator(Pending));
      MachineBasicBlock::iterator Second(MI);
      if (canMoveAcross(*Pending, Between, Second, TRI)) {
        if (Between != Second) {
          MBB.splice(Second, &MBB, Pending);
          if (LIS)
            LIS->handleMove(*Pending, /*UpdateFlags=*/true);
          Changed = true;
        }
      } else if (canMoveAcross(MI, Between, Second, TRI)) {
        MBB.splice(Between, &MBB, &MI);
        if (LIS)
          LIS->handleMove(MI, /*UpdateFlags=*/true);
        Changed = true;
      } else {
        Warn();
      }
      Pending = nullptr;
    }

    if (Pending)
      Warn();
  }

  return Changed;
}

class HexagonMergeActivateWeight : public MachineFunctionPass {
public:
  static char ID;
  HexagonMergeActivateWeight() : MachineFunctionPass(ID) {
    PassRegistry &Registry = *PassRegistry::getPassRegistry();
    initializeHexagonMergeActivateWeightPass(Registry);
  }

  StringRef getPassName() const override {
    return "Hexagon Merge Activate Weight Instructions After Machine Scheduler";
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    return runActivateWeight(MF,
                             &getAnalysis<LiveIntervalsWrapperPass>().getLIS());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char HexagonMergeActivateWeight::ID = 0;
char &HexagonMergeActivateWeightID = HexagonMergeActivateWeight::ID;

class HexagonMergeActivateWeight2 : public MachineFunctionPass {
public:
  static char ID;
  HexagonMergeActivateWeight2() : MachineFunctionPass(ID) {
    PassRegistry &Registry = *PassRegistry::getPassRegistry();
    initializeHexagonMergeActivateWeight2Pass(Registry);
  }

  StringRef getPassName() const override {
    return "Hexagon Merge Activate Weight Instructions Before Packetizer";
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    return runActivateWeight(MF, /*LIS=*/nullptr);
  }
};

char HexagonMergeActivateWeight2::ID = 0;

} // namespace llvm

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

INITIALIZE_PASS_BEGIN(HexagonMergeActivateWeight,
                      "hexagon-merge-activate-weight",
                      "Hexagon merge activate weight after machine scheduler",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(HexagonMergeActivateWeight, "hexagon-merge-activate-weight",
                    "Hexagon merge activate weight after machine scheduler",
                    false, false)

INITIALIZE_PASS(HexagonMergeActivateWeight2, "hexagon-merge-activate-weight-2",
                "Hexagon merge activate weight before packetizer", false, false)

FunctionPass *llvm::createHexagonMergeActivateWeight() {
  return new HexagonMergeActivateWeight();
}

FunctionPass *llvm::createHexagonMergeActivateWeight2() {
  return new HexagonMergeActivateWeight2();
}
