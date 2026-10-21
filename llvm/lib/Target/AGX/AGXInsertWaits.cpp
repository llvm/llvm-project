//===-- AGXInsertWaits.cpp - Insert scoreboard waits ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Memory loads, stack loads and atomics write their results asynchronously,
// tracked by one of two scoreboard groups; `wait N` waits for group N only
// (measured on the M2 Max: ptx-re-blockers B34). Like SIInsertWaitcnts, this
// pass inserts a wait on a group before any instruction reads or overwrites a
// register with a pending result in it, both groups before barriers and at the
// end of every block with pending results (so that no wait is needed across
// control flow). At most eight requests are kept outstanding.
//
// Each request goes to a group by where its result is first needed: requests
// needed at the same instruction share a group, and a request does not join a
// group that holds one needed earlier than itself (whose wait would then stall
// on it), so the wait for the first results does not wait for later ones.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "agx-insert-waits"

namespace {

class AGXInsertWaits : public MachineFunctionPass {
public:
  static char ID;
  AGXInsertWaits() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Insert Waits"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AGXInsertWaits::ID = 0;

INITIALIZE_PASS(AGXInsertWaits, DEBUG_TYPE, "AGX Insert Waits", false, false)

FunctionPass *llvm::createAGXInsertWaitsPass() { return new AGXInsertWaits(); }

static constexpr unsigned MaxPending = 8;

static bool isBarrier(const MachineInstr &MI) {
  return MI.getOpcode() == AGX::THREADGROUP_BARRIER ||
         MI.getOpcode() == AGX::BLOCK_BARRIER ||
         MI.getOpcode() == AGX::MEMORY_BARRIER;
}

/// Where the result of each asynchronous instruction of MBB is first needed:
/// the index of the first later instruction that touches one of its results,
/// is a barrier or ends the block.
static DenseMap<const MachineInstr *, unsigned>
firstUses(MachineBasicBlock &MBB, const TargetRegisterInfo &TRI) {
  DenseMap<const MachineInstr *, unsigned> Out;
  SmallVector<MachineInstr *, 64> Instrs;
  for (MachineInstr &MI : MBB)
    Instrs.push_back(&MI);
  for (unsigned I = 0; I < Instrs.size(); ++I) {
    const MachineInstr &MI = *Instrs[I];
    if (!AGXInstrInfo::isAsync(MI) || !MI.getNumExplicitDefs())
      continue;
    unsigned Use = Instrs.size();
    for (unsigned J = I + 1; J < Instrs.size() && Use == Instrs.size(); ++J) {
      const MachineInstr &UI = *Instrs[J];
      if (isBarrier(UI) || UI.isTerminator()) {
        Use = J;
        break;
      }
      for (const MachineOperand &D : MI.defs())
        for (const MachineOperand &MO : UI.operands())
          if (MO.isReg() && MO.getReg() && MO.getReg() != AGX::R0L &&
              TRI.regsOverlap(MO.getReg(), D.getReg()))
            Use = J;
    }
    Out[&MI] = Use;
  }
  return Out;
}

bool AGXInsertWaits::runOnMachineFunction(MachineFunction &MF) {
  const AGXSubtarget &ST = MF.getSubtarget<AGXSubtarget>();
  const AGXInstrInfo &TII = *ST.getInstrInfo();
  const TargetRegisterInfo &TRI = *ST.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    DenseMap<const MachineInstr *, unsigned> FirstUse = firstUses(MBB, TRI);
    struct Group {
      BitVector Pending;
      unsigned Count = 0;
      // The earliest first use of the group's pending results.
      unsigned Needed = ~0u;
    };
    Group Groups[2] = {{BitVector(TRI.getNumRegUnits())},
                       {BitVector(TRI.getNumRegUnits())}};

    auto Clear = [&](unsigned G) {
      Groups[G].Pending.reset();
      Groups[G].Count = 0;
      Groups[G].Needed = ~0u;
    };
    auto Wait = [&](MachineBasicBlock::iterator I, unsigned G) {
      BuildMI(MBB, I, DebugLoc(), TII.get(AGX::WAIT)).addImm(G);
      Clear(G);
      Changed = true;
    };
    auto Touches = [&](const MachineInstr &MI, unsigned G) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg() || MO.getReg() == AGX::R0L)
          continue;
        for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg()))
          if (Groups[G].Pending.test(static_cast<unsigned>(U)))
            return true;
      }
      return false;
    };

    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == AGX::WAIT) {
        Clear(MI.getOperand(0).getImm() & 1);
        continue;
      }
      bool All = isBarrier(MI) || MI.isTerminator();
      for (unsigned G = 0; G < 2; ++G)
        if (Groups[G].Count && (All || Touches(MI, G)))
          Wait(MI.getIterator(), G);
      // Asynchronous instructions without a result (atomics whose old value
      // is unused) leave nothing to wait for, like stores.
      if (!AGXInstrInfo::isAsync(MI) || !MI.getNumExplicitDefs())
        continue;
      unsigned Use = FirstUse.lookup(&MI);
      int GIdx = AGX::getNamedOperandIdx(MI.getOpcode(), AGX::OpName::g);
      unsigned G = 0;
      if (GIdx >= 0) {
        // A group none of whose results is needed before this one's,
        // preferring one needed at the same point, then an empty one;
        // failing both, the group needed last.
        auto Fits = [&](unsigned K) {
          return !Groups[K].Count || Groups[K].Needed >= Use;
        };
        if (Fits(0) && Fits(1))
          G = Groups[1].Count && Groups[1].Needed == Use ? 1
              : Groups[0].Count && Groups[0].Needed != Use && !Groups[1].Count
                  ? 1
                  : 0;
        else if (Fits(0) || Fits(1))
          G = Fits(0) ? 0 : 1;
        else
          G = Groups[1].Needed > Groups[0].Needed ? 1 : 0;
        MI.getOperand(GIdx).setImm(G);
      }
      if (Groups[0].Count + Groups[1].Count >= MaxPending) {
        // Make room: wait for the group needed first.
        unsigned E = Groups[1].Count && (!Groups[0].Count ||
                                         Groups[1].Needed < Groups[0].Needed)
                         ? 1
                         : 0;
        Wait(MI.getIterator(), E);
      }
      for (const MachineOperand &MO : MI.defs())
        for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg()))
          Groups[G].Pending.set(static_cast<unsigned>(U));
      ++Groups[G].Count;
      Groups[G].Needed = std::min(Groups[G].Needed, Use);
      Changed = true;
    }
    for (unsigned G = 0; G < 2; ++G)
      if (Groups[G].Count)
        Wait(MBB.end(), G);
  }
  return Changed;
}
