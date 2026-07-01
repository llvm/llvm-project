//===- HexagonHVXSaveRemark.cpp - Remark on HVX saves around calls --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostic pass that emits optimization remarks when HVX vector registers
// are live across function calls.  All HVX registers are caller-saved
// (Section 5.3 of the Hexagon ABI), so every HVX value that is live across a
// call requires a save/restore pair on the stack.  Each HVX vector is 64 or
// 128 bytes (depending on the mode), making this overhead expensive.  The
// remarks help programmers identify call sites where inlining, hoisting, or
// sinking the call could reduce the save/restore cost.
//
// The pass runs before register allocation while values are still in virtual
// registers.  A backward liveness scan over each basic block counts the HVX
// virtual registers (and their corresponding byte cost) live at each call
// instruction.
//
//===----------------------------------------------------------------------===//

#include "HexagonSubtarget.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "hexagon-hvx-save"

static cl::opt<unsigned> HVXSaveThreshold(
    "hexagon-hvx-save-threshold", cl::Hidden, cl::init(128 * 8),
    cl::desc("Minimum number of bytes of HVX caller-saved register data live "
             "across a call to trigger a remark (default: 8 x 128-byte "
             "vectors)"));

namespace {

struct HexagonHVXSaveRemark : public MachineFunctionPass {
  static char ID;

  HexagonHVXSaveRemark() : MachineFunctionPass(ID) {}

  // Returns the number of HVX vectors represented by VReg: 2 for HvxWR
  // (vector pair), 1 for HvxVR (single vector), 0 for non-HVX registers.
  static unsigned hvxVecCount(Register VReg, const MachineRegisterInfo &MRI) {
    const TargetRegisterClass *RC = MRI.getRegClass(VReg);
    if (RC == &Hexagon::HvxWRRegClass)
      return 2;
    if (RC == &Hexagon::HvxVRRegClass)
      return 1;
    return 0;
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MORE = getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();
    if (!MORE.allowExtraAnalysis(DEBUG_TYPE))
      return false;

    const HexagonSubtarget &HST = MF.getSubtarget<HexagonSubtarget>();
    if (!HST.useHVXOps())
      return false;

    const MachineRegisterInfo &MRI = MF.getRegInfo();
    unsigned HVXLen = HST.getVectorLength();

    // Compute LiveOut[B] for each block: the set of HVX virtual registers
    // that are live on exit from B.  We use a standard backward dataflow
    // fixed-point:
    //
    //   LiveIn[B]  = UEVar[B] union (LiveOut[B] - Def[B])
    //   LiveOut[B] = union over successors S of LiveIn[S]
    //
    // where UEVar[B] is the set of HVX vregs that are used in B before any
    // definition of that vreg in B (upward-exposed uses), and Def[B] is the
    // set of HVX vregs defined in B.
    //
    // Because MachineBasicBlock::liveins() only contains physical registers,
    // we cannot seed cross-block virtual register liveness from successor
    // liveins -- we must compute it ourselves.

    unsigned NumBlocks = MF.getNumBlockIDs();
    using VRegSet = SmallSet<Register, 8>;

    // Per-block UEVar and Def sets (HVX vregs only).
    SmallVector<VRegSet, 16> UEVar(NumBlocks), BlockDef(NumBlocks);

    for (const MachineBasicBlock &MBB : MF) {
      unsigned BN = MBB.getNumber();
      VRegSet Defs;
      for (const MachineInstr &MI : MBB) {
        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg())
            continue;
          Register R = MO.getReg();
          if (!R.isVirtual() || !hvxVecCount(R, MRI))
            continue;
          if (MO.isDef()) {
            Defs.insert(R);
          } else if (MO.isUse() && !Defs.count(R)) {
            UEVar[BN].insert(R); // upward-exposed use
          }
        }
      }
      BlockDef[BN] = Defs;
    }

    // LiveOut[B] and LiveIn[B] maps.
    SmallVector<VRegSet, 16> LiveOut(NumBlocks), LiveIn(NumBlocks);

    // Seed LiveIn from UEVar and iterate until stable.
    for (unsigned I = 0; I < NumBlocks; ++I)
      LiveIn[I] = UEVar[I];

    bool Changed = true;
    while (Changed) {
      Changed = false;
      for (const MachineBasicBlock &MBB : MF) {
        unsigned BN = MBB.getNumber();

        // LiveOut[B] = union of LiveIn[S] for each successor S.
        VRegSet NewLiveOut;
        for (const MachineBasicBlock *Succ : MBB.successors())
          for (Register R : LiveIn[Succ->getNumber()])
            NewLiveOut.insert(R);

        if (NewLiveOut != LiveOut[BN]) {
          LiveOut[BN] = NewLiveOut;
          Changed = true;
        }

        // LiveIn[B] = UEVar[B] union (LiveOut[B] - Def[B]).
        VRegSet NewLiveIn = UEVar[BN];
        for (Register R : LiveOut[BN])
          if (!BlockDef[BN].count(R))
            NewLiveIn.insert(R);

        if (NewLiveIn != LiveIn[BN]) {
          LiveIn[BN] = NewLiveIn;
          Changed = true;
        }
      }
    }

    // Now do the backward scan over each block, seeded from LiveOut[B].
    for (const MachineBasicBlock &MBB : MF) {
      // Backward liveness scan over virtual registers.  We track which
      // virtual registers are live at each point, then at call instructions
      // count those with HVX register classes.
      //
      // When walking backwards:
      //   - a def removes a vreg from the live set
      //   - a use adds a vreg to the live set
      // At each call, the live set holds vregs live after the call (i.e., the
      // values that must survive across it and therefore need save/restore).
      VRegSet LiveVRegs = LiveOut[MBB.getNumber()];

      for (const MachineInstr &MI : llvm::reverse(MBB)) {
        if (MI.isCall()) {
          // Count HVX virtual registers live after (and thus across) this
          // call.  HvxVR holds one vector (HVXLen bytes); HvxWR holds two
          // (2 * HVXLen bytes).
          unsigned NumVecs = 0;
          for (Register VReg : LiveVRegs)
            NumVecs += hvxVecCount(VReg, MRI);
          unsigned TotalBytes = NumVecs * HVXLen;

          LLVM_DEBUG(dbgs() << "HVXSaveRemark: call in " << MF.getName()
                            << " has " << NumVecs << " HVX vector(s) live ("
                            << TotalBytes << " bytes)\n");

          if (TotalBytes >= HVXSaveThreshold) {
            MORE.emit([&]() {
              MachineOptimizationRemarkAnalysis R(
                  DEBUG_TYPE, "HVXSaveAroundCall", MI.getDebugLoc(), &MBB);
              R << ore::NV("NumVecs", NumVecs)
                << " HVX caller-saved register(s) ("
                << ore::NV("TotalBytes", TotalBytes)
                << " bytes) live across call";
              return R;
            });
          }
        }

        // Update liveness: defs kill vregs, uses add them.
        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg() || !MO.getReg().isVirtual())
            continue;
          if (MO.isDef())
            LiveVRegs.erase(MO.getReg());
          else if (MO.isUse())
            LiveVRegs.insert(MO.getReg());
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
