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

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MORE = getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();
    if (!MORE.allowExtraAnalysis(DEBUG_TYPE))
      return false;

    const HexagonSubtarget &HST = MF.getSubtarget<HexagonSubtarget>();
    if (!HST.useHVXOps())
      return false;

    const MachineRegisterInfo &MRI = MF.getRegInfo();
    unsigned HVXLen = HST.getVectorLength();

    for (const MachineBasicBlock &MBB : MF) {
      // Backward liveness scan over virtual registers.  We track which
      // virtual registers are live at each point, then at call instructions
      // count those with HVX register classes.
      //
      // Use a SmallSet of virtual register numbers.  When walking backwards:
      //   - a def removes a vreg from the live set
      //   - a use adds a vreg to the live set
      // At each call, the live set holds vregs live AFTER the call (i.e., the
      // values that must survive across it and therefore need save/restore).
      SmallSet<Register, 32> LiveVRegs;

      // Seed with live-outs of the block (vregs used by successors).
      for (const MachineBasicBlock *Succ : MBB.successors()) {
        for (const auto &LI : Succ->liveins()) {
          Register R = LI.PhysReg;
          if (R.isVirtual())
            LiveVRegs.insert(R);
        }
      }

      for (const MachineInstr &MI : llvm::reverse(MBB)) {
        if (MI.isCall()) {
          // Count HVX virtual registers live after (and thus across) this
          // call.  HvxVR holds one vector (HVXLen bytes); HvxWR holds two
          // (2 * HVXLen bytes).
          unsigned NumVecs = 0;
          for (Register VReg : LiveVRegs) {
            if (!VReg.isVirtual())
              continue;
            const TargetRegisterClass *RC = MRI.getRegClass(VReg);
            if (RC == &Hexagon::HvxWRRegClass)
              NumVecs += 2;
            else if (RC == &Hexagon::HvxVRRegClass)
              NumVecs += 1;
          }
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
