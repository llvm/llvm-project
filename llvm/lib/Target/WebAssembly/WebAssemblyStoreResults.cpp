//===-- WebAssemblyStoreResults.cpp - Optimize using store result values --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements an optimization pass using store result values.
///
/// WebAssembly's store instructions return the stored value. This is to enable
/// an optimization wherein uses of the stored value can be replaced by uses of
/// the store's result value, making the stored value register more likely to
/// be single-use, thus more likely to be useful to register stackifying, and
/// potentially also exposing the store to register stackifying. These both can
/// reduce get_local/set_local traffic.
///
/// This pass also performs this optimization for memcpy, memmove, and memset
/// calls, since the LLVM intrinsics for these return void so they can't use the
/// returned attribute and consequently aren't handled by the OptimizeReturned
/// pass.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-store-results"

namespace {
class WebAssemblyStoreResults final : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyStoreResults() : MachineFunctionPass(ID) {}

  const char *getPassName() const override {
    return "WebAssembly Store Results";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineBlockFrequencyInfo>();
    AU.addPreserved<MachineBlockFrequencyInfo>();
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    AU.addRequired<LiveIntervals>();
    AU.addPreserved<SlotIndexes>();
    AU.addPreserved<LiveIntervals>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
};
} // end anonymous namespace

char WebAssemblyStoreResults::ID = 0;
FunctionPass *llvm::createWebAssemblyStoreResults() {
  return new WebAssemblyStoreResults();
}

// Replace uses of FromReg with ToReg if they are dominated by MI.
static bool ReplaceDominatedUses(MachineBasicBlock &MBB, MachineInstr &MI,
                                 unsigned FromReg, unsigned ToReg,
                                 const MachineRegisterInfo &MRI,
                                 MachineDominatorTree &MDT,
                                 LiveIntervals &LIS) {
  bool Changed = false;

  LiveInterval *FromLI = &LIS.getInterval(FromReg);
  LiveInterval *ToLI = &LIS.getInterval(ToReg);

  SlotIndex FromIdx = LIS.getInstructionIndex(MI).getRegSlot();
  VNInfo *FromVNI = FromLI->getVNInfoAt(FromIdx);

  SmallVector<SlotIndex, 4> Indices;

  for (auto I = MRI.use_nodbg_begin(FromReg), E = MRI.use_nodbg_end(); I != E;) {
    MachineOperand &O = *I++;
    MachineInstr *Where = O.getParent();

    // Check that MI dominates the instruction in the normal way.
    if (&MI == Where || !MDT.dominates(&MI, Where))
      continue;

    // If this use gets a different value, skip it.
    SlotIndex WhereIdx = LIS.getInstructionIndex(*Where);
    VNInfo *WhereVNI = FromLI->getVNInfoAt(WhereIdx);
    if (WhereVNI && WhereVNI != FromVNI)
      continue;

    // Make sure ToReg isn't clobbered before it gets there.
    VNInfo *ToVNI = ToLI->getVNInfoAt(WhereIdx);
    if (ToVNI && ToVNI != FromVNI)
      continue;

    Changed = true;
    DEBUG(dbgs() << "Setting operand " << O << " in " << *Where << " from "
                 << MI << "\n");
    O.setReg(ToReg);

    // If the store's def was previously dead, it is no longer.
    if (!O.isUndef()) {
      MI.getOperand(0).setIsDead(false);

      Indices.push_back(WhereIdx.getRegSlot());
    }
  }

  if (Changed) {
    // Extend ToReg's liveness.
    LIS.extendToIndices(*ToLI, Indices);

    // Shrink FromReg's liveness.
    LIS.shrinkToUses(FromLI);

    // If we replaced all dominated uses, FromReg is now killed at MI.
    if (!FromLI->liveAt(FromIdx.getDeadSlot()))
      MI.addRegisterKilled(FromReg,
                           MBB.getParent()->getSubtarget<WebAssemblySubtarget>()
                                 .getRegisterInfo());
  }

  return Changed;
}

static bool optimizeStore(MachineBasicBlock &MBB, MachineInstr &MI,
                          const MachineRegisterInfo &MRI,
                          MachineDominatorTree &MDT,
                          LiveIntervals &LIS) {
  unsigned ToReg = MI.getOperand(0).getReg();
  unsigned FromReg = MI.getOperand(WebAssembly::StoreValueOperandNo).getReg();
  return ReplaceDominatedUses(MBB, MI, FromReg, ToReg, MRI, MDT, LIS);
}

static bool optimizeCall(MachineBasicBlock &MBB, MachineInstr &MI,
                         const MachineRegisterInfo &MRI,
                         MachineDominatorTree &MDT,
                         LiveIntervals &LIS,
                         const WebAssemblyTargetLowering &TLI,
                         const TargetLibraryInfo &LibInfo) {
  MachineOperand &Op1 = MI.getOperand(1);
  if (!Op1.isSymbol())
    return false;

  StringRef Name(Op1.getSymbolName());
  bool callReturnsInput = Name == TLI.getLibcallName(RTLIB::MEMCPY) ||
                          Name == TLI.getLibcallName(RTLIB::MEMMOVE) ||
                          Name == TLI.getLibcallName(RTLIB::MEMSET);
  if (!callReturnsInput)
    return false;

  LibFunc::Func Func;
  if (!LibInfo.getLibFunc(Name, Func))
    return false;

  unsigned FromReg = MI.getOperand(2).getReg();
  unsigned ToReg = MI.getOperand(0).getReg();
  if (MRI.getRegClass(FromReg) != MRI.getRegClass(ToReg))
    report_fatal_error("Store results: call to builtin function with wrong "
                       "signature, from/to mismatch");
  return ReplaceDominatedUses(MBB, MI, FromReg, ToReg, MRI, MDT, LIS);
}

bool WebAssemblyStoreResults::runOnMachineFunction(MachineFunction &MF) {
  DEBUG({
    dbgs() << "********** Store Results **********\n"
           << "********** Function: " << MF.getName() << '\n';
  });

  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();
  const WebAssemblyTargetLowering &TLI =
      *MF.getSubtarget<WebAssemblySubtarget>().getTargetLowering();
  const auto &LibInfo = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  LiveIntervals &LIS = getAnalysis<LiveIntervals>();
  bool Changed = false;

  // We don't preserve SSA form.
  MRI.leaveSSA();

  assert(MRI.tracksLiveness() && "StoreResults expects liveness tracking");

  for (auto &MBB : MF) {
    DEBUG(dbgs() << "Basic Block: " << MBB.getName() << '\n');
    for (auto &MI : MBB)
      switch (MI.getOpcode()) {
      default:
        break;
      case WebAssembly::STORE8_I32:
      case WebAssembly::STORE16_I32:
      case WebAssembly::STORE8_I64:
      case WebAssembly::STORE16_I64:
      case WebAssembly::STORE32_I64:
      case WebAssembly::STORE_F32:
      case WebAssembly::STORE_F64:
      case WebAssembly::STORE_I32:
      case WebAssembly::STORE_I64:
        Changed |= optimizeStore(MBB, MI, MRI, MDT, LIS);
        break;
      case WebAssembly::CALL_I32:
      case WebAssembly::CALL_I64:
        Changed |= optimizeCall(MBB, MI, MRI, MDT, LIS, TLI, LibInfo);
        break;
      }
  }

  return Changed;
}
