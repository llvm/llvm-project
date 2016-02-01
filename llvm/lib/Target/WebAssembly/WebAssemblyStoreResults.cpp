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
                                 MachineDominatorTree &MDT) {
  bool Changed = false;
  for (auto I = MRI.use_begin(FromReg), E = MRI.use_end(); I != E;) {
    MachineOperand &O = *I++;
    MachineInstr *Where = O.getParent();
    if (Where->getOpcode() == TargetOpcode::PHI) {
      // PHIs use their operands on their incoming CFG edges rather than
      // in their parent blocks. Get the basic block paired with this use
      // of FromReg and check that MI's block dominates it.
      MachineBasicBlock *Pred =
          Where->getOperand(&O - &Where->getOperand(0) + 1).getMBB();
      if (!MDT.dominates(&MBB, Pred))
        continue;
    } else {
      // For a non-PHI, check that MI dominates the instruction in the
      // normal way.
      if (&MI == Where || !MDT.dominates(&MI, Where))
        continue;
    }
    Changed = true;
    DEBUG(dbgs() << "Setting operand " << O << " in " << *Where << " from "
                 << MI << "\n");
    O.setReg(ToReg);
    // If the store's def was previously dead, it is no longer. But the
    // dead flag shouldn't be set yet.
    assert(!MI.getOperand(0).isDead() && "Unexpected dead flag");
  }
  return Changed;
}

static bool optimizeStore(MachineBasicBlock &MBB, MachineInstr &MI,
                          const MachineRegisterInfo &MRI,
                          MachineDominatorTree &MDT) {
  const auto &Stored = MI.getOperand(WebAssembly::StoreValueOperandNo);
  switch (Stored.getType()) {
  case MachineOperand::MO_Register: {
    unsigned ToReg = MI.getOperand(0).getReg();
    unsigned FromReg = Stored.getReg();
    return ReplaceDominatedUses(MBB, MI, FromReg, ToReg, MRI, MDT);
  }
  case MachineOperand::MO_FrameIndex:
    // TODO: optimize.
    return false;
  default:
    report_fatal_error("Store results: store not consuming reg or frame index");
  }
}

static bool optimizeCall(MachineBasicBlock &MBB, MachineInstr &MI,
                         const MachineRegisterInfo &MRI,
                         MachineDominatorTree &MDT,
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

  const auto &Op2 = MI.getOperand(2);
  switch (Op2.getType()) {
  case MachineOperand::MO_Register: {
    unsigned FromReg = Op2.getReg();
    unsigned ToReg = MI.getOperand(0).getReg();
    if (MRI.getRegClass(FromReg) != MRI.getRegClass(ToReg))
      report_fatal_error("Store results: call to builtin function with wrong "
                         "signature, from/to mismatch");
    return ReplaceDominatedUses(MBB, MI, FromReg, ToReg, MRI, MDT);
  }
  case MachineOperand::MO_FrameIndex:
    // TODO: optimize.
    return false;
  default:
    report_fatal_error("Store results: call to builtin function with wrong "
                       "signature, not consuming reg or frame index");
  }
}

bool WebAssemblyStoreResults::runOnMachineFunction(MachineFunction &MF) {
  DEBUG({
    dbgs() << "********** Store Results **********\n"
           << "********** Function: " << MF.getName() << '\n';
  });

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();
  const WebAssemblyTargetLowering &TLI =
      *MF.getSubtarget<WebAssemblySubtarget>().getTargetLowering();
  const auto &LibInfo = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  bool Changed = false;

  assert(MRI.isSSA() && "StoreResults depends on SSA form");

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
        Changed |= optimizeStore(MBB, MI, MRI, MDT);
        break;
      case WebAssembly::CALL_I32:
      case WebAssembly::CALL_I64:
        Changed |= optimizeCall(MBB, MI, MRI, MDT, TLI, LibInfo);
        break;
      }
  }

  return Changed;
}
