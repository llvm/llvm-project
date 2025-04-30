//===-- Next32OrderCallChain.cpp - OrderCall instructions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass that adds feeder and add to order call
// instructions
//
//===----------------------------------------------------------------------===//

#include "Next32.h"
#include "Next32InstrInfo.h"
#include "Next32PassTrace.h"
#include "Next32Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace llvm {
void initializeOrderCallChainPassPass(PassRegistry &);
}

#define ORDERCALLCHAIN_DESC "Next32 OrderCallChain"
#define ORDERCALLCHAIN_NAME "Next32-ordercallchain"

#define DEBUG_TYPE ORDERCALLCHAIN_NAME

namespace {
class OrderCallChainPass : public MachineFunctionPass {

public:
  static char ID;

  StringRef getPassName() const override { return ORDERCALLCHAIN_DESC; }

  OrderCallChainPass() : MachineFunctionPass(ID) {
    initializeOrderCallChainPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const Next32InstrInfo *TII;

  void ProcessBasicBlock(MachineBasicBlock &MBB);
};
} // namespace

char OrderCallChainPass::ID = 0;

INITIALIZE_PASS(OrderCallChainPass, ORDERCALLCHAIN_NAME, ORDERCALLCHAIN_DESC,
                false, false)

FunctionPass *llvm::createNext32OrderCallChain() {
  return new OrderCallChainPass();
}

void OrderCallChainPass::ProcessBasicBlock(MachineBasicBlock &MBB) {
  LLVM_DEBUG(dbgs() << "Process basic block: " << MBB.getNumber() << "\n");

  if (!MBB.isLiveIn(Next32::TID))
    MBB.addLiveIn(Next32::TID);
}

bool OrderCallChainPass::runOnMachineFunction(MachineFunction &Func) {
  Next32PassTrace TFunc(DEBUG_TYPE, Func);
  TII = Func.getSubtarget<Next32Subtarget>().getInstrInfo();
  for (auto &MBB : TFunc)
    ProcessBasicBlock(MBB);
  return false;
}
