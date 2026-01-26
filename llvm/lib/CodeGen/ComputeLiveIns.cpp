//===---------------= Compute live-ins for all basic blocks
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass computes live-ins for all basic blocks in a machine function.
// The main use of this pass is to be able to run passes which require live-ins
// info on the mir which does not have it.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "compute-live-ins"

namespace {
class ComputeLiveIns : public MachineFunctionPass {
public:
  static char ID;

  ComputeLiveIns() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (MF.empty())
      return false;

    MachineFunctionProperties &Props = MF.getProperties();
    if (Props.hasTracksLiveness())
      return false;

    Props.setTracksLiveness();
    SmallVector<MachineBasicBlock *> AllMBBsInPostOrder;
    AllMBBsInPostOrder.reserve(MF.getNumBlockIDs());
    for (MachineBasicBlock *MBB : post_order(MF))
      AllMBBsInPostOrder.push_back(MBB);

    fullyRecomputeLiveIns(AllMBBsInPostOrder);

    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // end anonymous namespace

char ComputeLiveIns::ID = 0;
char &llvm::ComputeLiveInsID = ComputeLiveIns::ID;

INITIALIZE_PASS(ComputeLiveIns, DEBUG_TYPE, "Compute Live Ins", false, false)
