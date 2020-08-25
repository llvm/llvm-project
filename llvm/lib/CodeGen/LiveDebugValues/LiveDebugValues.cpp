//===- LiveDebugValues.cpp - Tracking Debug Value MIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LiveDebugValues.h"

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

/// \file LiveDebugValues.cpp
///
/// The LiveDebugValues pass extends the range of variable locations
/// (specified by DBG_VALUE instructions) from single blocks to successors
/// and any other code locations where the variable location is valid.
/// There are currently two implementations: the "VarLoc" implementation
/// explicitly tracks the location of a variable, while the "InstrRef"
/// implementation tracks the values defined by instructions through locations.
///
/// This file implements neither; it merely registers the pass, allows the
/// user to pick which implementation will be used to propagate variable
/// locations.

#define DEBUG_TYPE "livedebugvalues"

using namespace llvm;

/// Generic LiveDebugValues pass. Calls through to VarLocBasedLDV or
/// InstrRefBasedLDV to perform location propagation, via the LDVImpl
/// base class.
class LiveDebugValues : public MachineFunctionPass {
public:
  static char ID;

  LiveDebugValues();
  ~LiveDebugValues() {
    if (TheImpl)
      delete TheImpl;
  }

  /// Calculate the liveness information for the given machine function.
  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  LDVImpl *TheImpl;
  TargetPassConfig *TPC;
};

char LiveDebugValues::ID = 0;

char &llvm::LiveDebugValuesID = LiveDebugValues::ID;

INITIALIZE_PASS(LiveDebugValues, DEBUG_TYPE, "Live DEBUG_VALUE analysis", false,
                false)

/// Default construct and initialize the pass.
LiveDebugValues::LiveDebugValues() : MachineFunctionPass(ID) {
  initializeLiveDebugValuesPass(*PassRegistry::getPassRegistry());
  TheImpl = nullptr;
}

bool LiveDebugValues::runOnMachineFunction(MachineFunction &MF) {
  if (!TheImpl) {
    TPC = getAnalysisIfAvailable<TargetPassConfig>();

    bool InstrRefBased = false;
    if (TPC) {
      auto &TM = TPC->getTM<TargetMachine>();
      InstrRefBased = TM.Options.ValueTrackingVariableLocations;
    }

    if (InstrRefBased)
      TheImpl = llvm::makeInstrRefBasedLiveDebugValues();
    else
      TheImpl = llvm::makeVarLocBasedLiveDebugValues();
  }

  return TheImpl->ExtendRanges(MF, TPC);
}
