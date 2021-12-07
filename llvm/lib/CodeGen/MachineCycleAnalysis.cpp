//===- MachineCycleAnalysis.cpp - Compute CycleInfo for Machine IR --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/ADT/GenericCycleImpl.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineSSAContext.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

template class llvm::GenericCycleInfo<llvm::MachineSSAContext>;
template class llvm::GenericCycle<llvm::MachineSSAContext>;

//===----------------------------------------------------------------------===//
// MachineCycleInfoWrapperPass Implementation
//===----------------------------------------------------------------------===//
//
// The implementation details of the wrapper pass that holds a MachineCycleInfo
// suitable for use with the legacy pass manager.
//
//===----------------------------------------------------------------------===//

char MachineCycleInfoWrapperPass::ID = 0;

MachineCycleInfoWrapperPass::MachineCycleInfoWrapperPass()
    : MachineFunctionPass(ID) {
  initializeMachineCycleInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

INITIALIZE_PASS_BEGIN(MachineCycleInfoWrapperPass, "machine-cycles",
                      "Machine Cycle Info Analysis", true, true)
INITIALIZE_PASS_END(MachineCycleInfoWrapperPass, "machine-cycles",
                    "Machine Cycle Info Analysis", true, true)

void MachineCycleInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineCycleInfoWrapperPass::runOnMachineFunction(MachineFunction &Func) {
  CI.clear();

  F = &Func;
  CI.compute(Func);
  return false;
}

void MachineCycleInfoWrapperPass::print(raw_ostream &OS, const Module *) const {
  OS << "MachineCycleInfo for function: " << F->getName() << "\n";
  CI.print(OS);
}

void MachineCycleInfoWrapperPass::releaseMemory() {
  CI.clear();
  F = nullptr;
}

char MachineCycleInfoPrinterPass::ID = 0;

MachineCycleInfoPrinterPass::MachineCycleInfoPrinterPass()
    : MachineFunctionPass(ID) {
  initializeMachineCycleInfoPrinterPassPass(*PassRegistry::getPassRegistry());
}

INITIALIZE_PASS_BEGIN(MachineCycleInfoPrinterPass, "print-machine-cycles",
                      "Print Machine Cycle Info Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineCycleInfoWrapperPass)
INITIALIZE_PASS_END(MachineCycleInfoPrinterPass, "print-machine-cycles",
                    "Print Machine Cycle Info Analysis", true, true)

void MachineCycleInfoPrinterPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineCycleInfoWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineCycleInfoPrinterPass::runOnMachineFunction(MachineFunction &F) {
  auto &CI = getAnalysis<MachineCycleInfoWrapperPass>();
  CI.print(errs());
  return false;
}
