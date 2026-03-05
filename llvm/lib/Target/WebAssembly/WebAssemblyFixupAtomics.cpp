//===-- WebAssemblyFixupAtomics.cpp - Fixup Atomics -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Fixes memory ordering operands for atomic instructions.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-fixup-atomics"

namespace {
class WebAssemblyFixupAtomics final : public MachineFunctionPass {
  StringRef getPassName() const override { return "WebAssembly Fixup Atomics"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyFixupAtomics() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyFixupAtomics::ID = 0;
INITIALIZE_PASS(WebAssemblyFixupAtomics, DEBUG_TYPE,
                "Fixup the memory ordering of atomics", false, false)

FunctionPass *llvm::createWebAssemblyFixupAtomics() {
  return new WebAssemblyFixupAtomics();
}

bool WebAssemblyFixupAtomics::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** Fixup Atomics **********\n"
                    << "********** Function: " << MF.getName() << '\n');

  bool Changed = false;
  const auto &Subtarget = MF.getSubtarget<WebAssemblySubtarget>();
  bool HasRelaxedAtomics = Subtarget.hasRelaxedAtomics();

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      const MCInstrDesc &Desc = MI.getDesc();
      for (unsigned I = 0, E = MI.getNumExplicitOperands(); I < E; ++I) {
        if (I < Desc.getNumOperands() &&
            Desc.operands()[I].OperandType == WebAssembly::OPERAND_MEMORDER) {

          if (HasRelaxedAtomics && !MI.memoperands_empty()) {
            unsigned Order = 0; // seqcst
            auto *MMO = *MI.memoperands_begin();
            switch (MMO->getMergedOrdering()) {
            case AtomicOrdering::Acquire:
            case AtomicOrdering::Release:
            case AtomicOrdering::AcquireRelease:
            case AtomicOrdering::Monotonic:
              Order = 1; // acqrel
              break;
            default:
              Order = 0; // seqcst
              break;
            }
            if (MI.getOperand(I).getImm() != Order) {
              MI.getOperand(I).setImm(Order);
              Changed = true;
            }
          }
        }
      }
    }
  }

  return Changed;
}
