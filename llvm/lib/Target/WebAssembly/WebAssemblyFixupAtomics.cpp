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
/// This is used because ISel selects atomics with a default value for the
/// memory ordering immediate operand. Even though we run this pass early in
/// the MI pass pipeline, later MI passes should still use getMergedOrdering()
/// on the MachineMemOperand to get the ordering rather than the immediate.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblySubtarget.h"
#include "llvm/BinaryFormat/Wasm.h"
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
  if (!MF.getSubtarget<WebAssemblySubtarget>().hasRelaxedAtomics())
    return Changed;

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      const MCInstrDesc &Desc = MI.getDesc();
      for (unsigned I = 0, E = MI.getNumExplicitOperands(); I < E; ++I) {
        if (I < Desc.getNumOperands() &&
            Desc.operands()[I].OperandType == WebAssembly::OPERAND_MEMORDER &&
            // Fences are already selected with the correct ordering.
            MI.getOpcode() != WebAssembly::ATOMIC_FENCE) {
          assert(MI.getOperand(I).getImm() == wasm::WASM_MEM_ORDER_SEQ_CST &&
                 "Expected seqcst default atomics from ISel");
          assert(!MI.memoperands_empty());
          unsigned Order = wasm::WASM_MEM_ORDER_SEQ_CST;
          auto *MMO = *MI.memoperands_begin();
          switch (MMO->getMergedOrdering()) {
          case AtomicOrdering::Unordered:
          case AtomicOrdering::Monotonic:
          case AtomicOrdering::Acquire:
          case AtomicOrdering::Release:
          case AtomicOrdering::AcquireRelease:
            Order = wasm::WASM_MEM_ORDER_ACQ_REL;
            break;
          case AtomicOrdering::SequentiallyConsistent:
            Order = wasm::WASM_MEM_ORDER_SEQ_CST;
            break;
          default:
           report_fatal_error("Atomic instructions cannot have NotAtomic ordering");
          }
          if (MI.getOperand(I).getImm() != Order) {
            MI.getOperand(I).setImm(Order);
            Changed = true;
          }
        }
      }
    }
  }

  return Changed;
}
