//===-- PISAScopeSelector.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// syncscope("<target-scope>") support for atomics
// - atomic operations take a 'scope' argument (part of MachineMemOperand)
// - PISA instructions define $scope input used in instruction printing
// - this pass extract 'scope' from MI and encodes associated value into $scope
#include "MCTargetDesc/PISAInstPrinter.h"
#include "PISA.h"

#define GET_INSTRINFO_OPERAND_ENUM
#include "PISAGenInstrInfo.inc"
#define GET_AtomicScopeControl_DECL
#include "PISAGenSearchableTables.inc"

#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/IR/DiagnosticInfo.h"

#define DEBUG_TYPE "pisa-scope-selector"
#define DEBUG_NAME "PISA Scope Selector"

using namespace llvm;
using namespace llvm::PISA;

namespace {
class PISAScopeSelector : public llvm::MachineFunctionPass {
public:
  static char ID;

  PISAScopeSelector() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return DEBUG_NAME; }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // namespace

char PISAScopeSelector::ID = 0;
INITIALIZE_PASS(PISAScopeSelector, DEBUG_TYPE, DEBUG_NAME, false, false)

MachineFunctionPass *llvm::createPISAScopeSelectorPass() {
  return new PISAScopeSelector();
}

void PISAScopeSelector::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

namespace {
const StringMap<unsigned> ScopeName2Encoding = {
    {"workgroup", AtomicScopeControl_WORKGROUP},
    {"gpu", AtomicScopeControl_GPU},
    {"system", AtomicScopeControl_SYSTEM},
    // using workgroup scope
    {"subgroup", AtomicScopeControl_WORKGROUP},
    {"workitem", AtomicScopeControl_WORKGROUP},
};
} // namespace

bool PISAScopeSelector::runOnMachineFunction(MachineFunction &MF) {
  // SyncScopeID are dynamically assigned during parsing, so
  // we need to map them back to AtomicScopeControl definitions
  auto &Ctx = MF.getFunction().getContext();
  DenseMap<SyncScope::ID, unsigned> ScopeID2Encoding;
  for (const auto &[Name, Encoding] : ScopeName2Encoding) {
    auto ID = Ctx.getOrInsertSyncScopeID(Name);
    ScopeID2Encoding.emplace_or_assign(ID, Encoding);
  }
  ScopeID2Encoding.emplace_or_assign(SyncScope::System,
                                     AtomicScopeControl_SYSTEM);

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.memoperands_empty())
        continue;

      auto *MMO = *MI.memoperands_begin();
      if (!(MMO->isLoad() || MMO->isStore()))
        continue;

      auto Ordering = MMO->getSuccessOrdering();
      // Catch inconsistent atomic ordering
      if (!isValidAtomicOrdering(static_cast<unsigned>(Ordering))) {
        MI.emitGenericError("invalid atomic ordering in MachineMemOperand");
        continue;
      }

      if (!isStrongerThanMonotonic(Ordering))
        continue;

      auto OpName = PISA::OpName::scope;
      auto OpIdx = PISA::getNamedOperandIdx(MI.getOpcode(), OpName);
      if (OpIdx == -1)
        continue;

      if (MI.getOperand(OpIdx).getImm() != AtomicScopeControl_NONE)
        continue; // skip if already set (pisa2pisa)

      auto ScopeID = MMO->getSyncScopeID();
      auto Entry = ScopeID2Encoding.find(ScopeID);
      if (Entry == ScopeID2Encoding.end())
        llvm_unreachable("unsupported syncscope");
      MI.getOperand(OpIdx).setImm(Entry->second);
    }
  }
  return false;
}
