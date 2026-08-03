//===-- PISACacheHintSelector.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cachehint (.cc) support
// - LLVM IR specifies cache hints via the 'pisa.cache.ctrl' MMRA tag
//   (carried on the instruction's !mmra metadata)
//   - that value is encoded within MI's target-specific flags by
//     PISA TargetLowering::getTargetMMOFlags()
// - PISA instructions define $cachehint input used in instruction printing
// - this pass extract target-specific flags from MI and encodes $cachehint

#include "MCTargetDesc/PISAInstPrinter.h"
#include "PISA.h"

#define GET_INSTRINFO_OPERAND_ENUM
#include "PISAGenInstrInfo.inc"

#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/Support/PISAAddrSpace.h"

#define DEBUG_TYPE "pisa-cache-hint-selector"
#define DEBUG_NAME "PISA Cache Hint Selector"

using namespace llvm;
using namespace llvm::PISA;

namespace {
class PISACacheHintSelector : public llvm::MachineFunctionPass {
public:
  static char ID;

  PISACacheHintSelector() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return DEBUG_NAME; }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // namespace

char PISACacheHintSelector::ID = 0;
INITIALIZE_PASS(PISACacheHintSelector, DEBUG_TYPE, DEBUG_NAME, false, false)

MachineFunctionPass *llvm::createPISACacheHintSelectorPass() {
  return new PISACacheHintSelector();
}

void PISACacheHintSelector::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

static unsigned getLoadCacheHintsFromFlags(MachineMemOperand::Flags F) {
  static_assert(MachineMemOperand::MOTargetFlag1 == 1u << 6,
                "Unexpected flag value");

  if (F & MachineMemOperand::MONonTemporal)
    return LoadCacheControl_L1UC_L2UC_L3UC;

  return (static_cast<unsigned>(F) >> 6) & 0xFU;
}

bool PISACacheHintSelector::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "Running PISA Cache Hint Selector\n");
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.memoperands_empty())
        continue;

      LLVM_DEBUG(dbgs() << "MI: " << MI);

      const int CacheHintIdx =
          PISA::getNamedOperandIdx(MI.getOpcode(), PISA::OpName::cachehint);
      if (CacheHintIdx == -1)
        continue;

      auto *MMO = *MI.memoperands_begin();
      if (!(MMO->isLoad() || MMO->isStore()))
        continue;

      PISAAS::AddressSpace AS =
          static_cast<PISAAS::AddressSpace>(MMO->getAddrSpace());
      // Shared address space exists in local memory only and
      // cache hints make no sense for it.
      if (AS == PISAAS::AddressSpace::SHARED)
        continue;

      // If the cache hint operand is already set, skip it.
      if (MI.getOperand(CacheHintIdx).getImm() != 0)
        continue;

      unsigned int CacheHint = getLoadCacheHintsFromFlags(MMO->getFlags());
      if (CacheHint != 1 && (MMO->isStore() || CacheHint != 15))
        MI.getOperand(CacheHintIdx).setImm(CacheHint);

      LLVM_DEBUG(dbgs() << "Updated MI: " << MI);
    }
  }

  return false;
}
