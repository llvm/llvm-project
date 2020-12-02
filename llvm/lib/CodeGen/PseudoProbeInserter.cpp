//===- PseudoProbeInserter.cpp - Insert annotation for callsite profiling -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements PseudoProbeInserter pass, which inserts pseudo probe
// annotations for call instructions with a pseudo-probe-specific dwarf
// discriminator. such discriminator indicates that the call instruction comes
// with a pseudo probe, and the discriminator value holds information to
// identify the corresponding counter.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/PseudoProbe.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include <unordered_map>

#define DEBUG_TYPE "pseudo-probe-inserter"

using namespace llvm;

namespace {
class PseudoProbeInserter : public MachineFunctionPass {
public:
  static char ID;

  PseudoProbeInserter() : MachineFunctionPass(ID) {
    initializePseudoProbeInserterPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Pseudo Probe Inserter"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    bool Changed = false;
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (MI.isCall()) {
          if (DILocation *DL = MI.getDebugLoc()) {
            auto Value = DL->getDiscriminator();
            if (DILocation::isPseudoProbeDiscriminator(Value)) {
              BuildMI(MBB, MI, DL, TII->get(TargetOpcode::PSEUDO_PROBE))
                  .addImm(getFuncGUID(MF.getFunction().getParent(), DL))
                  .addImm(
                      PseudoProbeDwarfDiscriminator::extractProbeIndex(Value))
                  .addImm(
                      PseudoProbeDwarfDiscriminator::extractProbeType(Value))
                  .addImm(PseudoProbeDwarfDiscriminator::extractProbeAttributes(
                      Value));
              Changed = true;
            }
          }
        }
      }
    }

    return Changed;
  }

private:
  uint64_t getFuncGUID(Module *M, DILocation *DL) {
    auto *SP = DL->getScope()->getSubprogram();
    auto Name = SP->getLinkageName();
    if (Name.empty())
      Name = SP->getName();
    return Function::getGUID(Name);
  }
};
} // namespace

char PseudoProbeInserter::ID = 0;
INITIALIZE_PASS_BEGIN(PseudoProbeInserter, DEBUG_TYPE,
                      "Insert pseudo probe annotations for value profiling",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(PseudoProbeInserter, DEBUG_TYPE,
                    "Insert pseudo probe annotations for value profiling",
                    false, false)

FunctionPass *llvm::createPseudoProbeInserter() {
  return new PseudoProbeInserter();
}
