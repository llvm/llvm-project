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

#include "llvm/CodeGen/PseudoProbeInserter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PseudoProbe.h"
#include "llvm/InitializePasses.h"
#include "llvm/ProfileData/SampleProf.h"

#define DEBUG_TYPE "pseudo-probe-inserter"

using namespace llvm;

namespace {

// A real instruction is a non-meta, non-pseudo instruction. Calls (including
// call pseudos like tail-call returns) are also treated as real because they
// expand to a branch and any preceding call probe must be preserved.
static bool isCallOrRealInstruction(const MachineInstr &MI) {
  return MI.isCall() || (!MI.isPseudo() && !MI.isMetaInstruction());
}

static uint64_t getFuncGUID(Module *M, DILocation *DL) {
  auto Name = DL->getSubprogramLinkageName();
  // CoroSplit Pass will change the debug info with suffixes i.e. `.resume`,
  // `.destroy`, `.cleanup`. Strip these suffixes to make the GUID consistent
  // with the pseudo probe
  Name = SampleProfileAttributes::getCanonicalCoroFnName(Name);
  return Function::getGUIDAssumingExternalLinkage(Name);
}

static bool insertPseudoProbesInMF(MachineFunction &MF) {
  if (!MF.getFunction().getParent()->getNamedMetadata(
          PseudoProbeDescMetadataName))
    return false;
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    MachineInstr *FirstInstr = nullptr;
    for (MachineInstr &MI : MBB) {
      // Pseudo instructions like TCRETURNdi results in a branch instruction
      // and the call probe for that tail call should be preserved.
      if (isCallOrRealInstruction(MI))
        FirstInstr = &MI;
      if (MI.isCall()) {
        if (DILocation *DL = MI.getDebugLoc()) {
          auto Value = DL->getDiscriminator();
          if (DILocation::isPseudoProbeDiscriminator(Value)) {
            BuildMI(MBB, MI, DL, TII->get(TargetOpcode::PSEUDO_PROBE))
                .addImm(getFuncGUID(MF.getFunction().getParent(), DL))
                .addImm(PseudoProbeDwarfDiscriminator::extractProbeIndex(Value))
                .addImm(PseudoProbeDwarfDiscriminator::extractProbeType(Value))
                .addImm(PseudoProbeDwarfDiscriminator::extractProbeAttributes(
                    Value));
            Changed = true;
          }
        }
      }
    }

    // Walk the block backwards, move PSEUDO_PROBE before the first real
    // instruction to fix out-of-order probes. There is a problem with probes
    // as the terminator of the block. During the offline counts processing,
    // the samples collected on the first physical instruction following a
    // probe will be counted towards the probe. This logically equals to
    // treating the instruction next to a probe as if it is from the same
    // block of the probe. This is accurate most of the time unless the
    // instruction can be reached from multiple flows, which means it actually
    // starts a new block. Samples collected on such probes may cause
    // imprecision with the counts inference algorithm. Fortunately, if
    // there are still other native instructions preceding the probe we can
    // use them as a place holder to collect samples for the probe.
    if (FirstInstr) {
      auto MII = MBB.rbegin();
      while (MII != MBB.rend()) {
        // Skip all pseudo probes followed by a real instruction since they
        // are not dangling. Treat call pseudos (e.g. tail-call returns)
        // as real instructions to keep this consistent with the forward
        // scan above.
        if (isCallOrRealInstruction(*MII))
          break;
        auto Cur = MII++;
        if (Cur->getOpcode() != TargetOpcode::PSEUDO_PROBE)
          continue;
        // Move the dangling probe before FirstInstr.
        auto *ProbeInstr = &*Cur;
        MBB.remove(ProbeInstr);
        MBB.insert(FirstInstr, ProbeInstr);
        Changed = true;
      }
    } else {
      // Probes not surrounded by any real instructions in the same block are
      // called dangling probes. Since there's no good way to pick up a sample
      // collection point for dangling probes at compile time, they are being
      // removed so that the profile correlation tool will not report any
      // samples collected for them and it's up to the counts inference tool
      // to get them a reasonable count.
      SmallVector<MachineInstr *, 4> ToBeRemoved;
      for (MachineInstr &MI : MBB) {
        if (MI.isPseudoProbe())
          ToBeRemoved.push_back(&MI);
      }

      for (auto *MI : ToBeRemoved)
        MI->eraseFromParent();

      Changed |= !ToBeRemoved.empty();
    }
  }

  return Changed;
}

class PseudoProbeInserterLegacy : public MachineFunctionPass {
public:
  static char ID;

  PseudoProbeInserterLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "Pseudo Probe Inserter"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    return insertPseudoProbesInMF(MF);
  }
};
} // namespace

PreservedAnalyses
PseudoProbeInserterPass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &MFAM) {
  return insertPseudoProbesInMF(MF) ? getMachineFunctionPassPreservedAnalyses()
                                          .preserveSet<CFGAnalyses>()
                                    : PreservedAnalyses::all();
}

char PseudoProbeInserterLegacy::ID = 0;
INITIALIZE_PASS_BEGIN(PseudoProbeInserterLegacy, DEBUG_TYPE,
                      "Insert pseudo probe annotations for value profiling",
                      false, false)
INITIALIZE_PASS_END(PseudoProbeInserterLegacy, DEBUG_TYPE,
                    "Insert pseudo probe annotations for value profiling",
                    false, false)

FunctionPass *llvm::createPseudoProbeInserterLegacyPass() {
  return new PseudoProbeInserterLegacy();
}
