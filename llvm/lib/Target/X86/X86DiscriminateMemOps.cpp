//===- X86DiscriminateMemOps.cpp - Unique IDs for Mem Ops -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This pass aids profile-driven cache prefetch insertion by ensuring all
/// instructions that have a memory operand are distinguishible from each other.
///
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/Transforms/IPO/SampleProfile.h"
using namespace llvm;

namespace {

using Location = std::pair<StringRef, unsigned>;

Location diToLocation(const DILocation *Loc) {
  return std::make_pair(Loc->getFilename(), Loc->getLine());
}

/// Ensure each instruction having a memory operand has a distinct <LineNumber,
/// Discriminator> pair.
void updateDebugInfo(MachineInstr *MI, const DILocation *Loc) {
  DebugLoc DL(Loc);
  MI->setDebugLoc(DL);
}

class X86DiscriminateMemOps : public MachineFunctionPass {
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override {
    return "X86 Discriminate Memory Operands";
  }

public:
  static char ID;

  /// Default construct and initialize the pass.
  X86DiscriminateMemOps();
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//            Implementation
//===----------------------------------------------------------------------===//

char X86DiscriminateMemOps::ID = 0;

/// Default construct and initialize the pass.
X86DiscriminateMemOps::X86DiscriminateMemOps() : MachineFunctionPass(ID) {}

bool X86DiscriminateMemOps::runOnMachineFunction(MachineFunction &MF) {
  DISubprogram *FDI = MF.getFunction().getSubprogram();
  if (!FDI || !FDI->getUnit()->getDebugInfoForProfiling())
    return false;

  // Have a default DILocation, if we find instructions with memops that don't
  // have any debug info.
  const DILocation *ReferenceDI =
      DILocation::get(FDI->getContext(), FDI->getLine(), 0, FDI);

  DenseMap<Location, unsigned> MemOpDiscriminators;
  MemOpDiscriminators[diToLocation(ReferenceDI)] = 0;

  // Figure out the largest discriminator issued for each Location. When we
  // issue new discriminators, we can thus avoid issuing discriminators
  // belonging to instructions that don't have memops. This isn't a requirement
  // for the goals of this pass, however, it avoids unnecessary ambiguity.
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      const auto &DI = MI.getDebugLoc();
      if (!DI)
        continue;
      Location Loc = diToLocation(DI);
      MemOpDiscriminators[Loc] =
          std::max(MemOpDiscriminators[Loc], DI->getBaseDiscriminator());
    }
  }

  // Keep track of the discriminators seen at each Location. If an instruction's
  // DebugInfo has a Location and discriminator we've already seen, replace its
  // discriminator with a new one, to guarantee uniqueness.
  DenseMap<Location, DenseSet<unsigned>> Seen;

  bool Changed = false;
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (X86II::getMemoryOperandNo(MI.getDesc().TSFlags) < 0)
        continue;
      const DILocation *DI = MI.getDebugLoc();
      if (!DI) {
        DI = ReferenceDI;
      }
      DenseSet<unsigned> &Set = Seen[diToLocation(DI)];
      std::pair<DenseSet<unsigned>::iterator, bool> P =
          Set.insert(DI->getBaseDiscriminator());
      if (!P.second) {
        DI = DI->setBaseDiscriminator(++MemOpDiscriminators[diToLocation(DI)]);
        updateDebugInfo(&MI, DI);
        Changed = true;
        *P.first = DI->getBaseDiscriminator();
      }

      // Bump the reference DI to avoid cramming discriminators on line 0.
      // FIXME(mtrofin): pin ReferenceDI on blocks or first instruction with DI
      // in a block. It's more consistent than just relying on the last memop
      // instruction we happened to see.
      ReferenceDI = DI;
    }
  }
  return Changed;
}

FunctionPass *llvm::createX86DiscriminateMemOpsPass() {
  return new X86DiscriminateMemOps();
}
