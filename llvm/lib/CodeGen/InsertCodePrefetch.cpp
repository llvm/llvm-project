//===-- InsertCodePrefetch.cpp ---=========-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Prefetch insertion pass implementation.
//===----------------------------------------------------------------------===//
/// Prefetch insertion pass.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/BasicBlockSectionUtils.h"
#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
#define DEBUG_TYPE "prefetchinsertion"

namespace {
class InsertCodePrefetch : public MachineFunctionPass {
public:
  static char ID;

  InsertCodePrefetch() : MachineFunctionPass(ID) {
    initializeInsertCodePrefetchPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "X86 Cide Prefetch Inserter Pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Identify basic blocks that need separate sections and prepare to emit them
  /// accordingly.
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//            Implementation
//===----------------------------------------------------------------------===//

char InsertCodePrefetch::ID = 0;
INITIALIZE_PASS_BEGIN(InsertCodePrefetch, DEBUG_TYPE, "Reads prefetch", true,
                      false)
INITIALIZE_PASS_DEPENDENCY(BasicBlockSectionsProfileReaderWrapperPass)
INITIALIZE_PASS_END(InsertCodePrefetch, DEBUG_TYPE, "Reads prefetch", true,
                    false)

bool InsertCodePrefetch::runOnMachineFunction(MachineFunction &MF) {
  assert(MF.getTarget().getBBSectionsType() == BasicBlockSection::List &&
         "BB Sections list not enabled!");
  if (hasInstrProfHashMismatch(MF))
    return false;
  SmallVector<BBPosition> PrefetchTargets =
      getAnalysis<BasicBlockSectionsProfileReaderWrapperPass>()
          .getPrefetchTargetsForFunction(MF.getName());
  DenseMap<UniqueBBID, SmallVector<unsigned>> PrefetchTargetsByBBID;
  for (const auto &Target : PrefetchTargets)
    PrefetchTargetsByBBID[Target.BBID].push_back(Target.CallsiteIndex);
  for (auto &MBB : MF) {
    auto R = PrefetchTargetsByBBID.find(*MBB.getBBID());
    if (R == PrefetchTargetsByBBID.end())
      continue;
    MBB.setPrefetchTargetIndexes(R->second);
  }

  return false;
}

void InsertCodePrefetch::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<BasicBlockSectionsProfileReaderWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

MachineFunctionPass *llvm::createInsertCodePrefetchPass() {
  return new InsertCodePrefetch();
}
