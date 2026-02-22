//===-- InsertCodePrefetch.cpp ---=========--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Code Prefetch Insertion Pass.
//===----------------------------------------------------------------------===//
/// This pass inserts code prefetch instructions according to the prefetch
/// directives in the basic block section profile. The target of a prefetch can
/// be the beginning of any dynamic basic block, that is the beginning of a
/// machine basic block, or immediately after a callsite. A global symbol is
/// emitted at the position of the target so it can be addressed from the
/// prefetch instruction from any module.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/BasicBlockSectionUtils.h"
#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"

using namespace llvm;
#define DEBUG_TYPE "insert-code-prefetch"

namespace {
class InsertCodePrefetch : public MachineFunctionPass {
public:
  static char ID;

  InsertCodePrefetch() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "Code Prefetch Inserter Pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  // Sets prefetch targets based on the bb section profile.
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//            Implementation
//===----------------------------------------------------------------------===//

char InsertCodePrefetch::ID = 0;
INITIALIZE_PASS_BEGIN(InsertCodePrefetch, DEBUG_TYPE, "Code prefetch insertion",
                      true, false)
INITIALIZE_PASS_DEPENDENCY(BasicBlockSectionsProfileReaderWrapperPass)
INITIALIZE_PASS_END(InsertCodePrefetch, DEBUG_TYPE, "Code prefetch insertion",
                    true, false)

bool InsertCodePrefetch::runOnMachineFunction(MachineFunction &MF) {
  assert(MF.getTarget().getBBSectionsType() == BasicBlockSection::List &&
         "BB Sections list not enabled!");
  if (hasInstrProfHashMismatch(MF))
    return false;
  // Set each block's prefetch targets so AsmPrinter can emit a special symbol
  // there.
  SmallVector<CallsiteID> PrefetchTargets =
      getAnalysis<BasicBlockSectionsProfileReaderWrapperPass>()
          .getPrefetchTargetsForFunction(MF.getName());
  DenseMap<UniqueBBID, SmallVector<unsigned>> PrefetchTargetsByBBID;
  for (const auto &Target : PrefetchTargets)
    PrefetchTargetsByBBID[Target.BBID].push_back(Target.CallsiteIndex);
  // Sort and uniquify the callsite indices for every block.
  for (auto &[K, V] : PrefetchTargetsByBBID) {
    llvm::sort(V);
    V.erase(llvm::unique(V), V.end());
  }
  for (auto &MBB : MF) {
    auto R = PrefetchTargetsByBBID.find(*MBB.getBBID());
    if (R == PrefetchTargetsByBBID.end())
      continue;
    MBB.setPrefetchTargetCallsiteIndexes(R->second);
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
