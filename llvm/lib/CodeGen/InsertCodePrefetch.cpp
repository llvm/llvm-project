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
/// prefetch instruction from any module. In order to insert prefetch hints,
/// `TargetInstrInfo::insertCodePrefetchInstr` must be implemented by the
/// target.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/InsertCodePrefetch.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/BasicBlockSectionUtils.h"
#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCSymbolELF.h"

using namespace llvm;
#define DEBUG_TYPE "insert-code-prefetch"

namespace llvm {
SmallString<128> getPrefetchTargetSymbolName(StringRef FunctionName,
                                             const UniqueBBID &BBID,
                                             unsigned CallsiteIndex) {
  SmallString<128> R("__llvm_prefetch_target_");
  R += FunctionName;
  R += "_";
  R += utostr(BBID.BaseID);
  R += "_";
  R += utostr(CallsiteIndex);
  return R;
}
} // namespace llvm

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

static void setPrefetchTargets(MachineFunction &MF,
                               const SmallVector<CallsiteID> &PrefetchTargets) {
  // Set each block's prefetch targets so AsmPrinter can emit a special symbol
  // there.
  DenseMap<UniqueBBID, SmallVector<unsigned>> PrefetchTargetsByBBID;
  for (const auto &Target : PrefetchTargets)
    PrefetchTargetsByBBID[Target.BBID].push_back(Target.CallsiteIndex);
  // Sort and uniquify the callsite indices for every block.
  for (auto &[K, V] : PrefetchTargetsByBBID) {
    llvm::sort(V);
    V.erase(llvm::unique(V), V.end());
  }
  MF.setPrefetchTargets(PrefetchTargetsByBBID);
}

static void
insertPrefetchHints(MachineFunction &MF,
                    const SmallVector<PrefetchHint> &PrefetchHints) {
  bool IsELF = MF.getTarget().getTargetTriple().isOSBinFormatELF();
  const Module *M = MF.getFunction().getParent();
  DenseMap<UniqueBBID, SmallVector<PrefetchHint>> PrefetchHintsBySiteBBID;
  for (const auto &H : PrefetchHints)
    PrefetchHintsBySiteBBID[H.SiteID.BBID].push_back(H);
  // Sort prefetch hints by their callsite index so we can insert them by one
  // pass over the block's instructions.
  for (auto &[SiteBBID, Hints] : PrefetchHintsBySiteBBID) {
    llvm::sort(Hints, [](const PrefetchHint &H1, const PrefetchHint &H2) {
      return H1.SiteID.CallsiteIndex < H2.SiteID.CallsiteIndex;
    });
  }
  auto PtrTy =
      PointerType::getUnqual(MF.getFunction().getParent()->getContext());
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  for (auto &BB : MF) {
    auto It = PrefetchHintsBySiteBBID.find(*BB.getBBID());
    if (It == PrefetchHintsBySiteBBID.end())
      continue;
    const auto &BBHints = It->second;
    unsigned NumCallsInBB = 0;
    auto InstrIt = BB.begin();
    for (auto HintIt = BBHints.begin(); HintIt != BBHints.end();) {
      bool TargetFunctionDefined = false;
      if (Function *TargetFunction = M->getFunction(HintIt->TargetFunction))
        TargetFunctionDefined = !TargetFunction->isDeclaration();
      auto NextInstrIt = InstrIt == BB.end() ? BB.end() : std::next(InstrIt);
      // Insert all the prefetch hints which must be placed after this call (or
      // at the beginning of the block if `NumCallsInBB` is zero.
      while (HintIt != BBHints.end() &&
             NumCallsInBB >= HintIt->SiteID.CallsiteIndex) {
        auto TargetSymbolName = getPrefetchTargetSymbolName(HintIt->TargetFunction,
                                        HintIt->TargetID.BBID,
                                        HintIt->TargetID.CallsiteIndex);
        auto *GV = MF.getFunction().getParent()->getOrInsertGlobal(TargetSymbolName,
            PtrTy);
        MachineInstr *PrefetchInstr = TII->insertCodePrefetchInstr(BB, InstrIt, GV);
        if (!TargetFunctionDefined && IsELF) {
          // If the target function is not defined in this module, we guard
          // against undefined prefetch target symbol by emitting a fallback
          // symbol with weak linkage right after the prefetch instruction. If
          // there is no strong symbol, the fallback will be used and we
          // prefetch the next address.
          MCSymbolELF *WeakFallbackSym = static_cast<MCSymbolELF *>(MF.getContext().getOrCreateSymbol(TargetSymbolName));
          WeakFallbackSym->setIsWeakref();
          PrefetchInstr->setPostInstrSymbol(MF, WeakFallbackSym);
        }
        ++HintIt;
      }
      if (InstrIt == BB.end())
        break;
      if (InstrIt->isCall())
        ++NumCallsInBB;
      InstrIt = NextInstrIt;
    }
  }
}

bool InsertCodePrefetch::runOnMachineFunction(MachineFunction &MF) {
  assert(MF.getTarget().getBBSectionsType() == BasicBlockSection::List &&
         "BB Sections list not enabled!");
  if (hasInstrProfHashMismatch(MF))
    return false;

  auto &ProfileReader =
      getAnalysis<BasicBlockSectionsProfileReaderWrapperPass>();
  setPrefetchTargets(MF,
                     ProfileReader.getPrefetchTargetsForFunction(MF.getName()));
  insertPrefetchHints(MF,
                      ProfileReader.getPrefetchHintsForFunction(MF.getName()));

  return true;
}

void InsertCodePrefetch::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<BasicBlockSectionsProfileReaderWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

MachineFunctionPass *llvm::createInsertCodePrefetchPass() {
  return new InsertCodePrefetch();
}
