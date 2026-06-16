//===-- MachineFunctionPass.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the MachineFunctionPass members.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/LazyBranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/CodeGen/DroppedVariableStatsMIR.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineStableHash.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/PrintPasses.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>

using namespace llvm;
using namespace ore;

static cl::opt<bool> DroppedVarStatsMIR(
    "dropped-variable-stats-mir", cl::Hidden,
    cl::desc("Dump dropped debug variables stats for MIR passes"),
    cl::init(false));

static bool isDiffChangePrinter(ChangePrinter Printer) {
  return Printer == ChangePrinter::DiffQuiet ||
         Printer == ChangePrinter::DiffVerbose ||
         Printer == ChangePrinter::ColourDiffQuiet ||
         Printer == ChangePrinter::ColourDiffVerbose;
}

namespace {

struct MachineBasicBlockChangeHash {
  std::string Key;
  stable_hash Hash = 0;

  bool operator==(const MachineBasicBlockChangeHash &That) const {
    return Key == That.Key && Hash == That.Hash;
  }
};

struct MachineFunctionChangeHash {
  stable_hash Hash = 0;
  SmallVector<MachineBasicBlockChangeHash> Blocks;

  bool operator==(const MachineFunctionChangeHash &That) const {
    return Hash == That.Hash;
  }
};

// Legacy MachineFunctionPass handles -print-changed outside the new pass
// instrumentation callbacks. Keep its stateful hash cache here, keyed by live
// MachineFunction objects from the current module, and clear it when the module
// changes to avoid stale pointer reuse.
DenseMap<const MachineFunction *, MachineFunctionChangeHash>
    StatefulMachineFunctionHashes;
const Module *StatefulMachineFunctionHashModule = nullptr;

void setStatefulMachineFunctionHashModule(const Module *M) {
  if (StatefulMachineFunctionHashModule == M)
    return;
  StatefulMachineFunctionHashes.clear();
  StatefulMachineFunctionHashModule = M;
}

// MachineBasicBlock pointers in MachineFunctionStableHashInfo are only valid
// for the current snapshot. Across passes, hash-bb matches blocks by a derived
// key: the block name when present, otherwise the block's ordinal position in
// the function. This does not suppress transformed or duplicated blocks;
// changed and newly-created blocks are still printed. The limitation is
// association: passes that delete/recreate, duplicate, or reorder unnamed
// blocks can appear as added/deleted or position-matched block changes instead
// of being matched to a previous logical block. TODO: add a more stable block
// tracking key.
std::string getMachineBasicBlockChangeKey(
    const MachineBasicBlock &MBB,
    const DenseMap<const MachineBasicBlock *, unsigned> *Numbers = nullptr) {
  if (MBB.hasName())
    return MBB.getName().str();
  if (Numbers)
    return formatv("{0}", Numbers->lookup(&MBB)).str();
  return formatv("{0}", MBB.getNumber()).str();
}

MachineFunctionChangeHash
collectMachineFunctionChangeHash(const MachineFunction &MF) {
  ChangePrinterHashMode HashMode = getPrintChangedHashMode();
  MachineFunctionChangeHash FunctionHash;

  if (HashMode == ChangePrinterHashMode::Function ||
      (HashMode == ChangePrinterHashMode::BasicBlock && MF.size() <= 1)) {
    FunctionHash.Hash = stableHashValueForChangePrinter(MF);
    return FunctionHash;
  }

  MachineFunctionStableHashInfo Details =
      stableHashValueWithDetailsForChangePrinter(MF);
  FunctionHash.Hash = Details.Hash;
  DenseMap<const MachineBasicBlock *, unsigned> BlockNumbers;
  unsigned BlockNumber = 0;
  for (const MachineBasicBlock &MBB : MF)
    BlockNumbers.try_emplace(&MBB, BlockNumber++);
  for (const MachineBasicBlockStableHashInfo &BlockInfo : Details.Blocks) {
    MachineBasicBlockChangeHash BlockHash;
    BlockHash.Key =
        getMachineBasicBlockChangeKey(*BlockInfo.MBB, &BlockNumbers);
    BlockHash.Hash = BlockInfo.Hash;
    FunctionHash.Blocks.push_back(std::move(BlockHash));
  }

  return FunctionHash;
}

StringMap<const MachineBasicBlock *>
collectMachineBasicBlockMap(const MachineFunction &MF) {
  StringMap<const MachineBasicBlock *> Blocks;
  DenseMap<const MachineBasicBlock *, unsigned> BlockNumbers;
  unsigned BlockNumber = 0;
  for (const MachineBasicBlock &MBB : MF)
    BlockNumbers.try_emplace(&MBB, BlockNumber++);
  for (const MachineBasicBlock &MBB : MF)
    Blocks[getMachineBasicBlockChangeKey(MBB, &BlockNumbers)] = &MBB;
  return Blocks;
}

class MachineChangePrinterPrintContext {
  const MachineFunction &MF;
  std::optional<ModuleSlotTracker> MST;

  ModuleSlotTracker &getModuleSlotTracker() {
    if (!MST) {
      MST.emplace(MF.getFunction().getParent());
      MST->incorporateFunction(MF.getFunction());
    }
    return *MST;
  }

public:
  explicit MachineChangePrinterPrintContext(const MachineFunction &MF)
      : MF(MF) {}

  void printBasicBlock(raw_ostream &OS, const MachineBasicBlock &MBB) {
    MBB.print(OS, getModuleSlotTracker(), /*Indexes=*/nullptr,
              /*IsStandalone=*/true);
  }
};

void printMachineBasicBlockChangeHeader(raw_ostream &OS,
                                        const MachineFunction &MF,
                                        StringRef BlockKey, StringRef Change,
                                        const MachineBasicBlock *MBB) {
  OS << "*** BasicBlock " << MF.getName() << ":";
  if (MBB && MBB->hasName())
    OS << MBB->getName();
  else
    OS << BlockKey;
  OS << " " << Change << " ***\n";
}

bool printMachineBasicBlockHashChanges(
    raw_ostream &OS, const MachineFunction &MF,
    const MachineFunctionChangeHash &Before,
    const MachineFunctionChangeHash &After,
    MachineChangePrinterPrintContext &PrintCtx) {
  bool Printed = false;
  StringMap<const MachineBasicBlock *> CurrentBlocks =
      collectMachineBasicBlockMap(MF);
  DenseMap<StringRef, const MachineBasicBlockChangeHash *> BeforeBlocks;
  DenseMap<StringRef, const MachineBasicBlockChangeHash *> AfterBlocks;
  for (const MachineBasicBlockChangeHash &BeforeBlock : Before.Blocks)
    BeforeBlocks.try_emplace(BeforeBlock.Key, &BeforeBlock);
  for (const MachineBasicBlockChangeHash &AfterBlock : After.Blocks)
    AfterBlocks.try_emplace(AfterBlock.Key, &AfterBlock);

  unsigned ChangedBlockCount = 0;
  for (const MachineBasicBlockChangeHash &AfterBlock : After.Blocks) {
    const MachineBasicBlockChangeHash *BeforeBlock =
        BeforeBlocks.lookup(AfterBlock.Key);
    if (!BeforeBlock || BeforeBlock->Hash != AfterBlock.Hash)
      ++ChangedBlockCount;
  }
  for (const MachineBasicBlockChangeHash &BeforeBlock : Before.Blocks)
    if (!AfterBlocks.contains(BeforeBlock.Key))
      ++ChangedBlockCount;
  if (ChangedBlockCount == 0)
    return false;

  unsigned UnchangedBlockCount = 0;
  for (const MachineBasicBlockChangeHash &AfterBlock : After.Blocks) {
    const MachineBasicBlockChangeHash *BeforeBlock =
        BeforeBlocks.lookup(AfterBlock.Key);
    if (BeforeBlock && BeforeBlock->Hash == AfterBlock.Hash) {
      ++UnchangedBlockCount;
      continue;
    }
    const MachineBasicBlock *MBB = CurrentBlocks.lookup(AfterBlock.Key);
    printMachineBasicBlockChangeHeader(OS, MF, AfterBlock.Key, "changed", MBB);
    if (MBB)
      PrintCtx.printBasicBlock(OS, *MBB);
    Printed = true;
  }
  if (UnchangedBlockCount != 0)
    OS << "; " << UnchangedBlockCount << " unchanged basic blocks omitted\n";
  for (const MachineBasicBlockChangeHash &BeforeBlock : Before.Blocks) {
    if (AfterBlocks.contains(BeforeBlock.Key))
      continue;
    printMachineBasicBlockChangeHeader(OS, MF, BeforeBlock.Key, "deleted",
                                       /*MBB=*/nullptr);
    Printed = true;
  }
  return Printed;
}

void printMachineFunctionHashChanges(raw_ostream &OS, const MachineFunction &MF,
                                     const MachineFunctionChangeHash &Before,
                                     const MachineFunctionChangeHash &After) {
  bool Printed = false;
  MachineChangePrinterPrintContext PrintCtx(MF);
  switch (getPrintChangedHashMode()) {
  case ChangePrinterHashMode::Function:
    MF.print(OS);
    return;
  case ChangePrinterHashMode::BasicBlock:
    if (Before.Blocks.empty() || After.Blocks.empty()) {
      MF.print(OS);
      return;
    }
    Printed =
        printMachineBasicBlockHashChanges(OS, MF, Before, After, PrintCtx);
    break;
  case ChangePrinterHashMode::None:
    llvm_unreachable("expected hash mode");
  }

  if (!Printed && Before.Hash != After.Hash)
    MF.print(OS);
}

} // namespace

Pass *MachineFunctionPass::createPrinterPass(raw_ostream &O,
                                             const std::string &Banner) const {
  return createMachineFunctionPrinterPass(O, Banner);
}

bool MachineFunctionPass::runOnFunction(Function &F) {
  // Do not codegen any 'available_externally' functions at all, they have
  // definitions outside the translation unit.
  if (F.hasAvailableExternallyLinkage())
    return false;

  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
  MachineFunction &MF = MMI.getOrCreateMachineFunction(F);

  MachineFunctionProperties &MFProps = MF.getProperties();

#ifndef NDEBUG
  if (!MFProps.verifyRequiredProperties(RequiredProperties)) {
    errs() << "MachineFunctionProperties required by " << getPassName()
           << " pass are not met by function " << F.getName() << ".\n"
           << "Required properties: ";
    RequiredProperties.print(errs());
    errs() << "\nCurrent properties: ";
    MFProps.print(errs());
    errs() << "\n";
    reportFatalUsageError("MachineFunctionProperties check failed");
  }
#endif
  // Collect the MI count of the function before the pass.
  unsigned CountBefore, CountAfter;

  // Check if the user asked for size remarks.
  bool ShouldEmitSizeRemarks =
      F.getParent()->shouldEmitInstrCountChangedRemark();

  // If we want size remarks, collect the number of MachineInstrs in our
  // MachineFunction before the pass runs.
  if (ShouldEmitSizeRemarks)
    CountBefore = MF.getInstructionCount();

  // Hash-based --print-changed is handled here so the legacy pass manager does
  // not need to materialize full MIR text before and after every machine pass.
  // Text and diff modes are handled by FunctionPass::printIRUnit().
  MachineFunctionChangeHash BeforeHash, AfterHash;
  StringRef PassID;
  if (PrintChanged != ChangePrinter::None) {
    if (const PassInfo *PI = Pass::lookupPassInfo(getPassID()))
      PassID = PI->getPassArgument();
  }
  const bool IsInterestingPass = isPassInPrintList(PassID);
  const bool UseHashComparison =
      shouldUsePrintChangedHash() && !isDiffChangePrinter(PrintChanged);
  const bool UseStatefulHash = UseHashComparison && isFilterPassesEmpty();
  const bool ShouldTrackChangedHash =
      UseStatefulHash && isFunctionInPrintList(MF.getName());
  const bool ShouldPrintChangedHash = UseHashComparison && IsInterestingPass &&
                                      isFunctionInPrintList(MF.getName());
  if (UseStatefulHash)
    setStatefulMachineFunctionHashModule(F.getParent());
  if (ShouldPrintChangedHash || ShouldTrackChangedHash) {
    if (UseStatefulHash) {
      auto It = StatefulMachineFunctionHashes.find(&MF);
      if (It == StatefulMachineFunctionHashes.end()) {
        BeforeHash = collectMachineFunctionChangeHash(MF);
        MachineFunctionChangeHash CachedHash = BeforeHash;
        StatefulMachineFunctionHashes[&MF] = std::move(CachedHash);
      } else {
        BeforeHash = It->second;
      }
    } else {
      BeforeHash = collectMachineFunctionChangeHash(MF);
    }
  }
  MFProps.reset(ClearedProperties);

  bool RV;
  if (DroppedVarStatsMIR) {
    DroppedVariableStatsMIR DroppedVarStatsMF;
    auto PassName = getPassName();
    DroppedVarStatsMF.runBeforePass(PassName, &MF);
    RV = runOnMachineFunction(MF);
    DroppedVarStatsMF.runAfterPass(PassName, &MF);
  } else {
    RV = runOnMachineFunction(MF);
  }

  if (ShouldEmitSizeRemarks) {
    // We wanted size remarks. Check if there was a change to the number of
    // MachineInstrs in the module. Emit a remark if there was a change.
    CountAfter = MF.getInstructionCount();
    if (CountBefore != CountAfter) {
      MachineOptimizationRemarkEmitter MORE(MF, nullptr);
      MORE.emit([&]() {
        int64_t Delta = static_cast<int64_t>(CountAfter) -
                        static_cast<int64_t>(CountBefore);
        MachineOptimizationRemarkAnalysis R("size-info", "FunctionMISizeChange",
                                            MF.getFunction().getSubprogram(),
                                            &MF.front());
        R << NV("Pass", getPassName())
          << ": Function: " << NV("Function", F.getName()) << ": "
          << "MI Instruction count changed from "
          << NV("MIInstrsBefore", CountBefore) << " to "
          << NV("MIInstrsAfter", CountAfter)
          << "; Delta: " << NV("Delta", Delta);
        return R;
      });
    }
  }

  MFProps.set(SetProperties);

  // For --print-changed, print if the MF has changed. Modes other than
  // quiet/verbose are unimplemented and treated the same as 'quiet'.
  if (UseHashComparison && (ShouldPrintChangedHash || !IsInterestingPass ||
                            ShouldTrackChangedHash)) {
    if (ShouldPrintChangedHash || ShouldTrackChangedHash) {
      AfterHash = collectMachineFunctionChangeHash(MF);
    }
    if (ShouldPrintChangedHash && !(BeforeHash == AfterHash)) {
      errs() << ("*** IR Dump After " + getPassName() + " (" + PassID +
                 ") on " + MF.getName() + " ***\n");
      switch (PrintChanged) {
      case ChangePrinter::None:
        llvm_unreachable("");
      case ChangePrinter::Quiet:
      case ChangePrinter::Verbose:
      case ChangePrinter::DotCfgQuiet:   // unimplemented
      case ChangePrinter::DotCfgVerbose: // unimplemented
        if (getPrintChangedHashMode() != ChangePrinterHashMode::Function)
          printMachineFunctionHashChanges(errs(), MF, BeforeHash, AfterHash);
        else
          MF.print(errs());
        break;
      case ChangePrinter::DiffQuiet:
      case ChangePrinter::DiffVerbose:
      case ChangePrinter::ColourDiffQuiet:
      case ChangePrinter::ColourDiffVerbose:
        llvm_unreachable("hash mode is not supported with diff output");
        break;
      }
    } else if (llvm::is_contained({ChangePrinter::Verbose,
                                   ChangePrinter::DiffVerbose,
                                   ChangePrinter::ColourDiffVerbose},
                                  PrintChanged)) {
      const char *Reason =
          IsInterestingPass ? " omitted because no change" : " filtered out";
      errs() << "*** IR Dump After " << getPassName();
      if (!PassID.empty())
        errs() << " (" << PassID << ")";
      errs() << " on " << MF.getName() + Reason + " ***\n";
    }
    if (ShouldTrackChangedHash && IsInterestingPass) {
      MachineFunctionChangeHash CachedHash = AfterHash;
      StatefulMachineFunctionHashes[&MF] = std::move(CachedHash);
    } else if (ShouldTrackChangedHash) {
      StatefulMachineFunctionHashes.erase(&MF);
    }
  }
  return RV;
}

bool MachineFunctionPass::printIRUnit(raw_ostream &OS, Function &F) {
  // Hash mode is handled in runOnFunction() to avoid full before/after MIR
  // printing in the legacy pass manager.
  if (shouldUsePrintChangedHash())
    return false;

  // available_externally functions are not codegen'd (see runOnFunction).
  if (F.hasAvailableExternallyLinkage())
    return false;
  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
  MMI.getOrCreateMachineFunction(F).print(OS);
  return true;
}

bool MachineFunctionPass::doFinalization(Module &M) {
  StatefulMachineFunctionHashes.clear();
  if (StatefulMachineFunctionHashModule == &M)
    StatefulMachineFunctionHashModule = nullptr;
  return false;
}

void MachineFunctionPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineModuleInfoWrapperPass>();
  AU.addPreserved<MachineModuleInfoWrapperPass>();

  // MachineFunctionPass preserves all LLVM IR passes, but there's no
  // high-level way to express this. Instead, just list a bunch of
  // passes explicitly. This does not include setPreservesCFG,
  // because CodeGen overloads that to mean preserving the MachineBasicBlock
  // CFG in addition to the LLVM IR CFG.
  AU.addPreserved<BasicAAWrapperPass>();
  AU.addPreserved<DominanceFrontierWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<PostDominatorTreeWrapperPass>();
  AU.addPreserved<BranchProbabilityInfoWrapperPass>();
  AU.addPreserved<LazyBranchProbabilityInfoPass>();
  AU.addPreserved<LazyBlockFrequencyInfoPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
  AU.addPreserved<IVUsersWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();
  AU.addPreserved<MemoryDependenceWrapperPass>();
  AU.addPreserved<ScalarEvolutionWrapperPass>();
  AU.addPreserved<SCEVAAWrapperPass>();

  FunctionPass::getAnalysisUsage(AU);
}
