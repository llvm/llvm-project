//===--- MCPipeline.cpp - MCPipeline ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/MCLinker/MCPipeline.h"

#include "MCLinkerUtils.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;
using namespace llvm::mclinker;

namespace {
class SetMachineFunctionBasePass : public llvm::ImmutablePass {
public:
  static char ID; // Pass identification, replacement for typeid
  SetMachineFunctionBasePass(llvm::MachineModuleInfo &MMI, unsigned Base);

  // Initialization and Finalization
  bool doInitialization(llvm::Module &) override;
  bool doFinalization(llvm::Module &) override;

private:
  llvm::MachineModuleInfo &MMI;
  unsigned Base;
};
} // namespace

char SetMachineFunctionBasePass::ID;

SetMachineFunctionBasePass::SetMachineFunctionBasePass(
    llvm::MachineModuleInfo &MMI, unsigned Base)
    : llvm::ImmutablePass(ID), MMI(MMI), Base(Base) {}

// Initialization and Finalization
bool SetMachineFunctionBasePass::doInitialization(llvm::Module &) {
  setNextFnNum(MMI, Base);
  return false;
}

bool SetMachineFunctionBasePass::doFinalization(llvm::Module &) {
  return false;
}

/// Build a pipeline that does machine specific codgen but stops before
/// AsmPrint. Returns true if failed.
bool llvm::mclinker::addPassesToEmitMC(
    llvm::TargetMachine &TgtMachine, llvm::legacy::PassManagerBase &PM,
    bool DisableVerify,
    llvm::MachineModuleInfoWrapperPass *MMIWP, unsigned NumFnBase) {
  // Targets may override createPassConfig to provide a target-specific
  // subclass.
  TargetPassConfig *PassConfig = TgtMachine.createPassConfig(PM);

  // Set PassConfig options provided by TargetMachine.
  PassConfig->setDisableVerify(DisableVerify);
  PM.add(PassConfig);
  PM.add(MMIWP);

  auto *SetFnBaseP = new SetMachineFunctionBasePass(MMIWP->getMMI(), NumFnBase);
  PM.add(SetFnBaseP);

  if (PassConfig->addISelPasses())
    return true;

  PassConfig->addMachinePasses();
  PassConfig->setInitialized();

  return false;
}

/// Function pass to populate external MCSymbols to other llvm module split's
/// MCContext so that they can be unique across all splits. This uniqueing
/// is required for ORCJIT (not for generating binary .o).
namespace {
class SyncX86SymbolTables : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit SyncX86SymbolTables(SmallVectorImpl<MCInfo *> &);

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  SmallVectorImpl<MCInfo *> &McInfos;
  DenseSet<StringRef> ExternSymbols;

  // Populate MCSymbol to all the MCContexts.
  void populateSymbol(StringRef, const MCSymbolTableValue &, MCContext *);
};
} // namespace

char SyncX86SymbolTables::ID;

SyncX86SymbolTables::SyncX86SymbolTables(SmallVectorImpl<MCInfo *> &McInfos)
    : MachineFunctionPass(ID), McInfos(McInfos) {}

void SyncX86SymbolTables::populateSymbol(StringRef Name,
                                         const llvm::MCSymbolTableValue &Value,
                                         MCContext *SrcCtx) {
  for (MCInfo *McInfo : McInfos) {
    MCContext &CurrCtx = *McInfo->McContext;
    if (&CurrCtx == SrcCtx)
      continue;
    MCSymbolTableEntry &Entry =
        llvm::mclinker::getMCContextSymbolTableEntry(Name, CurrCtx);
    if (!Entry.second.Symbol) {
      Entry.second.Symbol = Value.Symbol;
      Entry.second.NextUniqueID = Value.NextUniqueID;
      Entry.second.Used = Value.Used;
    }
  }
}

bool SyncX86SymbolTables::runOnMachineFunction(MachineFunction &MF) {
  MCContext &Ctx = MF.getContext();
  for (auto &[Name, SymbolEntry] : Ctx.getSymbols()) {
    if (!SymbolEntry.Symbol || !SymbolEntry.Symbol->isExternal() ||
        ExternSymbols.contains(Name))
      continue;
    ExternSymbols.insert(Name);
    populateSymbol(Name, SymbolEntry, &Ctx);
  }
  return false;
}

/// Build a pipeline that does AsmPrint only.
/// Returns true if failed.
bool llvm::mclinker::addPassesToAsmPrint(
    llvm::TargetMachine &TgtMachine, llvm::legacy::PassManagerBase &PM,
    llvm::raw_pwrite_stream &Out, llvm::CodeGenFileType FileType,
    bool DisableVerify, llvm::MachineModuleInfoWrapperPass *MMIWP,
    llvm::SmallVectorImpl<MCInfo *> &McInfos) {
  TargetPassConfig *PassConfig = TgtMachine.createPassConfig(PM);
  if (!PassConfig)
    return true;
  // Set PassConfig options provided by TargetMachine.
  PassConfig->setDisableVerify(DisableVerify);
  PM.add(PassConfig);
  PM.add(MMIWP);
  PassConfig->setInitialized();

  bool Result = TgtMachine.addAsmPrinter(PM, Out, nullptr, FileType,
                                         MMIWP->getMMI().getContext());

  if (TgtMachine.getTargetTriple().isX86())
    PM.add(new SyncX86SymbolTables(McInfos));
  return Result;
}
