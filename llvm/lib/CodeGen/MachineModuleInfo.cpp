//===-- llvm/CodeGen/MachineModuleInfo.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolXCOFF.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::dwarf;

// Out of line virtual method.
MachineModuleInfoImpl::~MachineModuleInfoImpl() = default;

namespace llvm {

class MMIAddrLabelMapCallbackPtr final : CallbackVH {
  MMIAddrLabelMap *Map = nullptr;

public:
  MMIAddrLabelMapCallbackPtr() = default;
  MMIAddrLabelMapCallbackPtr(Value *V) : CallbackVH(V) {}

  void setPtr(BasicBlock *BB) {
    ValueHandleBase::operator=(BB);
  }

  void setMap(MMIAddrLabelMap *map) { Map = map; }

  void deleted() override;
  void allUsesReplacedWith(Value *V2) override;
};

class MMIAddrLabelMap {
  MCContext &Context;
  struct AddrLabelSymEntry {
    /// The symbols for the label.
    TinyPtrVector<MCSymbol *> Symbols;

    Function *Fn;   // The containing function of the BasicBlock.
    unsigned Index; // The index in BBCallbacks for the BasicBlock.
  };

  DenseMap<AssertingVH<BasicBlock>, AddrLabelSymEntry> AddrLabelSymbols;

  /// Callbacks for the BasicBlock's that we have entries for.  We use this so
  /// we get notified if a block is deleted or RAUWd.
  std::vector<MMIAddrLabelMapCallbackPtr> BBCallbacks;

public:
  MMIAddrLabelMap(MCContext &context) : Context(context) {}

  ArrayRef<MCSymbol *> getAddrLabelSymbolToEmit(BasicBlock *BB);

  void UpdateForDeletedBlock(BasicBlock *BB);
  void UpdateForRAUWBlock(BasicBlock *Old, BasicBlock *New);
};

} // end namespace llvm

ArrayRef<MCSymbol *> MMIAddrLabelMap::getAddrLabelSymbolToEmit(BasicBlock *BB) {
  assert(BB->hasAddressTaken() &&
         "Shouldn't get label for block without address taken");
  AddrLabelSymEntry &Entry = AddrLabelSymbols[BB];

  // If we already had an entry for this block, just return it.
  if (!Entry.Symbols.empty()) {
    assert(BB->getParent() == Entry.Fn && "Parent changed");
    return Entry.Symbols;
  }

  // Otherwise, this is a new entry, create a new symbol for it and add an
  // entry to BBCallbacks so we can be notified if the BB is deleted or RAUWd.
  BBCallbacks.emplace_back(BB);
  BBCallbacks.back().setMap(this);
  Entry.Index = BBCallbacks.size() - 1;
  Entry.Fn = BB->getParent();
  MCSymbol *Sym = Context.createTempSymbol(!BB->hasAddressTaken());
  Entry.Symbols.push_back(Sym);
  return Entry.Symbols;
}

void MMIAddrLabelMap::UpdateForDeletedBlock(BasicBlock *BB) {
  // If the block got deleted, there is no need for the symbol.  If the symbol
  // was already emitted, we can just forget about it, otherwise we need to
  // queue it up for later emission when the function is output.
  AddrLabelSymEntry Entry = std::move(AddrLabelSymbols[BB]);
  AddrLabelSymbols.erase(BB);
  assert(!Entry.Symbols.empty() && "Didn't have a symbol, why a callback?");
  BBCallbacks[Entry.Index] = nullptr;  // Clear the callback.

  assert((BB->getParent() == nullptr || BB->getParent() == Entry.Fn) &&
         "Block/parent mismatch");

  assert(llvm::all_of(Entry.Symbols, [](MCSymbol *Sym) {
    return Sym->isDefined(); }));
}

void MMIAddrLabelMap::UpdateForRAUWBlock(BasicBlock *Old, BasicBlock *New) {
  // Get the entry for the RAUW'd block and remove it from our map.
  AddrLabelSymEntry OldEntry = std::move(AddrLabelSymbols[Old]);
  AddrLabelSymbols.erase(Old);
  assert(!OldEntry.Symbols.empty() && "Didn't have a symbol, why a callback?");

  AddrLabelSymEntry &NewEntry = AddrLabelSymbols[New];

  // If New is not address taken, just move our symbol over to it.
  if (NewEntry.Symbols.empty()) {
    BBCallbacks[OldEntry.Index].setPtr(New);    // Update the callback.
    NewEntry = std::move(OldEntry);             // Set New's entry.
    return;
  }

  BBCallbacks[OldEntry.Index] = nullptr;    // Update the callback.

  // Otherwise, we need to add the old symbols to the new block's set.
  NewEntry.Symbols.insert(NewEntry.Symbols.end(), OldEntry.Symbols.begin(),
                          OldEntry.Symbols.end());
}

void MMIAddrLabelMapCallbackPtr::deleted() {
  Map->UpdateForDeletedBlock(cast<BasicBlock>(getValPtr()));
}

void MMIAddrLabelMapCallbackPtr::allUsesReplacedWith(Value *V2) {
  Map->UpdateForRAUWBlock(cast<BasicBlock>(getValPtr()), cast<BasicBlock>(V2));
}

void MachineModuleInfo::initialize() {
  ObjFileMMI = nullptr;
  CurCallSite = 0;
  UsesMSVCFloatingPoint = UsesMorestackAddr = false;
  HasSplitStack = HasNosplitStack = false;
  AddrLabelSymbols = nullptr;
}

void MachineModuleInfo::finalize() {
  Personalities.clear();

  delete AddrLabelSymbols;
  AddrLabelSymbols = nullptr;

  Context.reset();
  // We don't clear the ExternalContext.

  delete ObjFileMMI;
  ObjFileMMI = nullptr;
}

MachineModuleInfo::MachineModuleInfo(MachineModuleInfo &&MMI)
    : TM(std::move(MMI.TM)),
      Context(MMI.TM.getMCAsmInfo(), MMI.TM.getMCRegisterInfo(),
              MMI.TM.getObjFileLowering(), nullptr, nullptr, false),
      MachineFunctions(std::move(MMI.MachineFunctions)) {
  ObjFileMMI = MMI.ObjFileMMI;
  CurCallSite = MMI.CurCallSite;
  UsesMSVCFloatingPoint = MMI.UsesMSVCFloatingPoint;
  UsesMorestackAddr = MMI.UsesMorestackAddr;
  HasSplitStack = MMI.HasSplitStack;
  HasNosplitStack = MMI.HasNosplitStack;
  AddrLabelSymbols = MMI.AddrLabelSymbols;
  ExternalContext = MMI.ExternalContext;
  TheModule = MMI.TheModule;
}

MachineModuleInfo::MachineModuleInfo(const LLVMTargetMachine *TM)
    : TM(*TM), Context(TM->getMCAsmInfo(), TM->getMCRegisterInfo(),
                       TM->getObjFileLowering(), nullptr, nullptr, false) {
  initialize();
}

MachineModuleInfo::MachineModuleInfo(const LLVMTargetMachine *TM,
                                     MCContext *ExtContext)
    : TM(*TM), Context(TM->getMCAsmInfo(), TM->getMCRegisterInfo(),
                       TM->getObjFileLowering(), nullptr, nullptr, false),
      ExternalContext(ExtContext) {
  initialize();
}

MachineModuleInfo::~MachineModuleInfo() { finalize(); }

//===- Address of Block Management ----------------------------------------===//

ArrayRef<MCSymbol *>
MachineModuleInfo::getAddrLabelSymbolToEmit(const BasicBlock *BB) {
  // Lazily create AddrLabelSymbols.
  if (!AddrLabelSymbols)
    AddrLabelSymbols = new MMIAddrLabelMap(getContext());
 return AddrLabelSymbols->getAddrLabelSymbolToEmit(const_cast<BasicBlock*>(BB));
}

/// \name Exception Handling
/// \{

void MachineModuleInfo::addPersonality(const Function *Personality) {
  for (unsigned i = 0; i < Personalities.size(); ++i)
    if (Personalities[i] == Personality)
      return;
  Personalities.push_back(Personality);
}

/// \}

MachineFunction *
MachineModuleInfo::getMachineFunction(const Function &F) const {
  auto I = MachineFunctions.find(&F);
  return I != MachineFunctions.end() ? I->second.get() : nullptr;
}

MachineFunction &MachineModuleInfo::getOrCreateMachineFunction(Function &F) {
  // Shortcut for the common case where a sequence of MachineFunctionPasses
  // all query for the same Function.
  if (LastRequest == &F)
    return *LastResult;

  auto I = MachineFunctions.insert(
      std::make_pair(&F, std::unique_ptr<MachineFunction>()));
  MachineFunction *MF;
  if (I.second) {
    // No pre-existing machine function, create a new one.
    const TargetSubtargetInfo &STI = *TM.getSubtargetImpl(F);
    MF = new MachineFunction(F, TM, STI, NextFnNum++, *this);
    // Update the set entry.
    I.first->second.reset(MF);
  } else {
    MF = I.first->second.get();
  }

  LastRequest = &F;
  LastResult = MF;
  return *MF;
}

void MachineModuleInfo::deleteMachineFunctionFor(Function &F) {
  MachineFunctions.erase(&F);
  LastRequest = nullptr;
  LastResult = nullptr;
}

namespace {

/// This pass frees the MachineFunction object associated with a Function.
class FreeMachineFunction : public FunctionPass {
public:
  static char ID;

  FreeMachineFunction() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineModuleInfoWrapperPass>();
    AU.addPreserved<MachineModuleInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    MachineModuleInfo &MMI =
        getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
    MMI.deleteMachineFunctionFor(F);
    return true;
  }

  StringRef getPassName() const override {
    return "Free MachineFunction";
  }
};

} // end anonymous namespace

char FreeMachineFunction::ID;

FunctionPass *llvm::createFreeMachineFunctionPass() {
  return new FreeMachineFunction();
}

MachineModuleInfoWrapperPass::MachineModuleInfoWrapperPass(
    const LLVMTargetMachine *TM)
    : ImmutablePass(ID), MMI(TM) {
  initializeMachineModuleInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

MachineModuleInfoWrapperPass::MachineModuleInfoWrapperPass(
    const LLVMTargetMachine *TM, MCContext *ExtContext)
    : ImmutablePass(ID), MMI(TM, ExtContext) {
  initializeMachineModuleInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

// Handle the Pass registration stuff necessary to use DataLayout's.
INITIALIZE_PASS(MachineModuleInfoWrapperPass, "machinemoduleinfo",
                "Machine Module Information", false, false)
char MachineModuleInfoWrapperPass::ID = 0;

bool MachineModuleInfoWrapperPass::doInitialization(Module &M) {
  MMI.initialize();
  MMI.TheModule = &M;
  MMI.DbgInfoAvailable = !M.debug_compile_units().empty();
  return false;
}

bool MachineModuleInfoWrapperPass::doFinalization(Module &M) {
  MMI.finalize();
  return false;
}

AnalysisKey MachineModuleAnalysis::Key;

MachineModuleInfo MachineModuleAnalysis::run(Module &M,
                                             ModuleAnalysisManager &) {
  MachineModuleInfo MMI(TM);
  MMI.TheModule = &M;
  MMI.DbgInfoAvailable = !M.debug_compile_units().empty();
  return MMI;
}
