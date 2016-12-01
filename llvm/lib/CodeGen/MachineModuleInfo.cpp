//===-- llvm/CodeGen/MachineModuleInfo.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionInitializer.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;
using namespace llvm::dwarf;

// Handle the Pass registration stuff necessary to use DataLayout's.
INITIALIZE_TM_PASS(MachineModuleInfo, "machinemoduleinfo",
                   "Machine Module Information", false, false)
char MachineModuleInfo::ID = 0;

// Out of line virtual method.
MachineModuleInfoImpl::~MachineModuleInfoImpl() {}

namespace llvm {
class MMIAddrLabelMapCallbackPtr final : CallbackVH {
  MMIAddrLabelMap *Map;
public:
  MMIAddrLabelMapCallbackPtr() : Map(nullptr) {}
  MMIAddrLabelMapCallbackPtr(Value *V) : CallbackVH(V), Map(nullptr) {}

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

  /// This is a per-function list of symbols whose corresponding BasicBlock got
  /// deleted.  These symbols need to be emitted at some point in the file, so
  /// AsmPrinter emits them after the function body.
  DenseMap<AssertingVH<Function>, std::vector<MCSymbol*> >
    DeletedAddrLabelsNeedingEmission;
public:

  MMIAddrLabelMap(MCContext &context) : Context(context) {}
  ~MMIAddrLabelMap() {
    assert(DeletedAddrLabelsNeedingEmission.empty() &&
           "Some labels for deleted blocks never got emitted");
  }

  ArrayRef<MCSymbol *> getAddrLabelSymbolToEmit(BasicBlock *BB);

  void takeDeletedSymbolsForFunction(Function *F,
                                     std::vector<MCSymbol*> &Result);

  void UpdateForDeletedBlock(BasicBlock *BB);
  void UpdateForRAUWBlock(BasicBlock *Old, BasicBlock *New);
};
}

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
  Entry.Symbols.push_back(Context.createTempSymbol());
  return Entry.Symbols;
}

/// If we have any deleted symbols for F, return them.
void MMIAddrLabelMap::
takeDeletedSymbolsForFunction(Function *F, std::vector<MCSymbol*> &Result) {
  DenseMap<AssertingVH<Function>, std::vector<MCSymbol*> >::iterator I =
    DeletedAddrLabelsNeedingEmission.find(F);

  // If there are no entries for the function, just return.
  if (I == DeletedAddrLabelsNeedingEmission.end()) return;

  // Otherwise, take the list.
  std::swap(Result, I->second);
  DeletedAddrLabelsNeedingEmission.erase(I);
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

  for (MCSymbol *Sym : Entry.Symbols) {
    if (Sym->isDefined())
      return;

    // If the block is not yet defined, we need to emit it at the end of the
    // function.  Add the symbol to the DeletedAddrLabelsNeedingEmission list
    // for the containing Function.  Since the block is being deleted, its
    // parent may already be removed, we have to get the function from 'Entry'.
    DeletedAddrLabelsNeedingEmission[Entry.Fn].push_back(Sym);
  }
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


//===----------------------------------------------------------------------===//

MachineModuleInfo::MachineModuleInfo(const TargetMachine *TM)
  : ImmutablePass(ID), TM(*TM),
    Context(TM->getMCAsmInfo(), TM->getMCRegisterInfo(),
            TM->getObjFileLowering(), nullptr, false) {
  initializeMachineModuleInfoPass(*PassRegistry::getPassRegistry());
}

MachineModuleInfo::~MachineModuleInfo() {
}

bool MachineModuleInfo::doInitialization(Module &M) {

  ObjFileMMI = nullptr;
  CurCallSite = 0;
  CallsEHReturn = false;
  CallsUnwindInit = false;
  HasEHFunclets = false;
  DbgInfoAvailable = UsesVAFloatArgument = UsesMorestackAddr = false;
  PersonalityTypeCache = EHPersonality::Unknown;
  AddrLabelSymbols = nullptr;
  TheModule = &M;

  return false;
}

bool MachineModuleInfo::doFinalization(Module &M) {

  Personalities.clear();

  delete AddrLabelSymbols;
  AddrLabelSymbols = nullptr;

  Context.reset();

  delete ObjFileMMI;
  ObjFileMMI = nullptr;

  return false;
}

void MachineModuleInfo::EndFunction() {
  // Clean up exception info.
  LandingPads.clear();
  PersonalityTypeCache = EHPersonality::Unknown;
  CallSiteMap.clear();
  TypeInfos.clear();
  FilterIds.clear();
  FilterEnds.clear();
  CallsEHReturn = false;
  CallsUnwindInit = false;
  HasEHFunclets = false;
}

//===- Address of Block Management ----------------------------------------===//

ArrayRef<MCSymbol *>
MachineModuleInfo::getAddrLabelSymbolToEmit(const BasicBlock *BB) {
  // Lazily create AddrLabelSymbols.
  if (!AddrLabelSymbols)
    AddrLabelSymbols = new MMIAddrLabelMap(Context);
 return AddrLabelSymbols->getAddrLabelSymbolToEmit(const_cast<BasicBlock*>(BB));
}

void MachineModuleInfo::
takeDeletedSymbolsForFunction(const Function *F,
                              std::vector<MCSymbol*> &Result) {
  // If no blocks have had their addresses taken, we're done.
  if (!AddrLabelSymbols) return;
  return AddrLabelSymbols->
     takeDeletedSymbolsForFunction(const_cast<Function*>(F), Result);
}

//===- EH -----------------------------------------------------------------===//

LandingPadInfo &MachineModuleInfo::getOrCreateLandingPadInfo
    (MachineBasicBlock *LandingPad) {
  unsigned N = LandingPads.size();
  for (unsigned i = 0; i < N; ++i) {
    LandingPadInfo &LP = LandingPads[i];
    if (LP.LandingPadBlock == LandingPad)
      return LP;
  }

  LandingPads.push_back(LandingPadInfo(LandingPad));
  return LandingPads[N];
}

void MachineModuleInfo::addInvoke(MachineBasicBlock *LandingPad,
                                  MCSymbol *BeginLabel, MCSymbol *EndLabel) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.BeginLabels.push_back(BeginLabel);
  LP.EndLabels.push_back(EndLabel);
}

MCSymbol *MachineModuleInfo::addLandingPad(MachineBasicBlock *LandingPad) {
  MCSymbol *LandingPadLabel = Context.createTempSymbol();
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.LandingPadLabel = LandingPadLabel;
  return LandingPadLabel;
}

void MachineModuleInfo::addPersonality(const Function *Personality) {
  for (unsigned i = 0; i < Personalities.size(); ++i)
    if (Personalities[i] == Personality)
      return;
  Personalities.push_back(Personality);
}

void MachineModuleInfo::
addCatchTypeInfo(MachineBasicBlock *LandingPad,
                 ArrayRef<const GlobalValue *> TyInfo) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  for (unsigned N = TyInfo.size(); N; --N)
    LP.TypeIds.push_back(getTypeIDFor(TyInfo[N - 1]));
}

void MachineModuleInfo::
addFilterTypeInfo(MachineBasicBlock *LandingPad,
                  ArrayRef<const GlobalValue *> TyInfo) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  std::vector<unsigned> IdsInFilter(TyInfo.size());
  for (unsigned I = 0, E = TyInfo.size(); I != E; ++I)
    IdsInFilter[I] = getTypeIDFor(TyInfo[I]);
  LP.TypeIds.push_back(getFilterIDFor(IdsInFilter));
}

void MachineModuleInfo::addCleanup(MachineBasicBlock *LandingPad) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.TypeIds.push_back(0);
}

void MachineModuleInfo::addSEHCatchHandler(MachineBasicBlock *LandingPad,
                                           const Function *Filter,
                                           const BlockAddress *RecoverBA) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  SEHHandler Handler;
  Handler.FilterOrFinally = Filter;
  Handler.RecoverBA = RecoverBA;
  LP.SEHHandlers.push_back(Handler);
}

void MachineModuleInfo::addSEHCleanupHandler(MachineBasicBlock *LandingPad,
                                             const Function *Cleanup) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  SEHHandler Handler;
  Handler.FilterOrFinally = Cleanup;
  Handler.RecoverBA = nullptr;
  LP.SEHHandlers.push_back(Handler);
}

void MachineModuleInfo::TidyLandingPads(DenseMap<MCSymbol*, uintptr_t> *LPMap) {
  for (unsigned i = 0; i != LandingPads.size(); ) {
    LandingPadInfo &LandingPad = LandingPads[i];
    if (LandingPad.LandingPadLabel &&
        !LandingPad.LandingPadLabel->isDefined() &&
        (!LPMap || (*LPMap)[LandingPad.LandingPadLabel] == 0))
      LandingPad.LandingPadLabel = nullptr;

    // Special case: we *should* emit LPs with null LP MBB. This indicates
    // "nounwind" case.
    if (!LandingPad.LandingPadLabel && LandingPad.LandingPadBlock) {
      LandingPads.erase(LandingPads.begin() + i);
      continue;
    }

    for (unsigned j = 0, e = LandingPads[i].BeginLabels.size(); j != e; ++j) {
      MCSymbol *BeginLabel = LandingPad.BeginLabels[j];
      MCSymbol *EndLabel = LandingPad.EndLabels[j];
      if ((BeginLabel->isDefined() ||
           (LPMap && (*LPMap)[BeginLabel] != 0)) &&
          (EndLabel->isDefined() ||
           (LPMap && (*LPMap)[EndLabel] != 0))) continue;

      LandingPad.BeginLabels.erase(LandingPad.BeginLabels.begin() + j);
      LandingPad.EndLabels.erase(LandingPad.EndLabels.begin() + j);
      --j;
      --e;
    }

    // Remove landing pads with no try-ranges.
    if (LandingPads[i].BeginLabels.empty()) {
      LandingPads.erase(LandingPads.begin() + i);
      continue;
    }

    // If there is no landing pad, ensure that the list of typeids is empty.
    // If the only typeid is a cleanup, this is the same as having no typeids.
    if (!LandingPad.LandingPadBlock ||
        (LandingPad.TypeIds.size() == 1 && !LandingPad.TypeIds[0]))
      LandingPad.TypeIds.clear();
    ++i;
  }
}

void MachineModuleInfo::setCallSiteLandingPad(MCSymbol *Sym,
                                              ArrayRef<unsigned> Sites) {
  LPadToCallSiteMap[Sym].append(Sites.begin(), Sites.end());
}

unsigned MachineModuleInfo::getTypeIDFor(const GlobalValue *TI) {
  for (unsigned i = 0, N = TypeInfos.size(); i != N; ++i)
    if (TypeInfos[i] == TI) return i + 1;

  TypeInfos.push_back(TI);
  return TypeInfos.size();
}

int MachineModuleInfo::getFilterIDFor(std::vector<unsigned> &TyIds) {
  // If the new filter coincides with the tail of an existing filter, then
  // re-use the existing filter.  Folding filters more than this requires
  // re-ordering filters and/or their elements - probably not worth it.
  for (std::vector<unsigned>::iterator I = FilterEnds.begin(),
       E = FilterEnds.end(); I != E; ++I) {
    unsigned i = *I, j = TyIds.size();

    while (i && j)
      if (FilterIds[--i] != TyIds[--j])
        goto try_next;

    if (!j)
      // The new filter coincides with range [i, end) of the existing filter.
      return -(1 + i);

try_next:;
  }

  // Add the new filter.
  int FilterID = -(1 + FilterIds.size());
  FilterIds.reserve(FilterIds.size() + TyIds.size() + 1);
  FilterIds.insert(FilterIds.end(), TyIds.begin(), TyIds.end());
  FilterEnds.push_back(FilterIds.size());
  FilterIds.push_back(0); // terminator
  return FilterID;
}

MachineFunction &MachineModuleInfo::getMachineFunction(const Function &F) {
  // Shortcut for the common case where a sequence of MachineFunctionPasses
  // all query for the same Function.
  if (LastRequest == &F)
    return *LastResult;

  auto I = MachineFunctions.insert(
      std::make_pair(&F, std::unique_ptr<MachineFunction>()));
  MachineFunction *MF;
  if (I.second) {
    // No pre-existing machine function, create a new one.
    MF = new MachineFunction(&F, TM, NextFnNum++, *this);
    // Update the set entry.
    I.first->second.reset(MF);

    if (MFInitializer)
      if (MFInitializer->initializeMachineFunction(*MF))
        report_fatal_error("Unable to initialize machine function");
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
    AU.addRequired<MachineModuleInfo>();
    AU.addPreserved<MachineModuleInfo>();
  }

  bool runOnFunction(Function &F) override {
    MachineModuleInfo &MMI = getAnalysis<MachineModuleInfo>();
    MMI.deleteMachineFunctionFor(F);
    return true;
  }
};
char FreeMachineFunction::ID;
} // end anonymous namespace

namespace llvm {
FunctionPass *createFreeMachineFunctionPass() {
  return new FreeMachineFunction();
}
} // end namespace llvm

//===- MMI building helpers -----------------------------------------------===//

void llvm::computeUsesVAFloatArgument(const CallInst &I,
                                      MachineModuleInfo &MMI) {
  FunctionType *FT =
      cast<FunctionType>(I.getCalledValue()->getType()->getContainedType(0));
  if (FT->isVarArg() && !MMI.usesVAFloatArgument()) {
    for (unsigned i = 0, e = I.getNumArgOperands(); i != e; ++i) {
      Type *T = I.getArgOperand(i)->getType();
      for (auto i : post_order(T)) {
        if (i->isFloatingPointTy()) {
          MMI.setUsesVAFloatArgument(true);
          return;
        }
      }
    }
  }
}

void llvm::addLandingPadInfo(const LandingPadInst &I, MachineModuleInfo &MMI,
                             MachineBasicBlock &MBB) {
  if (const auto *PF = dyn_cast<Function>(
          I.getParent()->getParent()->getPersonalityFn()->stripPointerCasts()))
    MMI.addPersonality(PF);

  if (I.isCleanup())
    MMI.addCleanup(&MBB);

  // FIXME: New EH - Add the clauses in reverse order. This isn't 100% correct,
  //        but we need to do it this way because of how the DWARF EH emitter
  //        processes the clauses.
  for (unsigned i = I.getNumClauses(); i != 0; --i) {
    Value *Val = I.getClause(i - 1);
    if (I.isCatch(i - 1)) {
      MMI.addCatchTypeInfo(&MBB,
                           dyn_cast<GlobalValue>(Val->stripPointerCasts()));
    } else {
      // Add filters in a list.
      Constant *CVal = cast<Constant>(Val);
      SmallVector<const GlobalValue *, 4> FilterList;
      for (User::op_iterator II = CVal->op_begin(), IE = CVal->op_end();
           II != IE; ++II)
        FilterList.push_back(cast<GlobalValue>((*II)->stripPointerCasts()));

      MMI.addFilterTypeInfo(&MBB, FilterList);
    }
  }
}
