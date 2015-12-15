//===- lib/Linker/LinkModules.cpp - Module Linker Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM module linker.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker/Linker.h"
#include "LinkDiagnosticInfo.h"
#include "llvm-c/Linker.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
using namespace llvm;

namespace {

/// This is an implementation class for the LinkModules function, which is the
/// entrypoint for this file.
class ModuleLinker {
  IRMover &Mover;
  Module &SrcM;

  SetVector<GlobalValue *> ValuesToLink;
  StringSet<> Internalize;

  /// For symbol clashes, prefer those from Src.
  unsigned Flags;

  /// Function index passed into ModuleLinker for using in function
  /// importing/exporting handling.
  const FunctionInfoIndex *ImportIndex;

  /// Function to import from source module, all other functions are
  /// imported as declarations instead of definitions.
  DenseSet<const GlobalValue *> *ImportFunction;

  /// Set to true if the given FunctionInfoIndex contains any functions
  /// from this source module, in which case we must conservatively assume
  /// that any of its functions may be imported into another module
  /// as part of a different backend compilation process.
  bool HasExportedFunctions = false;

  /// Used as the callback for lazy linking.
  /// The mover has just hit GV and we have to decide if it, and other members
  /// of the same comdat, should be linked. Every member to be linked is passed
  /// to Add.
  void addLazyFor(GlobalValue &GV, IRMover::ValueAdder Add);

  bool shouldOverrideFromSrc() { return Flags & Linker::OverrideFromSrc; }
  bool shouldLinkOnlyNeeded() { return Flags & Linker::LinkOnlyNeeded; }
  bool shouldInternalizeLinkedSymbols() {
    return Flags & Linker::InternalizeLinkedSymbols;
  }

  /// Check if we should promote the given local value to global scope.
  bool doPromoteLocalToGlobal(const GlobalValue *SGV);

  bool shouldLinkFromSource(bool &LinkFromSrc, const GlobalValue &Dest,
                            const GlobalValue &Src);

  /// Should we have mover and linker error diag info?
  bool emitError(const Twine &Message) {
    SrcM.getContext().diagnose(LinkDiagnosticInfo(DS_Error, Message));
    return true;
  }

  bool getComdatLeader(Module &M, StringRef ComdatName,
                       const GlobalVariable *&GVar);
  bool computeResultingSelectionKind(StringRef ComdatName,
                                     Comdat::SelectionKind Src,
                                     Comdat::SelectionKind Dst,
                                     Comdat::SelectionKind &Result,
                                     bool &LinkFromSrc);
  std::map<const Comdat *, std::pair<Comdat::SelectionKind, bool>>
      ComdatsChosen;
  bool getComdatResult(const Comdat *SrcC, Comdat::SelectionKind &SK,
                       bool &LinkFromSrc);
  // Keep track of the global value members of each comdat in source.
  DenseMap<const Comdat *, std::vector<GlobalValue *>> ComdatMembers;

  /// Given a global in the source module, return the global in the
  /// destination module that is being linked to, if any.
  GlobalValue *getLinkedToGlobal(const GlobalValue *SrcGV) {
    Module &DstM = Mover.getModule();
    // If the source has no name it can't link.  If it has local linkage,
    // there is no name match-up going on.
    if (!SrcGV->hasName() || GlobalValue::isLocalLinkage(getLinkage(SrcGV)))
      return nullptr;

    // Otherwise see if we have a match in the destination module's symtab.
    GlobalValue *DGV = DstM.getNamedValue(getName(SrcGV));
    if (!DGV)
      return nullptr;

    // If we found a global with the same name in the dest module, but it has
    // internal linkage, we are really not doing any linkage here.
    if (DGV->hasLocalLinkage())
      return nullptr;

    // Otherwise, we do in fact link to the destination global.
    return DGV;
  }

  bool linkIfNeeded(GlobalValue &GV);

  /// Helper methods to check if we are importing from or potentially
  /// exporting from the current source module.
  bool isPerformingImport() { return ImportFunction != nullptr; }
  bool isModuleExporting() { return HasExportedFunctions; }

  /// If we are importing from the source module, checks if we should
  /// import SGV as a definition, otherwise import as a declaration.
  bool doImportAsDefinition(const GlobalValue *SGV);

  /// Get the name for SGV that should be used in the linked destination
  /// module. Specifically, this handles the case where we need to rename
  /// a local that is being promoted to global scope.
  std::string getName(const GlobalValue *SGV);

  /// Process globals so that they can be used in ThinLTO. This includes
  /// promoting local variables so that they can be reference externally by
  /// thin lto imported globals and converting strong external globals to
  /// available_externally.
  void processGlobalsForThinLTO();
  void processGlobalForThinLTO(GlobalValue &GV);

  /// Get the new linkage for SGV that should be used in the linked destination
  /// module. Specifically, for ThinLTO importing or exporting it may need
  /// to be adjusted.
  GlobalValue::LinkageTypes getLinkage(const GlobalValue *SGV);

  /// Copies the necessary global value attributes and name from the source
  /// to the newly cloned global value.
  void copyGVAttributes(GlobalValue *NewGV, const GlobalValue *SrcGV);

  /// Updates the visibility for the new global cloned from the source
  /// and, if applicable, linked with an existing destination global.
  /// Handles visibility change required for promoted locals.
  void setVisibility(GlobalValue *NewGV, const GlobalValue *SGV,
                     const GlobalValue *DGV = nullptr);

public:
  ModuleLinker(IRMover &Mover, Module &SrcM, unsigned Flags,
               const FunctionInfoIndex *Index = nullptr,
               DenseSet<const GlobalValue *> *FunctionsToImport = nullptr)
      : Mover(Mover), SrcM(SrcM), Flags(Flags), ImportIndex(Index),
        ImportFunction(FunctionsToImport) {
    assert((ImportIndex || !ImportFunction) &&
           "Expect a FunctionInfoIndex when importing");
    // If we have a FunctionInfoIndex but no function to import,
    // then this is the primary module being compiled in a ThinLTO
    // backend compilation, and we need to see if it has functions that
    // may be exported to another backend compilation.
    if (ImportIndex && !ImportFunction)
      HasExportedFunctions = ImportIndex->hasExportedFunctions(SrcM);
  }

  bool run();
};
}

/// The LLVM SymbolTable class autorenames globals that conflict in the symbol
/// table. This is good for all clients except for us. Go through the trouble
/// to force this back.
static void forceRenaming(GlobalValue *GV, StringRef Name) {
  // If the global doesn't force its name or if it already has the right name,
  // there is nothing for us to do.
  // Note that any required local to global promotion should already be done,
  // so promoted locals will not skip this handling as their linkage is no
  // longer local.
  if (GV->hasLocalLinkage() || GV->getName() == Name)
    return;

  Module *M = GV->getParent();

  // If there is a conflict, rename the conflict.
  if (GlobalValue *ConflictGV = M->getNamedValue(Name)) {
    GV->takeName(ConflictGV);
    ConflictGV->setName(Name); // This will cause ConflictGV to get renamed
    assert(ConflictGV->getName() != Name && "forceRenaming didn't work");
  } else {
    GV->setName(Name); // Force the name back
  }
}

/// copy additional attributes (those not needed to construct a GlobalValue)
/// from the SrcGV to the DestGV.
void ModuleLinker::copyGVAttributes(GlobalValue *NewGV,
                                    const GlobalValue *SrcGV) {
  NewGV->copyAttributesFrom(SrcGV);
  forceRenaming(NewGV, getName(SrcGV));
}

bool ModuleLinker::doImportAsDefinition(const GlobalValue *SGV) {
  if (!isPerformingImport())
    return false;
  auto *GA = dyn_cast<GlobalAlias>(SGV);
  if (GA) {
    if (GA->hasWeakAnyLinkage())
      return false;
    const GlobalObject *GO = GA->getBaseObject();
    if (!GO->hasLinkOnceODRLinkage())
      return false;
    return doImportAsDefinition(GO);
  }
  // Always import GlobalVariable definitions, except for the special
  // case of WeakAny which are imported as ExternalWeak declarations
  // (see comments in ModuleLinker::getLinkage). The linkage changes
  // described in ModuleLinker::getLinkage ensure the correct behavior (e.g.
  // global variables with external linkage are transformed to
  // available_externally definitions, which are ultimately turned into
  // declarations after the EliminateAvailableExternally pass).
  if (isa<GlobalVariable>(SGV) && !SGV->isDeclaration() &&
      !SGV->hasWeakAnyLinkage())
    return true;
  // Only import the function requested for importing.
  auto *SF = dyn_cast<Function>(SGV);
  if (SF && ImportFunction->count(SF))
    return true;
  // Otherwise no.
  return false;
}

bool ModuleLinker::doPromoteLocalToGlobal(const GlobalValue *SGV) {
  assert(SGV->hasLocalLinkage());
  // Both the imported references and the original local variable must
  // be promoted.
  if (!isPerformingImport() && !isModuleExporting())
    return false;

  // Local const variables never need to be promoted unless they are address
  // taken. The imported uses can simply use the clone created in this module.
  // For now we are conservative in determining which variables are not
  // address taken by checking the unnamed addr flag. To be more aggressive,
  // the address taken information must be checked earlier during parsing
  // of the module and recorded in the function index for use when importing
  // from that module.
  auto *GVar = dyn_cast<GlobalVariable>(SGV);
  if (GVar && GVar->isConstant() && GVar->hasUnnamedAddr())
    return false;

  // Eventually we only need to promote functions in the exporting module that
  // are referenced by a potentially exported function (i.e. one that is in the
  // function index).
  return true;
}

std::string ModuleLinker::getName(const GlobalValue *SGV) {
  // For locals that must be promoted to global scope, ensure that
  // the promoted name uniquely identifies the copy in the original module,
  // using the ID assigned during combined index creation. When importing,
  // we rename all locals (not just those that are promoted) in order to
  // avoid naming conflicts between locals imported from different modules.
  if (SGV->hasLocalLinkage() &&
      (doPromoteLocalToGlobal(SGV) || isPerformingImport()))
    return FunctionInfoIndex::getGlobalNameForLocal(
        SGV->getName(),
        ImportIndex->getModuleId(SGV->getParent()->getModuleIdentifier()));
  return SGV->getName();
}

GlobalValue::LinkageTypes ModuleLinker::getLinkage(const GlobalValue *SGV) {
  // Any local variable that is referenced by an exported function needs
  // to be promoted to global scope. Since we don't currently know which
  // functions reference which local variables/functions, we must treat
  // all as potentially exported if this module is exporting anything.
  if (isModuleExporting()) {
    if (SGV->hasLocalLinkage() && doPromoteLocalToGlobal(SGV))
      return GlobalValue::ExternalLinkage;
    return SGV->getLinkage();
  }

  // Otherwise, if we aren't importing, no linkage change is needed.
  if (!isPerformingImport())
    return SGV->getLinkage();

  switch (SGV->getLinkage()) {
  case GlobalValue::ExternalLinkage:
    // External defnitions are converted to available_externally
    // definitions upon import, so that they are available for inlining
    // and/or optimization, but are turned into declarations later
    // during the EliminateAvailableExternally pass.
    if (doImportAsDefinition(SGV) && !dyn_cast<GlobalAlias>(SGV))
      return GlobalValue::AvailableExternallyLinkage;
    // An imported external declaration stays external.
    return SGV->getLinkage();

  case GlobalValue::AvailableExternallyLinkage:
    // An imported available_externally definition converts
    // to external if imported as a declaration.
    if (!doImportAsDefinition(SGV))
      return GlobalValue::ExternalLinkage;
    // An imported available_externally declaration stays that way.
    return SGV->getLinkage();

  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
    // These both stay the same when importing the definition.
    // The ThinLTO pass will eventually force-import their definitions.
    return SGV->getLinkage();

  case GlobalValue::WeakAnyLinkage:
    // Can't import weak_any definitions correctly, or we might change the
    // program semantics, since the linker will pick the first weak_any
    // definition and importing would change the order they are seen by the
    // linker. The module linking caller needs to enforce this.
    assert(!doImportAsDefinition(SGV));
    // If imported as a declaration, it becomes external_weak.
    return GlobalValue::ExternalWeakLinkage;

  case GlobalValue::WeakODRLinkage:
    // For weak_odr linkage, there is a guarantee that all copies will be
    // equivalent, so the issue described above for weak_any does not exist,
    // and the definition can be imported. It can be treated similarly
    // to an imported externally visible global value.
    if (doImportAsDefinition(SGV) && !dyn_cast<GlobalAlias>(SGV))
      return GlobalValue::AvailableExternallyLinkage;
    else
      return GlobalValue::ExternalLinkage;

  case GlobalValue::AppendingLinkage:
    // It would be incorrect to import an appending linkage variable,
    // since it would cause global constructors/destructors to be
    // executed multiple times. This should have already been handled
    // by linkIfNeeded, and we will assert in shouldLinkFromSource
    // if we try to import, so we simply return AppendingLinkage here
    // as this helper is called more widely in getLinkedToGlobal.
    return GlobalValue::AppendingLinkage;

  case GlobalValue::InternalLinkage:
  case GlobalValue::PrivateLinkage:
    // If we are promoting the local to global scope, it is handled
    // similarly to a normal externally visible global.
    if (doPromoteLocalToGlobal(SGV)) {
      if (doImportAsDefinition(SGV) && !dyn_cast<GlobalAlias>(SGV))
        return GlobalValue::AvailableExternallyLinkage;
      else
        return GlobalValue::ExternalLinkage;
    }
    // A non-promoted imported local definition stays local.
    // The ThinLTO pass will eventually force-import their definitions.
    return SGV->getLinkage();

  case GlobalValue::ExternalWeakLinkage:
    // External weak doesn't apply to definitions, must be a declaration.
    assert(!doImportAsDefinition(SGV));
    // Linkage stays external_weak.
    return SGV->getLinkage();

  case GlobalValue::CommonLinkage:
    // Linkage stays common on definitions.
    // The ThinLTO pass will eventually force-import their definitions.
    return SGV->getLinkage();
  }

  llvm_unreachable("unknown linkage type");
}

static GlobalValue::VisibilityTypes
getMinVisibility(GlobalValue::VisibilityTypes A,
                 GlobalValue::VisibilityTypes B) {
  if (A == GlobalValue::HiddenVisibility || B == GlobalValue::HiddenVisibility)
    return GlobalValue::HiddenVisibility;
  if (A == GlobalValue::ProtectedVisibility ||
      B == GlobalValue::ProtectedVisibility)
    return GlobalValue::ProtectedVisibility;
  return GlobalValue::DefaultVisibility;
}

void ModuleLinker::setVisibility(GlobalValue *NewGV, const GlobalValue *SGV,
                                 const GlobalValue *DGV) {
  GlobalValue::VisibilityTypes Visibility = SGV->getVisibility();
  if (DGV)
    Visibility = getMinVisibility(DGV->getVisibility(), Visibility);
  // For promoted locals, mark them hidden so that they can later be
  // stripped from the symbol table to reduce bloat.
  if (SGV->hasLocalLinkage() && doPromoteLocalToGlobal(SGV))
    Visibility = GlobalValue::HiddenVisibility;
  NewGV->setVisibility(Visibility);
}

bool ModuleLinker::getComdatLeader(Module &M, StringRef ComdatName,
                                   const GlobalVariable *&GVar) {
  const GlobalValue *GVal = M.getNamedValue(ComdatName);
  if (const auto *GA = dyn_cast_or_null<GlobalAlias>(GVal)) {
    GVal = GA->getBaseObject();
    if (!GVal)
      // We cannot resolve the size of the aliasee yet.
      return emitError("Linking COMDATs named '" + ComdatName +
                       "': COMDAT key involves incomputable alias size.");
  }

  GVar = dyn_cast_or_null<GlobalVariable>(GVal);
  if (!GVar)
    return emitError(
        "Linking COMDATs named '" + ComdatName +
        "': GlobalVariable required for data dependent selection!");

  return false;
}

bool ModuleLinker::computeResultingSelectionKind(StringRef ComdatName,
                                                 Comdat::SelectionKind Src,
                                                 Comdat::SelectionKind Dst,
                                                 Comdat::SelectionKind &Result,
                                                 bool &LinkFromSrc) {
  Module &DstM = Mover.getModule();
  // The ability to mix Comdat::SelectionKind::Any with
  // Comdat::SelectionKind::Largest is a behavior that comes from COFF.
  bool DstAnyOrLargest = Dst == Comdat::SelectionKind::Any ||
                         Dst == Comdat::SelectionKind::Largest;
  bool SrcAnyOrLargest = Src == Comdat::SelectionKind::Any ||
                         Src == Comdat::SelectionKind::Largest;
  if (DstAnyOrLargest && SrcAnyOrLargest) {
    if (Dst == Comdat::SelectionKind::Largest ||
        Src == Comdat::SelectionKind::Largest)
      Result = Comdat::SelectionKind::Largest;
    else
      Result = Comdat::SelectionKind::Any;
  } else if (Src == Dst) {
    Result = Dst;
  } else {
    return emitError("Linking COMDATs named '" + ComdatName +
                     "': invalid selection kinds!");
  }

  switch (Result) {
  case Comdat::SelectionKind::Any:
    // Go with Dst.
    LinkFromSrc = false;
    break;
  case Comdat::SelectionKind::NoDuplicates:
    return emitError("Linking COMDATs named '" + ComdatName +
                     "': noduplicates has been violated!");
  case Comdat::SelectionKind::ExactMatch:
  case Comdat::SelectionKind::Largest:
  case Comdat::SelectionKind::SameSize: {
    const GlobalVariable *DstGV;
    const GlobalVariable *SrcGV;
    if (getComdatLeader(DstM, ComdatName, DstGV) ||
        getComdatLeader(SrcM, ComdatName, SrcGV))
      return true;

    const DataLayout &DstDL = DstM.getDataLayout();
    const DataLayout &SrcDL = SrcM.getDataLayout();
    uint64_t DstSize =
        DstDL.getTypeAllocSize(DstGV->getType()->getPointerElementType());
    uint64_t SrcSize =
        SrcDL.getTypeAllocSize(SrcGV->getType()->getPointerElementType());
    if (Result == Comdat::SelectionKind::ExactMatch) {
      if (SrcGV->getInitializer() != DstGV->getInitializer())
        return emitError("Linking COMDATs named '" + ComdatName +
                         "': ExactMatch violated!");
      LinkFromSrc = false;
    } else if (Result == Comdat::SelectionKind::Largest) {
      LinkFromSrc = SrcSize > DstSize;
    } else if (Result == Comdat::SelectionKind::SameSize) {
      if (SrcSize != DstSize)
        return emitError("Linking COMDATs named '" + ComdatName +
                         "': SameSize violated!");
      LinkFromSrc = false;
    } else {
      llvm_unreachable("unknown selection kind");
    }
    break;
  }
  }

  return false;
}

bool ModuleLinker::getComdatResult(const Comdat *SrcC,
                                   Comdat::SelectionKind &Result,
                                   bool &LinkFromSrc) {
  Module &DstM = Mover.getModule();
  Comdat::SelectionKind SSK = SrcC->getSelectionKind();
  StringRef ComdatName = SrcC->getName();
  Module::ComdatSymTabType &ComdatSymTab = DstM.getComdatSymbolTable();
  Module::ComdatSymTabType::iterator DstCI = ComdatSymTab.find(ComdatName);

  if (DstCI == ComdatSymTab.end()) {
    // Use the comdat if it is only available in one of the modules.
    LinkFromSrc = true;
    Result = SSK;
    return false;
  }

  const Comdat *DstC = &DstCI->second;
  Comdat::SelectionKind DSK = DstC->getSelectionKind();
  return computeResultingSelectionKind(ComdatName, SSK, DSK, Result,
                                       LinkFromSrc);
}

bool ModuleLinker::shouldLinkFromSource(bool &LinkFromSrc,
                                        const GlobalValue &Dest,
                                        const GlobalValue &Src) {
  // Should we unconditionally use the Src?
  if (shouldOverrideFromSrc()) {
    LinkFromSrc = true;
    return false;
  }

  // We always have to add Src if it has appending linkage.
  if (Src.hasAppendingLinkage()) {
    // Should have prevented importing for appending linkage in linkIfNeeded.
    assert(!isPerformingImport());
    LinkFromSrc = true;
    return false;
  }

  bool SrcIsDeclaration = Src.isDeclarationForLinker();
  bool DestIsDeclaration = Dest.isDeclarationForLinker();

  if (isPerformingImport()) {
    if (isa<Function>(&Src)) {
      // For functions, LinkFromSrc iff this is the function requested
      // for importing. For variables, decide below normally.
      LinkFromSrc = ImportFunction->count(&Src);
      return false;
    }

    // Check if this is an alias with an already existing definition
    // in Dest, which must have come from a prior importing pass from
    // the same Src module. Unlike imported function and variable
    // definitions, which are imported as available_externally and are
    // not definitions for the linker, that is not a valid linkage for
    // imported aliases which must be definitions. Simply use the existing
    // Dest copy.
    if (isa<GlobalAlias>(&Src) && !DestIsDeclaration) {
      assert(isa<GlobalAlias>(&Dest));
      LinkFromSrc = false;
      return false;
    }
  }

  if (SrcIsDeclaration) {
    // If Src is external or if both Src & Dest are external..  Just link the
    // external globals, we aren't adding anything.
    if (Src.hasDLLImportStorageClass()) {
      // If one of GVs is marked as DLLImport, result should be dllimport'ed.
      LinkFromSrc = DestIsDeclaration;
      return false;
    }
    // If the Dest is weak, use the source linkage.
    if (Dest.hasExternalWeakLinkage()) {
      LinkFromSrc = true;
      return false;
    }
    // Link an available_externally over a declaration.
    LinkFromSrc = !Src.isDeclaration() && Dest.isDeclaration();
    return false;
  }

  if (DestIsDeclaration) {
    // If Dest is external but Src is not:
    LinkFromSrc = true;
    return false;
  }

  if (Src.hasCommonLinkage()) {
    if (Dest.hasLinkOnceLinkage() || Dest.hasWeakLinkage()) {
      LinkFromSrc = true;
      return false;
    }

    if (!Dest.hasCommonLinkage()) {
      LinkFromSrc = false;
      return false;
    }

    const DataLayout &DL = Dest.getParent()->getDataLayout();
    uint64_t DestSize = DL.getTypeAllocSize(Dest.getType()->getElementType());
    uint64_t SrcSize = DL.getTypeAllocSize(Src.getType()->getElementType());
    LinkFromSrc = SrcSize > DestSize;
    return false;
  }

  if (Src.isWeakForLinker()) {
    assert(!Dest.hasExternalWeakLinkage());
    assert(!Dest.hasAvailableExternallyLinkage());

    if (Dest.hasLinkOnceLinkage() && Src.hasWeakLinkage()) {
      LinkFromSrc = true;
      return false;
    }

    LinkFromSrc = false;
    return false;
  }

  if (Dest.isWeakForLinker()) {
    assert(Src.hasExternalLinkage());
    LinkFromSrc = true;
    return false;
  }

  assert(!Src.hasExternalWeakLinkage());
  assert(!Dest.hasExternalWeakLinkage());
  assert(Dest.hasExternalLinkage() && Src.hasExternalLinkage() &&
         "Unexpected linkage type!");
  return emitError("Linking globals named '" + Src.getName() +
                   "': symbol multiply defined!");
}

bool ModuleLinker::linkIfNeeded(GlobalValue &GV) {
  GlobalValue *DGV = getLinkedToGlobal(&GV);

  if (shouldLinkOnlyNeeded() && !(DGV && DGV->isDeclaration()))
    return false;

  if (DGV && !GV.hasLocalLinkage() && !GV.hasAppendingLinkage()) {
    auto *DGVar = dyn_cast<GlobalVariable>(DGV);
    auto *SGVar = dyn_cast<GlobalVariable>(&GV);
    if (DGVar && SGVar) {
      if (DGVar->isDeclaration() && SGVar->isDeclaration() &&
          (!DGVar->isConstant() || !SGVar->isConstant())) {
        DGVar->setConstant(false);
        SGVar->setConstant(false);
      }
      if (DGVar->hasCommonLinkage() && SGVar->hasCommonLinkage()) {
        unsigned Align = std::max(DGVar->getAlignment(), SGVar->getAlignment());
        SGVar->setAlignment(Align);
        DGVar->setAlignment(Align);
      }
    }

    GlobalValue::VisibilityTypes Visibility =
        getMinVisibility(DGV->getVisibility(), GV.getVisibility());
    DGV->setVisibility(Visibility);
    GV.setVisibility(Visibility);

    bool HasUnnamedAddr = GV.hasUnnamedAddr() && DGV->hasUnnamedAddr();
    DGV->setUnnamedAddr(HasUnnamedAddr);
    GV.setUnnamedAddr(HasUnnamedAddr);
  }

  // Don't want to append to global_ctors list, for example, when we
  // are importing for ThinLTO, otherwise the global ctors and dtors
  // get executed multiple times for local variables (the latter causing
  // double frees).
  if (GV.hasAppendingLinkage() && isPerformingImport())
    return false;

  if (isPerformingImport() && !doImportAsDefinition(&GV))
    return false;

  if (!DGV && !shouldOverrideFromSrc() &&
      (GV.hasLocalLinkage() || GV.hasLinkOnceLinkage() ||
       GV.hasAvailableExternallyLinkage()))
    return false;

  if (GV.isDeclaration())
    return false;

  if (const Comdat *SC = GV.getComdat()) {
    bool LinkFromSrc;
    Comdat::SelectionKind SK;
    std::tie(SK, LinkFromSrc) = ComdatsChosen[SC];
    if (LinkFromSrc)
      ValuesToLink.insert(&GV);
    return false;
  }

  bool LinkFromSrc = true;
  if (DGV && shouldLinkFromSource(LinkFromSrc, *DGV, GV))
    return true;
  if (LinkFromSrc)
    ValuesToLink.insert(&GV);
  return false;
}

void ModuleLinker::addLazyFor(GlobalValue &GV, IRMover::ValueAdder Add) {
  // Add these to the internalize list
  if (!GV.hasLinkOnceLinkage())
    return;

  if (shouldInternalizeLinkedSymbols())
    Internalize.insert(GV.getName());
  Add(GV);

  const Comdat *SC = GV.getComdat();
  if (!SC)
    return;
  for (GlobalValue *GV2 : ComdatMembers[SC]) {
    if (!GV2->hasLocalLinkage() && shouldInternalizeLinkedSymbols())
      Internalize.insert(GV2->getName());
    Add(*GV2);
  }
}

void ModuleLinker::processGlobalForThinLTO(GlobalValue &GV) {
  if (GV.hasLocalLinkage() &&
      (doPromoteLocalToGlobal(&GV) || isPerformingImport())) {
    GV.setName(getName(&GV));
    GV.setLinkage(getLinkage(&GV));
    if (!GV.hasLocalLinkage())
      GV.setVisibility(GlobalValue::HiddenVisibility);
    if (isModuleExporting())
      ValuesToLink.insert(&GV);
    return;
  }
  GV.setLinkage(getLinkage(&GV));
}

void ModuleLinker::processGlobalsForThinLTO() {
  for (GlobalVariable &GV : SrcM.globals())
    processGlobalForThinLTO(GV);
  for (Function &SF : SrcM)
    processGlobalForThinLTO(SF);
  for (GlobalAlias &GA : SrcM.aliases())
    processGlobalForThinLTO(GA);
}

bool ModuleLinker::run() {
  for (const auto &SMEC : SrcM.getComdatSymbolTable()) {
    const Comdat &C = SMEC.getValue();
    if (ComdatsChosen.count(&C))
      continue;
    Comdat::SelectionKind SK;
    bool LinkFromSrc;
    if (getComdatResult(&C, SK, LinkFromSrc))
      return true;
    ComdatsChosen[&C] = std::make_pair(SK, LinkFromSrc);
  }

  for (GlobalVariable &GV : SrcM.globals())
    if (const Comdat *SC = GV.getComdat())
      ComdatMembers[SC].push_back(&GV);

  for (Function &SF : SrcM)
    if (const Comdat *SC = SF.getComdat())
      ComdatMembers[SC].push_back(&SF);

  for (GlobalAlias &GA : SrcM.aliases())
    if (const Comdat *SC = GA.getComdat())
      ComdatMembers[SC].push_back(&GA);

  // Insert all of the globals in src into the DstM module... without linking
  // initializers (which could refer to functions not yet mapped over).
  for (GlobalVariable &GV : SrcM.globals())
    if (linkIfNeeded(GV))
      return true;

  for (Function &SF : SrcM)
    if (linkIfNeeded(SF))
      return true;

  for (GlobalAlias &GA : SrcM.aliases())
    if (linkIfNeeded(GA))
      return true;

  processGlobalsForThinLTO();

  for (unsigned I = 0; I < ValuesToLink.size(); ++I) {
    GlobalValue *GV = ValuesToLink[I];
    const Comdat *SC = GV->getComdat();
    if (!SC)
      continue;
    for (GlobalValue *GV2 : ComdatMembers[SC])
      ValuesToLink.insert(GV2);
  }

  if (shouldInternalizeLinkedSymbols()) {
    for (GlobalValue *GV : ValuesToLink)
      Internalize.insert(GV->getName());
  }

  if (Mover.move(SrcM, ValuesToLink.getArrayRef(),
                 [this](GlobalValue &GV, IRMover::ValueAdder Add) {
                   addLazyFor(GV, Add);
                 }))
    return true;
  Module &DstM = Mover.getModule();
  for (auto &P : Internalize) {
    GlobalValue *GV = DstM.getNamedValue(P.first());
    GV->setLinkage(GlobalValue::InternalLinkage);
  }

  return false;
}

Linker::Linker(Module &M) : Mover(M) {}

bool Linker::linkInModule(Module &Src, unsigned Flags,
                          const FunctionInfoIndex *Index,
                          DenseSet<const GlobalValue *> *FunctionsToImport) {
  ModuleLinker TheLinker(Mover, Src, Flags, Index, FunctionsToImport);
  return TheLinker.run();
}

//===----------------------------------------------------------------------===//
// LinkModules entrypoint.
//===----------------------------------------------------------------------===//

/// This function links two modules together, with the resulting Dest module
/// modified to be the composite of the two input modules. If an error occurs,
/// true is returned and ErrorMsg (if not null) is set to indicate the problem.
/// Upon failure, the Dest module could be in a modified state, and shouldn't be
/// relied on to be consistent.
bool Linker::linkModules(Module &Dest, Module &Src, unsigned Flags) {
  Linker L(Dest);
  return L.linkInModule(Src, Flags);
}

std::unique_ptr<Module>
llvm::renameModuleForThinLTO(std::unique_ptr<Module> &M,
                             const FunctionInfoIndex *Index) {
  std::unique_ptr<llvm::Module> RenamedModule(
      new llvm::Module(M->getModuleIdentifier(), M->getContext()));
  Linker L(*RenamedModule.get());
  if (L.linkInModule(*M.get(), llvm::Linker::Flags::None, Index))
    return nullptr;
  return RenamedModule;
}

//===----------------------------------------------------------------------===//
// C API.
//===----------------------------------------------------------------------===//

static void diagnosticHandler(const DiagnosticInfo &DI, void *C) {
  auto *Message = reinterpret_cast<std::string *>(C);
  raw_string_ostream Stream(*Message);
  DiagnosticPrinterRawOStream DP(Stream);
  DI.print(DP);
}

LLVMBool LLVMLinkModules(LLVMModuleRef Dest, LLVMModuleRef Src,
                         LLVMLinkerMode Unused, char **OutMessages) {
  Module *D = unwrap(Dest);
  LLVMContext &Ctx = D->getContext();

  LLVMContext::DiagnosticHandlerTy OldDiagnosticHandler =
      Ctx.getDiagnosticHandler();
  void *OldDiagnosticContext = Ctx.getDiagnosticContext();
  std::string Message;
  Ctx.setDiagnosticHandler(diagnosticHandler, &Message, true);

  LLVMBool Result = Linker::linkModules(*D, *unwrap(Src));

  Ctx.setDiagnosticHandler(OldDiagnosticHandler, OldDiagnosticContext, true);

  if (OutMessages && Result)
    *OutMessages = strdup(Message.c_str());
  return Result;
}
