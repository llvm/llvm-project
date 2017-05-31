//===- lib/Transforms/Utils/FunctionImportUtils.cpp - Importing utilities -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the FunctionImportGlobalProcessing class, used
// to perform the necessary global value handling for function importing.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
using namespace llvm;

/// Checks if we should import SGV as a definition, otherwise import as a
/// declaration.
bool FunctionImportGlobalProcessing::doImportAsDefinition(
    const GlobalValue *SGV, DenseSet<const GlobalValue *> *GlobalsToImport) {

  // For alias, we tie the definition to the base object. Extract it and recurse
  if (auto *GA = dyn_cast<GlobalAlias>(SGV)) {
    if (GA->hasWeakAnyLinkage())
      return false;
    const GlobalObject *GO = GA->getBaseObject();
    if (!GO->hasLinkOnceODRLinkage())
      return false;
    return FunctionImportGlobalProcessing::doImportAsDefinition(
        GO, GlobalsToImport);
  }
  // Only import the globals requested for importing.
  if (GlobalsToImport->count(SGV))
    return true;
  // Otherwise no.
  return false;
}

bool FunctionImportGlobalProcessing::doImportAsDefinition(
    const GlobalValue *SGV) {
  if (!isPerformingImport())
    return false;
  return FunctionImportGlobalProcessing::doImportAsDefinition(SGV,
                                                              GlobalsToImport);
}

bool FunctionImportGlobalProcessing::doPromoteLocalToGlobal(
    const GlobalValue *SGV) {
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
  // of the module and recorded in the summary index for use when importing
  // from that module.
  auto *GVar = dyn_cast<GlobalVariable>(SGV);
  if (GVar && GVar->isConstant() && GVar->hasGlobalUnnamedAddr())
    return false;

  if (GVar && GVar->hasSection())
    // Some sections like "__DATA,__cfstring" are "magic" and promotion is not
    // allowed. Just disable promotion on any GVar with sections right now.
    return false;

  // Eventually we only need to promote functions in the exporting module that
  // are referenced by a potentially exported function (i.e. one that is in the
  // summary index).
  return true;
}

std::string FunctionImportGlobalProcessing::getName(const GlobalValue *SGV) {
  // For locals that must be promoted to global scope, ensure that
  // the promoted name uniquely identifies the copy in the original module,
  // using the ID assigned during combined index creation. When importing,
  // we rename all locals (not just those that are promoted) in order to
  // avoid naming conflicts between locals imported from different modules.
  if (SGV->hasLocalLinkage() &&
      (doPromoteLocalToGlobal(SGV) || isPerformingImport()))
    return ModuleSummaryIndex::getGlobalNameForLocal(
        SGV->getName(),
        ImportIndex.getModuleHash(SGV->getParent()->getModuleIdentifier()));
  return SGV->getName();
}

GlobalValue::LinkageTypes
FunctionImportGlobalProcessing::getLinkage(const GlobalValue *SGV) {
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
    return SGV->getLinkage();

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
    // if we try to import, so we simply return AppendingLinkage.
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

void FunctionImportGlobalProcessing::processGlobalForThinLTO(GlobalValue &GV) {
  if (GV.hasLocalLinkage() &&
      (doPromoteLocalToGlobal(&GV) || isPerformingImport())) {
    GV.setName(getName(&GV));
    GV.setLinkage(getLinkage(&GV));
    if (!GV.hasLocalLinkage())
      GV.setVisibility(GlobalValue::HiddenVisibility);
  } else
    GV.setLinkage(getLinkage(&GV));

  // Remove functions imported as available externally defs from comdats,
  // as this is a declaration for the linker, and will be dropped eventually.
  // It is illegal for comdats to contain declarations.
  auto *GO = dyn_cast_or_null<GlobalObject>(&GV);
  if (GO && GO->isDeclarationForLinker() && GO->hasComdat()) {
    // The IRMover should not have placed any imported declarations in
    // a comdat, so the only declaration that should be in a comdat
    // at this point would be a definition imported as available_externally.
    assert(GO->hasAvailableExternallyLinkage() &&
           "Expected comdat on definition (possibly available external)");
    GO->setComdat(nullptr);
  }
}

void FunctionImportGlobalProcessing::processGlobalsForThinLTO() {
  if (!moduleCanBeRenamedForThinLTO(M)) {
    // We would have blocked importing from this module by suppressing index
    // generation. We still may be able to import into this module though.
    assert(!isPerformingImport() &&
           "Should have blocked importing from module with local used in ASM");
    return;
  }

  for (GlobalVariable &GV : M.globals())
    processGlobalForThinLTO(GV);
  for (Function &SF : M)
    processGlobalForThinLTO(SF);
  for (GlobalAlias &GA : M.aliases())
    processGlobalForThinLTO(GA);
}

bool FunctionImportGlobalProcessing::run() {
  processGlobalsForThinLTO();
  return false;
}

bool llvm::renameModuleForThinLTO(
    Module &M, const ModuleSummaryIndex &Index,
    DenseSet<const GlobalValue *> *GlobalsToImport) {
  FunctionImportGlobalProcessing ThinLTOProcessing(M, Index, GlobalsToImport);
  return ThinLTOProcessing.run();
}
