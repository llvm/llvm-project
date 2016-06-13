//===-LTO.cpp - LLVM Link Time Optimizer ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements functions and classes used to support LTO.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTO.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

// Simple helper to load a module from bitcode
std::unique_ptr<Module> loadModuleFromBuffer(const MemoryBufferRef &Buffer,
                                             LLVMContext &Context, bool Lazy) {
  SMDiagnostic Err;
  ErrorOr<std::unique_ptr<Module>> ModuleOrErr(nullptr);
  if (Lazy) {
    ModuleOrErr =
        getLazyBitcodeModule(MemoryBuffer::getMemBuffer(Buffer, false), Context,
                             /* ShouldLazyLoadMetadata */ Lazy);
  } else {
    ModuleOrErr = parseBitcodeFile(Buffer, Context);
  }
  if (std::error_code EC = ModuleOrErr.getError()) {
    Err = SMDiagnostic(Buffer.getBufferIdentifier(), SourceMgr::DK_Error,
                       EC.message());
    Err.print("ThinLTO", errs());
    report_fatal_error("Can't load module, abort.");
  }
  return std::move(ModuleOrErr.get());
}

static void thinLTOResolveWeakForLinkerGUID(
    GlobalValueSummaryList &GVSummaryList, GlobalValue::GUID GUID,
    DenseSet<GlobalValueSummary *> &GlobalInvolvedWithAlias,
    function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
        isPrevailing,
    function_ref<bool(StringRef, GlobalValue::GUID)> isExported,
    function_ref<void(StringRef, GlobalValue::GUID, GlobalValue::LinkageTypes)>
        recordNewLinkage) {
  auto HasMultipleCopies = GVSummaryList.size() > 1;

  for (auto &S : GVSummaryList) {
    if (GlobalInvolvedWithAlias.count(S.get()))
      continue;
    GlobalValue::LinkageTypes OriginalLinkage = S->linkage();
    if (!GlobalValue::isWeakForLinker(OriginalLinkage))
      continue;
    // We need to emit only one of these, the first module will keep it,
    // but turned into a weak, while the others will drop it when possible.
    if (!HasMultipleCopies) {
      // Exported Linkonce needs to be promoted to not be discarded.
      if (GlobalValue::isLinkOnceLinkage(OriginalLinkage) &&
          isExported(S->modulePath(), GUID))
        S->setLinkage(GlobalValue::getWeakLinkage(
            GlobalValue::isLinkOnceODRLinkage(OriginalLinkage)));
    } else if (isPrevailing(GUID, S.get())) {
      if (GlobalValue::isLinkOnceLinkage(OriginalLinkage))
        S->setLinkage(GlobalValue::getWeakLinkage(
            GlobalValue::isLinkOnceODRLinkage(OriginalLinkage)));
    }
    // Alias can't be turned into available_externally.
    else if (!isa<AliasSummary>(S.get()) &&
             (GlobalValue::isLinkOnceODRLinkage(OriginalLinkage) ||
              GlobalValue::isWeakODRLinkage(OriginalLinkage)))
      S->setLinkage(GlobalValue::AvailableExternallyLinkage);
    if (S->linkage() != OriginalLinkage)
      recordNewLinkage(S->modulePath(), GUID, S->linkage());
  }
}

// Resolve Weak and LinkOnce values in the \p Index.
//
// We'd like to drop these functions if they are no longer referenced in the
// current module. However there is a chance that another module is still
// referencing them because of the import. We make sure we always emit at least
// one copy.
void thinLTOResolveWeakForLinkerInIndex(
    ModuleSummaryIndex &Index,
    function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
        isPrevailing,
    function_ref<bool(StringRef, GlobalValue::GUID)> isExported,
    function_ref<void(StringRef, GlobalValue::GUID, GlobalValue::LinkageTypes)>
        recordNewLinkage) {
  if (Index.modulePaths().size() == 1)
    // Nothing to do if we don't have multiple modules
    return;

  // We won't optimize the globals that are referenced by an alias for now
  // Ideally we should turn the alias into a global and duplicate the definition
  // when needed.
  DenseSet<GlobalValueSummary *> GlobalInvolvedWithAlias;
  for (auto &I : Index)
    for (auto &S : I.second)
      if (auto AS = dyn_cast<AliasSummary>(S.get()))
        GlobalInvolvedWithAlias.insert(&AS->getAliasee());

  for (auto &I : Index)
    thinLTOResolveWeakForLinkerGUID(I.second, I.first, GlobalInvolvedWithAlias,
                                    isPrevailing, isExported, recordNewLinkage);
}

static void thinLTOInternalizeAndPromoteGUID(
    GlobalValueSummaryList &GVSummaryList, GlobalValue::GUID GUID,
    function_ref<bool(StringRef, GlobalValue::GUID)> isExported) {
  for (auto &S : GVSummaryList) {
    if (isExported(S->modulePath(), GUID)) {
      if (GlobalValue::isLocalLinkage(S->linkage()))
        S->setLinkage(GlobalValue::ExternalLinkage);
    } else if (!GlobalValue::isLocalLinkage(S->linkage()))
      S->setLinkage(GlobalValue::InternalLinkage);
  }
}

// Update the linkages in the given \p Index to mark exported values
// as external and non-exported values as internal.
void thinLTOInternalizeAndPromoteInIndex(
    ModuleSummaryIndex &Index,
    function_ref<bool(StringRef, GlobalValue::GUID)> isExported) {
  for (auto &I : Index)
    thinLTOInternalizeAndPromoteGUID(I.second, I.first, isExported);
}
}
