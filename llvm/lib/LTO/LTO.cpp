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
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Metadata.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Linker/IRMover.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/VCSRevision.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/SplitModule.h"

#include <set>

using namespace llvm;
using namespace lto;
using namespace object;

#define DEBUG_TYPE "lto"

// The values are (type identifier, summary) pairs.
typedef DenseMap<
    GlobalValue::GUID,
    TinyPtrVector<const std::pair<const std::string, TypeIdSummary> *>>
    TypeIdSummariesByGuidTy;

// Returns a unique hash for the Module considering the current list of
// export/import and other global analysis results.
// The hash is produced in \p Key.
static void computeCacheKey(
    SmallString<40> &Key, const Config &Conf, const ModuleSummaryIndex &Index,
    StringRef ModuleID, const FunctionImporter::ImportMapTy &ImportList,
    const FunctionImporter::ExportSetTy &ExportList,
    const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
    const GVSummaryMapTy &DefinedGlobals,
    const TypeIdSummariesByGuidTy &TypeIdSummariesByGuid) {
  // Compute the unique hash for this entry.
  // This is based on the current compiler version, the module itself, the
  // export list, the hash for every single module in the import list, the
  // list of ResolvedODR for the module, and the list of preserved symbols.
  SHA1 Hasher;

  // Start with the compiler revision
  Hasher.update(LLVM_VERSION_STRING);
#ifdef LLVM_REVISION
  Hasher.update(LLVM_REVISION);
#endif

  // Include the parts of the LTO configuration that affect code generation.
  auto AddString = [&](StringRef Str) {
    Hasher.update(Str);
    Hasher.update(ArrayRef<uint8_t>{0});
  };
  auto AddUnsigned = [&](unsigned I) {
    uint8_t Data[4];
    Data[0] = I;
    Data[1] = I >> 8;
    Data[2] = I >> 16;
    Data[3] = I >> 24;
    Hasher.update(ArrayRef<uint8_t>{Data, 4});
  };
  auto AddUint64 = [&](uint64_t I) {
    uint8_t Data[8];
    Data[0] = I;
    Data[1] = I >> 8;
    Data[2] = I >> 16;
    Data[3] = I >> 24;
    Data[4] = I >> 32;
    Data[5] = I >> 40;
    Data[6] = I >> 48;
    Data[7] = I >> 56;
    Hasher.update(ArrayRef<uint8_t>{Data, 8});
  };
  AddString(Conf.CPU);
  // FIXME: Hash more of Options. For now all clients initialize Options from
  // command-line flags (which is unsupported in production), but may set
  // RelaxELFRelocations. The clang driver can also pass FunctionSections,
  // DataSections and DebuggerTuning via command line flags.
  AddUnsigned(Conf.Options.RelaxELFRelocations);
  AddUnsigned(Conf.Options.FunctionSections);
  AddUnsigned(Conf.Options.DataSections);
  AddUnsigned((unsigned)Conf.Options.DebuggerTuning);
  for (auto &A : Conf.MAttrs)
    AddString(A);
  if (Conf.RelocModel)
    AddUnsigned(*Conf.RelocModel);
  else
    AddUnsigned(-1);
  AddUnsigned(Conf.CodeModel);
  AddUnsigned(Conf.CGOptLevel);
  AddUnsigned(Conf.CGFileType);
  AddUnsigned(Conf.OptLevel);
  AddUnsigned(Conf.UseNewPM);
  AddString(Conf.OptPipeline);
  AddString(Conf.AAPipeline);
  AddString(Conf.OverrideTriple);
  AddString(Conf.DefaultTriple);

  // Include the hash for the current module
  auto ModHash = Index.getModuleHash(ModuleID);
  Hasher.update(ArrayRef<uint8_t>((uint8_t *)&ModHash[0], sizeof(ModHash)));
  for (auto F : ExportList)
    // The export list can impact the internalization, be conservative here
    Hasher.update(ArrayRef<uint8_t>((uint8_t *)&F, sizeof(F)));

  // Include the hash for every module we import functions from. The set of
  // imported symbols for each module may affect code generation and is
  // sensitive to link order, so include that as well.
  for (auto &Entry : ImportList) {
    auto ModHash = Index.getModuleHash(Entry.first());
    Hasher.update(ArrayRef<uint8_t>((uint8_t *)&ModHash[0], sizeof(ModHash)));

    AddUint64(Entry.second.size());
    for (auto &Fn : Entry.second)
      AddUint64(Fn.first);
  }

  // Include the hash for the resolved ODR.
  for (auto &Entry : ResolvedODR) {
    Hasher.update(ArrayRef<uint8_t>((const uint8_t *)&Entry.first,
                                    sizeof(GlobalValue::GUID)));
    Hasher.update(ArrayRef<uint8_t>((const uint8_t *)&Entry.second,
                                    sizeof(GlobalValue::LinkageTypes)));
  }

  std::set<GlobalValue::GUID> UsedTypeIds;

  auto AddUsedTypeIds = [&](GlobalValueSummary *GS) {
    auto *FS = dyn_cast_or_null<FunctionSummary>(GS);
    if (!FS)
      return;
    for (auto &TT : FS->type_tests())
      UsedTypeIds.insert(TT);
    for (auto &TT : FS->type_test_assume_vcalls())
      UsedTypeIds.insert(TT.GUID);
    for (auto &TT : FS->type_checked_load_vcalls())
      UsedTypeIds.insert(TT.GUID);
    for (auto &TT : FS->type_test_assume_const_vcalls())
      UsedTypeIds.insert(TT.VFunc.GUID);
    for (auto &TT : FS->type_checked_load_const_vcalls())
      UsedTypeIds.insert(TT.VFunc.GUID);
  };

  // Include the hash for the linkage type to reflect internalization and weak
  // resolution, and collect any used type identifier resolutions.
  for (auto &GS : DefinedGlobals) {
    GlobalValue::LinkageTypes Linkage = GS.second->linkage();
    Hasher.update(
        ArrayRef<uint8_t>((const uint8_t *)&Linkage, sizeof(Linkage)));
    AddUsedTypeIds(GS.second);
  }

  // Imported functions may introduce new uses of type identifier resolutions,
  // so we need to collect their used resolutions as well.
  for (auto &ImpM : ImportList)
    for (auto &ImpF : ImpM.second)
      AddUsedTypeIds(Index.findSummaryInModule(ImpF.first, ImpM.first()));

  auto AddTypeIdSummary = [&](StringRef TId, const TypeIdSummary &S) {
    AddString(TId);

    AddUnsigned(S.TTRes.TheKind);
    AddUnsigned(S.TTRes.SizeM1BitWidth);

    AddUint64(S.WPDRes.size());
    for (auto &WPD : S.WPDRes) {
      AddUnsigned(WPD.first);
      AddUnsigned(WPD.second.TheKind);
      AddString(WPD.second.SingleImplName);

      AddUint64(WPD.second.ResByArg.size());
      for (auto &ByArg : WPD.second.ResByArg) {
        AddUint64(ByArg.first.size());
        for (uint64_t Arg : ByArg.first)
          AddUint64(Arg);
        AddUnsigned(ByArg.second.TheKind);
        AddUint64(ByArg.second.Info);
      }
    }
  };

  // Include the hash for all type identifiers used by this module.
  for (GlobalValue::GUID TId : UsedTypeIds) {
    auto SummariesI = TypeIdSummariesByGuid.find(TId);
    if (SummariesI != TypeIdSummariesByGuid.end())
      for (auto *Summary : SummariesI->second)
        AddTypeIdSummary(Summary->first, Summary->second);
  }

  if (!Conf.SampleProfile.empty()) {
    auto FileOrErr = MemoryBuffer::getFile(Conf.SampleProfile);
    if (FileOrErr)
      Hasher.update(FileOrErr.get()->getBuffer());
  }

  Key = toHex(Hasher.result());
}

static void thinLTOResolveWeakForLinkerGUID(
    GlobalValueSummaryList &GVSummaryList, GlobalValue::GUID GUID,
    DenseSet<GlobalValueSummary *> &GlobalInvolvedWithAlias,
    function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
        isPrevailing,
    function_ref<void(StringRef, GlobalValue::GUID, GlobalValue::LinkageTypes)>
        recordNewLinkage) {
  for (auto &S : GVSummaryList) {
    GlobalValue::LinkageTypes OriginalLinkage = S->linkage();
    if (!GlobalValue::isWeakForLinker(OriginalLinkage))
      continue;
    // We need to emit only one of these. The prevailing module will keep it,
    // but turned into a weak, while the others will drop it when possible.
    // This is both a compile-time optimization and a correctness
    // transformation. This is necessary for correctness when we have exported
    // a reference - we need to convert the linkonce to weak to
    // ensure a copy is kept to satisfy the exported reference.
    // FIXME: We may want to split the compile time and correctness
    // aspects into separate routines.
    if (isPrevailing(GUID, S.get())) {
      if (GlobalValue::isLinkOnceLinkage(OriginalLinkage))
        S->setLinkage(GlobalValue::getWeakLinkage(
            GlobalValue::isLinkOnceODRLinkage(OriginalLinkage)));
    }
    // Alias and aliasee can't be turned into available_externally.
    else if (!isa<AliasSummary>(S.get()) &&
             !GlobalInvolvedWithAlias.count(S.get()))
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
void llvm::thinLTOResolveWeakForLinkerInIndex(
    ModuleSummaryIndex &Index,
    function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
        isPrevailing,
    function_ref<void(StringRef, GlobalValue::GUID, GlobalValue::LinkageTypes)>
        recordNewLinkage) {
  // We won't optimize the globals that are referenced by an alias for now
  // Ideally we should turn the alias into a global and duplicate the definition
  // when needed.
  DenseSet<GlobalValueSummary *> GlobalInvolvedWithAlias;
  for (auto &I : Index)
    for (auto &S : I.second.SummaryList)
      if (auto AS = dyn_cast<AliasSummary>(S.get()))
        GlobalInvolvedWithAlias.insert(&AS->getAliasee());

  for (auto &I : Index)
    thinLTOResolveWeakForLinkerGUID(I.second.SummaryList, I.first,
                                    GlobalInvolvedWithAlias, isPrevailing,
                                    recordNewLinkage);
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
void llvm::thinLTOInternalizeAndPromoteInIndex(
    ModuleSummaryIndex &Index,
    function_ref<bool(StringRef, GlobalValue::GUID)> isExported) {
  for (auto &I : Index)
    thinLTOInternalizeAndPromoteGUID(I.second.SummaryList, I.first, isExported);
}

// Requires a destructor for std::vector<InputModule>.
InputFile::~InputFile() = default;

Expected<std::unique_ptr<InputFile>> InputFile::create(MemoryBufferRef Object) {
  std::unique_ptr<InputFile> File(new InputFile);

  Expected<IRSymtabFile> FOrErr = readIRSymtab(Object);
  if (!FOrErr)
    return FOrErr.takeError();

  File->TargetTriple = FOrErr->TheReader.getTargetTriple();
  File->SourceFileName = FOrErr->TheReader.getSourceFileName();
  File->COFFLinkerOpts = FOrErr->TheReader.getCOFFLinkerOpts();
  File->ComdatTable = FOrErr->TheReader.getComdatTable();

  for (unsigned I = 0; I != FOrErr->Mods.size(); ++I) {
    size_t Begin = File->Symbols.size();
    for (const irsymtab::Reader::SymbolRef &Sym :
         FOrErr->TheReader.module_symbols(I))
      // Skip symbols that are irrelevant to LTO. Note that this condition needs
      // to match the one in Skip() in LTO::addRegularLTO().
      if (Sym.isGlobal() && !Sym.isFormatSpecific())
        File->Symbols.push_back(Sym);
    File->ModuleSymIndices.push_back({Begin, File->Symbols.size()});
  }

  File->Mods = FOrErr->Mods;
  File->Strtab = std::move(FOrErr->Strtab);
  return std::move(File);
}

StringRef InputFile::getName() const {
  return Mods[0].getModuleIdentifier();
}

LTO::RegularLTOState::RegularLTOState(unsigned ParallelCodeGenParallelismLevel,
                                      Config &Conf)
    : ParallelCodeGenParallelismLevel(ParallelCodeGenParallelismLevel),
      Ctx(Conf) {}

LTO::ThinLTOState::ThinLTOState(ThinBackend Backend) : Backend(Backend) {
  if (!Backend)
    this->Backend =
        createInProcessThinBackend(llvm::heavyweight_hardware_concurrency());
}

LTO::LTO(Config Conf, ThinBackend Backend,
         unsigned ParallelCodeGenParallelismLevel)
    : Conf(std::move(Conf)),
      RegularLTO(ParallelCodeGenParallelismLevel, this->Conf),
      ThinLTO(std::move(Backend)) {}

// Requires a destructor for MapVector<BitcodeModule>.
LTO::~LTO() = default;

// Add the symbols in the given module to the GlobalResolutions map, and resolve
// their partitions.
void LTO::addModuleToGlobalRes(ArrayRef<InputFile::Symbol> Syms,
                               ArrayRef<SymbolResolution> Res,
                               unsigned Partition, bool InSummary) {
  auto *ResI = Res.begin();
  auto *ResE = Res.end();
  (void)ResE;
  for (const InputFile::Symbol &Sym : Syms) {
    assert(ResI != ResE);
    SymbolResolution Res = *ResI++;

    auto &GlobalRes = GlobalResolutions[Sym.getName()];
    GlobalRes.UnnamedAddr &= Sym.isUnnamedAddr();
    if (Res.Prevailing)
      GlobalRes.IRName = Sym.getIRName();

    // Set the partition to external if we know it is re-defined by the linker
    // with -defsym or -wrap options, used elsewhere, e.g. it is visible to a
    // regular object, is referenced from llvm.compiler_used, or was already
    // recorded as being referenced from a different partition.
    if (Res.LinkerRedefined || Res.VisibleToRegularObj || Sym.isUsed() ||
        (GlobalRes.Partition != GlobalResolution::Unknown &&
         GlobalRes.Partition != Partition)) {
      GlobalRes.Partition = GlobalResolution::External;
    } else
      // First recorded reference, save the current partition.
      GlobalRes.Partition = Partition;

    // Flag as visible outside of summary if visible from a regular object or
    // from a module that does not have a summary.
    GlobalRes.VisibleOutsideSummary |=
        (Res.VisibleToRegularObj || Sym.isUsed() || !InSummary);
  }
}

static void writeToResolutionFile(raw_ostream &OS, InputFile *Input,
                                  ArrayRef<SymbolResolution> Res) {
  StringRef Path = Input->getName();
  OS << Path << '\n';
  auto ResI = Res.begin();
  for (const InputFile::Symbol &Sym : Input->symbols()) {
    assert(ResI != Res.end());
    SymbolResolution Res = *ResI++;

    OS << "-r=" << Path << ',' << Sym.getName() << ',';
    if (Res.Prevailing)
      OS << 'p';
    if (Res.FinalDefinitionInLinkageUnit)
      OS << 'l';
    if (Res.VisibleToRegularObj)
      OS << 'x';
    if (Res.LinkerRedefined)
      OS << 'r';
    OS << '\n';
  }
  OS.flush();
  assert(ResI == Res.end());
}

Error LTO::add(std::unique_ptr<InputFile> Input,
               ArrayRef<SymbolResolution> Res) {
  assert(!CalledGetMaxTasks);

  if (Conf.ResolutionFile)
    writeToResolutionFile(*Conf.ResolutionFile, Input.get(), Res);

  const SymbolResolution *ResI = Res.begin();
  for (unsigned I = 0; I != Input->Mods.size(); ++I)
    if (Error Err = addModule(*Input, I, ResI, Res.end()))
      return Err;

  assert(ResI == Res.end());
  return Error::success();
}

Error LTO::addModule(InputFile &Input, unsigned ModI,
                     const SymbolResolution *&ResI,
                     const SymbolResolution *ResE) {
  Expected<BitcodeLTOInfo> LTOInfo = Input.Mods[ModI].getLTOInfo();
  if (!LTOInfo)
    return LTOInfo.takeError();

  BitcodeModule BM = Input.Mods[ModI];
  auto ModSyms = Input.module_symbols(ModI);
  addModuleToGlobalRes(ModSyms, {ResI, ResE},
                       LTOInfo->IsThinLTO ? ThinLTO.ModuleMap.size() + 1 : 0,
                       LTOInfo->HasSummary);

  if (LTOInfo->IsThinLTO)
    return addThinLTO(BM, ModSyms, ResI, ResE);

  Expected<RegularLTOState::AddedModule> ModOrErr =
      addRegularLTO(BM, ModSyms, ResI, ResE);
  if (!ModOrErr)
    return ModOrErr.takeError();

  if (!LTOInfo->HasSummary)
    return linkRegularLTO(std::move(*ModOrErr), /*LivenessFromIndex=*/false);

  // Regular LTO module summaries are added to a dummy module that represents
  // the combined regular LTO module.
  if (Error Err = BM.readSummary(ThinLTO.CombinedIndex, "", -1ull))
    return Err;
  RegularLTO.ModsWithSummaries.push_back(std::move(*ModOrErr));
  return Error::success();
}

// Add a regular LTO object to the link.
// The resulting module needs to be linked into the combined LTO module with
// linkRegularLTO.
Expected<LTO::RegularLTOState::AddedModule>
LTO::addRegularLTO(BitcodeModule BM, ArrayRef<InputFile::Symbol> Syms,
                   const SymbolResolution *&ResI,
                   const SymbolResolution *ResE) {
  RegularLTOState::AddedModule Mod;
  Expected<std::unique_ptr<Module>> MOrErr =
      BM.getLazyModule(RegularLTO.Ctx, /*ShouldLazyLoadMetadata*/ true,
                       /*IsImporting*/ false);
  if (!MOrErr)
    return MOrErr.takeError();
  Module &M = **MOrErr;
  Mod.M = std::move(*MOrErr);

  if (Error Err = M.materializeMetadata())
    return std::move(Err);
  UpgradeDebugInfo(M);

  ModuleSymbolTable SymTab;
  SymTab.addModule(&M);

  for (GlobalVariable &GV : M.globals())
    if (GV.hasAppendingLinkage())
      Mod.Keep.push_back(&GV);

  DenseSet<GlobalObject *> AliasedGlobals;
  for (auto &GA : M.aliases())
    if (GlobalObject *GO = GA.getBaseObject())
      AliasedGlobals.insert(GO);

  // In this function we need IR GlobalValues matching the symbols in Syms
  // (which is not backed by a module), so we need to enumerate them in the same
  // order. The symbol enumeration order of a ModuleSymbolTable intentionally
  // matches the order of an irsymtab, but when we read the irsymtab in
  // InputFile::create we omit some symbols that are irrelevant to LTO. The
  // Skip() function skips the same symbols from the module as InputFile does
  // from the symbol table.
  auto MsymI = SymTab.symbols().begin(), MsymE = SymTab.symbols().end();
  auto Skip = [&]() {
    while (MsymI != MsymE) {
      auto Flags = SymTab.getSymbolFlags(*MsymI);
      if ((Flags & object::BasicSymbolRef::SF_Global) &&
          !(Flags & object::BasicSymbolRef::SF_FormatSpecific))
        return;
      ++MsymI;
    }
  };
  Skip();

  for (const InputFile::Symbol &Sym : Syms) {
    assert(ResI != ResE);
    SymbolResolution Res = *ResI++;

    assert(MsymI != MsymE);
    ModuleSymbolTable::Symbol Msym = *MsymI++;
    Skip();

    if (GlobalValue *GV = Msym.dyn_cast<GlobalValue *>()) {
      if (Res.Prevailing) {
        if (Sym.isUndefined())
          continue;
        Mod.Keep.push_back(GV);
        // For symbols re-defined with linker -wrap and -defsym options,
        // set the linkage to weak to inhibit IPO. The linkage will be
        // restored by the linker.
        if (Res.LinkerRedefined)
          GV->setLinkage(GlobalValue::WeakAnyLinkage);

        GlobalValue::LinkageTypes OriginalLinkage = GV->getLinkage();
        if (GlobalValue::isLinkOnceLinkage(OriginalLinkage))
          GV->setLinkage(GlobalValue::getWeakLinkage(
              GlobalValue::isLinkOnceODRLinkage(OriginalLinkage)));
      } else if (isa<GlobalObject>(GV) &&
                 (GV->hasLinkOnceODRLinkage() || GV->hasWeakODRLinkage() ||
                  GV->hasAvailableExternallyLinkage()) &&
                 !AliasedGlobals.count(cast<GlobalObject>(GV))) {
        // Any of the above three types of linkage indicates that the
        // chosen prevailing symbol will have the same semantics as this copy of
        // the symbol, so we may be able to link it with available_externally
        // linkage. We will decide later whether to do that when we link this
        // module (in linkRegularLTO), based on whether it is undefined.
        Mod.Keep.push_back(GV);
        GV->setLinkage(GlobalValue::AvailableExternallyLinkage);
        cast<GlobalObject>(GV)->setComdat(nullptr);
      }
    }
    // Common resolution: collect the maximum size/alignment over all commons.
    // We also record if we see an instance of a common as prevailing, so that
    // if none is prevailing we can ignore it later.
    if (Sym.isCommon()) {
      // FIXME: We should figure out what to do about commons defined by asm.
      // For now they aren't reported correctly by ModuleSymbolTable.
      auto &CommonRes = RegularLTO.Commons[Sym.getIRName()];
      CommonRes.Size = std::max(CommonRes.Size, Sym.getCommonSize());
      CommonRes.Align = std::max(CommonRes.Align, Sym.getCommonAlignment());
      CommonRes.Prevailing |= Res.Prevailing;
    }

    // FIXME: use proposed local attribute for FinalDefinitionInLinkageUnit.
  }
  assert(MsymI == MsymE);
  return std::move(Mod);
}

Error LTO::linkRegularLTO(RegularLTOState::AddedModule Mod,
                          bool LivenessFromIndex) {
  if (!RegularLTO.CombinedModule) {
    RegularLTO.CombinedModule =
        llvm::make_unique<Module>("ld-temp.o", RegularLTO.Ctx);
    RegularLTO.Mover = llvm::make_unique<IRMover>(*RegularLTO.CombinedModule);
  }

  std::vector<GlobalValue *> Keep;
  for (GlobalValue *GV : Mod.Keep) {
    if (LivenessFromIndex && !ThinLTO.CombinedIndex.isGUIDLive(GV->getGUID()))
      continue;

    if (!GV->hasAvailableExternallyLinkage()) {
      Keep.push_back(GV);
      continue;
    }

    // Only link available_externally definitions if we don't already have a
    // definition.
    GlobalValue *CombinedGV =
        RegularLTO.CombinedModule->getNamedValue(GV->getName());
    if (CombinedGV && !CombinedGV->isDeclaration())
      continue;

    Keep.push_back(GV);
  }

  return RegularLTO.Mover->move(std::move(Mod.M), Keep,
                                [](GlobalValue &, IRMover::ValueAdder) {},
                                /* IsPerformingImport */ false);
}

// Add a ThinLTO module to the link.
Error LTO::addThinLTO(BitcodeModule BM, ArrayRef<InputFile::Symbol> Syms,
                      const SymbolResolution *&ResI,
                      const SymbolResolution *ResE) {
  if (Error Err =
          BM.readSummary(ThinLTO.CombinedIndex, BM.getModuleIdentifier(),
                         ThinLTO.ModuleMap.size()))
    return Err;

  for (const InputFile::Symbol &Sym : Syms) {
    assert(ResI != ResE);
    SymbolResolution Res = *ResI++;

    if (Res.Prevailing) {
      if (!Sym.getIRName().empty()) {
        auto GUID = GlobalValue::getGUID(GlobalValue::getGlobalIdentifier(
            Sym.getIRName(), GlobalValue::ExternalLinkage, ""));
        ThinLTO.PrevailingModuleForGUID[GUID] = BM.getModuleIdentifier();
      }
    }
  }

  if (!ThinLTO.ModuleMap.insert({BM.getModuleIdentifier(), BM}).second)
    return make_error<StringError>(
        "Expected at most one ThinLTO module per bitcode file",
        inconvertibleErrorCode());

  return Error::success();
}

unsigned LTO::getMaxTasks() const {
  CalledGetMaxTasks = true;
  return RegularLTO.ParallelCodeGenParallelismLevel + ThinLTO.ModuleMap.size();
}

Error LTO::run(AddStreamFn AddStream, NativeObjectCache Cache) {
  // Compute "dead" symbols, we don't want to import/export these!
  DenseSet<GlobalValue::GUID> GUIDPreservedSymbols;
  for (auto &Res : GlobalResolutions) {
    if (Res.second.VisibleOutsideSummary &&
        // IRName will be defined if we have seen the prevailing copy of
        // this value. If not, no need to preserve any ThinLTO copies.
        !Res.second.IRName.empty())
      GUIDPreservedSymbols.insert(GlobalValue::getGUID(
          GlobalValue::dropLLVMManglingEscape(Res.second.IRName)));
  }

  computeDeadSymbols(ThinLTO.CombinedIndex, GUIDPreservedSymbols);

  // Save the status of having a regularLTO combined module, as
  // this is needed for generating the ThinLTO Task ID, and
  // the CombinedModule will be moved at the end of runRegularLTO.
  bool HasRegularLTO = RegularLTO.CombinedModule != nullptr ||
                       !RegularLTO.ModsWithSummaries.empty();
  // Invoke regular LTO if there was a regular LTO module to start with.
  if (HasRegularLTO)
    if (auto E = runRegularLTO(AddStream))
      return E;
  return runThinLTO(AddStream, Cache, HasRegularLTO);
}

Error LTO::runRegularLTO(AddStreamFn AddStream) {
  for (auto &M : RegularLTO.ModsWithSummaries)
    if (Error Err = linkRegularLTO(std::move(M),
                                   /*LivenessFromIndex=*/true))
      return Err;

  // Make sure commons have the right size/alignment: we kept the largest from
  // all the prevailing when adding the inputs, and we apply it here.
  const DataLayout &DL = RegularLTO.CombinedModule->getDataLayout();
  for (auto &I : RegularLTO.Commons) {
    if (!I.second.Prevailing)
      // Don't do anything if no instance of this common was prevailing.
      continue;
    GlobalVariable *OldGV = RegularLTO.CombinedModule->getNamedGlobal(I.first);
    if (OldGV && DL.getTypeAllocSize(OldGV->getValueType()) == I.second.Size) {
      // Don't create a new global if the type is already correct, just make
      // sure the alignment is correct.
      OldGV->setAlignment(I.second.Align);
      continue;
    }
    ArrayType *Ty =
        ArrayType::get(Type::getInt8Ty(RegularLTO.Ctx), I.second.Size);
    auto *GV = new GlobalVariable(*RegularLTO.CombinedModule, Ty, false,
                                  GlobalValue::CommonLinkage,
                                  ConstantAggregateZero::get(Ty), "");
    GV->setAlignment(I.second.Align);
    if (OldGV) {
      OldGV->replaceAllUsesWith(ConstantExpr::getBitCast(GV, OldGV->getType()));
      GV->takeName(OldGV);
      OldGV->eraseFromParent();
    } else {
      GV->setName(I.first);
    }
  }

  if (Conf.PreOptModuleHook &&
      !Conf.PreOptModuleHook(0, *RegularLTO.CombinedModule))
    return Error::success();

  if (!Conf.CodeGenOnly) {
    for (const auto &R : GlobalResolutions) {
      if (R.second.IRName.empty())
        continue;
      if (R.second.Partition != 0 &&
          R.second.Partition != GlobalResolution::External)
        continue;

      GlobalValue *GV =
          RegularLTO.CombinedModule->getNamedValue(R.second.IRName);
      // Ignore symbols defined in other partitions.
      if (!GV || GV->hasLocalLinkage())
        continue;
      GV->setUnnamedAddr(R.second.UnnamedAddr ? GlobalValue::UnnamedAddr::Global
                                              : GlobalValue::UnnamedAddr::None);
      if (R.second.Partition == 0)
        GV->setLinkage(GlobalValue::InternalLinkage);
    }

    if (Conf.PostInternalizeModuleHook &&
        !Conf.PostInternalizeModuleHook(0, *RegularLTO.CombinedModule))
      return Error::success();
  }
  return backend(Conf, AddStream, RegularLTO.ParallelCodeGenParallelismLevel,
                 std::move(RegularLTO.CombinedModule), ThinLTO.CombinedIndex);
}

/// This class defines the interface to the ThinLTO backend.
class lto::ThinBackendProc {
protected:
  Config &Conf;
  ModuleSummaryIndex &CombinedIndex;
  const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries;

public:
  ThinBackendProc(Config &Conf, ModuleSummaryIndex &CombinedIndex,
                  const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries)
      : Conf(Conf), CombinedIndex(CombinedIndex),
        ModuleToDefinedGVSummaries(ModuleToDefinedGVSummaries) {}

  virtual ~ThinBackendProc() {}
  virtual Error start(
      unsigned Task, BitcodeModule BM,
      const FunctionImporter::ImportMapTy &ImportList,
      const FunctionImporter::ExportSetTy &ExportList,
      const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
      MapVector<StringRef, BitcodeModule> &ModuleMap) = 0;
  virtual Error wait() = 0;
};

namespace {
class InProcessThinBackend : public ThinBackendProc {
  ThreadPool BackendThreadPool;
  AddStreamFn AddStream;
  NativeObjectCache Cache;
  TypeIdSummariesByGuidTy TypeIdSummariesByGuid;

  Optional<Error> Err;
  std::mutex ErrMu;

public:
  InProcessThinBackend(
      Config &Conf, ModuleSummaryIndex &CombinedIndex,
      unsigned ThinLTOParallelismLevel,
      const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
      AddStreamFn AddStream, NativeObjectCache Cache)
      : ThinBackendProc(Conf, CombinedIndex, ModuleToDefinedGVSummaries),
        BackendThreadPool(ThinLTOParallelismLevel),
        AddStream(std::move(AddStream)), Cache(std::move(Cache)) {
    // Create a mapping from type identifier GUIDs to type identifier summaries.
    // This allows backends to use the type identifier GUIDs stored in the
    // function summaries to determine which type identifier summaries affect
    // each function without needing to compute GUIDs in each backend.
    for (auto &TId : CombinedIndex.typeIds())
      TypeIdSummariesByGuid[GlobalValue::getGUID(TId.first)].push_back(&TId);
  }

  Error runThinLTOBackendThread(
      AddStreamFn AddStream, NativeObjectCache Cache, unsigned Task,
      BitcodeModule BM, ModuleSummaryIndex &CombinedIndex,
      const FunctionImporter::ImportMapTy &ImportList,
      const FunctionImporter::ExportSetTy &ExportList,
      const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
      const GVSummaryMapTy &DefinedGlobals,
      MapVector<StringRef, BitcodeModule> &ModuleMap,
      const TypeIdSummariesByGuidTy &TypeIdSummariesByGuid) {
    auto RunThinBackend = [&](AddStreamFn AddStream) {
      LTOLLVMContext BackendContext(Conf);
      Expected<std::unique_ptr<Module>> MOrErr = BM.parseModule(BackendContext);
      if (!MOrErr)
        return MOrErr.takeError();

      return thinBackend(Conf, Task, AddStream, **MOrErr, CombinedIndex,
                         ImportList, DefinedGlobals, ModuleMap);
    };

    auto ModuleID = BM.getModuleIdentifier();

    if (!Cache || !CombinedIndex.modulePaths().count(ModuleID) ||
        all_of(CombinedIndex.getModuleHash(ModuleID),
               [](uint32_t V) { return V == 0; }))
      // Cache disabled or no entry for this module in the combined index or
      // no module hash.
      return RunThinBackend(AddStream);

    SmallString<40> Key;
    // The module may be cached, this helps handling it.
    computeCacheKey(Key, Conf, CombinedIndex, ModuleID, ImportList, ExportList,
                    ResolvedODR, DefinedGlobals, TypeIdSummariesByGuid);
    if (AddStreamFn CacheAddStream = Cache(Task, Key))
      return RunThinBackend(CacheAddStream);

    return Error::success();
  }

  Error start(
      unsigned Task, BitcodeModule BM,
      const FunctionImporter::ImportMapTy &ImportList,
      const FunctionImporter::ExportSetTy &ExportList,
      const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
      MapVector<StringRef, BitcodeModule> &ModuleMap) override {
    StringRef ModulePath = BM.getModuleIdentifier();
    assert(ModuleToDefinedGVSummaries.count(ModulePath));
    const GVSummaryMapTy &DefinedGlobals =
        ModuleToDefinedGVSummaries.find(ModulePath)->second;
    BackendThreadPool.async(
        [=](BitcodeModule BM, ModuleSummaryIndex &CombinedIndex,
            const FunctionImporter::ImportMapTy &ImportList,
            const FunctionImporter::ExportSetTy &ExportList,
            const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes>
                &ResolvedODR,
            const GVSummaryMapTy &DefinedGlobals,
            MapVector<StringRef, BitcodeModule> &ModuleMap,
            const TypeIdSummariesByGuidTy &TypeIdSummariesByGuid) {
          Error E = runThinLTOBackendThread(
              AddStream, Cache, Task, BM, CombinedIndex, ImportList, ExportList,
              ResolvedODR, DefinedGlobals, ModuleMap, TypeIdSummariesByGuid);
          if (E) {
            std::unique_lock<std::mutex> L(ErrMu);
            if (Err)
              Err = joinErrors(std::move(*Err), std::move(E));
            else
              Err = std::move(E);
          }
        },
        BM, std::ref(CombinedIndex), std::ref(ImportList), std::ref(ExportList),
        std::ref(ResolvedODR), std::ref(DefinedGlobals), std::ref(ModuleMap),
        std::ref(TypeIdSummariesByGuid));
    return Error::success();
  }

  Error wait() override {
    BackendThreadPool.wait();
    if (Err)
      return std::move(*Err);
    else
      return Error::success();
  }
};
} // end anonymous namespace

ThinBackend lto::createInProcessThinBackend(unsigned ParallelismLevel) {
  return [=](Config &Conf, ModuleSummaryIndex &CombinedIndex,
             const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
             AddStreamFn AddStream, NativeObjectCache Cache) {
    return llvm::make_unique<InProcessThinBackend>(
        Conf, CombinedIndex, ParallelismLevel, ModuleToDefinedGVSummaries,
        AddStream, Cache);
  };
}

// Given the original \p Path to an output file, replace any path
// prefix matching \p OldPrefix with \p NewPrefix. Also, create the
// resulting directory if it does not yet exist.
std::string lto::getThinLTOOutputFile(const std::string &Path,
                                      const std::string &OldPrefix,
                                      const std::string &NewPrefix) {
  if (OldPrefix.empty() && NewPrefix.empty())
    return Path;
  SmallString<128> NewPath(Path);
  llvm::sys::path::replace_path_prefix(NewPath, OldPrefix, NewPrefix);
  StringRef ParentPath = llvm::sys::path::parent_path(NewPath.str());
  if (!ParentPath.empty()) {
    // Make sure the new directory exists, creating it if necessary.
    if (std::error_code EC = llvm::sys::fs::create_directories(ParentPath))
      llvm::errs() << "warning: could not create directory '" << ParentPath
                   << "': " << EC.message() << '\n';
  }
  return NewPath.str();
}

namespace {
class WriteIndexesThinBackend : public ThinBackendProc {
  std::string OldPrefix, NewPrefix;
  bool ShouldEmitImportsFiles;

  std::string LinkedObjectsFileName;
  std::unique_ptr<llvm::raw_fd_ostream> LinkedObjectsFile;

public:
  WriteIndexesThinBackend(
      Config &Conf, ModuleSummaryIndex &CombinedIndex,
      const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
      std::string OldPrefix, std::string NewPrefix, bool ShouldEmitImportsFiles,
      std::string LinkedObjectsFileName)
      : ThinBackendProc(Conf, CombinedIndex, ModuleToDefinedGVSummaries),
        OldPrefix(OldPrefix), NewPrefix(NewPrefix),
        ShouldEmitImportsFiles(ShouldEmitImportsFiles),
        LinkedObjectsFileName(LinkedObjectsFileName) {}

  Error start(
      unsigned Task, BitcodeModule BM,
      const FunctionImporter::ImportMapTy &ImportList,
      const FunctionImporter::ExportSetTy &ExportList,
      const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
      MapVector<StringRef, BitcodeModule> &ModuleMap) override {
    StringRef ModulePath = BM.getModuleIdentifier();
    std::string NewModulePath =
        getThinLTOOutputFile(ModulePath, OldPrefix, NewPrefix);

    std::error_code EC;
    if (!LinkedObjectsFileName.empty()) {
      if (!LinkedObjectsFile) {
        LinkedObjectsFile = llvm::make_unique<raw_fd_ostream>(
            LinkedObjectsFileName, EC, sys::fs::OpenFlags::F_None);
        if (EC)
          return errorCodeToError(EC);
      }
      *LinkedObjectsFile << NewModulePath << '\n';
    }

    std::map<std::string, GVSummaryMapTy> ModuleToSummariesForIndex;
    gatherImportedSummariesForModule(ModulePath, ModuleToDefinedGVSummaries,
                                     ImportList, ModuleToSummariesForIndex);

    raw_fd_ostream OS(NewModulePath + ".thinlto.bc", EC,
                      sys::fs::OpenFlags::F_None);
    if (EC)
      return errorCodeToError(EC);
    WriteIndexToFile(CombinedIndex, OS, &ModuleToSummariesForIndex);

    if (ShouldEmitImportsFiles)
      return errorCodeToError(
          EmitImportsFiles(ModulePath, NewModulePath + ".imports", ImportList));
    return Error::success();
  }

  Error wait() override { return Error::success(); }
};
} // end anonymous namespace

ThinBackend lto::createWriteIndexesThinBackend(std::string OldPrefix,
                                               std::string NewPrefix,
                                               bool ShouldEmitImportsFiles,
                                               std::string LinkedObjectsFile) {
  return [=](Config &Conf, ModuleSummaryIndex &CombinedIndex,
             const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
             AddStreamFn AddStream, NativeObjectCache Cache) {
    return llvm::make_unique<WriteIndexesThinBackend>(
        Conf, CombinedIndex, ModuleToDefinedGVSummaries, OldPrefix, NewPrefix,
        ShouldEmitImportsFiles, LinkedObjectsFile);
  };
}

Error LTO::runThinLTO(AddStreamFn AddStream, NativeObjectCache Cache,
                      bool HasRegularLTO) {
  if (ThinLTO.ModuleMap.empty())
    return Error::success();

  if (Conf.CombinedIndexHook && !Conf.CombinedIndexHook(ThinLTO.CombinedIndex))
    return Error::success();

  // Collect for each module the list of function it defines (GUID ->
  // Summary).
  StringMap<std::map<GlobalValue::GUID, GlobalValueSummary *>>
      ModuleToDefinedGVSummaries(ThinLTO.ModuleMap.size());
  ThinLTO.CombinedIndex.collectDefinedGVSummariesPerModule(
      ModuleToDefinedGVSummaries);
  // Create entries for any modules that didn't have any GV summaries
  // (either they didn't have any GVs to start with, or we suppressed
  // generation of the summaries because they e.g. had inline assembly
  // uses that couldn't be promoted/renamed on export). This is so
  // InProcessThinBackend::start can still launch a backend thread, which
  // is passed the map of summaries for the module, without any special
  // handling for this case.
  for (auto &Mod : ThinLTO.ModuleMap)
    if (!ModuleToDefinedGVSummaries.count(Mod.first))
      ModuleToDefinedGVSummaries.try_emplace(Mod.first);

  StringMap<FunctionImporter::ImportMapTy> ImportLists(
      ThinLTO.ModuleMap.size());
  StringMap<FunctionImporter::ExportSetTy> ExportLists(
      ThinLTO.ModuleMap.size());
  StringMap<std::map<GlobalValue::GUID, GlobalValue::LinkageTypes>> ResolvedODR;

  if (Conf.OptLevel > 0) {
    ComputeCrossModuleImport(ThinLTO.CombinedIndex, ModuleToDefinedGVSummaries,
                             ImportLists, ExportLists);

    std::set<GlobalValue::GUID> ExportedGUIDs;
    for (auto &Res : GlobalResolutions) {
      // First check if the symbol was flagged as having external references.
      if (Res.second.Partition != GlobalResolution::External)
        continue;
      // IRName will be defined if we have seen the prevailing copy of
      // this value. If not, no need to mark as exported from a ThinLTO
      // partition (and we can't get the GUID).
      if (Res.second.IRName.empty())
        continue;
      auto GUID = GlobalValue::getGUID(
          GlobalValue::dropLLVMManglingEscape(Res.second.IRName));
      // Mark exported unless index-based analysis determined it to be dead.
      if (ThinLTO.CombinedIndex.isGUIDLive(GUID))
        ExportedGUIDs.insert(GUID);
    }

    auto isExported = [&](StringRef ModuleIdentifier, GlobalValue::GUID GUID) {
      const auto &ExportList = ExportLists.find(ModuleIdentifier);
      return (ExportList != ExportLists.end() &&
              ExportList->second.count(GUID)) ||
             ExportedGUIDs.count(GUID);
    };
    thinLTOInternalizeAndPromoteInIndex(ThinLTO.CombinedIndex, isExported);
  }

  auto isPrevailing = [&](GlobalValue::GUID GUID,
                          const GlobalValueSummary *S) {
    return ThinLTO.PrevailingModuleForGUID[GUID] == S->modulePath();
  };
  auto recordNewLinkage = [&](StringRef ModuleIdentifier,
                              GlobalValue::GUID GUID,
                              GlobalValue::LinkageTypes NewLinkage) {
    ResolvedODR[ModuleIdentifier][GUID] = NewLinkage;
  };
  thinLTOResolveWeakForLinkerInIndex(ThinLTO.CombinedIndex, isPrevailing,
                                     recordNewLinkage);

  std::unique_ptr<ThinBackendProc> BackendProc =
      ThinLTO.Backend(Conf, ThinLTO.CombinedIndex, ModuleToDefinedGVSummaries,
                      AddStream, Cache);

  // Task numbers start at ParallelCodeGenParallelismLevel if an LTO
  // module is present, as tasks 0 through ParallelCodeGenParallelismLevel-1
  // are reserved for parallel code generation partitions.
  unsigned Task =
      HasRegularLTO ? RegularLTO.ParallelCodeGenParallelismLevel : 0;
  for (auto &Mod : ThinLTO.ModuleMap) {
    if (Error E = BackendProc->start(Task, Mod.second, ImportLists[Mod.first],
                                     ExportLists[Mod.first],
                                     ResolvedODR[Mod.first], ThinLTO.ModuleMap))
      return E;
    ++Task;
  }

  return BackendProc->wait();
}

Expected<std::unique_ptr<tool_output_file>>
lto::setupOptimizationRemarks(LLVMContext &Context,
                              StringRef LTORemarksFilename,
                              bool LTOPassRemarksWithHotness, int Count) {
  if (LTORemarksFilename.empty())
    return nullptr;

  std::string Filename = LTORemarksFilename;
  if (Count != -1)
    Filename += ".thin." + llvm::utostr(Count) + ".yaml";

  std::error_code EC;
  auto DiagnosticFile =
      llvm::make_unique<tool_output_file>(Filename, EC, sys::fs::F_None);
  if (EC)
    return errorCodeToError(EC);
  Context.setDiagnosticsOutputFile(
      llvm::make_unique<yaml::Output>(DiagnosticFile->os()));
  if (LTOPassRemarksWithHotness)
    Context.setDiagnosticHotnessRequested(true);
  DiagnosticFile->keep();
  return std::move(DiagnosticFile);
}
