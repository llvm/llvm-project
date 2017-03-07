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
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Metadata.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Linker/IRMover.h"
#include "llvm/Object/ModuleSummaryIndexObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
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

// Returns a unique hash for the Module considering the current list of
// export/import and other global analysis results.
// The hash is produced in \p Key.
static void computeCacheKey(
    SmallString<40> &Key, const Config &Conf, const ModuleSummaryIndex &Index,
    StringRef ModuleID, const FunctionImporter::ImportMapTy &ImportList,
    const FunctionImporter::ExportSetTy &ExportList,
    const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
    const GVSummaryMapTy &DefinedGlobals) {
  // Compute the unique hash for this entry.
  // This is based on the current compiler version, the module itself, the
  // export list, the hash for every single module in the import list, the
  // list of ResolvedODR for the module, and the list of preserved symbols.
  SHA1 Hasher;

  // Start with the compiler revision
  Hasher.update(LLVM_VERSION_STRING);
#ifdef HAVE_LLVM_REVISION
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
  AddUnsigned(Conf.RelocModel);
  AddUnsigned(Conf.CodeModel);
  AddUnsigned(Conf.CGOptLevel);
  AddUnsigned(Conf.CGFileType);
  AddUnsigned(Conf.OptLevel);
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

  // Include the hash for the linkage type to reflect internalization and weak
  // resolution.
  for (auto &GS : DefinedGlobals) {
    GlobalValue::LinkageTypes Linkage = GS.second->linkage();
    Hasher.update(
        ArrayRef<uint8_t>((const uint8_t *)&Linkage, sizeof(Linkage)));
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
    for (auto &S : I.second)
      if (auto AS = dyn_cast<AliasSummary>(S.get()))
        GlobalInvolvedWithAlias.insert(&AS->getAliasee());

  for (auto &I : Index)
    thinLTOResolveWeakForLinkerGUID(I.second, I.first, GlobalInvolvedWithAlias,
                                    isPrevailing, recordNewLinkage);
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
    thinLTOInternalizeAndPromoteGUID(I.second, I.first, isExported);
}

struct InputFile::InputModule {
  BitcodeModule BM;
  std::unique_ptr<Module> Mod;

  // The range of ModuleSymbolTable entries for this input module.
  size_t SymBegin, SymEnd;
};

// Requires a destructor for std::vector<InputModule>.
InputFile::~InputFile() = default;

Expected<std::unique_ptr<InputFile>> InputFile::create(MemoryBufferRef Object) {
  std::unique_ptr<InputFile> File(new InputFile);

  ErrorOr<MemoryBufferRef> BCOrErr =
      IRObjectFile::findBitcodeInMemBuffer(Object);
  if (!BCOrErr)
    return errorCodeToError(BCOrErr.getError());

  Expected<std::vector<BitcodeModule>> BMsOrErr =
      getBitcodeModuleList(*BCOrErr);
  if (!BMsOrErr)
    return BMsOrErr.takeError();

  if (BMsOrErr->empty())
    return make_error<StringError>("Bitcode file does not contain any modules",
                                   inconvertibleErrorCode());

  // Create an InputModule for each module in the InputFile, and add it to the
  // ModuleSymbolTable.
  for (auto BM : *BMsOrErr) {
    Expected<std::unique_ptr<Module>> MOrErr =
        BM.getLazyModule(File->Ctx, /*ShouldLazyLoadMetadata*/ true,
                         /*IsImporting*/ false);
    if (!MOrErr)
      return MOrErr.takeError();

    size_t SymBegin = File->SymTab.symbols().size();
    File->SymTab.addModule(MOrErr->get());
    size_t SymEnd = File->SymTab.symbols().size();

    for (const auto &C : (*MOrErr)->getComdatSymbolTable()) {
      auto P = File->ComdatMap.insert(
          std::make_pair(&C.second, File->Comdats.size()));
      assert(P.second);
      (void)P;
      File->Comdats.push_back(C.first());
    }

    File->Mods.push_back({BM, std::move(*MOrErr), SymBegin, SymEnd});
  }

  return std::move(File);
}

Expected<int> InputFile::Symbol::getComdatIndex() const {
  if (!isGV())
    return -1;
  const GlobalObject *GO = getGV()->getBaseObject();
  if (!GO)
    return make_error<StringError>("Unable to determine comdat of alias!",
                                   inconvertibleErrorCode());
  if (const Comdat *C = GO->getComdat()) {
    auto I = File->ComdatMap.find(C);
    assert(I != File->ComdatMap.end());
    return I->second;
  }
  return -1;
}

Expected<std::string> InputFile::getLinkerOpts() {
  std::string LinkerOpts;
  raw_string_ostream LOS(LinkerOpts);
  // Extract linker options from module metadata.
  for (InputModule &Mod : Mods) {
    std::unique_ptr<Module> &M = Mod.Mod;
    if (auto E = M->materializeMetadata())
      return std::move(E);
    if (Metadata *Val = M->getModuleFlag("Linker Options")) {
      MDNode *LinkerOptions = cast<MDNode>(Val);
      for (const MDOperand &MDOptions : LinkerOptions->operands())
        for (const MDOperand &MDOption : cast<MDNode>(MDOptions)->operands())
          LOS << " " << cast<MDString>(MDOption)->getString();
    }
  }

  // Synthesize export flags for symbols with dllexport storage.
  const Triple TT(Mods[0].Mod->getTargetTriple());
  Mangler M;
  for (const ModuleSymbolTable::Symbol &Sym : SymTab.symbols())
    if (auto *GV = Sym.dyn_cast<GlobalValue*>())
      emitLinkerFlagsForGlobalCOFF(LOS, GV, TT, M);
  LOS.flush();
  return LinkerOpts;
}

StringRef InputFile::getName() const {
  return Mods[0].BM.getModuleIdentifier();
}

StringRef InputFile::getSourceFileName() const {
  return Mods[0].Mod->getSourceFileName();
}

iterator_range<InputFile::symbol_iterator>
InputFile::module_symbols(InputModule &IM) {
  return llvm::make_range(
      symbol_iterator(SymTab.symbols().data() + IM.SymBegin, SymTab, this),
      symbol_iterator(SymTab.symbols().data() + IM.SymEnd, SymTab, this));
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

// Add the given symbol to the GlobalResolutions map, and resolve its partition.
void LTO::addSymbolToGlobalRes(SmallPtrSet<GlobalValue *, 8> &Used,
                               const InputFile::Symbol &Sym,
                               SymbolResolution Res, unsigned Partition) {
  GlobalValue *GV = Sym.isGV() ? Sym.getGV() : nullptr;

  auto &GlobalRes = GlobalResolutions[Sym.getName()];
  if (GV) {
    GlobalRes.UnnamedAddr &= GV->hasGlobalUnnamedAddr();
    if (Res.Prevailing)
      GlobalRes.IRName = GV->getName();
  }
  // Set the partition to external if we know it is used elsewhere, e.g.
  // it is visible to a regular object, is referenced from llvm.compiler_used,
  // or was already recorded as being referenced from a different partition.
  if (Res.VisibleToRegularObj || (GV && Used.count(GV)) ||
      (GlobalRes.Partition != GlobalResolution::Unknown &&
       GlobalRes.Partition != Partition)) {
    GlobalRes.Partition = GlobalResolution::External;
  } else
    // First recorded reference, save the current partition.
    GlobalRes.Partition = Partition;

  // Flag as visible outside of ThinLTO if visible from a regular object or
  // if this is a reference in the regular LTO partition.
  GlobalRes.VisibleOutsideThinLTO |=
      (Res.VisibleToRegularObj || (Partition == GlobalResolution::RegularLTO));
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
  for (InputFile::InputModule &IM : Input->Mods)
    if (Error Err = addModule(*Input, IM, ResI, Res.end()))
      return Err;

  assert(ResI == Res.end());
  return Error::success();
}

Error LTO::addModule(InputFile &Input, InputFile::InputModule &IM,
                     const SymbolResolution *&ResI,
                     const SymbolResolution *ResE) {
  // FIXME: move to backend
  Module &M = *IM.Mod;

  if (M.getDataLayoutStr().empty())
    return make_error<StringError>("input module has no datalayout",
                                    inconvertibleErrorCode());

  if (!Conf.OverrideTriple.empty())
    M.setTargetTriple(Conf.OverrideTriple);
  else if (M.getTargetTriple().empty())
    M.setTargetTriple(Conf.DefaultTriple);

  Expected<bool> HasThinLTOSummary = IM.BM.hasSummary();
  if (!HasThinLTOSummary)
    return HasThinLTOSummary.takeError();

  if (*HasThinLTOSummary)
    return addThinLTO(IM.BM, M, Input.module_symbols(IM), ResI, ResE);
  else
    return addRegularLTO(IM.BM, ResI, ResE);
}

// Add a regular LTO object to the link.
Error LTO::addRegularLTO(BitcodeModule BM, const SymbolResolution *&ResI,
                         const SymbolResolution *ResE) {
  if (!RegularLTO.CombinedModule) {
    RegularLTO.CombinedModule =
        llvm::make_unique<Module>("ld-temp.o", RegularLTO.Ctx);
    RegularLTO.Mover = llvm::make_unique<IRMover>(*RegularLTO.CombinedModule);
  }
  Expected<std::unique_ptr<Module>> MOrErr =
      BM.getLazyModule(RegularLTO.Ctx, /*ShouldLazyLoadMetadata*/ true,
                       /*IsImporting*/ false);
  if (!MOrErr)
    return MOrErr.takeError();

  Module &M = **MOrErr;
  if (Error Err = M.materializeMetadata())
    return Err;
  UpgradeDebugInfo(M);

  ModuleSymbolTable SymTab;
  SymTab.addModule(&M);

  SmallPtrSet<GlobalValue *, 8> Used;
  collectUsedGlobalVariables(M, Used, /*CompilerUsed*/ false);

  std::vector<GlobalValue *> Keep;

  for (GlobalVariable &GV : M.globals())
    if (GV.hasAppendingLinkage())
      Keep.push_back(&GV);

  DenseSet<GlobalObject *> AliasedGlobals;
  for (auto &GA : M.aliases())
    if (GlobalObject *GO = GA.getBaseObject())
      AliasedGlobals.insert(GO);

  for (const InputFile::Symbol &Sym :
       make_range(InputFile::symbol_iterator(SymTab.symbols().begin(), SymTab,
                                             nullptr),
                  InputFile::symbol_iterator(SymTab.symbols().end(), SymTab,
                                             nullptr))) {
    assert(ResI != ResE);
    SymbolResolution Res = *ResI++;
    addSymbolToGlobalRes(Used, Sym, Res, 0);

    if (Sym.isGV()) {
      GlobalValue *GV = Sym.getGV();
      if (Res.Prevailing) {
        if (Sym.getFlags() & object::BasicSymbolRef::SF_Undefined)
          continue;
        Keep.push_back(GV);
        switch (GV->getLinkage()) {
        default:
          break;
        case GlobalValue::LinkOnceAnyLinkage:
          GV->setLinkage(GlobalValue::WeakAnyLinkage);
          break;
        case GlobalValue::LinkOnceODRLinkage:
          GV->setLinkage(GlobalValue::WeakODRLinkage);
          break;
        }
      } else if (isa<GlobalObject>(GV) &&
                 (GV->hasLinkOnceODRLinkage() || GV->hasWeakODRLinkage() ||
                  GV->hasAvailableExternallyLinkage()) &&
                 !AliasedGlobals.count(cast<GlobalObject>(GV))) {
        // Either of the above three types of linkage indicates that the
        // chosen prevailing symbol will have the same semantics as this copy of
        // the symbol, so we can link it with available_externally linkage. We
        // only need to do this if the symbol is undefined.
        GlobalValue *CombinedGV =
            RegularLTO.CombinedModule->getNamedValue(GV->getName());
        if (!CombinedGV || CombinedGV->isDeclaration()) {
          Keep.push_back(GV);
          GV->setLinkage(GlobalValue::AvailableExternallyLinkage);
          cast<GlobalObject>(GV)->setComdat(nullptr);
        }
      }
    }
    // Common resolution: collect the maximum size/alignment over all commons.
    // We also record if we see an instance of a common as prevailing, so that
    // if none is prevailing we can ignore it later.
    if (Sym.getFlags() & object::BasicSymbolRef::SF_Common) {
      // FIXME: We should figure out what to do about commons defined by asm.
      // For now they aren't reported correctly by ModuleSymbolTable.
      auto &CommonRes = RegularLTO.Commons[Sym.getGV()->getName()];
      CommonRes.Size = std::max(CommonRes.Size, Sym.getCommonSize());
      CommonRes.Align = std::max(CommonRes.Align, Sym.getCommonAlignment());
      CommonRes.Prevailing |= Res.Prevailing;
    }

    // FIXME: use proposed local attribute for FinalDefinitionInLinkageUnit.
  }

  return RegularLTO.Mover->move(std::move(*MOrErr), Keep,
                                [](GlobalValue &, IRMover::ValueAdder) {},
                                /* IsPerformingImport */ false);
}

// Add a ThinLTO object to the link.
// FIXME: This function should not need to take as many parameters once we have
// a bitcode symbol table.
Error LTO::addThinLTO(BitcodeModule BM, Module &M,
                      iterator_range<InputFile::symbol_iterator> Syms,
                      const SymbolResolution *&ResI,
                      const SymbolResolution *ResE) {
  SmallPtrSet<GlobalValue *, 8> Used;
  collectUsedGlobalVariables(M, Used, /*CompilerUsed*/ false);

  Expected<std::unique_ptr<ModuleSummaryIndex>> SummaryOrErr = BM.getSummary();
  if (!SummaryOrErr)
    return SummaryOrErr.takeError();
  ThinLTO.CombinedIndex.mergeFrom(std::move(*SummaryOrErr),
                                  ThinLTO.ModuleMap.size());

  for (const InputFile::Symbol &Sym : Syms) {
    assert(ResI != ResE);
    SymbolResolution Res = *ResI++;
    addSymbolToGlobalRes(Used, Sym, Res, ThinLTO.ModuleMap.size() + 1);

    if (Res.Prevailing && Sym.isGV())
      ThinLTO.PrevailingModuleForGUID[Sym.getGV()->getGUID()] =
          BM.getModuleIdentifier();
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
  // Save the status of having a regularLTO combined module, as
  // this is needed for generating the ThinLTO Task ID, and
  // the CombinedModule will be moved at the end of runRegularLTO.
  bool HasRegularLTO = RegularLTO.CombinedModule != nullptr;
  // Invoke regular LTO if there was a regular LTO module to start with.
  if (HasRegularLTO)
    if (auto E = runRegularLTO(AddStream))
      return E;
  return runThinLTO(AddStream, Cache, HasRegularLTO);
}

Error LTO::runRegularLTO(AddStreamFn AddStream) {
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
        AddStream(std::move(AddStream)), Cache(std::move(Cache)) {}

  Error runThinLTOBackendThread(
      AddStreamFn AddStream, NativeObjectCache Cache, unsigned Task,
      BitcodeModule BM, ModuleSummaryIndex &CombinedIndex,
      const FunctionImporter::ImportMapTy &ImportList,
      const FunctionImporter::ExportSetTy &ExportList,
      const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
      const GVSummaryMapTy &DefinedGlobals,
      MapVector<StringRef, BitcodeModule> &ModuleMap) {
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
                    ResolvedODR, DefinedGlobals);
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
            MapVector<StringRef, BitcodeModule> &ModuleMap) {
          Error E = runThinLTOBackendThread(
              AddStream, Cache, Task, BM, CombinedIndex, ImportList,
              ExportList, ResolvedODR, DefinedGlobals, ModuleMap);
          if (E) {
            std::unique_lock<std::mutex> L(ErrMu);
            if (Err)
              Err = joinErrors(std::move(*Err), std::move(E));
            else
              Err = std::move(E);
          }
        },
        BM, std::ref(CombinedIndex), std::ref(ImportList),
        std::ref(ExportList), std::ref(ResolvedODR), std::ref(DefinedGlobals),
        std::ref(ModuleMap));
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
    // Compute "dead" symbols, we don't want to import/export these!
    DenseSet<GlobalValue::GUID> GUIDPreservedSymbols;
    for (auto &Res : GlobalResolutions) {
      if (Res.second.VisibleOutsideThinLTO &&
          // IRName will be defined if we have seen the prevailing copy of
          // this value. If not, no need to preserve any ThinLTO copies.
          !Res.second.IRName.empty())
        GUIDPreservedSymbols.insert(GlobalValue::getGUID(Res.second.IRName));
    }

    auto DeadSymbols =
        computeDeadSymbols(ThinLTO.CombinedIndex, GUIDPreservedSymbols);

    ComputeCrossModuleImport(ThinLTO.CombinedIndex, ModuleToDefinedGVSummaries,
                             ImportLists, ExportLists, &DeadSymbols);

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
      auto GUID = GlobalValue::getGUID(Res.second.IRName);
      // Mark exported unless index-based analysis determined it to be dead.
      if (!DeadSymbols.count(GUID))
        ExportedGUIDs.insert(GlobalValue::getGUID(Res.second.IRName));
    }

    auto isPrevailing = [&](GlobalValue::GUID GUID,
                            const GlobalValueSummary *S) {
      return ThinLTO.PrevailingModuleForGUID[GUID] == S->modulePath();
    };
    auto isExported = [&](StringRef ModuleIdentifier, GlobalValue::GUID GUID) {
      const auto &ExportList = ExportLists.find(ModuleIdentifier);
      return (ExportList != ExportLists.end() &&
              ExportList->second.count(GUID)) ||
             ExportedGUIDs.count(GUID);
    };
    thinLTOInternalizeAndPromoteInIndex(ThinLTO.CombinedIndex, isExported);

    auto recordNewLinkage = [&](StringRef ModuleIdentifier,
                                GlobalValue::GUID GUID,
                                GlobalValue::LinkageTypes NewLinkage) {
      ResolvedODR[ModuleIdentifier][GUID] = NewLinkage;
    };

    thinLTOResolveWeakForLinkerInIndex(ThinLTO.CombinedIndex, isPrevailing,
                                       recordNewLinkage);
  }

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
