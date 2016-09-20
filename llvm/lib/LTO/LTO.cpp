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
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Linker/IRMover.h"
#include "llvm/Object/ModuleSummaryIndexObjectFile.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ThreadPool.h"
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
    SmallString<40> &Key, const ModuleSummaryIndex &Index, StringRef ModuleID,
    const FunctionImporter::ImportMapTy &ImportList,
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

  // Include the hash for the current module
  auto ModHash = Index.getModuleHash(ModuleID);
  Hasher.update(ArrayRef<uint8_t>((uint8_t *)&ModHash[0], sizeof(ModHash)));
  for (auto F : ExportList)
    // The export list can impact the internalization, be conservative here
    Hasher.update(ArrayRef<uint8_t>((uint8_t *)&F, sizeof(F)));

  // Include the hash for every module we import functions from
  for (auto &Entry : ImportList) {
    auto ModHash = Index.getModuleHash(Entry.first());
    Hasher.update(ArrayRef<uint8_t>((uint8_t *)&ModHash[0], sizeof(ModHash)));
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

  Key = toHex(Hasher.result());
}

// Simple helper to load a module from bitcode
std::unique_ptr<Module>
llvm::loadModuleFromBuffer(const MemoryBufferRef &Buffer, LLVMContext &Context,
                           bool Lazy) {
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
    function_ref<void(StringRef, GlobalValue::GUID, GlobalValue::LinkageTypes)>
        recordNewLinkage) {
  for (auto &S : GVSummaryList) {
    if (GlobalInvolvedWithAlias.count(S.get()))
      continue;
    GlobalValue::LinkageTypes OriginalLinkage = S->linkage();
    if (!GlobalValue::isWeakForLinker(OriginalLinkage))
      continue;
    // We need to emit only one of these. The prevailing module will keep it,
    // but turned into a weak, while the others will drop it when possible.
    if (isPrevailing(GUID, S.get())) {
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

Expected<std::unique_ptr<InputFile>> InputFile::create(MemoryBufferRef Object) {
  std::unique_ptr<InputFile> File(new InputFile);
  std::string Msg;
  auto DiagHandler = [](const DiagnosticInfo &DI, void *MsgP) {
    auto *Msg = reinterpret_cast<std::string *>(MsgP);
    raw_string_ostream OS(*Msg);
    DiagnosticPrinterRawOStream DP(OS);
    DI.print(DP);
  };
  File->Ctx.setDiagnosticHandler(DiagHandler, static_cast<void *>(&Msg));

  ErrorOr<std::unique_ptr<object::IRObjectFile>> IRObj =
      IRObjectFile::create(Object, File->Ctx);
  if (!Msg.empty())
    return make_error<StringError>(Msg, inconvertibleErrorCode());
  if (!IRObj)
    return errorCodeToError(IRObj.getError());
  File->Obj = std::move(*IRObj);

  File->Ctx.setDiagnosticHandler(nullptr, nullptr);

  return std::move(File);
}

LTO::RegularLTOState::RegularLTOState(unsigned ParallelCodeGenParallelismLevel,
                                      Config &Conf)
    : ParallelCodeGenParallelismLevel(ParallelCodeGenParallelismLevel),
      Ctx(Conf) {}

LTO::ThinLTOState::ThinLTOState(ThinBackend Backend) : Backend(Backend) {
  if (!Backend)
    this->Backend = createInProcessThinBackend(thread::hardware_concurrency());
}

LTO::LTO(Config Conf, ThinBackend Backend,
         unsigned ParallelCodeGenParallelismLevel)
    : Conf(std::move(Conf)),
      RegularLTO(ParallelCodeGenParallelismLevel, this->Conf),
      ThinLTO(std::move(Backend)) {}

// Add the given symbol to the GlobalResolutions map, and resolve its partition.
void LTO::addSymbolToGlobalRes(IRObjectFile *Obj,
                               SmallPtrSet<GlobalValue *, 8> &Used,
                               const InputFile::Symbol &Sym,
                               SymbolResolution Res, unsigned Partition) {
  GlobalValue *GV = Obj->getSymbolGV(Sym.I->getRawDataRefImpl());

  auto &GlobalRes = GlobalResolutions[Sym.getName()];
  if (GV) {
    GlobalRes.UnnamedAddr &= GV->hasGlobalUnnamedAddr();
    if (Res.Prevailing)
      GlobalRes.IRName = GV->getName();
  }
  if (Res.VisibleToRegularObj || (GV && Used.count(GV)) ||
      (GlobalRes.Partition != GlobalResolution::Unknown &&
       GlobalRes.Partition != Partition))
    GlobalRes.Partition = GlobalResolution::External;
  else
    GlobalRes.Partition = Partition;
}

static void writeToResolutionFile(raw_ostream &OS, InputFile *Input,
                                  ArrayRef<SymbolResolution> Res) {
  StringRef Path = Input->getMemoryBufferRef().getBufferIdentifier();
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
  assert(ResI == Res.end());
}

Error LTO::add(std::unique_ptr<InputFile> Input,
               ArrayRef<SymbolResolution> Res) {
  assert(!CalledGetMaxTasks);

  if (Conf.ResolutionFile)
    writeToResolutionFile(*Conf.ResolutionFile, Input.get(), Res);

  // FIXME: move to backend
  Module &M = Input->Obj->getModule();
  if (!Conf.OverrideTriple.empty())
    M.setTargetTriple(Conf.OverrideTriple);
  else if (M.getTargetTriple().empty())
    M.setTargetTriple(Conf.DefaultTriple);

  MemoryBufferRef MBRef = Input->Obj->getMemoryBufferRef();
  bool HasThinLTOSummary = hasGlobalValueSummary(MBRef, Conf.DiagHandler);

  if (HasThinLTOSummary)
    return addThinLTO(std::move(Input), Res);
  else
    return addRegularLTO(std::move(Input), Res);
}

// Add a regular LTO object to the link.
Error LTO::addRegularLTO(std::unique_ptr<InputFile> Input,
                         ArrayRef<SymbolResolution> Res) {
  if (!RegularLTO.CombinedModule) {
    RegularLTO.CombinedModule =
        llvm::make_unique<Module>("ld-temp.o", RegularLTO.Ctx);
    RegularLTO.Mover = llvm::make_unique<IRMover>(*RegularLTO.CombinedModule);
  }
  ErrorOr<std::unique_ptr<object::IRObjectFile>> ObjOrErr =
      IRObjectFile::create(Input->Obj->getMemoryBufferRef(), RegularLTO.Ctx);
  if (!ObjOrErr)
    return errorCodeToError(ObjOrErr.getError());
  std::unique_ptr<object::IRObjectFile> Obj = std::move(*ObjOrErr);

  Module &M = Obj->getModule();
  M.materializeMetadata();
  UpgradeDebugInfo(M);

  SmallPtrSet<GlobalValue *, 8> Used;
  collectUsedGlobalVariables(M, Used, /*CompilerUsed*/ false);

  std::vector<GlobalValue *> Keep;

  for (GlobalVariable &GV : M.globals())
    if (GV.hasAppendingLinkage())
      Keep.push_back(&GV);

  auto ResI = Res.begin();
  for (const InputFile::Symbol &Sym :
       make_range(InputFile::symbol_iterator(Obj->symbol_begin()),
                  InputFile::symbol_iterator(Obj->symbol_end()))) {
    assert(ResI != Res.end());
    SymbolResolution Res = *ResI++;
    addSymbolToGlobalRes(Obj.get(), Used, Sym, Res, 0);

    GlobalValue *GV = Obj->getSymbolGV(Sym.I->getRawDataRefImpl());
    if (Sym.getFlags() & object::BasicSymbolRef::SF_Undefined)
      continue;
    if (Res.Prevailing && GV) {
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
    }
    // Common resolution: collect the maximum size/alignment over all commons.
    // We also record if we see an instance of a common as prevailing, so that
    // if none is prevailing we can ignore it later.
    if (Sym.getFlags() & object::BasicSymbolRef::SF_Common) {
      auto &CommonRes = RegularLTO.Commons[Sym.getIRName()];
      CommonRes.Size = std::max(CommonRes.Size, Sym.getCommonSize());
      CommonRes.Align = std::max(CommonRes.Align, Sym.getCommonAlignment());
      CommonRes.Prevailing |= Res.Prevailing;
    }

    // FIXME: use proposed local attribute for FinalDefinitionInLinkageUnit.
  }
  assert(ResI == Res.end());

  return RegularLTO.Mover->move(Obj->takeModule(), Keep,
                                [](GlobalValue &, IRMover::ValueAdder) {});
}

// Add a ThinLTO object to the link.
Error LTO::addThinLTO(std::unique_ptr<InputFile> Input,
                      ArrayRef<SymbolResolution> Res) {
  Module &M = Input->Obj->getModule();
  SmallPtrSet<GlobalValue *, 8> Used;
  collectUsedGlobalVariables(M, Used, /*CompilerUsed*/ false);

  MemoryBufferRef MBRef = Input->Obj->getMemoryBufferRef();
  ErrorOr<std::unique_ptr<object::ModuleSummaryIndexObjectFile>>
      SummaryObjOrErr =
          object::ModuleSummaryIndexObjectFile::create(MBRef, Conf.DiagHandler);
  if (!SummaryObjOrErr)
    return errorCodeToError(SummaryObjOrErr.getError());
  ThinLTO.CombinedIndex.mergeFrom((*SummaryObjOrErr)->takeIndex(),
                                  ThinLTO.ModuleMap.size());

  auto ResI = Res.begin();
  for (const InputFile::Symbol &Sym : Input->symbols()) {
    assert(ResI != Res.end());
    SymbolResolution Res = *ResI++;
    addSymbolToGlobalRes(Input->Obj.get(), Used, Sym, Res,
                         ThinLTO.ModuleMap.size() + 1);

    GlobalValue *GV = Input->Obj->getSymbolGV(Sym.I->getRawDataRefImpl());
    if (Res.Prevailing && GV)
      ThinLTO.PrevailingModuleForGUID[GV->getGUID()] =
          MBRef.getBufferIdentifier();
  }
  assert(ResI == Res.end());

  ThinLTO.ModuleMap[MBRef.getBufferIdentifier()] = MBRef;
  return Error();
}

unsigned LTO::getMaxTasks() const {
  CalledGetMaxTasks = true;
  return RegularLTO.ParallelCodeGenParallelismLevel + ThinLTO.ModuleMap.size();
}

Error LTO::run(AddOutputFn AddOutput) {
  // Save the status of having a regularLTO combined module, as
  // this is needed for generating the ThinLTO Task ID, and
  // the CombinedModule will be moved at the end of runRegularLTO.
  bool HasRegularLTO = RegularLTO.CombinedModule != nullptr;
  // Invoke regular LTO if there was a regular LTO module to start with.
  if (HasRegularLTO)
    if (auto E = runRegularLTO(AddOutput))
      return E;
  return runThinLTO(AddOutput, HasRegularLTO);
}

Error LTO::runRegularLTO(AddOutputFn AddOutput) {
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
    return Error();

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
      return Error();
  }
  return backend(Conf, AddOutput, RegularLTO.ParallelCodeGenParallelismLevel,
                 std::move(RegularLTO.CombinedModule));
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
      unsigned Task, MemoryBufferRef MBRef,
      const FunctionImporter::ImportMapTy &ImportList,
      const FunctionImporter::ExportSetTy &ExportList,
      const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
      MapVector<StringRef, MemoryBufferRef> &ModuleMap) = 0;
  virtual Error wait() = 0;
};

class InProcessThinBackend : public ThinBackendProc {
  ThreadPool BackendThreadPool;
  AddOutputFn AddOutput;

  Optional<Error> Err;
  std::mutex ErrMu;

public:
  InProcessThinBackend(
      Config &Conf, ModuleSummaryIndex &CombinedIndex,
      unsigned ThinLTOParallelismLevel,
      const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
      AddOutputFn AddOutput)
      : ThinBackendProc(Conf, CombinedIndex, ModuleToDefinedGVSummaries),
        BackendThreadPool(ThinLTOParallelismLevel),
        AddOutput(std::move(AddOutput)) {}

  Error runThinLTOBackendThread(
      AddOutputFn AddOutput, unsigned Task, MemoryBufferRef MBRef,
      ModuleSummaryIndex &CombinedIndex,
      const FunctionImporter::ImportMapTy &ImportList,
      const FunctionImporter::ExportSetTy &ExportList,
      const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
      const GVSummaryMapTy &DefinedGlobals,
      MapVector<StringRef, MemoryBufferRef> &ModuleMap) {

    auto ModuleIdentifier = MBRef.getBufferIdentifier();
    auto Output = AddOutput(Task);
    if (Output->isCachingEnabled()) {
      SmallString<40> Key;
      // The module may be cached, this helps handling it.
      computeCacheKey(Key, CombinedIndex, ModuleIdentifier, ImportList,
                      ExportList, ResolvedODR, DefinedGlobals);
      if (Output->tryLoadFromCache(Key))
        return Error();
    }

    LTOLLVMContext BackendContext(Conf);
    ErrorOr<std::unique_ptr<Module>> MOrErr =
        parseBitcodeFile(MBRef, BackendContext);
    assert(MOrErr && "Unable to load module in thread?");

    auto AddOutputWrapper = [&](unsigned TaskId) {
      assert(Task == TaskId && "Unexpexted TaskId mismatch");
      return std::move(Output);
    };
    return thinBackend(Conf, Task, AddOutputWrapper, **MOrErr, CombinedIndex,
                       ImportList, DefinedGlobals, ModuleMap);
  }

  Error start(
      unsigned Task, MemoryBufferRef MBRef,
      const FunctionImporter::ImportMapTy &ImportList,
      const FunctionImporter::ExportSetTy &ExportList,
      const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
      MapVector<StringRef, MemoryBufferRef> &ModuleMap) override {
    StringRef ModulePath = MBRef.getBufferIdentifier();
    assert(ModuleToDefinedGVSummaries.count(ModulePath));
    const GVSummaryMapTy &DefinedGlobals =
        ModuleToDefinedGVSummaries.find(ModulePath)->second;
    BackendThreadPool.async(
        [=](MemoryBufferRef MBRef, ModuleSummaryIndex &CombinedIndex,
            const FunctionImporter::ImportMapTy &ImportList,
            const FunctionImporter::ExportSetTy &ExportList,
            const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes>
                &ResolvedODR,
            const GVSummaryMapTy &DefinedGlobals,
            MapVector<StringRef, MemoryBufferRef> &ModuleMap) {
          Error E = runThinLTOBackendThread(
              AddOutput, Task, MBRef, CombinedIndex, ImportList, ExportList,
              ResolvedODR, DefinedGlobals, ModuleMap);
          if (E) {
            std::unique_lock<std::mutex> L(ErrMu);
            if (Err)
              Err = joinErrors(std::move(*Err), std::move(E));
            else
              Err = std::move(E);
          }
        },
        MBRef, std::ref(CombinedIndex), std::ref(ImportList),
        std::ref(ExportList), std::ref(ResolvedODR), std::ref(DefinedGlobals),
        std::ref(ModuleMap));
    return Error();
  }

  Error wait() override {
    BackendThreadPool.wait();
    if (Err)
      return std::move(*Err);
    else
      return Error();
  }
};

ThinBackend lto::createInProcessThinBackend(unsigned ParallelismLevel) {
  return [=](Config &Conf, ModuleSummaryIndex &CombinedIndex,
             const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
             AddOutputFn AddOutput) {
    return llvm::make_unique<InProcessThinBackend>(
        Conf, CombinedIndex, ParallelismLevel, ModuleToDefinedGVSummaries,
        AddOutput);
  };
}

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

  /// Given the original \p Path to an output file, replace any path
  /// prefix matching \p OldPrefix with \p NewPrefix. Also, create the
  /// resulting directory if it does not yet exist.
  std::string getThinLTOOutputFile(const std::string &Path,
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

  Error start(
      unsigned Task, MemoryBufferRef MBRef,
      const FunctionImporter::ImportMapTy &ImportList,
      const FunctionImporter::ExportSetTy &ExportList,
      const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
      MapVector<StringRef, MemoryBufferRef> &ModuleMap) override {
    StringRef ModulePath = MBRef.getBufferIdentifier();
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
    return Error();
  }

  Error wait() override { return Error(); }
};

ThinBackend lto::createWriteIndexesThinBackend(std::string OldPrefix,
                                               std::string NewPrefix,
                                               bool ShouldEmitImportsFiles,
                                               std::string LinkedObjectsFile) {
  return [=](Config &Conf, ModuleSummaryIndex &CombinedIndex,
             const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
             AddOutputFn AddOutput) {
    return llvm::make_unique<WriteIndexesThinBackend>(
        Conf, CombinedIndex, ModuleToDefinedGVSummaries, OldPrefix, NewPrefix,
        ShouldEmitImportsFiles, LinkedObjectsFile);
  };
}

Error LTO::runThinLTO(AddOutputFn AddOutput, bool HasRegularLTO) {
  if (ThinLTO.ModuleMap.empty())
    return Error();

  if (Conf.CombinedIndexHook && !Conf.CombinedIndexHook(ThinLTO.CombinedIndex))
    return Error();

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
  ComputeCrossModuleImport(ThinLTO.CombinedIndex, ModuleToDefinedGVSummaries,
                           ImportLists, ExportLists);

  std::set<GlobalValue::GUID> ExportedGUIDs;
  for (auto &Res : GlobalResolutions) {
    if (!Res.second.IRName.empty() &&
        Res.second.Partition == GlobalResolution::External)
      ExportedGUIDs.insert(GlobalValue::getGUID(Res.second.IRName));
  }

  auto isPrevailing = [&](GlobalValue::GUID GUID, const GlobalValueSummary *S) {
    return ThinLTO.PrevailingModuleForGUID[GUID] == S->modulePath();
  };
  auto isExported = [&](StringRef ModuleIdentifier, GlobalValue::GUID GUID) {
    const auto &ExportList = ExportLists.find(ModuleIdentifier);
    return (ExportList != ExportLists.end() &&
            ExportList->second.count(GUID)) ||
           ExportedGUIDs.count(GUID);
  };
  thinLTOInternalizeAndPromoteInIndex(ThinLTO.CombinedIndex, isExported);

  StringMap<std::map<GlobalValue::GUID, GlobalValue::LinkageTypes>> ResolvedODR;
  auto recordNewLinkage = [&](StringRef ModuleIdentifier,
                              GlobalValue::GUID GUID,
                              GlobalValue::LinkageTypes NewLinkage) {
    ResolvedODR[ModuleIdentifier][GUID] = NewLinkage;
  };

  thinLTOResolveWeakForLinkerInIndex(ThinLTO.CombinedIndex, isPrevailing,
                                     recordNewLinkage);

  std::unique_ptr<ThinBackendProc> BackendProc = ThinLTO.Backend(
      Conf, ThinLTO.CombinedIndex, ModuleToDefinedGVSummaries, AddOutput);

  // Partition numbers for ThinLTO jobs start at 1 (see comments for
  // GlobalResolution in LTO.h). Task numbers, however, start at
  // ParallelCodeGenParallelismLevel if an LTO module is present, as tasks 0
  // through ParallelCodeGenParallelismLevel-1 are reserved for parallel code
  // generation partitions.
  unsigned Task =
      HasRegularLTO ? RegularLTO.ParallelCodeGenParallelismLevel : 0;
  unsigned Partition = 1;

  for (auto &Mod : ThinLTO.ModuleMap) {
    if (Error E = BackendProc->start(Task, Mod.second, ImportLists[Mod.first],
                                     ExportLists[Mod.first],
                                     ResolvedODR[Mod.first], ThinLTO.ModuleMap))
      return E;

    ++Task;
    ++Partition;
  }

  return BackendProc->wait();
}
