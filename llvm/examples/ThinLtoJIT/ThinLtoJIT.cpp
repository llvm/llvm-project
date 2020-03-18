#include "ThinLtoJIT.h"

#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Host.h"

#include "ThinLtoDiscoveryThread.h"
#include "ThinLtoInstrumentationLayer.h"
#include "ThinLtoModuleIndex.h"

#include <set>
#include <string>
#include <thread>

#ifndef NDEBUG
#include <chrono>
#endif

#define DEBUG_TYPE "thinltojit"

namespace llvm {
namespace orc {

class ThinLtoDefinitionGenerator : public JITDylib::DefinitionGenerator {
public:
  ThinLtoDefinitionGenerator(ThinLtoModuleIndex &GlobalIndex,
                             ThinLtoInstrumentationLayer &InstrumentationLayer,
                             ThinLtoJIT::AddModuleFunction AddModule,
                             char Prefix, bool AllowNudge, bool PrintStats)
      : GlobalIndex(GlobalIndex), InstrumentationLayer(InstrumentationLayer),
        AddModule(std::move(AddModule)), ManglePrefix(Prefix),
        AllowNudgeIntoDiscovery(AllowNudge), PrintStats(PrintStats) {}

  ~ThinLtoDefinitionGenerator() {
    if (PrintStats)
      dump(errs());
  }

  Error tryToGenerate(LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &Symbols) override;

  void dump(raw_ostream &OS) {
    OS << format("Modules submitted synchronously: %d\n", NumModulesMissed);
  }

private:
  ThinLtoModuleIndex &GlobalIndex;
  ThinLtoInstrumentationLayer &InstrumentationLayer;
  ThinLtoJIT::AddModuleFunction AddModule;
  char ManglePrefix;
  bool AllowNudgeIntoDiscovery;
  bool PrintStats;
  unsigned NumModulesMissed{0};

  // ThinLTO summaries encode unprefixed names.
  StringRef stripGlobalManglePrefix(StringRef Symbol) const {
    bool Strip = (ManglePrefix != '\0' && Symbol[0] == ManglePrefix);
    return Strip ? StringRef(Symbol.data() + 1, Symbol.size() - 1) : Symbol;
  }
};

Error ThinLtoDefinitionGenerator::tryToGenerate(
    LookupKind K, JITDylib &JD, JITDylibLookupFlags JDLookupFlags,
    const SymbolLookupSet &Symbols) {
  std::set<StringRef> ModulePaths;
  std::vector<GlobalValue::GUID> NewDiscoveryRoots;

  for (const auto &KV : Symbols) {
    StringRef UnmangledName = stripGlobalManglePrefix(*KV.first);
    auto Guid = GlobalValue::getGUID(UnmangledName);
    if (GlobalValueSummary *S = GlobalIndex.getSummary(Guid)) {
      // We could have discovered it ahead of time.
      LLVM_DEBUG(dbgs() << format("Failed to discover symbol: %s\n",
                                  UnmangledName.str().c_str()));
      ModulePaths.insert(S->modulePath());
      if (AllowNudgeIntoDiscovery && isa<FunctionSummary>(S)) {
        NewDiscoveryRoots.push_back(Guid);
      }
    }
  }

  NumModulesMissed += ModulePaths.size();

  // Parse the requested modules if it hasn't happened yet.
  GlobalIndex.scheduleModuleParsing(ModulePaths);

  for (StringRef Path : ModulePaths) {
    ThreadSafeModule TSM = GlobalIndex.takeModule(Path);
    assert(TSM && "We own the session lock, no asynchronous access possible");

    if (Error LoadErr = AddModule(std::move(TSM)))
      // Failed to add the module to the session.
      return LoadErr;

    LLVM_DEBUG(dbgs() << "Generator: added " << Path << " synchronously\n");
  }

  // Requested functions that we failed to discover ahead of time, are likely
  // close to the execution front. We can anticipate to run into them as soon
  // as execution continues and trigger their discovery flags already now. This
  // behavior is enabled with the 'allow-nudge' option and implemented below.
  // On the one hand, it may give us a head start in a moment where discovery
  // was lacking behind. On the other hand, we may bet on the wrong horse and
  // waste extra time speculating in the wrong direction.
  if (!NewDiscoveryRoots.empty()) {
    assert(AllowNudgeIntoDiscovery);
    InstrumentationLayer.nudgeIntoDiscovery(std::move(NewDiscoveryRoots));
  }

  return Error::success();
}

ThinLtoJIT::ThinLtoJIT(ArrayRef<std::string> InputFiles,
                       StringRef MainFunctionName, unsigned LookaheadLevels,
                       unsigned NumCompileThreads, unsigned NumLoadThreads,
                       unsigned DiscoveryFlagsPerBucket,
                       ExplicitMemoryBarrier MemFence,
                       bool AllowNudgeIntoDiscovery, bool PrintStats,
                       Error &Err) {
  ErrorAsOutParameter ErrAsOutParam(&Err);

  // Populate the module index, so we know which modules exist and we can find
  // the one that defines the main function.
  GlobalIndex = std::make_unique<ThinLtoModuleIndex>(ES, NumLoadThreads);
  for (StringRef F : InputFiles) {
    if (auto Err = GlobalIndex->add(F))
      ES.reportError(std::move(Err));
  }

  // Load the module that defines the main function.
  auto TSM = setupMainModule(MainFunctionName);
  if (!TSM) {
    Err = TSM.takeError();
    return;
  }

  // Infer target-specific utils from the main module.
  ThreadSafeModule MainModule = std::move(*TSM);
  auto JTMB = setupTargetUtils(MainModule.getModuleUnlocked());
  if (!JTMB) {
    Err = JTMB.takeError();
    return;
  }

  // Set up the JIT compile pipeline.
  setupLayers(std::move(*JTMB), NumCompileThreads, DiscoveryFlagsPerBucket,
              MemFence);

  // We can use the mangler now. Remember the mangled name of the main function.
  MainFunctionMangled = (*Mangle)(MainFunctionName);

  // We are restricted to a single dylib currently. Add runtime overrides and
  // symbol generators.
  MainJD = &ES.createBareJITDylib("main");
  Err = setupJITDylib(MainJD, AllowNudgeIntoDiscovery, PrintStats);
  if (Err)
    return;

  // Spawn discovery thread and let it add newly discovered modules to the JIT.
  setupDiscovery(MainJD, LookaheadLevels, PrintStats);

  Err = AddModule(std::move(MainModule));
  if (Err)
    return;

  if (AllowNudgeIntoDiscovery) {
    auto MainFunctionGuid = GlobalValue::getGUID(MainFunctionName);
    InstrumentationLayer->nudgeIntoDiscovery({MainFunctionGuid});
  }
}

Expected<ThreadSafeModule> ThinLtoJIT::setupMainModule(StringRef MainFunction) {
  Optional<StringRef> M = GlobalIndex->getModulePathForSymbol(MainFunction);
  if (!M) {
    std::string Buffer;
    raw_string_ostream OS(Buffer);
    OS << "No ValueInfo for symbol '" << MainFunction;
    OS << "' in provided modules: ";
    for (StringRef P : GlobalIndex->getAllModulePaths())
      OS << P << " ";
    OS << "\n";
    return createStringError(inconvertibleErrorCode(), OS.str());
  }

  if (auto TSM = GlobalIndex->parseModuleFromFile(*M))
    return std::move(TSM); // Not a redundant move: fix build on gcc-7.5

  return createStringError(inconvertibleErrorCode(),
                           "Failed to parse main module");
}

Expected<JITTargetMachineBuilder> ThinLtoJIT::setupTargetUtils(Module *M) {
  std::string T = M->getTargetTriple();
  JITTargetMachineBuilder JTMB(Triple(T.empty() ? sys::getProcessTriple() : T));

  // CallThroughManager is ABI-specific
  auto LCTM = createLocalLazyCallThroughManager(
      JTMB.getTargetTriple(), ES,
      pointerToJITTargetAddress(exitOnLazyCallThroughFailure));
  if (!LCTM)
    return LCTM.takeError();
  CallThroughManager = std::move(*LCTM);

  // Use DataLayout or the given module or fall back to the host's default.
  DL = DataLayout(M);
  if (DL.getStringRepresentation().empty()) {
    auto HostDL = JTMB.getDefaultDataLayoutForTarget();
    if (!HostDL)
      return HostDL.takeError();
    DL = std::move(*HostDL);
    if (Error Err = applyDataLayout(M))
      return std::move(Err);
  }

  // Now that we know the target data layout we can setup the mangler.
  Mangle = std::make_unique<MangleAndInterner>(ES, DL);
  return JTMB;
}

Error ThinLtoJIT::applyDataLayout(Module *M) {
  if (M->getDataLayout().isDefault())
    M->setDataLayout(DL);

  if (M->getDataLayout() != DL)
    return make_error<StringError>(
        "Added modules have incompatible data layouts",
        inconvertibleErrorCode());

  return Error::success();
}

static bool IsTrivialModule(MaterializationUnit *MU) {
  StringRef ModuleName = MU->getName();
  return ModuleName == "<Lazy Reexports>" || ModuleName == "<Reexports>" ||
         ModuleName == "<Absolute Symbols>";
}

void ThinLtoJIT::setupLayers(JITTargetMachineBuilder JTMB,
                             unsigned NumCompileThreads,
                             unsigned DiscoveryFlagsPerBucket,
                             ExplicitMemoryBarrier MemFence) {
  ObjLinkingLayer = std::make_unique<RTDyldObjectLinkingLayer>(
      ES, []() { return std::make_unique<SectionMemoryManager>(); });

  CompileLayer = std::make_unique<IRCompileLayer>(
      ES, *ObjLinkingLayer, std::make_unique<ConcurrentIRCompiler>(JTMB));

  InstrumentationLayer = std::make_unique<ThinLtoInstrumentationLayer>(
      ES, *CompileLayer, MemFence, DiscoveryFlagsPerBucket);

  OnDemandLayer = std::make_unique<CompileOnDemandLayer>(
      ES, *InstrumentationLayer, *CallThroughManager,
      createLocalIndirectStubsManagerBuilder(JTMB.getTargetTriple()));
  // Don't break up modules. Insert stubs on module boundaries.
  OnDemandLayer->setPartitionFunction(CompileOnDemandLayer::compileWholeModule);

  // Delegate compilation to the thread pool.
  CompileThreads = std::make_unique<ThreadPool>(
      llvm::hardware_concurrency(NumCompileThreads));
  ES.setDispatchMaterialization(
      [this](JITDylib &JD, std::unique_ptr<MaterializationUnit> MU) {
        if (IsTrivialModule(MU.get())) {
          // This should be quick and we may save a few session locks.
          MU->doMaterialize(JD);
        } else {
          // FIXME: Drop the std::shared_ptr workaround once ThreadPool::async()
          // accepts llvm::unique_function to define jobs.
          auto SharedMU = std::shared_ptr<MaterializationUnit>(std::move(MU));
          CompileThreads->async(
              [MU = std::move(SharedMU), &JD]() { MU->doMaterialize(JD); });
        }
      });

  AddModule = [this](ThreadSafeModule TSM) -> Error {
    assert(MainJD && "Setup MainJD JITDylib before calling");
    Module *M = TSM.getModuleUnlocked();
    if (Error Err = applyDataLayout(M))
      return Err;
    VModuleKey Id = GlobalIndex->getModuleId(M->getName());
    return OnDemandLayer->add(*MainJD, std::move(TSM), Id);
  };
}

void ThinLtoJIT::setupDiscovery(JITDylib *MainJD, unsigned LookaheadLevels,
                                bool PrintStats) {
  JitRunning.store(true);
  DiscoveryThreadWorker = std::make_unique<ThinLtoDiscoveryThread>(
      JitRunning, ES, MainJD, *InstrumentationLayer, *GlobalIndex, AddModule,
      LookaheadLevels, PrintStats);

  DiscoveryThread = std::thread(std::ref(*DiscoveryThreadWorker));
}

Error ThinLtoJIT::setupJITDylib(JITDylib *JD, bool AllowNudge,
                                bool PrintStats) {
  // Register symbols for C++ static destructors.
  LocalCXXRuntimeOverrides CXXRuntimeoverrides;
  Error Err = CXXRuntimeoverrides.enable(*JD, *Mangle);
  if (Err)
    return Err;

  // Lookup symbol names in the global ThinLTO module index first
  char Prefix = DL.getGlobalPrefix();
  JD->addGenerator(std::make_unique<ThinLtoDefinitionGenerator>(
      *GlobalIndex, *InstrumentationLayer, AddModule, Prefix, AllowNudge,
      PrintStats));

  // Then try lookup in the host process.
  auto HostLookup = DynamicLibrarySearchGenerator::GetForCurrentProcess(Prefix);
  if (!HostLookup)
    return HostLookup.takeError();
  JD->addGenerator(std::move(*HostLookup));

  return Error::success();
}

ThinLtoJIT::~ThinLtoJIT() {
  // Signal the DiscoveryThread to shut down.
  JitRunning.store(false);
  DiscoveryThread.join();

  // Wait for potential compile actions to finish.
  CompileThreads->wait();
}

} // namespace orc
} // namespace llvm
