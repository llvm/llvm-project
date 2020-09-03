#ifndef LLVM_EXAMPLES_THINLTOJIT_THINLTOJIT_H
#define LLVM_EXAMPLES_THINLTOJIT_THINLTOJIT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ThreadPool.h"

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace llvm {
namespace orc {

class ThinLtoDiscoveryThread;
class ThinLtoInstrumentationLayer;
class ThinLtoModuleIndex;

class CompileOnDemandLayer;
class IRCompileLayer;
class RTDyldObjectLinkingLayer;

class JITDylib;
class JITTargetMachineBuilder;
class LazyCallThroughManager;
class MangleAndInterner;

class ThinLtoJIT {
public:
  using AddModuleFunction = std::function<Error(ThreadSafeModule)>;

  enum ExplicitMemoryBarrier {
    NeverFence = 0,
    FenceStaticCode = 1,
    FenceJITedCode = 2,
    AlwaysFence = 3
  };

  ThinLtoJIT(ArrayRef<std::string> InputFiles, StringRef MainFunctionName,
             unsigned LookaheadLevels, unsigned NumCompileThreads,
             unsigned NumLoadThreads, unsigned DiscoveryFlagsPerBucket,
             ExplicitMemoryBarrier MemFence, bool AllowNudgeIntoDiscovery,
             bool PrintStats, Error &Err);
  ~ThinLtoJIT();

  ThinLtoJIT(const ThinLtoJIT &) = delete;
  ThinLtoJIT &operator=(const ThinLtoJIT &) = delete;
  ThinLtoJIT(ThinLtoJIT &&) = delete;
  ThinLtoJIT &operator=(ThinLtoJIT &&) = delete;

  Expected<int> main(ArrayRef<std::string> Args) {
    auto MainSym = ES.lookup({MainJD}, MainFunctionMangled);
    if (!MainSym)
      return MainSym.takeError();

    using MainFn = int(int, char *[]);
    auto Main = jitTargetAddressToFunction<MainFn *>(MainSym->getAddress());

    return runAsMain(Main, Args, StringRef("ThinLtoJIT"));
  }

private:
  ExecutionSession ES;
  DataLayout DL{""};

  JITDylib *MainJD;
  SymbolStringPtr MainFunctionMangled;
  std::unique_ptr<ThreadPool> CompileThreads;
  std::unique_ptr<ThinLtoModuleIndex> GlobalIndex;

  AddModuleFunction AddModule;
  std::unique_ptr<RTDyldObjectLinkingLayer> ObjLinkingLayer;
  std::unique_ptr<IRCompileLayer> CompileLayer;
  std::unique_ptr<ThinLtoInstrumentationLayer> InstrumentationLayer;
  std::unique_ptr<CompileOnDemandLayer> OnDemandLayer;

  std::atomic<bool> JitRunning;
  std::thread DiscoveryThread;
  std::unique_ptr<ThinLtoDiscoveryThread> DiscoveryThreadWorker;

  std::unique_ptr<MangleAndInterner> Mangle;
  std::unique_ptr<LazyCallThroughManager> CallThroughManager;

  void setupLayers(JITTargetMachineBuilder JTMB, unsigned NumCompileThreads,
                   unsigned DiscoveryFlagsPerBucket,
                   ExplicitMemoryBarrier MemFence);
  Error setupJITDylib(JITDylib *JD, bool AllowNudge, bool PrintStats);
  void setupDiscovery(JITDylib *MainJD, unsigned LookaheadLevels,
                      bool PrintStats);
  Expected<ThreadSafeModule> setupMainModule(StringRef MainFunction);
  Expected<JITTargetMachineBuilder> setupTargetUtils(Module *M);
  Error applyDataLayout(Module *M);

  static void exitOnLazyCallThroughFailure() {
    errs() << "Compilation failed. Aborting.\n";
    exit(1);
  }
};

} // namespace orc
} // namespace llvm

#endif
