//===----- LLJIT.h -- An ORC-based JIT for compiling LLVM IR ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An ORC-based JIT for compiling LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LLJIT_H
#define LLVM_EXECUTIONENGINE_ORC_LLJIT_H

#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ThreadPool.h"

namespace llvm {
namespace orc {

class LLJITBuilderState;
class LLLazyJITBuilderState;

/// A pre-fabricated ORC JIT stack that can serve as an alternative to MCJIT.
///
/// Create instances using LLJITBuilder.
class LLJIT {
  template <typename, typename, typename> friend class LLJITBuilderSetters;

  friend void setUpGenericLLVMIRPlatform(LLJIT &J);

public:
  /// Initializer support for LLJIT.
  class PlatformSupport {
  public:
    virtual ~PlatformSupport();

    virtual Error initialize(JITDylib &JD) = 0;

    virtual Error deinitialize(JITDylib &JD) = 0;

  protected:
    static void setInitTransform(LLJIT &J,
                                 IRTransformLayer::TransformFunction T);
  };

  /// Destruct this instance. If a multi-threaded instance, waits for all
  /// compile threads to complete.
  ~LLJIT();

  /// Returns the ExecutionSession for this instance.
  ExecutionSession &getExecutionSession() { return *ES; }

  /// Returns a reference to the triple for this instance.
  const Triple &getTargetTriple() const { return TT; }

  /// Returns a reference to the DataLayout for this instance.
  const DataLayout &getDataLayout() const { return DL; }

  /// Returns a reference to the JITDylib representing the JIT'd main program.
  JITDylib &getMainJITDylib() { return *Main; }

  /// Returns the JITDylib with the given name, or nullptr if no JITDylib with
  /// that name exists.
  JITDylib *getJITDylibByName(StringRef Name) {
    return ES->getJITDylibByName(Name);
  }

  /// Create a new JITDylib with the given name and return a reference to it.
  ///
  /// JITDylib names must be unique. If the given name is derived from user
  /// input or elsewhere in the environment then the client should check
  /// (e.g. by calling getJITDylibByName) that the given name is not already in
  /// use.
  Expected<JITDylib &> createJITDylib(std::string Name) {
    return ES->createJITDylib(std::move(Name));
  }

  /// A convenience method for defining MUs in LLJIT's Main JITDylib. This can
  /// be useful for succinctly defining absolute symbols, aliases and
  /// re-exports.
  template <typename MUType>
  Error define(std::unique_ptr<MUType> &&MU) {
    return Main->define(std::move(MU));
  }

  /// A convenience method for defining MUs in LLJIT's Main JITDylib. This can
  /// be usedful for succinctly defining absolute symbols, aliases and
  /// re-exports.
  template <typename MUType>
  Error define(std::unique_ptr<MUType> &MU) {
    return Main->define(MU);
  }

  /// Adds an IR module to the given JITDylib.
  Error addIRModule(JITDylib &JD, ThreadSafeModule TSM);

  /// Adds an IR module to the Main JITDylib.
  Error addIRModule(ThreadSafeModule TSM) {
    return addIRModule(*Main, std::move(TSM));
  }

  /// Adds an object file to the given JITDylib.
  Error addObjectFile(JITDylib &JD, std::unique_ptr<MemoryBuffer> Obj);

  /// Adds an object file to the given JITDylib.
  Error addObjectFile(std::unique_ptr<MemoryBuffer> Obj) {
    return addObjectFile(*Main, std::move(Obj));
  }

  /// Look up a symbol in JITDylib JD by the symbol's linker-mangled name (to
  /// look up symbols based on their IR name use the lookup function instead).
  Expected<JITEvaluatedSymbol> lookupLinkerMangled(JITDylib &JD,
                                                   SymbolStringPtr Name);

  /// Look up a symbol in JITDylib JD by the symbol's linker-mangled name (to
  /// look up symbols based on their IR name use the lookup function instead).
  Expected<JITEvaluatedSymbol> lookupLinkerMangled(JITDylib &JD,
                                                   StringRef Name) {
    return lookupLinkerMangled(JD, ES->intern(Name));
  }

  /// Look up a symbol in the main JITDylib by the symbol's linker-mangled name
  /// (to look up symbols based on their IR name use the lookup function
  /// instead).
  Expected<JITEvaluatedSymbol> lookupLinkerMangled(StringRef Name) {
    return lookupLinkerMangled(*Main, Name);
  }

  /// Look up a symbol in JITDylib JD based on its IR symbol name.
  Expected<JITEvaluatedSymbol> lookup(JITDylib &JD, StringRef UnmangledName) {
    return lookupLinkerMangled(JD, mangle(UnmangledName));
  }

  /// Look up a symbol in the main JITDylib based on its IR symbol name.
  Expected<JITEvaluatedSymbol> lookup(StringRef UnmangledName) {
    return lookup(*Main, UnmangledName);
  }

  /// Set the PlatformSupport instance.
  void setPlatformSupport(std::unique_ptr<PlatformSupport> PS) {
    this->PS = std::move(PS);
  }

  /// Get the PlatformSupport instance.
  PlatformSupport *getPlatformSupport() { return PS.get(); }

  /// Run the initializers for the given JITDylib.
  Error initialize(JITDylib &JD) {
    DEBUG_WITH_TYPE("orc", {
      dbgs() << "LLJIT running initializers for JITDylib \"" << JD.getName()
             << "\"\n";
    });
    assert(PS && "PlatformSupport must be set to run initializers.");
    return PS->initialize(JD);
  }

  /// Run the deinitializers for the given JITDylib.
  Error deinitialize(JITDylib &JD) {
    DEBUG_WITH_TYPE("orc", {
      dbgs() << "LLJIT running deinitializers for JITDylib \"" << JD.getName()
             << "\"\n";
    });
    assert(PS && "PlatformSupport must be set to run initializers.");
    return PS->deinitialize(JD);
  }

  /// Returns a reference to the ObjLinkingLayer
  ObjectLayer &getObjLinkingLayer() { return *ObjLinkingLayer; }

  /// Returns a reference to the object transform layer.
  ObjectTransformLayer &getObjTransformLayer() { return ObjTransformLayer; }

  /// Returns a reference to the IR transform layer.
  IRTransformLayer &getIRTransformLayer() { return *TransformLayer; }

  /// Returns a reference to the IR compile layer.
  IRCompileLayer &getIRCompileLayer() { return *CompileLayer; }

  /// Returns a linker-mangled version of UnmangledName.
  std::string mangle(StringRef UnmangledName) const;

  /// Returns an interned, linker-mangled version of UnmangledName.
  SymbolStringPtr mangleAndIntern(StringRef UnmangledName) const {
    return ES->intern(mangle(UnmangledName));
  }

protected:
  static std::unique_ptr<ObjectLayer>
  createObjectLinkingLayer(LLJITBuilderState &S, ExecutionSession &ES);

  static Expected<std::unique_ptr<IRCompileLayer::IRCompiler>>
  createCompileFunction(LLJITBuilderState &S, JITTargetMachineBuilder JTMB);

  /// Create an LLJIT instance with a single compile thread.
  LLJIT(LLJITBuilderState &S, Error &Err);

  Error applyDataLayout(Module &M);

  void recordCtorDtors(Module &M);

  std::unique_ptr<ExecutionSession> ES;
  std::unique_ptr<PlatformSupport> PS;

  JITDylib *Main = nullptr;

  DataLayout DL;
  Triple TT;
  std::unique_ptr<ThreadPool> CompileThreads;

  std::unique_ptr<ObjectLayer> ObjLinkingLayer;
  ObjectTransformLayer ObjTransformLayer;
  std::unique_ptr<IRCompileLayer> CompileLayer;
  std::unique_ptr<IRTransformLayer> TransformLayer;
  std::unique_ptr<IRTransformLayer> InitHelperTransformLayer;
};

/// An extended version of LLJIT that supports lazy function-at-a-time
/// compilation of LLVM IR.
class LLLazyJIT : public LLJIT {
  template <typename, typename, typename> friend class LLJITBuilderSetters;

public:

  /// Sets the partition function.
  void
  setPartitionFunction(CompileOnDemandLayer::PartitionFunction Partition) {
    CODLayer->setPartitionFunction(std::move(Partition));
  }

  /// Add a module to be lazily compiled to JITDylib JD.
  Error addLazyIRModule(JITDylib &JD, ThreadSafeModule M);

  /// Add a module to be lazily compiled to the main JITDylib.
  Error addLazyIRModule(ThreadSafeModule M) {
    return addLazyIRModule(*Main, std::move(M));
  }

private:

  // Create a single-threaded LLLazyJIT instance.
  LLLazyJIT(LLLazyJITBuilderState &S, Error &Err);

  std::unique_ptr<LazyCallThroughManager> LCTMgr;
  std::unique_ptr<CompileOnDemandLayer> CODLayer;
};

class LLJITBuilderState {
public:
  using ObjectLinkingLayerCreator = std::function<std::unique_ptr<ObjectLayer>(
      ExecutionSession &, const Triple &TT)>;

  using CompileFunctionCreator =
      std::function<Expected<std::unique_ptr<IRCompileLayer::IRCompiler>>(
          JITTargetMachineBuilder JTMB)>;

  using PlatformSetupFunction = std::function<Error(LLJIT &J)>;

  std::unique_ptr<ExecutionSession> ES;
  Optional<JITTargetMachineBuilder> JTMB;
  Optional<DataLayout> DL;
  ObjectLinkingLayerCreator CreateObjectLinkingLayer;
  CompileFunctionCreator CreateCompileFunction;
  PlatformSetupFunction SetUpPlatform;
  unsigned NumCompileThreads = 0;

  /// Called prior to JIT class construcion to fix up defaults.
  Error prepareForConstruction();
};

template <typename JITType, typename SetterImpl, typename State>
class LLJITBuilderSetters {
public:

  /// Set an ExecutionSession for this instance.
  SetterImpl &setExecutionSession(std::unique_ptr<ExecutionSession> ES) {
    impl().ES = std::move(ES);
    return impl();
  }

  /// Set the JITTargetMachineBuilder for this instance.
  ///
  /// If this method is not called, JITTargetMachineBuilder::detectHost will be
  /// used to construct a default target machine builder for the host platform.
  SetterImpl &setJITTargetMachineBuilder(JITTargetMachineBuilder JTMB) {
    impl().JTMB = std::move(JTMB);
    return impl();
  }

  /// Return a reference to the JITTargetMachineBuilder.
  ///
  Optional<JITTargetMachineBuilder> &getJITTargetMachineBuilder() {
    return impl().JTMB;
  }

  /// Set a DataLayout for this instance. If no data layout is specified then
  /// the target's default data layout will be used.
  SetterImpl &setDataLayout(Optional<DataLayout> DL) {
    impl().DL = std::move(DL);
    return impl();
  }

  /// Set an ObjectLinkingLayer creation function.
  ///
  /// If this method is not called, a default creation function will be used
  /// that will construct an RTDyldObjectLinkingLayer.
  SetterImpl &setObjectLinkingLayerCreator(
      LLJITBuilderState::ObjectLinkingLayerCreator CreateObjectLinkingLayer) {
    impl().CreateObjectLinkingLayer = std::move(CreateObjectLinkingLayer);
    return impl();
  }

  /// Set a CompileFunctionCreator.
  ///
  /// If this method is not called, a default creation function wil be used
  /// that will construct a basic IR compile function that is compatible with
  /// the selected number of threads (SimpleCompiler for '0' compile threads,
  /// ConcurrentIRCompiler otherwise).
  SetterImpl &setCompileFunctionCreator(
      LLJITBuilderState::CompileFunctionCreator CreateCompileFunction) {
    impl().CreateCompileFunction = std::move(CreateCompileFunction);
    return impl();
  }

  /// Set up an PlatformSetupFunction.
  ///
  /// If this method is not called then setUpGenericLLVMIRPlatform
  /// will be used to configure the JIT's platform support.
  SetterImpl &
  setPlatformSetUp(LLJITBuilderState::PlatformSetupFunction SetUpPlatform) {
    impl().SetUpPlatform = std::move(SetUpPlatform);
    return impl();
  }

  /// Set the number of compile threads to use.
  ///
  /// If set to zero, compilation will be performed on the execution thread when
  /// JITing in-process. If set to any other number N, a thread pool of N
  /// threads will be created for compilation.
  ///
  /// If this method is not called, behavior will be as if it were called with
  /// a zero argument.
  SetterImpl &setNumCompileThreads(unsigned NumCompileThreads) {
    impl().NumCompileThreads = NumCompileThreads;
    return impl();
  }

  /// Create an instance of the JIT.
  Expected<std::unique_ptr<JITType>> create() {
    if (auto Err = impl().prepareForConstruction())
      return std::move(Err);

    Error Err = Error::success();
    std::unique_ptr<JITType> J(new JITType(impl(), Err));
    if (Err)
      return std::move(Err);
    return std::move(J);
  }

protected:
  SetterImpl &impl() { return static_cast<SetterImpl &>(*this); }
};

/// Constructs LLJIT instances.
class LLJITBuilder
    : public LLJITBuilderState,
      public LLJITBuilderSetters<LLJIT, LLJITBuilder, LLJITBuilderState> {};

class LLLazyJITBuilderState : public LLJITBuilderState {
  friend class LLLazyJIT;

public:
  using IndirectStubsManagerBuilderFunction =
      std::function<std::unique_ptr<IndirectStubsManager>()>;

  Triple TT;
  JITTargetAddress LazyCompileFailureAddr = 0;
  std::unique_ptr<LazyCallThroughManager> LCTMgr;
  IndirectStubsManagerBuilderFunction ISMBuilder;

  Error prepareForConstruction();
};

template <typename JITType, typename SetterImpl, typename State>
class LLLazyJITBuilderSetters
    : public LLJITBuilderSetters<JITType, SetterImpl, State> {
public:
  /// Set the address in the target address to call if a lazy compile fails.
  ///
  /// If this method is not called then the value will default to 0.
  SetterImpl &setLazyCompileFailureAddr(JITTargetAddress Addr) {
    this->impl().LazyCompileFailureAddr = Addr;
    return this->impl();
  }

  /// Set the lazy-callthrough manager.
  ///
  /// If this method is not called then a default, in-process lazy callthrough
  /// manager for the host platform will be used.
  SetterImpl &
  setLazyCallthroughManager(std::unique_ptr<LazyCallThroughManager> LCTMgr) {
    this->impl().LCTMgr = std::move(LCTMgr);
    return this->impl();
  }

  /// Set the IndirectStubsManager builder function.
  ///
  /// If this method is not called then a default, in-process
  /// IndirectStubsManager builder for the host platform will be used.
  SetterImpl &setIndirectStubsManagerBuilder(
      LLLazyJITBuilderState::IndirectStubsManagerBuilderFunction ISMBuilder) {
    this->impl().ISMBuilder = std::move(ISMBuilder);
    return this->impl();
  }
};

/// Constructs LLLazyJIT instances.
class LLLazyJITBuilder
    : public LLLazyJITBuilderState,
      public LLLazyJITBuilderSetters<LLLazyJIT, LLLazyJITBuilder,
                                     LLLazyJITBuilderState> {};

/// Configure the LLJIT instance to scrape modules for llvm.global_ctors and
/// llvm.global_dtors variables and (if present) build initialization and
/// deinitialization functions. Platform specific initialization configurations
/// should be preferred where available.
void setUpGenericLLVMIRPlatform(LLJIT &J);

/// Configure the LLJIT instance to use MachOPlatform support.
///
/// Warning: MachOPlatform *requires* that LLJIT be configured to use
/// ObjectLinkingLayer (default on platforms supported by JITLink). If
/// MachOPlatform is used with RTDyldObjectLinkingLayer it will result in
/// undefined behavior).
///
/// MachOPlatform installs an ObjectLinkingLayer plugin to scrape initializers
/// from the __mod_inits section. It also provides interposes for the dlfcn
/// functions (dlopen, dlclose, dlsym, dlerror) that work for JITDylibs as
/// well as regular libraries (JITDylibs will be preferenced, so make sure
/// your JITDylib names do not shadow any real library paths).
Error setUpMachOPlatform(LLJIT &J);

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_LLJIT_H
