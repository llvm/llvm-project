//===-- MachOPlatform.h - Utilities for executing MachO in Orc --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for executing JIT'd MachO in Orc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_MACHOPLATFORM_H
#define LLVM_EXECUTIONENGINE_ORC_MACHOPLATFORM_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"

#include <future>
#include <thread>
#include <vector>

namespace llvm {
namespace orc {

/// Mediates between MachO initialization and ExecutionSession state.
class MachOPlatform : public Platform {
public:
  // Used internally by MachOPlatform, but made public to enable serialization.
  struct MachOJITDylibDepInfo {
    bool Sealed = false;
    std::vector<ExecutorAddr> DepHeaders;
  };

  // Used internally by MachOPlatform, but made public to enable serialization.
  using MachOJITDylibDepInfoMap =
      std::vector<std::pair<ExecutorAddr, MachOJITDylibDepInfo>>;

  /// Try to create a MachOPlatform instance, adding the ORC runtime to the
  /// given JITDylib.
  ///
  /// The ORC runtime requires access to a number of symbols in libc++, and
  /// requires access to symbols in libobjc, and libswiftCore to support
  /// Objective-C and Swift code. It is up to the caller to ensure that the
  /// requried symbols can be referenced by code added to PlatformJD. The
  /// standard way to achieve this is to first attach dynamic library search
  /// generators for either the given process, or for the specific required
  /// libraries, to PlatformJD, then to create the platform instance:
  ///
  /// \code{.cpp}
  ///   auto &PlatformJD = ES.createBareJITDylib("stdlib");
  ///   PlatformJD.addGenerator(
  ///     ExitOnErr(EPCDynamicLibrarySearchGenerator
  ///                 ::GetForTargetProcess(EPC)));
  ///   ES.setPlatform(
  ///     ExitOnErr(MachOPlatform::Create(ES, ObjLayer, EPC, PlatformJD,
  ///                                     "/path/to/orc/runtime")));
  /// \endcode
  ///
  /// Alternatively, these symbols could be added to another JITDylib that
  /// PlatformJD links against.
  ///
  /// Clients are also responsible for ensuring that any JIT'd code that
  /// depends on runtime functions (including any code using TLV or static
  /// destructors) can reference the runtime symbols. This is usually achieved
  /// by linking any JITDylibs containing regular code against
  /// PlatformJD.
  ///
  /// By default, MachOPlatform will add the set of aliases returned by the
  /// standardPlatformAliases function. This includes both required aliases
  /// (e.g. __cxa_atexit -> __orc_rt_macho_cxa_atexit for static destructor
  /// support), and optional aliases that provide JIT versions of common
  /// functions (e.g. dlopen -> __orc_rt_macho_jit_dlopen). Clients can
  /// override these defaults by passing a non-None value for the
  /// RuntimeAliases function, in which case the client is responsible for
  /// setting up all aliases (including the required ones).
  static Expected<std::unique_ptr<MachOPlatform>>
  Create(ExecutionSession &ES, ObjectLinkingLayer &ObjLinkingLayer,
         JITDylib &PlatformJD, std::unique_ptr<DefinitionGenerator> OrcRuntime,
         std::optional<SymbolAliasMap> RuntimeAliases = std::nullopt);

  /// Construct using a path to the ORC runtime.
  static Expected<std::unique_ptr<MachOPlatform>>
  Create(ExecutionSession &ES, ObjectLinkingLayer &ObjLinkingLayer,
         JITDylib &PlatformJD, const char *OrcRuntimePath,
         std::optional<SymbolAliasMap> RuntimeAliases = std::nullopt);

  ExecutionSession &getExecutionSession() const { return ES; }
  ObjectLinkingLayer &getObjectLinkingLayer() const { return ObjLinkingLayer; }

  Error setupJITDylib(JITDylib &JD) override;
  Error teardownJITDylib(JITDylib &JD) override;
  Error notifyAdding(ResourceTracker &RT,
                     const MaterializationUnit &MU) override;
  Error notifyRemoving(ResourceTracker &RT) override;

  /// Returns an AliasMap containing the default aliases for the MachOPlatform.
  /// This can be modified by clients when constructing the platform to add
  /// or remove aliases.
  static SymbolAliasMap standardPlatformAliases(ExecutionSession &ES);

  /// Returns the array of required CXX aliases.
  static ArrayRef<std::pair<const char *, const char *>> requiredCXXAliases();

  /// Returns the array of standard runtime utility aliases for MachO.
  static ArrayRef<std::pair<const char *, const char *>>
  standardRuntimeUtilityAliases();

private:
  // Data needed for bootstrap only.
  struct BootstrapInfo {
    std::mutex Mutex;
    std::condition_variable CV;
    size_t ActiveGraphs = 0;
    shared::AllocActions DeferredAAs;
    ExecutorAddr MachOHeaderAddr;
  };

  // The MachOPlatformPlugin scans/modifies LinkGraphs to support MachO
  // platform features including initializers, exceptions, TLV, and language
  // runtime registration.
  class MachOPlatformPlugin : public ObjectLinkingLayer::Plugin {
  public:
    MachOPlatformPlugin(MachOPlatform &MP) : MP(MP) {}

    void modifyPassConfig(MaterializationResponsibility &MR,
                          jitlink::LinkGraph &G,
                          jitlink::PassConfiguration &Config) override;

    SyntheticSymbolDependenciesMap
    getSyntheticSymbolDependencies(MaterializationResponsibility &MR) override;

    // FIXME: We should be tentatively tracking scraped sections and discarding
    // if the MR fails.
    Error notifyFailed(MaterializationResponsibility &MR) override {
      return Error::success();
    }

    Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override {
      return Error::success();
    }

    void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                     ResourceKey SrcKey) override {}

  private:
    using InitSymbolDepMap =
        DenseMap<MaterializationResponsibility *, JITLinkSymbolSet>;

    struct UnwindSections {
      SmallVector<ExecutorAddrRange> CodeRanges;
      ExecutorAddrRange DwarfSection;
      ExecutorAddrRange CompactUnwindSection;
    };

    Error bootstrapPipelineStart(jitlink::LinkGraph &G);
    Error bootstrapPipelineRecordRuntimeFunctions(jitlink::LinkGraph &G);
    Error bootstrapPipelineEnd(jitlink::LinkGraph &G);

    Error recordRuntimeRegistrationFunctions(jitlink::LinkGraph &G);

    Error associateJITDylibHeaderSymbol(jitlink::LinkGraph &G,
                                        MaterializationResponsibility &MR);

    Error preserveInitSections(jitlink::LinkGraph &G,
                               MaterializationResponsibility &MR);

    Error processObjCImageInfo(jitlink::LinkGraph &G,
                               MaterializationResponsibility &MR);

    Error fixTLVSectionsAndEdges(jitlink::LinkGraph &G, JITDylib &JD);

    std::optional<UnwindSections> findUnwindSectionInfo(jitlink::LinkGraph &G);

    Error registerObjectPlatformSections(jitlink::LinkGraph &G, JITDylib &JD,
                                         bool InBootstrapPhase);

    std::mutex PluginMutex;
    MachOPlatform &MP;

    // FIXME: ObjCImageInfos and HeaderAddrs need to be cleared when
    // JITDylibs are removed.
    DenseMap<JITDylib *, std::pair<uint32_t, uint32_t>> ObjCImageInfos;
    DenseMap<JITDylib *, ExecutorAddr> HeaderAddrs;
    InitSymbolDepMap InitSymbolDeps;
  };

  using GetJITDylibHeaderSendResultFn =
      unique_function<void(Expected<ExecutorAddr>)>;
  using GetJITDylibNameSendResultFn =
      unique_function<void(Expected<StringRef>)>;
  using PushInitializersSendResultFn =
      unique_function<void(Expected<MachOJITDylibDepInfoMap>)>;
  using SendSymbolAddressFn = unique_function<void(Expected<ExecutorAddr>)>;

  static bool supportedTarget(const Triple &TT);

  MachOPlatform(ExecutionSession &ES, ObjectLinkingLayer &ObjLinkingLayer,
                JITDylib &PlatformJD,
                std::unique_ptr<DefinitionGenerator> OrcRuntimeGenerator,
                Error &Err);

  // Associate MachOPlatform JIT-side runtime support functions with handlers.
  Error associateRuntimeSupportFunctions();

  // Implements rt_pushInitializers by making repeat async lookups for
  // initializer symbols (each lookup may spawn more initializer symbols if
  // it pulls in new materializers, e.g. from objects in a static library).
  void pushInitializersLoop(PushInitializersSendResultFn SendResult,
                            JITDylibSP JD);

  // Handle requests from the ORC runtime to push MachO initializer info.
  void rt_pushInitializers(PushInitializersSendResultFn SendResult,
                           ExecutorAddr JDHeaderAddr);

  // Handle requests for symbol addresses from the ORC runtime.
  void rt_lookupSymbol(SendSymbolAddressFn SendResult, ExecutorAddr Handle,
                       StringRef SymbolName);

  // Call the ORC runtime to create a pthread key.
  Expected<uint64_t> createPThreadKey();

  ExecutionSession &ES;
  JITDylib &PlatformJD;
  ObjectLinkingLayer &ObjLinkingLayer;

  SymbolStringPtr MachOHeaderStartSymbol = ES.intern("___dso_handle");

  struct RuntimeFunction {
    RuntimeFunction(SymbolStringPtr Name) : Name(std::move(Name)) {}
    SymbolStringPtr Name;
    ExecutorAddr Addr;
  };

  RuntimeFunction PlatformBootstrap{
      ES.intern("___orc_rt_macho_platform_bootstrap")};
  RuntimeFunction PlatformShutdown{
      ES.intern("___orc_rt_macho_platform_shutdown")};
  RuntimeFunction RegisterEHFrameSection{
      ES.intern("___orc_rt_macho_register_ehframe_section")};
  RuntimeFunction DeregisterEHFrameSection{
      ES.intern("___orc_rt_macho_deregister_ehframe_section")};
  RuntimeFunction RegisterJITDylib{
      ES.intern("___orc_rt_macho_register_jitdylib")};
  RuntimeFunction DeregisterJITDylib{
      ES.intern("___orc_rt_macho_deregister_jitdylib")};
  RuntimeFunction RegisterObjectPlatformSections{
      ES.intern("___orc_rt_macho_register_object_platform_sections")};
  RuntimeFunction DeregisterObjectPlatformSections{
      ES.intern("___orc_rt_macho_deregister_object_platform_sections")};
  RuntimeFunction CreatePThreadKey{
      ES.intern("___orc_rt_macho_create_pthread_key")};

  DenseMap<JITDylib *, SymbolLookupSet> RegisteredInitSymbols;

  std::mutex PlatformMutex;
  DenseMap<JITDylib *, ExecutorAddr> JITDylibToHeaderAddr;
  DenseMap<ExecutorAddr, JITDylib *> HeaderAddrToJITDylib;
  DenseMap<JITDylib *, uint64_t> JITDylibToPThreadKey;

  std::atomic<BootstrapInfo *> Bootstrap;
};

namespace shared {

using SPSNamedExecutorAddrRangeSequence =
    SPSSequence<SPSTuple<SPSString, SPSExecutorAddrRange>>;

} // end namespace shared
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_MACHOPLATFORM_H
