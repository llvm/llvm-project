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
#include "llvm/ExecutionEngine/Orc/LLVMSPSSerializers.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"

#include <future>
#include <thread>
#include <vector>

namespace llvm {
namespace orc {

struct MachOPerObjectSectionsToRegister {
  ExecutorAddressRange EHFrameSection;
  ExecutorAddressRange ThreadDataSection;
};

struct MachOJITDylibInitializers {
  using SectionList = std::vector<ExecutorAddressRange>;

  MachOJITDylibInitializers(std::string Name,
                            ExecutorAddress MachOHeaderAddress)
      : Name(std::move(Name)),
        MachOHeaderAddress(std::move(MachOHeaderAddress)) {}

  std::string Name;
  ExecutorAddress MachOHeaderAddress;
  ExecutorAddress ObjCImageInfoAddress;

  StringMap<SectionList> InitSections;
};

class MachOJITDylibDeinitializers {};

using MachOJITDylibInitializerSequence = std::vector<MachOJITDylibInitializers>;

using MachOJITDylibDeinitializerSequence =
    std::vector<MachOJITDylibDeinitializers>;

/// Mediates between MachO initialization and ExecutionSession state.
class MachOPlatform : public Platform {
public:
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
         JITDylib &PlatformJD, const char *OrcRuntimePath,
         Optional<SymbolAliasMap> RuntimeAliases = None);

  ExecutionSession &getExecutionSession() const { return ES; }
  ObjectLinkingLayer &getObjectLinkingLayer() const { return ObjLinkingLayer; }

  Error setupJITDylib(JITDylib &JD) override;
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

  /// Returns true if the given section name is an initializer section.
  static bool isInitializerSection(StringRef SegName, StringRef SectName);

private:
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

    Error notifyRemovingResources(ResourceKey K) override {
      return Error::success();
    }

    void notifyTransferringResources(ResourceKey DstKey,
                                     ResourceKey SrcKey) override {}

  private:
    using InitSymbolDepMap =
        DenseMap<MaterializationResponsibility *, JITLinkSymbolSet>;

    void addInitializerSupportPasses(MaterializationResponsibility &MR,
                                     jitlink::PassConfiguration &Config);

    void addMachOHeaderSupportPasses(MaterializationResponsibility &MR,
                                     jitlink::PassConfiguration &Config);

    void addEHAndTLVSupportPasses(MaterializationResponsibility &MR,
                                  jitlink::PassConfiguration &Config);

    Error preserveInitSections(jitlink::LinkGraph &G,
                               MaterializationResponsibility &MR);

    Error processObjCImageInfo(jitlink::LinkGraph &G,
                               MaterializationResponsibility &MR);

    Error registerInitSections(jitlink::LinkGraph &G, JITDylib &JD);

    Error fixTLVSectionsAndEdges(jitlink::LinkGraph &G, JITDylib &JD);

    std::mutex PluginMutex;
    MachOPlatform &MP;
    DenseMap<JITDylib *, std::pair<uint32_t, uint32_t>> ObjCImageInfos;
    InitSymbolDepMap InitSymbolDeps;
  };

  using SendInitializerSequenceFn =
      unique_function<void(Expected<MachOJITDylibInitializerSequence>)>;

  using SendDeinitializerSequenceFn =
      unique_function<void(Expected<MachOJITDylibDeinitializerSequence>)>;

  using SendSymbolAddressFn = unique_function<void(Expected<ExecutorAddress>)>;

  static bool supportedTarget(const Triple &TT);

  MachOPlatform(ExecutionSession &ES, ObjectLinkingLayer &ObjLinkingLayer,
                JITDylib &PlatformJD,
                std::unique_ptr<DefinitionGenerator> OrcRuntimeGenerator,
                Error &Err);

  // Associate MachOPlatform JIT-side runtime support functions with handlers.
  Error associateRuntimeSupportFunctions(JITDylib &PlatformJD);

  void getInitializersBuildSequencePhase(SendInitializerSequenceFn SendResult,
                                         JITDylib &JD,
                                         std::vector<JITDylibSP> DFSLinkOrder);

  void getInitializersLookupPhase(SendInitializerSequenceFn SendResult,
                                  JITDylib &JD);

  void rt_getInitializers(SendInitializerSequenceFn SendResult,
                          StringRef JDName);

  void rt_getDeinitializers(SendDeinitializerSequenceFn SendResult,
                            ExecutorAddress Handle);

  void rt_lookupSymbol(SendSymbolAddressFn SendResult, ExecutorAddress Handle,
                       StringRef SymbolName);

  // Records the addresses of runtime symbols used by the platform.
  Error bootstrapMachORuntime(JITDylib &PlatformJD);

  Error registerInitInfo(JITDylib &JD, ExecutorAddress ObjCImageInfoAddr,
                         ArrayRef<jitlink::Section *> InitSections);

  Error registerPerObjectSections(const MachOPerObjectSectionsToRegister &POSR);

  Expected<uint64_t> createPThreadKey();

  ExecutionSession &ES;
  ObjectLinkingLayer &ObjLinkingLayer;

  SymbolStringPtr MachOHeaderStartSymbol;
  std::atomic<bool> RuntimeBootstrapped{false};

  ExecutorAddress orc_rt_macho_platform_bootstrap;
  ExecutorAddress orc_rt_macho_platform_shutdown;
  ExecutorAddress orc_rt_macho_register_object_sections;
  ExecutorAddress orc_rt_macho_create_pthread_key;

  DenseMap<JITDylib *, SymbolLookupSet> RegisteredInitSymbols;

  // InitSeqs gets its own mutex to avoid locking the whole session when
  // aggregating data from the jitlink.
  std::mutex PlatformMutex;
  DenseMap<JITDylib *, MachOJITDylibInitializers> InitSeqs;
  std::vector<MachOPerObjectSectionsToRegister> BootstrapPOSRs;

  DenseMap<JITTargetAddress, JITDylib *> HeaderAddrToJITDylib;
  DenseMap<JITDylib *, uint64_t> JITDylibToPThreadKey;
};

namespace shared {

using SPSMachOPerObjectSectionsToRegister =
    SPSTuple<SPSExecutorAddressRange, SPSExecutorAddressRange>;

template <>
class SPSSerializationTraits<SPSMachOPerObjectSectionsToRegister,
                             MachOPerObjectSectionsToRegister> {

public:
  static size_t size(const MachOPerObjectSectionsToRegister &MOPOSR) {
    return SPSMachOPerObjectSectionsToRegister::AsArgList::size(
        MOPOSR.EHFrameSection, MOPOSR.ThreadDataSection);
  }

  static bool serialize(SPSOutputBuffer &OB,
                        const MachOPerObjectSectionsToRegister &MOPOSR) {
    return SPSMachOPerObjectSectionsToRegister::AsArgList::serialize(
        OB, MOPOSR.EHFrameSection, MOPOSR.ThreadDataSection);
  }

  static bool deserialize(SPSInputBuffer &IB,
                          MachOPerObjectSectionsToRegister &MOPOSR) {
    return SPSMachOPerObjectSectionsToRegister::AsArgList::deserialize(
        IB, MOPOSR.EHFrameSection, MOPOSR.ThreadDataSection);
  }
};

using SPSNamedExecutorAddressRangeSequenceMap =
    SPSSequence<SPSTuple<SPSString, SPSExecutorAddressRangeSequence>>;

using SPSMachOJITDylibInitializers =
    SPSTuple<SPSString, SPSExecutorAddress, SPSExecutorAddress,
             SPSNamedExecutorAddressRangeSequenceMap>;

using SPSMachOJITDylibInitializerSequence =
    SPSSequence<SPSMachOJITDylibInitializers>;

/// Serialization traits for MachOJITDylibInitializers.
template <>
class SPSSerializationTraits<SPSMachOJITDylibInitializers,
                             MachOJITDylibInitializers> {
public:
  static size_t size(const MachOJITDylibInitializers &MOJDIs) {
    return SPSMachOJITDylibInitializers::AsArgList::size(
        MOJDIs.Name, MOJDIs.MachOHeaderAddress, MOJDIs.ObjCImageInfoAddress,
        MOJDIs.InitSections);
  }

  static bool serialize(SPSOutputBuffer &OB,
                        const MachOJITDylibInitializers &MOJDIs) {
    return SPSMachOJITDylibInitializers::AsArgList::serialize(
        OB, MOJDIs.Name, MOJDIs.MachOHeaderAddress, MOJDIs.ObjCImageInfoAddress,
        MOJDIs.InitSections);
  }

  static bool deserialize(SPSInputBuffer &IB,
                          MachOJITDylibInitializers &MOJDIs) {
    return SPSMachOJITDylibInitializers::AsArgList::deserialize(
        IB, MOJDIs.Name, MOJDIs.MachOHeaderAddress, MOJDIs.ObjCImageInfoAddress,
        MOJDIs.InitSections);
  }
};

using SPSMachOJITDylibDeinitializers = SPSEmpty;

using SPSMachOJITDylibDeinitializerSequence =
    SPSSequence<SPSMachOJITDylibDeinitializers>;

template <>
class SPSSerializationTraits<SPSMachOJITDylibDeinitializers,
                             MachOJITDylibDeinitializers> {
public:
  static size_t size(const MachOJITDylibDeinitializers &MOJDDs) { return 0; }

  static bool serialize(SPSOutputBuffer &OB,
                        const MachOJITDylibDeinitializers &MOJDDs) {
    return true;
  }

  static bool deserialize(SPSInputBuffer &IB,
                          MachOJITDylibDeinitializers &MOJDDs) {
    MOJDDs = MachOJITDylibDeinitializers();
    return true;
  }
};

} // end namespace shared
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_MACHOPLATFORM_H
