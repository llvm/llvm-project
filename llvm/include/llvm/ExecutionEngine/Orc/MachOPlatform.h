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
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"

#include <future>
#include <thread>
#include <vector>

namespace llvm {
namespace orc {


class MachOJITDylibInitializers {
public:
  using RawPointerSectionList = std::vector<ExecutorAddressRange>;

  void setObjCImageInfoAddr(JITTargetAddress ObjCImageInfoAddr) {
    this->ObjCImageInfoAddr = ObjCImageInfoAddr;
  }

  void addModInitsSection(ExecutorAddressRange ModInit) {
    ModInitSections.push_back(std::move(ModInit));
  }

  const RawPointerSectionList &getModInitsSections() const {
    return ModInitSections;
  }

  void addObjCSelRefsSection(ExecutorAddressRange ObjCSelRefs) {
    ObjCSelRefsSections.push_back(std::move(ObjCSelRefs));
  }

  const RawPointerSectionList &getObjCSelRefsSections() const {
    return ObjCSelRefsSections;
  }

  void addObjCClassListSection(ExecutorAddressRange ObjCClassList) {
    ObjCClassListSections.push_back(std::move(ObjCClassList));
  }

  const RawPointerSectionList &getObjCClassListSections() const {
    return ObjCClassListSections;
  }

  void runModInits() const;
  void registerObjCSelectors() const;
  Error registerObjCClasses() const;

private:

  JITTargetAddress ObjCImageInfoAddr;
  RawPointerSectionList ModInitSections;
  RawPointerSectionList ObjCSelRefsSections;
  RawPointerSectionList ObjCClassListSections;
};

class MachOJITDylibDeinitializers {};

/// Mediates between MachO initialization and ExecutionSession state.
class MachOPlatform : public Platform {
public:
  using InitializerSequence =
      std::vector<std::pair<JITDylib *, MachOJITDylibInitializers>>;

  using DeinitializerSequence =
      std::vector<std::pair<JITDylib *, MachOJITDylibDeinitializers>>;

  MachOPlatform(ExecutionSession &ES, ObjectLinkingLayer &ObjLinkingLayer,
                std::unique_ptr<MemoryBuffer> StandardSymbolsObject);

  ExecutionSession &getExecutionSession() const { return ES; }

  Error setupJITDylib(JITDylib &JD) override;
  Error notifyAdding(ResourceTracker &RT,
                     const MaterializationUnit &MU) override;
  Error notifyRemoving(ResourceTracker &RT) override;

  Expected<InitializerSequence> getInitializerSequence(JITDylib &JD);

  Expected<DeinitializerSequence> getDeinitializerSequence(JITDylib &JD);

private:
  // This ObjectLinkingLayer plugin scans JITLink graphs for __mod_init_func,
  // __objc_classlist and __sel_ref sections and records their extents so that
  // they can be run in the target process.
  class InitScraperPlugin : public ObjectLinkingLayer::Plugin {
  public:
    InitScraperPlugin(MachOPlatform &MP) : MP(MP) {}

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

    void preserveInitSectionIfPresent(JITLinkSymbolSet &Symbols,
                                      jitlink::LinkGraph &G,
                                      StringRef SectionName);

    Error processObjCImageInfo(jitlink::LinkGraph &G,
                               MaterializationResponsibility &MR);

    std::mutex InitScraperMutex;
    MachOPlatform &MP;
    DenseMap<JITDylib *, std::pair<uint32_t, uint32_t>> ObjCImageInfos;
    InitSymbolDepMap InitSymbolDeps;
  };

  void registerInitInfo(JITDylib &JD, JITTargetAddress ObjCImageInfoAddr,
                        ExecutorAddressRange ModInits,
                        ExecutorAddressRange ObjCSelRefs,
                        ExecutorAddressRange ObjCClassList);

  ExecutionSession &ES;
  ObjectLinkingLayer &ObjLinkingLayer;
  std::unique_ptr<MemoryBuffer> StandardSymbolsObject;

  DenseMap<JITDylib *, SymbolLookupSet> RegisteredInitSymbols;

  // InitSeqs gets its own mutex to avoid locking the whole session when
  // aggregating data from the jitlink.
  std::mutex InitSeqsMutex;
  DenseMap<JITDylib *, MachOJITDylibInitializers> InitSeqs;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_MACHOPLATFORM_H
