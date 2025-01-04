//===-- LinkGraphLinkingLayer.h - Link LinkGraphs with JITLink --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LinkGraphLinkingLayer and associated utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LINKGRAPHLINKINGLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_LINKGRAPHLINKINGLAYER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/LinkGraphLayer.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace llvm {

namespace jitlink {
class EHFrameRegistrar;
} // namespace jitlink

namespace orc {

/// LinkGraphLinkingLayer links LinkGraphs into the Executor using JITLink.
///
/// Clients can use this class to add LinkGraphs to an ExecutionSession, and it
/// serves as a base for the ObjectLinkingLayer that can link object files.
class LinkGraphLinkingLayer : public LinkGraphLayer, private ResourceManager {
  class JITLinkCtx;

public:
  /// Plugin instances can be added to the ObjectLinkingLayer to receive
  /// callbacks when code is loaded or emitted, and when JITLink is being
  /// configured.
  class Plugin {
  public:
    virtual ~Plugin();
    virtual void modifyPassConfig(MaterializationResponsibility &MR,
                                  jitlink::LinkGraph &G,
                                  jitlink::PassConfiguration &Config) {}

    // Deprecated. Don't use this in new code. There will be a proper mechanism
    // for capturing object buffers.
    virtual void notifyMaterializing(MaterializationResponsibility &MR,
                                     jitlink::LinkGraph &G,
                                     jitlink::JITLinkContext &Ctx,
                                     MemoryBufferRef InputObject) {}

    virtual void notifyLoaded(MaterializationResponsibility &MR) {}
    virtual Error notifyEmitted(MaterializationResponsibility &MR) {
      return Error::success();
    }
    virtual Error notifyFailed(MaterializationResponsibility &MR) = 0;
    virtual Error notifyRemovingResources(JITDylib &JD, ResourceKey K) = 0;
    virtual void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                             ResourceKey SrcKey) = 0;
  };

  /// Construct a LinkGraphLinkingLayer using the ExecutorProcessControl
  /// instance's memory manager.
  LinkGraphLinkingLayer(ExecutionSession &ES);

  /// Construct a LinkGraphLinkingLayer using a custom memory manager.
  LinkGraphLinkingLayer(ExecutionSession &ES,
                        jitlink::JITLinkMemoryManager &MemMgr);

  /// Construct an LinkGraphLinkingLayer. Takes ownership of the given
  /// JITLinkMemoryManager. This method is a temporary hack to simplify
  /// co-existence with RTDyldObjectLinkingLayer (which also owns its
  /// allocators).
  LinkGraphLinkingLayer(ExecutionSession &ES,
                        std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr);

  /// Destroy the LinkGraphLinkingLayer.
  ~LinkGraphLinkingLayer();

  /// Add a plugin.
  LinkGraphLinkingLayer &addPlugin(std::shared_ptr<Plugin> P) {
    std::lock_guard<std::mutex> Lock(LayerMutex);
    Plugins.push_back(std::move(P));
    return *this;
  }

  /// Remove a plugin. This remove applies only to subsequent links (links
  /// already underway will continue to use the plugin), and does not of itself
  /// destroy the plugin -- destruction will happen once all shared pointers
  /// (including those held by in-progress links) are destroyed.
  void removePlugin(Plugin &P) {
    std::lock_guard<std::mutex> Lock(LayerMutex);
    auto I = llvm::find_if(Plugins, [&](const std::shared_ptr<Plugin> &Elem) {
      return Elem.get() == &P;
    });
    assert(I != Plugins.end() && "Plugin not present");
    Plugins.erase(I);
  }

  /// Emit a LinkGraph.
  void emit(std::unique_ptr<MaterializationResponsibility> R,
            std::unique_ptr<jitlink::LinkGraph> G) override;

  /// Instructs this LinkgraphLinkingLayer instance to override the symbol flags
  /// found in the LinkGraph with the flags supplied by the
  /// MaterializationResponsibility instance. This is a workaround to support
  /// symbol visibility in COFF, which does not use the libObject's
  /// SF_Exported flag. Use only when generating / adding COFF object files.
  ///
  /// FIXME: We should be able to remove this if/when COFF properly tracks
  /// exported symbols.
  LinkGraphLinkingLayer &
  setOverrideObjectFlagsWithResponsibilityFlags(bool OverrideObjectFlags) {
    this->OverrideObjectFlags = OverrideObjectFlags;
    return *this;
  }

  /// If set, this LinkGraphLinkingLayer instance will claim responsibility
  /// for any symbols provided by a given object file that were not already in
  /// the MaterializationResponsibility instance. Setting this flag allows
  /// higher-level program representations (e.g. LLVM IR) to be added based on
  /// only a subset of the symbols they provide, without having to write
  /// intervening layers to scan and add the additional symbols. This trades
  /// diagnostic quality for convenience however: If all symbols are enumerated
  /// up-front then clashes can be detected and reported early (and usually
  /// deterministically). If this option is set, clashes for the additional
  /// symbols may not be detected until late, and detection may depend on
  /// the flow of control through JIT'd code. Use with care.
  LinkGraphLinkingLayer &
  setAutoClaimResponsibilityForObjectSymbols(bool AutoClaimObjectSymbols) {
    this->AutoClaimObjectSymbols = AutoClaimObjectSymbols;
    return *this;
  }

protected:
  /// Emit a LinkGraph with the given backing buffer.
  ///
  /// This overload is intended for use by ObjectLinkingLayer.
  void emit(std::unique_ptr<MaterializationResponsibility> R,
            std::unique_ptr<jitlink::LinkGraph> G,
            std::unique_ptr<MemoryBuffer> ObjBuf);

  std::function<void(std::unique_ptr<MemoryBuffer>)> ReturnObjectBuffer;

private:
  using FinalizedAlloc = jitlink::JITLinkMemoryManager::FinalizedAlloc;

  Error recordFinalizedAlloc(MaterializationResponsibility &MR,
                             FinalizedAlloc FA);

  Error handleRemoveResources(JITDylib &JD, ResourceKey K) override;
  void handleTransferResources(JITDylib &JD, ResourceKey DstKey,
                               ResourceKey SrcKey) override;

  mutable std::mutex LayerMutex;
  jitlink::JITLinkMemoryManager &MemMgr;
  std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgrOwnership;
  bool OverrideObjectFlags = false;
  bool AutoClaimObjectSymbols = false;
  DenseMap<ResourceKey, std::vector<FinalizedAlloc>> Allocs;
  std::vector<std::shared_ptr<Plugin>> Plugins;
};

class EHFrameRegistrationPlugin : public LinkGraphLinkingLayer::Plugin {
public:
  EHFrameRegistrationPlugin(
      ExecutionSession &ES,
      std::unique_ptr<jitlink::EHFrameRegistrar> Registrar);
  void modifyPassConfig(MaterializationResponsibility &MR,
                        jitlink::LinkGraph &G,
                        jitlink::PassConfiguration &PassConfig) override;
  Error notifyEmitted(MaterializationResponsibility &MR) override;
  Error notifyFailed(MaterializationResponsibility &MR) override;
  Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override;
  void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                   ResourceKey SrcKey) override;

private:
  std::mutex EHFramePluginMutex;
  ExecutionSession &ES;
  std::unique_ptr<jitlink::EHFrameRegistrar> Registrar;
  DenseMap<MaterializationResponsibility *, ExecutorAddrRange> InProcessLinks;
  DenseMap<ResourceKey, std::vector<ExecutorAddrRange>> EHFrameRanges;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_LINKGRAPHLINKINGLAYER_H
