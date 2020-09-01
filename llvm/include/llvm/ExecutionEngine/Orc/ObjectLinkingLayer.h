//===-- ObjectLinkingLayer.h - JITLink-based jit linking layer --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition for an JITLink-based, in-process object linking
// layer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llvm {

namespace jitlink {
class EHFrameRegistrar;
class Symbol;
} // namespace jitlink

namespace object {
class ObjectFile;
} // namespace object

namespace orc {

class ObjectLinkingLayerJITLinkContext;

/// An ObjectLayer implementation built on JITLink.
///
/// Clients can use this class to add relocatable object files to an
/// ExecutionSession, and it typically serves as the base layer (underneath
/// a compiling layer like IRCompileLayer) for the rest of the JIT.
class ObjectLinkingLayer : public ObjectLayer {
  friend class ObjectLinkingLayerJITLinkContext;

public:
  /// Plugin instances can be added to the ObjectLinkingLayer to receive
  /// callbacks when code is loaded or emitted, and when JITLink is being
  /// configured.
  class Plugin {
  public:
    using JITLinkSymbolVector = std::vector<const jitlink::Symbol *>;
    using LocalDependenciesMap = DenseMap<SymbolStringPtr, JITLinkSymbolVector>;

    virtual ~Plugin();
    virtual void modifyPassConfig(MaterializationResponsibility &MR,
                                  const Triple &TT,
                                  jitlink::PassConfiguration &Config) {}

    virtual void notifyLoaded(MaterializationResponsibility &MR) {}
    virtual Error notifyEmitted(MaterializationResponsibility &MR) {
      return Error::success();
    }
    virtual Error notifyRemovingModule(VModuleKey K) {
      return Error::success();
    }
    virtual Error notifyRemovingAllModules() { return Error::success(); }

    /// Return any dependencies that synthetic symbols (e.g. init symbols)
    /// have on locally scoped jitlink::Symbols. This is used by the
    /// ObjectLinkingLayer to update the dependencies for the synthetic
    /// symbols.
    virtual LocalDependenciesMap
    getSyntheticSymbolLocalDependencies(MaterializationResponsibility &MR) {
      return LocalDependenciesMap();
    }
  };

  using ReturnObjectBufferFunction =
      std::function<void(std::unique_ptr<MemoryBuffer>)>;

  /// Construct an ObjectLinkingLayer.
  ObjectLinkingLayer(ExecutionSession &ES,
                     jitlink::JITLinkMemoryManager &MemMgr);

  /// Construct an ObjectLinkingLayer. Takes ownership of the given
  /// JITLinkMemoryManager. This method is a temporary hack to simplify
  /// co-existence with RTDyldObjectLinkingLayer (which also owns its
  /// allocators).
  ObjectLinkingLayer(ExecutionSession &ES,
                     std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr);

  /// Destruct an ObjectLinkingLayer.
  ~ObjectLinkingLayer();

  /// Set an object buffer return function. By default object buffers are
  /// deleted once the JIT has linked them. If a return function is set then
  /// it will be called to transfer ownership of the buffer instead.
  void setReturnObjectBuffer(ReturnObjectBufferFunction ReturnObjectBuffer) {
    this->ReturnObjectBuffer = std::move(ReturnObjectBuffer);
  }

  /// Add a pass-config modifier.
  ObjectLinkingLayer &addPlugin(std::unique_ptr<Plugin> P) {
    std::lock_guard<std::mutex> Lock(LayerMutex);
    Plugins.push_back(std::move(P));
    return *this;
  }

  /// Emit the object.
  void emit(MaterializationResponsibility R,
            std::unique_ptr<MemoryBuffer> O) override;

  /// Instructs this ObjectLinkingLayer instance to override the symbol flags
  /// found in the AtomGraph with the flags supplied by the
  /// MaterializationResponsibility instance. This is a workaround to support
  /// symbol visibility in COFF, which does not use the libObject's
  /// SF_Exported flag. Use only when generating / adding COFF object files.
  ///
  /// FIXME: We should be able to remove this if/when COFF properly tracks
  /// exported symbols.
  ObjectLinkingLayer &
  setOverrideObjectFlagsWithResponsibilityFlags(bool OverrideObjectFlags) {
    this->OverrideObjectFlags = OverrideObjectFlags;
    return *this;
  }

  /// If set, this ObjectLinkingLayer instance will claim responsibility
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
  ObjectLinkingLayer &
  setAutoClaimResponsibilityForObjectSymbols(bool AutoClaimObjectSymbols) {
    this->AutoClaimObjectSymbols = AutoClaimObjectSymbols;
    return *this;
  }

private:
  using AllocPtr = std::unique_ptr<jitlink::JITLinkMemoryManager::Allocation>;

  void modifyPassConfig(MaterializationResponsibility &MR, const Triple &TT,
                        jitlink::PassConfiguration &PassConfig);
  void notifyLoaded(MaterializationResponsibility &MR);
  Error notifyEmitted(MaterializationResponsibility &MR, AllocPtr Alloc);

  Error removeModule(VModuleKey K);
  Error removeAllModules();

  mutable std::mutex LayerMutex;
  jitlink::JITLinkMemoryManager &MemMgr;
  std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgrOwnership;
  bool OverrideObjectFlags = false;
  bool AutoClaimObjectSymbols = false;
  ReturnObjectBufferFunction ReturnObjectBuffer;
  DenseMap<VModuleKey, AllocPtr> TrackedAllocs;
  std::vector<AllocPtr> UntrackedAllocs;
  std::vector<std::unique_ptr<Plugin>> Plugins;
};

class EHFrameRegistrationPlugin : public ObjectLinkingLayer::Plugin {
public:
  EHFrameRegistrationPlugin(jitlink::EHFrameRegistrar &Registrar);
  Error notifyEmitted(MaterializationResponsibility &MR) override;
  void modifyPassConfig(MaterializationResponsibility &MR, const Triple &TT,
                        jitlink::PassConfiguration &PassConfig) override;
  Error notifyRemovingModule(VModuleKey K) override;
  Error notifyRemovingAllModules() override;

private:

  struct EHFrameRange {
    JITTargetAddress Addr = 0;
    size_t Size;
  };

  std::mutex EHFramePluginMutex;
  jitlink::EHFrameRegistrar &Registrar;
  DenseMap<MaterializationResponsibility *, EHFrameRange> InProcessLinks;
  DenseMap<VModuleKey, EHFrameRange> TrackedEHFrameRanges;
  std::vector<EHFrameRange> UntrackedEHFrameRanges;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H
