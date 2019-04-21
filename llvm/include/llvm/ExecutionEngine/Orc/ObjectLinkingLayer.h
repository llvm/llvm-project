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

namespace object {
class ObjectFile;
} // namespace object

namespace orc {

class ObjectLinkingLayerJITLinkContext;

class ObjectLinkingLayer : public ObjectLayer {
  friend class ObjectLinkingLayerJITLinkContext;

public:
  /// Function object for receiving object-loaded notifications.
  using NotifyLoadedFunction = std::function<void(VModuleKey)>;

  /// Function object for receiving finalization notifications.
  using NotifyEmittedFunction = std::function<void(VModuleKey)>;

  /// Function object for modifying PassConfiguration objects.
  using ModifyPassConfigFunction =
      std::function<void(const Triple &TT, jitlink::PassConfiguration &Config)>;

  /// Construct an ObjectLinkingLayer with the given NotifyLoaded,
  ///        and NotifyEmitted functors.
  ObjectLinkingLayer(
      ExecutionSession &ES, jitlink::JITLinkMemoryManager &MemMgr,
      NotifyLoadedFunction NotifyLoaded = NotifyLoadedFunction(),
      NotifyEmittedFunction NotifyEmitted = NotifyEmittedFunction(),
      ModifyPassConfigFunction ModifyPassConfig = ModifyPassConfigFunction());

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

  class ObjectResources {
  public:
    ObjectResources() = default;
    ObjectResources(AllocPtr Alloc, JITTargetAddress EHFrameAddr);
    ObjectResources(ObjectResources &&Other);
    ObjectResources &operator=(ObjectResources &&Other);
    ~ObjectResources();

  private:
    AllocPtr Alloc;
    JITTargetAddress EHFrameAddr = 0;
  };

  void notifyFinalized(ObjectResources OR) {
    ObjResources.push_back(std::move(OR));
  }

  mutable std::mutex LayerMutex;
  jitlink::JITLinkMemoryManager &MemMgr;
  NotifyLoadedFunction NotifyLoaded;
  NotifyEmittedFunction NotifyEmitted;
  ModifyPassConfigFunction ModifyPassConfig;
  bool OverrideObjectFlags = false;
  bool AutoClaimObjectSymbols = false;
  std::vector<ObjectResources> ObjResources;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H
