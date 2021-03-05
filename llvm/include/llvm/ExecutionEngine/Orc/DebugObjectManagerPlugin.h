//===---- DebugObjectManagerPlugin.h - JITLink debug objects ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ObjectLinkingLayer plugin for emitting debug objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_DEBUGOBJECTMANAGERPLUGIN_H
#define LLVM_EXECUTIONENGINE_ORC_DEBUGOBJECTMANAGERPLUGIN_H

#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TPCDebugObjectRegistrar.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"

#include <functional>
#include <map>
#include <memory>
#include <mutex>

namespace llvm {
namespace orc {

class DebugObject;

/// Creates and manages DebugObjects for JITLink artifacts.
///
/// DebugObjects are created when linking for a MaterializationResponsibility
/// starts. They are pending as long as materialization is in progress.
///
/// There can only be one pending DebugObject per MaterializationResponsibility.
/// If materialization fails, pending DebugObjects are discarded.
///
/// Once executable code for the MaterializationResponsibility is emitted, the
/// corresponding DebugObject is finalized to target memory and the provided
/// DebugObjectRegistrar is notified. Ownership of DebugObjects remains with the
/// plugin.
///
class DebugObjectManagerPlugin : public ObjectLinkingLayer::Plugin {
public:
  DebugObjectManagerPlugin(ExecutionSession &ES,
                           std::unique_ptr<DebugObjectRegistrar> Target);
  ~DebugObjectManagerPlugin();

  void notifyMaterializing(MaterializationResponsibility &MR,
                           jitlink::LinkGraph &G, jitlink::JITLinkContext &Ctx,
                           MemoryBufferRef InputObject) override;

  Error notifyEmitted(MaterializationResponsibility &MR) override;
  Error notifyFailed(MaterializationResponsibility &MR) override;
  Error notifyRemovingResources(ResourceKey K) override;

  void notifyTransferringResources(ResourceKey DstKey,
                                   ResourceKey SrcKey) override;

  void modifyPassConfig(MaterializationResponsibility &MR, const Triple &TT,
                        jitlink::PassConfiguration &PassConfig) override;

private:
  ExecutionSession &ES;

  using OwnedDebugObject = std::unique_ptr<DebugObject>;
  std::map<ResourceKey, OwnedDebugObject> PendingObjs;
  std::map<ResourceKey, std::vector<OwnedDebugObject>> RegisteredObjs;

  std::mutex PendingObjsLock;
  std::mutex RegisteredObjsLock;

  std::unique_ptr<DebugObjectRegistrar> Target;
};

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_DEBUGOBJECTMANAGERPLUGIN_H
