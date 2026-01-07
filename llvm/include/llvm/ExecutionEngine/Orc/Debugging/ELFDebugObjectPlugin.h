//===------ ELFDebugObjectPlugin.h - JITLink debug objects ------*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_ORC_ELFDEBUGOBJECTPLUGIN_H
#define LLVM_EXECUTIONENGINE_ORC_ELFDEBUGOBJECTPLUGIN_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/TargetParser/Triple.h"

#include <map>
#include <memory>
#include <mutex>

namespace llvm {
namespace orc {

class DebugObject;

/// Debugger support for ELF platforms with the GDB JIT Interface. The plugin
/// emits and manages a separate debug object allocation in addition to the
/// LinkGraph's own allocation and it notifies the debugger when necessary.
///
class LLVM_ABI ELFDebugObjectPlugin : public ObjectLinkingLayer::Plugin {
public:
  /// Create the plugin for the given session and set additional options
  ///
  /// RequireDebugSections:
  ///   Emit debug objects only if the LinkGraph contains debug info. Turning
  ///   this off allows minimal debugging based on raw symbol names, but it
  ///   comes with significant overhead for release configurations.
  ///
  /// AutoRegisterCode:
  ///   Notify the debugger for each new debug object. This is a good default
  ///   mode, but it may cause significant overhead when adding many modules in
  ///   sequence. Otherwise the user must call __jit_debug_register_code() in
  ///   the debug session manually.
  ///
  ELFDebugObjectPlugin(ExecutionSession &ES, bool RequireDebugSections,
                       bool AutoRegisterCode, Error &Err);
  ~ELFDebugObjectPlugin() override;

  void notifyMaterializing(MaterializationResponsibility &MR,
                           jitlink::LinkGraph &G, jitlink::JITLinkContext &Ctx,
                           MemoryBufferRef InputObj) override;

  Error notifyFailed(MaterializationResponsibility &MR) override;
  Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override;

  void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                   ResourceKey SrcKey) override;

  void modifyPassConfig(MaterializationResponsibility &MR,
                        jitlink::LinkGraph &LG,
                        jitlink::PassConfiguration &PassConfig) override;

private:
  ExecutionSession &ES;

  using OwnedDebugObject = std::unique_ptr<DebugObject>;
  std::map<MaterializationResponsibility *, OwnedDebugObject> PendingObjs;
  std::map<ResourceKey, std::vector<OwnedDebugObject>> RegisteredObjs;

  std::mutex PendingObjsLock;
  std::mutex RegisteredObjsLock;

  ExecutorAddr RegistrationAction;
  bool RequireDebugSections;
  bool AutoRegisterCode;

  DebugObject *getPendingDebugObj(MaterializationResponsibility &MR);
};

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_ELFDEBUGOBJECTPLUGIN_H
