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

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/LinkGraphLinkingLayer.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace orc {

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
class DebugObjectManagerPlugin : public LinkGraphLinkingLayer::Plugin {
public:
  // DEPRECATED - Please specify options explicitly
  DebugObjectManagerPlugin(ExecutionSession &ES);

  /// Create the plugin to submit DebugObjects for JITLink artifacts. For all
  /// options the recommended setting is true.
  ///
  /// RequireDebugSections:
  ///   Submit debug objects to the executor only if they contain actual debug
  ///   info. Turning this off may allow minimal debugging based on raw symbol
  ///   names. Note that this may cause significant memory and transport
  ///   overhead for objects built with a release configuration.
  ///
  /// AutoRegisterCode:
  ///   Notify the debugger for each new debug object. This is a good default
  ///   mode, but it may cause significant overhead when adding many modules in
  ///   sequence. When turning this off, the user has to issue the call to
  ///   __jit_debug_register_code() on the executor side manually.
  ///
  DebugObjectManagerPlugin(ExecutorSymbolDef RegisterDebugObject,
                           ExecutorSymbolDef DeregisterDebugObject,
                           bool RequireDebugSections, bool AutoRegisterCode);
  ~DebugObjectManagerPlugin();

  Error fixUpDebugObject(jitlink::LinkGraph &LG);

  Error notifyFailed(MaterializationResponsibility &MR) override {
    return Error::success();
  }

  Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override {
    return Error::success();
  }

  void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                   ResourceKey SrcKey) override {}

  void modifyPassConfig(MaterializationResponsibility &MR,
                        jitlink::LinkGraph &LG,
                        jitlink::PassConfiguration &PassConfig) override;

private:
  ExecutorSymbolDef RegisterDebugObject;
  ExecutorSymbolDef DeregisterDebugObject;
};

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_DEBUGOBJECTMANAGERPLUGIN_H