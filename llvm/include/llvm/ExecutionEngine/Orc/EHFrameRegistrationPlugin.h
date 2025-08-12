//===----- EHFrameRegistrationPlugin.h - Register eh-frames -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Register eh-frame sections with a registrar.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EHFRAMEREGISTRATIONPLUGIN_H
#define LLVM_EXECUTIONENGINE_ORC_EHFRAMEREGISTRATIONPLUGIN_H

#include "llvm/ExecutionEngine/Orc/LinkGraphLinkingLayer.h"
#include "llvm/Support/Compiler.h"

#include <memory>
#include <mutex>
#include <vector>

namespace llvm::orc {

/// Adds AllocationActions to register and deregister eh-frame sections in the
/// absence of native Platform support.
class LLVM_ABI EHFrameRegistrationPlugin
    : public LinkGraphLinkingLayer::Plugin {
public:
  static Expected<std::unique_ptr<EHFrameRegistrationPlugin>>
  Create(ExecutionSession &ES);

  EHFrameRegistrationPlugin(ExecutorAddr RegisterEHFrame,
                            ExecutorAddr DeregisterEHFrame)
      : RegisterEHFrame(RegisterEHFrame), DeregisterEHFrame(DeregisterEHFrame) {
  }

  void modifyPassConfig(MaterializationResponsibility &MR,
                        jitlink::LinkGraph &G,
                        jitlink::PassConfiguration &PassConfig) override;
  Error notifyFailed(MaterializationResponsibility &MR) override {
    return Error::success();
  }
  Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override {
    return Error::success();
  }
  void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                   ResourceKey SrcKey) override {}

private:
  ExecutorAddr RegisterEHFrame;
  ExecutorAddr DeregisterEHFrame;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_EHFRAMEREGISTRATIONPLUGIN_H
