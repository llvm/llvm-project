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

#include <memory>
#include <mutex>
#include <vector>

namespace llvm {

namespace jitlink {
class EHFrameRegistrar;
} // namespace jitlink

namespace orc {

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

#endif // LLVM_EXECUTIONENGINE_ORC_EHFRAMEREGISTRATIONPLUGIN_H
