//===- UnwindInfoRegistrationPlugin.h -- libunwind registration -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Register eh-frame and compact-unwind sections with libunwind
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_UNWINDINFOREGISTRATIONPLUGIN_H
#define LLVM_EXECUTIONENGINE_ORC_UNWINDINFOREGISTRATIONPLUGIN_H

#include "llvm/ExecutionEngine/Orc/LinkGraphLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/Support/Compiler.h"

namespace llvm::orc {

class LLVM_ABI UnwindInfoRegistrationPlugin
    : public LinkGraphLinkingLayer::Plugin {
public:
  UnwindInfoRegistrationPlugin(ExecutionSession &ES,
                               ExecutorAddr RegisterSections,
                               ExecutorAddr DeregisterSections)
      : RegisterSections(RegisterSections),
        DeregisterSections(DeregisterSections) {
    DSOBaseName = ES.intern("__jitlink$libunwind_dso_base");
  }

  static Expected<std::shared_ptr<UnwindInfoRegistrationPlugin>>
  Create(ExecutionSession &ES,
         rt::MachOUnwindInfoRegistrarSymbolNames SNs =
             rt::orc_rt_MachOUnwindInfoRegistrarSPSSymbols);

  void modifyPassConfig(MaterializationResponsibility &MR,
                        jitlink::LinkGraph &G,
                        jitlink::PassConfiguration &PassConfig) override;

  Error notifyEmitted(MaterializationResponsibility &MR) override {
    return Error::success();
  }

  Error notifyFailed(MaterializationResponsibility &MR) override {
    return Error::success();
  }

  Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override {
    return Error::success();
  }

  void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                   ResourceKey SrcKey) override {}

private:
  Error addUnwindInfoRegistrationActions(jitlink::LinkGraph &G);

  SymbolStringPtr DSOBaseName;
  ExecutorAddr RegisterSections, DeregisterSections;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_UNWINDINFOREGISTRATIONPLUGIN_H
