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

namespace llvm::orc {

class UnwindInfoRegistrationPlugin : public LinkGraphLinkingLayer::Plugin {
public:
  UnwindInfoRegistrationPlugin(ExecutionSession &ES, ExecutorAddr Register,
                               ExecutorAddr Deregister)
      : ES(ES), Register(Register), Deregister(Deregister) {
    DSOBaseName = ES.intern("__jitlink$libunwind_dso_base");
  }

  static Expected<std::shared_ptr<UnwindInfoRegistrationPlugin>>
  Create(ExecutionSession &ES, ExecutorAddr Register, ExecutorAddr Deregister);

  static Expected<std::shared_ptr<UnwindInfoRegistrationPlugin>>
  Create(ExecutionSession &ES);

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

  ExecutionSession &ES;
  SymbolStringPtr DSOBaseName;
  ExecutorAddr Register, Deregister;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_UNWINDINFOREGISTRATIONPLUGIN_H
