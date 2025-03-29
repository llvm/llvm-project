//===- JITLinkReentryTrampolines.h -- JITLink-based trampolines -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit reentry trampolines via JITLink.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_JITLINKREENTRYTRAMPOLINES_H
#define LLVM_EXECUTIONENGINE_ORC_JITLINKREENTRYTRAMPOLINES_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/LazyReexports.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/Support/Error.h"

namespace llvm::jitlink {
class Block;
class LinkGraph;
class Section;
class Symbol;
} // namespace llvm::jitlink

namespace llvm::orc {

class ObjectLinkingLayer;
class RedirectableSymbolManager;

/// Produces trampolines on request using JITLink.
class JITLinkReentryTrampolines {
public:
  using EmitTrampolineFn = unique_function<jitlink::Symbol &(
      jitlink::LinkGraph &G, jitlink::Section &Sec,
      jitlink::Symbol &ReentrySym)>;
  using OnTrampolinesReadyFn = unique_function<void(
      Expected<std::vector<ExecutorSymbolDef>> EntryAddrs)>;

  /// Create trampolines using the default reentry trampoline function for
  /// the session triple.
  static Expected<std::unique_ptr<JITLinkReentryTrampolines>>
  Create(ObjectLinkingLayer &ObjLinkingLayer);

  JITLinkReentryTrampolines(ObjectLinkingLayer &ObjLinkingLayer,
                            EmitTrampolineFn EmitTrampoline);
  JITLinkReentryTrampolines(JITLinkReentryTrampolines &&) = delete;
  JITLinkReentryTrampolines &operator=(JITLinkReentryTrampolines &&) = delete;

  void emit(ResourceTrackerSP RT, size_t NumTrampolines,
            OnTrampolinesReadyFn OnTrampolinesReady);

private:
  class TrampolineAddrScraperPlugin;

  ObjectLinkingLayer &ObjLinkingLayer;
  TrampolineAddrScraperPlugin *TrampolineAddrScraper = nullptr;
  EmitTrampolineFn EmitTrampoline;
  std::atomic<size_t> ReentryGraphIdx{0};
};

Expected<std::unique_ptr<LazyReexportsManager>>
createJITLinkLazyReexportsManager(ObjectLinkingLayer &ObjLinkingLayer,
                                  RedirectableSymbolManager &RSMgr,
                                  JITDylib &PlatformJD,
                                  LazyReexportsManager::Listener *L = nullptr);

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_JITLINKREENTRYTRAMPOLINES_H
