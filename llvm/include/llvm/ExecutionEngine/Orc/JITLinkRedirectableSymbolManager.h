//===- JITLinkRedirectableSymbolManager.h - JITLink redirection -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Redirectable Symbol Manager implementation using JITLink
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_JITLINKREDIRECABLEMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_JITLINKREDIRECABLEMANAGER_H

#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/RedirectionManager.h"
#include "llvm/Support/StringSaver.h"

namespace llvm {
namespace orc {

class JITLinkRedirectableSymbolManager : public RedirectableSymbolManager,
                                         public ResourceManager {
public:
  /// Create redirection manager that uses JITLink based implementaion.
  static Expected<std::unique_ptr<RedirectableSymbolManager>>
  Create(ObjectLinkingLayer &ObjLinkingLayer, JITDylib &JD) {
    auto AnonymousPtrCreator(jitlink::getAnonymousPointerCreator(
        ObjLinkingLayer.getExecutionSession().getTargetTriple()));
    auto PtrJumpStubCreator(jitlink::getPointerJumpStubCreator(
        ObjLinkingLayer.getExecutionSession().getTargetTriple()));
    if (!AnonymousPtrCreator || !PtrJumpStubCreator)
      return make_error<StringError>("Architecture not supported",
                                     inconvertibleErrorCode());
    return std::unique_ptr<RedirectableSymbolManager>(
        new JITLinkRedirectableSymbolManager(
            ObjLinkingLayer, JD, AnonymousPtrCreator, PtrJumpStubCreator));
  }

  void emitRedirectableSymbols(std::unique_ptr<MaterializationResponsibility> R,
                               const SymbolAddrMap &InitialDests) override;

  Error redirect(JITDylib &TargetJD, const SymbolAddrMap &NewDests) override;

  Error handleRemoveResources(JITDylib &TargetJD, ResourceKey K) override;

  void handleTransferResources(JITDylib &TargetJD, ResourceKey DstK,
                               ResourceKey SrcK) override;

private:
  using StubHandle = unsigned;
  constexpr static unsigned StubBlockSize = 256;
  constexpr static StringRef JumpStubPrefix = "$__IND_JUMP_STUBS";
  constexpr static StringRef StubPtrPrefix = "$IND_JUMP_PTR_";
  constexpr static StringRef JumpStubTableName = "$IND_JUMP_";
  constexpr static StringRef StubPtrTableName = "$__IND_JUMP_PTRS";

  JITLinkRedirectableSymbolManager(
      ObjectLinkingLayer &ObjLinkingLayer, JITDylib &JD,
      jitlink::AnonymousPointerCreator &AnonymousPtrCreator,
      jitlink::PointerJumpStubCreator &PtrJumpStubCreator)
      : ObjLinkingLayer(ObjLinkingLayer), JD(JD),
        AnonymousPtrCreator(std::move(AnonymousPtrCreator)),
        PtrJumpStubCreator(std::move(PtrJumpStubCreator)) {
    ObjLinkingLayer.getExecutionSession().registerResourceManager(*this);
  }

  ~JITLinkRedirectableSymbolManager() {
    ObjLinkingLayer.getExecutionSession().deregisterResourceManager(*this);
  }

  StringRef JumpStubSymbolName(unsigned I) {
    return *ObjLinkingLayer.getExecutionSession().intern(
        (JumpStubPrefix + Twine(I)).str());
  }

  StringRef StubPtrSymbolName(unsigned I) {
    return *ObjLinkingLayer.getExecutionSession().intern(
        (StubPtrPrefix + Twine(I)).str());
  }

  unsigned GetNumAvailableStubs() const { return AvailableStubs.size(); }

  Error redirectInner(JITDylib &TargetJD, const SymbolAddrMap &NewDests);
  Error grow(unsigned Need);

  ObjectLinkingLayer &ObjLinkingLayer;
  JITDylib &JD;
  jitlink::AnonymousPointerCreator AnonymousPtrCreator;
  jitlink::PointerJumpStubCreator PtrJumpStubCreator;

  std::vector<StubHandle> AvailableStubs;
  using SymbolToStubMap = DenseMap<SymbolStringPtr, StubHandle>;
  DenseMap<JITDylib *, SymbolToStubMap> SymbolToStubs;
  std::vector<ExecutorSymbolDef> JumpStubs;
  std::vector<ExecutorSymbolDef> StubPointers;
  DenseMap<ResourceKey, std::vector<SymbolStringPtr>> TrackedResources;

  std::mutex Mutex;
};

} // namespace orc
} // namespace llvm

#endif
