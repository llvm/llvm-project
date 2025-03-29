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

#ifndef LLVM_EXECUTIONENGINE_ORC_JITLINKREDIRECABLESYMBOLMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_JITLINKREDIRECABLESYMBOLMANAGER_H

#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/RedirectionManager.h"
#include "llvm/Support/StringSaver.h"

#include <atomic>

namespace llvm {
namespace orc {

class JITLinkRedirectableSymbolManager : public RedirectableSymbolManager {
public:
  /// Create redirection manager that uses JITLink based implementaion.
  static Expected<std::unique_ptr<RedirectableSymbolManager>>
  Create(ObjectLinkingLayer &ObjLinkingLayer) {
    auto AnonymousPtrCreator(jitlink::getAnonymousPointerCreator(
        ObjLinkingLayer.getExecutionSession().getTargetTriple()));
    auto PtrJumpStubCreator(jitlink::getPointerJumpStubCreator(
        ObjLinkingLayer.getExecutionSession().getTargetTriple()));
    if (!AnonymousPtrCreator || !PtrJumpStubCreator)
      return make_error<StringError>("Architecture not supported",
                                     inconvertibleErrorCode());
    return std::unique_ptr<RedirectableSymbolManager>(
        new JITLinkRedirectableSymbolManager(
            ObjLinkingLayer, AnonymousPtrCreator, PtrJumpStubCreator));
  }

  JITLinkRedirectableSymbolManager(
      ObjectLinkingLayer &ObjLinkingLayer,
      jitlink::AnonymousPointerCreator &AnonymousPtrCreator,
      jitlink::PointerJumpStubCreator &PtrJumpStubCreator)
      : ObjLinkingLayer(ObjLinkingLayer),
        AnonymousPtrCreator(std::move(AnonymousPtrCreator)),
        PtrJumpStubCreator(std::move(PtrJumpStubCreator)) {}

  ObjectLinkingLayer &getObjectLinkingLayer() const { return ObjLinkingLayer; }

  void emitRedirectableSymbols(std::unique_ptr<MaterializationResponsibility> R,
                               SymbolMap InitialDests) override;

  Error redirect(JITDylib &JD, const SymbolMap &NewDests) override;

private:
  ObjectLinkingLayer &ObjLinkingLayer;
  jitlink::AnonymousPointerCreator AnonymousPtrCreator;
  jitlink::PointerJumpStubCreator PtrJumpStubCreator;
  std::atomic_size_t StubGraphIdx{0};
};

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_JITLINKREDIRECABLESYMBOLMANAGER_H
