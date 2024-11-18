//===- RedirectionManager.h - Redirection manager interface -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Redirection manager interface that redirects a call to symbol to another.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_EXECUTIONENGINE_ORC_LAZYOBJECTLINKINGLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_LAZYOBJECTLINKINGLAYER_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"

namespace llvm::orc {

class ObjectLinkingLayer;
class LazyCallThroughManager;
class RedirectableSymbolManager;

class LazyObjectLinkingLayer : public ObjectLayer {
public:
  LazyObjectLinkingLayer(ObjectLinkingLayer &BaseLayer,
                         LazyCallThroughManager &LCTMgr,
                         RedirectableSymbolManager &RSMgr);

  llvm::Error add(llvm::orc::ResourceTrackerSP RT,
                  std::unique_ptr<llvm::MemoryBuffer> O,
                  llvm::orc::MaterializationUnit::Interface I) override;

  void emit(std::unique_ptr<MaterializationResponsibility> R,
            std::unique_ptr<MemoryBuffer> O) override;

private:
  class RenamerPlugin;

  ObjectLinkingLayer &BaseLayer;
  LazyCallThroughManager &LCTMgr;
  RedirectableSymbolManager &RSMgr;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_LAZYOBJECTLINKINGLAYER_H
