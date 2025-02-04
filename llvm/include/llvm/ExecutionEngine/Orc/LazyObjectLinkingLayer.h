//===- LazyObjectLinkingLayer.h - Link objects on first fn call -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Link object files lazily on first call.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_EXECUTIONENGINE_ORC_LAZYOBJECTLINKINGLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_LAZYOBJECTLINKINGLAYER_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"

namespace llvm::orc {

class ObjectLinkingLayer;
class LazyReexportsManager;
class RedirectableSymbolManager;

/// LazyObjectLinkingLayer is an adapter for ObjectLinkingLayer that builds
/// lazy reexports for all function symbols in objects that are/ added to defer
/// linking until the first call to a function defined in the object.
///
/// Linking is performed by emitting the object file via the base
/// ObjectLinkingLayer.
///
/// No partitioning is performed: The first call to any function in the object
/// will trigger linking of the whole object.
///
/// References to data symbols are not lazy and will trigger immediate linking
/// (same os ObjectlinkingLayer).
class LazyObjectLinkingLayer : public ObjectLayer {
public:
  LazyObjectLinkingLayer(ObjectLinkingLayer &BaseLayer,
                         LazyReexportsManager &LRMgr);

  /// Add an object file to the JITDylib targeted by the given tracker.
  llvm::Error add(llvm::orc::ResourceTrackerSP RT,
                  std::unique_ptr<MemoryBuffer> O,
                  MaterializationUnit::Interface I) override;

  void emit(std::unique_ptr<MaterializationResponsibility> R,
            std::unique_ptr<MemoryBuffer> O) override;

private:
  class RenamerPlugin;

  ObjectLinkingLayer &BaseLayer;
  LazyReexportsManager &LRMgr;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_LAZYOBJECTLINKINGLAYER_H
