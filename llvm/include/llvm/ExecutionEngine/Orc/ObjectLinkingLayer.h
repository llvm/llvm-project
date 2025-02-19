//===-- ObjectLinkingLayer.h - JITLink-based jit linking layer --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition for an JITLink-based, in-process object linking
// layer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/LinkGraphLinkingLayer.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace llvm {

namespace jitlink {
class EHFrameRegistrar;
class LinkGraph;
class Symbol;
} // namespace jitlink

namespace orc {

/// An ObjectLayer implementation built on JITLink.
///
/// Clients can use this class to add relocatable object files to an
/// ExecutionSession, and it typically serves as the base layer (underneath
/// a compiling layer like IRCompileLayer) for the rest of the JIT.
class ObjectLinkingLayer : public LinkGraphLinkingLayer,
                           public RTTIExtends<ObjectLinkingLayer, ObjectLayer> {
private:
  using BaseObjectLayer = RTTIExtends<ObjectLinkingLayer, ObjectLayer>;

public:
  static char ID;

  using ReturnObjectBufferFunction =
      std::function<void(std::unique_ptr<MemoryBuffer>)>;

  /// Construct an ObjectLinkingLayer using the ExecutorProcessControl
  /// instance's memory manager.
  ObjectLinkingLayer(ExecutionSession &ES)
      : LinkGraphLinkingLayer(ES), BaseObjectLayer(ES) {}

  /// Construct an ObjectLinkingLayer using a custom memory manager.
  ObjectLinkingLayer(ExecutionSession &ES,
                     jitlink::JITLinkMemoryManager &MemMgr)
      : LinkGraphLinkingLayer(ES, MemMgr), BaseObjectLayer(ES) {}

  /// Construct an ObjectLinkingLayer. Takes ownership of the given
  /// JITLinkMemoryManager. This method is a temporary hack to simplify
  /// co-existence with RTDyldObjectLinkingLayer (which also owns its
  /// allocators).
  ObjectLinkingLayer(ExecutionSession &ES,
                     std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr)
      : LinkGraphLinkingLayer(ES, std::move(MemMgr)), BaseObjectLayer(ES) {}

  using LinkGraphLinkingLayer::getExecutionSession;

  /// Set an object buffer return function. By default object buffers are
  /// deleted once the JIT has linked them. If a return function is set then
  /// it will be called to transfer ownership of the buffer instead.
  void setReturnObjectBuffer(ReturnObjectBufferFunction ReturnObjectBuffer) {
    this->ReturnObjectBuffer = std::move(ReturnObjectBuffer);
  }

  using LinkGraphLinkingLayer::add;
  using LinkGraphLinkingLayer::emit;

  using ObjectLayer::add;

  /// Emit an object file.
  void emit(std::unique_ptr<MaterializationResponsibility> R,
            std::unique_ptr<MemoryBuffer> O) override;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H
