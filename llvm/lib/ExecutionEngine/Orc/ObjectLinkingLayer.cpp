//===------- ObjectLinkingLayer.cpp - JITLink backed ORC ObjectLayer ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "orc"

namespace llvm::orc {

char ObjectLinkingLayer::ID;

void ObjectLinkingLayer::emit(std::unique_ptr<MaterializationResponsibility> R,
                              std::unique_ptr<MemoryBuffer> O) {
  assert(O && "Object must not be null");
  MemoryBufferRef ObjBuffer = O->getMemBufferRef();

  if (auto G = jitlink::createLinkGraphFromObject(
          ObjBuffer, getExecutionSession().getSymbolStringPool())) {
    emit(std::move(R), std::move(*G), std::move(O));
  } else {
    R->getExecutionSession().reportError(G.takeError());
    R->failMaterialization();
    return;
  }
}

} // namespace llvm::orc
