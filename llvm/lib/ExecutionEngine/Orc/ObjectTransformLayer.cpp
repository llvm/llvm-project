//===---------- ObjectTransformLayer.cpp - Object Transform Layer ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace orc {

ObjectTransformLayer::ObjectTransformLayer(ExecutionSession &ES,
                                            ObjectLayer &BaseLayer,
                                            TransformFunction Transform)
    : ObjectLayer(ES), BaseLayer(BaseLayer), Transform(std::move(Transform)) {}

void ObjectTransformLayer::emit(MaterializationResponsibility R,
                                std::unique_ptr<MemoryBuffer> O) {
  assert(O && "Module must not be null");

  if (auto TransformedObj = Transform(std::move(O)))
    BaseLayer.emit(std::move(R), std::move(*TransformedObj));
  else {
    R.failMaterialization();
    getExecutionSession().reportError(TransformedObj.takeError());
  }
}

} // End namespace orc.
} // End namespace llvm.
