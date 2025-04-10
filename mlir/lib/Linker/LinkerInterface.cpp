//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Linker/LinkerInterface.h"

#define DEBUG_TYPE "mlir-linker-interface"

using namespace mlir;
using namespace mlir::link;

//===----------------------------------------------------------------------===//
// LinkState
//===----------------------------------------------------------------------===//

template <typename CloneFunc>
Operation *cloneImpl(Operation *src, IRMapping &mapping, CloneFunc cloneFunc) {
  assert(!mapping.contains(src));
  Operation *dst = cloneFunc(src);
  mapping.map(src, dst);
  return dst;
}

Operation *LinkState::clone(Operation *src) {
    return cloneImpl(src, mapping, [this](Operation *op) {
        return builder.clone(*op, mapping);
    });
}

Operation *LinkState::cloneWithoutRegions(Operation *src) {
    return cloneImpl(src, mapping, [this](Operation *op) {
        return builder.cloneWithoutRegions(*op, mapping);
    });
}

Operation *LinkState::getDestinationOp() const {
  return builder.getInsertionBlock()->getParentOp();
}

Operation *LinkState::remapped(Operation *src) const {
  return mapping.lookupOrNull(src);
}

//===----------------------------------------------------------------------===//
// SymbolAttrLinkerInterface
//===----------------------------------------------------------------------===//

StringRef SymbolAttrLinkerInterface::getSymbol(Operation *op) const {
  return op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
      .getValue();
}

Conflict SymbolAttrLinkerInterface::findConflict(Operation *src) const {
  assert(canBeLinked(src) && "expected linkable operation");
  StringRef symbol = getSymbol(src);
  auto it = summary.find(symbol);
  if (it == summary.end())
    return Conflict::noConflict(src);
  return {it->second, src};
}
