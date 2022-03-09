//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::memref;

#include "mlir/Dialect/MemRef/IR/MemRefOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MemRefDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct MemRefInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // namespace

void mlir::memref::MemRefDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MemRef/IR/MemRefOps.cpp.inc"
      >();
  addInterfaces<MemRefInlinerInterface>();
}

/// Finds a single dealloc operation for the given allocated value.
llvm::Optional<Operation *> mlir::memref::findDealloc(Value allocValue) {
  Operation *dealloc = nullptr;
  for (Operation *user : allocValue.getUsers()) {
    auto effectInterface = dyn_cast<MemoryEffectOpInterface>(user);
    if (!effectInterface)
      continue;
    // Try to find a free effect that is applied to one of our values
    // that will be automatically freed by our pass.
    SmallVector<MemoryEffects::EffectInstance, 2> effects;
    effectInterface.getEffectsOnValue(allocValue, effects);
    const bool isFree =
        llvm::any_of(effects, [&](MemoryEffects::EffectInstance &it) {
          return isa<MemoryEffects::Free>(it.getEffect());
        });
    if (!isFree)
      continue;
    // If we found > 1 dealloc, return None.
    if (dealloc)
      return llvm::None;
    dealloc = user;
  }
  return dealloc;
}
