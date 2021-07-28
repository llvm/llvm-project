//===- MemRefUtils.cpp - Utilities to support the MemRef dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the MemRef dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

/// Finds a single dealloc operation for the given allocated value.
llvm::Optional<Operation *> mlir::findDealloc(Value allocValue) {
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
