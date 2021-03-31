//===- Utils.cpp - Utilities to support the MemRef dialect ----------------===//
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

/// Finds associated deallocs that can be linked to our allocation nodes (if
/// any).
Operation *mlir::findDealloc(Value allocValue) {
  auto userIt = llvm::find_if(allocValue.getUsers(), [&](Operation *user) {
    auto effectInterface = dyn_cast<MemoryEffectOpInterface>(user);
    if (!effectInterface)
      return false;
    // Try to find a free effect that is applied to one of our values
    // that will be automatically freed by our pass.
    SmallVector<MemoryEffects::EffectInstance, 2> effects;
    effectInterface.getEffectsOnValue(allocValue, effects);
    return llvm::any_of(effects, [&](MemoryEffects::EffectInstance &it) {
      return isa<MemoryEffects::Free>(it.getEffect());
    });
  });
  // Assign the associated dealloc operation (if any).
  return userIt != allocValue.user_end() ? *userIt : nullptr;
}
