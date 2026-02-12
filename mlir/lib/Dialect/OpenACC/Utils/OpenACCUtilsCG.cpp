//===- OpenACCUtilsCG.cpp - OpenACC Code Generation Utilities -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for OpenACC code generation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace acc {

std::optional<DataLayout> getDataLayout(Operation *op, bool allowDefault) {
  if (!op)
    return std::nullopt;

  // Walk up the parent chain to find the nearest operation with an explicit
  // data layout spec. Check ModuleOp explicitly since it does not actually
  // implement DataLayoutOpInterface as a trait (it just has the same methods).
  Operation *current = op;
  while (current) {
    // Check for ModuleOp with explicit data layout spec
    if (auto mod = llvm::dyn_cast<ModuleOp>(current)) {
      if (mod.getDataLayoutSpec())
        return DataLayout(mod);
    } else if (auto dataLayoutOp =
                   llvm::dyn_cast<DataLayoutOpInterface>(current)) {
      // Check other DataLayoutOpInterface implementations
      if (dataLayoutOp.getDataLayoutSpec())
        return DataLayout(dataLayoutOp);
    }
    current = current->getParentOp();
  }

  // No explicit data layout found; return default if allowed
  if (allowDefault) {
    // Check if op itself is a ModuleOp
    if (auto mod = llvm::dyn_cast<ModuleOp>(op))
      return DataLayout(mod);
    // Otherwise check parents
    if (auto mod = op->getParentOfType<ModuleOp>())
      return DataLayout(mod);
  }

  return std::nullopt;
}

} // namespace acc
} // namespace mlir
