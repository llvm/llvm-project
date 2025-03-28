//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINKER_IRMOVER_H
#define MLIR_LINKER_IRMOVER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/Hashing.h"

namespace mlir::link {

struct ConflictPair {
  Operation *dst;
  Operation *src;

  bool hasConflict() const { return dst; }

  static ConflictPair noConflict(Operation *src) { return {nullptr, src}; }
};

using Summary = llvm::StringMap<ConflictPair>;

struct IRMover {

  ModuleOp composite;

  std::vector<ConflictPair> worklist;

  /// Mapping of values to their cloned counterpart.
  IRMapping mapping;

  explicit IRMover(ModuleOp composite) : composite(composite) {}

  LogicalResult move(const Summary &summary);

private:
  Operation * remap(ConflictPair pair);

  Operation * materialize(ConflictPair pair) const;
};

} // namespace mlir::link

#endif // MLIR_LINKER_IRMOVER_H
