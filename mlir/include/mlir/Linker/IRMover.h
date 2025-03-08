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
};

struct IRMover {

  ModuleOp composite;

  std::vector<ConflictPair> worklist;

  /// Mapping of values to their cloned counterpart.
  IRMapping mapping;

  explicit IRMover(ModuleOp composite) : composite(composite) {}

  LogicalResult move(ArrayRef<ConflictPair> valuesToLink);

private:
  Operation * remap(ConflictPair pair);

  Operation * materialize(ConflictPair pair) const;
};

} // namespace mlir::link

namespace llvm {

template <>
struct DenseMapInfo<mlir::link::ConflictPair> {
  static mlir::link::ConflictPair getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return {{}, static_cast<mlir::Operation*>(pointer)};
  }
  static mlir::link::ConflictPair getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return {{}, static_cast<mlir::Operation *>(pointer)};
  }
  static unsigned getHashValue(mlir::link::ConflictPair val) {
    return DenseMapInfo<const mlir::Operation *>::getHashValue(val.src);
  }
  static bool isEqual(mlir::link::ConflictPair lhs, mlir::link::ConflictPair rhs) {
    return lhs.src == rhs.src;
  }
};

} // namespace llvm

#endif // MLIR_LINKER_IRMOVER_H
