//===- SymbolDceUtils.h.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_SYMBOLDCEUTILS_H
#define MLIR_TRANSFORMS_SYMBOLDCEUTILS_H

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {

class Operation;
class SymbolTableCollection;

/// Eliminate dead symbols in the symbolTableOp.
LogicalResult symbolDce(Operation *);

/// Compute the liveness of the symbols within the given symbol table.
/// `symbolTableIsHidden` is true if this symbol table is known to be
/// unaccessible from operations in its parent regions.
LogicalResult computeLiveness(Operation *, SymbolTableCollection &, bool,
                              DenseSet<Operation *> &);
} // end namespace mlir

#endif // MLIR_TRANSFORMS_SYMBOLDCEUTILS_H
