//===- OpenACCOpsInterfaces.h - OpenACC op interfaces -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds C++ types used by OpenACC operation interface signatures
// and includes the generated interface declarations. StoreLikeOpInterface
// uses the stored-value types defined here. Types for another interface go
// in this file as well, above the generated include.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCOPSINTERFACES_H
#define MLIR_DIALECT_OPENACC_OPENACCOPSINTERFACES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class OpOperand;

namespace acc {

/// Where the stored bits come from.
enum class StoredSourceKind {
  Operand,  // directly from an operand, e.g. fir.store %operand to %loc
  Contents, // bits loaded from an address operand, e.g. a swap
  Symbol,   // a symbol, or the bits loaded from that symbol
  Unknown   // a source this operation cannot name
};

struct StoredSource {
  StoredSourceKind kind = StoredSourceKind::Unknown;
  OpOperand *operand = nullptr; // Operand or Contents
  SymbolRefAttr symbol = {};    // Symbol
  // For Symbol, true means the symbol's contents.
  bool dereferenced = false;
};

/// Null destination means an unknown place.
struct StoreLocation {
  OpOperand *destination = nullptr;
};

struct StoredValue {
  StoredSource source;
  llvm::SmallVector<StoreLocation, 1> locations;
};

} // namespace acc
} // namespace mlir

#include "mlir/Dialect/OpenACC/OpenACCOpsInterfaces.h.inc"

#endif // MLIR_DIALECT_OPENACC_OPENACCOPSINTERFACES_H
