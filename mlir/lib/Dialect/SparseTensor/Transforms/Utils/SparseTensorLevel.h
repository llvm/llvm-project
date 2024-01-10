//===- SparseTensorLevel.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_SPARSETENSORLEVEL_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_SPARSETENSORLEVEL_H_

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

namespace mlir {
namespace sparse_tensor {

class SparseTensorLevel {
  SparseTensorLevel(SparseTensorLevel &&) = delete;
  SparseTensorLevel(const SparseTensorLevel &) = delete;
  SparseTensorLevel &operator=(SparseTensorLevel &&) = delete;
  SparseTensorLevel &operator=(const SparseTensorLevel &) = delete;

public:
  SparseTensorLevel() : SparseTensorLevel(LevelType::Undef, nullptr){};
  virtual ~SparseTensorLevel() = default;

  virtual Value peekCrdAt(OpBuilder &b, Location l, Value p) const = 0;

  /// Peeks the lower and upper bound to *fully* traverse the level with
  /// the given position `p` that the immediate parent level is current at.
  /// `bound` is only used when the level is `non-unique` and deduplication is
  /// required. It specifies the max upper bound of the non-unique segment.
  virtual std::pair<Value, Value> peekRangeAt(OpBuilder &b, Location l, Value p,
                                              Value bound = Value()) const = 0;

  LevelType getLT() const { return lt; }
  Value getPos() const { return pos; }
  Value getCrd() const { return crd; }
  Value getLoopHi() const { return loopHi; }
  Value getLoopLo() const { return loopLo; }

protected:
  SparseTensorLevel(LevelType lt, Value lvlSize)
      : lt(lt), lvlSize(lvlSize), pos(nullptr), crd(nullptr), loopHi(nullptr),
        loopLo(nullptr){};

  const LevelType lt;
  const Value lvlSize;

public: // TODO: make these values private upon feature complete.
  Value pos;
  Value crd;
  Value loopHi;
  Value loopLo;
};

/// Helper function to create a TensorLevel object from given `tensor`.
std::unique_ptr<SparseTensorLevel>
makeSparseTensorLevel(OpBuilder &builder, Location loc, Value t, Level l);

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_SPARSETENSORLEVEL_H_
