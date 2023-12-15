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

public:
  SparseTensorLevel() : SparseTensorLevel(LevelType::Undef, nullptr){};
  virtual ~SparseTensorLevel() = default;

  virtual Value peekCrdAt(OpBuilder &b, Location l, Value p) const = 0;

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

class DenseLevel : public SparseTensorLevel {
public:
  DenseLevel(Value lvlSize) : SparseTensorLevel(LevelType::Dense, lvlSize) {
    // Dense level, loop upper bound equals to the level size.
    loopHi = lvlSize;
  }

  Value peekCrdAt(OpBuilder &, Location, Value pos) const override {
    return pos;
  }
};

class SparseLevel : public SparseTensorLevel {
public:
  SparseLevel(LevelType lt, Value lvlSize, Value crdBuffer)
      : SparseTensorLevel(lt, lvlSize), crdBuffer(crdBuffer) {}

  Value peekCrdAt(OpBuilder &b, Location l, Value pos) const override;

public: // TODO: make these values private upon feature complete.
  const Value crdBuffer;
};

class CompressedLevel : public SparseLevel {
public:
  CompressedLevel(LevelType lt, Value lvlSize, Value posBuffer, Value crdBuffer)
      : SparseLevel(lt, lvlSize, crdBuffer), posBuffer(posBuffer) {}

public: // TODO: make these values private upon feature complete.
  const Value posBuffer;
};

class LooseCompressedLevel : public SparseLevel {
public:
  LooseCompressedLevel(LevelType lt, Value lvlSize, Value posBuffer,
                       Value crdBuffer)
      : SparseLevel(lt, lvlSize, crdBuffer), posBuffer(posBuffer) {}

public: // TODO: make these values private upon feature complete.
  const Value posBuffer;
};

class SingletonLevel : public SparseLevel {
public:
  SingletonLevel(LevelType lt, Value lvlSize, Value crdBuffer)
      : SparseLevel(lt, lvlSize, crdBuffer) {}
};

class TwoOutFourLevel : public SparseLevel {
public:
  TwoOutFourLevel(LevelType lt, Value lvlSize, Value crdBuffer)
      : SparseLevel(lt, lvlSize, crdBuffer) {}
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_SPARSETENSORLEVEL_H_
