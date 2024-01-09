//===- SparseTensorLevel.cpp - Tensor management class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SparseTensorLevel.h"
#include "CodegenUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::sparse_tensor;
using ValuePair = std::pair<Value, Value>;

//===----------------------------------------------------------------------===//
// File local helper functions/macros.
//===----------------------------------------------------------------------===//
#define CMPI(p, lhs, rhs)                                                      \
  (b.create<arith::CmpIOp>(l, arith::CmpIPredicate::p, (lhs), (rhs)))

#define C_IDX(v) (constantIndex(b, l, (v)))
#define YIELD(vs) (b.create<scf::YieldOp>(l, (vs)))
#define ADDI(lhs, rhs) (b.create<arith::AddIOp>(l, (lhs), (rhs)))
#define ANDI(lhs, rhs) (b.create<arith::AndIOp>(l, (lhs), (rhs)))
#define SUBI(lhs, rhs) (b.create<arith::SubIOp>(l, (lhs), (rhs)))
#define MULI(lhs, rhs) (b.create<arith::MulIOp>(l, (lhs), (rhs)))
#define REMUI(lhs, rhs) (b.create<arith::RemUIOp>(l, (lhs), (rhs)))
#define DIVUI(lhs, rhs) (b.create<arith::DivUIOp>(l, (lhs), (rhs)))
#define SELECT(c, lhs, rhs) (b.create<arith::SelectOp>(l, (c), (lhs), (rhs)))

static ValuePair constantRange(OpBuilder &b, Location l, Value lo, Value sz) {
  return std::make_pair(lo, ADDI(lo, sz));
}

//===----------------------------------------------------------------------===//
// SparseTensorLevel derived classes.
//===----------------------------------------------------------------------===//

namespace {

class SparseLevel : public SparseTensorLevel {
public:
  SparseLevel(LevelType lt, Value lvlSize, Value crdBuffer)
      : SparseTensorLevel(lt, lvlSize), crdBuffer(crdBuffer) {}

  Value peekCrdAt(OpBuilder &b, Location l, Value pos) const override {
    return genIndexLoad(b, l, crdBuffer, pos);
  }

protected:
  const Value crdBuffer;
};

class DenseLevel : public SparseTensorLevel {
public:
  DenseLevel(Value lvlSize) : SparseTensorLevel(LevelType::Dense, lvlSize) {
    // Dense level, loop upper bound equals to the level size.
    loopHi = lvlSize;
  }

  Value peekCrdAt(OpBuilder &, Location, Value pos) const override {
    return pos;
  }

  ValuePair peekRangeAt(OpBuilder &b, Location l, Value p,
                        Value max) const override {
    assert(max == nullptr && "Dense level can not be non-unique.");
    return constantRange(b, l, C_IDX(0), lvlSize);
  }
};

class CompressedLevel : public SparseLevel {
public:
  CompressedLevel(LevelType lt, Value lvlSize, Value posBuffer, Value crdBuffer)
      : SparseLevel(lt, lvlSize, crdBuffer), posBuffer(posBuffer) {}

  ValuePair peekRangeAt(OpBuilder &b, Location l, Value p,
                        Value max) const override {
    if (max == nullptr) {
      Value pLo = genIndexLoad(b, l, posBuffer, p);
      Value pHi = genIndexLoad(b, l, posBuffer, ADDI(p, C_IDX(1)));
      return {pLo, pHi};
    }
    llvm_unreachable("TODO: dedup not implemented");
  }

private:
  const Value posBuffer;
};

class LooseCompressedLevel : public SparseLevel {
public:
  LooseCompressedLevel(LevelType lt, Value lvlSize, Value posBuffer,
                       Value crdBuffer)
      : SparseLevel(lt, lvlSize, crdBuffer), posBuffer(posBuffer) {}

  ValuePair peekRangeAt(OpBuilder &b, Location l, Value p,
                        Value max) const override {
    // Allows this?
    assert(max == nullptr && "loss compressed level can not be non-unique.");

    p = MULI(p, C_IDX(2));
    Value pLo = genIndexLoad(b, l, posBuffer, p);
    Value pHi = genIndexLoad(b, l, posBuffer, ADDI(p, C_IDX(1)));
    return {pLo, pHi};
  }

private:
  const Value posBuffer;
};

class SingletonLevel : public SparseLevel {
public:
  SingletonLevel(LevelType lt, Value lvlSize, Value crdBuffer)
      : SparseLevel(lt, lvlSize, crdBuffer) {}

  ValuePair peekRangeAt(OpBuilder &b, Location l, Value p,
                        Value max) const override {
    if (max == nullptr)
      return constantRange(b, l, p, C_IDX(1));
    llvm_unreachable("TODO: dedup not implemented");
  }
};

class TwoOutFourLevel : public SparseLevel {
public:
  TwoOutFourLevel(LevelType lt, Value lvlSize, Value crdBuffer)
      : SparseLevel(lt, lvlSize, crdBuffer) {}

  ValuePair peekRangeAt(OpBuilder &b, Location l, Value p,
                        Value max) const override {
    assert(max == nullptr && "2:4 level can not be non-unique.");
    // Each 2:4 block has exactly two specified elements.
    Value c2 = C_IDX(2);
    return constantRange(b, l, MULI(p, c2), c2);
  }
};

} // namespace

std::unique_ptr<SparseTensorLevel>
sparse_tensor::makeSparseTensorLevel(OpBuilder &builder, Location loc, Value t,
                                     Level l) {
  auto stt = getSparseTensorType(t);

  LevelType lt = stt.getLvlType(l);
  Value lvlSz = stt.hasEncoding()
                    ? builder.create<LvlOp>(loc, t, l).getResult()
                    : builder.create<tensor::DimOp>(loc, t, l).getResult();

  switch (*getLevelFormat(lt)) {
  case LevelFormat::Dense:
    return std::make_unique<DenseLevel>(lvlSz);
  case LevelFormat::Compressed: {
    Value posBuf = genToPositions(builder, loc, t, l);
    Value crdBuf = genToCoordinates(builder, loc, t, l);
    return std::make_unique<CompressedLevel>(lt, lvlSz, posBuf, crdBuf);
  }
  case LevelFormat::LooseCompressed: {
    Value posBuf = genToPositions(builder, loc, t, l);
    Value crdBuf = genToCoordinates(builder, loc, t, l);
    return std::make_unique<LooseCompressedLevel>(lt, lvlSz, posBuf, crdBuf);
  }
  case LevelFormat::Singleton: {
    Value crdBuf = genToCoordinates(builder, loc, t, l);
    return std::make_unique<SingletonLevel>(lt, lvlSz, crdBuf);
  }
  case LevelFormat::TwoOutOfFour: {
    Value crdBuf = genToCoordinates(builder, loc, t, l);
    return std::make_unique<TwoOutFourLevel>(lt, lvlSz, crdBuf);
  }
  }
  llvm_unreachable("unrecognizable level format");
}

#undef CMPI
#undef C_IDX
#undef YIELD
#undef ADDI
#undef ANDI
#undef SUBI
#undef MULI
#undef SELECT
