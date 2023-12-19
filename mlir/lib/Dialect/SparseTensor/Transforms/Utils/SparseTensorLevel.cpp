//===- SparseTensorLevel.cpp - Tensor management class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SparseTensorLevel.h"
#include "CodegenUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::sparse_tensor;
using ValuePair = std::pair<Value, Value>;
using ValueTuple = std::tuple<Value, Value, Value>;

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

// Helper functions that load/store into the position buffer for slice-driven
// loops.
static constexpr unsigned kSliceIterWidth = 3;
// The sliced pointer buffer is organized as:
//     [[pLo0, pLo1, pLo2, ...],
//      [pHi0, pHi1, pHi2, ...],
//      [pNx0, pNx1, pNx2, ...]]
static Value allocSlicePosBuf(OpBuilder &b, Location l, Value tupleCnt) {
  Value bufSz = MULI(tupleCnt, C_IDX(kSliceIterWidth));
  // Additional two metadata {memSize, idx} at head.
  return genAlloca(b, l, bufSz, b.getIndexType());
}

// Gets and sets position values for slice-driven loops.
enum class SlicePosKind { kLo, kHi, kNext };
static Value getSlicePosIdx(OpBuilder &b, Location l, Value posBuf,
                            Value tupleIdx, SlicePosKind posKind) {
  Value dim = b.create<memref::DimOp>(l, posBuf, C_IDX(0));
  Value tupleCnt = DIVUI(dim, C_IDX(kSliceIterWidth));
  switch (posKind) {
  case SlicePosKind::kLo:
    return tupleIdx;
  case SlicePosKind::kHi:
    return ADDI(tupleIdx, tupleCnt);
  case SlicePosKind::kNext:
    return ADDI(tupleIdx, MULI(tupleCnt, C_IDX(2)));
  }
  llvm_unreachable("unexpected kind");
}
static Value loadSlicePos(OpBuilder &b, Location l, Value sPosBuf,
                          Value tupleIdx, SlicePosKind posKind) {
  return genIndexLoad(b, l, sPosBuf,
                      getSlicePosIdx(b, l, sPosBuf, tupleIdx, posKind));
}
static void updateSlicePos(OpBuilder &b, Location l, Value sPosBuf, Value pos,
                           Value tupleIdx, SlicePosKind posKind) {
  b.create<memref::StoreOp>(l, pos, sPosBuf,
                            getSlicePosIdx(b, l, sPosBuf, tupleIdx, posKind));
}

//===----------------------------------------------------------------------===//
// SparseTensorLevel derived classes.
//===----------------------------------------------------------------------===//

namespace {

class SparseLevel : public SparseTensorLevel {
public:
  SparseLevel(unsigned tid, Level lvl, LevelType lt, Value lvlSize,
              Value crdBuffer)
      : SparseTensorLevel(tid, lvl, lt, lvlSize), crdBuffer(crdBuffer) {}

  Value peekCrdAt(OpBuilder &b, Location l, Value iv) const override {
    return genIndexLoad(b, l, crdBuffer, iv);
  }

protected:
  const Value crdBuffer;
};

class DenseLevel : public SparseTensorLevel {
public:
  DenseLevel(unsigned tid, Level lvl, Value lvlSize, bool encoded)
      : SparseTensorLevel(tid, lvl, LevelType::Dense, lvlSize),
        encoded(encoded) {}

  Value peekCrdAt(OpBuilder &, Location, Value pos) const override {
    return pos;
  }

  ValuePair peekRangeAt(OpBuilder &b, Location l, Value p,
                        Value max) const override {
    assert(max == nullptr && "Dense level can not be non-unique.");
    if (encoded) {
      Value posLo = MULI(p, lvlSize);
      return {posLo, lvlSize};
    }
    // No need to linearize the position for non-annotated tensors.
    return {C_IDX(0), lvlSize};
  }

  const bool encoded;
};

class CompressedLevel : public SparseLevel {
public:
  CompressedLevel(unsigned tid, Level lvl, LevelType lt, Value lvlSize,
                  Value posBuffer, Value crdBuffer)
      : SparseLevel(tid, lvl, lt, lvlSize, crdBuffer), posBuffer(posBuffer) {}

  ValuePair peekRangeAt(OpBuilder &b, Location l, Value p,
                        Value max) const override {
    if (max == nullptr) {
      Value pLo = genIndexLoad(b, l, posBuffer, p);
      Value pHi = genIndexLoad(b, l, posBuffer, ADDI(p, C_IDX(1)));
      return {pLo, pHi};
    }
    llvm_unreachable("compressed-nu should be the first non-unique level.");
  }

private:
  const Value posBuffer;
};

class LooseCompressedLevel : public SparseLevel {
public:
  LooseCompressedLevel(unsigned tid, Level lvl, LevelType lt, Value lvlSize,
                       Value posBuffer, Value crdBuffer)
      : SparseLevel(tid, lvl, lt, lvlSize, crdBuffer), posBuffer(posBuffer) {}

  ValuePair peekRangeAt(OpBuilder &b, Location l, Value p,
                        Value max) const override {
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
  SingletonLevel(unsigned tid, Level lvl, LevelType lt, Value lvlSize,
                 Value crdBuffer)
      : SparseLevel(tid, lvl, lt, lvlSize, crdBuffer) {}

  ValuePair peekRangeAt(OpBuilder &b, Location l, Value p,
                        Value segHi) const override {
    if (segHi == nullptr)
      return {p, ADDI(p, C_IDX(1))};

    // Use the segHi as the loop upper bound.
    return {p, segHi};
  }
};

class TwoOutFourLevel : public SparseLevel {
public:
  TwoOutFourLevel(unsigned tid, Level lvl, LevelType lt, Value lvlSize,
                  Value crdBuffer)
      : SparseLevel(tid, lvl, lt, lvlSize, crdBuffer) {}

  ValuePair peekRangeAt(OpBuilder &b, Location l, Value p,
                        Value max) const override {
    assert(max == nullptr && isUnique() && "2:4 level can not be non-unique.");
    // Each 2:4 blk has exactly two specified elements.
    Value posLo = MULI(p, C_IDX(2));
    return {posLo, ADDI(posLo, C_IDX(2))};
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// SparseIterator derived classes.
//===----------------------------------------------------------------------===//

namespace {

class TrivialIterator : public SparseIterator {
  Value getLoopLo(OpBuilder &b, Location l) const {
    // Dense loop are traversed by coordinate, delinearize the position to get
    // the coordinate.
    if (randomAccessible())
      return SUBI(itPos, posLo);
    return itPos;
  }

public:
  TrivialIterator(const SparseTensorLevel &stl,
                  const IterKind kind = IterKind::kTrivial)
      : SparseIterator(kind, stl.tid, stl.lvl, itPos), stl(stl) {}

  // For LLVM-style RTTI.
  static bool classof(const SparseIterator *from) {
    return from->kind == IterKind::kTrivial;
  }

  bool randomAccessible() const override { return isDenseLT(stl.getLT()); };
  bool iteratableByFor() const override { return true; };

  ValuePair peekNxLvlRange(OpBuilder &b, Location l,
                           const SparseTensorLevel &stl) const override {
    assert(stl.tid == this->tid && stl.lvl - 1 == this->lvl);
    return stl.peekRangeAt(b, l, itPos);
  }

  void genInit(OpBuilder &b, Location l,
               const SparseIterator *parent) override {
    if (parent)
      std::tie(posLo, loopHi) = parent->peekNxLvlRange(b, l, stl);
    else
      std::tie(posLo, loopHi) = stl.peekRangeAt(b, l, C_IDX(0));

    // Only randomly accessible iterator's position need to be linearized.
    seek(posLo);
  }

  ValuePair genForCond(OpBuilder &b, Location l) override {
    assert(iteratableByFor());
    return std::make_pair(getLoopLo(b, l), loopHi);
  }

  Value genIsEnd(OpBuilder &b, Location l) override {
    // We used the first level bound as the bound the collapsed set of levels.
    return CMPI(ult, itPos, loopHi);
  }

  Value deref(OpBuilder &b, Location l) override {
    updateCrd(stl.peekCrdAt(b, l, itPos));
    return getCrd();
  };

  ValueRange forward(OpBuilder &b, Location l) override {
    seek(ADDI(itPos, C_IDX(1)).getResult());
    return getItVals();
  }

  void locate(OpBuilder &b, Location l, Value crd) override {
    assert(randomAccessible());
    // Seek to the linearized position.
    seek(ADDI(crd, posLo).getResult());
    updateCrd(crd);
  }

  Value itPos; // the position that represent the iterator

  Value posLo, loopHi;
  const SparseTensorLevel &stl;
};

class DedupIterator : public SparseIterator {
private:
  Value genSegmentHigh(OpBuilder &b, Location l, Value pos);

public:
  DedupIterator(const SparseTensorLevel &stl)
      : SparseIterator(IterKind::kDedup, stl.tid, stl.lvl, posAndSegHi),
        stl(stl) {
    assert(!stl.isUnique());
  }
  // For LLVM-style RTTI.
  static bool classof(const SparseIterator *from) {
    return from->kind == IterKind::kDedup;
  }

  bool randomAccessible() const override { return false; };
  bool iteratableByFor() const override { return false; };

  ValuePair peekNxLvlRange(OpBuilder &b, Location l,
                           const SparseTensorLevel &stl) const override {
    assert(stl.tid == this->tid && stl.lvl - 1 == this->lvl);
    return stl.peekRangeAt(b, l, getPos(), getSegHi());
  }

  void genInit(OpBuilder &b, Location l,
               const SparseIterator *parent) override {
    Value posLo;

    if (parent)
      std::tie(posLo, loopHi) = parent->peekNxLvlRange(b, l, stl);
    else
      std::tie(posLo, loopHi) = stl.peekRangeAt(b, l, C_IDX(0));

    seek({posLo, genSegmentHigh(b, l, posLo)});
  }

  Value genIsEnd(OpBuilder &b, Location l) override {
    return CMPI(ult, getPos(), loopHi);
  }

  Value deref(OpBuilder &b, Location l) override {
    updateCrd(stl.peekCrdAt(b, l, getPos()));
    return getCrd();
  };

  ValueRange forward(OpBuilder &b, Location l) override {
    Value nxPos = getSegHi(); // forward the position to the next segment.
    seek({nxPos, genSegmentHigh(b, l, nxPos)});
    return getItVals();
  }

  Value getPos() const { return posAndSegHi[0]; }
  Value getSegHi() const { return posAndSegHi[1]; }

  Value loopHi;
  Value posAndSegHi[2]; // position and segment high
  const SparseTensorLevel &stl;
};

} // namespace

//===----------------------------------------------------------------------===//
// SparseIterator derived classes impl.
//===----------------------------------------------------------------------===//

ValueRange SparseIterator::forwardIf(OpBuilder &b, Location l, Value cond) {
  auto ifOp = b.create<scf::IfOp>(l, getItVals().getTypes(), cond, true);
  // Generate else branch first, otherwise iterator values will be updated by
  // `forward()`.
  b.setInsertionPointToStart(ifOp.elseBlock());
  YIELD(getItVals());

  b.setInsertionPointToStart(ifOp.thenBlock());
  YIELD(forward(b, l));

  b.setInsertionPointAfter(ifOp);
  seek(ifOp.getResults());
  return getItVals();
}

Value DedupIterator::genSegmentHigh(OpBuilder &b, Location l, Value pos) {
  auto whileOp = b.create<scf::WhileOp>(
      l, pos.getType(), pos,
      /*beforeBuilder=*/
      [this, pos](OpBuilder &b, Location l, ValueRange ivs) {
        Value inBound = CMPI(ult, ivs.front(), loopHi);
        auto ifInBound = b.create<scf::IfOp>(l, b.getI1Type(), inBound, true);
        {
          OpBuilder::InsertionGuard guard(b);
          // If in bound, load the next coordinates and check duplication.
          b.setInsertionPointToStart(ifInBound.thenBlock());
          Value headCrd = stl.peekCrdAt(b, l, pos);
          Value tailCrd = stl.peekCrdAt(b, l, ivs.front());
          Value isDup = CMPI(eq, headCrd, tailCrd);
          YIELD(isDup);
          // Else, the position is out of bound, yield false.
          b.setInsertionPointToStart(ifInBound.elseBlock());
          YIELD(constantI1(b, l, false));
        }
        b.create<scf::ConditionOp>(l, ifInBound.getResults()[0], ivs);
      },
      /*afterBuilder=*/
      [](OpBuilder &b, Location l, ValueRange ivs) {
        // pos ++
        Value nxPos = ADDI(ivs[0], C_IDX(1));
        YIELD(nxPos);
      });
  // Return the segment high.
  return whileOp.getResult(0);
}

Value FilterIterator::genShouldFilter(OpBuilder &b, Location l) {
  Value end = wrap->genIsEnd(b, l);

  auto shouldFilter = b.create<scf::IfOp>(l, b.getI1Type(), end, true);
  // it.end() ? false : should_filter(*it);
  b.setInsertionPointToStart(shouldFilter.thenBlock());
  YIELD(constantI1(b, l, false));

  // Iterator not at the end.
  b.setInsertionPointToStart(shouldFilter.elseBlock());
  Value wrapCrd = wrap->deref(b, l);
  Value crd = fromWrapCrd(b, l, wrapCrd);
  // on stride
  Value legit = CMPI(eq, toWrapCrd(b, l, crd), wrapCrd);
  // wrapCrd >= offset
  legit = ANDI(CMPI(uge, wrapCrd, offset), legit);
  //  crd < length
  legit = ANDI(CMPI(ult, crd, size), legit);
  YIELD(legit);

  b.setInsertionPointAfter(shouldFilter);
  return shouldFilter.getResult(0);
}

std::unique_ptr<SparseTensorLevel>
sparse_tensor::makeSparseTensorLevel(OpBuilder &b, Location l, Value t,
                                     unsigned tid, Level lvl) {
  auto stt = getSparseTensorType(t);

  LevelType lt = stt.getLvlType(lvl);
  Value sz = stt.hasEncoding() ? b.create<LvlOp>(l, t, lvl).getResult()
                               : b.create<tensor::DimOp>(l, t, lvl).getResult();

  switch (*getLevelFormat(lt)) {
  case LevelFormat::Dense:
    return std::make_unique<DenseLevel>(tid, lvl, sz, stt.hasEncoding());
  case LevelFormat::Compressed: {
    Value pos = genToPositions(b, l, t, lvl);
    Value crd = genToCoordinates(b, l, t, lvl);
    return std::make_unique<CompressedLevel>(tid, lvl, lt, sz, pos, crd);
  }
  case LevelFormat::LooseCompressed: {
    Value pos = genToPositions(b, l, t, lvl);
    Value crd = genToCoordinates(b, l, t, lvl);
    return std::make_unique<LooseCompressedLevel>(tid, lvl, lt, sz, pos, crd);
  }
  case LevelFormat::Singleton: {
    Value crd = genToCoordinates(b, l, t, lvl);
    return std::make_unique<SingletonLevel>(tid, lvl, lt, sz, crd);
  }
  case LevelFormat::TwoOutOfFour: {
    Value crd = genToCoordinates(b, l, t, lvl);
    return std::make_unique<TwoOutFourLevel>(tid, lvl, lt, sz, crd);
  }
  }
  llvm_unreachable("unrecognizable level format");
}

std::pair<std::unique_ptr<SparseTensorLevel>, std::unique_ptr<SparseIterator>>
sparse_tensor::makeSynLevelAndIterator(Value sz, unsigned tid, unsigned lvl) {
  auto stl = std::make_unique<DenseLevel>(tid, lvl, sz, /*encoded=*/false);
  auto it = std::make_unique<TrivialIterator>(*stl);
  return std::make_pair(std::move(stl), std::move(it));
}

std::unique_ptr<SparseIterator>
sparse_tensor::makeSimpleIterator(const SparseTensorLevel &stl, bool dedup) {
  dedup = dedup && !isUniqueLT(stl.getLT());
  if (dedup)
    return std::make_unique<DedupIterator>(stl);
  return std::make_unique<TrivialIterator>(stl);
}

std::unique_ptr<SparseIterator>
sparse_tensor::makeSlicedLevelIterator(std::unique_ptr<SparseIterator> &&sit,
                                       Value offset, Value stride, Value size) {
  return nullptr;
  // return std::make_unique<FilterIterator>(std::move(sit), offset, stride,
  // size);
}

std::unique_ptr<SparseIterator> sparse_tensor::makeNonEmptySubSectIterator(
    OpBuilder &b, Location l, const SparseIterator *parent,
    std::unique_ptr<SparseIterator> &&lvlIt, Value size, unsigned stride) {
  return nullptr;
  // return std::make_unique<NonEmptySubSectIterator>(
  //     b, l, parent, std::move(lvlIt), size, stride);
}

std::unique_ptr<SparseIterator> sparse_tensor::makeTraverseSubSectIterator(
    const SparseIterator *parent, std::unique_ptr<SparseIterator> &&lvlIt) {
  // return std::make_unique<SubSectIterator>(parent, std::move(lvlIt));
  return nullptr;
}

#undef CMPI
#undef C_IDX
#undef YIELD
#undef ADDI
#undef ANDI
#undef SUBI
#undef MULI
#undef SELECT
