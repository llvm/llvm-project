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
  (b.create<arith::CmpIOp>(l, arith::CmpIPredicate::p, (lhs), (rhs))           \
       .getResult())

#define C_FALSE (constantI1(b, l, false))
#define C_IDX(v) (constantIndex(b, l, (v)))
#define YIELD(vs) (b.create<scf::YieldOp>(l, (vs)))
#define ADDI(lhs, rhs) (b.create<arith::AddIOp>(l, (lhs), (rhs)).getResult())
#define ORI(lhs, rhs) (b.create<arith::OrIOp>(l, (lhs), (rhs)).getResult())
#define ANDI(lhs, rhs) (b.create<arith::AndIOp>(l, (lhs), (rhs)).getResult())
#define SUBI(lhs, rhs) (b.create<arith::SubIOp>(l, (lhs), (rhs)).getResult())
#define MULI(lhs, rhs) (b.create<arith::MulIOp>(l, (lhs), (rhs)).getResult())
#define REMUI(lhs, rhs) (b.create<arith::RemUIOp>(l, (lhs), (rhs)).getResult())
#define DIVUI(lhs, rhs) (b.create<arith::DivUIOp>(l, (lhs), (rhs)).getResult())
#define SELECT(c, lhs, rhs)                                                    \
  (b.create<arith::SelectOp>(l, (c), (lhs), (rhs)).getResult())

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

  ValuePair getCurPosition() const override { return {itPos, nullptr}; }

  void genInit(OpBuilder &b, Location l,
               const SparseIterator *parent) override {
    Value pos = C_IDX(0);
    Value hi = nullptr;
    if (parent)
      std::tie(pos, hi) = parent->getCurPosition();

    std::tie(posLo, loopHi) = stl.peekRangeAt(b, l, pos, hi);
    // Seek to the lowest position.
    seek(posLo);
  }

  ValuePair genForCond(OpBuilder &b, Location l) override {
    assert(iteratableByFor());
    return std::make_pair(getLoopLo(b, l), loopHi);
  }

  Value genNotEnd(OpBuilder &b, Location l) override {
    // We used the first level bound as the bound the collapsed set of levels.
    return CMPI(ult, itPos, loopHi);
  }

  Value deref(OpBuilder &b, Location l) override {
    updateCrd(stl.peekCrdAt(b, l, itPos));
    return getCrd();
  };

  ValueRange forward(OpBuilder &b, Location l) override {
    seek(ADDI(itPos, C_IDX(1)));
    return getItVals();
  }

  void locate(OpBuilder &b, Location l, Value crd) override {
    assert(randomAccessible());
    // Seek to the linearized position.
    seek(ADDI(crd, posLo));
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

  ValuePair getCurPosition() const override { return {getPos(), getSegHi()}; }

  void genInit(OpBuilder &b, Location l,
               const SparseIterator *parent) override {

    Value pos = C_IDX(0);
    Value hi = nullptr;
    if (parent)
      std::tie(pos, hi) = parent->getCurPosition();

    Value posLo;
    std::tie(posLo, posHi) = stl.peekRangeAt(b, l, pos, hi);

    seek({posLo, genSegmentHigh(b, l, posLo)});
  }

  Value genNotEnd(OpBuilder &b, Location l) override {
    return CMPI(ult, getPos(), posHi);
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

  Value posHi;
  Value posAndSegHi[2]; // position and segment high
  const SparseTensorLevel &stl;
};

class FilterIterator : public SparseIterator {
  // Coorindate translation between crd loaded from the wrap iterator and the
  // filter iterator.
  Value fromWrapCrd(OpBuilder &b, Location l, Value wrapCrd) {
    // crd = (wrapCrd - offset) / stride
    return DIVUI(SUBI(wrapCrd, offset), stride);
  }
  Value toWrapCrd(OpBuilder &b, Location l, Value crd) {
    // wrapCrd = crd * stride + offset
    return ADDI(MULI(crd, stride), offset);
  }

  ValueRange genWhenWrapInBound(
      OpBuilder &b, Location l, ValueRange elseRet,
      llvm::function_ref<ValueRange(OpBuilder &, Location, Value)> builder);

  Value genCrdNotLegitPredicate(OpBuilder &b, Location l, Value wrapCrd);

  Value genShouldFilter(OpBuilder &b, Location l);

public:
  FilterIterator(std::unique_ptr<SparseIterator> &&w, Value offset,
                 Value stride, Value size)
      : SparseIterator(IterKind::kFilter, w.get()), offset(offset),
        stride(stride), size(size), wrap(std::move(w)) {}

  // For LLVM-style RTTI.
  static bool classof(const SparseIterator *from) {
    return from->kind == IterKind::kFilter;
  }

  bool randomAccessible() const override { return wrap->randomAccessible(); };
  bool iteratableByFor() const override { return randomAccessible(); };

  ValuePair getCurPosition() const override { return wrap->getCurPosition(); }

  void genInit(OpBuilder &b, Location l,
               const SparseIterator *parent) override {
    wrap->genInit(b, l, parent);
    if (!randomAccessible()) {
      // TODO: we can skip this when stride == 1 and offset == 0, we can also
      // use binary search here.
      forwardIf(b, l, genShouldFilter(b, l));
    }
  }

  ValuePair genForCond(OpBuilder &b, Location l) override {
    assert(randomAccessible());

    auto [lo, hi] = wrap->genForCond(b, l);
    // if offset < lo, we use lo - offset as the new lower bound, else we use 0.
    Value loInBound = CMPI(ult, offset, lo);
    lo = SELECT(loInBound, SUBI(lo, offset), C_IDX(0));
    return {lo, size};
  }

  Value genNotEnd(OpBuilder &b, Location l) override;

  Value deref(OpBuilder &b, Location l) override {
    updateCrd(fromWrapCrd(b, l, wrap->deref(b, l)));
    return getCrd();
  }

  void locate(OpBuilder &b, Location l, Value crd) override {
    assert(randomAccessible());
    wrap->locate(b, l, toWrapCrd(b, l, crd));
    updateCrd(crd);
  }

  ValueRange forward(OpBuilder &b, Location l) override;

  const Value offset, stride, size;
  std::unique_ptr<SparseIterator> wrap;
};

/*
class NonEmptySubSectIterator : public SparseIterator {
public:
  NonEmptySubSectIterator(OpBuilder &b, Location l,
                          const SparseIterator *parent,
                          std::unique_ptr<SparseIterator> &&w, Value size)
      : SparseIterator(IterKind::kNonEmptySubSect, w->tid, w->lvl),
        parent(parent), wrap(std::move(w)), size(size), stride(stride) {

    auto *p = dyn_cast_or_null<NonEmptySubSectIterator>(parent);
    if (p == nullptr) {
      // Extract subsections along the root level.
      prevUnResCnt = C_IDX(1);
    } else if (p->lvl == lvl) {
      // Extract subsections along the same level.
      prevUnResCnt = p->prevUnResCnt;
    } else {
      // Extract subsections along the previous level.
      assert(p->lvl + 1 == lvl);
      prevUnResCnt = MULI(p->prevUnResCnt, p->size);
    }

    // We don't need an extra buffer to find subsections on dense levels.
    if (randomAccessible())
      return;
    subSectPosBuf = allocSlicePosBuf(b, l, prevUnResCnt);
  }

  // For LLVM-style RTTI.
  static bool classof(const SparseIterator *from) {
    return from->kind == IterKind::kNonEmptySubSect;
  }

  bool randomAccessible() const override { return wrap->randomAccessible(); };
  bool iteratableByFor() const override { return randomAccessible(); };

  Value size, prevUnResCnt, subSectPosBuf;
  unsigned stride;
};

class SubSectIterator : public SparseIterator {
public:
  SubSectIterator(const SparseIterator *parent,
                  std::unique_ptr<SparseIterator> &&w)
      : SparseIterator(IterKind::kSubSect, w->tid, w->lvl), parent(parent),
        wrap(std::move(w)) {}

  // For LLVM-style RTTI.
  static bool classof(const SparseIterator *from) {
    return from->kind == IterKind::kSubSect;
  }

  bool randomAccessible() const override { return wrap->randomAccessible(); };
  bool iteratableByFor() const override { return randomAccessible(); };

  const SparseIterator *parent;
  std::unique_ptr<SparseIterator> wrap;
};
*/
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
        Value inBound = CMPI(ult, ivs.front(), posHi);
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

ValueRange FilterIterator::genWhenWrapInBound(
    OpBuilder &b, Location l, ValueRange elseRet,
    llvm::function_ref<ValueRange(OpBuilder &, Location, Value)> builder) {
  // !it.end() ? callback(*crd) : resOOB;
  TypeRange ifRetTypes = elseRet.getTypes();
  auto ifOp = b.create<scf::IfOp>(l, ifRetTypes, wrap->genNotEnd(b, l), true);

  b.setInsertionPointToStart(ifOp.thenBlock());
  Value wrapCrd = wrap->deref(b, l);
  YIELD(builder(b, l, wrapCrd));

  b.setInsertionPointToStart(ifOp.elseBlock());
  YIELD(elseRet);

  b.setInsertionPointAfter(ifOp);
  return ifOp.getResults();
}

Value FilterIterator::genCrdNotLegitPredicate(OpBuilder &b, Location l,
                                              Value wrapCrd) {
  Value crd = fromWrapCrd(b, l, wrapCrd);
  // not on stride
  Value notlegit = CMPI(ne, toWrapCrd(b, l, crd), wrapCrd);
  // wrapCrd < offset
  notlegit = ORI(CMPI(ult, wrapCrd, offset), notlegit);
  //  crd >= length
  notlegit = ORI(CMPI(uge, crd, size), notlegit);
  return notlegit;
}

Value FilterIterator::genShouldFilter(OpBuilder &b, Location l) {
  ValueRange r = genWhenWrapInBound(
      b, l, C_FALSE, [this](OpBuilder &b, Location l, Value wrapCrd) {
        Value notLegit = genCrdNotLegitPredicate(b, l, wrapCrd);
        return notLegit.getDefiningOp()->getResults();
      });

  assert(r.size() == 1);
  return r.front();
}

Value FilterIterator::genNotEnd(OpBuilder &b, Location l) {
  assert(!wrap->randomAccessible());
  ValueRange r = genWhenWrapInBound(
      b, l, C_FALSE, [this](OpBuilder &b, Location l, Value wrapCrd) {
        Value crd = fromWrapCrd(b, l, wrapCrd);
        // crd < size
        return CMPI(ult, crd, size).getDefiningOp()->getResults();
      });
  assert(r.size() == 1);
  return r.front();
}

ValueRange FilterIterator::forward(OpBuilder &b, Location l) {
  assert(!randomAccessible());
  // Generates
  //
  // wrap ++;
  // while !it.end() && !legit(*it)
  //   wrap ++;
  wrap->forward(b, l);
  auto whileOp = b.create<scf::WhileOp>(
      l, getItVals().getTypes(), getItVals(),
      /*beforeBuilder=*/
      [this](OpBuilder &b, Location l, ValueRange ivs) {
        linkNewScope(ivs);
        ValueRange cont = genWhenWrapInBound(
            b, l, C_FALSE, [this](OpBuilder &b, Location l, Value wrapCrd) {
              // crd < size && !legit();
              Value notLegit = genCrdNotLegitPredicate(b, l, wrapCrd);
              Value crd = fromWrapCrd(b, l, wrapCrd);
              Value ret = ANDI(CMPI(ult, crd, size), notLegit);
              return ret.getDefiningOp()->getResults();
            });
        b.create<scf::ConditionOp>(l, cont.front(), ivs);
      },
      /*afterBuilder=*/
      [this](OpBuilder &b, Location l, ValueRange ivs) {
        linkNewScope(ivs);
        wrap->forward(b, l);
        YIELD(getItVals());
      });

  b.setInsertionPointAfter(whileOp);
  linkNewScope(whileOp.getResults());
  return getItVals();
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
sparse_tensor::makeSimpleIterator(const SparseTensorLevel &stl) {
  if (!isUniqueLT(stl.getLT())) {
    // We always dedupliate the non-unique level, but we should optimize it away
    // if possible.
    return std::make_unique<DedupIterator>(stl);
  }
  return std::make_unique<TrivialIterator>(stl);
}

std::unique_ptr<SparseIterator>
sparse_tensor::makeSlicedLevelIterator(std::unique_ptr<SparseIterator> &&sit,
                                       Value offset, Value stride, Value size) {

  return std::make_unique<FilterIterator>(std::move(sit), offset, stride, size);
}

std::unique_ptr<SparseIterator> sparse_tensor::makeNonEmptySubSectIterator(
    OpBuilder &b, Location l, const SparseIterator *parent,
    std::unique_ptr<SparseIterator> &&delegate, Value size, unsigned stride) {
  return nullptr;
  //  return std::make_unique<NonEmptySubSectIterator>(
  //      b, l, parent, std::move(lvlIt), size, stride);
}

std::unique_ptr<SparseIterator> sparse_tensor::makeTraverseSubSectIterator(
    const SparseIterator *, std::unique_ptr<SparseIterator> &&delegate) {
  return nullptr;
  //  return std::make_unique<SubSectIterator>(parent, std::move(lvlIt));
}

#undef CMPI
#undef C_IDX
#undef YIELD
#undef ADDI
#undef ANDI
#undef SUBI
#undef MULI
#undef SELECT
