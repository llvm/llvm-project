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
#define C_TRUE (constantI1(b, l, true))
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
// File local helpers
//===----------------------------------------------------------------------===//

static ValueRange
genWhenInBound(OpBuilder &b, Location l, SparseIterator &it, ValueRange elseRet,
               llvm::function_ref<void(OpBuilder &, Location, Value)> builder) {
  // !it.end() ? callback(*crd) : resOOB;
  TypeRange ifRetTypes = elseRet.getTypes();
  auto ifOp = b.create<scf::IfOp>(l, ifRetTypes, it.genNotEnd(b, l), true);

  b.setInsertionPointToStart(ifOp.thenBlock());
  Value crd = it.deref(b, l);
  builder(b, l, crd);

  b.setInsertionPointToStart(ifOp.elseBlock());
  YIELD(elseRet);

  b.setInsertionPointAfter(ifOp);
  return ifOp.getResults();
}

/// Generates code to compute the *absolute* offset of the slice based on the
/// provide minimum coordinates in the slice.
/// E.g., when reducing d0 + d1 + d2, we need two slices to fully reduced the
/// expression, i,e, s1 = slice(T, d0), s2 = slice(s1, d1). The *absolute*
/// offset is the offset computed relative to the initial tensors T.
///
/// When isNonEmpty == true, the computed offset is meaningless and should not
/// be used during runtime, the method generates code to return 0 currently in
/// that case.
///
/// offset = minCrd >= size ? minCrd - size + 1 : 0;
static Value offsetFromMinCrd(OpBuilder &b, Location l, Value minCrd,
                              Value size) {
  Value geSize = CMPI(uge, minCrd, size);
  // Computes minCrd - size + 1
  Value mms = SUBI(ADDI(minCrd, C_IDX(1)), size);
  // This is the absolute offset related to the actual tensor.
  return SELECT(geSize, mms, C_IDX(0));
}

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
  Value upperBound(OpBuilder &b, Location l) const override {
    return stl.size();
  };

  SmallVector<Value> serialize() const override {
    SmallVector<Value> ret;
    ret.push_back(itPos);
    if (randomAccessible()) {
      // Loop high is implicit (defined by `upperBound()`) for random-access
      // iterator, but we need to memorize posLo for linearization.
      ret.push_back(posLo);
    } else {
      ret.push_back(posHi);
    }
    return ret;
  };

  void deserialize(ValueRange vs) override {
    assert(vs.size() == 2);
    seek(vs.front());
    if (randomAccessible())
      posLo = vs.back();
    else
      posHi = vs.back();
  };

  ValuePair getCurPosition() const override { return {itPos, nullptr}; }

  void genInit(OpBuilder &b, Location l,
               const SparseIterator *parent) override {
    Value pos = C_IDX(0);
    Value hi = nullptr;
    if (parent)
      std::tie(pos, hi) = parent->getCurPosition();

    std::tie(posLo, posHi) = stl.peekRangeAt(b, l, pos, hi);
    // Seek to the lowest position.
    seek(posLo);
  }

  ValuePair genForCond(OpBuilder &b, Location l) override {
    if (randomAccessible())
      return {deref(b, l), upperBound(b, l)};
    return std::make_pair(getLoopLo(b, l), posHi);
  }

  Value genNotEnd(OpBuilder &b, Location l) override {
    // We used the first level bound as the bound the collapsed set of levels.
    return CMPI(ult, itPos, posHi);
  }

  Value deref(OpBuilder &b, Location l) override {
    if (randomAccessible()) {
      updateCrd(SUBI(itPos, posLo));
    } else {
      updateCrd(stl.peekCrdAt(b, l, itPos));
    }
    return getCrd();
  };

  ValueRange forward(OpBuilder &b, Location l) override {
    seek(ADDI(itPos, C_IDX(1)));
    return getItVals();
  }

  ValueRange forwardIf(OpBuilder &b, Location l, Value cond) override {
    Value curPos = getItVals().front();
    Value nxPos = forward(b, l).front();
    seek(SELECT(cond, nxPos, curPos));
    return getItVals();
  }

  void locate(OpBuilder &b, Location l, Value crd) override {
    assert(randomAccessible());
    // Seek to the linearized position.
    seek(ADDI(crd, posLo));
    updateCrd(crd);
  }

  Value itPos; // the position that represent the iterator

  Value posLo, posHi;
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
  Value upperBound(OpBuilder &b, Location l) const override {
    return stl.size();
  };

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

  SmallVector<Value> serialize() const override {
    SmallVector<Value> ret;
    ret.append(getItVals().begin(), getItVals().end());
    ret.push_back(posHi);
    return ret;
  };
  void deserialize(ValueRange vs) override {
    assert(vs.size() == 3);
    seek(vs.take_front(getItVals().size()));
    posHi = vs.back();
  };

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
  Value fromWrapCrd(OpBuilder &b, Location l, Value wrapCrd) const {
    // crd = (wrapCrd - offset) / stride
    return DIVUI(SUBI(wrapCrd, offset), stride);
  }
  Value toWrapCrd(OpBuilder &b, Location l, Value crd) const {
    // wrapCrd = crd * stride + offset
    return ADDI(MULI(crd, stride), offset);
  }

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
  Value upperBound(OpBuilder &b, Location l) const override { return size; };

  SmallVector<Value> serialize() const override { return wrap->serialize(); };
  void deserialize(ValueRange vs) override { wrap->deserialize(vs); };
  ValuePair getCurPosition() const override { return wrap->getCurPosition(); }

  void genInit(OpBuilder &b, Location l,
               const SparseIterator *parent) override {
    wrap->genInit(b, l, parent);
    if (!randomAccessible()) {
      // TODO: we can skip this when stride == 1 and offset == 0, we can also
      // use binary search here.
      forwardIf(b, l, genShouldFilter(b, l));
    } else {
      // Else, locate to the slice.offset, which is the first coordinate
      // included by the slice.
      wrap->locate(b, l, offset);
    }
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

class SubSectIterator;
class NonEmptySubSectIterator : public SparseIterator {

  // The sliced pointer buffer is organized as:
  //     [[itVal0, itVal1, ..., pNx0],
  //      [itVal0, itVal1, ..., pNx0],
  //      ...]
  Value allocSubSectPosBuf(OpBuilder &b, Location l) {
    return b.create<memref::AllocaOp>(
        l,
        MemRefType::get({ShapedType::kDynamic, tupleSz + 1}, b.getIndexType()),
        maxTupleCnt);
  }

  SmallVector<Value> loadItVals(OpBuilder &b, Location l, Value tupleId) const {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < tupleSz; i++) {
      Value v = b.create<memref::LoadOp>(l, subSectPosBuf,
                                         ValueRange{tupleId, C_IDX(i)});
      ret.push_back(v);
    }
    return ret;
  }

  void storeItVals(OpBuilder &b, Location l, Value tupleId, ValueRange itVals) {
    assert(itVals.size() == tupleSz);
    for (unsigned i = 0; i < tupleSz; i++) {
      b.create<memref::StoreOp>(l, itVals[i], subSectPosBuf,
                                ValueRange{tupleId, C_IDX(i)});
    }
  }

public:
  NonEmptySubSectIterator(OpBuilder &b, Location l,
                          const SparseIterator *parent,
                          std::unique_ptr<SparseIterator> &&wrap,
                          Value subSectSz, unsigned stride)
      : SparseIterator(IterKind::kNonEmptySubSect, wrap->tid, wrap->lvl,
                       /*itVals=*/subSectMeta),
        subSectSz(subSectSz), stride(stride), parent(parent),
        wrap(std::move(wrap)) {

    auto *p = dyn_cast_or_null<NonEmptySubSectIterator>(parent);
    assert(stride == 1);
    if (p == nullptr) {
      // Extract subsections along the root level.
      maxTupleCnt = C_IDX(1);
    } else if (p->lvl == lvl) {
      // Extract subsections along the same level.
      maxTupleCnt = p->maxTupleCnt;
      assert(false && "Not implemented.");
    } else {
      // Extract subsections along the previous level.
      assert(p->lvl + 1 == lvl);
      maxTupleCnt = MULI(p->maxTupleCnt, p->subSectSz);
    }
    // We don't need an extra buffer to find subsections on dense levels.
    if (randomAccessible())
      return;

    tupleSz = this->wrap->serialize().size();
    subSectPosBuf = allocSubSectPosBuf(b, l);
  }

  bool randomAccessible() const override { return wrap->randomAccessible(); };
  bool iteratableByFor() const override { return randomAccessible(); };
  Value upperBound(OpBuilder &b, Location l) const override {
    auto *p = dyn_cast_or_null<NonEmptySubSectIterator>(parent);
    Value parentUB =
        p && p->lvl == lvl ? p->upperBound(b, l) : wrap->upperBound(b, l);
    return ADDI(SUBI(parentUB, subSectSz), C_IDX(1));
  };

  // For LLVM-style RTTI.
  static bool classof(const SparseIterator *from) {
    return from->kind == IterKind::kNonEmptySubSect;
  }

  void genInit(OpBuilder &b, Location l, const SparseIterator *) override;

  void locate(OpBuilder &b, Location l, Value crd) override {
    Value absOff = crd;
    auto *p = dyn_cast_or_null<NonEmptySubSectIterator>(parent);
    if (p && p->lvl == lvl)
      absOff = ADDI(crd, p->getAbsOff());

    wrap->locate(b, l, absOff);
    seek(ValueRange{absOff, absOff, C_TRUE});
    updateCrd(crd);
  }

  Value genNotEnd(OpBuilder &b, Location l) override { return getNotEnd(); };

  Value deref(OpBuilder &b, Location l) override {
    // Use the relative offset to coiterate.
    Value crd;
    auto *p = dyn_cast_or_null<NonEmptySubSectIterator>(parent);
    if (p && p->lvl == lvl)
      crd = SUBI(getAbsOff(), p->getAbsOff());
    crd = getAbsOff();

    updateCrd(crd);
    return crd;
  };

  ValueRange forward(OpBuilder &b, Location l) override;

  Value getMinCrd() const { return subSectMeta[0]; }
  Value getAbsOff() const { return subSectMeta[1]; }
  Value getNotEnd() const { return subSectMeta[2]; }

  // Number of values required to serialize the wrapped iterator.
  unsigned tupleSz;
  // Max number of tuples, and the actual number of tuple.
  Value maxTupleCnt, tupleCnt;
  // The memory used to cache the tuple serialized from the wrapped iterator.
  Value subSectPosBuf;

  const Value subSectSz;
  const unsigned stride;

  const SparseIterator *parent;
  std::unique_ptr<SparseIterator> wrap;

  Value subSectMeta[3]; // minCrd, absolute offset, notEnd

  friend SubSectIterator;
};

class SubSectIterator : public SparseIterator {
  Value fromWrapCrd(OpBuilder &b, Location l, Value wrapCrd) {
    assert(stride == 1);
    return SUBI(wrapCrd, subSect.getAbsOff());
  }

public:
  SubSectIterator(const NonEmptySubSectIterator &subSect,
                  const SparseIterator &parent,
                  std::unique_ptr<SparseIterator> &&wrap, Value size,
                  unsigned stride)
      : SparseIterator(IterKind::kSubSect, wrap.get()), subSect(subSect),
        parent(parent), wrap(std::move(wrap)), size(size), stride(stride) {
    assert(stride == 1 && "Not implemented.");
    assert(subSect.tid == tid && subSect.lvl == lvl);
    // The immediate parents of a subsection iterator is either a non-empty
    // subsect iterator or another subsection iterator for the previous level
    // depending on the index varaiables' reduction order.
    assert(parent.kind == IterKind::kNonEmptySubSect ||
           parent.kind == IterKind::kSubSect);
    assert(parent.kind != IterKind::kNonEmptySubSect || &parent == &subSect);
    assert(parent.kind != IterKind::kSubSect || parent.lvl + 1 == lvl);
  };

  bool randomAccessible() const override { return wrap->randomAccessible(); };
  bool iteratableByFor() const override { return randomAccessible(); };
  Value upperBound(OpBuilder &b, Location l) const override { return size; }
  std::pair<Value, Value> getCurPosition() const override {
    return wrap->getCurPosition();
  };

  void genInit(OpBuilder &b, Location l, const SparseIterator *) override {
    if (llvm::isa<NonEmptySubSectIterator>(parent)) {
      if (randomAccessible()) {
        // We continue from the parent's offset.
        wrap->deserialize(subSect.wrap->serialize());
        return;
      }
      // Else deserializing from the cached values.
      wrap->deserialize(subSect.loadItVals(b, l, C_IDX(0)));
    } else {
      llvm_unreachable("Not implemented");
    }
  }

  void locate(OpBuilder &b, Location l, Value crd) override {
    Value absCrd = ADDI(crd, subSect.getAbsOff());
    wrap->locate(b, l, absCrd);
    updateCrd(crd);
  }

  Value genNotEnd(OpBuilder &b, Location l) override {
    assert(!wrap->randomAccessible());
    ValueRange r = genWhenInBound(
        b, l, *wrap, C_FALSE, [this](OpBuilder &b, Location l, Value wrapCrd) {
          Value crd = fromWrapCrd(b, l, wrapCrd);
          // crd < size
          YIELD(CMPI(ult, crd, size));
        });
    assert(r.size() == 1);
    return r.front();
  }

  Value deref(OpBuilder &b, Location l) override {
    Value wrapCrd = wrap->deref(b, l);
    Value crd = fromWrapCrd(b, l, wrapCrd);
    updateCrd(crd);
    return crd;
  };

  ValueRange forward(OpBuilder &b, Location l) override {
    return wrap->forward(b, l);
  };

  const NonEmptySubSectIterator &subSect;
  const SparseIterator &parent;

  std::unique_ptr<SparseIterator> wrap;
  Value size;
  unsigned stride;
};

} // namespace

//===----------------------------------------------------------------------===//
// Complex SparseIterator derived classes impl.
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
  ValueRange r = genWhenInBound(
      b, l, *wrap, C_FALSE, [this](OpBuilder &b, Location l, Value wrapCrd) {
        Value notLegit = genCrdNotLegitPredicate(b, l, wrapCrd);
        YIELD(notLegit);
      });

  assert(r.size() == 1);
  return r.front();
}

Value FilterIterator::genNotEnd(OpBuilder &b, Location l) {
  assert(!wrap->randomAccessible());
  ValueRange r = genWhenInBound(
      b, l, *wrap, C_FALSE, [this](OpBuilder &b, Location l, Value wrapCrd) {
        Value crd = fromWrapCrd(b, l, wrapCrd);
        // crd < size
        YIELD(CMPI(ult, crd, size));
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
        ValueRange cont =
            genWhenInBound(b, l, *wrap, C_FALSE,
                           [this](OpBuilder &b, Location l, Value wrapCrd) {
                             // crd < size && !legit();
                             Value notLegit =
                                 genCrdNotLegitPredicate(b, l, wrapCrd);
                             Value crd = fromWrapCrd(b, l, wrapCrd);
                             Value ret = ANDI(CMPI(ult, crd, size), notLegit);
                             YIELD(ret);
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

void NonEmptySubSectIterator::genInit(OpBuilder &b, Location l,
                                      const SparseIterator *) {
  auto *p = dyn_cast_or_null<NonEmptySubSectIterator>(parent);
  if (p) {
    llvm_unreachable("Not implemented");
  } else {
    wrap->genInit(b, l, parent);
    Value c0 = C_IDX(0);
    if (randomAccessible()) {
      seek({/*minCrd=*/c0, /*offset=*/c0, /*notEnd=*/C_TRUE});
      return;
    }
    // Handle sparse subsection iterator.
    tupleCnt = C_IDX(1);
    SmallVector<Value> elseRet{c0, c0, /*notEnd=*/C_FALSE};
    ValueRange meta = genWhenInBound(
        b, l, *wrap, elseRet, [this](OpBuilder &b, Location l, Value crd) {
          Value offset = offsetFromMinCrd(b, l, crd, subSectSz);
          YIELD((ValueRange{crd, offset, C_TRUE}));
        });

    seek(meta);
    SmallVector<Value> itVals = wrap->serialize();
    storeItVals(b, l, c0, itVals);
  }
}

ValueRange NonEmptySubSectIterator::forward(OpBuilder &b, Location l) {
  assert(!randomAccessible());
  Value c0 = C_IDX(0), c1 = C_IDX(1);
  // Forward to the next non empty slice by generating
  //
  // if (minCrd > offset) {
  //   offset += 1
  // } else {
  //    minCrd = nextMinInSlice();
  //    offset = minCrd - size + 1;
  // }
  //
  // if (offset + size > parents.size)
  //   isNonEmpty = false;
  Value fastPathP = CMPI(ugt, getMinCrd(), getAbsOff());
  auto ifOp = b.create<scf::IfOp>(l, getItVals().getTypes(), fastPathP, true);
  {
    OpBuilder::InsertionGuard guard(b);
    // Take the fast path
    // if (minCrd > offset)
    //   offset += 1
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    Value nxOffset = ADDI(getAbsOff(), c1);
    YIELD((ValueRange{getMinCrd(), nxOffset, getNotEnd()}));

    // else /*minCrd == offset*/ {
    //    for (i = 0; i < tupleCnt; i++) {
    //       wrap->deserialize(pos[i]);
    //       minCrd=min(minCrd, *wrap);
    //    }
    //    offset = minCrd - size + 1;
    // }
    b.setInsertionPointToStart(&ifOp.getElseRegion().front());
    ValueRange loopArgs{upperBound(b, l), // nextMinCrd
                        C_FALSE};         // isNotEnd
    auto loopNest = scf::buildLoopNest(
        b, l, c0, tupleCnt, c1, loopArgs,
        [this](OpBuilder &b, Location l, ValueRange ivs,
               ValueRange iterArgs) -> scf::ValueVector {
          Value tupleId = ivs.front();
          SmallVector<Value> itVals = loadItVals(b, l, tupleId);
          wrap->deserialize(itVals);
          return genWhenInBound(
              b, l, *wrap, /*elseRet=*/iterArgs,
              [this, iterArgs, tupleId](OpBuilder &b, Location l, Value crd) {
                // if coord == minCrd
                //   wrap->forward();
                Value isMin = CMPI(eq, crd, getMinCrd());
                wrap->forwardIf(b, l, isMin);
                // Update the forwarded iterator values if needed.
                auto ifIsMin = b.create<scf::IfOp>(l, isMin, false);
                b.setInsertionPointToStart(&ifIsMin.getThenRegion().front());
                storeItVals(b, l, tupleId, wrap->serialize());
                b.setInsertionPointAfter(ifIsMin);
                // if (!wrap.end())
                //  yield(min(nxMinCrd, *wrap), true)
                Value nxMin = iterArgs[0];
                ValueRange ret = genWhenInBound(
                    b, l, *wrap, /*elseRet=*/iterArgs,
                    [nxMin](OpBuilder &b, Location l, Value crd) {
                      Value nx = SELECT(CMPI(ult, crd, nxMin), crd, nxMin);
                      YIELD((ValueRange{nx, C_TRUE}));
                    });
                YIELD(ret);
              });
        });

    scf::ForOp forOp = loopNest.loops.front();
    b.setInsertionPointAfter(forOp);

    Value nxMinCrd = forOp.getResult(0);
    Value nxNotEnd = forOp.getResult(1);
    Value nxAbsOff = offsetFromMinCrd(b, l, nxMinCrd, subSectSz);
    YIELD((ValueRange{nxMinCrd, nxAbsOff, nxNotEnd}));
  }

  Value nxMinCrd = ifOp.getResult(0);
  Value nxAbsOff = ifOp.getResult(1);
  Value nxNotEnd = ifOp.getResult(2);

  // We should at least forward the offset by one.
  Value minAbsOff = ADDI(getAbsOff(), c1);
  nxAbsOff = SELECT(CMPI(ugt, minAbsOff, nxAbsOff), minAbsOff, nxAbsOff);

  assert(stride == 1 && "Not yet implemented");

  seek(ValueRange{nxMinCrd, nxAbsOff, nxNotEnd});
  // The coordinate should not exceeds the space upper bound.
  Value crd = deref(b, l);
  nxNotEnd = ANDI(nxNotEnd, CMPI(ult, crd, upperBound(b, l)));

  seek(ValueRange{nxMinCrd, nxAbsOff, nxNotEnd});
  return getItVals();
}

//===----------------------------------------------------------------------===//
// SparseIterator factory functions.
//===----------------------------------------------------------------------===//

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
  return std::make_unique<NonEmptySubSectIterator>(
      b, l, parent, std::move(delegate), size, stride);
}

std::unique_ptr<SparseIterator> sparse_tensor::makeTraverseSubSectIterator(
    const SparseIterator &subsectIter, const SparseIterator &parent,
    std::unique_ptr<SparseIterator> &&wrap, Value size, unsigned stride) {
  return std::make_unique<SubSectIterator>(
      llvm::cast<NonEmptySubSectIterator>(subsectIter), parent, std::move(wrap),
      size, stride);
}

#undef CMPI
#undef C_IDX
#undef YIELD
#undef ADDI
#undef ANDI
#undef SUBI
#undef MULI
#undef SELECT
