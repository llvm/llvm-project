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
#define MINUI(lhs, rhs) (b.create<arith::MinUIOp>(l, (lhs), (rhs)).getResult())
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
      : SparseTensorLevel(tid, lvl, LevelFormat::Dense, lvlSize),
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

class NOutOfMLevel : public SparseLevel {
public:
  NOutOfMLevel(unsigned tid, Level lvl, LevelType lt, Value lvlSize,
               Value crdBuffer)
      : SparseLevel(tid, lvl, lt, lvlSize, crdBuffer) {}

  ValuePair peekRangeAt(OpBuilder &b, Location l, Value p,
                        Value max) const override {
    assert(max == nullptr && isUnique() && "n:m level can not be non-unique.");
    // Each n:m blk has exactly n specified elements.
    auto n = getN(lt);
    Value posLo = MULI(p, C_IDX(n));
    return {posLo, ADDI(posLo, C_IDX(n))};
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// File local helpers
//===----------------------------------------------------------------------===//

static scf::ValueVector genWhenInBound(
    OpBuilder &b, Location l, SparseIterator &it, ValueRange elseRet,
    llvm::function_ref<scf::ValueVector(OpBuilder &, Location, Value)>
        builder) {
  TypeRange ifRetTypes = elseRet.getTypes();
  auto ifOp = b.create<scf::IfOp>(l, ifRetTypes, it.genNotEnd(b, l), true);

  b.setInsertionPointToStart(ifOp.thenBlock());
  Value crd = it.deref(b, l);
  scf::ValueVector ret = builder(b, l, crd);
  YIELD(ret);

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
  // Compute minCrd - size + 1.
  Value mms = SUBI(ADDI(minCrd, C_IDX(1)), size);
  // This is the absolute offset related to the actual tensor.
  return SELECT(geSize, mms, C_IDX(0));
}

//===----------------------------------------------------------------------===//
// SparseIterator derived classes.
//===----------------------------------------------------------------------===//

namespace {

// The iterator that traverses a concrete sparse tensor levels. High-level
// abstract iterators wrap it to achieve more complex goals (such as collapsing
// several levels). It also holds the common storage to hold the mlir::Values
// for itself as well as for wrappers.
class ConcreteIterator : public SparseIterator {
protected:
  ConcreteIterator(const SparseTensorLevel &stl, IterKind kind,
                   unsigned cursorValCnt)
      : SparseIterator(kind, stl.tid, stl.lvl, cursorValCnt, cursorValsStorage),
        stl(stl), cursorValsStorage(cursorValCnt, nullptr) {
    assert(getCursor().size() == cursorValCnt);
  };

public:
  // For LLVM-style RTTI.
  static bool classof(const SparseIterator *from) {
    return from->kind == IterKind::kTrivial;
  }

  bool randomAccessible() const override { return isDenseLT(stl.getLT()); };
  bool iteratableByFor() const override { return kind != IterKind::kDedup; };
  Value upperBound(OpBuilder &b, Location l) const override {
    return stl.getSize();
  };

protected:
  const SparseTensorLevel &stl;
  // Owner of the storage, all wrappers build on top of a concrete iterator
  // share the same storage such that the iterator values are always
  // synchronized.
  SmallVector<Value> cursorValsStorage;
};

class TrivialIterator : public ConcreteIterator {
public:
  TrivialIterator(const SparseTensorLevel &stl)
      : ConcreteIterator(stl, IterKind::kTrivial, /*itValCnt=*/1) {}

  std::string getDebugInterfacePrefix() const override {
    return std::string("trivial<") + stl.toString() + ">";
  }
  SmallVector<Type> getCursorValTypes(OpBuilder &b) const override {
    return {b.getIndexType()};
  }

  SmallVector<Value> serialize() const override {
    SmallVector<Value> ret;
    ret.push_back(getItPos());
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

  ValuePair getCurPosition() const override { return {getItPos(), nullptr}; }

  void genInitImpl(OpBuilder &b, Location l,
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
    return std::make_pair(getItPos(), posHi);
  }

  Value genNotEndImpl(OpBuilder &b, Location l) override {
    // We used the first level bound as the bound the collapsed set of levels.
    return CMPI(ult, getItPos(), posHi);
  }

  Value derefImpl(OpBuilder &b, Location l) override {
    if (randomAccessible()) {
      updateCrd(SUBI(getItPos(), posLo));
    } else {
      updateCrd(stl.peekCrdAt(b, l, getItPos()));
    }
    return getCrd();
  };

  ValueRange forwardImpl(OpBuilder &b, Location l) override {
    seek(ADDI(getItPos(), C_IDX(1)));
    return getCursor();
  }

  ValueRange forwardIf(OpBuilder &b, Location l, Value cond) override {
    Value curPos = getCursor().front();
    Value nxPos = forward(b, l).front();
    seek(SELECT(cond, nxPos, curPos));
    return getCursor();
  }

  void locateImpl(OpBuilder &b, Location l, Value crd) override {
    assert(randomAccessible());
    // Seek to the linearized position.
    seek(ADDI(crd, posLo));
    updateCrd(crd);
  }

  Value getItPos() const { return getCursor().front(); }
  Value posLo, posHi;
};

class DedupIterator : public ConcreteIterator {
private:
  Value genSegmentHigh(OpBuilder &b, Location l, Value pos);

public:
  DedupIterator(const SparseTensorLevel &stl)
      : ConcreteIterator(stl, IterKind::kDedup, /*itValCnt=*/2) {
    assert(!stl.isUnique());
  }
  // For LLVM-style RTTI.
  static bool classof(const SparseIterator *from) {
    return from->kind == IterKind::kDedup;
  }

  std::string getDebugInterfacePrefix() const override {
    return std::string("dedup<") + stl.toString() + ">";
  }
  SmallVector<Type> getCursorValTypes(OpBuilder &b) const override {
    return {b.getIndexType(), b.getIndexType()};
  }

  ValuePair getCurPosition() const override { return {getPos(), getSegHi()}; }

  void genInitImpl(OpBuilder &b, Location l,
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
    ret.append(getCursor().begin(), getCursor().end());
    ret.push_back(posHi);
    return ret;
  };
  void deserialize(ValueRange vs) override {
    assert(vs.size() == 3);
    seek(vs.take_front(getCursor().size()));
    posHi = vs.back();
  };

  Value genNotEndImpl(OpBuilder &b, Location l) override {
    return CMPI(ult, getPos(), posHi);
  }

  Value derefImpl(OpBuilder &b, Location l) override {
    updateCrd(stl.peekCrdAt(b, l, getPos()));
    return getCrd();
  };

  ValueRange forwardImpl(OpBuilder &b, Location l) override {
    Value nxPos = getSegHi(); // forward the position to the next segment.
    seek({nxPos, genSegmentHigh(b, l, nxPos)});
    return getCursor();
  }

  Value getPos() const { return getCursor()[0]; }
  Value getSegHi() const { return getCursor()[1]; }

  Value posHi;
};

//
// A filter iterator wrapped from another iterator. The filter iterator update
// the wrapped iterator *in-place*.
//
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
  // TODO: avoid unnessary check when offset == 0 and/or when stride == 1 and/or
  // when crd always < size.
  FilterIterator(std::unique_ptr<SparseIterator> &&wrap, Value offset,
                 Value stride, Value size)
      : SparseIterator(IterKind::kFilter, *wrap), offset(offset),
        stride(stride), size(size), wrap(std::move(wrap)) {}

  // For LLVM-style RTTI.
  static bool classof(const SparseIterator *from) {
    return from->kind == IterKind::kFilter;
  }

  std::string getDebugInterfacePrefix() const override {
    return std::string("filter<") + wrap->getDebugInterfacePrefix() + ">";
  }
  SmallVector<Type> getCursorValTypes(OpBuilder &b) const override {
    return wrap->getCursorValTypes(b);
  }

  bool randomAccessible() const override { return wrap->randomAccessible(); };
  bool iteratableByFor() const override { return randomAccessible(); };
  Value upperBound(OpBuilder &b, Location l) const override { return size; };

  SmallVector<Value> serialize() const override { return wrap->serialize(); };
  void deserialize(ValueRange vs) override { wrap->deserialize(vs); };
  ValuePair getCurPosition() const override { return wrap->getCurPosition(); }

  void genInitImpl(OpBuilder &b, Location l,
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

  Value genNotEndImpl(OpBuilder &b, Location l) override;

  Value derefImpl(OpBuilder &b, Location l) override {
    updateCrd(fromWrapCrd(b, l, wrap->deref(b, l)));
    return getCrd();
  }

  void locateImpl(OpBuilder &b, Location l, Value crd) override {
    assert(randomAccessible());
    wrap->locate(b, l, toWrapCrd(b, l, crd));
    updateCrd(crd);
  }

  ValueRange forwardImpl(OpBuilder &b, Location l) override;

  Value offset, stride, size;
  std::unique_ptr<SparseIterator> wrap;
};

class NonEmptySubSectIterator : public SparseIterator {
public:
  using TraverseBuilder = llvm::function_ref<scf::ValueVector(
      OpBuilder &, Location, const SparseIterator *, ValueRange)>;

  NonEmptySubSectIterator(OpBuilder &b, Location l,
                          const SparseIterator *parent,
                          std::unique_ptr<SparseIterator> &&delegate,
                          Value subSectSz)
      : SparseIterator(IterKind::kNonEmptySubSect, 3, subSectMeta, *delegate),
        parent(parent), delegate(std::move(delegate)),
        tupleSz(this->delegate->serialize().size()), subSectSz(subSectSz) {
    auto *p = dyn_cast_or_null<NonEmptySubSectIterator>(parent);
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
    subSectPosBuf = allocSubSectPosBuf(b, l);
  }

  // For LLVM-style RTTI.
  static bool classof(const SparseIterator *from) {
    return from->kind == IterKind::kNonEmptySubSect;
  }

  std::string getDebugInterfacePrefix() const override {
    return std::string("ne_sub<") + delegate->getDebugInterfacePrefix() + ">";
  }
  SmallVector<Type> getCursorValTypes(OpBuilder &b) const override {
    // minCrd, absolute offset, notEnd
    return {b.getIndexType(), b.getIndexType(), b.getI1Type()};
  }

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

  void storeNxLvlStart(OpBuilder &b, Location l, Value tupleId,
                       Value start) const {
    b.create<memref::StoreOp>(l, start, subSectPosBuf,
                              ValueRange{tupleId, C_IDX(tupleSz)});
  }

  Value loadNxLvlStart(OpBuilder &b, Location l, Value tupleId) const {
    return b.create<memref::LoadOp>(l, subSectPosBuf,
                                    ValueRange{tupleId, C_IDX(tupleSz)});
  }

  void storeCursorVals(OpBuilder &b, Location l, Value tupleId,
                       ValueRange itVals) const {
    assert(itVals.size() == tupleSz);
    for (unsigned i = 0; i < tupleSz; i++) {
      b.create<memref::StoreOp>(l, itVals[i], subSectPosBuf,
                                ValueRange{tupleId, C_IDX(i)});
    }
  }

  SmallVector<Value> loadCursorVals(OpBuilder &b, Location l,
                                    Value tupleId) const {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < tupleSz; i++) {
      Value v = b.create<memref::LoadOp>(l, subSectPosBuf,
                                         ValueRange{tupleId, C_IDX(i)});
      ret.push_back(v);
    }
    return ret;
  }

  bool isSubSectRoot() const {
    return !parent || !llvm::isa<NonEmptySubSectIterator>(parent);
  }

  // Generate code that inflate the current subsection tree till the current
  // level such that every leaf node is visited.
  ValueRange inflateSubSectTree(OpBuilder &b, Location l, ValueRange reduc,
                                TraverseBuilder builder) const;

  bool randomAccessible() const override {
    return delegate->randomAccessible();
  };
  bool iteratableByFor() const override { return randomAccessible(); };
  Value upperBound(OpBuilder &b, Location l) const override {
    auto *p = dyn_cast_or_null<NonEmptySubSectIterator>(parent);
    Value parentUB =
        p && p->lvl == lvl ? p->upperBound(b, l) : delegate->upperBound(b, l);
    return ADDI(SUBI(parentUB, subSectSz), C_IDX(1));
  };

  void genInitImpl(OpBuilder &b, Location l, const SparseIterator *) override;

  void locateImpl(OpBuilder &b, Location l, Value crd) override {
    Value absOff = crd;

    if (isSubSectRoot())
      delegate->locate(b, l, absOff);
    else
      assert(parent->lvl + 1 == lvl);

    seek(ValueRange{absOff, absOff, C_TRUE});
    updateCrd(crd);
  }

  Value toSubSectCrd(OpBuilder &b, Location l, Value wrapCrd) const {
    return SUBI(wrapCrd, getAbsOff());
  }

  Value genNotEndImpl(OpBuilder &b, Location l) override {
    return getNotEnd();
  };

  Value derefImpl(OpBuilder &b, Location l) override {
    // Use the relative offset to coiterate.
    Value crd;
    auto *p = dyn_cast_or_null<NonEmptySubSectIterator>(parent);
    if (p && p->lvl == lvl)
      crd = SUBI(getAbsOff(), p->getAbsOff());
    crd = getAbsOff();

    updateCrd(crd);
    return crd;
  };

  ValueRange forwardImpl(OpBuilder &b, Location l) override;

  Value getMinCrd() const { return subSectMeta[0]; }
  Value getAbsOff() const { return subSectMeta[1]; }
  Value getNotEnd() const { return subSectMeta[2]; }

  const SparseIterator *parent;
  std::unique_ptr<SparseIterator> delegate;

  // Number of values required to serialize the wrapped iterator.
  const unsigned tupleSz;
  // Max number of tuples, and the actual number of tuple.
  Value maxTupleCnt, tupleCnt;
  // The memory used to cache the tuple serialized from the wrapped iterator.
  Value subSectPosBuf;

  const Value subSectSz;

  // minCrd, absolute offset, notEnd
  SmallVector<Value, 3> subSectMeta{nullptr, nullptr, nullptr};
};

class SubSectIterator;

// A wrapper that helps generating code to traverse a subsection, used
// by both `NonEmptySubSectIterator`and `SubSectIterator`.
struct SubSectIterHelper {
  explicit SubSectIterHelper(const SubSectIterator &iter);
  explicit SubSectIterHelper(const NonEmptySubSectIterator &subSect);

  // Delegate methods.
  void deserializeFromTupleId(OpBuilder &b, Location l, Value tupleId);
  void locate(OpBuilder &b, Location l, Value crd);
  Value genNotEnd(OpBuilder &b, Location l);
  Value deref(OpBuilder &b, Location l);
  ValueRange forward(OpBuilder &b, Location l);

  const NonEmptySubSectIterator &subSect;
  SparseIterator &wrap;
};

class SubSectIterator : public SparseIterator {
public:
  SubSectIterator(const NonEmptySubSectIterator &subSect,
                  const SparseIterator &parent,
                  std::unique_ptr<SparseIterator> &&wrap)
      : SparseIterator(IterKind::kSubSect, *wrap,
                       /*extraCursorCnt=*/wrap->randomAccessible() ? 0 : 1),
        subSect(subSect), wrap(std::move(wrap)), parent(parent), helper(*this) {
    assert(subSect.tid == tid && subSect.lvl == lvl);
    assert(parent.kind != IterKind::kSubSect || parent.lvl + 1 == lvl);
  };

  // For LLVM-style RTTI.
  static bool classof(const SparseIterator *from) {
    return from->kind == IterKind::kSubSect;
  }

  std::string getDebugInterfacePrefix() const override {
    return std::string("subsect<") + wrap->getDebugInterfacePrefix() + ">";
  }
  SmallVector<Type> getCursorValTypes(OpBuilder &b) const override {
    SmallVector<Type> ret = wrap->getCursorValTypes(b);
    if (!randomAccessible())
      ret.push_back(b.getIndexType()); // The extra counter.
    return ret;
  }

  bool randomAccessible() const override { return wrap->randomAccessible(); };
  bool iteratableByFor() const override { return randomAccessible(); };
  Value upperBound(OpBuilder &b, Location l) const override {
    return subSect.subSectSz;
  }
  std::pair<Value, Value> getCurPosition() const override {
    return wrap->getCurPosition();
  };

  Value getNxLvlTupleId(OpBuilder &b, Location l) const {
    if (randomAccessible()) {
      return ADDI(getCrd(), nxLvlTupleStart);
    };
    return ADDI(getCursor().back(), nxLvlTupleStart);
  }

  void genInitImpl(OpBuilder &b, Location l, const SparseIterator *) override {
    if (randomAccessible()) {
      if (auto *p = llvm::dyn_cast<SubSectIterator>(&parent)) {
        assert(p->lvl + 1 == lvl);
        wrap->genInit(b, l, p);
        // Linearize the dense subsection index.
        nxLvlTupleStart = MULI(subSect.subSectSz, p->getNxLvlTupleId(b, l));
      } else {
        assert(subSect.lvl == lvl && subSect.isSubSectRoot());
        wrap->deserialize(subSect.delegate->serialize());
        nxLvlTupleStart = C_IDX(0);
      }
      return;
    }
    assert(!randomAccessible());
    assert(getCursor().size() == wrap->getCursor().size() + 1);
    // Extra counter that counts the number of actually visited coordinates in
    // the sparse subsection.
    getMutCursorVals().back() = C_IDX(0);
    Value tupleId;
    if (auto *p = llvm::dyn_cast<SubSectIterator>(&parent)) {
      assert(p->lvl + 1 == lvl);
      tupleId = p->getNxLvlTupleId(b, l);
    } else {
      assert(subSect.lvl == lvl && subSect.isSubSectRoot());
      tupleId = C_IDX(0);
    }
    nxLvlTupleStart = subSect.loadNxLvlStart(b, l, tupleId);
    helper.deserializeFromTupleId(b, l, tupleId);
  }

  void locateImpl(OpBuilder &b, Location l, Value crd) override {
    helper.locate(b, l, crd);
    updateCrd(crd);
  }

  Value genNotEndImpl(OpBuilder &b, Location l) override {
    return helper.genNotEnd(b, l);
  }

  Value derefImpl(OpBuilder &b, Location l) override {
    Value crd = helper.deref(b, l);
    updateCrd(crd);
    return crd;
  };

  ValueRange forwardImpl(OpBuilder &b, Location l) override {
    helper.forward(b, l);
    assert(!randomAccessible());
    assert(getCursor().size() == wrap->getCursor().size() + 1);
    getMutCursorVals().back() = ADDI(getCursor().back(), C_IDX(1));
    return getCursor();
  };

  Value nxLvlTupleStart;

  const NonEmptySubSectIterator &subSect;
  std::unique_ptr<SparseIterator> wrap;
  const SparseIterator &parent;

  SubSectIterHelper helper;
};

} // namespace

//===----------------------------------------------------------------------===//
// SparseIterator derived classes implementation.
//===----------------------------------------------------------------------===//

void SparseIterator::genInit(OpBuilder &b, Location l,
                             const SparseIterator *p) {
  if (emitStrategy == SparseEmitStrategy::kDebugInterface) {
    std::string prefix = getDebugInterfacePrefix();
    Operation *begin = b.create(l, b.getStringAttr(prefix + ".begin"), {},
                                getCursorValTypes(b));
    seek(begin->getResults());
    return;
  }
  // TODO: support lowering to function call.
  return genInitImpl(b, l, p);
}

Value SparseIterator::genNotEnd(OpBuilder &b, Location l) {
  if (emitStrategy == SparseEmitStrategy::kDebugInterface) {
    std::string prefix = getDebugInterfacePrefix();
    Operation *notEnd = b.create(l, b.getStringAttr(prefix + ".not_end"),
                                 getCursor(), b.getI1Type());
    return notEnd->getResult(0);
  }
  // TODO: support lowering to function call.
  return genNotEndImpl(b, l);
}

void SparseIterator::locate(OpBuilder &b, Location l, Value crd) {
  if (emitStrategy == SparseEmitStrategy::kDebugInterface) {
    std::string prefix = getDebugInterfacePrefix();
    SmallVector<Value> args = getCursor();
    args.push_back(crd);
    Operation *locate = b.create(l, b.getStringAttr(prefix + ".locate"), args,
                                 getCursorValTypes(b));
    seek(locate->getResults());
    updateCrd(crd);
    return;
  }
  return locateImpl(b, l, crd);
}

Value SparseIterator::deref(OpBuilder &b, Location l) {
  if (emitStrategy == SparseEmitStrategy::kDebugInterface) {
    std::string prefix = getDebugInterfacePrefix();
    SmallVector<Value> args = getCursor();
    Operation *deref = b.create(l, b.getStringAttr(prefix + ".deref"),
                                getCursor(), b.getIndexType());
    updateCrd(deref->getResult(0));
    return getCrd();
  }
  return derefImpl(b, l);
}

ValueRange SparseIterator::forward(OpBuilder &b, Location l) {
  if (emitStrategy == SparseEmitStrategy::kDebugInterface) {
    std::string prefix = getDebugInterfacePrefix();
    Operation *next = b.create(l, b.getStringAttr(prefix + ".next"),
                               getCursor(), getCursorValTypes(b));
    seek(next->getResults());
    return getCursor();
  }
  return forwardImpl(b, l);
}

ValueRange SparseIterator::forwardIf(OpBuilder &b, Location l, Value cond) {
  auto ifOp = b.create<scf::IfOp>(l, getCursor().getTypes(), cond, true);
  // Generate else branch first, otherwise iterator values will be updated by
  // `forward()`.
  b.setInsertionPointToStart(ifOp.elseBlock());
  YIELD(getCursor());

  b.setInsertionPointToStart(ifOp.thenBlock());
  YIELD(forward(b, l));

  b.setInsertionPointAfter(ifOp);
  seek(ifOp.getResults());
  return getCursor();
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
        Value nxPos = ADDI(ivs[0], C_IDX(1));
        YIELD(nxPos);
      });
  // Return the segment high.
  return whileOp.getResult(0);
}

Value FilterIterator::genCrdNotLegitPredicate(OpBuilder &b, Location l,
                                              Value wrapCrd) {
  Value crd = fromWrapCrd(b, l, wrapCrd);
  // Test whether the coordinate is on stride.
  Value notlegit = CMPI(ne, toWrapCrd(b, l, crd), wrapCrd);
  // Test wrapCrd < offset
  notlegit = ORI(CMPI(ult, wrapCrd, offset), notlegit);
  // Test crd >= length
  notlegit = ORI(CMPI(uge, crd, size), notlegit);
  return notlegit;
}

Value FilterIterator::genShouldFilter(OpBuilder &b, Location l) {
  auto r = genWhenInBound(
      b, l, *wrap, C_FALSE,
      [this](OpBuilder &b, Location l, Value wrapCrd) -> scf::ValueVector {
        Value notLegit = genCrdNotLegitPredicate(b, l, wrapCrd);
        return {notLegit};
      });

  assert(r.size() == 1);
  return r.front();
}

Value FilterIterator::genNotEndImpl(OpBuilder &b, Location l) {
  assert(!wrap->randomAccessible());
  auto r = genWhenInBound(
      b, l, *wrap, C_FALSE,
      [this](OpBuilder &b, Location l, Value wrapCrd) -> scf::ValueVector {
        Value crd = fromWrapCrd(b, l, wrapCrd);
        // crd < size
        return {CMPI(ult, crd, size)};
      });
  assert(r.size() == 1);
  return r.front();
}

ValueRange FilterIterator::forwardImpl(OpBuilder &b, Location l) {
  assert(!randomAccessible());
  // Generates
  //
  // bool isFirst = true;
  // while !it.end() && (!legit(*it) || isFirst)
  //   wrap ++;
  //   isFirst = false;
  //
  // We do not hoist the first `wrap++` outside the loop but use a `isFirst`
  // flag here because `wrap++` might have a complex implementation (e.g., to
  // forward a subsection).
  Value isFirst = constantI1(b, l, true);

  SmallVector<Value> whileArgs(getCursor().begin(), getCursor().end());
  whileArgs.push_back(isFirst);
  auto whileOp = b.create<scf::WhileOp>(
      l, ValueRange(whileArgs).getTypes(), whileArgs,
      /*beforeBuilder=*/
      [this](OpBuilder &b, Location l, ValueRange ivs) {
        ValueRange isFirst = linkNewScope(ivs);
        assert(isFirst.size() == 1);
        scf::ValueVector cont =
            genWhenInBound(b, l, *wrap, C_FALSE,
                           [this, isFirst](OpBuilder &b, Location l,
                                           Value wrapCrd) -> scf::ValueVector {
                             // crd < size && !legit();
                             Value notLegit =
                                 genCrdNotLegitPredicate(b, l, wrapCrd);
                             Value crd = fromWrapCrd(b, l, wrapCrd);
                             Value ret = ANDI(CMPI(ult, crd, size), notLegit);
                             ret = ORI(ret, isFirst.front());
                             return {ret};
                           });
        b.create<scf::ConditionOp>(l, cont.front(), ivs);
      },
      /*afterBuilder=*/
      [this](OpBuilder &b, Location l, ValueRange ivs) {
        linkNewScope(ivs);
        wrap->forward(b, l);
        SmallVector<Value> yieldVals(getCursor().begin(), getCursor().end());
        yieldVals.push_back(constantI1(b, l, false));
        YIELD(yieldVals);
      });

  b.setInsertionPointAfter(whileOp);
  linkNewScope(whileOp.getResults());
  return getCursor();
}

SubSectIterHelper::SubSectIterHelper(const NonEmptySubSectIterator &subSect)
    : subSect(subSect), wrap(*subSect.delegate) {}

SubSectIterHelper::SubSectIterHelper(const SubSectIterator &iter)
    : subSect(iter.subSect), wrap(*iter.wrap) {}

void SubSectIterHelper::deserializeFromTupleId(OpBuilder &b, Location l,
                                               Value tupleId) {
  assert(!subSect.randomAccessible());
  wrap.deserialize(subSect.loadCursorVals(b, l, tupleId));
}

void SubSectIterHelper::locate(OpBuilder &b, Location l, Value crd) {
  Value absCrd = ADDI(crd, subSect.getAbsOff());
  wrap.locate(b, l, absCrd);
}

Value SubSectIterHelper::genNotEnd(OpBuilder &b, Location l) {
  assert(!wrap.randomAccessible());
  auto r = genWhenInBound(
      b, l, wrap, C_FALSE,
      [this](OpBuilder &b, Location l, Value wrapCrd) -> scf::ValueVector {
        Value crd = SUBI(wrapCrd, subSect.getAbsOff());
        // crd < size
        return {CMPI(ult, crd, subSect.subSectSz)};
      });
  assert(r.size() == 1);
  return r.front();
}

Value SubSectIterHelper::deref(OpBuilder &b, Location l) {
  Value wrapCrd = wrap.deref(b, l);
  Value crd = subSect.toSubSectCrd(b, l, wrapCrd);
  return crd;
}

ValueRange SubSectIterHelper::forward(OpBuilder &b, Location l) {
  return wrap.forward(b, l);
}

ValueRange NonEmptySubSectIterator::inflateSubSectTree(
    OpBuilder &b, Location l, ValueRange reduc, TraverseBuilder builder) const {
  // Set up the helper to help traverse a sparse subsection.
  SubSectIterHelper helper(*this);
  if (!randomAccessible()) {
    // The subsection tree have been expanded till the level and cached,
    // traverse all the leaves and expanded to the next level.
    SmallVector<Value> iterArgs;
    iterArgs.push_back(C_IDX(0));
    iterArgs.append(reduc.begin(), reduc.end());
    auto forEachLeaf = b.create<scf::ForOp>(
        l, /*lb=*/C_IDX(0), /*ub=*/tupleCnt, /*step=*/C_IDX(1), iterArgs,
        [&helper, &builder](OpBuilder &b, Location l, Value tupleId,
                            ValueRange iterArgs) {
          // Deserialize the iterator at the cached position (tupleId).
          helper.deserializeFromTupleId(b, l, tupleId);

          Value cnt = iterArgs.front();
          // Record the number of leaf nodes included in the subsection.
          // The number indicates the starting tupleId for the next level that
          // is corresponding to the current node.
          helper.subSect.storeNxLvlStart(b, l, tupleId, cnt);

          SmallVector<Value> whileArgs(helper.wrap.getCursor());
          whileArgs.append(iterArgs.begin(), iterArgs.end());

          auto whileOp = b.create<scf::WhileOp>(
              l, ValueRange(whileArgs).getTypes(), whileArgs,
              /*beforeBuilder=*/
              [&helper](OpBuilder &b, Location l, ValueRange ivs) {
                helper.wrap.linkNewScope(ivs);
                b.create<scf::ConditionOp>(l, helper.genNotEnd(b, l), ivs);
              },
              /*afterBuilder=*/
              [&helper, &builder](OpBuilder &b, Location l, ValueRange ivs) {
                ValueRange remIter = helper.wrap.linkNewScope(ivs);
                Value cnt = remIter.front();
                ValueRange userIter = remIter.drop_front();
                scf::ValueVector userNx = builder(b, l, &helper.wrap, userIter);

                SmallVector<Value> nxIter = helper.forward(b, l);
                nxIter.push_back(ADDI(cnt, C_IDX(1)));
                nxIter.append(userNx.begin(), userNx.end());
                YIELD(nxIter);
              });
          ValueRange res = helper.wrap.linkNewScope(whileOp.getResults());
          YIELD(res);
        });
    return forEachLeaf.getResults().drop_front();
  }

  assert(randomAccessible());
  // Helper lambda that traverse the current dense subsection range.
  auto visitDenseSubSect = [&, this](OpBuilder &b, Location l,
                                     const SparseIterator *parent,
                                     ValueRange reduc) {
    assert(!parent || parent->lvl + 1 == lvl);
    delegate->genInit(b, l, parent);
    auto forOp = b.create<scf::ForOp>(
        l, /*lb=*/C_IDX(0), /*ub=*/subSectSz, /*step=*/C_IDX(1), reduc,
        [&](OpBuilder &b, Location l, Value crd, ValueRange iterArgs) {
          helper.locate(b, l, crd);
          scf::ValueVector nx = builder(b, l, &helper.wrap, iterArgs);
          YIELD(nx);
        });
    return forOp.getResults();
  };

  if (isSubSectRoot()) {
    return visitDenseSubSect(b, l, parent, reduc);
  }
  // Else, this is not the root, recurse until root.
  auto *p = llvm::cast<NonEmptySubSectIterator>(parent);
  assert(p->lvl + 1 == lvl);
  return p->inflateSubSectTree(b, l, reduc, visitDenseSubSect);
}

void NonEmptySubSectIterator::genInitImpl(OpBuilder &b, Location l,
                                          const SparseIterator *) {
  Value c0 = C_IDX(0);
  if (!isSubSectRoot()) {
    assert(parent->lvl + 1 == lvl);
    if (randomAccessible()) {
      // We can not call wrap->genInit() here to initialize the wrapped
      // iterator, because the parent of the curent iterator is still
      // unresolved.
      seek({/*minCrd=*/c0, /*offset=*/c0, /*notEnd=*/C_TRUE});
      return;
    }

    auto *p = cast<NonEmptySubSectIterator>(parent);
    SmallVector<Value, 3> reduc = {
        C_IDX(-1), // minCrd (max signless integer)
        c0,        // tupleId
    };

    // Expand the subsection tree from the parent level to the current level.
    ValueRange result = p->inflateSubSectTree(
        b, l, reduc,
        [this](OpBuilder &b, Location l, const SparseIterator *parent,
               ValueRange reduc) -> scf::ValueVector {
          assert(parent->lvl + 1 == lvl && reduc.size() == 2);
          Value minCrd = reduc.front();
          Value tupleId = reduc.back();

          // Initialize the subsection range.
          SubSectIterHelper helper(*this);
          helper.wrap.genInit(b, l, parent);

          // Update minCrd.
          minCrd = genWhenInBound(b, l, helper.wrap, minCrd,
                                  [minCrd](OpBuilder &b, Location l,
                                           Value crd) -> scf::ValueVector {
                                    Value min = MINUI(crd, minCrd);
                                    return {min};
                                  })
                       .front();

          // Cache the sparse range.
          storeCursorVals(b, l, tupleId, helper.wrap.serialize());
          tupleId = ADDI(tupleId, C_IDX(1));
          return {minCrd, tupleId};
        });
    assert(result.size() == 2);
    tupleCnt = result.back();

    Value minCrd = result.front();
    Value absOff = offsetFromMinCrd(b, l, minCrd, subSectSz);
    Value notEnd = CMPI(ne, minCrd, C_IDX(-1));
    seek({minCrd, absOff, notEnd});
    return;
  }

  // This is the root level of the subsection, which means that it is resolved
  // to one node.
  assert(isSubSectRoot());

  // Initialize the position, the position marks the *lower bound* of the
  // subRange. The higher bound is determined by the size of the subsection.
  delegate->genInit(b, l, parent);
  if (randomAccessible()) {
    seek({/*minCrd=*/c0, /*offset=*/c0, /*notEnd=*/C_TRUE});
    return;
  }

  // Only have one root node.
  tupleCnt = C_IDX(1);
  // Cache the sparse range.
  storeCursorVals(b, l, c0, delegate->serialize());
  SmallVector<Value> elseRet{c0, c0, /*notEnd=*/C_FALSE};
  auto meta = genWhenInBound(
      b, l, *delegate, elseRet,
      [this](OpBuilder &b, Location l, Value crd) -> scf::ValueVector {
        Value offset = offsetFromMinCrd(b, l, crd, subSectSz);
        return {crd, offset, C_TRUE};
      });

  seek(meta);
}

ValueRange NonEmptySubSectIterator::forwardImpl(OpBuilder &b, Location l) {
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
  auto ifOp = b.create<scf::IfOp>(l, getCursor().getTypes(), fastPathP, true);
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
    SmallVector<Value, 2> loopArgs{C_IDX(-1), // nextMinCrd
                                   C_FALSE};  // isNotEnd
    auto loopNest = scf::buildLoopNest(
        b, l, c0, tupleCnt, c1, loopArgs,
        [this](OpBuilder &b, Location l, ValueRange ivs,
               ValueRange iterArgs) -> scf::ValueVector {
          Value tupleId = ivs.front();
          SubSectIterHelper helper(*this);
          helper.deserializeFromTupleId(b, l, tupleId);

          return genWhenInBound(
              b, l, *delegate, /*elseRet=*/iterArgs,
              [this, iterArgs, tupleId](OpBuilder &b, Location l,
                                        Value crd) -> scf::ValueVector {
                // if coord == minCrd
                //   wrap->forward();
                Value isMin = CMPI(eq, crd, getMinCrd());
                delegate->forwardIf(b, l, isMin);
                // Update the forwarded iterator values if needed.
                auto ifIsMin = b.create<scf::IfOp>(l, isMin, false);
                b.setInsertionPointToStart(&ifIsMin.getThenRegion().front());
                storeCursorVals(b, l, tupleId, delegate->serialize());
                b.setInsertionPointAfter(ifIsMin);
                // if (!wrap.end())
                //  yield(min(nxMinCrd, *wrap), true)
                Value nxMin = iterArgs[0];
                return genWhenInBound(b, l, *delegate, /*elseRet=*/iterArgs,
                                      [nxMin](OpBuilder &b, Location l,
                                              Value crd) -> scf::ValueVector {
                                        Value nx = b.create<arith::MinUIOp>(
                                            l, crd, nxMin);
                                        return {nx, C_TRUE};
                                      });
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
  nxAbsOff = b.create<arith::MaxUIOp>(l, minAbsOff, nxAbsOff);

  seek(ValueRange{nxMinCrd, nxAbsOff, nxNotEnd});
  // The coordinate should not exceeds the space upper bound.
  Value crd = deref(b, l);
  nxNotEnd = ANDI(nxNotEnd, CMPI(ult, crd, upperBound(b, l)));

  seek(ValueRange{nxMinCrd, nxAbsOff, nxNotEnd});
  return getCursor();
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

  switch (lt.getLvlFmt()) {
  case LevelFormat::Dense:
    return std::make_unique<DenseLevel>(tid, lvl, sz, stt.hasEncoding());
  case LevelFormat::Batch:
    llvm_unreachable("not implemented");
  case LevelFormat::Compressed: {
    Value pos = b.create<ToPositionsOp>(l, t, lvl);
    Value crd = b.create<ToCoordinatesOp>(l, t, lvl);
    return std::make_unique<CompressedLevel>(tid, lvl, lt, sz, pos, crd);
  }
  case LevelFormat::LooseCompressed: {
    Value pos = b.create<ToPositionsOp>(l, t, lvl);
    Value crd = b.create<ToCoordinatesOp>(l, t, lvl);
    return std::make_unique<LooseCompressedLevel>(tid, lvl, lt, sz, pos, crd);
  }
  case LevelFormat::Singleton: {
    Value crd = b.create<ToCoordinatesOp>(l, t, lvl);
    return std::make_unique<SingletonLevel>(tid, lvl, lt, sz, crd);
  }
  case LevelFormat::NOutOfM: {
    Value crd = b.create<ToCoordinatesOp>(l, t, lvl);
    return std::make_unique<NOutOfMLevel>(tid, lvl, lt, sz, crd);
  }
  case LevelFormat::Undef:
    llvm_unreachable("undefined level format");
  }
  llvm_unreachable("unrecognizable level format");
}

std::pair<std::unique_ptr<SparseTensorLevel>, std::unique_ptr<SparseIterator>>
sparse_tensor::makeSynLevelAndIterator(Value sz, unsigned tid, unsigned lvl,
                                       SparseEmitStrategy strategy) {
  auto stl = std::make_unique<DenseLevel>(tid, lvl, sz, /*encoded=*/false);
  auto it = std::make_unique<TrivialIterator>(*stl);
  it->setSparseEmitStrategy(strategy);
  return std::make_pair(std::move(stl), std::move(it));
}

std::unique_ptr<SparseIterator>
sparse_tensor::makeSimpleIterator(const SparseTensorLevel &stl,
                                  SparseEmitStrategy strategy) {
  std::unique_ptr<SparseIterator> ret;
  if (!isUniqueLT(stl.getLT())) {
    // We always dedupliate the non-unique level, but we should optimize it away
    // if possible.
    ret = std::make_unique<DedupIterator>(stl);
  } else {
    ret = std::make_unique<TrivialIterator>(stl);
  }
  ret->setSparseEmitStrategy(strategy);
  return ret;
}

std::unique_ptr<SparseIterator>
sparse_tensor::makeSlicedLevelIterator(std::unique_ptr<SparseIterator> &&sit,
                                       Value offset, Value stride, Value size,
                                       SparseEmitStrategy strategy) {

  auto ret =
      std::make_unique<FilterIterator>(std::move(sit), offset, stride, size);
  ret->setSparseEmitStrategy(strategy);
  return ret;
}

static const SparseIterator *tryUnwrapFilter(const SparseIterator *it) {
  auto *filter = llvm::dyn_cast_or_null<FilterIterator>(it);
  if (filter)
    return filter->wrap.get();
  return it;
}

std::unique_ptr<SparseIterator> sparse_tensor::makeNonEmptySubSectIterator(
    OpBuilder &b, Location l, const SparseIterator *parent, Value loopBound,
    std::unique_ptr<SparseIterator> &&delegate, Value size, unsigned stride,
    SparseEmitStrategy strategy) {

  // Try unwrap the NonEmptySubSectIterator from a filter parent.
  parent = tryUnwrapFilter(parent);
  std::unique_ptr<SparseIterator> it =
      std::make_unique<NonEmptySubSectIterator>(b, l, parent,
                                                std::move(delegate), size);

  if (stride != 1) {
    // TODO: We can safely skip bound checking on sparse levels, but for dense
    // iteration space, we need the bound to infer the dense loop range.
    it = std::make_unique<FilterIterator>(std::move(it), /*offset=*/C_IDX(0),
                                          C_IDX(stride), /*size=*/loopBound);
  }
  it->setSparseEmitStrategy(strategy);
  return it;
}

std::unique_ptr<SparseIterator> sparse_tensor::makeTraverseSubSectIterator(
    OpBuilder &b, Location l, const SparseIterator &subSectIter,
    const SparseIterator &parent, std::unique_ptr<SparseIterator> &&wrap,
    Value loopBound, unsigned stride, SparseEmitStrategy strategy) {

  // This must be a subsection iterator or a filtered subsection iterator.
  auto &subSect =
      llvm::cast<NonEmptySubSectIterator>(*tryUnwrapFilter(&subSectIter));

  std::unique_ptr<SparseIterator> it = std::make_unique<SubSectIterator>(
      subSect, *tryUnwrapFilter(&parent), std::move(wrap));

  if (stride != 1) {
    it = std::make_unique<FilterIterator>(std::move(it), /*offset=*/C_IDX(0),
                                          C_IDX(stride), /*size=*/loopBound);
  }
  it->setSparseEmitStrategy(strategy);
  return it;
}

#undef CMPI
#undef C_IDX
#undef YIELD
#undef ADDI
#undef ANDI
#undef SUBI
#undef MULI
#undef SELECT
