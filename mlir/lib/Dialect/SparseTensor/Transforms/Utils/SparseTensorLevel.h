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

/// The base class for all types of sparse tensor levels. It provides interfaces
/// to query the loop range (see `peekRangeAt`) and look up the coordinates (see
/// `peekCrdAt`).
class SparseTensorLevel {
  SparseTensorLevel(SparseTensorLevel &&) = delete;
  SparseTensorLevel(const SparseTensorLevel &) = delete;
  SparseTensorLevel &operator=(SparseTensorLevel &&) = delete;
  SparseTensorLevel &operator=(const SparseTensorLevel &) = delete;

public:
  virtual ~SparseTensorLevel() = default;

  virtual Value peekCrdAt(OpBuilder &b, Location l, Value iv) const = 0;

  /// Peeks the lower and upper bound to *fully* traverse the level with
  /// the given position `p` that the immediate parent level is current at.
  /// Returns a pair of values for *posLo* and *loopHi* respectively.
  ///
  /// For a dense level, the *posLo* is the linearized position at beginning,
  /// while *loopHi* is the largest *coordinate*, it also implies that the
  /// smallest *coordinate* to start the loop is 0.
  ///
  /// For a sparse level, [posLo, loopHi) specifies the range of index pointer
  /// to load coordinate from the coordinate buffer.
  ///
  /// `bound` is only used when the level is `non-unique` and deduplication is
  /// required. It specifies the max upper bound of the non-unique segment.
  virtual std::pair<Value, Value> peekRangeAt(OpBuilder &b, Location l, Value p,
                                              Value segHi = Value()) const = 0;

  Level getLevel() const { return lvl; }
  LevelType getLT() const { return lt; }
  Value size() const { return lvlSize; }

  //
  // Level properties
  //
  bool isUnique() const { return isUniqueLT(lt); }

protected:
  SparseTensorLevel(unsigned tid, unsigned lvl, LevelType lt, Value lvlSize)
      : tid(tid), lvl(lvl), lt(lt), lvlSize(lvlSize){};

public:
  const unsigned tid, lvl;
  const LevelType lt;
  const Value lvlSize;
};

enum class IterKind : uint8_t {
  kTrivial,
  kDedup,
  kSubSect,
  kNonEmptySubSect,
  kFilter,
};

/// Helper class that generates loop conditions, etc, to traverse a
/// sparse tensor level.
class SparseIterator {
  SparseIterator(SparseIterator &&) = delete;
  SparseIterator(const SparseIterator &) = delete;
  SparseIterator &operator=(SparseIterator &&) = delete;
  SparseIterator &operator=(const SparseIterator &) = delete;

protected:
  SparseIterator(IterKind kind, unsigned tid, unsigned lvl, unsigned itValsCnt,
                 SmallVectorImpl<Value> &storage)
      : kind(kind), tid(tid), lvl(lvl), crd(nullptr), itValsCnt(itValsCnt),
        itValsStorageRef(storage){};

  SparseIterator(IterKind kind, const SparseIterator &wrap)
      : kind(kind), tid(wrap.tid), lvl(wrap.lvl), crd(nullptr),
        itValsCnt(wrap.itValsCnt), itValsStorageRef(wrap.itValsStorageRef) {
    assert(wrap.itValsCnt == itValsStorageRef.size());
  };

  SparseIterator(IterKind kind, const SparseIterator &wrap, unsigned extraVal)
      : kind(kind), tid(wrap.tid), lvl(wrap.lvl), crd(nullptr),
        itValsCnt(wrap.itValsCnt + extraVal),
        itValsStorageRef(wrap.itValsStorageRef) {
    itValsStorageRef.append(extraVal, nullptr);
    assert(itValsCnt == itValsStorageRef.size());
  };

public:
  virtual ~SparseIterator() = default;

  Value getCrd() const { return crd; }
  ValueRange getItVals() const {
    return ValueRange(itValsStorageRef).take_front(itValsCnt);
  };

  // Sets the iterate to the specified position.
  void seek(ValueRange vals) {
    assert(vals.size() == itValsCnt);
    std::copy(vals.begin(), vals.end(), itValsStorageRef.begin());
    // Now that the iterator is re-positioned, the coordinate becomes invalid.
    crd = nullptr;
  }

  //
  // Iterator properties.
  //

  // Whether the iterator support random access (i.e., support look up by
  // *coordinate*). A random access iterator must also traverses a dense space.
  virtual bool randomAccessible() const = 0;

  // Whether the iterator can simply traversed by a for loop.
  virtual bool iteratableByFor() const { return false; };

  // Get the upper bound of the sparse space that the iterator might visited. A
  // sparse space is a subset of a dense space [0, bound), this function returns
  // *bound*.
  virtual Value upperBound(OpBuilder &b, Location l) const = 0;

  // Serializes and deserializes the current status to/from a set of values. The
  // ValueRange should contain values that are sufficient to recover the current
  // iterating postion (i.e., itVals) as well as loop bound.
  //
  // Not every type of iterator supports the operations, e.g., non-empty
  // subsection iterator does not because the the number of non-empty
  // subsections can not be determined easily.
  //
  // NOTE: All the values should have index type.
  virtual SmallVector<Value> serialize() const {
    llvm_unreachable("unsupported");
  };
  virtual void deserialize(ValueRange vs) { llvm_unreachable("unsupported"); };

  //
  // Core functions.
  //

  // Initializes the iterator according to the parent iterator's state.
  void genInit(OpBuilder &b, Location l, const SparseIterator *p);

  // Forwards the iterator to the next element.
  ValueRange forward(OpBuilder &b, Location l);

  // Actual Implementation provided by derived class.
  virtual void genInitImpl(OpBuilder &, Location, const SparseIterator *) = 0;
  virtual ValueRange forwardImpl(OpBuilder &b, Location l) = 0;

  // Returns a boolean value that equals `!it.end()`
  virtual Value genNotEnd(OpBuilder &b, Location l) = 0;

  // Dereferences the iterator, loads the coordinate at the current position.
  //
  // The method assumes that the iterator is not currently exhausted (i.e.,
  // it != it.end()).
  virtual Value deref(OpBuilder &b, Location l) = 0;

  // Gets the current position and the optional *position high* (for non-unique
  // iterators), the value is essentially the number of sparse coordinate that
  // the iterator is current visiting. It should be able to uniquely identify
  // the sparse range for the next level. See SparseTensorLevel::peekRangeAt();
  //
  // Not every type of iterator supports the operation, e.g., non-empty
  // subsection iterator does not because it represent a range of coordinates
  // instead of just one.
  virtual std::pair<Value, Value> getCurPosition() const {
    llvm_unreachable("unsupported");
  };

  // Returns a pair of values for *upper*, *lower* bound respectively.
  virtual std::pair<Value, Value> genForCond(OpBuilder &b, Location l) {
    assert(randomAccessible());
    // Random-access iterator is traversed by coordinate, i.e., [curCrd, UB).
    return {getCrd(), upperBound(b, l)};
  }

  // Generates a bool value for scf::ConditionOp.
  std::pair<Value, ValueRange> genWhileCond(OpBuilder &b, Location l,
                                            ValueRange vs) {
    ValueRange rem = linkNewScope(vs);
    return std::make_pair(genNotEnd(b, l), rem);
  }

  // Generate a conditional it.next() in the following form
  //
  // if (cond)
  //    yield it.next
  // else
  //    yield it
  //
  // The function is virtual to allow alternative implementation. For example,
  // if it.next() is trivial to compute, we can use a select operation instead.
  // E.g.,
  //
  //  it = select cond ? it+1 : it
  virtual ValueRange forwardIf(OpBuilder &b, Location l, Value cond);

  // Locate the iterator to the position specified by *crd*, this can only
  // be done on an iterator that supports randm access.
  virtual void locate(OpBuilder &b, Location l, Value crd) {
    llvm_unreachable("Unsupported");
  }

  // Update the SSA value for the iterator after entering a new scope.
  ValueRange linkNewScope(ValueRange pos) {
    assert(!randomAccessible() && "random accessible iterators are traversed "
                                  "by coordinate, call locate() instead.");
    seek(pos.take_front(itValsCnt));
    return pos.drop_front(itValsCnt);
  };

protected:
  void updateCrd(Value crd) { this->crd = crd; }
  MutableArrayRef<Value> getMutItVals() {
    MutableArrayRef<Value> ref = itValsStorageRef;
    return ref.take_front(itValsCnt);
  }

public:
  const IterKind kind;     // For LLVM-style RTTI.
  const unsigned tid, lvl; // tensor level identifier.

private:
  Value crd; // The sparse coordinate used to coiterate;

  // A range of value that together defines the current state of the
  // iterator. Only loop variants should be included.
  //
  // For trivial iterators, it is the position; for dedup iterators, it consists
  // of the positon and the segment high, for non-empty subsection iterator, it
  // is the metadata that specifies the subsection.
  // Note that the wrapped iterator shares the same storage to maintain itVals
  // with it wrapper, which means the wrapped iterator might only own a subset
  // of all the values stored in itValStorage.
  const unsigned itValsCnt;
  SmallVectorImpl<Value> &itValsStorageRef;
};

/// Helper function to create a TensorLevel object from given `tensor`.
std::unique_ptr<SparseTensorLevel> makeSparseTensorLevel(OpBuilder &builder,
                                                         Location loc, Value t,
                                                         unsigned tid, Level l);

/// Helper function to create a simple SparseIterator object that iterate over
/// the SparseTensorLevel.
std::unique_ptr<SparseIterator>
makeSimpleIterator(const SparseTensorLevel &stl);

/// Helper function to create a synthetic SparseIterator object that iterate
/// over a dense space specified by [0,`sz`).
std::pair<std::unique_ptr<SparseTensorLevel>, std::unique_ptr<SparseIterator>>
makeSynLevelAndIterator(Value sz, unsigned tid, unsigned lvl);

/// Helper function to create a SparseIterator object that iterate over a
/// sliced space, the orignal space (before slicing) is traversed by `sit`.
std::unique_ptr<SparseIterator>
makeSlicedLevelIterator(std::unique_ptr<SparseIterator> &&sit, Value offset,
                        Value stride, Value size);

/// Helper function to create a SparseIterator object that iterate over the
/// non-empty subsections set.
std::unique_ptr<SparseIterator> makeNonEmptySubSectIterator(
    OpBuilder &b, Location l, const SparseIterator *parent, Value loopBound,
    std::unique_ptr<SparseIterator> &&delegate, Value size, unsigned stride);

/// Helper function to create a SparseIterator object that iterate over a
/// non-empty subsection created by NonEmptySubSectIterator.
std::unique_ptr<SparseIterator> makeTraverseSubSectIterator(
    const SparseIterator &subsectIter, const SparseIterator &parent,
    std::unique_ptr<SparseIterator> &&delegate, Value size, unsigned stride);

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_SPARSETENSORLEVEL_H_
