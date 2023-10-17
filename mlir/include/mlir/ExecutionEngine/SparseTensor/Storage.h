//===- Storage.h - TACO-flavored sparse tensor representation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the following classes:
//
// * `SparseTensorStorageBase`
// * `SparseTensorStorage<P, C, V>`
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_STORAGE_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_STORAGE_H

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/ExecutionEngine/SparseTensor/ArithmeticUtils.h"
#include "mlir/ExecutionEngine/SparseTensor/COO.h"
#include "mlir/ExecutionEngine/SparseTensor/ErrorHandling.h"
#include "mlir/ExecutionEngine/SparseTensor/MapRef.h"

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
//
//  SparseTensorStorage Classes
//
//===----------------------------------------------------------------------===//

/// Abstract base class for `SparseTensorStorage<P,C,V>`. This class
/// takes responsibility for all the `<P,C,V>`-independent aspects
/// of the tensor (e.g., sizes, sparsity, mapping). In addition,
/// we use function overloading to implement "partial" method
/// specialization, which the C-API relies on to catch type errors
/// arising from our use of opaque pointers.
///
/// Because this class forms a bridge between the denotational semantics
/// of "tensors" and the operational semantics of how we store and
/// compute with them, it also distinguishes between two different
/// coordinate spaces (and their associated rank, sizes, etc).
/// Denotationally, we have the *dimensions* of the tensor represented
/// by this object.  Operationally, we have the *levels* of the storage
/// representation itself.
///
/// The *size* of an axis is the cardinality of possible coordinate
/// values along that axis (regardless of which coordinates have stored
/// element values). As such, each size must be non-zero since if any
/// axis has size-zero then the whole tensor would have trivial storage
/// (since there are no possible coordinates). Thus we use the plural
/// term *sizes* for a collection of non-zero cardinalities, and use
/// this term whenever referring to run-time cardinalities. Whereas we
/// use the term *shape* for a collection of compile-time cardinalities,
/// where zero is used to indicate cardinalities which are dynamic (i.e.,
/// unknown/unspecified at compile-time). At run-time, these dynamic
/// cardinalities will be inferred from or checked against sizes otherwise
/// specified. Thus, dynamic cardinalities always have an "immutable but
/// unknown" value; so the term "dynamic" should not be taken to indicate
/// run-time mutability.
class SparseTensorStorageBase {
protected:
  SparseTensorStorageBase(const SparseTensorStorageBase &) = default;
  SparseTensorStorageBase &operator=(const SparseTensorStorageBase &) = delete;

public:
  /// Constructs a new sparse-tensor storage object with the given encoding.
  SparseTensorStorageBase(uint64_t dimRank, const uint64_t *dimSizes,
                          uint64_t lvlRank, const uint64_t *lvlSizes,
                          const DimLevelType *lvlTypes, const uint64_t *dim2lvl,
                          const uint64_t *lvl2dim);
  virtual ~SparseTensorStorageBase() = default;

  /// Gets the number of tensor-dimensions.
  uint64_t getDimRank() const { return dimSizes.size(); }

  /// Gets the number of storage-levels.
  uint64_t getLvlRank() const { return lvlSizes.size(); }

  /// Gets the tensor-dimension sizes array.
  const std::vector<uint64_t> &getDimSizes() const { return dimSizes; }

  /// Safely looks up the size of the given tensor-dimension.
  uint64_t getDimSize(uint64_t d) const {
    assert(d < getDimRank());
    return dimSizes[d];
  }

  /// Gets the storage-level sizes array.
  const std::vector<uint64_t> &getLvlSizes() const { return lvlSizes; }

  /// Safely looks up the size of the given storage-level.
  uint64_t getLvlSize(uint64_t l) const {
    assert(l < getLvlRank());
    return lvlSizes[l];
  }

  /// Gets the level-types array.
  const std::vector<DimLevelType> &getLvlTypes() const { return lvlTypes; }

  /// Safely looks up the type of the given level.
  DimLevelType getLvlType(uint64_t l) const {
    assert(l < getLvlRank());
    return lvlTypes[l];
  }

  /// Safely checks if the level uses dense storage.
  bool isDenseLvl(uint64_t l) const { return isDenseDLT(getLvlType(l)); }

  /// Safely checks if the level uses compressed storage.
  bool isCompressedLvl(uint64_t l) const {
    return isCompressedDLT(getLvlType(l));
  }

  /// Safely checks if the level uses singleton storage.
  bool isSingletonLvl(uint64_t l) const {
    return isSingletonDLT(getLvlType(l));
  }

  /// Safely checks if the level is ordered.
  bool isOrderedLvl(uint64_t l) const { return isOrderedDLT(getLvlType(l)); }

  /// Safely checks if the level is unique.
  bool isUniqueLvl(uint64_t l) const { return isUniqueDLT(getLvlType(l)); }

  /// Gets positions-overhead storage for the given level.
#define DECL_GETPOSITIONS(PNAME, P)                                            \
  virtual void getPositions(std::vector<P> **, uint64_t);
  MLIR_SPARSETENSOR_FOREVERY_FIXED_O(DECL_GETPOSITIONS)
#undef DECL_GETPOSITIONS

  /// Gets coordinates-overhead storage for the given level.
#define DECL_GETCOORDINATES(INAME, C)                                          \
  virtual void getCoordinates(std::vector<C> **, uint64_t);
  MLIR_SPARSETENSOR_FOREVERY_FIXED_O(DECL_GETCOORDINATES)
#undef DECL_GETCOORDINATES

  /// Gets the coordinate-value stored at the given level and position.
  virtual uint64_t getCrd(uint64_t lvl, uint64_t pos) const = 0;

  /// Gets primary storage.
#define DECL_GETVALUES(VNAME, V) virtual void getValues(std::vector<V> **);
  MLIR_SPARSETENSOR_FOREVERY_V(DECL_GETVALUES)
#undef DECL_GETVALUES

  /// Element-wise forwarding insertions. The first argument is the
  /// dimension-coordinates for the value being inserted.
#define DECL_FORWARDINGINSERT(VNAME, V)                                        \
  virtual void forwardingInsert(const uint64_t *, V);
  MLIR_SPARSETENSOR_FOREVERY_V(DECL_FORWARDINGINSERT)
#undef DECL_FORWARDINGINSERT

  /// Element-wise insertion in lexicographic coordinate order. The first
  /// argument is the level-coordinates for the value being inserted.
#define DECL_LEXINSERT(VNAME, V) virtual void lexInsert(const uint64_t *, V);
  MLIR_SPARSETENSOR_FOREVERY_V(DECL_LEXINSERT)
#undef DECL_LEXINSERT

  /// Expanded insertion.  Note that this method resets the
  /// values/filled-switch array back to all-zero/false while only
  /// iterating over the nonzero elements.
#define DECL_EXPINSERT(VNAME, V)                                               \
  virtual void expInsert(uint64_t *, V *, bool *, uint64_t *, uint64_t,        \
                         uint64_t);
  MLIR_SPARSETENSOR_FOREVERY_V(DECL_EXPINSERT)
#undef DECL_EXPINSERT

  /// Finalizes forwarding insertions.
  virtual void endForwardingInsert() = 0;

  /// Finalizes lexicographic insertions.
  virtual void endLexInsert() = 0;

private:
  const std::vector<uint64_t> dimSizes;
  const std::vector<uint64_t> lvlSizes;
  const std::vector<DimLevelType> lvlTypes;
  const std::vector<uint64_t> dim2lvlVec;
  const std::vector<uint64_t> lvl2dimVec;

protected:
  const MapRef map; // non-owning pointers into dim2lvl/lvl2dim vectors
};

/// A memory-resident sparse tensor using a storage scheme based on
/// per-level sparse/dense annotations. This data structure provides
/// a bufferized form of a sparse tensor type. In contrast to generating
/// setup methods for each differently annotated sparse tensor, this
/// method provides a convenient "one-size-fits-all" solution that simply
/// takes an input tensor and annotations to implement all required setup
/// in a general manner.
template <typename P, typename C, typename V>
class SparseTensorStorage final : public SparseTensorStorageBase {
  /// Private constructor to share code between the other constructors.
  /// Beware that the object is not necessarily guaranteed to be in a
  /// valid state after this constructor alone; e.g., `isCompressedLvl(l)`
  /// doesn't entail `!(positions[l].empty())`.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const uint64_t *lvlSizes,
                      const DimLevelType *lvlTypes, const uint64_t *dim2lvl,
                      const uint64_t *lvl2dim)
      : SparseTensorStorageBase(dimRank, dimSizes, lvlRank, lvlSizes, lvlTypes,
                                dim2lvl, lvl2dim),
        positions(lvlRank), coordinates(lvlRank), lvlCursor(lvlRank), coo() {}

public:
  /// Constructs a sparse tensor with the given encoding, and allocates
  /// overhead storage according to some simple heuristics. When the
  /// `bool` argument is true and `lvlTypes` are all dense, then this
  /// ctor will also initialize the values array with zeros. That
  /// argument should be true when an empty tensor is intended; whereas
  /// it should usually be false when the ctor will be followed up by
  /// some other form of initialization.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const uint64_t *lvlSizes,
                      const DimLevelType *lvlTypes, const uint64_t *dim2lvl,
                      const uint64_t *lvl2dim, SparseTensorCOO<V> *lvlCOO,
                      bool initializeValuesIfAllDense);

  /// Constructs a sparse tensor with the given encoding, and initializes
  /// the contents from the COO. This ctor performs the same heuristic
  /// overhead-storage allocation as the ctor above.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const uint64_t *lvlSizes,
                      const DimLevelType *lvlTypes, const uint64_t *dim2lvl,
                      const uint64_t *lvl2dim, SparseTensorCOO<V> &lvlCOO);

  /// Constructs a sparse tensor with the given encoding, and initializes
  /// the contents from the level buffers. This ctor allocates exactly
  /// the required amount of overhead storage, not using any heuristics.
  /// It assumes that the data provided by `lvlBufs` can be directly used to
  /// interpret the result sparse tensor and performs *NO* integrity test on the
  /// input data. It also assume that the trailing COO coordinate buffer is
  /// passed in as a single AoS memory.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const uint64_t *lvlSizes,
                      const DimLevelType *lvlTypes, const uint64_t *dim2lvl,
                      const uint64_t *lvl2dim, const intptr_t *lvlBufs);

  /// Allocates a new empty sparse tensor.
  static SparseTensorStorage<P, C, V> *
  newEmpty(uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
           const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
           const uint64_t *dim2lvl, const uint64_t *lvl2dim, bool forwarding);

  /// Allocates a new sparse tensor and initializes it from the given COO.
  static SparseTensorStorage<P, C, V> *
  newFromCOO(uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
             const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
             const uint64_t *dim2lvl, const uint64_t *lvl2dim,
             SparseTensorCOO<V> &lvlCOO);

  /// Allocates a new sparse tensor and initialize it with the data stored level
  /// buffers directly.
  static SparseTensorStorage<P, C, V> *
  packFromLvlBuffers(uint64_t dimRank, const uint64_t *dimSizes,
                     uint64_t lvlRank, const uint64_t *lvlSizes,
                     const DimLevelType *lvlTypes, const uint64_t *dim2lvl,
                     const uint64_t *lvl2dim, uint64_t srcRank,
                     const intptr_t *buffers);

  ~SparseTensorStorage() final = default;

  /// Partially specialize these getter methods based on template types.
  void getPositions(std::vector<P> **out, uint64_t lvl) final {
    assert(out && "Received nullptr for out parameter");
    assert(lvl < getLvlRank());
    *out = &positions[lvl];
  }
  void getCoordinates(std::vector<C> **out, uint64_t lvl) final {
    assert(out && "Received nullptr for out parameter");
    assert(lvl < getLvlRank());
    *out = &coordinates[lvl];
  }
  void getValues(std::vector<V> **out) final {
    assert(out && "Received nullptr for out parameter");
    *out = &values;
  }

  /// Returns coordinate at given position.
  uint64_t getCrd(uint64_t lvl, uint64_t pos) const final {
    assert(isCompressedDLT(getLvlType(lvl)) || isSingletonDLT(getLvlType(lvl)));
    assert(pos < coordinates[lvl].size());
    return coordinates[lvl][pos]; // Converts the stored `C` into `uint64_t`.
  }

  /// Partially specialize forwarding insertions based on template types.
  void forwardingInsert(const uint64_t *dimCoords, V val) final {
    assert(dimCoords && coo);
    map.pushforward(dimCoords, lvlCursor.data());
    coo->add(lvlCursor, val);
  }

  /// Partially specialize lexicographical insertions based on template types.
  void lexInsert(const uint64_t *lvlCoords, V val) final {
    assert(lvlCoords);
    bool allDense = std::all_of(getLvlTypes().begin(), getLvlTypes().end(),
                                [](DimLevelType lt) { return isDenseDLT(lt); });
    if (allDense) {
      uint64_t lvlRank = getLvlRank();
      uint64_t valIdx = 0;
      // Linearize the address
      for (size_t lvl = 0; lvl < lvlRank; lvl++)
        valIdx = valIdx * getLvlSize(lvl) + lvlCoords[lvl];
      values[valIdx] = val;
      return;
    }
    // First, wrap up pending insertion path.
    uint64_t diffLvl = 0;
    uint64_t full = 0;
    if (!values.empty()) {
      diffLvl = lexDiff(lvlCoords);
      endPath(diffLvl + 1);
      full = lvlCursor[diffLvl] + 1;
    }
    // Then continue with insertion path.
    insPath(lvlCoords, diffLvl, full, val);
  }

  /// Partially specialize expanded insertions based on template types.
  void expInsert(uint64_t *lvlCoords, V *values, bool *filled, uint64_t *added,
                 uint64_t count, uint64_t expsz) final {
    assert((lvlCoords && values && filled && added) && "Received nullptr");
    if (count == 0)
      return;
    // Sort.
    std::sort(added, added + count);
    // Restore insertion path for first insert.
    const uint64_t lastLvl = getLvlRank() - 1;
    uint64_t c = added[0];
    assert(c <= expsz);
    assert(filled[c] && "added coordinate is not filled");
    lvlCoords[lastLvl] = c;
    lexInsert(lvlCoords, values[c]);
    values[c] = 0;
    filled[c] = false;
    // Subsequent insertions are quick.
    for (uint64_t i = 1; i < count; ++i) {
      assert(c < added[i] && "non-lexicographic insertion");
      c = added[i];
      assert(c <= expsz);
      assert(filled[c] && "added coordinate is not filled");
      lvlCoords[lastLvl] = c;
      insPath(lvlCoords, lastLvl, added[i - 1] + 1, values[c]);
      values[c] = 0;
      filled[c] = false;
    }
  }

  /// Finalizes forwarding insertions.
  void endForwardingInsert() final {
    // Ensure COO is sorted.
    assert(coo);
    coo->sort();
    // Now actually insert the `elements`.
    const auto &elements = coo->getElements();
    const uint64_t nse = elements.size();
    assert(values.size() == 0);
    values.reserve(nse);
    fromCOO(elements, 0, nse, 0);
    delete coo;
    coo = nullptr;
  }

  /// Finalizes lexicographic insertions.
  void endLexInsert() final {
    if (values.empty())
      finalizeSegment(0);
    else
      endPath(0);
  }

  /// Allocates a new COO object and initializes it with the contents.
  /// Callers must make sure to delete the COO when they're done with it.
  SparseTensorCOO<V> *toCOO() {
    std::vector<uint64_t> dimCoords(getDimRank());
    coo = new SparseTensorCOO<V>(getDimSizes(), values.size());
    toCOO(0, 0, dimCoords);
    assert(coo->getElements().size() == values.size());
    return coo;
  }

  /// Sort the unordered tensor in place, the method assumes that it is
  /// an unordered COO tensor.
  void sortInPlace() {
    uint64_t nnz = values.size();
#ifndef NDEBUG
    for (uint64_t l = 0; l < getLvlRank(); l++)
      assert(nnz == coordinates[l].size());
#endif

    // In-place permutation.
    auto applyPerm = [this](std::vector<uint64_t> &perm) {
      size_t length = perm.size();
      size_t lvlRank = getLvlRank();
      // Cache for the current level coordinates.
      std::vector<P> lvlCrds(lvlRank);
      for (size_t i = 0; i < length; i++) {
        size_t current = i;
        if (i != perm[current]) {
          for (size_t l = 0; l < lvlRank; l++)
            lvlCrds[l] = coordinates[l][i];
          V val = values[i];
          // Deals with a permutation cycle.
          while (i != perm[current]) {
            size_t next = perm[current];
            // Swaps the level coordinates and value.
            for (size_t l = 0; l < lvlRank; l++)
              coordinates[l][current] = coordinates[l][next];
            values[current] = values[next];
            perm[current] = current;
            current = next;
          }
          for (size_t l = 0; l < lvlRank; l++)
            coordinates[l][current] = lvlCrds[l];
          values[current] = val;
          perm[current] = current;
        }
      }
    };

    std::vector<uint64_t> sortedIdx(nnz, 0);
    for (uint64_t i = 0; i < nnz; i++)
      sortedIdx[i] = i;

    std::sort(sortedIdx.begin(), sortedIdx.end(),
              [this](uint64_t lhs, uint64_t rhs) {
                for (uint64_t l = 0; l < getLvlRank(); l++) {
                  if (coordinates[l][lhs] == coordinates[l][rhs])
                    continue;
                  return coordinates[l][lhs] < coordinates[l][rhs];
                }
                assert(lhs == rhs && "duplicate coordinates");
                return false;
              });

    applyPerm(sortedIdx);
  }

private:
  /// Appends an arbitrary new position to `positions[lvl]`.  This method
  /// checks that `pos` is representable in the `P` type; however, it
  /// does not check that `pos` is semantically valid (i.e., larger than
  /// the previous position and smaller than `coordinates[lvl].capacity()`).
  void appendPos(uint64_t lvl, uint64_t pos, uint64_t count = 1) {
    assert(isCompressedLvl(lvl));
    positions[lvl].insert(positions[lvl].end(), count,
                          detail::checkOverflowCast<P>(pos));
  }

  /// Appends coordinate `crd` to level `lvl`, in the semantically
  /// general sense.  For non-dense levels, that means appending to the
  /// `coordinates[lvl]` array, checking that `crd` is representable in
  /// the `C` type; however, we do not verify other semantic requirements
  /// (e.g., that `crd` is in bounds for `lvlSizes[lvl]`, and not previously
  /// occurring in the same segment).  For dense levels, this method instead
  /// appends the appropriate number of zeros to the `values` array, where
  /// `full` is the number of "entries" already written to `values` for this
  /// segment (aka one after the highest coordinate previously appended).
  void appendCrd(uint64_t lvl, uint64_t full, uint64_t crd) {
    const auto dlt = getLvlType(lvl); // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt) || isSingletonDLT(dlt)) {
      coordinates[lvl].push_back(detail::checkOverflowCast<C>(crd));
    } else { // Dense level.
      assert(isDenseDLT(dlt));
      assert(crd >= full && "Coordinate was already filled");
      if (crd == full)
        return; // Short-circuit, since it'll be a nop.
      if (lvl + 1 == getLvlRank())
        values.insert(values.end(), crd - full, 0);
      else
        finalizeSegment(lvl + 1, 0, crd - full);
    }
  }

  /// Computes the assembled-size associated with the `l`-th level,
  /// given the assembled-size associated with the `(l-1)`-th level.
  /// "Assembled-sizes" correspond to the (nominal) sizes of overhead
  /// storage, as opposed to "level-sizes" which are the cardinality
  /// of possible coordinates for that level.
  uint64_t assembledSize(uint64_t parentSz, uint64_t l) const {
    const auto dlt = getLvlType(l); // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt))
      return positions[l][parentSz];
    if (isSingletonDLT(dlt))
      return parentSz; // New size is same as the parent.
    if (isDenseDLT(dlt))
      return parentSz * getLvlSize(l);
    MLIR_SPARSETENSOR_FATAL("unsupported level type: %d\n",
                            static_cast<uint8_t>(dlt));
  }

  /// Initializes sparse tensor storage scheme from a memory-resident sparse
  /// tensor in coordinate scheme. This method prepares the positions and
  /// coordinates arrays under the given per-level dense/sparse annotations.
  void fromCOO(const std::vector<Element<V>> &lvlElements, uint64_t lo,
               uint64_t hi, uint64_t l) {
    const uint64_t lvlRank = getLvlRank();
    assert(l <= lvlRank && hi <= lvlElements.size());
    // Once levels are exhausted, insert the numerical values.
    if (l == lvlRank) {
      assert(lo < hi);
      values.push_back(lvlElements[lo].value);
      return;
    }
    // Visit all elements in this interval.
    uint64_t full = 0;
    while (lo < hi) { // If `hi` is unchanged, then `lo < lvlElements.size()`.
      // Find segment in interval with same coordinate at this level.
      const uint64_t c = lvlElements[lo].coords[l];
      uint64_t seg = lo + 1;
      if (isUniqueLvl(l))
        while (seg < hi && lvlElements[seg].coords[l] == c)
          ++seg;
      // Handle segment in interval for sparse or dense level.
      appendCrd(l, full, c);
      full = c + 1;
      fromCOO(lvlElements, lo, seg, l + 1);
      // And move on to next segment in interval.
      lo = seg;
    }
    // Finalize the sparse position structure at this level.
    finalizeSegment(l, full);
  }

  /// Finalizes the sparse position structure at this level.
  void finalizeSegment(uint64_t l, uint64_t full = 0, uint64_t count = 1) {
    if (count == 0)
      return;                       // Short-circuit, since it'll be a nop.
    const auto dlt = getLvlType(l); // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt)) {
      appendPos(l, coordinates[l].size(), count);
    } else if (isSingletonDLT(dlt)) {
      return; // Nothing to finalize.
    } else {  // Dense dimension.
      assert(isDenseDLT(dlt));
      const uint64_t sz = getLvlSizes()[l];
      assert(sz >= full && "Segment is overfull");
      count = detail::checkedMul(count, sz - full);
      // For dense storage we must enumerate all the remaining coordinates
      // in this level (i.e., coordinates after the last non-zero
      // element), and either fill in their zero values or else recurse
      // to finalize some deeper level.
      if (l + 1 == getLvlRank())
        values.insert(values.end(), count, 0);
      else
        finalizeSegment(l + 1, 0, count);
    }
  }

  /// Wraps up a single insertion path, inner to outer.
  void endPath(uint64_t diffLvl) {
    const uint64_t lvlRank = getLvlRank();
    const uint64_t lastLvl = lvlRank - 1;
    assert(diffLvl <= lvlRank);
    const uint64_t stop = lvlRank - diffLvl;
    for (uint64_t i = 0; i < stop; ++i) {
      const uint64_t l = lastLvl - i;
      finalizeSegment(l, lvlCursor[l] + 1);
    }
  }

  /// Continues a single insertion path, outer to inner. The first
  /// argument is the level-coordinates for the value being inserted.
  void insPath(const uint64_t *lvlCoords, uint64_t diffLvl, uint64_t full,
               V val) {
    const uint64_t lvlRank = getLvlRank();
    assert(diffLvl <= lvlRank);
    for (uint64_t l = diffLvl; l < lvlRank; ++l) {
      const uint64_t c = lvlCoords[l];
      appendCrd(l, full, c);
      full = 0;
      lvlCursor[l] = c;
    }
    values.push_back(val);
  }

  /// Finds the lexicographically first level where the level-coordinates
  /// in the argument differ from those in the current cursor.
  uint64_t lexDiff(const uint64_t *lvlCoords) const {
    const uint64_t lvlRank = getLvlRank();
    for (uint64_t l = 0; l < lvlRank; ++l) {
      const auto crd = lvlCoords[l];
      const auto cur = lvlCursor[l];
      if (crd > cur || (crd == cur && !isUniqueLvl(l)) ||
          (crd < cur && !isOrderedLvl(l))) {
        return l;
      }

      if (crd < cur) {
        assert(false && "non-lexicographic insertion");
        return -1u;
      }
    }
    assert(false && "duplicate insertion");
    return -1u;
  }

  // Performs forall on level entries and inserts into dim COO.
  void toCOO(uint64_t parentPos, uint64_t l, std::vector<uint64_t> &dimCoords) {
    if (l == getLvlRank()) {
      map.pushbackward(lvlCursor.data(), dimCoords.data());
      assert(coo);
      assert(parentPos < values.size());
      coo->add(dimCoords, values[parentPos]);
      return;
    }
    if (isCompressedLvl(l)) {
      // Look up the bounds of the `l`-level segment determined by the
      // `(l - 1)`-level position `parentPos`.
      const std::vector<P> &positionsL = positions[l];
      assert(parentPos + 1 < positionsL.size());
      const uint64_t pstart = static_cast<uint64_t>(positionsL[parentPos]);
      const uint64_t pstop = static_cast<uint64_t>(positionsL[parentPos + 1]);
      // Loop-invariant code for looking up the `l`-level coordinates.
      const std::vector<C> &coordinatesL = coordinates[l];
      assert(pstop <= coordinatesL.size());
      for (uint64_t pos = pstart; pos < pstop; ++pos) {
        lvlCursor[l] = static_cast<uint64_t>(coordinatesL[pos]);
        toCOO(pos, l + 1, dimCoords);
      }
    } else if (isSingletonLvl(l)) {
      lvlCursor[l] = getCrd(l, parentPos);
      toCOO(parentPos, l + 1, dimCoords);
    } else { // Dense level.
      assert(isDenseLvl(l));
      const uint64_t sz = getLvlSizes()[l];
      const uint64_t pstart = parentPos * sz;
      for (uint64_t c = 0; c < sz; ++c) {
        lvlCursor[l] = c;
        toCOO(pstart + c, l + 1, dimCoords);
      }
    }
  }

  std::vector<std::vector<P>> positions;
  std::vector<std::vector<C>> coordinates;
  std::vector<V> values;
  std::vector<uint64_t> lvlCursor;
  SparseTensorCOO<V> *coo;
};

//===----------------------------------------------------------------------===//
//
//  SparseTensorStorage Factories
//
//===----------------------------------------------------------------------===//

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V> *SparseTensorStorage<P, C, V>::newEmpty(
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
    const uint64_t *dim2lvl, const uint64_t *lvl2dim, bool forwarding) {
  SparseTensorCOO<V> *lvlCOO = nullptr;
  if (forwarding)
    lvlCOO = new SparseTensorCOO<V>(lvlRank, lvlSizes);
  return new SparseTensorStorage<P, C, V>(dimRank, dimSizes, lvlRank, lvlSizes,
                                          lvlTypes, dim2lvl, lvl2dim, lvlCOO,
                                          !forwarding);
}

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V> *SparseTensorStorage<P, C, V>::newFromCOO(
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
    const uint64_t *dim2lvl, const uint64_t *lvl2dim,
    SparseTensorCOO<V> &lvlCOO) {
  return new SparseTensorStorage<P, C, V>(dimRank, dimSizes, lvlRank, lvlSizes,
                                          lvlTypes, dim2lvl, lvl2dim, lvlCOO);
}

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V> *SparseTensorStorage<P, C, V>::packFromLvlBuffers(
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
    const uint64_t *dim2lvl, const uint64_t *lvl2dim, uint64_t srcRank,
    const intptr_t *buffers) {
  return new SparseTensorStorage<P, C, V>(dimRank, dimSizes, lvlRank, lvlSizes,
                                          lvlTypes, dim2lvl, lvl2dim, buffers);
}

//===----------------------------------------------------------------------===//
//
//  SparseTensorStorage Constructors
//
//===----------------------------------------------------------------------===//

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V>::SparseTensorStorage(
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
    const uint64_t *dim2lvl, const uint64_t *lvl2dim,
    SparseTensorCOO<V> *lvlCOO, bool initializeValuesIfAllDense)
    : SparseTensorStorage(dimRank, dimSizes, lvlRank, lvlSizes, lvlTypes,
                          dim2lvl, lvl2dim) {
  assert(!lvlCOO || lvlRank == lvlCOO->getRank());
  coo = lvlCOO;
  // Provide hints on capacity of positions and coordinates.
  // TODO: needs much fine-tuning based on actual sparsity; currently
  // we reserve position/coordinate space based on all previous dense
  // levels, which works well up to first sparse level; but we should
  // really use nnz and dense/sparse distribution.
  bool allDense = true;
  uint64_t sz = 1;
  for (uint64_t l = 0; l < lvlRank; ++l) {
    const DimLevelType dlt = lvlTypes[l]; // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt)) {
      positions[l].reserve(sz + 1);
      positions[l].push_back(0);
      coordinates[l].reserve(sz);
      sz = 1;
      allDense = false;
    } else if (isSingletonDLT(dlt)) {
      coordinates[l].reserve(sz);
      sz = 1;
      allDense = false;
    } else { // Dense level.
      assert(isDenseDLT(dlt));
      sz = detail::checkedMul(sz, lvlSizes[l]);
    }
  }
  if (allDense && initializeValuesIfAllDense)
    values.resize(sz, 0);
}

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V>::SparseTensorStorage( // NOLINT
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
    const uint64_t *dim2lvl, const uint64_t *lvl2dim,
    SparseTensorCOO<V> &lvlCOO)
    : SparseTensorStorage(dimRank, dimSizes, lvlRank, lvlSizes, lvlTypes,
                          dim2lvl, lvl2dim, nullptr, false) {
  // Ensure lvlCOO is sorted.
  assert(lvlRank == lvlCOO.getRank());
  lvlCOO.sort();
  // Now actually insert the `elements`.
  const auto &elements = lvlCOO.getElements();
  const uint64_t nse = elements.size();
  assert(values.size() == 0);
  values.reserve(nse);
  fromCOO(elements, 0, nse, 0);
}

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V>::SparseTensorStorage(
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
    const uint64_t *dim2lvl, const uint64_t *lvl2dim, const intptr_t *lvlBufs)
    : SparseTensorStorage(dimRank, dimSizes, lvlRank, lvlSizes, lvlTypes,
                          dim2lvl, lvl2dim) {
  uint64_t trailCOOLen = 0, parentSz = 1, bufIdx = 0;
  for (uint64_t l = 0; l < lvlRank; l++) {
    if (!isUniqueLvl(l) && isCompressedLvl(l)) {
      // A `compressed_nu` level marks the start of trailing COO start level.
      // Since the coordinate buffer used for trailing COO are passed in as AoS
      // scheme, and SparseTensorStorage uses a SoA scheme, we can not simply
      // copy the value from the provided buffers.
      trailCOOLen = lvlRank - l;
      break;
    }
    assert(!isSingletonLvl(l) &&
           "Singleton level not following a compressed_nu level");
    if (isCompressedLvl(l)) {
      P *posPtr = reinterpret_cast<P *>(lvlBufs[bufIdx++]);
      C *crdPtr = reinterpret_cast<C *>(lvlBufs[bufIdx++]);
      // Copies the lvlBuf into the vectors. The buffer can not be simply reused
      // because the memory passed from users is not necessarily allocated on
      // heap.
      positions[l].assign(posPtr, posPtr + parentSz + 1);
      coordinates[l].assign(crdPtr, crdPtr + positions[l][parentSz]);
    } else {
      assert(isDenseLvl(l));
    }
    parentSz = assembledSize(parentSz, l);
  }

  if (trailCOOLen != 0) {
    uint64_t cooStartLvl = lvlRank - trailCOOLen;
    assert(!isUniqueLvl(cooStartLvl) && isCompressedLvl(cooStartLvl));
    P *posPtr = reinterpret_cast<P *>(lvlBufs[bufIdx++]);
    C *aosCrdPtr = reinterpret_cast<C *>(lvlBufs[bufIdx++]);
    positions[cooStartLvl].assign(posPtr, posPtr + parentSz + 1);
    P crdLen = positions[cooStartLvl][parentSz];
    for (uint64_t l = cooStartLvl; l < lvlRank; l++) {
      coordinates[l].resize(crdLen);
      for (uint64_t n = 0; n < crdLen; n++) {
        coordinates[l][n] = *(aosCrdPtr + (l - cooStartLvl) + n * trailCOOLen);
      }
    }
    parentSz = assembledSize(parentSz, cooStartLvl);
  }

  V *valPtr = reinterpret_cast<V *>(lvlBufs[bufIdx]);
  values.assign(valPtr, valPtr + parentSz);
}

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_STORAGE_H
