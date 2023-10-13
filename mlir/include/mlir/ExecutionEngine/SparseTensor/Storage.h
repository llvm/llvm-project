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
// * `SparseTensorEnumeratorBase<V>`
// * `SparseTensorEnumerator<P, C, V>`
// * `SparseTensorNNZ`
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

#define ASSERT_COMPRESSED_OR_SINGLETON_LVL(l)                                  \
  do {                                                                         \
    const DimLevelType dlt = getLvlType(l);                                    \
    (void)dlt;                                                                 \
    assert((isCompressedDLT(dlt) || isSingletonDLT(dlt)) &&                    \
           "Level is neither compressed nor singleton");                       \
  } while (false)

namespace mlir {
namespace sparse_tensor {

// Forward references.
template <typename V>
class SparseTensorEnumeratorBase;
template <typename P, typename C, typename V>
class SparseTensorEnumerator;

//===----------------------------------------------------------------------===//
//
//  SparseTensorStorage
//
//===----------------------------------------------------------------------===//

/// Abstract base class for `SparseTensorStorage<P,C,V>`. This class
/// takes responsibility for all the `<P,C,V>`-independent aspects
/// of the tensor (e.g., shape, sparsity, mapping). In addition,
/// we use function overloading to implement "partial" method
/// specialization, which the C-API relies on to catch type errors
/// arising from our use of opaque pointers.
///
/// Because this class forms a bridge between the denotational semantics
/// of "tensors" and the operational semantics of how we store and
/// compute with them, it also distinguishes between two different
/// coordinate spaces (and their associated rank, shape, sizes, etc).
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

  /// Gets the level-to-dimension mapping.
  // TODO: REMOVE THIS
  const std::vector<uint64_t> &getLvl2Dim() const { return lvl2dimVec; }

  /// Allocates a new enumerator.  Callers must make sure to delete
  /// the enumerator when they're done with it. The first argument
  /// is the out-parameter for storing the newly allocated enumerator;
  /// all other arguments are passed along to the `SparseTensorEnumerator`
  /// ctor and must satisfy the preconditions/assertions thereof.
#define DECL_NEWENUMERATOR(VNAME, V)                                           \
  virtual void newEnumerator(SparseTensorEnumeratorBase<V> **, uint64_t,       \
                             const uint64_t *, uint64_t, const uint64_t *)     \
      const;
  MLIR_SPARSETENSOR_FOREVERY_V(DECL_NEWENUMERATOR)
#undef DECL_NEWENUMERATOR

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
        positions(lvlRank), coordinates(lvlRank), lvlCursor(lvlRank), lvlCOO() {
  }

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
                      const uint64_t *lvl2dim, SparseTensorCOO<V> *coo,
                      bool initializeValuesIfAllDense);

  /// Constructs a sparse tensor with the given encoding, and initializes
  /// the contents from the COO. This ctor performs the same heuristic
  /// overhead-storage allocation as the ctor above.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const DimLevelType *lvlTypes,
                      const uint64_t *dim2lvl, const uint64_t *lvl2dim,
                      SparseTensorCOO<V> &lvlCOO);

  /// Constructs a sparse tensor with the given encoding, and initializes
  /// the contents from the enumerator. This ctor allocates exactly
  /// the required amount of overhead storage, not using any heuristics.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const DimLevelType *lvlTypes,
                      const uint64_t *dim2lvl, const uint64_t *lvl2dim,
                      SparseTensorEnumeratorBase<V> &lvlEnumerator);

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

  /// Allocates a new empty sparse tensor. The preconditions/assertions
  /// are as per the `SparseTensorStorageBase` ctor; which is to say,
  /// the `dimSizes` and `lvlSizes` must both be "sizes" not "shapes",
  /// since there's nowhere to reconstruct dynamic sizes from.
  static SparseTensorStorage<P, C, V> *
  newEmpty(uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
           const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
           const uint64_t *dim2lvl, const uint64_t *lvl2dim, bool forwarding);

  /// Allocates a new sparse tensor and initializes it from the given COO.
  /// The preconditions are as per the `SparseTensorStorageBase` ctor
  /// (where we define `lvlSizes = lvlCOO.getDimSizes().data()`), but
  /// using the following assertions in lieu of the base ctor's assertions:
  //
  // TODO: The ability to reconstruct dynamic dimensions-sizes does not
  // easily generalize to arbitrary `lvl2dim` mappings.  When compiling
  // MLIR programs to use this library, we should be able to generate
  // code for effectively computing the reconstruction, but it's not clear
  // that there's a feasible way to do so from within the library itself.
  // Therefore, when we functionalize the `lvl2dim` mapping we'll have
  // to update the type/preconditions of this factory too.
  static SparseTensorStorage<P, C, V> *
  newFromCOO(uint64_t dimRank, const uint64_t *dimShape, uint64_t lvlRank,
             const DimLevelType *lvlTypes, const uint64_t *dim2lvl,
             const uint64_t *lvl2dim, SparseTensorCOO<V> &lvlCOO);

  /// Allocates a new sparse tensor and initializes it with the contents
  /// of another sparse tensor.
  //
  // TODO: The `dimRank` and `dimShape` arguments are only used for
  // verifying that the source tensor has the expected shape.  So if we
  // wanted to skip that verification, then we could remove those arguments.
  // Alternatively, if we required the `dimShape` to be "sizes" instead,
  // then that would remove any constraints on `source.getDimSizes()`
  // (other than compatibility with `src2lvl`) as well as removing the
  // requirement that `src2lvl` be the inverse of `lvl2dim`.  Which would
  // enable this factory to be used for performing a much larger class of
  // transformations (which can already be handled by the `SparseTensorNNZ`
  // implementation).
  static SparseTensorStorage<P, C, V> *
  newFromSparseTensor(uint64_t dimRank, const uint64_t *dimShape,
                      uint64_t lvlRank, const uint64_t *lvlSizes,
                      const DimLevelType *lvlTypes,
                      const uint64_t *src2lvl, // FIXME: dim2lvl,
                      const uint64_t *lvl2dim, uint64_t srcRank,
                      const SparseTensorStorageBase &source);

  /// Allocates a new sparse tensor and initialize it with the data stored level
  /// buffers directly.
  static SparseTensorStorage<P, C, V> *packFromLvlBuffers(
      uint64_t dimRank, const uint64_t *dimShape, uint64_t lvlRank,
      const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
      const uint64_t *src2lvl, // FIXME: dim2lvl
      const uint64_t *lvl2dim, uint64_t srcRank, const intptr_t *buffers);

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
    ASSERT_COMPRESSED_OR_SINGLETON_LVL(lvl);
    assert(pos < coordinates[lvl].size());
    return coordinates[lvl][pos]; // Converts the stored `C` into `uint64_t`.
  }

  /// Partially specialize forwarding insertions based on template types.
  void forwardingInsert(const uint64_t *dimCoords, V val) final {
    assert(dimCoords && lvlCOO);
    map.pushforward(dimCoords, lvlCursor.data());
    lvlCOO->add(lvlCursor, val);
  }

  /// Partially specialize lexicographical insertions based on template types.
  void lexInsert(const uint64_t *lvlCoords, V val) final {
    assert(lvlCoords);
    // TODO: get rid of this! canonicalize all-dense "sparse" array into dense
    // tensors.
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
    // Ensure lvlCOO is sorted.
    assert(lvlCOO);
    lvlCOO->sort();
    // Now actually insert the `elements`.
    const auto &elements = lvlCOO->getElements();
    const uint64_t nse = elements.size();
    assert(values.size() == 0);
    values.reserve(nse);
    fromCOO(elements, 0, nse, 0);
    delete lvlCOO;
  }

  /// Finalizes lexicographic insertions.
  void endLexInsert() final {
    if (values.empty())
      finalizeSegment(0);
    else
      endPath(0);
  }

  /// Allocates a new enumerator for this class's `<P,C,V>` types and
  /// erase the `<P,C>` parts from the type.  Callers must make sure to
  /// delete the enumerator when they're done with it.
  void newEnumerator(SparseTensorEnumeratorBase<V> **out, uint64_t trgRank,
                     const uint64_t *trgSizes, uint64_t srcRank,
                     const uint64_t *src2trg) const final {
    assert(out && "Received nullptr for out parameter");
    *out = new SparseTensorEnumerator<P, C, V>(*this, trgRank, trgSizes,
                                               srcRank, src2trg);
  }

  /// Allocates a new COO object and initializes it with the contents
  /// of this tensor under the given mapping from the `getDimSizes()`
  /// coordinate-space to the `trgSizes` coordinate-space. Callers must
  /// make sure to delete the COO when they're done with it.
  SparseTensorCOO<V> *toCOO(uint64_t trgRank, const uint64_t *trgSizes,
                            uint64_t srcRank,
                            const uint64_t *src2trg, // FIXME: dim2lvl
                            const uint64_t *lvl2dim) const {
    // We inline `newEnumerator` to avoid virtual dispatch and allocation.
    // TODO: use MapRef here too for the translation
    SparseTensorEnumerator<P, C, V> enumerator(*this, trgRank, trgSizes,
                                               srcRank, src2trg);
    auto *coo = new SparseTensorCOO<V>(trgRank, trgSizes, values.size());
    enumerator.forallElements(
        [&coo](const auto &trgCoords, V val) { coo->add(trgCoords, val); });
    // TODO: This assertion assumes there are no stored zeros,
    // or if there are then that we don't filter them out.
    // <https://github.com/llvm/llvm-project/issues/54179>
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
                assert(false && "duplicate coordinates");
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

  /// Writes the given coordinate to `coordinates[lvl][pos]`.  This method
  /// checks that `crd` is representable in the `C` type; however, it
  /// does not check that `crd` is semantically valid (i.e., in bounds
  /// for `dimSizes[lvl]` and not elsewhere occurring in the same segment).
  void writeCrd(uint64_t lvl, uint64_t pos, uint64_t crd) {
    ASSERT_COMPRESSED_OR_SINGLETON_LVL(lvl);
    // Subscript assignment to `std::vector` requires that the `pos`-th
    // entry has been initialized; thus we must be sure to check `size()`
    // here, instead of `capacity()` as would be ideal.
    assert(pos < coordinates[lvl].size());
    coordinates[lvl][pos] = detail::checkOverflowCast<C>(crd);
  }

  /// Computes the assembled-size associated with the `l`-th level,
  /// given the assembled-size associated with the `(l-1)`-th level.
  /// "Assembled-sizes" correspond to the (nominal) sizes of overhead
  /// storage, as opposed to "level-sizes" which are the cardinality
  /// of possible coordinates for that level.
  ///
  /// Precondition: the `positions[l]` array must be fully initialized
  /// before calling this method.
  uint64_t assembledSize(uint64_t parentSz, uint64_t l) const {
    const auto dlt = getLvlType(l); // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt))
      return positions[l][parentSz];
    if (isSingletonDLT(dlt))
      return parentSz; // New size is same as the parent.
    if (isDenseDLT(dlt))
      return parentSz * getLvlSizes()[l];
    MLIR_SPARSETENSOR_FATAL("unsupported level type: %d\n",
                            static_cast<uint8_t>(dlt));
  }

  /// Initializes sparse tensor storage scheme from a memory-resident sparse
  /// tensor in coordinate scheme. This method prepares the positions and
  /// coordinates arrays under the given per-level dense/sparse annotations.
  ///
  /// Preconditions:
  /// * the `lvlElements` must be lexicographically sorted.
  /// * the coordinates of every element are valid for `getLvlSizes()`
  ///   (i.e., equal rank and pointwise less-than).
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

  // Allow `SparseTensorEnumerator` to access the data-members (to avoid
  // the cost of virtual-function dispatch in inner loops), without
  // making them public to other client code.
  friend class SparseTensorEnumerator<P, C, V>;

  std::vector<std::vector<P>> positions;
  std::vector<std::vector<C>> coordinates;
  std::vector<V> values;
  std::vector<uint64_t> lvlCursor; // cursor for lexicographic insertion.
  SparseTensorCOO<V> *lvlCOO;      // COO used during forwarding
};

#undef ASSERT_COMPRESSED_OR_SINGLETON_LVL

//===----------------------------------------------------------------------===//
//
//  SparseTensorEnumerator
//
//===----------------------------------------------------------------------===//

/// A (higher-order) function object for enumerating the elements of some
/// `SparseTensorStorage` under a permutation.  That is, the `forallElements`
/// method encapsulates the loop-nest for enumerating the elements of
/// the source tensor (in whatever order is best for the source tensor),
/// and applies a permutation to the coordinates before handing
/// each element to the callback.  A single enumerator object can be
/// freely reused for several calls to `forallElements`, just so long
/// as each call is sequential with respect to one another.
///
/// N.B., this class stores a reference to the `SparseTensorStorageBase`
/// passed to the constructor; thus, objects of this class must not
/// outlive the sparse tensor they depend on.
///
/// Design Note: The reason we define this class instead of simply using
/// `SparseTensorEnumerator<P,C,V>` is because we need to hide/generalize
/// the `<P,C>` template parameters from MLIR client code (to simplify the
/// type parameters used for direct sparse-to-sparse conversion).  And the
/// reason we define the `SparseTensorEnumerator<P,C,V>` subclasses rather
/// than simply using this class, is to avoid the cost of virtual-method
/// dispatch within the loop-nest.
template <typename V>
class SparseTensorEnumeratorBase {
public:
  /// Constructs an enumerator which automatically applies the given
  /// mapping from the source tensor's dimensions to the desired
  /// target tensor dimensions.
  ///
  /// Preconditions:
  /// * the `src` must have the same `V` value type.
  /// * `trgSizes` must be valid for `trgRank`.
  /// * `src2trg` must be valid for `srcRank`, and must map coordinates
  ///   valid for `src.getDimSizes()` to coordinates valid for `trgSizes`.
  ///
  /// Asserts:
  /// * `trgSizes` must be nonnull and must contain only nonzero sizes.
  /// * `srcRank == src.getDimRank()`.
  /// * `src2trg` must be nonnull.
  SparseTensorEnumeratorBase(const SparseTensorStorageBase &src,
                             uint64_t trgRank, const uint64_t *trgSizes,
                             uint64_t srcRank, const uint64_t *src2trg)
      : src(src), trgSizes(trgSizes, trgSizes + trgRank),
        lvl2trg(src.getLvlRank()), trgCursor(trgRank) {
    assert(trgSizes && "Received nullptr for target-sizes");
    assert(src2trg && "Received nullptr for source-to-target mapping");
    assert(srcRank == src.getDimRank() && "Source-rank mismatch");
    for (uint64_t t = 0; t < trgRank; ++t)
      assert(trgSizes[t] > 0 && "Target-size zero has trivial storage");
    const auto &lvl2src = src.getLvl2Dim();
    for (uint64_t lvlRank = src.getLvlRank(), l = 0; l < lvlRank; ++l)
      lvl2trg[l] = src2trg[lvl2src[l]];
  }

  virtual ~SparseTensorEnumeratorBase() = default;

  // We disallow copying to help avoid leaking the `src` reference.
  // (In addition to avoiding the problem of slicing.)
  SparseTensorEnumeratorBase(const SparseTensorEnumeratorBase &) = delete;
  SparseTensorEnumeratorBase &
  operator=(const SparseTensorEnumeratorBase &) = delete;

  /// Gets the source's dimension-rank.
  uint64_t getSrcDimRank() const { return src.getDimRank(); }

  /// Gets the target's dimension-/level-rank.  (This is usually
  /// "dimension-rank", though that may coincide with "level-rank"
  /// depending on usage.)
  uint64_t getTrgRank() const { return trgSizes.size(); }

  /// Gets the target's dimension-/level-sizes.  (These are usually
  /// "dimensions", though that may coincide with "level-rank" depending
  /// on usage.)
  const std::vector<uint64_t> &getTrgSizes() const { return trgSizes; }

  /// Enumerates all elements of the source tensor, permutes their
  /// coordinates, and passes the permuted element to the callback.
  /// The callback must not store the cursor reference directly,
  /// since this function reuses the storage.  Instead, the callback
  /// must copy it if they want to keep it.
  virtual void forallElements(ElementConsumer<V> yield) = 0;

protected:
  const SparseTensorStorageBase &src;
  std::vector<uint64_t> trgSizes;  // in target order.
  std::vector<uint64_t> lvl2trg;   // source-levels -> target-dims/lvls.
  std::vector<uint64_t> trgCursor; // in target order.
};

template <typename P, typename C, typename V>
class SparseTensorEnumerator final : public SparseTensorEnumeratorBase<V> {
  using Base = SparseTensorEnumeratorBase<V>;
  using StorageImpl = SparseTensorStorage<P, C, V>;

public:
  /// Constructs an enumerator which automatically applies the given
  /// mapping from the source tensor's dimensions to the desired
  /// target tensor dimensions.
  ///
  /// Preconditions/assertions are as per the `SparseTensorEnumeratorBase` ctor.
  SparseTensorEnumerator(const StorageImpl &src, uint64_t trgRank,
                         const uint64_t *trgSizes, uint64_t srcRank,
                         const uint64_t *src2trg)
      : Base(src, trgRank, trgSizes, srcRank, src2trg) {}

  ~SparseTensorEnumerator() final = default;

  void forallElements(ElementConsumer<V> yield) final {
    forallElements(yield, 0, 0);
  }

private:
  // TODO: Once we functionalize the mappings, then we'll no longer
  // be able to use the current approach of constructing `lvl2trg` in the
  // ctor and using it to incrementally fill the `trgCursor` cursor as we
  // recurse through `forallElements`.  Instead we'll want to incrementally
  // fill a `lvlCursor` as we recurse, and then use `src.getLvl2Dim()`
  // and `src2trg` to convert it just before yielding to the callback.
  // It's probably most efficient to just store the `srcCursor` and
  // `trgCursor` buffers in this object, but we may want to benchmark
  // that against using `std::calloc` to stack-allocate them instead.
  //
  /// The recursive component of the public `forallElements`.
  void forallElements(ElementConsumer<V> yield, uint64_t parentPos,
                      uint64_t l) {
    // Recover the `<P,C,V>` type parameters of `src`.
    const auto &src = static_cast<const StorageImpl &>(this->src);
    if (l == src.getLvlRank()) {
      assert(parentPos < src.values.size());
      // TODO: <https://github.com/llvm/llvm-project/issues/54179>
      yield(this->trgCursor, src.values[parentPos]);
      return;
    }
    uint64_t &cursorL = this->trgCursor[this->lvl2trg[l]];
    const auto dlt = src.getLvlType(l); // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt)) {
      // Look up the bounds of the `l`-level segment determined by the
      // `(l - 1)`-level position `parentPos`.
      const std::vector<P> &positionsL = src.positions[l];
      assert(parentPos + 1 < positionsL.size());
      const uint64_t pstart = static_cast<uint64_t>(positionsL[parentPos]);
      const uint64_t pstop = static_cast<uint64_t>(positionsL[parentPos + 1]);
      // Loop-invariant code for looking up the `l`-level coordinates.
      const std::vector<C> &coordinatesL = src.coordinates[l];
      assert(pstop <= coordinatesL.size());
      for (uint64_t pos = pstart; pos < pstop; ++pos) {
        cursorL = static_cast<uint64_t>(coordinatesL[pos]);
        forallElements(yield, pos, l + 1);
      }
    } else if (isSingletonDLT(dlt)) {
      cursorL = src.getCrd(l, parentPos);
      forallElements(yield, parentPos, l + 1);
    } else { // Dense level.
      assert(isDenseDLT(dlt));
      const uint64_t sz = src.getLvlSizes()[l];
      const uint64_t pstart = parentPos * sz;
      for (uint64_t c = 0; c < sz; ++c) {
        cursorL = c;
        forallElements(yield, pstart + c, l + 1);
      }
    }
  }
};

//===----------------------------------------------------------------------===//
//
//  SparseTensorNNZ
//
//===----------------------------------------------------------------------===//

/// Statistics regarding the number of nonzero subtensors in
/// a source tensor, for direct sparse=>sparse conversion a la
/// <https://arxiv.org/abs/2001.02609>.
///
/// N.B., this class stores references to the parameters passed to
/// the constructor; thus, objects of this class must not outlive
/// those parameters.
///
/// This class does not have the "dimension" vs "level" distinction, but
/// since it is used for initializing the levels of a `SparseTensorStorage`
/// object, we use the "level" name throughout for the sake of consistency.
class SparseTensorNNZ final {
public:
  /// Allocates the statistics structure for the desired target-tensor
  /// level structure (i.e., sizes and types).  This constructor does not
  /// actually populate the statistics, however; for that see `initialize`.
  ///
  /// Precondition: `lvlSizes` must not contain zeros.
  /// Asserts: `lvlSizes.size() == lvlTypes.size()`.
  SparseTensorNNZ(const std::vector<uint64_t> &lvlSizes,
                  const std::vector<DimLevelType> &lvlTypes);

  // We disallow copying to help avoid leaking the stored references.
  SparseTensorNNZ(const SparseTensorNNZ &) = delete;
  SparseTensorNNZ &operator=(const SparseTensorNNZ &) = delete;

  /// Gets the target-tensor's level-rank.
  uint64_t getLvlRank() const { return lvlSizes.size(); }

  /// Enumerates the source tensor to fill in the statistics.
  /// The enumerator should already incorporate the mapping from
  /// the source tensor-dimensions to the target storage-levels.
  ///
  /// Asserts:
  /// * `enumerator.getTrgRank() == getLvlRank()`.
  /// * `enumerator.getTrgSizes() == lvlSizes`.
  template <typename V>
  void initialize(SparseTensorEnumeratorBase<V> &enumerator) {
    assert(enumerator.getTrgRank() == getLvlRank() && "Tensor rank mismatch");
    assert(enumerator.getTrgSizes() == lvlSizes && "Tensor size mismatch");
    enumerator.forallElements(
        [this](const std::vector<uint64_t> &lvlCoords, V) { add(lvlCoords); });
  }

  /// The type of callback functions which receive an nnz-statistic.
  using NNZConsumer = const std::function<void(uint64_t)> &;

  /// Lexicographically enumerates all coordinates for levels strictly
  /// less than `stopLvl`, and passes their nnz statistic to the callback.
  /// Since our use-case only requires the statistic not the coordinates
  /// themselves, we do not bother to construct those coordinates.
  void forallCoords(uint64_t stopLvl, NNZConsumer yield) const;

private:
  /// Adds a new element (i.e., increment its statistics).  We use
  /// a method rather than inlining into the lambda in `initialize`,
  /// to avoid spurious templating over `V`.  And this method is private
  /// to avoid needing to re-assert validity of `lvlCoords` (which is
  /// guaranteed by `forallElements`).
  void add(const std::vector<uint64_t> &lvlCoords);

  /// Recursive component of the public `forallCoords`.
  void forallCoords(NNZConsumer yield, uint64_t stopLvl, uint64_t parentPos,
                    uint64_t l) const;

  // All of these are in the target storage-order.
  const std::vector<uint64_t> &lvlSizes;
  const std::vector<DimLevelType> &lvlTypes;
  std::vector<std::vector<uint64_t>> nnz;
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

// TODO: MapRef
template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V> *SparseTensorStorage<P, C, V>::newFromCOO(
    uint64_t dimRank, const uint64_t *dimShape, uint64_t lvlRank,
    const DimLevelType *lvlTypes, const uint64_t *dim2lvl,
    const uint64_t *lvl2dim, SparseTensorCOO<V> &lvlCOO) {
  assert(dimShape && dim2lvl && lvl2dim);
  const auto &lvlSizes = lvlCOO.getDimSizes();
  assert(lvlRank == lvlSizes.size() && "Level-rank mismatch");
  // Must reconstruct `dimSizes` from `lvlSizes`.  While this is easy
  // enough to do when `lvl2dim` is a permutation, this approach will
  // not work for more general mappings; so we will need to move this
  // computation off to codegen.
  std::vector<uint64_t> dimSizes(dimRank);
  for (uint64_t l = 0; l < lvlRank; ++l) {
    const uint64_t d = lvl2dim[l];
    assert((dimShape[d] == 0 || dimShape[d] == lvlSizes[l]) &&
           "Dimension sizes do not match expected shape");
    dimSizes[d] = lvlSizes[l];
  }
  return new SparseTensorStorage<P, C, V>(dimRank, dimSizes.data(), lvlRank,
                                          lvlTypes, dim2lvl, lvl2dim, lvlCOO);
}

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V> *SparseTensorStorage<P, C, V>::newFromSparseTensor(
    uint64_t dimRank, const uint64_t *dimShape, uint64_t lvlRank,
    const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
    const uint64_t *src2lvl, // dim2lvl
    const uint64_t *lvl2dim, uint64_t srcRank,
    const SparseTensorStorageBase &source) {
  // Verify that the `source` dimensions match the expected `dimShape`.
  assert(dimShape && "Got nullptr for dimension shape");
  assert(dimRank == source.getDimRank() && "Dimension-rank mismatch");
  const auto &dimSizes = source.getDimSizes();
#ifndef NDEBUG
  for (uint64_t d = 0; d < dimRank; ++d) {
    const uint64_t sz = dimShape[d];
    assert((sz == 0 || sz == dimSizes[d]) &&
           "Dimension-sizes do not match expected shape");
  }
#endif
  SparseTensorEnumeratorBase<V> *lvlEnumerator;
  source.newEnumerator(&lvlEnumerator, lvlRank, lvlSizes, srcRank, src2lvl);
  auto *tensor = new SparseTensorStorage<P, C, V>(dimRank, dimSizes.data(),
                                                  lvlRank, lvlTypes, src2lvl,
                                                  lvl2dim, *lvlEnumerator);
  delete lvlEnumerator;
  return tensor;
}

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V> *SparseTensorStorage<P, C, V>::packFromLvlBuffers(
    uint64_t dimRank, const uint64_t *dimShape, uint64_t lvlRank,
    const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
    const uint64_t *src2lvl, // FIXME: dim2lvl
    const uint64_t *lvl2dim, uint64_t srcRank, const intptr_t *buffers) {
  assert(dimShape && "Got nullptr for dimension shape");
  auto *tensor =
      new SparseTensorStorage<P, C, V>(dimRank, dimShape, lvlRank, lvlSizes,
                                       lvlTypes, src2lvl, lvl2dim, buffers);
  return tensor;
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
    const uint64_t *dim2lvl, const uint64_t *lvl2dim, SparseTensorCOO<V> *coo,
    bool initializeValuesIfAllDense)
    : SparseTensorStorage(dimRank, dimSizes, lvlRank, lvlSizes, lvlTypes,
                          dim2lvl, lvl2dim) {
  lvlCOO = coo;
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

// TODO: share more code with forwarding methods?
template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V>::SparseTensorStorage( // NOLINT
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const DimLevelType *lvlTypes, const uint64_t *dim2lvl,
    const uint64_t *lvl2dim, SparseTensorCOO<V> &lvlCOO)
    : SparseTensorStorage(dimRank, dimSizes, lvlRank,
                          lvlCOO.getDimSizes().data(), lvlTypes, dim2lvl,
                          lvl2dim, nullptr, false) {
  // Ensure lvlCOO is sorted.
  assert(lvlRank == lvlCOO.getDimSizes().size() && "Level-rank mismatch");
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
    const DimLevelType *lvlTypes, const uint64_t *dim2lvl,
    const uint64_t *lvl2dim, SparseTensorEnumeratorBase<V> &lvlEnumerator)
    : SparseTensorStorage(dimRank, dimSizes, lvlRank,
                          lvlEnumerator.getTrgSizes().data(), lvlTypes, dim2lvl,
                          lvl2dim) {
  assert(lvlRank == lvlEnumerator.getTrgRank() && "Level-rank mismatch");
  {
    // Initialize the statistics structure.
    SparseTensorNNZ nnz(getLvlSizes(), getLvlTypes());
    nnz.initialize(lvlEnumerator);
    // Initialize "positions" overhead (and allocate "coordinates", "values").
    uint64_t parentSz = 1; // assembled-size of the `(l - 1)`-level.
    for (uint64_t l = 0; l < lvlRank; ++l) {
      const auto dlt = lvlTypes[l]; // Avoid redundant bounds checking.
      if (isCompressedDLT(dlt)) {
        positions[l].reserve(parentSz + 1);
        positions[l].push_back(0);
        uint64_t currentPos = 0;
        nnz.forallCoords(l, [this, &currentPos, l](uint64_t n) {
          currentPos += n;
          appendPos(l, currentPos);
        });
        assert(positions[l].size() == parentSz + 1 &&
               "Final positions size doesn't match allocated size");
        // That assertion entails `assembledSize(parentSz, l)`
        // is now in a valid state.  That is, `positions[l][parentSz]`
        // equals the present value of `currentPos`, which is the
        // correct assembled-size for `coordinates[l]`.
      }
      // Update assembled-size for the next iteration.
      parentSz = assembledSize(parentSz, l);
      // Ideally we need only `coordinates[l].reserve(parentSz)`, however
      // the `std::vector` implementation forces us to initialize it too.
      // That is, in the yieldPos loop we need random-access assignment
      // to `coordinates[l]`; however, `std::vector`'s subscript-assignment
      // only allows assigning to already-initialized positions.
      if (isCompressedDLT(dlt) || isSingletonDLT(dlt))
        coordinates[l].resize(parentSz, 0);
      else
        assert(isDenseDLT(dlt));
    }
    values.resize(parentSz, 0); // Both allocate and zero-initialize.
  }
  // The yieldPos loop
  lvlEnumerator.forallElements([this](const auto &lvlCoords, V val) {
    uint64_t parentSz = 1, parentPos = 0;
    for (uint64_t lvlRank = getLvlRank(), l = 0; l < lvlRank; ++l) {
      const auto dlt = getLvlTypes()[l]; // Avoid redundant bounds checking.
      if (isCompressedDLT(dlt)) {
        // If `parentPos == parentSz` then it's valid as an array-lookup;
        // however, it's semantically invalid here since that entry
        // does not represent a segment of `coordinates[l]`.  Moreover, that
        // entry must be immutable for `assembledSize` to remain valid.
        assert(parentPos < parentSz);
        const uint64_t currentPos = positions[l][parentPos];
        // This increment won't overflow the `P` type, since it can't
        // exceed the original value of `positions[l][parentPos+1]`
        // which was already verified to be within bounds for `P`
        // when it was written to the array.
        positions[l][parentPos]++;
        writeCrd(l, currentPos, lvlCoords[l]);
        parentPos = currentPos;
      } else if (isSingletonDLT(dlt)) {
        writeCrd(l, parentPos, lvlCoords[l]);
        // the new parentPos equals the old parentPos.
      } else { // Dense level.
        assert(isDenseDLT(dlt));
        parentPos = parentPos * getLvlSizes()[l] + lvlCoords[l];
      }
      parentSz = assembledSize(parentSz, l);
    }
    assert(parentPos < values.size());
    values[parentPos] = val;
  });
  // The finalizeYieldPos loop
  for (uint64_t parentSz = 1, l = 0; l < lvlRank; ++l) {
    const auto dlt = lvlTypes[l]; // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt)) {
      assert(parentSz == positions[l].size() - 1 &&
             "Actual positions size doesn't match the expected size");
      // Can't check all of them, but at least we can check the last one.
      assert(positions[l][parentSz - 1] == positions[l][parentSz] &&
             "Positions got corrupted");
      for (uint64_t n = 0; n < parentSz; ++n) {
        const uint64_t parentPos = parentSz - n;
        positions[l][parentPos] = positions[l][parentPos - 1];
      }
      positions[l][0] = 0;
    } else {
      // Both dense and singleton are no-ops for the finalizeYieldPos loop.
      // This assertion is for future-proofing.
      assert((isDenseDLT(dlt) || isSingletonDLT(dlt)));
    }
    parentSz = assembledSize(parentSz, l);
  }
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
