//===- Storage.h - TACO-flavored sparse tensor representation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is part of the lightweight runtime support library for sparse
// tensor manipulations.  The functionality of the support library is meant
// to simplify benchmarking, testing, and debugging MLIR code operating on
// sparse tensors.  However, the provided functionality is **not** part of
// core MLIR itself.
//
// This file contains definitions for the following classes:
//
// * `SparseTensorStorageBase`
// * `SparseTensorStorage<P, I, V>`
// * `SparseTensorEnumeratorBase<V>`
// * `SparseTensorEnumerator<P, I, V>`
// * `SparseTensorNNZ`
//
// Ideally we would split the storage classes and enumerator classes
// into separate files, to improve legibility.  But alas: because these
// are template-classes, they must therefore provide *definitions* in the
// header; and those definitions cause circular dependencies that make it
// impossible to split the file up along the desired lines.  (We could
// split the base classes from the derived classes, but that doesn't
// particularly help improve legibility.)
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_STORAGE_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_STORAGE_H

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/ExecutionEngine/SparseTensor/Attributes.h"
#include "mlir/ExecutionEngine/SparseTensor/COO.h"
#include "mlir/ExecutionEngine/SparseTensor/CheckedMul.h"
#include "mlir/ExecutionEngine/SparseTensor/ErrorHandling.h"

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// This forward decl is sufficient to split `SparseTensorStorageBase` into
// its own header, but isn't sufficient for `SparseTensorStorage` to join it.
template <typename V>
class SparseTensorEnumeratorBase;

// These macros ensure consistent error messages, without risk of incuring
// an additional method call to do so.
#define ASSERT_VALID_LVL(l)                                                    \
  assert(l < getLvlRank() && "Level index is out of bounds");
#define ASSERT_COMPRESSED_LVL(l)                                               \
  assert(isCompressedLvl(l) && "Level is not compressed");
#define ASSERT_COMPRESSED_OR_SINGLETON_LVL(l)                                  \
  do {                                                                         \
    const DimLevelType dlt = getLvlType(l);                                    \
    (void)dlt;                                                                 \
    assert((isCompressedDLT(dlt) || isSingletonDLT(dlt)) &&                    \
           "Level is neither compressed nor singleton");                       \
  } while (false)
// Because the `SparseTensorStorageBase` ctor uses `MLIR_SPARSETENSOR_FATAL`
// (rather than `assert`) when validating level-types, all the uses of
// `ASSERT_DENSE_DLT` are technically unnecessary.  However, they are
// retained for the sake of future-proofing.
#define ASSERT_DENSE_DLT(dlt) assert(isDenseDLT(dlt) && "Level is not dense");

/// Abstract base class for `SparseTensorStorage<P,I,V>`.  This class
/// takes responsibility for all the `<P,I,V>`-independent aspects
/// of the tensor (e.g., shape, sparsity, permutation).  In addition,
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
/// representation itself.  We use this "dimension" vs "level" terminology
/// throughout, since alternative terminology like "tensor-dimension",
/// "original-dimension", "storage-dimension", etc, is both more verbose
/// and prone to introduce confusion whenever the qualifiers are dropped.
/// Where necessary, we use "axis" as the generic term.
///
/// The *size* of an axis is the cardinality of possible coordinate/index
/// values along that axis (regardless of which coordinates have stored
/// element values).  As such, each size must be non-zero since if any
/// axis has size-zero then the whole tensor would have trivial storage
/// (since there are no possible coordinates).  Thus we use the plural
/// term *sizes* for a collection of non-zero cardinalities, and use
/// this term whenever referring to run-time cardinalities.  Whereas we
/// use the term *shape* for a collection of compile-time cardinalities,
/// where zero is used to indicate cardinalities which are dynamic (i.e.,
/// unknown/unspecified at compile-time).  At run-time, these dynamic
/// cardinalities will be inferred from or checked against sizes otherwise
/// specified.  Thus, dynamic cardinalities always have an "immutable but
/// unknown" value; so the term "dynamic" should not be taken to indicate
/// run-time mutability.
//
// TODO: we'd like to factor out a class akin to `PermutationRef` for
// capturing known-valid sizes to avoid redundant validity assertions.
// But calling that class "SizesRef" would be a terrible name (and
// "ValidSizesRef" isn't much better).  Whereas, calling it "ShapeRef"
// would be a lot nicer, but then that conflicts with the terminology
// introduced above.  So we need to come up with some new terminology
// for distinguishing things, which allows a reasonable class name too.
class SparseTensorStorageBase {
protected:
  // Since this class is virtual, we must disallow public copying in
  // order to avoid "slicing".  Since this class has data members,
  // that means making copying protected.
  // <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-copy-virtual>
  SparseTensorStorageBase(const SparseTensorStorageBase &) = default;
  // Copy-assignment would be implicitly deleted (because our fields
  // are const), so we explicitly delete it for clarity.
  SparseTensorStorageBase &operator=(const SparseTensorStorageBase &) = delete;

public:
  /// Constructs a new sparse-tensor storage object with the given encoding.
  ///
  /// Preconditions:
  /// * `dimSizes`, `lvlSizes`, `lvlTypes`, and `lvl2dim` must be nonnull.
  /// * `dimSizes` must be valid for `dimRank`.
  /// * `lvlSizes`, `lvlTypes`, and `lvl2dim` must be valid for `lvlRank`.
  /// * `lvl2dim` must map indices valid for `lvlSizes` to indices valid
  ///   for `dimSizes`.
  ///
  /// Asserts:
  /// * `dimRank` and `lvlRank` are nonzero.
  /// * `dimSizes` and `lvlSizes` contain only nonzero sizes.
  SparseTensorStorageBase(uint64_t dimRank, const uint64_t *dimSizes,
                          uint64_t lvlRank, const uint64_t *lvlSizes,
                          const DimLevelType *lvlTypes,
                          const uint64_t *lvl2dim);
  // NOTE: For the most part we only need the `dimRank`.  But we need
  // `dimSizes` for `toCOO` to support the identity permutation nicely
  // (i.e., without the caller needing to already know the tensor's
  // dimension-sizes; e.g., as in `fromMLIRSparseTensor`).

  virtual ~SparseTensorStorageBase() = default;

  /// Gets the number of tensor-dimensions.
  uint64_t getDimRank() const { return dimSizes.size(); }

  /// Gets the number of storage-levels.
  uint64_t getLvlRank() const { return lvlSizes.size(); }

  /// Gets the tensor-dimension sizes array.
  const std::vector<uint64_t> &getDimSizes() const { return dimSizes; }

  /// Gets the storage-level sizes array.
  const std::vector<uint64_t> &getLvlSizes() const { return lvlSizes; }

  /// Safely looks up the size of the given storage-level.
  uint64_t getLvlSize(uint64_t l) const {
    ASSERT_VALID_LVL(l);
    return lvlSizes[l];
  }

  /// Gets the level-to-dimension mapping.
  const std::vector<uint64_t> &getLvl2Dim() const { return lvl2dim; }

  /// Gets the level-types array.
  const std::vector<DimLevelType> &getLvlTypes() const { return lvlTypes; }

  /// Safely looks up the type of the given level.
  DimLevelType getLvlType(uint64_t l) const {
    ASSERT_VALID_LVL(l);
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

  /// Allocates a new enumerator.  Callers must make sure to delete
  /// the enumerator when they're done with it.  The first argument
  /// is the out-parameter for storing the newly allocated enumerator;
  /// all other arguments are passed along to the `SparseTensorEnumerator`
  /// ctor and must satisfy the preconditions/assertions thereof.
#define DECL_NEWENUMERATOR(VNAME, V)                                           \
  virtual void newEnumerator(SparseTensorEnumeratorBase<V> **, uint64_t,       \
                             const uint64_t *, uint64_t, const uint64_t *)     \
      const;
  MLIR_SPARSETENSOR_FOREVERY_V(DECL_NEWENUMERATOR)
#undef DECL_NEWENUMERATOR

  /// Gets pointers-overhead storage.
#define DECL_GETPOINTERS(PNAME, P)                                             \
  virtual void getPointers(std::vector<P> **, uint64_t);
  MLIR_SPARSETENSOR_FOREVERY_FIXED_O(DECL_GETPOINTERS)
#undef DECL_GETPOINTERS

  /// Gets indices-overhead storage.
#define DECL_GETINDICES(INAME, I)                                              \
  virtual void getIndices(std::vector<I> **, uint64_t);
  MLIR_SPARSETENSOR_FOREVERY_FIXED_O(DECL_GETINDICES)
#undef DECL_GETINDICES
  virtual uint64_t getIndex(uint64_t l, uint64_t pos) const = 0;

  /// Gets primary storage.
#define DECL_GETVALUES(VNAME, V) virtual void getValues(std::vector<V> **);
  MLIR_SPARSETENSOR_FOREVERY_V(DECL_GETVALUES)
#undef DECL_GETVALUES

  /// Element-wise insertion in lexicographic index order.  The first
  /// argument is the level-indices for the value being inserted.
  // TODO: For better safety, this should take a parameter for the
  // length of `lvlInd` and check that against `getLvlRank()`.
#define DECL_LEXINSERT(VNAME, V) virtual void lexInsert(const uint64_t *, V);
  MLIR_SPARSETENSOR_FOREVERY_V(DECL_LEXINSERT)
#undef DECL_LEXINSERT

  /// Expanded insertion.  Note that this method resets the
  /// values/filled-switch array back to all-zero/false while only
  /// iterating over the nonzero elements.
  ///
  /// Arguments:
  /// * `lvlInd` the level-indices shared by the values being inserted.
  /// * `values` a map from last-level coordinates to their associated value.
  /// * `filled` a map from last-level coordinates to bool, indicating
  ///   whether `values` contains a valid value to be inserted.
  /// * `added` a map from `[0..count)` to last-level coordinates for
  ///   which `filled` is true and `values` contains the assotiated value.
  /// * `count` the size of `added`.
#define DECL_EXPINSERT(VNAME, V)                                               \
  virtual void expInsert(uint64_t *, V *, bool *, uint64_t *, uint64_t);
  MLIR_SPARSETENSOR_FOREVERY_V(DECL_EXPINSERT)
#undef DECL_EXPINSERT

  /// Finishes insertion.
  virtual void endInsert() = 0;

private:
  const std::vector<uint64_t> dimSizes;
  const std::vector<uint64_t> lvlSizes;
  const std::vector<DimLevelType> lvlTypes;
  const std::vector<uint64_t> lvl2dim;
};

//===----------------------------------------------------------------------===//
// This forward decl is necessary for defining `SparseTensorStorage`,
// but isn't sufficient for splitting it off.
template <typename P, typename I, typename V>
class SparseTensorEnumerator;

/// A memory-resident sparse tensor using a storage scheme based on
/// per-dimension sparse/dense annotations.  This data structure provides
/// a bufferized form of a sparse tensor type.  In contrast to generating
/// setup methods for each differently annotated sparse tensor, this
/// method provides a convenient "one-size-fits-all" solution that simply
/// takes an input tensor and annotations to implement all required setup
/// in a general manner.
template <typename P, typename I, typename V>
class SparseTensorStorage final : public SparseTensorStorageBase {
  /// Private constructor to share code between the other constructors.
  /// Beware that the object is not necessarily guaranteed to be in a
  /// valid state after this constructor alone; e.g., `isCompressedLvl(l)`
  /// doesn't entail `!(pointers[l].empty())`.
  ///
  /// Preconditions/assertions are as per the `SparseTensorStorageBase` ctor.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const uint64_t *lvlSizes,
                      const DimLevelType *lvlTypes, const uint64_t *lvl2dim)
      : SparseTensorStorageBase(dimRank, dimSizes, lvlRank, lvlSizes, lvlTypes,
                                lvl2dim),
        pointers(lvlRank), indices(lvlRank), lvlCursor(lvlRank) {}

public:
  /// Constructs a sparse tensor with the given encoding, and allocates
  /// overhead storage according to some simple heuristics.  When the
  /// `bool` argument is true and `lvlTypes` are all dense, then this
  /// ctor will also initialize the values array with zeros.  That
  /// argument should be true when an empty tensor is intended; whereas
  /// it should usually be false when the ctor will be followed up by
  /// some other form of initialization.
  ///
  /// Preconditions/assertions are as per the `SparseTensorStorageBase` ctor.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const uint64_t *lvlSizes,
                      const DimLevelType *lvlTypes, const uint64_t *lvl2dim,
                      bool initializeValuesIfAllDense);

  /// Constructs a sparse tensor with the given encoding, and initializes
  /// the contents from the COO.  This ctor performs the same heuristic
  /// overhead-storage allocation as the ctor taking a `bool`, and
  /// has the same preconditions/assertions (where we define `lvlSizes =
  /// lvlCOO.getDimSizes().data()`), with the following addition:
  ///
  /// Asserts:
  /// * `lvlRank == lvlCOO.getRank()`.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const DimLevelType *lvlTypes,
                      const uint64_t *lvl2dim, SparseTensorCOO<V> &lvlCOO);

  /// Constructs a sparse tensor with the given encoding, and initializes
  /// the contents from the enumerator.  This ctor allocates exactly
  /// the required amount of overhead storage, not using any heuristics.
  /// Preconditions/assertions are as per the `SparseTensorStorageBase`
  /// ctor (where we define `lvlSizes = lvlEnumerator.getTrgSizes().data()`),
  /// with the following addition:
  ///
  /// Asserts:
  /// * `lvlRank == lvlEnumerator.getTrgRank()`.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const DimLevelType *lvlTypes,
                      const uint64_t *lvl2dim,
                      SparseTensorEnumeratorBase<V> &lvlEnumerator);

  /// Allocates a new empty sparse tensor.  The preconditions/assertions
  /// are as per the `SparseTensorStorageBase` ctor; which is to say,
  /// the `dimSizes` and `lvlSizes` must both be "sizes" not "shapes",
  /// since there's nowhere to reconstruct dynamic sizes from.
  static SparseTensorStorage<P, I, V> *
  newEmpty(uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
           const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
           const uint64_t *lvl2dim) {
    return new SparseTensorStorage<P, I, V>(dimRank, dimSizes, lvlRank,
                                            lvlSizes, lvlTypes, lvl2dim, true);
  }

  /// Allocates a new sparse tensor and initializes it from the given COO.
  /// The preconditions are as per the `SparseTensorStorageBase` ctor
  /// (where we define `lvlSizes = lvlCOO.getDimSizes().data()`), but
  /// using the following assertions in lieu of the base ctor's assertions:
  ///
  /// Asserts:
  /// * `dimRank` and `lvlRank` are nonzero.
  /// * `lvlRank == lvlCOO.getRank()`.
  /// * `lvlCOO.getDimSizes()` under the `lvl2dim` mapping is a refinement
  ///   of `dimShape`.
  //
  // TODO: The ability to reconstruct dynamic dimensions-sizes does not
  // easily generalize to arbitrary `lvl2dim` mappings.  When compiling
  // MLIR programs to use this library, we should be able to generate
  // code for effectively computing the reconstruction, but it's not clear
  // that there's a feasible way to do so from within the library itself.
  // Therefore, when we functionalize the `lvl2dim` mapping we'll have
  // to update the type/preconditions of this factory too.
  static SparseTensorStorage<P, I, V> *
  newFromCOO(uint64_t dimRank, const uint64_t *dimShape, uint64_t lvlRank,
             const DimLevelType *lvlTypes, const uint64_t *lvl2dim,
             SparseTensorCOO<V> &lvlCOO);

  /// Allocates a new sparse tensor and initializes it with the contents
  /// of another sparse tensor.
  ///
  /// Preconditions:
  /// * as per the `SparseTensorStorageBase` ctor.
  /// * `src2lvl` must be valid for `srcRank`, must map indices valid for
  ///   `source.getDimSizes()` to indices valid for `lvlSizes`, and therefore
  ///   must be the inverse of `lvl2dim`.
  /// * `source` must have the same value type `V`.
  ///
  /// Asserts:
  /// * `dimRank` and `lvlRank` are nonzero.
  /// * `srcRank == source.getDimRank()`.
  /// * `lvlSizes` contains only nonzero sizes.
  /// * `source.getDimSizes()` is a refinement of `dimShape`.
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
  static SparseTensorStorage<P, I, V> *
  newFromSparseTensor(uint64_t dimRank, const uint64_t *dimShape,
                      uint64_t lvlRank, const uint64_t *lvlSizes,
                      const DimLevelType *lvlTypes, const uint64_t *lvl2dim,
                      uint64_t srcRank, const uint64_t *src2lvl,
                      const SparseTensorStorageBase &source);

  ~SparseTensorStorage() final = default;

  /// Partially specialize these getter methods based on template types.
  void getPointers(std::vector<P> **out, uint64_t l) final {
    assert(out && "Received nullptr for out parameter");
    ASSERT_VALID_LVL(l);
    *out = &pointers[l];
  }
  void getIndices(std::vector<I> **out, uint64_t l) final {
    assert(out && "Received nullptr for out parameter");
    ASSERT_VALID_LVL(l);
    *out = &indices[l];
  }
  void getValues(std::vector<V> **out) final {
    assert(out && "Received nullptr for out parameter");
    *out = &values;
  }

  uint64_t getIndex(uint64_t l, uint64_t pos) const final {
    ASSERT_COMPRESSED_OR_SINGLETON_LVL(l);
    assert(pos < indices[l].size() && "Index position is out of bounds");
    return indices[l][pos]; // Converts the stored `I` into `uint64_t`.
  }

  /// Partially specialize lexicographical insertions based on template types.
  void lexInsert(const uint64_t *lvlInd, V val) final {
    assert(lvlInd && "Received nullptr for level-indices");
    // First, wrap up pending insertion path.
    uint64_t diffLvl = 0;
    uint64_t topIdx = 0;
    if (!values.empty()) {
      diffLvl = lexDiff(lvlInd);
      endPath(diffLvl + 1);
      topIdx = lvlCursor[diffLvl] + 1;
    }
    // Then continue with insertion path.
    insPath(lvlInd, diffLvl, topIdx, val);
  }

  /// Partially specialize expanded insertions based on template types.
  void expInsert(uint64_t *lvlInd, V *values, bool *filled, uint64_t *added,
                 uint64_t count) final {
    assert((lvlInd && values && filled && added) && "Received nullptr");
    if (count == 0)
      return;
    // Sort.
    std::sort(added, added + count);
    // Restore insertion path for first insert.
    const uint64_t lastLvl = getLvlRank() - 1;
    uint64_t index = added[0];
    assert(filled[index] && "added index is not filled");
    lvlInd[lastLvl] = index;
    lexInsert(lvlInd, values[index]);
    values[index] = 0;
    filled[index] = false;
    // Subsequent insertions are quick.
    for (uint64_t i = 1; i < count; ++i) {
      assert(index < added[i] && "non-lexicographic insertion");
      index = added[i];
      assert(filled[index] && "added index is not filled");
      lvlInd[lastLvl] = index;
      insPath(lvlInd, lastLvl, added[i - 1] + 1, values[index]);
      values[index] = 0;
      filled[index] = false;
    }
  }

  /// Finalizes lexicographic insertions.
  void endInsert() final {
    if (values.empty())
      finalizeSegment(0);
    else
      endPath(0);
  }

  /// Allocates a new enumerator for this class's `<P,I,V>` types and
  /// erase the `<P,I>` parts from the type.  Callers must make sure to
  /// delete the enumerator when they're done with it.
  void newEnumerator(SparseTensorEnumeratorBase<V> **out, uint64_t trgRank,
                     const uint64_t *trgSizes, uint64_t srcRank,
                     const uint64_t *src2trg) const final {
    assert(out && "Received nullptr for out parameter");
    *out = new SparseTensorEnumerator<P, I, V>(*this, trgRank, trgSizes,
                                               srcRank, src2trg);
  }

  /// Allocates a new COO object and initializes it with the contents
  /// of this tensor under the given mapping from the `getDimSizes()`
  /// coordinate-space to the `trgSizes` coordinate-space.  Callers must
  /// make sure to delete the COO when they're done with it.
  ///
  /// Preconditions/assertions are as per the `SparseTensorEnumerator` ctor.
  SparseTensorCOO<V> *toCOO(uint64_t trgRank, const uint64_t *trgSizes,
                            uint64_t srcRank, const uint64_t *src2trg) const {
    // We inline `newEnumerator` to avoid virtual dispatch and allocation.
    SparseTensorEnumerator<P, I, V> enumerator(*this, trgRank, trgSizes,
                                               srcRank, src2trg);
    auto *coo = new SparseTensorCOO<V>(trgRank, trgSizes, values.size());
    enumerator.forallElements(
        [&coo](const auto &trgInd, V val) { coo->add(trgInd, val); });
    // TODO: This assertion assumes there are no stored zeros,
    // or if there are then that we don't filter them out.
    // Cf., <https://github.com/llvm/llvm-project/issues/54179>
    assert(coo->getElements().size() == values.size());
    return coo;
  }

private:
  /// Appends an arbitrary new position to `pointers[l]`.  This method
  /// checks that `pos` is representable in the `P` type; however, it
  /// does not check that `pos` is semantically valid (i.e., larger than
  /// the previous position and smaller than `indices[l].capacity()`).
  void appendPointer(uint64_t l, uint64_t pos, uint64_t count = 1) {
    ASSERT_COMPRESSED_LVL(l);
    assert(pos <= std::numeric_limits<P>::max() &&
           "Pointer value is too large for the P-type");
    pointers[l].insert(pointers[l].end(), count, static_cast<P>(pos));
  }

  /// Appends index `i` to level `l`, in the semantically general sense.
  /// For non-dense levels, that means appending to the `indices[l]` array,
  /// checking that `i` is representable in the `I` type; however, we do
  /// not verify other semantic requirements (e.g., that `i` is in bounds
  /// for `lvlSizes[l]`, and not previously occurring in the same segment).
  /// For dense levels, this method instead appends the appropriate number
  /// of zeros to the `values` array, where `full` is the number of "entries"
  /// already written to `values` for this segment (aka one after the highest
  /// index previously appended).
  void appendIndex(uint64_t l, uint64_t full, uint64_t i) {
    const auto dlt = getLvlType(l); // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt) || isSingletonDLT(dlt)) {
      assert(i <= std::numeric_limits<I>::max() &&
             "Index value is too large for the I-type");
      indices[l].push_back(static_cast<I>(i));
    } else { // Dense dimension.
      ASSERT_DENSE_DLT(dlt);
      assert(i >= full && "Index was already filled");
      if (i == full)
        return; // Short-circuit, since it'll be a nop.
      if (l + 1 == getLvlRank())
        values.insert(values.end(), i - full, 0);
      else
        finalizeSegment(l + 1, 0, i - full);
    }
  }

  /// Writes the given coordinate to `indices[l][pos]`.  This method
  /// checks that `i` is representable in the `I` type; however, it
  /// does not check that `i` is semantically valid (i.e., in bounds
  /// for `dimSizes[l]` and not elsewhere occurring in the same segment).
  void writeIndex(uint64_t l, uint64_t pos, uint64_t i) {
    ASSERT_COMPRESSED_OR_SINGLETON_LVL(l);
    // Subscript assignment to `std::vector` requires that the `pos`-th
    // entry has been initialized; thus we must be sure to check `size()`
    // here, instead of `capacity()` as would be ideal.
    assert(pos < indices[l].size() && "Index position is out of bounds");
    assert(i <= std::numeric_limits<I>::max() &&
           "Index value is too large for the I-type");
    indices[l][pos] = static_cast<I>(i);
  }

  /// Computes the assembled-size associated with the `l`-th level,
  /// given the assembled-size associated with the `(l-1)`-th level.
  /// "Assembled-sizes" correspond to the (nominal) sizes of overhead
  /// storage, as opposed to "level-sizes" which are the cardinality
  /// of possible coordinates for that level.
  ///
  /// Precondition: the `pointers[l]` array must be fully initialized
  /// before calling this method.
  uint64_t assembledSize(uint64_t parentSz, uint64_t l) const {
    const auto dlt = getLvlType(l); // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt))
      return pointers[l][parentSz];
    if (isSingletonDLT(dlt))
      return parentSz; // New size is same as the parent.
    if (isDenseDLT(dlt))
      return parentSz * getLvlSizes()[l];
    MLIR_SPARSETENSOR_FATAL("unsupported level type: %d\n",
                            static_cast<uint8_t>(dlt));
  }

  /// Initializes sparse tensor storage scheme from a memory-resident sparse
  /// tensor in coordinate scheme. This method prepares the pointers and
  /// indices arrays under the given per-dimension dense/sparse annotations.
  ///
  /// Preconditions:
  /// * the `lvlElements` must be lexicographically sorted.
  /// * the indices of every element are valid for `getLvlSizes()`
  ///   (i.e., equal rank and pointwise less-than).
  void fromCOO(const std::vector<Element<V>> &lvlElements, uint64_t lo,
               uint64_t hi, uint64_t l) {
    const uint64_t lvlRank = getLvlRank();
    assert(l <= lvlRank && hi <= lvlElements.size());
    // Once dimensions are exhausted, insert the numerical values.
    if (l == lvlRank) {
      assert(lo < hi);
      values.push_back(lvlElements[lo].value);
      return;
    }
    // Visit all elements in this interval.
    uint64_t full = 0;
    while (lo < hi) { // If `hi` is unchanged, then `lo < lvlElements.size()`.
      // Find segment in interval with same index elements in this level.
      const uint64_t i = lvlElements[lo].indices[l];
      uint64_t seg = lo + 1;
      if (isUniqueLvl(l))
        while (seg < hi && lvlElements[seg].indices[l] == i)
          ++seg;
      // Handle segment in interval for sparse or dense level.
      appendIndex(l, full, i);
      full = i + 1;
      fromCOO(lvlElements, lo, seg, l + 1);
      // And move on to next segment in interval.
      lo = seg;
    }
    // Finalize the sparse pointer structure at this level.
    finalizeSegment(l, full);
  }

  /// Finalizes the sparse pointer structure at this level.
  void finalizeSegment(uint64_t l, uint64_t full = 0, uint64_t count = 1) {
    if (count == 0)
      return; // Short-circuit, since it'll be a nop.
    const auto dlt = getLvlType(l); // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt)) {
      appendPointer(l, indices[l].size(), count);
    } else if (isSingletonDLT(dlt)) {
      return; // Nothing to finalize.
    } else {  // Dense dimension.
      ASSERT_DENSE_DLT(dlt);
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
    assert(diffLvl <= lvlRank && "Level-diff is out of bounds");
    const uint64_t stop = lvlRank - diffLvl;
    for (uint64_t i = 0; i < stop; ++i) {
      const uint64_t l = lastLvl - i;
      finalizeSegment(l, lvlCursor[l] + 1);
    }
  }

  /// Continues a single insertion path, outer to inner.  The first
  /// argument is the storage-level indices for the value being inserted.
  void insPath(const uint64_t *lvlInd, uint64_t diffLvl, uint64_t topIdx,
               V val) {
    const uint64_t lvlRank = getLvlRank();
    assert(diffLvl <= lvlRank && "Level-diff is out of bounds");
    for (uint64_t l = diffLvl; l < lvlRank; ++l) {
      const uint64_t i = lvlInd[l];
      appendIndex(l, topIdx, i);
      topIdx = 0;
      lvlCursor[l] = i;
    }
    values.push_back(val);
  }

  /// Finds the lexicographically first level where the level-indices
  /// in the argument differ from those in the current cursor.
  uint64_t lexDiff(const uint64_t *lvlInd) const {
    const uint64_t lvlRank = getLvlRank();
    for (uint64_t l = 0; l < lvlRank; ++l)
      if (lvlInd[l] > lvlCursor[l])
        return l;
      else
        assert(lvlInd[l] == lvlCursor[l] && "non-lexicographic insertion");
    assert(0 && "duplicate insertion");
    return -1u;
  }

  // Allow `SparseTensorEnumerator` to access the data-members (to avoid
  // the cost of virtual-function dispatch in inner loops), without
  // making them public to other client code.
  friend class SparseTensorEnumerator<P, I, V>;

  std::vector<std::vector<P>> pointers;
  std::vector<std::vector<I>> indices;
  std::vector<V> values;
  std::vector<uint64_t> lvlCursor; // cursor for lexicographic insertion.
};

#undef ASSERT_COMPRESSED_OR_SINGLETON_LVL
#undef ASSERT_COMPRESSED_LVL
#undef ASSERT_VALID_LVL

//===----------------------------------------------------------------------===//
/// A (higher-order) function object for enumerating the elements of some
/// `SparseTensorStorage` under a permutation.  That is, the `forallElements`
/// method encapsulates the loop-nest for enumerating the elements of
/// the source tensor (in whatever order is best for the source tensor),
/// and applies a permutation to the coordinates/indices before handing
/// each element to the callback.  A single enumerator object can be
/// freely reused for several calls to `forallElements`, just so long
/// as each call is sequential with respect to one another.
///
/// N.B., this class stores a reference to the `SparseTensorStorageBase`
/// passed to the constructor; thus, objects of this class must not
/// outlive the sparse tensor they depend on.
///
/// Design Note: The reason we define this class instead of simply using
/// `SparseTensorEnumerator<P,I,V>` is because we need to hide/generalize
/// the `<P,I>` template parameters from MLIR client code (to simplify the
/// type parameters used for direct sparse-to-sparse conversion).  And the
/// reason we define the `SparseTensorEnumerator<P,I,V>` subclasses rather
/// than simply using this class, is to avoid the cost of virtual-method
/// dispatch within the loop-nest.
template <typename V>
class MLIR_SPARSETENSOR_GSL_POINTER [[nodiscard]] SparseTensorEnumeratorBase {
public:
  /// Constructs an enumerator which automatically applies the given
  /// mapping from the source tensor's dimensions to the desired
  /// target tensor dimensions.
  ///
  /// Preconditions:
  /// * the `src` must have the same `V` value type.
  /// * `trgSizes` must be valid for `trgRank`.
  /// * `src2trg` must be valid for `srcRank`, and must map indices
  ///   valid for `src.getDimSizes()` to indices valid for `trgSizes`.
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
  /// indices, and passes the permuted element to the callback.
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

//===----------------------------------------------------------------------===//
template <typename P, typename I, typename V>
class MLIR_SPARSETENSOR_GSL_POINTER [[nodiscard]] SparseTensorEnumerator final
    : public SparseTensorEnumeratorBase<V> {
  using Base = SparseTensorEnumeratorBase<V>;
  using StorageImpl = SparseTensorStorage<P, I, V>;

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
    // Recover the `<P,I,V>` type parameters of `src`.
    const auto &src = static_cast<const StorageImpl &>(this->src);
    if (l == src.getLvlRank()) {
      assert(parentPos < src.values.size() &&
             "Value position is out of bounds");
      // TODO: <https://github.com/llvm/llvm-project/issues/54179>
      yield(this->trgCursor, src.values[parentPos]);
      return;
    }
    uint64_t &cursorL = this->trgCursor[this->lvl2trg[l]];
    const auto dlt = src.getLvlType(l); // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt)) {
      // Look up the bounds of the `l`-level segment determined by the
      // `(l - 1)`-level position `parentPos`.
      const std::vector<P> &pointersL = src.pointers[l];
      assert(parentPos + 1 < pointersL.size() &&
             "Parent pointer position is out of bounds");
      const uint64_t pstart = static_cast<uint64_t>(pointersL[parentPos]);
      const uint64_t pstop = static_cast<uint64_t>(pointersL[parentPos + 1]);
      // Loop-invariant code for looking up the `l`-level coordinates/indices.
      const std::vector<I> &indicesL = src.indices[l];
      assert(pstop <= indicesL.size() && "Index position is out of bounds");
      for (uint64_t pos = pstart; pos < pstop; ++pos) {
        cursorL = static_cast<uint64_t>(indicesL[pos]);
        forallElements(yield, pos, l + 1);
      }
    } else if (isSingletonDLT(dlt)) {
      cursorL = src.getIndex(l, parentPos);
      forallElements(yield, parentPos, l + 1);
    } else { // Dense dimension.
      ASSERT_DENSE_DLT(dlt);
      const uint64_t sz = src.getLvlSizes()[l];
      const uint64_t pstart = parentPos * sz;
      for (uint64_t i = 0; i < sz; ++i) {
        cursorL = i;
        forallElements(yield, pstart + i, l + 1);
      }
    }
  }
};

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
class MLIR_SPARSETENSOR_GSL_POINTER [[nodiscard]] SparseTensorNNZ final {
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
        [this](const std::vector<uint64_t> &ind, V) { add(ind); });
  }

  /// The type of callback functions which receive an nnz-statistic.
  using NNZConsumer = const std::function<void(uint64_t)> &;

  /// Lexicographically enumerates all indicies for levels strictly
  /// less than `stopLvl`, and passes their nnz statistic to the callback.
  /// Since our use-case only requires the statistic not the coordinates
  /// themselves, we do not bother to construct those coordinates.
  void forallIndices(uint64_t stopLvl, NNZConsumer yield) const;

private:
  /// Adds a new element (i.e., increment its statistics).  We use
  /// a method rather than inlining into the lambda in `initialize`,
  /// to avoid spurious templating over `V`.  And this method is private
  /// to avoid needing to re-assert validity of `lvlInd` (which is
  /// guaranteed by `forallElements`).
  void add(const std::vector<uint64_t> &lvlInd);

  /// Recursive component of the public `forallIndices`.
  void forallIndices(NNZConsumer yield, uint64_t stopLvl, uint64_t parentPos,
                     uint64_t l) const;

  // All of these are in the target storage-order.
  const std::vector<uint64_t> &lvlSizes;
  const std::vector<DimLevelType> &lvlTypes;
  std::vector<std::vector<uint64_t>> nnz;
};

//===----------------------------------------------------------------------===//
// Definitions of the ctors and factories of `SparseTensorStorage<P,I,V>`.

template <typename P, typename I, typename V>
SparseTensorStorage<P, I, V> *SparseTensorStorage<P, I, V>::newFromCOO(
    uint64_t dimRank, const uint64_t *dimShape, uint64_t lvlRank,
    const DimLevelType *lvlTypes, const uint64_t *lvl2dim,
    SparseTensorCOO<V> &lvlCOO) {
  assert(dimShape && "Got nullptr for dimension shape");
  assert(lvl2dim && "Got nullptr for level-to-dimension mapping");
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
  return new SparseTensorStorage<P, I, V>(dimRank, dimSizes.data(), lvlRank,
                                          lvlTypes, lvl2dim, lvlCOO);
}

template <typename P, typename I, typename V>
SparseTensorStorage<P, I, V> *SparseTensorStorage<P, I, V>::newFromSparseTensor(
    uint64_t dimRank, const uint64_t *dimShape, uint64_t lvlRank,
    const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
    const uint64_t *lvl2dim, uint64_t srcRank, const uint64_t *src2lvl,
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
  auto *tensor = new SparseTensorStorage<P, I, V>(
      dimRank, dimSizes.data(), lvlRank, lvlTypes, lvl2dim, *lvlEnumerator);
  delete lvlEnumerator;
  return tensor;
}

template <typename P, typename I, typename V>
SparseTensorStorage<P, I, V>::SparseTensorStorage(
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
    const uint64_t *lvl2dim, bool initializeValuesIfAllDense)
    : SparseTensorStorage(dimRank, dimSizes, lvlRank, lvlSizes, lvlTypes,
                          lvl2dim) {
  // Provide hints on capacity of pointers and indices.
  // TODO: needs much fine-tuning based on actual sparsity; currently
  //       we reserve pointer/index space based on all previous dense
  //       dimensions, which works well up to first sparse dim; but
  //       we should really use nnz and dense/sparse distribution.
  bool allDense = true;
  uint64_t sz = 1;
  for (uint64_t l = 0; l < lvlRank; ++l) {
    const DimLevelType dlt = lvlTypes[l]; // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt)) {
      // TODO: Take a parameter between 1 and `lvlSizes[l]`, and multiply
      // `sz` by that before reserving. (For now we just use 1.)
      pointers[l].reserve(sz + 1);
      pointers[l].push_back(0);
      indices[l].reserve(sz);
      sz = 1;
      allDense = false;
    } else if (isSingletonDLT(dlt)) {
      indices[l].reserve(sz);
      sz = 1;
      allDense = false;
    } else { // Dense dimension.
      ASSERT_DENSE_DLT(dlt);
      sz = detail::checkedMul(sz, lvlSizes[l]);
    }
  }
  if (allDense && initializeValuesIfAllDense)
    values.resize(sz, 0);
}

template <typename P, typename I, typename V>
SparseTensorStorage<P, I, V>::SparseTensorStorage( // NOLINT
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const DimLevelType *lvlTypes, const uint64_t *lvl2dim,
    SparseTensorCOO<V> &lvlCOO)
    : SparseTensorStorage(dimRank, dimSizes, lvlRank,
                          lvlCOO.getDimSizes().data(), lvlTypes, lvl2dim,
                          false) {
  assert(lvlRank == lvlCOO.getDimSizes().size() && "Level-rank mismatch");
  // Ensure the preconditions of `fromCOO`.  (One is already ensured by
  // using `lvlSizes = lvlCOO.getDimSizes()` in the ctor above.)
  lvlCOO.sort();
  // Now actually insert the `elements`.
  const auto &elements = lvlCOO.getElements();
  uint64_t nnz = elements.size();
  values.reserve(nnz);
  fromCOO(elements, 0, nnz, 0);
}

template <typename P, typename I, typename V>
SparseTensorStorage<P, I, V>::SparseTensorStorage(
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const DimLevelType *lvlTypes, const uint64_t *lvl2dim,
    SparseTensorEnumeratorBase<V> &lvlEnumerator)
    : SparseTensorStorage(dimRank, dimSizes, lvlRank,
                          lvlEnumerator.getTrgSizes().data(), lvlTypes,
                          lvl2dim) {
  assert(lvlRank == lvlEnumerator.getTrgRank() && "Level-rank mismatch");
  {
    // Initialize the statistics structure.
    SparseTensorNNZ nnz(getLvlSizes(), getLvlTypes());
    nnz.initialize(lvlEnumerator);
    // Initialize "pointers" overhead (and allocate "indices", "values").
    uint64_t parentSz = 1; // assembled-size of the `(l - 1)`-level.
    for (uint64_t l = 0; l < lvlRank; ++l) {
      const auto dlt = lvlTypes[l]; // Avoid redundant bounds checking.
      if (isCompressedDLT(dlt)) {
        pointers[l].reserve(parentSz + 1);
        pointers[l].push_back(0);
        uint64_t currentPos = 0;
        nnz.forallIndices(l, [this, &currentPos, l](uint64_t n) {
          currentPos += n;
          appendPointer(l, currentPos);
        });
        assert(pointers[l].size() == parentSz + 1 &&
               "Final pointers size doesn't match allocated size");
        // That assertion entails `assembledSize(parentSz, l)`
        // is now in a valid state.  That is, `pointers[l][parentSz]`
        // equals the present value of `currentPos`, which is the
        // correct assembled-size for `indices[l]`.
      }
      // Update assembled-size for the next iteration.
      parentSz = assembledSize(parentSz, l);
      // Ideally we need only `indices[l].reserve(parentSz)`, however
      // the `std::vector` implementation forces us to initialize it too.
      // That is, in the yieldPos loop we need random-access assignment
      // to `indices[l]`; however, `std::vector`'s subscript-assignment
      // only allows assigning to already-initialized positions.
      if (isCompressedDLT(dlt) || isSingletonDLT(dlt))
        indices[l].resize(parentSz, 0);
      else
        ASSERT_DENSE_DLT(dlt); // Future-proofing.
    }
    values.resize(parentSz, 0); // Both allocate and zero-initialize.
  }
  // The yieldPos loop
  lvlEnumerator.forallElements([this](const auto &lvlInd, V val) {
    uint64_t parentSz = 1, parentPos = 0;
    for (uint64_t lvlRank = getLvlRank(), l = 0; l < lvlRank; ++l) {
      const auto dlt = getLvlTypes()[l]; // Avoid redundant bounds checking.
      if (isCompressedDLT(dlt)) {
        // If `parentPos == parentSz` then it's valid as an array-lookup;
        // however, it's semantically invalid here since that entry
        // does not represent a segment of `indices[l]`.  Moreover, that
        // entry must be immutable for `assembledSize` to remain valid.
        assert(parentPos < parentSz && "Pointers position is out of bounds");
        const uint64_t currentPos = pointers[l][parentPos];
        // This increment won't overflow the `P` type, since it can't
        // exceed the original value of `pointers[l][parentPos+1]`
        // which was already verified to be within bounds for `P`
        // when it was written to the array.
        pointers[l][parentPos]++;
        writeIndex(l, currentPos, lvlInd[l]);
        parentPos = currentPos;
      } else if (isSingletonDLT(dlt)) {
        writeIndex(l, parentPos, lvlInd[l]);
        // the new parentPos equals the old parentPos.
      } else { // Dense dimension.
        ASSERT_DENSE_DLT(dlt);
        parentPos = parentPos * getLvlSizes()[l] + lvlInd[l];
      }
      parentSz = assembledSize(parentSz, l);
    }
    assert(parentPos < values.size() && "Value position is out of bounds");
    values[parentPos] = val;
  });
  // The finalizeYieldPos loop
  for (uint64_t parentSz = 1, l = 0; l < lvlRank; ++l) {
    const auto dlt = lvlTypes[l]; // Avoid redundant bounds checking.
    if (isCompressedDLT(dlt)) {
      assert(parentSz == pointers[l].size() - 1 &&
             "Actual pointers size doesn't match the expected size");
      // Can't check all of them, but at least we can check the last one.
      assert(pointers[l][parentSz - 1] == pointers[l][parentSz] &&
             "Pointers got corrupted");
      // TODO: optimize this by using `memmove` or similar.
      for (uint64_t n = 0; n < parentSz; ++n) {
        const uint64_t parentPos = parentSz - n;
        pointers[l][parentPos] = pointers[l][parentPos - 1];
      }
      pointers[l][0] = 0;
    } else {
      // Both dense and singleton are no-ops for the finalizeYieldPos loop.
      // This assertion is for future-proofing.
      assert((isDenseDLT(dlt) || isSingletonDLT(dlt)) &&
             "Level is neither dense nor singleton");
    }
    parentSz = assembledSize(parentSz, l);
  }
}

#undef ASSERT_DENSE_DLT

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_STORAGE_H
