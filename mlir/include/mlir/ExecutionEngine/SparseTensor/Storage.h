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

#include "mlir/ExecutionEngine/SparseTensor/COO.h"
#include "mlir/ExecutionEngine/SparseTensor/CheckedMul.h"
#include "mlir/ExecutionEngine/SparseTensor/Enums.h"
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
#define ASSERT_VALID_DIM(d)                                                    \
  assert(d < getRank() && "Dimension index is out of bounds");
#define ASSERT_COMPRESSED_DIM(d)                                               \
  assert(isCompressedDim(d) && "Dimension is not compressed");
#define ASSERT_DENSE_DIM(d) assert(isDenseDim(d) && "Dimension is not dense");

/// Abstract base class for `SparseTensorStorage<P,I,V>`.  This class
/// takes responsibility for all the `<P,I,V>`-independent aspects
/// of the tensor (e.g., shape, sparsity, permutation).  In addition,
/// we use function overloading to implement "partial" method
/// specialization, which the C-API relies on to catch type errors
/// arising from our use of opaque pointers.
class SparseTensorStorageBase {
protected:
  // Since this class is virtual, we must disallow public copying in
  // order to avoid "slicing".  Since this class has data members,
  // that means making copying protected.
  // <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-copy-virtual>
  SparseTensorStorageBase(const SparseTensorStorageBase &) = default;
  // Copy-assignment would be implicitly deleted (because `dimSizes`
  // is const), so we explicitly delete it for clarity.
  SparseTensorStorageBase &operator=(const SparseTensorStorageBase &) = delete;

public:
  /// Constructs a new storage object.  The `perm` maps the tensor's
  /// semantic-ordering of dimensions to this object's storage-order.
  /// The `dimSizes` and `sparsity` arrays are already in storage-order.
  ///
  /// Precondition: `perm` and `sparsity` must be valid for `dimSizes.size()`.
  SparseTensorStorageBase(const std::vector<uint64_t> &dimSizes,
                          const uint64_t *perm, const DimLevelType *sparsity);

  virtual ~SparseTensorStorageBase() = default;

  /// Get the rank of the tensor.
  uint64_t getRank() const { return dimSizes.size(); }

  /// Getter for the dimension-sizes array, in storage-order.
  const std::vector<uint64_t> &getDimSizes() const { return dimSizes; }

  /// Safely lookup the size of the given (storage-order) dimension.
  uint64_t getDimSize(uint64_t d) const {
    ASSERT_VALID_DIM(d);
    return dimSizes[d];
  }

  /// Getter for the "reverse" permutation, which maps this object's
  /// storage-order to the tensor's semantic-order.
  const std::vector<uint64_t> &getRev() const { return rev; }

  /// Getter for the dimension-types array, in storage-order.
  const std::vector<DimLevelType> &getDimTypes() const { return dimTypes; }

  /// Safely check if the (storage-order) dimension uses dense storage.
  bool isDenseDim(uint64_t d) const {
    ASSERT_VALID_DIM(d);
    return dimTypes[d] == DimLevelType::kDense;
  }

  /// Safely check if the (storage-order) dimension uses compressed storage.
  bool isCompressedDim(uint64_t d) const {
    ASSERT_VALID_DIM(d);
    switch (dimTypes[d]) {
    case DimLevelType::kCompressed:
    case DimLevelType::kCompressedNu:
    case DimLevelType::kCompressedNo:
    case DimLevelType::kCompressedNuNo:
      return true;
    default:
      return false;
    }
  }

  /// Safely check if the (storage-order) dimension uses singleton storage.
  bool isSingletonDim(uint64_t d) const {
    ASSERT_VALID_DIM(d);
    switch (dimTypes[d]) {
    case DimLevelType::kSingleton:
    case DimLevelType::kSingletonNu:
    case DimLevelType::kSingletonNo:
    case DimLevelType::kSingletonNuNo:
      return true;
    default:
      return false;
    }
  }

  /// Safely check if the (storage-order) dimension is ordered.
  bool isOrderedDim(uint64_t d) const {
    ASSERT_VALID_DIM(d);
    switch (dimTypes[d]) {
    case DimLevelType::kCompressedNo:
    case DimLevelType::kCompressedNuNo:
    case DimLevelType::kSingletonNo:
    case DimLevelType::kSingletonNuNo:
      return false;
    default:
      return true;
    }
  }

  /// Safely check if the (storage-order) dimension is unique.
  bool isUniqueDim(uint64_t d) const {
    ASSERT_VALID_DIM(d);
    switch (dimTypes[d]) {
    case DimLevelType::kCompressedNu:
    case DimLevelType::kCompressedNuNo:
    case DimLevelType::kSingletonNu:
    case DimLevelType::kSingletonNuNo:
      return false;
    default:
      return true;
    }
  }

  /// Allocate a new enumerator.
#define DECL_NEWENUMERATOR(VNAME, V)                                           \
  virtual void newEnumerator(SparseTensorEnumeratorBase<V> **, uint64_t,       \
                             const uint64_t *) const;
  FOREVERY_V(DECL_NEWENUMERATOR)
#undef DECL_NEWENUMERATOR

  /// Overhead storage.
#define DECL_GETPOINTERS(PNAME, P)                                             \
  virtual void getPointers(std::vector<P> **, uint64_t);
  FOREVERY_FIXED_O(DECL_GETPOINTERS)
#undef DECL_GETPOINTERS
#define DECL_GETINDICES(INAME, I)                                              \
  virtual void getIndices(std::vector<I> **, uint64_t);
  FOREVERY_FIXED_O(DECL_GETINDICES)
#undef DECL_GETINDICES

  /// Primary storage.
#define DECL_GETVALUES(VNAME, V) virtual void getValues(std::vector<V> **);
  FOREVERY_V(DECL_GETVALUES)
#undef DECL_GETVALUES

  /// Element-wise insertion in lexicographic index order.
#define DECL_LEXINSERT(VNAME, V) virtual void lexInsert(const uint64_t *, V);
  FOREVERY_V(DECL_LEXINSERT)
#undef DECL_LEXINSERT

  /// Expanded insertion.
#define DECL_EXPINSERT(VNAME, V)                                               \
  virtual void expInsert(uint64_t *, V *, bool *, uint64_t *, uint64_t);
  FOREVERY_V(DECL_EXPINSERT)
#undef DECL_EXPINSERT

  /// Finishes insertion.
  virtual void endInsert() = 0;

private:
  const std::vector<uint64_t> dimSizes;
  std::vector<uint64_t> rev; // conceptually `const`
  const std::vector<DimLevelType> dimTypes;
};

//===----------------------------------------------------------------------===//
// This forward decl is necessary for defining `SparseTensorStorage`,
// but isn't sufficient for splitting it off.
template <typename P, typename I, typename V>
class SparseTensorEnumerator;

/// A memory-resident sparse tensor using a storage scheme based on
/// per-dimension sparse/dense annotations. This data structure provides a
/// bufferized form of a sparse tensor type. In contrast to generating setup
/// methods for each differently annotated sparse tensor, this method provides
/// a convenient "one-size-fits-all" solution that simply takes an input tensor
/// and annotations to implement all required setup in a general manner.
template <typename P, typename I, typename V>
class SparseTensorStorage final : public SparseTensorStorageBase {
  /// Private constructor to share code between the other constructors.
  /// Beware that the object is not necessarily guaranteed to be in a
  /// valid state after this constructor alone; e.g., `isCompressedDim(d)`
  /// doesn't entail `!(pointers[d].empty())`.
  ///
  /// Precondition: `perm` and `sparsity` must be valid for `dimSizes.size()`.
  SparseTensorStorage(const std::vector<uint64_t> &dimSizes,
                      const uint64_t *perm, const DimLevelType *sparsity)
      : SparseTensorStorageBase(dimSizes, perm, sparsity), pointers(getRank()),
        indices(getRank()), idx(getRank()) {}

public:
  /// Constructs a sparse tensor storage scheme with the given dimensions,
  /// permutation, and per-dimension dense/sparse annotations, using
  /// the coordinate scheme tensor for the initial contents if provided.
  ///
  /// Precondition: `perm` and `sparsity` must be valid for `dimSizes.size()`.
  SparseTensorStorage(const std::vector<uint64_t> &dimSizes,
                      const uint64_t *perm, const DimLevelType *sparsity,
                      SparseTensorCOO<V> *coo);

  /// Constructs a sparse tensor storage scheme with the given dimensions,
  /// permutation, and per-dimension dense/sparse annotations, using
  /// the given sparse tensor for the initial contents.
  ///
  /// Preconditions:
  /// * `perm` and `sparsity` must be valid for `dimSizes.size()`.
  /// * The `tensor` must have the same value type `V`.
  SparseTensorStorage(const std::vector<uint64_t> &dimSizes,
                      const uint64_t *perm, const DimLevelType *sparsity,
                      const SparseTensorStorageBase &tensor);

  /// Factory method. Constructs a sparse tensor storage scheme with the given
  /// dimensions, permutation, and per-dimension dense/sparse annotations,
  /// using the coordinate scheme tensor for the initial contents if provided.
  /// In the latter case, the coordinate scheme must respect the same
  /// permutation as is desired for the new sparse tensor storage.
  ///
  /// Precondition: `shape`, `perm`, and `sparsity` must be valid for `rank`.
  static SparseTensorStorage<P, I, V> *
  newSparseTensor(uint64_t rank, const uint64_t *shape, const uint64_t *perm,
                  const DimLevelType *sparsity, SparseTensorCOO<V> *coo);

  /// Factory method. Constructs a sparse tensor storage scheme with
  /// the given dimensions, permutation, and per-dimension dense/sparse
  /// annotations, using the sparse tensor for the initial contents.
  ///
  /// Preconditions:
  /// * `shape`, `perm`, and `sparsity` must be valid for `rank`.
  /// * The `tensor` must have the same value type `V`.
  static SparseTensorStorage<P, I, V> *
  newSparseTensor(uint64_t rank, const uint64_t *shape, const uint64_t *perm,
                  const DimLevelType *sparsity,
                  const SparseTensorStorageBase *source);

  ~SparseTensorStorage() final = default;

  /// Partially specialize these getter methods based on template types.
  void getPointers(std::vector<P> **out, uint64_t d) final {
    ASSERT_VALID_DIM(d);
    *out = &pointers[d];
  }
  void getIndices(std::vector<I> **out, uint64_t d) final {
    ASSERT_VALID_DIM(d);
    *out = &indices[d];
  }
  void getValues(std::vector<V> **out) final { *out = &values; }

  /// Partially specialize lexicographical insertions based on template types.
  void lexInsert(const uint64_t *cursor, V val) final {
    // First, wrap up pending insertion path.
    uint64_t diff = 0;
    uint64_t top = 0;
    if (!values.empty()) {
      diff = lexDiff(cursor);
      endPath(diff + 1);
      top = idx[diff] + 1;
    }
    // Then continue with insertion path.
    insPath(cursor, diff, top, val);
  }

  /// Partially specialize expanded insertions based on template types.
  /// Note that this method resets the values/filled-switch array back
  /// to all-zero/false while only iterating over the nonzero elements.
  void expInsert(uint64_t *cursor, V *values, bool *filled, uint64_t *added,
                 uint64_t count) final {
    if (count == 0)
      return;
    // Sort.
    std::sort(added, added + count);
    // Restore insertion path for first insert.
    const uint64_t lastDim = getRank() - 1;
    uint64_t index = added[0];
    assert(filled[index] && "added index is not filled");
    cursor[lastDim] = index;
    lexInsert(cursor, values[index]);
    values[index] = 0;
    filled[index] = false;
    // Subsequent insertions are quick.
    for (uint64_t i = 1; i < count; ++i) {
      assert(index < added[i] && "non-lexicographic insertion");
      index = added[i];
      assert(filled[index] && "added index is not filled");
      cursor[lastDim] = index;
      insPath(cursor, lastDim, added[i - 1] + 1, values[index]);
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

  void newEnumerator(SparseTensorEnumeratorBase<V> **out, uint64_t rank,
                     const uint64_t *perm) const final {
    *out = new SparseTensorEnumerator<P, I, V>(*this, rank, perm);
  }

  /// Returns this sparse tensor storage scheme as a new memory-resident
  /// sparse tensor in coordinate scheme with the given dimension order.
  ///
  /// Precondition: `perm` must be valid for `getRank()`.
  SparseTensorCOO<V> *toCOO(const uint64_t *perm) const {
    SparseTensorEnumeratorBase<V> *enumerator;
    newEnumerator(&enumerator, getRank(), perm);
    SparseTensorCOO<V> *coo =
        new SparseTensorCOO<V>(enumerator->permutedSizes(), values.size());
    enumerator->forallElements([&coo](const std::vector<uint64_t> &ind, V val) {
      coo->add(ind, val);
    });
    // TODO: This assertion assumes there are no stored zeros,
    // or if there are then that we don't filter them out.
    // Cf., <https://github.com/llvm/llvm-project/issues/54179>
    assert(coo->getElements().size() == values.size());
    delete enumerator;
    return coo;
  }

private:
  /// Appends an arbitrary new position to `pointers[d]`.  This method
  /// checks that `pos` is representable in the `P` type; however, it
  /// does not check that `pos` is semantically valid (i.e., larger than
  /// the previous position and smaller than `indices[d].capacity()`).
  void appendPointer(uint64_t d, uint64_t pos, uint64_t count = 1) {
    ASSERT_COMPRESSED_DIM(d);
    assert(pos <= std::numeric_limits<P>::max() &&
           "Pointer value is too large for the P-type");
    pointers[d].insert(pointers[d].end(), count, static_cast<P>(pos));
  }

  /// Appends index `i` to dimension `d`, in the semantically general
  /// sense.  For non-dense dimensions, that means appending to the
  /// `indices[d]` array, checking that `i` is representable in the `I`
  /// type; however, we do not verify other semantic requirements (e.g.,
  /// that `i` is in bounds for `dimSizes[d]`, and not previously occurring
  /// in the same segment).  For dense dimensions, this method instead
  /// appends the appropriate number of zeros to the `values` array,
  /// where `full` is the number of "entries" already written to `values`
  /// for this segment (aka one after the highest index previously appended).
  void appendIndex(uint64_t d, uint64_t full, uint64_t i) {
    if (isCompressedDim(d) || isSingletonDim(d)) {
      assert(i <= std::numeric_limits<I>::max() &&
             "Index value is too large for the I-type");
      indices[d].push_back(static_cast<I>(i));
    } else { // Dense dimension.
      ASSERT_DENSE_DIM(d);
      assert(i >= full && "Index was already filled");
      if (i == full)
        return; // Short-circuit, since it'll be a nop.
      if (d + 1 == getRank())
        values.insert(values.end(), i - full, 0);
      else
        finalizeSegment(d + 1, 0, i - full);
    }
  }

  /// Writes the given coordinate to `indices[d][pos]`.  This method
  /// checks that `i` is representable in the `I` type; however, it
  /// does not check that `i` is semantically valid (i.e., in bounds
  /// for `dimSizes[d]` and not elsewhere occurring in the same segment).
  void writeIndex(uint64_t d, uint64_t pos, uint64_t i) {
    ASSERT_COMPRESSED_DIM(d);
    // Subscript assignment to `std::vector` requires that the `pos`-th
    // entry has been initialized; thus we must be sure to check `size()`
    // here, instead of `capacity()` as would be ideal.
    assert(pos < indices[d].size() && "Index position is out of bounds");
    assert(i <= std::numeric_limits<I>::max() &&
           "Index value is too large for the I-type");
    indices[d][pos] = static_cast<I>(i);
  }

  /// Computes the assembled-size associated with the `d`-th dimension,
  /// given the assembled-size associated with the `(d-1)`-th dimension.
  /// "Assembled-sizes" correspond to the (nominal) sizes of overhead
  /// storage, as opposed to "dimension-sizes" which are the cardinality
  /// of coordinates for that dimension.
  ///
  /// Precondition: the `pointers[d]` array must be fully initialized
  /// before calling this method.
  uint64_t assembledSize(uint64_t parentSz, uint64_t d) const {
    if (isCompressedDim(d))
      return pointers[d][parentSz];
    // else if dense:
    return parentSz * getDimSizes()[d];
  }

  /// Initializes sparse tensor storage scheme from a memory-resident sparse
  /// tensor in coordinate scheme. This method prepares the pointers and
  /// indices arrays under the given per-dimension dense/sparse annotations.
  ///
  /// Preconditions:
  /// (1) the `elements` must be lexicographically sorted.
  /// (2) the indices of every element are valid for `dimSizes` (equal rank
  ///     and pointwise less-than).
  void fromCOO(const std::vector<Element<V>> &elements, uint64_t lo,
               uint64_t hi, uint64_t d) {
    const uint64_t rank = getRank();
    assert(d <= rank && hi <= elements.size());
    // Once dimensions are exhausted, insert the numerical values.
    if (d == rank) {
      assert(lo < hi);
      values.push_back(elements[lo].value);
      return;
    }
    // Visit all elements in this interval.
    uint64_t full = 0;
    while (lo < hi) { // If `hi` is unchanged, then `lo < elements.size()`.
      // Find segment in interval with same index elements in this dimension.
      const uint64_t i = elements[lo].indices[d];
      uint64_t seg = lo + 1;
      if (isUniqueDim(d))
        while (seg < hi && elements[seg].indices[d] == i)
          ++seg;
      // Handle segment in interval for sparse or dense dimension.
      appendIndex(d, full, i);
      full = i + 1;
      fromCOO(elements, lo, seg, d + 1);
      // And move on to next segment in interval.
      lo = seg;
    }
    // Finalize the sparse pointer structure at this dimension.
    finalizeSegment(d, full);
  }

  /// Finalize the sparse pointer structure at this dimension.
  void finalizeSegment(uint64_t d, uint64_t full = 0, uint64_t count = 1) {
    if (count == 0)
      return; // Short-circuit, since it'll be a nop.
    if (isCompressedDim(d)) {
      appendPointer(d, indices[d].size(), count);
    } else if (isSingletonDim(d)) {
      return;
    } else { // Dense dimension.
      ASSERT_DENSE_DIM(d);
      const uint64_t sz = getDimSizes()[d];
      assert(sz >= full && "Segment is overfull");
      count = detail::checkedMul(count, sz - full);
      // For dense storage we must enumerate all the remaining coordinates
      // in this dimension (i.e., coordinates after the last non-zero
      // element), and either fill in their zero values or else recurse
      // to finalize some deeper dimension.
      if (d + 1 == getRank())
        values.insert(values.end(), count, 0);
      else
        finalizeSegment(d + 1, 0, count);
    }
  }

  /// Wraps up a single insertion path, inner to outer.
  void endPath(uint64_t diff) {
    const uint64_t rank = getRank();
    assert(diff <= rank && "Dimension-diff is out of bounds");
    for (uint64_t i = 0; i < rank - diff; ++i) {
      const uint64_t d = rank - i - 1;
      finalizeSegment(d, idx[d] + 1);
    }
  }

  /// Continues a single insertion path, outer to inner.
  void insPath(const uint64_t *cursor, uint64_t diff, uint64_t top, V val) {
    ASSERT_VALID_DIM(diff);
    const uint64_t rank = getRank();
    for (uint64_t d = diff; d < rank; ++d) {
      const uint64_t i = cursor[d];
      appendIndex(d, top, i);
      top = 0;
      idx[d] = i;
    }
    values.push_back(val);
  }

  /// Finds the lexicographic differing dimension.
  uint64_t lexDiff(const uint64_t *cursor) const {
    const uint64_t rank = getRank();
    for (uint64_t r = 0; r < rank; ++r)
      if (cursor[r] > idx[r])
        return r;
      else
        assert(cursor[r] == idx[r] && "non-lexicographic insertion");
    assert(0 && "duplication insertion");
    return -1u;
  }

  // Allow `SparseTensorEnumerator` to access the data-members (to avoid
  // the cost of virtual-function dispatch in inner loops), without
  // making them public to other client code.
  friend class SparseTensorEnumerator<P, I, V>;

  std::vector<std::vector<P>> pointers;
  std::vector<std::vector<I>> indices;
  std::vector<V> values;
  std::vector<uint64_t> idx; // index cursor for lexicographic insertion.
};

#undef ASSERT_COMPRESSED_DIM
#undef ASSERT_VALID_DIM

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
class SparseTensorEnumeratorBase {
public:
  /// Constructs an enumerator with the given permutation for mapping
  /// the semantic-ordering of dimensions to the desired target-ordering.
  ///
  /// Preconditions:
  /// * the `tensor` must have the same `V` value type.
  /// * `perm` must be valid for `rank`.
  SparseTensorEnumeratorBase(const SparseTensorStorageBase &tensor,
                             uint64_t rank, const uint64_t *perm)
      : src(tensor), permsz(src.getRev().size()), reord(getRank()),
        cursor(getRank()) {
    assert(perm && "Received nullptr for permutation");
    assert(rank == getRank() && "Permutation rank mismatch");
    const auto &rev = src.getRev();           // source-order -> semantic-order
    const auto &dimSizes = src.getDimSizes(); // in source storage-order
    for (uint64_t s = 0; s < rank; ++s) {     // `s` source storage-order
      uint64_t t = perm[rev[s]];              // `t` target-order
      reord[s] = t;
      permsz[t] = dimSizes[s];
    }
  }

  virtual ~SparseTensorEnumeratorBase() = default;

  // We disallow copying to help avoid leaking the `src` reference.
  // (In addition to avoiding the problem of slicing.)
  SparseTensorEnumeratorBase(const SparseTensorEnumeratorBase &) = delete;
  SparseTensorEnumeratorBase &
  operator=(const SparseTensorEnumeratorBase &) = delete;

  /// Returns the source/target tensor's rank.  (The source-rank and
  /// target-rank are always equal since we only support permutations.
  /// Though once we add support for other dimension mappings, this
  /// method will have to be split in two.)
  uint64_t getRank() const { return permsz.size(); }

  /// Returns the target tensor's dimension sizes.
  const std::vector<uint64_t> &permutedSizes() const { return permsz; }

  /// Enumerates all elements of the source tensor, permutes their
  /// indices, and passes the permuted element to the callback.
  /// The callback must not store the cursor reference directly,
  /// since this function reuses the storage.  Instead, the callback
  /// must copy it if they want to keep it.
  virtual void forallElements(ElementConsumer<V> yield) = 0;

protected:
  const SparseTensorStorageBase &src;
  std::vector<uint64_t> permsz; // in target order.
  std::vector<uint64_t> reord;  // source storage-order -> target order.
  std::vector<uint64_t> cursor; // in target order.
};

//===----------------------------------------------------------------------===//
template <typename P, typename I, typename V>
class SparseTensorEnumerator final : public SparseTensorEnumeratorBase<V> {
  using Base = SparseTensorEnumeratorBase<V>;
  using StorageImpl = SparseTensorStorage<P, I, V>;

public:
  /// Constructs an enumerator with the given permutation for mapping
  /// the semantic-ordering of dimensions to the desired target-ordering.
  ///
  /// Precondition: `perm` must be valid for `rank`.
  SparseTensorEnumerator(const StorageImpl &tensor, uint64_t rank,
                         const uint64_t *perm)
      : Base(tensor, rank, perm) {}

  ~SparseTensorEnumerator() final = default;

  void forallElements(ElementConsumer<V> yield) final {
    forallElements(yield, 0, 0);
  }

private:
  /// The recursive component of the public `forallElements`.
  void forallElements(ElementConsumer<V> yield, uint64_t parentPos,
                      uint64_t d) {
    // Recover the `<P,I,V>` type parameters of `src`.
    const auto &src = static_cast<const StorageImpl &>(this->src);
    if (d == Base::getRank()) {
      assert(parentPos < src.values.size() &&
             "Value position is out of bounds");
      // TODO: <https://github.com/llvm/llvm-project/issues/54179>
      yield(this->cursor, src.values[parentPos]);
    } else if (src.isCompressedDim(d)) {
      // Look up the bounds of the `d`-level segment determined by the
      // `d-1`-level position `parentPos`.
      const std::vector<P> &pointersD = src.pointers[d];
      assert(parentPos + 1 < pointersD.size() &&
             "Parent pointer position is out of bounds");
      const uint64_t pstart = static_cast<uint64_t>(pointersD[parentPos]);
      const uint64_t pstop = static_cast<uint64_t>(pointersD[parentPos + 1]);
      // Loop-invariant code for looking up the `d`-level coordinates/indices.
      const std::vector<I> &indicesD = src.indices[d];
      assert(pstop <= indicesD.size() && "Index position is out of bounds");
      uint64_t &cursorReordD = this->cursor[this->reord[d]];
      for (uint64_t pos = pstart; pos < pstop; ++pos) {
        cursorReordD = static_cast<uint64_t>(indicesD[pos]);
        forallElements(yield, pos, d + 1);
      }
    } else if (src.isSingletonDim(d)) {
      MLIR_SPARSETENSOR_FATAL("unsupported dimension level type");
    } else { // Dense dimension.
      assert(src.isDenseDim(d)); // TODO: reuse the ASSERT_DENSE_DIM message
      const uint64_t sz = src.getDimSizes()[d];
      const uint64_t pstart = parentPos * sz;
      uint64_t &cursorReordD = this->cursor[this->reord[d]];
      for (uint64_t i = 0; i < sz; ++i) {
        cursorReordD = i;
        forallElements(yield, pstart + i, d + 1);
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
class SparseTensorNNZ final {
public:
  /// Allocate the statistics structure for the desired sizes and
  /// sparsity (in the target tensor's storage-order).  This constructor
  /// does not actually populate the statistics, however; for that see
  /// `initialize`.
  ///
  /// Precondition: `dimSizes` must not contain zeros.
  SparseTensorNNZ(const std::vector<uint64_t> &dimSizes,
                  const std::vector<DimLevelType> &sparsity);

  // We disallow copying to help avoid leaking the stored references.
  SparseTensorNNZ(const SparseTensorNNZ &) = delete;
  SparseTensorNNZ &operator=(const SparseTensorNNZ &) = delete;

  /// Returns the rank of the target tensor.
  uint64_t getRank() const { return dimSizes.size(); }

  /// Enumerate the source tensor to fill in the statistics.  The
  /// enumerator should already incorporate the permutation (from
  /// semantic-order to the target storage-order).
  template <typename V>
  void initialize(SparseTensorEnumeratorBase<V> &enumerator) {
    assert(enumerator.getRank() == getRank() && "Tensor rank mismatch");
    assert(enumerator.permutedSizes() == dimSizes && "Tensor size mismatch");
    enumerator.forallElements(
        [this](const std::vector<uint64_t> &ind, V) { add(ind); });
  }

  /// The type of callback functions which receive an nnz-statistic.
  using NNZConsumer = const std::function<void(uint64_t)> &;

  /// Lexicographically enumerates all indicies for dimensions strictly
  /// less than `stopDim`, and passes their nnz statistic to the callback.
  /// Since our use-case only requires the statistic not the coordinates
  /// themselves, we do not bother to construct those coordinates.
  void forallIndices(uint64_t stopDim, NNZConsumer yield) const;

private:
  /// Adds a new element (i.e., increment its statistics).  We use
  /// a method rather than inlining into the lambda in `initialize`,
  /// to avoid spurious templating over `V`.  And this method is private
  /// to avoid needing to re-assert validity of `ind` (which is guaranteed
  /// by `forallElements`).
  void add(const std::vector<uint64_t> &ind);

  /// Recursive component of the public `forallIndices`.
  void forallIndices(NNZConsumer yield, uint64_t stopDim, uint64_t parentPos,
                     uint64_t d) const;

  // All of these are in the target storage-order.
  const std::vector<uint64_t> &dimSizes;
  const std::vector<DimLevelType> &dimTypes;
  std::vector<std::vector<uint64_t>> nnz;
};

//===----------------------------------------------------------------------===//
// Definitions of the ctors and factories of `SparseTensorStorage<P,I,V>`.

namespace detail {
/// Asserts that the `dimSizes` (in target-order) under the `perm` (mapping
/// semantic-order to target-order) are a refinement of the desired `shape`
/// (in semantic-order).
///
/// Precondition: `perm` and `shape` must be valid for `rank`.
void assertPermutedSizesMatchShape(const std::vector<uint64_t> &dimSizes,
                                   uint64_t rank, const uint64_t *perm,
                                   const uint64_t *shape);
} // namespace detail

template <typename P, typename I, typename V>
SparseTensorStorage<P, I, V> *SparseTensorStorage<P, I, V>::newSparseTensor(
    uint64_t rank, const uint64_t *shape, const uint64_t *perm,
    const DimLevelType *sparsity, SparseTensorCOO<V> *coo) {
  if (coo) {
    const auto &coosz = coo->getDimSizes();
#ifndef NDEBUG
    detail::assertPermutedSizesMatchShape(coosz, rank, perm, shape);
#endif
    return new SparseTensorStorage<P, I, V>(coosz, perm, sparsity, coo);
  }
  // else
  std::vector<uint64_t> permsz(rank);
  for (uint64_t r = 0; r < rank; ++r) {
    assert(shape[r] > 0 && "Dimension size zero has trivial storage");
    permsz[perm[r]] = shape[r];
  }
  // We pass the null `coo` to ensure we select the intended constructor.
  return new SparseTensorStorage<P, I, V>(permsz, perm, sparsity, coo);
}

template <typename P, typename I, typename V>
SparseTensorStorage<P, I, V> *SparseTensorStorage<P, I, V>::newSparseTensor(
    uint64_t rank, const uint64_t *shape, const uint64_t *perm,
    const DimLevelType *sparsity, const SparseTensorStorageBase *source) {
  assert(source && "Got nullptr for source");
  SparseTensorEnumeratorBase<V> *enumerator;
  source->newEnumerator(&enumerator, rank, perm);
  const auto &permsz = enumerator->permutedSizes();
#ifndef NDEBUG
  detail::assertPermutedSizesMatchShape(permsz, rank, perm, shape);
#endif
  auto *tensor =
      new SparseTensorStorage<P, I, V>(permsz, perm, sparsity, *source);
  delete enumerator;
  return tensor;
}

template <typename P, typename I, typename V>
SparseTensorStorage<P, I, V>::SparseTensorStorage(
    const std::vector<uint64_t> &dimSizes, const uint64_t *perm,
    const DimLevelType *sparsity, SparseTensorCOO<V> *coo)
    : SparseTensorStorage(dimSizes, perm, sparsity) {
  // Provide hints on capacity of pointers and indices.
  // TODO: needs much fine-tuning based on actual sparsity; currently
  //       we reserve pointer/index space based on all previous dense
  //       dimensions, which works well up to first sparse dim; but
  //       we should really use nnz and dense/sparse distribution.
  bool allDense = true;
  uint64_t sz = 1;
  for (uint64_t r = 0, rank = getRank(); r < rank; ++r) {
    if (isCompressedDim(r)) {
      // TODO: Take a parameter between 1 and `dimSizes[r]`, and multiply
      // `sz` by that before reserving. (For now we just use 1.)
      pointers[r].reserve(sz + 1);
      pointers[r].push_back(0);
      indices[r].reserve(sz);
      sz = 1;
      allDense = false;
    } else if (isSingletonDim(r)) {
      indices[r].reserve(sz);
      sz = 1;
      allDense = false;
    } else { // Dense dimension.
      ASSERT_DENSE_DIM(r);
      sz = detail::checkedMul(sz, getDimSizes()[r]);
    }
  }
  // Then assign contents from coordinate scheme tensor if provided.
  if (coo) {
    // Ensure both preconditions of `fromCOO`.
    assert(coo->getDimSizes() == getDimSizes() && "Tensor size mismatch");
    coo->sort();
    // Now actually insert the `elements`.
    const std::vector<Element<V>> &elements = coo->getElements();
    uint64_t nnz = elements.size();
    values.reserve(nnz);
    fromCOO(elements, 0, nnz, 0);
  } else if (allDense) {
    values.resize(sz, 0);
  }
}

template <typename P, typename I, typename V>
SparseTensorStorage<P, I, V>::SparseTensorStorage(
    const std::vector<uint64_t> &dimSizes, const uint64_t *perm,
    const DimLevelType *sparsity, const SparseTensorStorageBase &tensor)
    : SparseTensorStorage(dimSizes, perm, sparsity) {
  SparseTensorEnumeratorBase<V> *enumerator;
  tensor.newEnumerator(&enumerator, getRank(), perm);
  {
    // Initialize the statistics structure.
    SparseTensorNNZ nnz(getDimSizes(), getDimTypes());
    nnz.initialize(*enumerator);
    // Initialize "pointers" overhead (and allocate "indices", "values").
    uint64_t parentSz = 1; // assembled-size (not dimension-size) of `r-1`.
    for (uint64_t rank = getRank(), r = 0; r < rank; ++r) {
      if (isCompressedDim(r)) {
        pointers[r].reserve(parentSz + 1);
        pointers[r].push_back(0);
        uint64_t currentPos = 0;
        nnz.forallIndices(r, [this, &currentPos, r](uint64_t n) {
          currentPos += n;
          appendPointer(r, currentPos);
        });
        assert(pointers[r].size() == parentSz + 1 &&
               "Final pointers size doesn't match allocated size");
        // That assertion entails `assembledSize(parentSz, r)`
        // is now in a valid state.  That is, `pointers[r][parentSz]`
        // equals the present value of `currentPos`, which is the
        // correct assembled-size for `indices[r]`.
      }
      // Update assembled-size for the next iteration.
      parentSz = assembledSize(parentSz, r);
      // Ideally we need only `indices[r].reserve(parentSz)`, however
      // the `std::vector` implementation forces us to initialize it too.
      // That is, in the yieldPos loop we need random-access assignment
      // to `indices[r]`; however, `std::vector`'s subscript-assignment
      // only allows assigning to already-initialized positions.
      if (isCompressedDim(r))
        indices[r].resize(parentSz, 0);
    }
    values.resize(parentSz, 0); // Both allocate and zero-initialize.
  }
  // The yieldPos loop
  enumerator->forallElements([this](const std::vector<uint64_t> &ind, V val) {
    uint64_t parentSz = 1, parentPos = 0;
    for (uint64_t rank = getRank(), r = 0; r < rank; ++r) {
      if (isCompressedDim(r)) {
        // If `parentPos == parentSz` then it's valid as an array-lookup;
        // however, it's semantically invalid here since that entry
        // does not represent a segment of `indices[r]`.  Moreover, that
        // entry must be immutable for `assembledSize` to remain valid.
        assert(parentPos < parentSz && "Pointers position is out of bounds");
        const uint64_t currentPos = pointers[r][parentPos];
        // This increment won't overflow the `P` type, since it can't
        // exceed the original value of `pointers[r][parentPos+1]`
        // which was already verified to be within bounds for `P`
        // when it was written to the array.
        pointers[r][parentPos]++;
        writeIndex(r, currentPos, ind[r]);
        parentPos = currentPos;
      } else if (isSingletonDim(r)) {
        // the new parentPos equals the old parentPos.
      } else { // Dense dimension.
        ASSERT_DENSE_DIM(r);
        parentPos = parentPos * getDimSizes()[r] + ind[r];
      }
      parentSz = assembledSize(parentSz, r);
    }
    assert(parentPos < values.size() && "Value position is out of bounds");
    values[parentPos] = val;
  });
  // No longer need the enumerator, so we'll delete it ASAP.
  delete enumerator;
  // The finalizeYieldPos loop
  for (uint64_t parentSz = 1, rank = getRank(), r = 0; r < rank; ++r) {
    if (isCompressedDim(r)) {
      assert(parentSz == pointers[r].size() - 1 &&
             "Actual pointers size doesn't match the expected size");
      // Can't check all of them, but at least we can check the last one.
      assert(pointers[r][parentSz - 1] == pointers[r][parentSz] &&
             "Pointers got corrupted");
      // TODO: optimize this by using `memmove` or similar.
      for (uint64_t n = 0; n < parentSz; ++n) {
        const uint64_t parentPos = parentSz - n;
        pointers[r][parentPos] = pointers[r][parentPos - 1];
      }
      pointers[r][0] = 0;
    }
    parentSz = assembledSize(parentSz, r);
  }
}

} // namespace sparse_tensor
} // namespace mlir

#undef ASSERT_DENSE_DIM

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_STORAGE_H
