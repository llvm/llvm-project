//===- SparseTensorUtils.cpp - Sparse Tensor Utils for MLIR execution -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a light-weight runtime support library that is useful
// for sparse tensor manipulations. The functionality provided in this library
// is meant to simplify benchmarking, testing, and debugging MLIR code that
// operates on sparse tensors. The provided functionality is **not** part
// of core MLIR, however.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/SparseTensorUtils.h"

#ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>

//===----------------------------------------------------------------------===//
//
// Internal support for storing and reading sparse tensors.
//
// The following memory-resident sparse storage schemes are supported:
//
// (a) A coordinate scheme for temporarily storing and lexicographically
//     sorting a sparse tensor by index (SparseTensorCOO).
//
// (b) A "one-size-fits-all" sparse tensor storage scheme defined by
//     per-dimension sparse/dense annnotations together with a dimension
//     ordering used by MLIR compiler-generated code (SparseTensorStorage).
//
// The following external formats are supported:
//
// (1) Matrix Market Exchange (MME): *.mtx
//     https://math.nist.gov/MatrixMarket/formats.html
//
// (2) Formidable Repository of Open Sparse Tensors and Tools (FROSTT): *.tns
//     http://frostt.io/tensors/file-formats.html
//
// Two public APIs are supported:
//
// (I) Methods operating on MLIR buffers (memrefs) to interact with sparse
//     tensors. These methods should be used exclusively by MLIR
//     compiler-generated code.
//
// (II) Methods that accept C-style data structures to interact with sparse
//      tensors. These methods can be used by any external runtime that wants
//      to interact with MLIR compiler-generated code.
//
// In both cases (I) and (II), the SparseTensorStorage format is externally
// only visible as an opaque pointer.
//
//===----------------------------------------------------------------------===//

namespace {

static constexpr int kColWidth = 1025;

/// A version of `operator*` on `uint64_t` which checks for overflows.
static inline uint64_t checkedMul(uint64_t lhs, uint64_t rhs) {
  assert((lhs == 0 || rhs <= std::numeric_limits<uint64_t>::max() / lhs) &&
         "Integer overflow");
  return lhs * rhs;
}

// This macro helps minimize repetition of this idiom, as well as ensuring
// we have some additional output indicating where the error is coming from.
// (Since `fprintf` doesn't provide a stacktrace, this helps make it easier
// to track down whether an error is coming from our code vs somewhere else
// in MLIR.)
#define FATAL(...)                                                             \
  {                                                                            \
    fprintf(stderr, "SparseTensorUtils: " __VA_ARGS__);                        \
    exit(1);                                                                   \
  }

// TODO: adjust this so it can be used by `openSparseTensorCOO` too.
// That version doesn't have the permutation, and the `dimSizes` are
// a pointer/C-array rather than `std::vector`.
//
/// Asserts that the `dimSizes` (in target-order) under the `perm` (mapping
/// semantic-order to target-order) are a refinement of the desired `shape`
/// (in semantic-order).
///
/// Precondition: `perm` and `shape` must be valid for `rank`.
static inline void
assertPermutedSizesMatchShape(const std::vector<uint64_t> &dimSizes,
                              uint64_t rank, const uint64_t *perm,
                              const uint64_t *shape) {
  assert(perm && shape);
  assert(rank == dimSizes.size() && "Rank mismatch");
  for (uint64_t r = 0; r < rank; r++)
    assert((shape[r] == 0 || shape[r] == dimSizes[perm[r]]) &&
           "Dimension size mismatch");
}

/// A sparse tensor element in coordinate scheme (value and indices).
/// For example, a rank-1 vector element would look like
///   ({i}, a[i])
/// and a rank-5 tensor element like
///   ({i,j,k,l,m}, a[i,j,k,l,m])
/// We use pointer to a shared index pool rather than e.g. a direct
/// vector since that (1) reduces the per-element memory footprint, and
/// (2) centralizes the memory reservation and (re)allocation to one place.
template <typename V>
struct Element final {
  Element(uint64_t *ind, V val) : indices(ind), value(val){};
  uint64_t *indices; // pointer into shared index pool
  V value;
};

/// The type of callback functions which receive an element.  We avoid
/// packaging the coordinates and value together as an `Element` object
/// because this helps keep code somewhat cleaner.
template <typename V>
using ElementConsumer =
    const std::function<void(const std::vector<uint64_t> &, V)> &;

/// A memory-resident sparse tensor in coordinate scheme (collection of
/// elements). This data structure is used to read a sparse tensor from
/// any external format into memory and sort the elements lexicographically
/// by indices before passing it back to the client (most packed storage
/// formats require the elements to appear in lexicographic index order).
template <typename V>
struct SparseTensorCOO final {
public:
  SparseTensorCOO(const std::vector<uint64_t> &dimSizes, uint64_t capacity)
      : dimSizes(dimSizes) {
    if (capacity) {
      elements.reserve(capacity);
      indices.reserve(capacity * getRank());
    }
  }

  /// Adds element as indices and value.
  void add(const std::vector<uint64_t> &ind, V val) {
    assert(!iteratorLocked && "Attempt to add() after startIterator()");
    uint64_t *base = indices.data();
    uint64_t size = indices.size();
    uint64_t rank = getRank();
    assert(ind.size() == rank && "Element rank mismatch");
    for (uint64_t r = 0; r < rank; r++) {
      assert(ind[r] < dimSizes[r] && "Index is too large for the dimension");
      indices.push_back(ind[r]);
    }
    // This base only changes if indices were reallocated. In that case, we
    // need to correct all previous pointers into the vector. Note that this
    // only happens if we did not set the initial capacity right, and then only
    // for every internal vector reallocation (which with the doubling rule
    // should only incur an amortized linear overhead).
    uint64_t *newBase = indices.data();
    if (newBase != base) {
      for (uint64_t i = 0, n = elements.size(); i < n; i++)
        elements[i].indices = newBase + (elements[i].indices - base);
      base = newBase;
    }
    // Add element as (pointer into shared index pool, value) pair.
    elements.emplace_back(base + size, val);
  }

  /// Sorts elements lexicographically by index.
  void sort() {
    assert(!iteratorLocked && "Attempt to sort() after startIterator()");
    // TODO: we may want to cache an `isSorted` bit, to avoid
    // unnecessary/redundant sorting.
    uint64_t rank = getRank();
    std::sort(elements.begin(), elements.end(),
              [rank](const Element<V> &e1, const Element<V> &e2) {
                for (uint64_t r = 0; r < rank; r++) {
                  if (e1.indices[r] == e2.indices[r])
                    continue;
                  return e1.indices[r] < e2.indices[r];
                }
                return false;
              });
  }

  /// Get the rank of the tensor.
  uint64_t getRank() const { return dimSizes.size(); }

  /// Getter for the dimension-sizes array.
  const std::vector<uint64_t> &getDimSizes() const { return dimSizes; }

  /// Getter for the elements array.
  const std::vector<Element<V>> &getElements() const { return elements; }

  /// Switch into iterator mode.
  void startIterator() {
    iteratorLocked = true;
    iteratorPos = 0;
  }

  /// Get the next element.
  const Element<V> *getNext() {
    assert(iteratorLocked && "Attempt to getNext() before startIterator()");
    if (iteratorPos < elements.size())
      return &(elements[iteratorPos++]);
    iteratorLocked = false;
    return nullptr;
  }

  /// Factory method. Permutes the original dimensions according to
  /// the given ordering and expects subsequent add() calls to honor
  /// that same ordering for the given indices. The result is a
  /// fully permuted coordinate scheme.
  ///
  /// Precondition: `dimSizes` and `perm` must be valid for `rank`.
  static SparseTensorCOO<V> *newSparseTensorCOO(uint64_t rank,
                                                const uint64_t *dimSizes,
                                                const uint64_t *perm,
                                                uint64_t capacity = 0) {
    std::vector<uint64_t> permsz(rank);
    for (uint64_t r = 0; r < rank; r++) {
      assert(dimSizes[r] > 0 && "Dimension size zero has trivial storage");
      permsz[perm[r]] = dimSizes[r];
    }
    return new SparseTensorCOO<V>(permsz, capacity);
  }

private:
  const std::vector<uint64_t> dimSizes; // per-dimension sizes
  std::vector<Element<V>> elements;     // all COO elements
  std::vector<uint64_t> indices;        // shared index pool
  bool iteratorLocked = false;
  unsigned iteratorPos = 0;
};

// Forward.
template <typename V>
class SparseTensorEnumeratorBase;

// Helper macro for generating error messages when some
// `SparseTensorStorage<P,I,V>` is cast to `SparseTensorStorageBase`
// and then the wrong "partial method specialization" is called.
#define FATAL_PIV(NAME) FATAL("<P,I,V> type mismatch for: " #NAME);

/// Abstract base class for `SparseTensorStorage<P,I,V>`.  This class
/// takes responsibility for all the `<P,I,V>`-independent aspects
/// of the tensor (e.g., shape, sparsity, permutation).  In addition,
/// we use function overloading to implement "partial" method
/// specialization, which the C-API relies on to catch type errors
/// arising from our use of opaque pointers.
class SparseTensorStorageBase {
public:
  /// Constructs a new storage object.  The `perm` maps the tensor's
  /// semantic-ordering of dimensions to this object's storage-order.
  /// The `dimSizes` and `sparsity` arrays are already in storage-order.
  ///
  /// Precondition: `perm` and `sparsity` must be valid for `dimSizes.size()`.
  SparseTensorStorageBase(const std::vector<uint64_t> &dimSizes,
                          const uint64_t *perm, const DimLevelType *sparsity)
      : dimSizes(dimSizes), rev(getRank()),
        dimTypes(sparsity, sparsity + getRank()) {
    assert(perm && sparsity);
    const uint64_t rank = getRank();
    // Validate parameters.
    assert(rank > 0 && "Trivial shape is unsupported");
    for (uint64_t r = 0; r < rank; r++) {
      assert(dimSizes[r] > 0 && "Dimension size zero has trivial storage");
      assert((dimTypes[r] == DimLevelType::kDense ||
              dimTypes[r] == DimLevelType::kCompressed) &&
             "Unsupported DimLevelType");
    }
    // Construct the "reverse" (i.e., inverse) permutation.
    for (uint64_t r = 0; r < rank; r++)
      rev[perm[r]] = r;
  }

  virtual ~SparseTensorStorageBase() = default;

  /// Get the rank of the tensor.
  uint64_t getRank() const { return dimSizes.size(); }

  /// Getter for the dimension-sizes array, in storage-order.
  const std::vector<uint64_t> &getDimSizes() const { return dimSizes; }

  /// Safely lookup the size of the given (storage-order) dimension.
  uint64_t getDimSize(uint64_t d) const {
    assert(d < getRank());
    return dimSizes[d];
  }

  /// Getter for the "reverse" permutation, which maps this object's
  /// storage-order to the tensor's semantic-order.
  const std::vector<uint64_t> &getRev() const { return rev; }

  /// Getter for the dimension-types array, in storage-order.
  const std::vector<DimLevelType> &getDimTypes() const { return dimTypes; }

  /// Safely check if the (storage-order) dimension uses compressed storage.
  bool isCompressedDim(uint64_t d) const {
    assert(d < getRank());
    return (dimTypes[d] == DimLevelType::kCompressed);
  }

  /// Allocate a new enumerator.
#define DECL_NEWENUMERATOR(VNAME, V)                                           \
  virtual void newEnumerator(SparseTensorEnumeratorBase<V> **, uint64_t,       \
                             const uint64_t *) const {                         \
    FATAL_PIV("newEnumerator" #VNAME);                                         \
  }
  FOREVERY_V(DECL_NEWENUMERATOR)
#undef DECL_NEWENUMERATOR

  /// Overhead storage.
#define DECL_GETPOINTERS(PNAME, P)                                             \
  virtual void getPointers(std::vector<P> **, uint64_t) {                      \
    FATAL_PIV("getPointers" #PNAME);                                           \
  }
  FOREVERY_FIXED_O(DECL_GETPOINTERS)
#undef DECL_GETPOINTERS
#define DECL_GETINDICES(INAME, I)                                              \
  virtual void getIndices(std::vector<I> **, uint64_t) {                       \
    FATAL_PIV("getIndices" #INAME);                                            \
  }
  FOREVERY_FIXED_O(DECL_GETINDICES)
#undef DECL_GETINDICES

  /// Primary storage.
#define DECL_GETVALUES(VNAME, V)                                               \
  virtual void getValues(std::vector<V> **) { FATAL_PIV("getValues" #VNAME); }
  FOREVERY_V(DECL_GETVALUES)
#undef DECL_GETVALUES

  /// Element-wise insertion in lexicographic index order.
#define DECL_LEXINSERT(VNAME, V)                                               \
  virtual void lexInsert(const uint64_t *, V) { FATAL_PIV("lexInsert" #VNAME); }
  FOREVERY_V(DECL_LEXINSERT)
#undef DECL_LEXINSERT

  /// Expanded insertion.
#define DECL_EXPINSERT(VNAME, V)                                               \
  virtual void expInsert(uint64_t *, V *, bool *, uint64_t *, uint64_t) {      \
    FATAL_PIV("expInsert" #VNAME);                                             \
  }
  FOREVERY_V(DECL_EXPINSERT)
#undef DECL_EXPINSERT

  /// Finishes insertion.
  virtual void endInsert() = 0;

protected:
  // Since this class is virtual, we must disallow public copying in
  // order to avoid "slicing".  Since this class has data members,
  // that means making copying protected.
  // <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-copy-virtual>
  SparseTensorStorageBase(const SparseTensorStorageBase &) = default;
  // Copy-assignment would be implicitly deleted (because `dimSizes`
  // is const), so we explicitly delete it for clarity.
  SparseTensorStorageBase &operator=(const SparseTensorStorageBase &) = delete;

private:
  const std::vector<uint64_t> dimSizes;
  std::vector<uint64_t> rev;
  const std::vector<DimLevelType> dimTypes;
};

#undef FATAL_PIV

// Forward.
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
                      SparseTensorCOO<V> *coo)
      : SparseTensorStorage(dimSizes, perm, sparsity) {
    // Provide hints on capacity of pointers and indices.
    // TODO: needs much fine-tuning based on actual sparsity; currently
    //       we reserve pointer/index space based on all previous dense
    //       dimensions, which works well up to first sparse dim; but
    //       we should really use nnz and dense/sparse distribution.
    bool allDense = true;
    uint64_t sz = 1;
    for (uint64_t r = 0, rank = getRank(); r < rank; r++) {
      if (isCompressedDim(r)) {
        // TODO: Take a parameter between 1 and `dimSizes[r]`, and multiply
        // `sz` by that before reserving. (For now we just use 1.)
        pointers[r].reserve(sz + 1);
        pointers[r].push_back(0);
        indices[r].reserve(sz);
        sz = 1;
        allDense = false;
      } else { // Dense dimension.
        sz = checkedMul(sz, getDimSizes()[r]);
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

  ~SparseTensorStorage() final = default;

  /// Partially specialize these getter methods based on template types.
  void getPointers(std::vector<P> **out, uint64_t d) final {
    assert(d < getRank());
    *out = &pointers[d];
  }
  void getIndices(std::vector<I> **out, uint64_t d) final {
    assert(d < getRank());
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
    cursor[lastDim] = index;
    lexInsert(cursor, values[index]);
    assert(filled[index]);
    values[index] = 0;
    filled[index] = false;
    // Subsequent insertions are quick.
    for (uint64_t i = 1; i < count; i++) {
      assert(index < added[i] && "non-lexicographic insertion");
      index = added[i];
      cursor[lastDim] = index;
      insPath(cursor, lastDim, added[i - 1] + 1, values[index]);
      assert(filled[index]);
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

  /// Factory method. Constructs a sparse tensor storage scheme with the given
  /// dimensions, permutation, and per-dimension dense/sparse annotations,
  /// using the coordinate scheme tensor for the initial contents if provided.
  /// In the latter case, the coordinate scheme must respect the same
  /// permutation as is desired for the new sparse tensor storage.
  ///
  /// Precondition: `shape`, `perm`, and `sparsity` must be valid for `rank`.
  static SparseTensorStorage<P, I, V> *
  newSparseTensor(uint64_t rank, const uint64_t *shape, const uint64_t *perm,
                  const DimLevelType *sparsity, SparseTensorCOO<V> *coo) {
    SparseTensorStorage<P, I, V> *n = nullptr;
    if (coo) {
      const auto &coosz = coo->getDimSizes();
      assertPermutedSizesMatchShape(coosz, rank, perm, shape);
      n = new SparseTensorStorage<P, I, V>(coosz, perm, sparsity, coo);
    } else {
      std::vector<uint64_t> permsz(rank);
      for (uint64_t r = 0; r < rank; r++) {
        assert(shape[r] > 0 && "Dimension size zero has trivial storage");
        permsz[perm[r]] = shape[r];
      }
      // We pass the null `coo` to ensure we select the intended constructor.
      n = new SparseTensorStorage<P, I, V>(permsz, perm, sparsity, coo);
    }
    return n;
  }

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
                  const SparseTensorStorageBase *source) {
    assert(source && "Got nullptr for source");
    SparseTensorEnumeratorBase<V> *enumerator;
    source->newEnumerator(&enumerator, rank, perm);
    const auto &permsz = enumerator->permutedSizes();
    assertPermutedSizesMatchShape(permsz, rank, perm, shape);
    auto *tensor =
        new SparseTensorStorage<P, I, V>(permsz, perm, sparsity, *source);
    delete enumerator;
    return tensor;
  }

private:
  /// Appends an arbitrary new position to `pointers[d]`.  This method
  /// checks that `pos` is representable in the `P` type; however, it
  /// does not check that `pos` is semantically valid (i.e., larger than
  /// the previous position and smaller than `indices[d].capacity()`).
  void appendPointer(uint64_t d, uint64_t pos, uint64_t count = 1) {
    assert(isCompressedDim(d));
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
    if (isCompressedDim(d)) {
      assert(i <= std::numeric_limits<I>::max() &&
             "Index value is too large for the I-type");
      indices[d].push_back(static_cast<I>(i));
    } else { // Dense dimension.
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
    assert(isCompressedDim(d));
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
    uint64_t rank = getRank();
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
      uint64_t i = elements[lo].indices[d];
      uint64_t seg = lo + 1;
      while (seg < hi && elements[seg].indices[d] == i)
        seg++;
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
    } else { // Dense dimension.
      const uint64_t sz = getDimSizes()[d];
      assert(sz >= full && "Segment is overfull");
      count = checkedMul(count, sz - full);
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
    uint64_t rank = getRank();
    assert(diff <= rank);
    for (uint64_t i = 0; i < rank - diff; i++) {
      const uint64_t d = rank - i - 1;
      finalizeSegment(d, idx[d] + 1);
    }
  }

  /// Continues a single insertion path, outer to inner.
  void insPath(const uint64_t *cursor, uint64_t diff, uint64_t top, V val) {
    uint64_t rank = getRank();
    assert(diff < rank);
    for (uint64_t d = diff; d < rank; d++) {
      uint64_t i = cursor[d];
      appendIndex(d, top, i);
      top = 0;
      idx[d] = i;
    }
    values.push_back(val);
  }

  /// Finds the lexicographic differing dimension.
  uint64_t lexDiff(const uint64_t *cursor) const {
    for (uint64_t r = 0, rank = getRank(); r < rank; r++)
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
    for (uint64_t s = 0; s < rank; s++) {     // `s` source storage-order
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

template <typename P, typename I, typename V>
class SparseTensorEnumerator final : public SparseTensorEnumeratorBase<V> {
  using Base = SparseTensorEnumeratorBase<V>;

public:
  /// Constructs an enumerator with the given permutation for mapping
  /// the semantic-ordering of dimensions to the desired target-ordering.
  ///
  /// Precondition: `perm` must be valid for `rank`.
  SparseTensorEnumerator(const SparseTensorStorage<P, I, V> &tensor,
                         uint64_t rank, const uint64_t *perm)
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
    const auto &src =
        static_cast<const SparseTensorStorage<P, I, V> &>(this->src);
    if (d == Base::getRank()) {
      assert(parentPos < src.values.size() &&
             "Value position is out of bounds");
      // TODO: <https://github.com/llvm/llvm-project/issues/54179>
      yield(this->cursor, src.values[parentPos]);
    } else if (src.isCompressedDim(d)) {
      // Look up the bounds of the `d`-level segment determined by the
      // `d-1`-level position `parentPos`.
      const std::vector<P> &pointers_d = src.pointers[d];
      assert(parentPos + 1 < pointers_d.size() &&
             "Parent pointer position is out of bounds");
      const uint64_t pstart = static_cast<uint64_t>(pointers_d[parentPos]);
      const uint64_t pstop = static_cast<uint64_t>(pointers_d[parentPos + 1]);
      // Loop-invariant code for looking up the `d`-level coordinates/indices.
      const std::vector<I> &indices_d = src.indices[d];
      assert(pstop <= indices_d.size() && "Index position is out of bounds");
      uint64_t &cursor_reord_d = this->cursor[this->reord[d]];
      for (uint64_t pos = pstart; pos < pstop; pos++) {
        cursor_reord_d = static_cast<uint64_t>(indices_d[pos]);
        forallElements(yield, pos, d + 1);
      }
    } else { // Dense dimension.
      const uint64_t sz = src.getDimSizes()[d];
      const uint64_t pstart = parentPos * sz;
      uint64_t &cursor_reord_d = this->cursor[this->reord[d]];
      for (uint64_t i = 0; i < sz; i++) {
        cursor_reord_d = i;
        forallElements(yield, pstart + i, d + 1);
      }
    }
  }
};

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
                  const std::vector<DimLevelType> &sparsity)
      : dimSizes(dimSizes), dimTypes(sparsity), nnz(getRank()) {
    assert(dimSizes.size() == dimTypes.size() && "Rank mismatch");
    bool uncompressed = true;
    uint64_t sz = 1; // the product of all `dimSizes` strictly less than `r`.
    for (uint64_t rank = getRank(), r = 0; r < rank; r++) {
      switch (dimTypes[r]) {
      case DimLevelType::kCompressed:
        assert(uncompressed &&
               "Multiple compressed layers not currently supported");
        uncompressed = false;
        nnz[r].resize(sz, 0); // Both allocate and zero-initialize.
        break;
      case DimLevelType::kDense:
        assert(uncompressed &&
               "Dense after compressed not currently supported");
        break;
      case DimLevelType::kSingleton:
        // Singleton after Compressed causes no problems for allocating
        // `nnz` nor for the yieldPos loop.  This remains true even
        // when adding support for multiple compressed dimensions or
        // for dense-after-compressed.
        break;
      }
      sz = checkedMul(sz, dimSizes[r]);
    }
  }

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
  void forallIndices(uint64_t stopDim, NNZConsumer yield) const {
    assert(stopDim < getRank() && "Stopping-dimension is out of bounds");
    assert(dimTypes[stopDim] == DimLevelType::kCompressed &&
           "Cannot look up non-compressed dimensions");
    forallIndices(yield, stopDim, 0, 0);
  }

private:
  /// Adds a new element (i.e., increment its statistics).  We use
  /// a method rather than inlining into the lambda in `initialize`,
  /// to avoid spurious templating over `V`.  And this method is private
  /// to avoid needing to re-assert validity of `ind` (which is guaranteed
  /// by `forallElements`).
  void add(const std::vector<uint64_t> &ind) {
    uint64_t parentPos = 0;
    for (uint64_t rank = getRank(), r = 0; r < rank; r++) {
      if (dimTypes[r] == DimLevelType::kCompressed)
        nnz[r][parentPos]++;
      parentPos = parentPos * dimSizes[r] + ind[r];
    }
  }

  /// Recursive component of the public `forallIndices`.
  void forallIndices(NNZConsumer yield, uint64_t stopDim, uint64_t parentPos,
                     uint64_t d) const {
    assert(d <= stopDim);
    if (d == stopDim) {
      assert(parentPos < nnz[d].size() && "Cursor is out of range");
      yield(nnz[d][parentPos]);
    } else {
      const uint64_t sz = dimSizes[d];
      const uint64_t pstart = parentPos * sz;
      for (uint64_t i = 0; i < sz; i++)
        forallIndices(yield, stopDim, pstart + i, d + 1);
    }
  }

  // All of these are in the target storage-order.
  const std::vector<uint64_t> &dimSizes;
  const std::vector<DimLevelType> &dimTypes;
  std::vector<std::vector<uint64_t>> nnz;
};

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
    for (uint64_t rank = getRank(), r = 0; r < rank; r++) {
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
    for (uint64_t rank = getRank(), r = 0; r < rank; r++) {
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
      } else { // Dense dimension.
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
  for (uint64_t parentSz = 1, rank = getRank(), r = 0; r < rank; r++) {
    if (isCompressedDim(r)) {
      assert(parentSz == pointers[r].size() - 1 &&
             "Actual pointers size doesn't match the expected size");
      // Can't check all of them, but at least we can check the last one.
      assert(pointers[r][parentSz - 1] == pointers[r][parentSz] &&
             "Pointers got corrupted");
      // TODO: optimize this by using `memmove` or similar.
      for (uint64_t n = 0; n < parentSz; n++) {
        const uint64_t parentPos = parentSz - n;
        pointers[r][parentPos] = pointers[r][parentPos - 1];
      }
      pointers[r][0] = 0;
    }
    parentSz = assembledSize(parentSz, r);
  }
}

/// Helper to convert string to lower case.
static char *toLower(char *token) {
  for (char *c = token; *c; c++)
    *c = tolower(*c);
  return token;
}

/// Read the MME header of a general sparse matrix of type real.
static void readMMEHeader(FILE *file, char *filename, char *line,
                          uint64_t *idata, bool *isPattern, bool *isSymmetric) {
  char header[64];
  char object[64];
  char format[64];
  char field[64];
  char symmetry[64];
  // Read header line.
  if (fscanf(file, "%63s %63s %63s %63s %63s\n", header, object, format, field,
             symmetry) != 5)
    FATAL("Corrupt header in %s\n", filename);
  // Set properties
  *isPattern = (strcmp(toLower(field), "pattern") == 0);
  *isSymmetric = (strcmp(toLower(symmetry), "symmetric") == 0);
  // Make sure this is a general sparse matrix.
  if (strcmp(toLower(header), "%%matrixmarket") ||
      strcmp(toLower(object), "matrix") ||
      strcmp(toLower(format), "coordinate") ||
      (strcmp(toLower(field), "real") && !(*isPattern)) ||
      (strcmp(toLower(symmetry), "general") && !(*isSymmetric)))
    FATAL("Cannot find a general sparse matrix in %s\n", filename);
  // Skip comments.
  while (true) {
    if (!fgets(line, kColWidth, file))
      FATAL("Cannot find data in %s\n", filename);
    if (line[0] != '%')
      break;
  }
  // Next line contains M N NNZ.
  idata[0] = 2; // rank
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "%" PRIu64 "\n", idata + 2, idata + 3,
             idata + 1) != 3)
    FATAL("Cannot find size in %s\n", filename);
}

/// Read the "extended" FROSTT header. Although not part of the documented
/// format, we assume that the file starts with optional comments followed
/// by two lines that define the rank, the number of nonzeros, and the
/// dimensions sizes (one per rank) of the sparse tensor.
static void readExtFROSTTHeader(FILE *file, char *filename, char *line,
                                uint64_t *idata) {
  // Skip comments.
  while (true) {
    if (!fgets(line, kColWidth, file))
      FATAL("Cannot find data in %s\n", filename);
    if (line[0] != '#')
      break;
  }
  // Next line contains RANK and NNZ.
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "\n", idata, idata + 1) != 2)
    FATAL("Cannot find metadata in %s\n", filename);
  // Followed by a line with the dimension sizes (one per rank).
  for (uint64_t r = 0; r < idata[0]; r++)
    if (fscanf(file, "%" PRIu64, idata + 2 + r) != 1)
      FATAL("Cannot find dimension size %s\n", filename);
  fgets(line, kColWidth, file); // end of line
}

/// Reads a sparse tensor with the given filename into a memory-resident
/// sparse tensor in coordinate scheme.
template <typename V>
static SparseTensorCOO<V> *openSparseTensorCOO(char *filename, uint64_t rank,
                                               const uint64_t *shape,
                                               const uint64_t *perm) {
  // Open the file.
  assert(filename && "Received nullptr for filename");
  FILE *file = fopen(filename, "r");
  if (!file)
    FATAL("Cannot find file %s\n", filename);
  // Perform some file format dependent set up.
  char line[kColWidth];
  uint64_t idata[512];
  bool isPattern = false;
  bool isSymmetric = false;
  if (strstr(filename, ".mtx")) {
    readMMEHeader(file, filename, line, idata, &isPattern, &isSymmetric);
  } else if (strstr(filename, ".tns")) {
    readExtFROSTTHeader(file, filename, line, idata);
  } else {
    FATAL("Unknown format %s\n", filename);
  }
  // Prepare sparse tensor object with per-dimension sizes
  // and the number of nonzeros as initial capacity.
  assert(rank == idata[0] && "rank mismatch");
  uint64_t nnz = idata[1];
  for (uint64_t r = 0; r < rank; r++)
    assert((shape[r] == 0 || shape[r] == idata[2 + r]) &&
           "dimension size mismatch");
  SparseTensorCOO<V> *tensor =
      SparseTensorCOO<V>::newSparseTensorCOO(rank, idata + 2, perm, nnz);
  // Read all nonzero elements.
  std::vector<uint64_t> indices(rank);
  for (uint64_t k = 0; k < nnz; k++) {
    if (!fgets(line, kColWidth, file))
      FATAL("Cannot find next line of data in %s\n", filename);
    char *linePtr = line;
    for (uint64_t r = 0; r < rank; r++) {
      uint64_t idx = strtoul(linePtr, &linePtr, 10);
      // Add 0-based index.
      indices[perm[r]] = idx - 1;
    }
    // The external formats always store the numerical values with the type
    // double, but we cast these values to the sparse tensor object type.
    // For a pattern tensor, we arbitrarily pick the value 1 for all entries.
    double value = isPattern ? 1.0 : strtod(linePtr, &linePtr);
    tensor->add(indices, value);
    // We currently chose to deal with symmetric matrices by fully constructing
    // them. In the future, we may want to make symmetry implicit for storage
    // reasons.
    if (isSymmetric && indices[0] != indices[1])
      tensor->add({indices[1], indices[0]}, value);
  }
  // Close the file and return tensor.
  fclose(file);
  return tensor;
}

/// Writes the sparse tensor to `dest` in extended FROSTT format.
template <typename V>
static void outSparseTensor(void *tensor, void *dest, bool sort) {
  assert(tensor && dest);
  auto coo = static_cast<SparseTensorCOO<V> *>(tensor);
  if (sort)
    coo->sort();
  char *filename = static_cast<char *>(dest);
  auto &dimSizes = coo->getDimSizes();
  auto &elements = coo->getElements();
  uint64_t rank = coo->getRank();
  uint64_t nnz = elements.size();
  std::fstream file;
  file.open(filename, std::ios_base::out | std::ios_base::trunc);
  assert(file.is_open());
  file << "; extended FROSTT format\n" << rank << " " << nnz << std::endl;
  for (uint64_t r = 0; r < rank - 1; r++)
    file << dimSizes[r] << " ";
  file << dimSizes[rank - 1] << std::endl;
  for (uint64_t i = 0; i < nnz; i++) {
    auto &idx = elements[i].indices;
    for (uint64_t r = 0; r < rank; r++)
      file << (idx[r] + 1) << " ";
    file << elements[i].value << std::endl;
  }
  file.flush();
  file.close();
  assert(file.good());
}

/// Initializes sparse tensor from an external COO-flavored format.
template <typename V>
static SparseTensorStorage<uint64_t, uint64_t, V> *
toMLIRSparseTensor(uint64_t rank, uint64_t nse, uint64_t *shape, V *values,
                   uint64_t *indices, uint64_t *perm, uint8_t *sparse) {
  const DimLevelType *sparsity = (DimLevelType *)(sparse);
#ifndef NDEBUG
  // Verify that perm is a permutation of 0..(rank-1).
  std::vector<uint64_t> order(perm, perm + rank);
  std::sort(order.begin(), order.end());
  for (uint64_t i = 0; i < rank; ++i)
    if (i != order[i])
      FATAL("Not a permutation of 0..%" PRIu64 "\n", rank);

  // Verify that the sparsity values are supported.
  for (uint64_t i = 0; i < rank; ++i)
    if (sparsity[i] != DimLevelType::kDense &&
        sparsity[i] != DimLevelType::kCompressed)
      FATAL("Unsupported sparsity value %d\n", static_cast<int>(sparsity[i]));
#endif

  // Convert external format to internal COO.
  auto *coo = SparseTensorCOO<V>::newSparseTensorCOO(rank, shape, perm, nse);
  std::vector<uint64_t> idx(rank);
  for (uint64_t i = 0, base = 0; i < nse; i++) {
    for (uint64_t r = 0; r < rank; r++)
      idx[perm[r]] = indices[base + r];
    coo->add(idx, values[i]);
    base += rank;
  }
  // Return sparse tensor storage format as opaque pointer.
  auto *tensor = SparseTensorStorage<uint64_t, uint64_t, V>::newSparseTensor(
      rank, shape, perm, sparsity, coo);
  delete coo;
  return tensor;
}

/// Converts a sparse tensor to an external COO-flavored format.
template <typename V>
static void fromMLIRSparseTensor(void *tensor, uint64_t *pRank, uint64_t *pNse,
                                 uint64_t **pShape, V **pValues,
                                 uint64_t **pIndices) {
  assert(tensor);
  auto sparseTensor =
      static_cast<SparseTensorStorage<uint64_t, uint64_t, V> *>(tensor);
  uint64_t rank = sparseTensor->getRank();
  std::vector<uint64_t> perm(rank);
  std::iota(perm.begin(), perm.end(), 0);
  SparseTensorCOO<V> *coo = sparseTensor->toCOO(perm.data());

  const std::vector<Element<V>> &elements = coo->getElements();
  uint64_t nse = elements.size();

  uint64_t *shape = new uint64_t[rank];
  for (uint64_t i = 0; i < rank; i++)
    shape[i] = coo->getDimSizes()[i];

  V *values = new V[nse];
  uint64_t *indices = new uint64_t[rank * nse];

  for (uint64_t i = 0, base = 0; i < nse; i++) {
    values[i] = elements[i].value;
    for (uint64_t j = 0; j < rank; j++)
      indices[base + j] = elements[i].indices[j];
    base += rank;
  }

  delete coo;
  *pRank = rank;
  *pNse = nse;
  *pShape = shape;
  *pValues = values;
  *pIndices = indices;
}

} // anonymous namespace

extern "C" {

//===----------------------------------------------------------------------===//
//
// Public functions which operate on MLIR buffers (memrefs) to interact
// with sparse tensors (which are only visible as opaque pointers externally).
//
//===----------------------------------------------------------------------===//

#define CASE(p, i, v, P, I, V)                                                 \
  if (ptrTp == (p) && indTp == (i) && valTp == (v)) {                          \
    SparseTensorCOO<V> *coo = nullptr;                                         \
    if (action <= Action::kFromCOO) {                                          \
      if (action == Action::kFromFile) {                                       \
        char *filename = static_cast<char *>(ptr);                             \
        coo = openSparseTensorCOO<V>(filename, rank, shape, perm);             \
      } else if (action == Action::kFromCOO) {                                 \
        coo = static_cast<SparseTensorCOO<V> *>(ptr);                          \
      } else {                                                                 \
        assert(action == Action::kEmpty);                                      \
      }                                                                        \
      auto *tensor = SparseTensorStorage<P, I, V>::newSparseTensor(            \
          rank, shape, perm, sparsity, coo);                                   \
      if (action == Action::kFromFile)                                         \
        delete coo;                                                            \
      return tensor;                                                           \
    }                                                                          \
    if (action == Action::kSparseToSparse) {                                   \
      auto *tensor = static_cast<SparseTensorStorageBase *>(ptr);              \
      return SparseTensorStorage<P, I, V>::newSparseTensor(rank, shape, perm,  \
                                                           sparsity, tensor);  \
    }                                                                          \
    if (action == Action::kEmptyCOO)                                           \
      return SparseTensorCOO<V>::newSparseTensorCOO(rank, shape, perm);        \
    coo = static_cast<SparseTensorStorage<P, I, V> *>(ptr)->toCOO(perm);       \
    if (action == Action::kToIterator) {                                       \
      coo->startIterator();                                                    \
    } else {                                                                   \
      assert(action == Action::kToCOO);                                        \
    }                                                                          \
    return coo;                                                                \
  }

#define CASE_SECSAME(p, v, P, V) CASE(p, p, v, P, P, V)

// Assume index_type is in fact uint64_t, so that _mlir_ciface_newSparseTensor
// can safely rewrite kIndex to kU64.  We make this assertion to guarantee
// that this file cannot get out of sync with its header.
static_assert(std::is_same<index_type, uint64_t>::value,
              "Expected index_type == uint64_t");

void *
_mlir_ciface_newSparseTensor(StridedMemRefType<DimLevelType, 1> *aref, // NOLINT
                             StridedMemRefType<index_type, 1> *sref,
                             StridedMemRefType<index_type, 1> *pref,
                             OverheadType ptrTp, OverheadType indTp,
                             PrimaryType valTp, Action action, void *ptr) {
  assert(aref && sref && pref);
  assert(aref->strides[0] == 1 && sref->strides[0] == 1 &&
         pref->strides[0] == 1);
  assert(aref->sizes[0] == sref->sizes[0] && sref->sizes[0] == pref->sizes[0]);
  const DimLevelType *sparsity = aref->data + aref->offset;
  const index_type *shape = sref->data + sref->offset;
  const index_type *perm = pref->data + pref->offset;
  uint64_t rank = aref->sizes[0];

  // Rewrite kIndex to kU64, to avoid introducing a bunch of new cases.
  // This is safe because of the static_assert above.
  if (ptrTp == OverheadType::kIndex)
    ptrTp = OverheadType::kU64;
  if (indTp == OverheadType::kIndex)
    indTp = OverheadType::kU64;

  // Double matrices with all combinations of overhead storage.
  CASE(OverheadType::kU64, OverheadType::kU64, PrimaryType::kF64, uint64_t,
       uint64_t, double);
  CASE(OverheadType::kU64, OverheadType::kU32, PrimaryType::kF64, uint64_t,
       uint32_t, double);
  CASE(OverheadType::kU64, OverheadType::kU16, PrimaryType::kF64, uint64_t,
       uint16_t, double);
  CASE(OverheadType::kU64, OverheadType::kU8, PrimaryType::kF64, uint64_t,
       uint8_t, double);
  CASE(OverheadType::kU32, OverheadType::kU64, PrimaryType::kF64, uint32_t,
       uint64_t, double);
  CASE(OverheadType::kU32, OverheadType::kU32, PrimaryType::kF64, uint32_t,
       uint32_t, double);
  CASE(OverheadType::kU32, OverheadType::kU16, PrimaryType::kF64, uint32_t,
       uint16_t, double);
  CASE(OverheadType::kU32, OverheadType::kU8, PrimaryType::kF64, uint32_t,
       uint8_t, double);
  CASE(OverheadType::kU16, OverheadType::kU64, PrimaryType::kF64, uint16_t,
       uint64_t, double);
  CASE(OverheadType::kU16, OverheadType::kU32, PrimaryType::kF64, uint16_t,
       uint32_t, double);
  CASE(OverheadType::kU16, OverheadType::kU16, PrimaryType::kF64, uint16_t,
       uint16_t, double);
  CASE(OverheadType::kU16, OverheadType::kU8, PrimaryType::kF64, uint16_t,
       uint8_t, double);
  CASE(OverheadType::kU8, OverheadType::kU64, PrimaryType::kF64, uint8_t,
       uint64_t, double);
  CASE(OverheadType::kU8, OverheadType::kU32, PrimaryType::kF64, uint8_t,
       uint32_t, double);
  CASE(OverheadType::kU8, OverheadType::kU16, PrimaryType::kF64, uint8_t,
       uint16_t, double);
  CASE(OverheadType::kU8, OverheadType::kU8, PrimaryType::kF64, uint8_t,
       uint8_t, double);

  // Float matrices with all combinations of overhead storage.
  CASE(OverheadType::kU64, OverheadType::kU64, PrimaryType::kF32, uint64_t,
       uint64_t, float);
  CASE(OverheadType::kU64, OverheadType::kU32, PrimaryType::kF32, uint64_t,
       uint32_t, float);
  CASE(OverheadType::kU64, OverheadType::kU16, PrimaryType::kF32, uint64_t,
       uint16_t, float);
  CASE(OverheadType::kU64, OverheadType::kU8, PrimaryType::kF32, uint64_t,
       uint8_t, float);
  CASE(OverheadType::kU32, OverheadType::kU64, PrimaryType::kF32, uint32_t,
       uint64_t, float);
  CASE(OverheadType::kU32, OverheadType::kU32, PrimaryType::kF32, uint32_t,
       uint32_t, float);
  CASE(OverheadType::kU32, OverheadType::kU16, PrimaryType::kF32, uint32_t,
       uint16_t, float);
  CASE(OverheadType::kU32, OverheadType::kU8, PrimaryType::kF32, uint32_t,
       uint8_t, float);
  CASE(OverheadType::kU16, OverheadType::kU64, PrimaryType::kF32, uint16_t,
       uint64_t, float);
  CASE(OverheadType::kU16, OverheadType::kU32, PrimaryType::kF32, uint16_t,
       uint32_t, float);
  CASE(OverheadType::kU16, OverheadType::kU16, PrimaryType::kF32, uint16_t,
       uint16_t, float);
  CASE(OverheadType::kU16, OverheadType::kU8, PrimaryType::kF32, uint16_t,
       uint8_t, float);
  CASE(OverheadType::kU8, OverheadType::kU64, PrimaryType::kF32, uint8_t,
       uint64_t, float);
  CASE(OverheadType::kU8, OverheadType::kU32, PrimaryType::kF32, uint8_t,
       uint32_t, float);
  CASE(OverheadType::kU8, OverheadType::kU16, PrimaryType::kF32, uint8_t,
       uint16_t, float);
  CASE(OverheadType::kU8, OverheadType::kU8, PrimaryType::kF32, uint8_t,
       uint8_t, float);

  // Integral matrices with both overheads of the same type.
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kI64, uint64_t, int64_t);
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kI32, uint64_t, int32_t);
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kI16, uint64_t, int16_t);
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kI8, uint64_t, int8_t);
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kI64, uint32_t, int64_t);
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kI32, uint32_t, int32_t);
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kI16, uint32_t, int16_t);
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kI8, uint32_t, int8_t);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kI64, uint16_t, int64_t);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kI32, uint16_t, int32_t);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kI16, uint16_t, int16_t);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kI8, uint16_t, int8_t);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kI64, uint8_t, int64_t);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kI32, uint8_t, int32_t);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kI16, uint8_t, int16_t);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kI8, uint8_t, int8_t);

  // Complex matrices with wide overhead.
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kC64, uint64_t, complex64);
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kC32, uint64_t, complex32);

  // Unsupported case (add above if needed).
  // TODO: better pretty-printing of enum values!
  FATAL("unsupported combination of types: <P=%d, I=%d, V=%d>\n",
        static_cast<int>(ptrTp), static_cast<int>(indTp),
        static_cast<int>(valTp));
}
#undef CASE
#undef CASE_SECSAME

#define IMPL_SPARSEVALUES(VNAME, V)                                            \
  void _mlir_ciface_sparseValues##VNAME(StridedMemRefType<V, 1> *ref,          \
                                        void *tensor) {                        \
    assert(ref &&tensor);                                                      \
    std::vector<V> *v;                                                         \
    static_cast<SparseTensorStorageBase *>(tensor)->getValues(&v);             \
    ref->basePtr = ref->data = v->data();                                      \
    ref->offset = 0;                                                           \
    ref->sizes[0] = v->size();                                                 \
    ref->strides[0] = 1;                                                       \
  }
FOREVERY_V(IMPL_SPARSEVALUES)
#undef IMPL_SPARSEVALUES

#define IMPL_GETOVERHEAD(NAME, TYPE, LIB)                                      \
  void _mlir_ciface_##NAME(StridedMemRefType<TYPE, 1> *ref, void *tensor,      \
                           index_type d) {                                     \
    assert(ref &&tensor);                                                      \
    std::vector<TYPE> *v;                                                      \
    static_cast<SparseTensorStorageBase *>(tensor)->LIB(&v, d);                \
    ref->basePtr = ref->data = v->data();                                      \
    ref->offset = 0;                                                           \
    ref->sizes[0] = v->size();                                                 \
    ref->strides[0] = 1;                                                       \
  }
#define IMPL_SPARSEPOINTERS(PNAME, P)                                          \
  IMPL_GETOVERHEAD(sparsePointers##PNAME, P, getPointers)
FOREVERY_O(IMPL_SPARSEPOINTERS)
#undef IMPL_SPARSEPOINTERS

#define IMPL_SPARSEINDICES(INAME, I)                                           \
  IMPL_GETOVERHEAD(sparseIndices##INAME, I, getIndices)
FOREVERY_O(IMPL_SPARSEINDICES)
#undef IMPL_SPARSEINDICES
#undef IMPL_GETOVERHEAD

#define IMPL_ADDELT(VNAME, V)                                                  \
  void *_mlir_ciface_addElt##VNAME(void *coo, V value,                         \
                                   StridedMemRefType<index_type, 1> *iref,     \
                                   StridedMemRefType<index_type, 1> *pref) {   \
    assert(coo &&iref &&pref);                                                 \
    assert(iref->strides[0] == 1 && pref->strides[0] == 1);                    \
    assert(iref->sizes[0] == pref->sizes[0]);                                  \
    const index_type *indx = iref->data + iref->offset;                        \
    const index_type *perm = pref->data + pref->offset;                        \
    uint64_t isize = iref->sizes[0];                                           \
    std::vector<index_type> indices(isize);                                    \
    for (uint64_t r = 0; r < isize; r++)                                       \
      indices[perm[r]] = indx[r];                                              \
    static_cast<SparseTensorCOO<V> *>(coo)->add(indices, value);               \
    return coo;                                                                \
  }
FOREVERY_SIMPLEX_V(IMPL_ADDELT)
IMPL_ADDELT(C64, complex64)
// Marked static because it's not part of the public API.
// NOTE: the `static` keyword confuses clang-format here, causing
// the strange indentation of the `_mlir_ciface_addEltC32` prototype.
// In C++11 we can add a semicolon after the call to `IMPL_ADDELT`
// and that will correct clang-format.  Alas, this file is compiled
// in C++98 mode where that semicolon is illegal (and there's no portable
// macro magic to license a no-op semicolon at the top level).
static IMPL_ADDELT(C32ABI, complex32)
#undef IMPL_ADDELT
    void *_mlir_ciface_addEltC32(void *coo, float r, float i,
                                 StridedMemRefType<index_type, 1> *iref,
                                 StridedMemRefType<index_type, 1> *pref) {
  return _mlir_ciface_addEltC32ABI(coo, complex32(r, i), iref, pref);
}

#define IMPL_GETNEXT(VNAME, V)                                                 \
  bool _mlir_ciface_getNext##VNAME(void *coo,                                  \
                                   StridedMemRefType<index_type, 1> *iref,     \
                                   StridedMemRefType<V, 0> *vref) {            \
    assert(coo &&iref &&vref);                                                 \
    assert(iref->strides[0] == 1);                                             \
    index_type *indx = iref->data + iref->offset;                              \
    V *value = vref->data + vref->offset;                                      \
    const uint64_t isize = iref->sizes[0];                                     \
    const Element<V> *elem =                                                   \
        static_cast<SparseTensorCOO<V> *>(coo)->getNext();                     \
    if (elem == nullptr)                                                       \
      return false;                                                            \
    for (uint64_t r = 0; r < isize; r++)                                       \
      indx[r] = elem->indices[r];                                              \
    *value = elem->value;                                                      \
    return true;                                                               \
  }
FOREVERY_V(IMPL_GETNEXT)
#undef IMPL_GETNEXT

#define IMPL_LEXINSERT(VNAME, V)                                               \
  void _mlir_ciface_lexInsert##VNAME(                                          \
      void *tensor, StridedMemRefType<index_type, 1> *cref, V val) {           \
    assert(tensor &&cref);                                                     \
    assert(cref->strides[0] == 1);                                             \
    index_type *cursor = cref->data + cref->offset;                            \
    assert(cursor);                                                            \
    static_cast<SparseTensorStorageBase *>(tensor)->lexInsert(cursor, val);    \
  }
FOREVERY_SIMPLEX_V(IMPL_LEXINSERT)
IMPL_LEXINSERT(C64, complex64)
// Marked static because it's not part of the public API.
// NOTE: see the note for `_mlir_ciface_addEltC32ABI`
static IMPL_LEXINSERT(C32ABI, complex32)
#undef IMPL_LEXINSERT
    void _mlir_ciface_lexInsertC32(void *tensor,
                                   StridedMemRefType<index_type, 1> *cref,
                                   float r, float i) {
  _mlir_ciface_lexInsertC32ABI(tensor, cref, complex32(r, i));
}

#define IMPL_EXPINSERT(VNAME, V)                                               \
  void _mlir_ciface_expInsert##VNAME(                                          \
      void *tensor, StridedMemRefType<index_type, 1> *cref,                    \
      StridedMemRefType<V, 1> *vref, StridedMemRefType<bool, 1> *fref,         \
      StridedMemRefType<index_type, 1> *aref, index_type count) {              \
    assert(tensor &&cref &&vref &&fref &&aref);                                \
    assert(cref->strides[0] == 1);                                             \
    assert(vref->strides[0] == 1);                                             \
    assert(fref->strides[0] == 1);                                             \
    assert(aref->strides[0] == 1);                                             \
    assert(vref->sizes[0] == fref->sizes[0]);                                  \
    index_type *cursor = cref->data + cref->offset;                            \
    V *values = vref->data + vref->offset;                                     \
    bool *filled = fref->data + fref->offset;                                  \
    index_type *added = aref->data + aref->offset;                             \
    static_cast<SparseTensorStorageBase *>(tensor)->expInsert(                 \
        cursor, values, filled, added, count);                                 \
  }
FOREVERY_V(IMPL_EXPINSERT)
#undef IMPL_EXPINSERT

//===----------------------------------------------------------------------===//
//
// Public functions which accept only C-style data structures to interact
// with sparse tensors (which are only visible as opaque pointers externally).
//
//===----------------------------------------------------------------------===//

index_type sparseDimSize(void *tensor, index_type d) {
  return static_cast<SparseTensorStorageBase *>(tensor)->getDimSize(d);
}

void endInsert(void *tensor) {
  return static_cast<SparseTensorStorageBase *>(tensor)->endInsert();
}

#define IMPL_OUTSPARSETENSOR(VNAME, V)                                         \
  void outSparseTensor##VNAME(void *coo, void *dest, bool sort) {              \
    return outSparseTensor<V>(coo, dest, sort);                                \
  }
FOREVERY_V(IMPL_OUTSPARSETENSOR)
#undef IMPL_OUTSPARSETENSOR

void delSparseTensor(void *tensor) {
  delete static_cast<SparseTensorStorageBase *>(tensor);
}

#define IMPL_DELCOO(VNAME, V)                                                  \
  void delSparseTensorCOO##VNAME(void *coo) {                                  \
    delete static_cast<SparseTensorCOO<V> *>(coo);                             \
  }
FOREVERY_V(IMPL_DELCOO)
#undef IMPL_DELCOO

char *getTensorFilename(index_type id) {
  char var[80];
  sprintf(var, "TENSOR%" PRIu64, id);
  char *env = getenv(var);
  if (!env)
    FATAL("Environment variable %s is not set\n", var);
  return env;
}

// TODO: generalize beyond 64-bit indices.
#define IMPL_CONVERTTOMLIRSPARSETENSOR(VNAME, V)                               \
  void *convertToMLIRSparseTensor##VNAME(                                      \
      uint64_t rank, uint64_t nse, uint64_t *shape, V *values,                 \
      uint64_t *indices, uint64_t *perm, uint8_t *sparse) {                    \
    return toMLIRSparseTensor<V>(rank, nse, shape, values, indices, perm,      \
                                 sparse);                                      \
  }
FOREVERY_V(IMPL_CONVERTTOMLIRSPARSETENSOR)
#undef IMPL_CONVERTTOMLIRSPARSETENSOR

// TODO: Currently, values are copied from SparseTensorStorage to
// SparseTensorCOO, then to the output.  We may want to reduce the number
// of copies.
//
// TODO: generalize beyond 64-bit indices, no dim ordering, all dimensions
// compressed
#define IMPL_CONVERTFROMMLIRSPARSETENSOR(VNAME, V)                             \
  void convertFromMLIRSparseTensor##VNAME(void *tensor, uint64_t *pRank,       \
                                          uint64_t *pNse, uint64_t **pShape,   \
                                          V **pValues, uint64_t **pIndices) {  \
    fromMLIRSparseTensor<V>(tensor, pRank, pNse, pShape, pValues, pIndices);   \
  }
FOREVERY_V(IMPL_CONVERTFROMMLIRSPARSETENSOR)
#undef IMPL_CONVERTFROMMLIRSPARSETENSOR

} // extern "C"

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
