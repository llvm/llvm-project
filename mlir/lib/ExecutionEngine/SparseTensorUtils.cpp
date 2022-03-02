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
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

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

/// A sparse tensor element in coordinate scheme (value and indices).
/// For example, a rank-1 vector element would look like
///   ({i}, a[i])
/// and a rank-5 tensor element like
///   ({i,j,k,l,m}, a[i,j,k,l,m])
template <typename V>
struct Element {
  Element(const std::vector<uint64_t> &ind, V val) : indices(ind), value(val){};
  std::vector<uint64_t> indices;
  V value;
};

/// A memory-resident sparse tensor in coordinate scheme (collection of
/// elements). This data structure is used to read a sparse tensor from
/// any external format into memory and sort the elements lexicographically
/// by indices before passing it back to the client (most packed storage
/// formats require the elements to appear in lexicographic index order).
template <typename V>
struct SparseTensorCOO {
public:
  SparseTensorCOO(const std::vector<uint64_t> &szs, uint64_t capacity)
      : sizes(szs), iteratorLocked(false), iteratorPos(0) {
    if (capacity)
      elements.reserve(capacity);
  }
  /// Adds element as indices and value.
  void add(const std::vector<uint64_t> &ind, V val) {
    assert(!iteratorLocked && "Attempt to add() after startIterator()");
    uint64_t rank = getRank();
    assert(rank == ind.size());
    for (uint64_t r = 0; r < rank; r++)
      assert(ind[r] < sizes[r]); // within bounds
    elements.emplace_back(ind, val);
  }
  /// Sorts elements lexicographically by index.
  void sort() {
    assert(!iteratorLocked && "Attempt to sort() after startIterator()");
    // TODO: we may want to cache an `isSorted` bit, to avoid
    // unnecessary/redundant sorting.
    std::sort(elements.begin(), elements.end(), lexOrder);
  }
  /// Returns rank.
  uint64_t getRank() const { return sizes.size(); }
  /// Getter for sizes array.
  const std::vector<uint64_t> &getSizes() const { return sizes; }
  /// Getter for elements array.
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
  static SparseTensorCOO<V> *newSparseTensorCOO(uint64_t rank,
                                                const uint64_t *sizes,
                                                const uint64_t *perm,
                                                uint64_t capacity = 0) {
    std::vector<uint64_t> permsz(rank);
    for (uint64_t r = 0; r < rank; r++)
      permsz[perm[r]] = sizes[r];
    return new SparseTensorCOO<V>(permsz, capacity);
  }

private:
  /// Returns true if indices of e1 < indices of e2.
  static bool lexOrder(const Element<V> &e1, const Element<V> &e2) {
    uint64_t rank = e1.indices.size();
    assert(rank == e2.indices.size());
    for (uint64_t r = 0; r < rank; r++) {
      if (e1.indices[r] == e2.indices[r])
        continue;
      return e1.indices[r] < e2.indices[r];
    }
    return false;
  }
  const std::vector<uint64_t> sizes; // per-dimension sizes
  std::vector<Element<V>> elements;
  bool iteratorLocked;
  unsigned iteratorPos;
};

/// Abstract base class of sparse tensor storage. Note that we use
/// function overloading to implement "partial" method specialization.
class SparseTensorStorageBase {
public:
  /// Dimension size query.
  virtual uint64_t getDimSize(uint64_t) = 0;

  /// Overhead storage.
  virtual void getPointers(std::vector<uint64_t> **, uint64_t) { fatal("p64"); }
  virtual void getPointers(std::vector<uint32_t> **, uint64_t) { fatal("p32"); }
  virtual void getPointers(std::vector<uint16_t> **, uint64_t) { fatal("p16"); }
  virtual void getPointers(std::vector<uint8_t> **, uint64_t) { fatal("p8"); }
  virtual void getIndices(std::vector<uint64_t> **, uint64_t) { fatal("i64"); }
  virtual void getIndices(std::vector<uint32_t> **, uint64_t) { fatal("i32"); }
  virtual void getIndices(std::vector<uint16_t> **, uint64_t) { fatal("i16"); }
  virtual void getIndices(std::vector<uint8_t> **, uint64_t) { fatal("i8"); }

  /// Primary storage.
  virtual void getValues(std::vector<double> **) { fatal("valf64"); }
  virtual void getValues(std::vector<float> **) { fatal("valf32"); }
  virtual void getValues(std::vector<int64_t> **) { fatal("vali64"); }
  virtual void getValues(std::vector<int32_t> **) { fatal("vali32"); }
  virtual void getValues(std::vector<int16_t> **) { fatal("vali16"); }
  virtual void getValues(std::vector<int8_t> **) { fatal("vali8"); }

  /// Element-wise insertion in lexicographic index order.
  virtual void lexInsert(const uint64_t *, double) { fatal("insf64"); }
  virtual void lexInsert(const uint64_t *, float) { fatal("insf32"); }
  virtual void lexInsert(const uint64_t *, int64_t) { fatal("insi64"); }
  virtual void lexInsert(const uint64_t *, int32_t) { fatal("insi32"); }
  virtual void lexInsert(const uint64_t *, int16_t) { fatal("ins16"); }
  virtual void lexInsert(const uint64_t *, int8_t) { fatal("insi8"); }

  /// Expanded insertion.
  virtual void expInsert(uint64_t *, double *, bool *, uint64_t *, uint64_t) {
    fatal("expf64");
  }
  virtual void expInsert(uint64_t *, float *, bool *, uint64_t *, uint64_t) {
    fatal("expf32");
  }
  virtual void expInsert(uint64_t *, int64_t *, bool *, uint64_t *, uint64_t) {
    fatal("expi64");
  }
  virtual void expInsert(uint64_t *, int32_t *, bool *, uint64_t *, uint64_t) {
    fatal("expi32");
  }
  virtual void expInsert(uint64_t *, int16_t *, bool *, uint64_t *, uint64_t) {
    fatal("expi16");
  }
  virtual void expInsert(uint64_t *, int8_t *, bool *, uint64_t *, uint64_t) {
    fatal("expi8");
  }

  /// Finishes insertion.
  virtual void endInsert() = 0;

  virtual ~SparseTensorStorageBase() = default;

private:
  void fatal(const char *tp) {
    fprintf(stderr, "unsupported %s\n", tp);
    exit(1);
  }
};

/// A memory-resident sparse tensor using a storage scheme based on
/// per-dimension sparse/dense annotations. This data structure provides a
/// bufferized form of a sparse tensor type. In contrast to generating setup
/// methods for each differently annotated sparse tensor, this method provides
/// a convenient "one-size-fits-all" solution that simply takes an input tensor
/// and annotations to implement all required setup in a general manner.
template <typename P, typename I, typename V>
class SparseTensorStorage : public SparseTensorStorageBase {
public:
  /// Constructs a sparse tensor storage scheme with the given dimensions,
  /// permutation, and per-dimension dense/sparse annotations, using
  /// the coordinate scheme tensor for the initial contents if provided.
  SparseTensorStorage(const std::vector<uint64_t> &szs, const uint64_t *perm,
                      const DimLevelType *sparsity,
                      SparseTensorCOO<V> *tensor = nullptr)
      : sizes(szs), rev(getRank()), idx(getRank()), pointers(getRank()),
        indices(getRank()) {
    uint64_t rank = getRank();
    // Store "reverse" permutation.
    for (uint64_t r = 0; r < rank; r++)
      rev[perm[r]] = r;
    // Provide hints on capacity of pointers and indices.
    // TODO: needs fine-tuning based on sparsity
    bool allDense = true;
    uint64_t sz = 1;
    for (uint64_t r = 0; r < rank; r++) {
      assert(sizes[r] > 0 && "Dimension size zero has trivial storage");
      sz *= sizes[r];
      if (sparsity[r] == DimLevelType::kCompressed) {
        pointers[r].reserve(sz + 1);
        indices[r].reserve(sz);
        sz = 1;
        allDense = false;
        // Prepare the pointer structure.  We cannot use `addPointer`
        // here, because `isCompressedDim` won't work until after this
        // preparation has been done.
        pointers[r].push_back(0);
      } else {
        assert(sparsity[r] == DimLevelType::kDense &&
               "singleton not yet supported");
      }
    }
    // Then assign contents from coordinate scheme tensor if provided.
    if (tensor) {
      // Ensure both preconditions of `fromCOO`.
      assert(tensor->getSizes() == sizes && "Tensor size mismatch");
      tensor->sort();
      // Now actually insert the `elements`.
      const std::vector<Element<V>> &elements = tensor->getElements();
      uint64_t nnz = elements.size();
      values.reserve(nnz);
      fromCOO(elements, 0, nnz, 0);
    } else if (allDense) {
      values.resize(sz, 0);
    }
  }

  ~SparseTensorStorage() override = default;

  /// Get the rank of the tensor.
  uint64_t getRank() const { return sizes.size(); }

  /// Get the size in the given dimension of the tensor.
  uint64_t getDimSize(uint64_t d) override {
    assert(d < getRank());
    return sizes[d];
  }

  /// Partially specialize these getter methods based on template types.
  void getPointers(std::vector<P> **out, uint64_t d) override {
    assert(d < getRank());
    *out = &pointers[d];
  }
  void getIndices(std::vector<I> **out, uint64_t d) override {
    assert(d < getRank());
    *out = &indices[d];
  }
  void getValues(std::vector<V> **out) override { *out = &values; }

  /// Partially specialize lexicographical insertions based on template types.
  void lexInsert(const uint64_t *cursor, V val) override {
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
                 uint64_t count) override {
    if (count == 0)
      return;
    // Sort.
    std::sort(added, added + count);
    // Restore insertion path for first insert.
    uint64_t rank = getRank();
    uint64_t index = added[0];
    cursor[rank - 1] = index;
    lexInsert(cursor, values[index]);
    assert(filled[index]);
    values[index] = 0;
    filled[index] = false;
    // Subsequent insertions are quick.
    for (uint64_t i = 1; i < count; i++) {
      assert(index < added[i] && "non-lexicographic insertion");
      index = added[i];
      cursor[rank - 1] = index;
      insPath(cursor, rank - 1, added[i - 1] + 1, values[index]);
      assert(filled[index]);
      values[index] = 0.0;
      filled[index] = false;
    }
  }

  /// Finalizes lexicographic insertions.
  void endInsert() override {
    if (values.empty())
      endDim(0);
    else
      endPath(0);
  }

  /// Returns this sparse tensor storage scheme as a new memory-resident
  /// sparse tensor in coordinate scheme with the given dimension order.
  SparseTensorCOO<V> *toCOO(const uint64_t *perm) {
    // Restore original order of the dimension sizes and allocate coordinate
    // scheme with desired new ordering specified in perm.
    uint64_t rank = getRank();
    std::vector<uint64_t> orgsz(rank);
    for (uint64_t r = 0; r < rank; r++)
      orgsz[rev[r]] = sizes[r];
    SparseTensorCOO<V> *tensor = SparseTensorCOO<V>::newSparseTensorCOO(
        rank, orgsz.data(), perm, values.size());
    // Populate coordinate scheme restored from old ordering and changed with
    // new ordering. Rather than applying both reorderings during the recursion,
    // we compute the combine permutation in advance.
    std::vector<uint64_t> reord(rank);
    for (uint64_t r = 0; r < rank; r++)
      reord[r] = perm[rev[r]];
    toCOO(*tensor, reord, 0, 0);
    assert(tensor->getElements().size() == values.size());
    return tensor;
  }

  /// Factory method. Constructs a sparse tensor storage scheme with the given
  /// dimensions, permutation, and per-dimension dense/sparse annotations,
  /// using the coordinate scheme tensor for the initial contents if provided.
  /// In the latter case, the coordinate scheme must respect the same
  /// permutation as is desired for the new sparse tensor storage.
  static SparseTensorStorage<P, I, V> *
  newSparseTensor(uint64_t rank, const uint64_t *sizes, const uint64_t *perm,
                  const DimLevelType *sparsity, SparseTensorCOO<V> *tensor) {
    SparseTensorStorage<P, I, V> *n = nullptr;
    if (tensor) {
      assert(tensor->getRank() == rank);
      for (uint64_t r = 0; r < rank; r++)
        assert(sizes[r] == 0 || tensor->getSizes()[perm[r]] == sizes[r]);
      n = new SparseTensorStorage<P, I, V>(tensor->getSizes(), perm, sparsity,
                                           tensor);
      delete tensor;
    } else {
      std::vector<uint64_t> permsz(rank);
      for (uint64_t r = 0; r < rank; r++)
        permsz[perm[r]] = sizes[r];
      n = new SparseTensorStorage<P, I, V>(permsz, perm, sparsity);
    }
    return n;
  }

private:
  /// Appends the next free position of `indices[d]` to `pointers[d]`.
  /// Thus, when called after inserting the last element of a segment,
  /// it will append the position where the next segment begins.
  inline void addPointer(uint64_t d) {
    assert(isCompressedDim(d)); // Entails `d < getRank()`.
    uint64_t p = indices[d].size();
    assert(p <= std::numeric_limits<P>::max() &&
           "Pointer value is too large for the P-type");
    pointers[d].push_back(p); // Here is where we convert to `P`.
  }

  /// Appends the given index to `indices[d]`.
  inline void addIndex(uint64_t d, uint64_t i) {
    assert(isCompressedDim(d)); // Entails `d < getRank()`.
    assert(i <= std::numeric_limits<I>::max() &&
           "Index value is too large for the I-type");
    indices[d].push_back(i); // Here is where we convert to `I`.
  }

  /// Initializes sparse tensor storage scheme from a memory-resident sparse
  /// tensor in coordinate scheme. This method prepares the pointers and
  /// indices arrays under the given per-dimension dense/sparse annotations.
  ///
  /// Preconditions:
  /// (1) the `elements` must be lexicographically sorted.
  /// (2) the indices of every element are valid for `sizes` (equal rank
  ///     and pointwise less-than).
  void fromCOO(const std::vector<Element<V>> &elements, uint64_t lo,
               uint64_t hi, uint64_t d) {
    // Once dimensions are exhausted, insert the numerical values.
    assert(d <= getRank() && hi <= elements.size());
    if (d == getRank()) {
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
      if (isCompressedDim(d)) {
        addIndex(d, i);
      } else {
        // For dense storage we must fill in all the zero values between
        // the previous element (when last we ran this for-loop) and the
        // current element.
        for (; full < i; full++)
          endDim(d + 1);
        full++;
      }
      fromCOO(elements, lo, seg, d + 1);
      // And move on to next segment in interval.
      lo = seg;
    }
    // Finalize the sparse pointer structure at this dimension.
    if (isCompressedDim(d)) {
      addPointer(d);
    } else {
      // For dense storage we must fill in all the zero values after
      // the last element.
      for (uint64_t sz = sizes[d]; full < sz; full++)
        endDim(d + 1);
    }
  }

  /// Stores the sparse tensor storage scheme into a memory-resident sparse
  /// tensor in coordinate scheme.
  void toCOO(SparseTensorCOO<V> &tensor, std::vector<uint64_t> &reord,
             uint64_t pos, uint64_t d) {
    assert(d <= getRank());
    if (d == getRank()) {
      assert(pos < values.size());
      tensor.add(idx, values[pos]);
    } else if (isCompressedDim(d)) {
      // Sparse dimension.
      for (uint64_t ii = pointers[d][pos]; ii < pointers[d][pos + 1]; ii++) {
        idx[reord[d]] = indices[d][ii];
        toCOO(tensor, reord, ii, d + 1);
      }
    } else {
      // Dense dimension.
      for (uint64_t i = 0, sz = sizes[d], off = pos * sz; i < sz; i++) {
        idx[reord[d]] = i;
        toCOO(tensor, reord, off + i, d + 1);
      }
    }
  }

  /// Ends a deeper, never seen before dimension.
  void endDim(uint64_t d) {
    assert(d <= getRank());
    if (d == getRank()) {
      values.push_back(0);
    } else if (isCompressedDim(d)) {
      addPointer(d);
    } else {
      for (uint64_t full = 0, sz = sizes[d]; full < sz; full++)
        endDim(d + 1);
    }
  }

  /// Wraps up a single insertion path, inner to outer.
  void endPath(uint64_t diff) {
    uint64_t rank = getRank();
    assert(diff <= rank);
    for (uint64_t i = 0; i < rank - diff; i++) {
      uint64_t d = rank - i - 1;
      if (isCompressedDim(d)) {
        addPointer(d);
      } else {
        for (uint64_t full = idx[d] + 1, sz = sizes[d]; full < sz; full++)
          endDim(d + 1);
      }
    }
  }

  /// Continues a single insertion path, outer to inner.
  void insPath(const uint64_t *cursor, uint64_t diff, uint64_t top, V val) {
    uint64_t rank = getRank();
    assert(diff < rank);
    for (uint64_t d = diff; d < rank; d++) {
      uint64_t i = cursor[d];
      if (isCompressedDim(d)) {
        addIndex(d, i);
      } else {
        for (uint64_t full = top; full < i; full++)
          endDim(d + 1);
      }
      top = 0;
      idx[d] = i;
    }
    values.push_back(val);
  }

  /// Finds the lexicographic differing dimension.
  uint64_t lexDiff(const uint64_t *cursor) {
    for (uint64_t r = 0, rank = getRank(); r < rank; r++)
      if (cursor[r] > idx[r])
        return r;
      else
        assert(cursor[r] == idx[r] && "non-lexicographic insertion");
    assert(0 && "duplication insertion");
    return -1u;
  }

  /// Returns true if dimension is compressed.
  inline bool isCompressedDim(uint64_t d) const {
    assert(d < getRank());
    return (!pointers[d].empty());
  }

private:
  std::vector<uint64_t> sizes; // per-dimension sizes
  std::vector<uint64_t> rev;   // "reverse" permutation
  std::vector<uint64_t> idx;   // index cursor
  std::vector<std::vector<P>> pointers;
  std::vector<std::vector<I>> indices;
  std::vector<V> values;
};

/// Helper to convert string to lower case.
static char *toLower(char *token) {
  for (char *c = token; *c; c++)
    *c = tolower(*c);
  return token;
}

/// Read the MME header of a general sparse matrix of type real.
static void readMMEHeader(FILE *file, char *filename, char *line,
                          uint64_t *idata, bool *isSymmetric) {
  char header[64];
  char object[64];
  char format[64];
  char field[64];
  char symmetry[64];
  // Read header line.
  if (fscanf(file, "%63s %63s %63s %63s %63s\n", header, object, format, field,
             symmetry) != 5) {
    fprintf(stderr, "Corrupt header in %s\n", filename);
    exit(1);
  }
  *isSymmetric = (strcmp(toLower(symmetry), "symmetric") == 0);
  // Make sure this is a general sparse matrix.
  if (strcmp(toLower(header), "%%matrixmarket") ||
      strcmp(toLower(object), "matrix") ||
      strcmp(toLower(format), "coordinate") || strcmp(toLower(field), "real") ||
      (strcmp(toLower(symmetry), "general") && !(*isSymmetric))) {
    fprintf(stderr,
            "Cannot find a general sparse matrix with type real in %s\n",
            filename);
    exit(1);
  }
  // Skip comments.
  while (true) {
    if (!fgets(line, kColWidth, file)) {
      fprintf(stderr, "Cannot find data in %s\n", filename);
      exit(1);
    }
    if (line[0] != '%')
      break;
  }
  // Next line contains M N NNZ.
  idata[0] = 2; // rank
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "%" PRIu64 "\n", idata + 2, idata + 3,
             idata + 1) != 3) {
    fprintf(stderr, "Cannot find size in %s\n", filename);
    exit(1);
  }
}

/// Read the "extended" FROSTT header. Although not part of the documented
/// format, we assume that the file starts with optional comments followed
/// by two lines that define the rank, the number of nonzeros, and the
/// dimensions sizes (one per rank) of the sparse tensor.
static void readExtFROSTTHeader(FILE *file, char *filename, char *line,
                                uint64_t *idata) {
  // Skip comments.
  while (true) {
    if (!fgets(line, kColWidth, file)) {
      fprintf(stderr, "Cannot find data in %s\n", filename);
      exit(1);
    }
    if (line[0] != '#')
      break;
  }
  // Next line contains RANK and NNZ.
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "\n", idata, idata + 1) != 2) {
    fprintf(stderr, "Cannot find metadata in %s\n", filename);
    exit(1);
  }
  // Followed by a line with the dimension sizes (one per rank).
  for (uint64_t r = 0; r < idata[0]; r++) {
    if (fscanf(file, "%" PRIu64, idata + 2 + r) != 1) {
      fprintf(stderr, "Cannot find dimension size %s\n", filename);
      exit(1);
    }
  }
  fgets(line, kColWidth, file); // end of line
}

/// Reads a sparse tensor with the given filename into a memory-resident
/// sparse tensor in coordinate scheme.
template <typename V>
static SparseTensorCOO<V> *openSparseTensorCOO(char *filename, uint64_t rank,
                                               const uint64_t *sizes,
                                               const uint64_t *perm) {
  // Open the file.
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Cannot find %s\n", filename);
    exit(1);
  }
  // Perform some file format dependent set up.
  char line[kColWidth];
  uint64_t idata[512];
  bool isSymmetric = false;
  if (strstr(filename, ".mtx")) {
    readMMEHeader(file, filename, line, idata, &isSymmetric);
  } else if (strstr(filename, ".tns")) {
    readExtFROSTTHeader(file, filename, line, idata);
  } else {
    fprintf(stderr, "Unknown format %s\n", filename);
    exit(1);
  }
  // Prepare sparse tensor object with per-dimension sizes
  // and the number of nonzeros as initial capacity.
  assert(rank == idata[0] && "rank mismatch");
  uint64_t nnz = idata[1];
  for (uint64_t r = 0; r < rank; r++)
    assert((sizes[r] == 0 || sizes[r] == idata[2 + r]) &&
           "dimension size mismatch");
  SparseTensorCOO<V> *tensor =
      SparseTensorCOO<V>::newSparseTensorCOO(rank, idata + 2, perm, nnz);
  //  Read all nonzero elements.
  std::vector<uint64_t> indices(rank);
  for (uint64_t k = 0; k < nnz; k++) {
    if (!fgets(line, kColWidth, file)) {
      fprintf(stderr, "Cannot find next line of data in %s\n", filename);
      exit(1);
    }
    char *linePtr = line;
    for (uint64_t r = 0; r < rank; r++) {
      uint64_t idx = strtoul(linePtr, &linePtr, 10);
      // Add 0-based index.
      indices[perm[r]] = idx - 1;
    }
    // The external formats always store the numerical values with the type
    // double, but we cast these values to the sparse tensor object type.
    double value = strtod(linePtr, &linePtr);
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

/// Writes the sparse tensor to extended FROSTT format.
template <typename V>
void outSparseTensor(void *tensor, void *dest, bool sort) {
  assert(tensor && dest);
  auto coo = static_cast<SparseTensorCOO<V> *>(tensor);
  if (sort)
    coo->sort();
  char *filename = static_cast<char *>(dest);
  auto &sizes = coo->getSizes();
  auto &elements = coo->getElements();
  uint64_t rank = coo->getRank();
  uint64_t nnz = elements.size();
  std::fstream file;
  file.open(filename, std::ios_base::out | std::ios_base::trunc);
  assert(file.is_open());
  file << "; extended FROSTT format\n" << rank << " " << nnz << std::endl;
  for (uint64_t r = 0; r < rank - 1; r++)
    file << sizes[r] << " ";
  file << sizes[rank - 1] << std::endl;
  for (uint64_t i = 0; i < nnz; i++) {
    auto &idx = elements[i].indices;
    for (uint64_t r = 0; r < rank; r++)
      file << (idx[r] + 1) << " ";
    file << elements[i].value << std::endl;
  }
  file.flush();
  file.close();
  assert(file.good());
  delete coo;
}

/// Initializes sparse tensor from an external COO-flavored format.
template <typename V>
SparseTensorStorage<uint64_t, uint64_t, V> *
toMLIRSparseTensor(uint64_t rank, uint64_t nse, uint64_t *shape, V *values,
                   uint64_t *indices, uint64_t *perm, uint8_t *sparse) {
  const DimLevelType *sparsity = (DimLevelType *)(sparse);
#ifndef NDEBUG
  // Verify that perm is a permutation of 0..(rank-1).
  std::vector<uint64_t> order(perm, perm + rank);
  std::sort(order.begin(), order.end());
  for (uint64_t i = 0; i < rank; ++i) {
    if (i != order[i]) {
      fprintf(stderr, "Permutation is not a permutation of 0..%lu\n", rank);
      exit(1);
    }
  }

  // Verify that the sparsity values are supported.
  for (uint64_t i = 0; i < rank; ++i) {
    if (sparsity[i] != DimLevelType::kDense &&
        sparsity[i] != DimLevelType::kCompressed) {
      fprintf(stderr, "Unsupported sparsity value %d\n",
              static_cast<int>(sparsity[i]));
      exit(1);
    }
  }
#endif

  // Convert external format to internal COO.
  auto *tensor = SparseTensorCOO<V>::newSparseTensorCOO(rank, shape, perm, nse);
  std::vector<uint64_t> idx(rank);
  for (uint64_t i = 0, base = 0; i < nse; i++) {
    for (uint64_t r = 0; r < rank; r++)
      idx[r] = indices[base + r];
    tensor->add(idx, values[i]);
    base += rank;
  }
  // Return sparse tensor storage format as opaque pointer.
  return SparseTensorStorage<uint64_t, uint64_t, V>::newSparseTensor(
      rank, shape, perm, sparsity, tensor);
}

/// Converts a sparse tensor to an external COO-flavored format.
template <typename V>
void fromMLIRSparseTensor(void *tensor, uint64_t *pRank, uint64_t *pNse,
                          uint64_t **pShape, V **pValues, uint64_t **pIndices) {
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
    shape[i] = coo->getSizes()[i];

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

} // namespace

extern "C" {

//===----------------------------------------------------------------------===//
//
// Public API with methods that operate on MLIR buffers (memrefs) to interact
// with sparse tensors, which are only visible as opaque pointers externally.
// These methods should be used exclusively by MLIR compiler-generated code.
//
// Some macro magic is used to generate implementations for all required type
// combinations that can be called from MLIR compiler-generated code.
//
//===----------------------------------------------------------------------===//

#define CASE(p, i, v, P, I, V)                                                 \
  if (ptrTp == (p) && indTp == (i) && valTp == (v)) {                          \
    SparseTensorCOO<V> *tensor = nullptr;                                      \
    if (action <= Action::kFromCOO) {                                          \
      if (action == Action::kFromFile) {                                       \
        char *filename = static_cast<char *>(ptr);                             \
        tensor = openSparseTensorCOO<V>(filename, rank, sizes, perm);          \
      } else if (action == Action::kFromCOO) {                                 \
        tensor = static_cast<SparseTensorCOO<V> *>(ptr);                       \
      } else {                                                                 \
        assert(action == Action::kEmpty);                                      \
      }                                                                        \
      return SparseTensorStorage<P, I, V>::newSparseTensor(rank, sizes, perm,  \
                                                           sparsity, tensor);  \
    }                                                                          \
    if (action == Action::kEmptyCOO)                                           \
      return SparseTensorCOO<V>::newSparseTensorCOO(rank, sizes, perm);        \
    tensor = static_cast<SparseTensorStorage<P, I, V> *>(ptr)->toCOO(perm);    \
    if (action == Action::kToIterator) {                                       \
      tensor->startIterator();                                                 \
    } else {                                                                   \
      assert(action == Action::kToCOO);                                        \
    }                                                                          \
    return tensor;                                                             \
  }

#define CASE_SECSAME(p, v, P, V) CASE(p, p, v, P, P, V)

#define IMPL_SPARSEVALUES(NAME, TYPE, LIB)                                     \
  void _mlir_ciface_##NAME(StridedMemRefType<TYPE, 1> *ref, void *tensor) {    \
    assert(ref &&tensor);                                                      \
    std::vector<TYPE> *v;                                                      \
    static_cast<SparseTensorStorageBase *>(tensor)->LIB(&v);                   \
    ref->basePtr = ref->data = v->data();                                      \
    ref->offset = 0;                                                           \
    ref->sizes[0] = v->size();                                                 \
    ref->strides[0] = 1;                                                       \
  }

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

#define IMPL_ADDELT(NAME, TYPE)                                                \
  void *_mlir_ciface_##NAME(void *tensor, TYPE value,                          \
                            StridedMemRefType<index_type, 1> *iref,            \
                            StridedMemRefType<index_type, 1> *pref) {          \
    assert(tensor &&iref &&pref);                                              \
    assert(iref->strides[0] == 1 && pref->strides[0] == 1);                    \
    assert(iref->sizes[0] == pref->sizes[0]);                                  \
    const index_type *indx = iref->data + iref->offset;                        \
    const index_type *perm = pref->data + pref->offset;                        \
    uint64_t isize = iref->sizes[0];                                           \
    std::vector<index_type> indices(isize);                                    \
    for (uint64_t r = 0; r < isize; r++)                                       \
      indices[perm[r]] = indx[r];                                              \
    static_cast<SparseTensorCOO<TYPE> *>(tensor)->add(indices, value);         \
    return tensor;                                                             \
  }

#define IMPL_GETNEXT(NAME, V)                                                  \
  bool _mlir_ciface_##NAME(void *tensor,                                       \
                           StridedMemRefType<index_type, 1> *iref,             \
                           StridedMemRefType<V, 0> *vref) {                    \
    assert(tensor &&iref &&vref);                                              \
    assert(iref->strides[0] == 1);                                             \
    index_type *indx = iref->data + iref->offset;                              \
    V *value = vref->data + vref->offset;                                      \
    const uint64_t isize = iref->sizes[0];                                     \
    auto iter = static_cast<SparseTensorCOO<V> *>(tensor);                     \
    const Element<V> *elem = iter->getNext();                                  \
    if (elem == nullptr) {                                                     \
      delete iter;                                                             \
      return false;                                                            \
    }                                                                          \
    for (uint64_t r = 0; r < isize; r++)                                       \
      indx[r] = elem->indices[r];                                              \
    *value = elem->value;                                                      \
    return true;                                                               \
  }

#define IMPL_LEXINSERT(NAME, V)                                                \
  void _mlir_ciface_##NAME(void *tensor,                                       \
                           StridedMemRefType<index_type, 1> *cref, V val) {    \
    assert(tensor &&cref);                                                     \
    assert(cref->strides[0] == 1);                                             \
    index_type *cursor = cref->data + cref->offset;                            \
    assert(cursor);                                                            \
    static_cast<SparseTensorStorageBase *>(tensor)->lexInsert(cursor, val);    \
  }

#define IMPL_EXPINSERT(NAME, V)                                                \
  void _mlir_ciface_##NAME(                                                    \
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

// Assume index_type is in fact uint64_t, so that _mlir_ciface_newSparseTensor
// can safely rewrite kIndex to kU64.  We make this assertion to guarantee
// that this file cannot get out of sync with its header.
static_assert(std::is_same<index_type, uint64_t>::value,
              "Expected index_type == uint64_t");

/// Constructs a new sparse tensor. This is the "swiss army knife"
/// method for materializing sparse tensors into the computation.
///
/// Action:
/// kEmpty = returns empty storage to fill later
/// kFromFile = returns storage, where ptr contains filename to read
/// kFromCOO = returns storage, where ptr contains coordinate scheme to assign
/// kEmptyCOO = returns empty coordinate scheme to fill and use with kFromCOO
/// kToCOO = returns coordinate scheme from storage in ptr to use with kFromCOO
/// kToIterator = returns iterator from storage in ptr (call getNext() to use)
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
  const index_type *sizes = sref->data + sref->offset;
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
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kI32, uint32_t, int32_t);
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kI16, uint32_t, int16_t);
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kI8, uint32_t, int8_t);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kI32, uint16_t, int32_t);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kI16, uint16_t, int16_t);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kI8, uint16_t, int8_t);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kI32, uint8_t, int32_t);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kI16, uint8_t, int16_t);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kI8, uint8_t, int8_t);

  // Unsupported case (add above if needed).
  fputs("unsupported combination of types\n", stderr);
  exit(1);
}

/// Methods that provide direct access to pointers.
IMPL_GETOVERHEAD(sparsePointers, index_type, getPointers)
IMPL_GETOVERHEAD(sparsePointers64, uint64_t, getPointers)
IMPL_GETOVERHEAD(sparsePointers32, uint32_t, getPointers)
IMPL_GETOVERHEAD(sparsePointers16, uint16_t, getPointers)
IMPL_GETOVERHEAD(sparsePointers8, uint8_t, getPointers)

/// Methods that provide direct access to indices.
IMPL_GETOVERHEAD(sparseIndices, index_type, getIndices)
IMPL_GETOVERHEAD(sparseIndices64, uint64_t, getIndices)
IMPL_GETOVERHEAD(sparseIndices32, uint32_t, getIndices)
IMPL_GETOVERHEAD(sparseIndices16, uint16_t, getIndices)
IMPL_GETOVERHEAD(sparseIndices8, uint8_t, getIndices)

/// Methods that provide direct access to values.
IMPL_SPARSEVALUES(sparseValuesF64, double, getValues)
IMPL_SPARSEVALUES(sparseValuesF32, float, getValues)
IMPL_SPARSEVALUES(sparseValuesI64, int64_t, getValues)
IMPL_SPARSEVALUES(sparseValuesI32, int32_t, getValues)
IMPL_SPARSEVALUES(sparseValuesI16, int16_t, getValues)
IMPL_SPARSEVALUES(sparseValuesI8, int8_t, getValues)

/// Helper to add value to coordinate scheme, one per value type.
IMPL_ADDELT(addEltF64, double)
IMPL_ADDELT(addEltF32, float)
IMPL_ADDELT(addEltI64, int64_t)
IMPL_ADDELT(addEltI32, int32_t)
IMPL_ADDELT(addEltI16, int16_t)
IMPL_ADDELT(addEltI8, int8_t)

/// Helper to enumerate elements of coordinate scheme, one per value type.
IMPL_GETNEXT(getNextF64, double)
IMPL_GETNEXT(getNextF32, float)
IMPL_GETNEXT(getNextI64, int64_t)
IMPL_GETNEXT(getNextI32, int32_t)
IMPL_GETNEXT(getNextI16, int16_t)
IMPL_GETNEXT(getNextI8, int8_t)

/// Insert elements in lexicographical index order, one per value type.
IMPL_LEXINSERT(lexInsertF64, double)
IMPL_LEXINSERT(lexInsertF32, float)
IMPL_LEXINSERT(lexInsertI64, int64_t)
IMPL_LEXINSERT(lexInsertI32, int32_t)
IMPL_LEXINSERT(lexInsertI16, int16_t)
IMPL_LEXINSERT(lexInsertI8, int8_t)

/// Insert using expansion, one per value type.
IMPL_EXPINSERT(expInsertF64, double)
IMPL_EXPINSERT(expInsertF32, float)
IMPL_EXPINSERT(expInsertI64, int64_t)
IMPL_EXPINSERT(expInsertI32, int32_t)
IMPL_EXPINSERT(expInsertI16, int16_t)
IMPL_EXPINSERT(expInsertI8, int8_t)

#undef CASE
#undef IMPL_SPARSEVALUES
#undef IMPL_GETOVERHEAD
#undef IMPL_ADDELT
#undef IMPL_GETNEXT
#undef IMPL_LEXINSERT
#undef IMPL_EXPINSERT

/// Output a sparse tensor, one per value type.
void outSparseTensorF64(void *tensor, void *dest, bool sort) {
  return outSparseTensor<double>(tensor, dest, sort);
}
void outSparseTensorF32(void *tensor, void *dest, bool sort) {
  return outSparseTensor<float>(tensor, dest, sort);
}
void outSparseTensorI64(void *tensor, void *dest, bool sort) {
  return outSparseTensor<int64_t>(tensor, dest, sort);
}
void outSparseTensorI32(void *tensor, void *dest, bool sort) {
  return outSparseTensor<int32_t>(tensor, dest, sort);
}
void outSparseTensorI16(void *tensor, void *dest, bool sort) {
  return outSparseTensor<int16_t>(tensor, dest, sort);
}
void outSparseTensorI8(void *tensor, void *dest, bool sort) {
  return outSparseTensor<int8_t>(tensor, dest, sort);
}

//===----------------------------------------------------------------------===//
//
// Public API with methods that accept C-style data structures to interact
// with sparse tensors, which are only visible as opaque pointers externally.
// These methods can be used both by MLIR compiler-generated code as well as by
// an external runtime that wants to interact with MLIR compiler-generated code.
//
//===----------------------------------------------------------------------===//

/// Helper method to read a sparse tensor filename from the environment,
/// defined with the naming convention ${TENSOR0}, ${TENSOR1}, etc.
char *getTensorFilename(index_type id) {
  char var[80];
  sprintf(var, "TENSOR%" PRIu64, id);
  char *env = getenv(var);
  return env;
}

/// Returns size of sparse tensor in given dimension.
index_type sparseDimSize(void *tensor, index_type d) {
  return static_cast<SparseTensorStorageBase *>(tensor)->getDimSize(d);
}

/// Finalizes lexicographic insertions.
void endInsert(void *tensor) {
  return static_cast<SparseTensorStorageBase *>(tensor)->endInsert();
}

/// Releases sparse tensor storage.
void delSparseTensor(void *tensor) {
  delete static_cast<SparseTensorStorageBase *>(tensor);
}

/// Initializes sparse tensor from a COO-flavored format expressed using C-style
/// data structures. The expected parameters are:
///
///   rank:    rank of tensor
///   nse:     number of specified elements (usually the nonzeros)
///   shape:   array with dimension size for each rank
///   values:  a "nse" array with values for all specified elements
///   indices: a flat "nse x rank" array with indices for all specified elements
///   perm:    the permutation of the dimensions in the storage
///   sparse:  the sparsity for the dimensions
///
/// For example, the sparse matrix
///     | 1.0 0.0 0.0 |
///     | 0.0 5.0 3.0 |
/// can be passed as
///      rank    = 2
///      nse     = 3
///      shape   = [2, 3]
///      values  = [1.0, 5.0, 3.0]
///      indices = [ 0, 0,  1, 1,  1, 2]
//
// TODO: generalize beyond 64-bit indices.
//
void *convertToMLIRSparseTensorF64(uint64_t rank, uint64_t nse, uint64_t *shape,
                                   double *values, uint64_t *indices,
                                   uint64_t *perm, uint8_t *sparse) {
  return toMLIRSparseTensor<double>(rank, nse, shape, values, indices, perm,
                                    sparse);
}
void *convertToMLIRSparseTensorF32(uint64_t rank, uint64_t nse, uint64_t *shape,
                                   float *values, uint64_t *indices,
                                   uint64_t *perm, uint8_t *sparse) {
  return toMLIRSparseTensor<float>(rank, nse, shape, values, indices, perm,
                                   sparse);
}

/// Converts a sparse tensor to COO-flavored format expressed using C-style
/// data structures. The expected output parameters are pointers for these
/// values:
///
///   rank:    rank of tensor
///   nse:     number of specified elements (usually the nonzeros)
///   shape:   array with dimension size for each rank
///   values:  a "nse" array with values for all specified elements
///   indices: a flat "nse x rank" array with indices for all specified elements
///
/// The input is a pointer to SparseTensorStorage<P, I, V>, typically returned
/// from convertToMLIRSparseTensor.
///
//  TODO: Currently, values are copied from SparseTensorStorage to
//  SparseTensorCOO, then to the output. We may want to reduce the number of
//  copies.
//
// TODO: generalize beyond 64-bit indices, no dim ordering, all dimensions
// compressed
//
void convertFromMLIRSparseTensorF64(void *tensor, uint64_t *pRank,
                                    uint64_t *pNse, uint64_t **pShape,
                                    double **pValues, uint64_t **pIndices) {
  fromMLIRSparseTensor<double>(tensor, pRank, pNse, pShape, pValues, pIndices);
}
void convertFromMLIRSparseTensorF32(void *tensor, uint64_t *pRank,
                                    uint64_t *pNse, uint64_t **pShape,
                                    float **pValues, uint64_t **pIndices) {
  fromMLIRSparseTensor<float>(tensor, pRank, pNse, pShape, pValues, pIndices);
}

} // extern "C"

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
