//===- SparseUtils.cpp - Sparse Utils for MLIR execution ------------------===//
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

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
      : sizes(szs) {
    if (capacity)
      elements.reserve(capacity);
  }
  /// Adds element as indices and value.
  void add(const std::vector<uint64_t> &ind, V val) {
    uint64_t rank = getRank();
    assert(rank == ind.size());
    for (uint64_t r = 0; r < rank; r++)
      assert(ind[r] < sizes[r]); // within bounds
    elements.emplace_back(ind, val);
  }
  /// Sorts elements lexicographically by index.
  void sort() { std::sort(elements.begin(), elements.end(), lexOrder); }
  /// Returns rank.
  uint64_t getRank() const { return sizes.size(); }
  /// Getter for sizes array.
  const std::vector<uint64_t> &getSizes() const { return sizes; }
  /// Getter for elements array.
  const std::vector<Element<V>> &getElements() const { return elements; }

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
  std::vector<uint64_t> sizes; // per-dimension sizes
  std::vector<Element<V>> elements;
};

/// Abstract base class of sparse tensor storage. Note that we use
/// function overloading to implement "partial" method specialization.
class SparseTensorStorageBase {
public:
  enum DimLevelType : uint8_t { kDense = 0, kCompressed = 1, kSingleton = 2 };

  virtual uint64_t getDimSize(uint64_t) = 0;

  // Overhead storage.
  virtual void getPointers(std::vector<uint64_t> **, uint64_t) { fatal("p64"); }
  virtual void getPointers(std::vector<uint32_t> **, uint64_t) { fatal("p32"); }
  virtual void getPointers(std::vector<uint16_t> **, uint64_t) { fatal("p16"); }
  virtual void getPointers(std::vector<uint8_t> **, uint64_t) { fatal("p8"); }
  virtual void getIndices(std::vector<uint64_t> **, uint64_t) { fatal("i64"); }
  virtual void getIndices(std::vector<uint32_t> **, uint64_t) { fatal("i32"); }
  virtual void getIndices(std::vector<uint16_t> **, uint64_t) { fatal("i16"); }
  virtual void getIndices(std::vector<uint8_t> **, uint64_t) { fatal("i8"); }

  // Primary storage.
  virtual void getValues(std::vector<double> **) { fatal("valf64"); }
  virtual void getValues(std::vector<float> **) { fatal("valf32"); }
  virtual void getValues(std::vector<int64_t> **) { fatal("vali64"); }
  virtual void getValues(std::vector<int32_t> **) { fatal("vali32"); }
  virtual void getValues(std::vector<int16_t> **) { fatal("vali16"); }
  virtual void getValues(std::vector<int8_t> **) { fatal("vali8"); }

  virtual ~SparseTensorStorageBase() {}

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
                      const uint8_t *sparsity, SparseTensorCOO<V> *tensor)
      : sizes(szs), rev(getRank()), pointers(getRank()), indices(getRank()) {
    uint64_t rank = getRank();
    // Store "reverse" permutation.
    for (uint64_t r = 0; r < rank; r++)
      rev[perm[r]] = r;
    // Provide hints on capacity of pointers and indices.
    // TODO: needs fine-tuning based on sparsity
    for (uint64_t r = 0, s = 1; r < rank; r++) {
      s *= sizes[r];
      if (sparsity[r] == kCompressed) {
        pointers[r].reserve(s + 1);
        indices[r].reserve(s);
        s = 1;
      } else {
        assert(sparsity[r] == kDense && "singleton not yet supported");
      }
    }
    // Prepare sparse pointer structures for all dimensions.
    for (uint64_t r = 0; r < rank; r++)
      if (sparsity[r] == kCompressed)
        pointers[r].push_back(0);
    // Then assign contents from coordinate scheme tensor if provided.
    if (tensor) {
      uint64_t nnz = tensor->getElements().size();
      values.reserve(nnz);
      fromCOO(tensor, sparsity, 0, nnz, 0);
    }
  }

  virtual ~SparseTensorStorage() {}

  /// Get the rank of the tensor.
  uint64_t getRank() const { return sizes.size(); }

  /// Get the size in the given dimension of the tensor.
  uint64_t getDimSize(uint64_t d) override {
    assert(d < getRank());
    return sizes[d];
  }

  // Partially specialize these three methods based on template types.
  void getPointers(std::vector<P> **out, uint64_t d) override {
    assert(d < getRank());
    *out = &pointers[d];
  }
  void getIndices(std::vector<I> **out, uint64_t d) override {
    assert(d < getRank());
    *out = &indices[d];
  }
  void getValues(std::vector<V> **out) override { *out = &values; }

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
    std::vector<uint64_t> idx(rank);
    toCOO(tensor, reord, idx, 0, 0);
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
                  const uint8_t *sparsity, SparseTensorCOO<V> *tensor) {
    SparseTensorStorage<P, I, V> *n = nullptr;
    if (tensor) {
      assert(tensor->getRank() == rank);
      for (uint64_t r = 0; r < rank; r++)
        assert(tensor->getSizes()[perm[r]] == sizes[r] || sizes[r] == 0);
      tensor->sort(); // sort lexicographically
      n = new SparseTensorStorage<P, I, V>(tensor->getSizes(), perm, sparsity,
                                           tensor);
      delete tensor;
    } else {
      std::vector<uint64_t> permsz(rank);
      for (uint64_t r = 0; r < rank; r++)
        permsz[perm[r]] = sizes[r];
      n = new SparseTensorStorage<P, I, V>(permsz, perm, sparsity, tensor);
    }
    return n;
  }

private:
  /// Initializes sparse tensor storage scheme from a memory-resident sparse
  /// tensor in coordinate scheme. This method prepares the pointers and
  /// indices arrays under the given per-dimension dense/sparse annotations.
  void fromCOO(SparseTensorCOO<V> *tensor, const uint8_t *sparsity, uint64_t lo,
               uint64_t hi, uint64_t d) {
    const std::vector<Element<V>> &elements = tensor->getElements();
    // Once dimensions are exhausted, insert the numerical values.
    if (d == getRank()) {
      assert(lo >= hi || lo < elements.size());
      values.push_back(lo < hi ? elements[lo].value : 0);
      return;
    }
    assert(d < getRank());
    // Visit all elements in this interval.
    uint64_t full = 0;
    while (lo < hi) {
      assert(lo < elements.size() && hi <= elements.size());
      // Find segment in interval with same index elements in this dimension.
      unsigned idx = elements[lo].indices[d];
      unsigned seg = lo + 1;
      while (seg < hi && elements[seg].indices[d] == idx)
        seg++;
      // Handle segment in interval for sparse or dense dimension.
      if (sparsity[d] == kCompressed) {
        indices[d].push_back(idx);
      } else {
        // For dense storage we must fill in all the zero values between
        // the previous element (when last we ran this for-loop) and the
        // current element.
        for (; full < idx; full++)
          fromCOO(tensor, sparsity, 0, 0, d + 1); // pass empty
        full++;
      }
      fromCOO(tensor, sparsity, lo, seg, d + 1);
      // And move on to next segment in interval.
      lo = seg;
    }
    // Finalize the sparse pointer structure at this dimension.
    if (sparsity[d] == kCompressed) {
      pointers[d].push_back(indices[d].size());
    } else {
      // For dense storage we must fill in all the zero values after
      // the last element.
      for (uint64_t sz = sizes[d]; full < sz; full++)
        fromCOO(tensor, sparsity, 0, 0, d + 1); // pass empty
    }
  }

  /// Stores the sparse tensor storage scheme into a memory-resident sparse
  /// tensor in coordinate scheme.
  void toCOO(SparseTensorCOO<V> *tensor, std::vector<uint64_t> &reord,
             std::vector<uint64_t> &idx, uint64_t pos, uint64_t d) {
    assert(d <= getRank());
    if (d == getRank()) {
      assert(pos < values.size());
      tensor->add(idx, values[pos]);
    } else if (pointers[d].empty()) {
      // Dense dimension.
      for (uint64_t i = 0, sz = sizes[d], off = pos * sz; i < sz; i++) {
        idx[reord[d]] = i;
        toCOO(tensor, reord, idx, off + i, d + 1);
      }
    } else {
      // Sparse dimension.
      for (uint64_t ii = pointers[d][pos]; ii < pointers[d][pos + 1]; ii++) {
        idx[reord[d]] = indices[d][ii];
        toCOO(tensor, reord, idx, ii, d + 1);
      }
    }
  }

private:
  std::vector<uint64_t> sizes; // per-dimension sizes
  std::vector<uint64_t> rev;   // "reverse" permutation
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
static void readMMEHeader(FILE *file, char *name, uint64_t *idata) {
  char line[1025];
  char header[64];
  char object[64];
  char format[64];
  char field[64];
  char symmetry[64];
  // Read header line.
  if (fscanf(file, "%63s %63s %63s %63s %63s\n", header, object, format, field,
             symmetry) != 5) {
    fprintf(stderr, "Corrupt header in %s\n", name);
    exit(1);
  }
  // Make sure this is a general sparse matrix.
  if (strcmp(toLower(header), "%%matrixmarket") ||
      strcmp(toLower(object), "matrix") ||
      strcmp(toLower(format), "coordinate") || strcmp(toLower(field), "real") ||
      strcmp(toLower(symmetry), "general")) {
    fprintf(stderr,
            "Cannot find a general sparse matrix with type real in %s\n", name);
    exit(1);
  }
  // Skip comments.
  while (1) {
    if (!fgets(line, 1025, file)) {
      fprintf(stderr, "Cannot find data in %s\n", name);
      exit(1);
    }
    if (line[0] != '%')
      break;
  }
  // Next line contains M N NNZ.
  idata[0] = 2; // rank
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "%" PRIu64 "\n", idata + 2, idata + 3,
             idata + 1) != 3) {
    fprintf(stderr, "Cannot find size in %s\n", name);
    exit(1);
  }
}

/// Read the "extended" FROSTT header. Although not part of the documented
/// format, we assume that the file starts with optional comments followed
/// by two lines that define the rank, the number of nonzeros, and the
/// dimensions sizes (one per rank) of the sparse tensor.
static void readExtFROSTTHeader(FILE *file, char *name, uint64_t *idata) {
  char line[1025];
  // Skip comments.
  while (1) {
    if (!fgets(line, 1025, file)) {
      fprintf(stderr, "Cannot find data in %s\n", name);
      exit(1);
    }
    if (line[0] != '#')
      break;
  }
  // Next line contains RANK and NNZ.
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "\n", idata, idata + 1) != 2) {
    fprintf(stderr, "Cannot find metadata in %s\n", name);
    exit(1);
  }
  // Followed by a line with the dimension sizes (one per rank).
  for (uint64_t r = 0; r < idata[0]; r++) {
    if (fscanf(file, "%" PRIu64, idata + 2 + r) != 1) {
      fprintf(stderr, "Cannot find dimension size %s\n", name);
      exit(1);
    }
  }
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
  uint64_t idata[512];
  if (strstr(filename, ".mtx")) {
    readMMEHeader(file, filename, idata);
  } else if (strstr(filename, ".tns")) {
    readExtFROSTTHeader(file, filename, idata);
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
    uint64_t idx = -1;
    for (uint64_t r = 0; r < rank; r++) {
      if (fscanf(file, "%" PRIu64, &idx) != 1) {
        fprintf(stderr, "Cannot find next index in %s\n", filename);
        exit(1);
      }
      // Add 0-based index.
      indices[perm[r]] = idx - 1;
    }
    // The external formats always store the numerical values with the type
    // double, but we cast these values to the sparse tensor object type.
    double value;
    if (fscanf(file, "%lg\n", &value) != 1) {
      fprintf(stderr, "Cannot find next value in %s\n", filename);
      exit(1);
    }
    tensor->add(indices, value);
  }
  // Close the file and return tensor.
  fclose(file);
  return tensor;
}

} // anonymous namespace

extern "C" {

/// Helper method to read a sparse tensor filename from the environment,
/// defined with the naming convention ${TENSOR0}, ${TENSOR1}, etc.
char *getTensorFilename(uint64_t id) {
  char var[80];
  sprintf(var, "TENSOR%" PRIu64, id);
  char *env = getenv(var);
  return env;
}

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

enum OverheadTypeEnum : uint64_t { kU64 = 1, kU32 = 2, kU16 = 3, kU8 = 4 };

enum PrimaryTypeEnum : uint64_t {
  kF64 = 1,
  kF32 = 2,
  kI64 = 3,
  kI32 = 4,
  kI16 = 5,
  kI8 = 6
};

enum Action : uint32_t {
  kEmpty = 0,
  kFromFile = 1,
  kFromCOO = 2,
  kEmptyCOO = 3,
  kToCOO = 4
};

#define CASE(p, i, v, P, I, V)                                                 \
  if (ptrTp == (p) && indTp == (i) && valTp == (v)) {                          \
    SparseTensorCOO<V> *tensor = nullptr;                                      \
    if (action == kFromFile)                                                   \
      tensor =                                                                 \
          openSparseTensorCOO<V>(static_cast<char *>(ptr), rank, sizes, perm); \
    else if (action == kFromCOO)                                               \
      tensor = static_cast<SparseTensorCOO<V> *>(ptr);                         \
    else if (action == kEmptyCOO)                                              \
      return SparseTensorCOO<V>::newSparseTensorCOO(rank, sizes, perm);        \
    else if (action == kToCOO)                                                 \
      return static_cast<SparseTensorStorage<P, I, V> *>(ptr)->toCOO(perm);    \
    else                                                                       \
      assert(action == kEmpty);                                                \
    return SparseTensorStorage<P, I, V>::newSparseTensor(rank, sizes, perm,    \
                                                         sparsity, tensor);    \
  }

#define IMPL1(NAME, TYPE, LIB)                                                 \
  void _mlir_ciface_##NAME(StridedMemRefType<TYPE, 1> *ref, void *tensor) {    \
    assert(ref);                                                               \
    assert(tensor);                                                            \
    std::vector<TYPE> *v;                                                      \
    static_cast<SparseTensorStorageBase *>(tensor)->LIB(&v);                   \
    ref->basePtr = ref->data = v->data();                                      \
    ref->offset = 0;                                                           \
    ref->sizes[0] = v->size();                                                 \
    ref->strides[0] = 1;                                                       \
  }

#define IMPL2(NAME, TYPE, LIB)                                                 \
  void _mlir_ciface_##NAME(StridedMemRefType<TYPE, 1> *ref, void *tensor,      \
                           uint64_t d) {                                       \
    assert(ref);                                                               \
    assert(tensor);                                                            \
    std::vector<TYPE> *v;                                                      \
    static_cast<SparseTensorStorageBase *>(tensor)->LIB(&v, d);                \
    ref->basePtr = ref->data = v->data();                                      \
    ref->offset = 0;                                                           \
    ref->sizes[0] = v->size();                                                 \
    ref->strides[0] = 1;                                                       \
  }

#define IMPL3(NAME, TYPE)                                                      \
  void *_mlir_ciface_##NAME(void *tensor, TYPE value,                          \
                            StridedMemRefType<uint64_t, 1> *iref,              \
                            StridedMemRefType<uint64_t, 1> *pref) {            \
    assert(tensor);                                                            \
    assert(iref);                                                              \
    assert(pref);                                                              \
    assert(iref->strides[0] == 1 && pref->strides[0] == 1);                    \
    assert(iref->sizes[0] == pref->sizes[0]);                                  \
    const uint64_t *indx = iref->data + iref->offset;                          \
    const uint64_t *perm = pref->data + pref->offset;                          \
    uint64_t isize = iref->sizes[0];                                           \
    std::vector<uint64_t> indices(isize);                                      \
    for (uint64_t r = 0; r < isize; r++)                                       \
      indices[perm[r]] = indx[r];                                              \
    static_cast<SparseTensorCOO<TYPE> *>(tensor)->add(indices, value);         \
    return tensor;                                                             \
  }

/// Constructs a new sparse tensor. This is the "swiss army knife"
/// method for materializing sparse tensors into the computation.
///
/// action:
/// kEmpty = returns empty storage to fill later
/// kFromFile = returns storage, where ptr contains filename to read
/// kFromCOO = returns storage, where ptr contains coordinate scheme to assign
/// kEmptyCOO = returns empty coordinate scheme to fill and use with kFromCOO
/// kToCOO = returns coordinate scheme from storage in ptr to use with kFromCOO
void *
_mlir_ciface_newSparseTensor(StridedMemRefType<uint8_t, 1> *aref, // NOLINT
                             StridedMemRefType<uint64_t, 1> *sref,
                             StridedMemRefType<uint64_t, 1> *pref,
                             uint64_t ptrTp, uint64_t indTp, uint64_t valTp,
                             uint32_t action, void *ptr) {
  assert(aref && sref && pref);
  assert(aref->strides[0] == 1 && sref->strides[0] == 1 &&
         pref->strides[0] == 1);
  assert(aref->sizes[0] == sref->sizes[0] && sref->sizes[0] == pref->sizes[0]);
  const uint8_t *sparsity = aref->data + aref->offset;
  const uint64_t *sizes = sref->data + sref->offset;
  const uint64_t *perm = pref->data + pref->offset;
  uint64_t rank = aref->sizes[0];

  // Double matrices with all combinations of overhead storage.
  CASE(kU64, kU64, kF64, uint64_t, uint64_t, double);
  CASE(kU64, kU32, kF64, uint64_t, uint32_t, double);
  CASE(kU64, kU16, kF64, uint64_t, uint16_t, double);
  CASE(kU64, kU8, kF64, uint64_t, uint8_t, double);
  CASE(kU32, kU64, kF64, uint32_t, uint64_t, double);
  CASE(kU32, kU32, kF64, uint32_t, uint32_t, double);
  CASE(kU32, kU16, kF64, uint32_t, uint16_t, double);
  CASE(kU32, kU8, kF64, uint32_t, uint8_t, double);
  CASE(kU16, kU64, kF64, uint16_t, uint64_t, double);
  CASE(kU16, kU32, kF64, uint16_t, uint32_t, double);
  CASE(kU16, kU16, kF64, uint16_t, uint16_t, double);
  CASE(kU16, kU8, kF64, uint16_t, uint8_t, double);
  CASE(kU8, kU64, kF64, uint8_t, uint64_t, double);
  CASE(kU8, kU32, kF64, uint8_t, uint32_t, double);
  CASE(kU8, kU16, kF64, uint8_t, uint16_t, double);
  CASE(kU8, kU8, kF64, uint8_t, uint8_t, double);

  // Float matrices with all combinations of overhead storage.
  CASE(kU64, kU64, kF32, uint64_t, uint64_t, float);
  CASE(kU64, kU32, kF32, uint64_t, uint32_t, float);
  CASE(kU64, kU16, kF32, uint64_t, uint16_t, float);
  CASE(kU64, kU8, kF32, uint64_t, uint8_t, float);
  CASE(kU32, kU64, kF32, uint32_t, uint64_t, float);
  CASE(kU32, kU32, kF32, uint32_t, uint32_t, float);
  CASE(kU32, kU16, kF32, uint32_t, uint16_t, float);
  CASE(kU32, kU8, kF32, uint32_t, uint8_t, float);
  CASE(kU16, kU64, kF32, uint16_t, uint64_t, float);
  CASE(kU16, kU32, kF32, uint16_t, uint32_t, float);
  CASE(kU16, kU16, kF32, uint16_t, uint16_t, float);
  CASE(kU16, kU8, kF32, uint16_t, uint8_t, float);
  CASE(kU8, kU64, kF32, uint8_t, uint64_t, float);
  CASE(kU8, kU32, kF32, uint8_t, uint32_t, float);
  CASE(kU8, kU16, kF32, uint8_t, uint16_t, float);
  CASE(kU8, kU8, kF32, uint8_t, uint8_t, float);

  // Integral matrices with same overhead storage.
  CASE(kU64, kU64, kI64, uint64_t, uint64_t, int64_t);
  CASE(kU64, kU64, kI32, uint64_t, uint64_t, int32_t);
  CASE(kU64, kU64, kI16, uint64_t, uint64_t, int16_t);
  CASE(kU64, kU64, kI8, uint64_t, uint64_t, int8_t);
  CASE(kU32, kU32, kI32, uint32_t, uint32_t, int32_t);
  CASE(kU32, kU32, kI16, uint32_t, uint32_t, int16_t);
  CASE(kU32, kU32, kI8, uint32_t, uint32_t, int8_t);
  CASE(kU16, kU16, kI32, uint16_t, uint16_t, int32_t);
  CASE(kU16, kU16, kI16, uint16_t, uint16_t, int16_t);
  CASE(kU16, kU16, kI8, uint16_t, uint16_t, int8_t);
  CASE(kU8, kU8, kI32, uint8_t, uint8_t, int32_t);
  CASE(kU8, kU8, kI16, uint8_t, uint8_t, int16_t);
  CASE(kU8, kU8, kI8, uint8_t, uint8_t, int8_t);

  // Unsupported case (add above if needed).
  fputs("unsupported combination of types\n", stderr);
  exit(1);
}

/// Methods that provide direct access to pointers, indices, and values.
IMPL2(sparsePointers, uint64_t, getPointers)
IMPL2(sparsePointers64, uint64_t, getPointers)
IMPL2(sparsePointers32, uint32_t, getPointers)
IMPL2(sparsePointers16, uint16_t, getPointers)
IMPL2(sparsePointers8, uint8_t, getPointers)
IMPL2(sparseIndices, uint64_t, getIndices)
IMPL2(sparseIndices64, uint64_t, getIndices)
IMPL2(sparseIndices32, uint32_t, getIndices)
IMPL2(sparseIndices16, uint16_t, getIndices)
IMPL2(sparseIndices8, uint8_t, getIndices)
IMPL1(sparseValuesF64, double, getValues)
IMPL1(sparseValuesF32, float, getValues)
IMPL1(sparseValuesI64, int64_t, getValues)
IMPL1(sparseValuesI32, int32_t, getValues)
IMPL1(sparseValuesI16, int16_t, getValues)
IMPL1(sparseValuesI8, int8_t, getValues)

/// Helper to add value to coordinate scheme, one per value type.
IMPL3(addEltF64, double)
IMPL3(addEltF32, float)
IMPL3(addEltI64, int64_t)
IMPL3(addEltI32, int32_t)
IMPL3(addEltI16, int16_t)
IMPL3(addEltI8, int8_t)

#undef CASE
#undef IMPL1
#undef IMPL2
#undef IMPL3

//===----------------------------------------------------------------------===//
//
// Public API with methods that accept C-style data structures to interact
// with sparse tensors, which are only visible as opaque pointers externally.
// These methods can be used both by MLIR compiler-generated code as well as by
// an external runtime that wants to interact with MLIR compiler-generated code.
//
//===----------------------------------------------------------------------===//

/// Returns size of sparse tensor in given dimension.
uint64_t sparseDimSize(void *tensor, uint64_t d) {
  return static_cast<SparseTensorStorageBase *>(tensor)->getDimSize(d);
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
// TODO: for now f64 tensors only, no dim ordering, all dimensions compressed
//
void *convertToMLIRSparseTensor(uint64_t rank, uint64_t nse, uint64_t *shape,
                                double *values, uint64_t *indices) {
  // Setup all-dims compressed and default ordering.
  std::vector<uint8_t> sparse(rank, SparseTensorStorageBase::kCompressed);
  std::vector<uint64_t> perm(rank);
  std::iota(perm.begin(), perm.end(), 0);
  // Convert external format to internal COO.
  SparseTensorCOO<double> *tensor = SparseTensorCOO<double>::newSparseTensorCOO(
      rank, shape, perm.data(), nse);
  std::vector<uint64_t> idx(rank);
  for (uint64_t i = 0, base = 0; i < nse; i++) {
    for (uint64_t r = 0; r < rank; r++)
      idx[r] = indices[base + r];
    tensor->add(idx, values[i]);
    base += rank;
  }
  // Return sparse tensor storage format as opaque pointer.
  return SparseTensorStorage<uint64_t, uint64_t, double>::newSparseTensor(
      rank, shape, perm.data(), sparse.data(), tensor);
}

} // extern "C"

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
