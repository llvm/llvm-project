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
#include <vector>

//===----------------------------------------------------------------------===//
//
// Internal support for storing and reading sparse tensors.
//
// The following memory-resident sparse storage schemes are supported:
//
// (a) A coordinate scheme for temporarily storing and lexicographically
//     sorting a sparse tensor by index.
//
// (b) A "one-size-fits-all" sparse storage scheme defined by per-rank
//     sparse/dense annnotations to be used by generated MLIR code.
//
// The following external formats are supported:
//
// (1) Matrix Market Exchange (MME): *.mtx
//     https://math.nist.gov/MatrixMarket/formats.html
//
// (2) Formidable Repository of Open Sparse Tensors and Tools (FROSTT): *.tns
//     http://frostt.io/tensors/file-formats.html
//
//===----------------------------------------------------------------------===//

namespace {

/// A sparse tensor element in coordinate scheme (value and indices).
/// For example, a rank-1 vector element would look like
///   ({i}, a[i])
/// and a rank-5 tensor element like
///   ({i,j,k,l,m}, a[i,j,k,l,m])
struct Element {
  Element(const std::vector<uint64_t> &ind, double val)
      : indices(ind), value(val){};
  std::vector<uint64_t> indices;
  double value;
};

/// A memory-resident sparse tensor in coordinate scheme (collection of
/// elements). This data structure is used to read a sparse tensor from
/// external file format into memory and sort the elements lexicographically
/// by indices before passing it back to the client (most packed storage
/// formats require the elements to appear in lexicographic index order).
struct SparseTensor {
public:
  SparseTensor(const std::vector<uint64_t> &szs, uint64_t capacity)
      : sizes(szs), pos(0) {
    elements.reserve(capacity);
  }
  /// Adds element as indices and value.
  void add(const std::vector<uint64_t> &ind, double val) {
    assert(getRank() == ind.size());
    for (int64_t r = 0, rank = getRank(); r < rank; r++)
      assert(ind[r] < sizes[r]); // within bounds
    elements.emplace_back(Element(ind, val));
  }
  /// Sorts elements lexicographically by index.
  void sort() { std::sort(elements.begin(), elements.end(), lexOrder); }
  /// Primitive one-time iteration.
  const Element &next() { return elements[pos++]; }
  /// Returns rank.
  uint64_t getRank() const { return sizes.size(); }
  /// Getter for sizes array.
  const std::vector<uint64_t> &getSizes() const { return sizes; }
  /// Getter for elements array.
  const std::vector<Element> &getElements() const { return elements; }

private:
  /// Returns true if indices of e1 < indices of e2.
  static bool lexOrder(const Element &e1, const Element &e2) {
    assert(e1.indices.size() == e2.indices.size());
    for (int64_t r = 0, rank = e1.indices.size(); r < rank; r++) {
      if (e1.indices[r] == e2.indices[r])
        continue;
      return e1.indices[r] < e2.indices[r];
    }
    return false;
  }
  std::vector<uint64_t> sizes; // per-rank dimension sizes
  std::vector<Element> elements;
  uint64_t pos;
};

/// Abstract base class of sparse tensor storage. Note that we use
/// function overloading to implement "partial" method specialization.
class SparseTensorStorageBase {
public:
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

  virtual ~SparseTensorStorageBase() {}

private:
  void fatal(const char *tp) {
    fprintf(stderr, "unsupported %s\n", tp);
    exit(1);
  }
};

/// A memory-resident sparse tensor using a storage scheme based on per-rank
/// annotations on dense/sparse. This data structure provides a bufferized
/// form of an imaginary SparseTensorType, until such a type becomes a
/// first-class citizen of MLIR. In contrast to generating setup methods for
/// each differently annotated sparse tensor, this method provides a convenient
/// "one-size-fits-all" solution that simply takes an input tensor and
/// annotations to implement all required setup in a general manner.
template <typename P, typename I, typename V>
class SparseTensorStorage : public SparseTensorStorageBase {
public:
  /// Constructs sparse tensor storage scheme following the given
  /// per-rank dimension dense/sparse annotations.
  SparseTensorStorage(SparseTensor *tensor, bool *sparsity)
      : sizes(tensor->getSizes()), pointers(getRank()), indices(getRank()) {
    // Provide hints on capacity.
    // TODO: needs fine-tuning based on sparsity
    uint64_t nnz = tensor->getElements().size();
    values.reserve(nnz);
    for (uint64_t d = 0, s = 1, rank = getRank(); d < rank; d++) {
      s *= sizes[d];
      if (sparsity[d]) {
        pointers[d].reserve(s + 1);
        indices[d].reserve(s);
        s = 1;
      }
    }
    // Then setup the tensor.
    traverse(tensor, sparsity, 0, nnz, 0);
  }

  virtual ~SparseTensorStorage() {}

  uint64_t getRank() const { return sizes.size(); }

  uint64_t getDimSize(uint64_t d) override { return sizes[d]; }

  // Partially specialize these three methods based on template types.
  void getPointers(std::vector<P> **out, uint64_t d) override {
    *out = &pointers[d];
  }
  void getIndices(std::vector<I> **out, uint64_t d) override {
    *out = &indices[d];
  }
  void getValues(std::vector<V> **out) override { *out = &values; }

private:
  /// Initializes sparse tensor storage scheme from a memory-resident
  /// representation of an external sparse tensor. This method prepares
  /// the pointers and indices arrays under the given per-rank dimension
  /// dense/sparse annotations.
  void traverse(SparseTensor *tensor, bool *sparsity, uint64_t lo, uint64_t hi,
                uint64_t d) {
    const std::vector<Element> &elements = tensor->getElements();
    // Once dimensions are exhausted, insert the numerical values.
    if (d == getRank()) {
      values.push_back(lo < hi ? elements[lo].value : 0.0);
      return;
    }
    // Prepare a sparse pointer structure at this dimension.
    if (sparsity[d] && pointers[d].empty())
      pointers[d].push_back(0);
    // Visit all elements in this interval.
    uint64_t full = 0;
    while (lo < hi) {
      // Find segment in interval with same index elements in this dimension.
      unsigned idx = elements[lo].indices[d];
      unsigned seg = lo + 1;
      while (seg < hi && elements[seg].indices[d] == idx)
        seg++;
      // Handle segment in interval for sparse or dense dimension.
      if (sparsity[d]) {
        indices[d].push_back(idx);
      } else {
        for (; full < idx; full++)
          traverse(tensor, sparsity, 0, 0, d + 1); // pass empty
        full++;
      }
      traverse(tensor, sparsity, lo, seg, d + 1);
      // And move on to next segment in interval.
      lo = seg;
    }
    // Finalize the sparse pointer structure at this dimension.
    if (sparsity[d]) {
      pointers[d].push_back(indices[d].size());
    } else {
      for (uint64_t sz = tensor->getSizes()[d]; full < sz; full++)
        traverse(tensor, sparsity, 0, 0, d + 1); // pass empty
    }
  }

private:
  std::vector<uint64_t> sizes; // per-rank dimension sizes
  std::vector<std::vector<P>> pointers;
  std::vector<std::vector<I>> indices;
  std::vector<V> values;
};

/// Templated reader.
template <typename P, typename I, typename V>
void *newSparseTensor(char *filename, bool *sparsity, uint64_t size) {
  uint64_t idata[64];
  SparseTensor *t = static_cast<SparseTensor *>(openTensorC(filename, idata));
  assert(size == t->getRank()); // sparsity array must match rank
  SparseTensorStorageBase *tensor =
      new SparseTensorStorage<P, I, V>(t, sparsity);
  delete t;
  return tensor;
}

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

} // anonymous namespace

//===----------------------------------------------------------------------===//
//
// Public API of the sparse runtime support library that enables MLIR code
// to read a sparse tensor from an external format (MME for FROSTT).
//
// For example, a sparse matrix in MME can be read as follows.
//
//   %tensor = call @openTensor(%fileName, %idata)
//     : (!llvm.ptr<i8>, memref<?xindex>) -> (!llvm.ptr<i8>)
//   %rank = load %idata[%c0] : memref<?xindex>    # always 2 for MME
//   %nnz  = load %idata[%c1] : memref<?xindex>
//   %m    = load %idata[%c2] : memref<?xindex>
//   %n    = load %idata[%c3] : memref<?xindex>
//   .. prepare reading in m x n sparse tensor A with nnz nonzero elements ..
//   scf.for %k = %c0 to %nnz step %c1 {
//     call @readTensorItem(%tensor, %idata, %ddata)
//       : (!llvm.ptr<i8>, memref<?xindex>, memref<?xf64>) -> ()
//     %i = load %idata[%c0] : memref<?xindex>
//     %j = load %idata[%c1] : memref<?xindex>
//     %d = load %ddata[%c0] : memref<?xf64>
//     .. process next nonzero element A[i][j] = d
//        where the elements appear in lexicographic order ..
//   }
//   call @closeTensor(%tensor) : (!llvm.ptr<i8>) -> ()
//
//
// Note that input parameters in the "MLIRized" version of a function mimic
// the data layout of a MemRef<?xT> (but cannot use a direct struct). The
// output parameter uses a direct struct.
//
//===----------------------------------------------------------------------===//

extern "C" {

/// Reads in a sparse tensor with the given filename. The call yields a
/// pointer to an opaque memory-resident sparse tensor object that is only
/// understood by other methods in the sparse runtime support library. An
/// array parameter is used to pass the rank, the number of nonzero elements,
/// and the dimension sizes (one per rank).
void *openTensorC(char *filename, uint64_t *idata) {
  // Open the file.
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Cannot find %s\n", filename);
    exit(1);
  }
  // Perform some file format dependent set up.
  if (strstr(filename, ".mtx")) {
    readMMEHeader(file, filename, idata);
  } else if (strstr(filename, ".tns")) {
    readExtFROSTTHeader(file, filename, idata);
  } else {
    fprintf(stderr, "Unknown format %s\n", filename);
    exit(1);
  }
  // Prepare sparse tensor object with per-rank dimension sizes
  // and the number of nonzeros as initial capacity.
  uint64_t rank = idata[0];
  uint64_t nnz = idata[1];
  std::vector<uint64_t> indices(rank);
  for (uint64_t r = 0; r < rank; r++)
    indices[r] = idata[2 + r];
  SparseTensor *tensor = new SparseTensor(indices, nnz);
  // Read all nonzero elements.
  for (uint64_t k = 0; k < nnz; k++) {
    for (uint64_t r = 0; r < rank; r++) {
      if (fscanf(file, "%" PRIu64, &indices[r]) != 1) {
        fprintf(stderr, "Cannot find next index in %s\n", filename);
        exit(1);
      }
      indices[r]--; // 0-based index
    }
    double value;
    if (fscanf(file, "%lg\n", &value) != 1) {
      fprintf(stderr, "Cannot find next value in %s\n", filename);
      exit(1);
    }
    tensor->add(indices, value);
  }
  // Close the file and return sorted tensor.
  fclose(file);
  tensor->sort(); // sort lexicographically
  return tensor;
}

/// "MLIRized" version.
void *openTensor(char *filename, uint64_t *ibase, uint64_t *idata,
                 uint64_t ioff, uint64_t isize, uint64_t istride) {
  assert(istride == 1);
  return openTensorC(filename, idata + ioff);
}

/// Yields the next element from the given opaque sparse tensor object.
void readTensorItemC(void *tensor, uint64_t *idata, double *ddata) {
  const Element &e = static_cast<SparseTensor *>(tensor)->next();
  for (uint64_t r = 0, rank = e.indices.size(); r < rank; r++)
    idata[r] = e.indices[r];
  ddata[0] = e.value;
}

/// "MLIRized" version.
void readTensorItem(void *tensor, uint64_t *ibase, uint64_t *idata,
                    uint64_t ioff, uint64_t isize, uint64_t istride,
                    double *dbase, double *ddata, uint64_t doff, uint64_t dsize,
                    uint64_t dstride) {
  assert(istride == 1 && dstride == 1);
  readTensorItemC(tensor, idata + ioff, ddata + doff);
}

/// Closes the given opaque sparse tensor object, releasing its memory
/// resources. After this call, the opaque object cannot be used anymore.
void closeTensor(void *tensor) { delete static_cast<SparseTensor *>(tensor); }

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
// Public API of the sparse runtime support library that support an opaque
// implementation of a bufferized SparseTensor in MLIR. This could be replaced
// by actual codegen in MLIR.
//
//===----------------------------------------------------------------------===//

// Cannot use templates with C linkage.

struct MemRef1DU64 {
  const uint64_t *base;
  const uint64_t *data;
  uint64_t off;
  uint64_t sizes[1];
  uint64_t strides[1];
};

struct MemRef1DU32 {
  const uint32_t *base;
  const uint32_t *data;
  uint64_t off;
  uint64_t sizes[1];
  uint64_t strides[1];
};

struct MemRef1DU16 {
  const uint16_t *base;
  const uint16_t *data;
  uint64_t off;
  uint64_t sizes[1];
  uint64_t strides[1];
};

struct MemRef1DU8 {
  const uint8_t *base;
  const uint8_t *data;
  uint64_t off;
  uint64_t sizes[1];
  uint64_t strides[1];
};

struct MemRef1DF64 {
  const double *base;
  const double *data;
  uint64_t off;
  uint64_t sizes[1];
  uint64_t strides[1];
};

struct MemRef1DF32 {
  const float *base;
  const float *data;
  uint64_t off;
  uint64_t sizes[1];
  uint64_t strides[1];
};

enum OverheadTypeEnum : uint64_t { kU64 = 1, kU32 = 2, kU16 = 3, kU8 = 4 };
enum PrimaryTypeEnum : uint64_t { kF64 = 1, kF32 = 2 };

#define CASE(p, i, v, P, I, V)                                                 \
  if (ptrTp == (p) && indTp == (i) && valTp == (v))                            \
  return newSparseTensor<P, I, V>(filename, sparsity, asize)

void *newSparseTensor(char *filename, bool *abase, bool *adata, uint64_t aoff,
                      uint64_t asize, uint64_t astride, uint64_t ptrTp,
                      uint64_t indTp, uint64_t valTp) {
  assert(astride == 1);
  bool *sparsity = abase + aoff;

  // The most common cases: 64-bit or 32-bit overhead, double/float values.
  CASE(kU64, kU64, kF64, uint64_t, uint64_t, double);
  CASE(kU64, kU64, kF32, uint64_t, uint64_t, float);
  CASE(kU64, kU32, kF64, uint64_t, uint32_t, double);
  CASE(kU64, kU32, kF32, uint64_t, uint32_t, float);
  CASE(kU32, kU64, kF64, uint32_t, uint64_t, double);
  CASE(kU32, kU64, kF32, uint32_t, uint64_t, float);
  CASE(kU32, kU32, kF64, uint32_t, uint32_t, double);
  CASE(kU32, kU32, kF32, uint32_t, uint32_t, float);

  // Some special cases: low overhead storage, double/float values.
  CASE(kU16, kU16, kF64, uint16_t, uint16_t, double);
  CASE(kU8, kU8, kF64, uint8_t, uint8_t, double);
  CASE(kU16, kU16, kF32, uint16_t, uint16_t, float);
  CASE(kU8, kU8, kF32, uint8_t, uint8_t, float);

  // Unsupported case (add above if needed).
  fputs("unsupported combination of types\n", stderr);
  exit(1);
}

#undef CASE

uint64_t sparseDimSize(void *tensor, uint64_t d) {
  return static_cast<SparseTensorStorageBase *>(tensor)->getDimSize(d);
}

MemRef1DU64 sparsePointers64(void *tensor, uint64_t d) {
  std::vector<uint64_t> *v;
  static_cast<SparseTensorStorageBase *>(tensor)->getPointers(&v, d);
  return {v->data(), v->data(), 0, {v->size()}, {1}};
}

MemRef1DU32 sparsePointers32(void *tensor, uint64_t d) {
  std::vector<uint32_t> *v;
  static_cast<SparseTensorStorageBase *>(tensor)->getPointers(&v, d);
  return {v->data(), v->data(), 0, {v->size()}, {1}};
}

MemRef1DU16 sparsePointers16(void *tensor, uint64_t d) {
  std::vector<uint16_t> *v;
  static_cast<SparseTensorStorageBase *>(tensor)->getPointers(&v, d);
  return {v->data(), v->data(), 0, {v->size()}, {1}};
}

MemRef1DU8 sparsePointers8(void *tensor, uint64_t d) {
  std::vector<uint8_t> *v;
  static_cast<SparseTensorStorageBase *>(tensor)->getPointers(&v, d);
  return {v->data(), v->data(), 0, {v->size()}, {1}};
}

MemRef1DU64 sparseIndices64(void *tensor, uint64_t d) {
  std::vector<uint64_t> *v;
  static_cast<SparseTensorStorageBase *>(tensor)->getIndices(&v, d);
  return {v->data(), v->data(), 0, {v->size()}, {1}};
}

MemRef1DU32 sparseIndices32(void *tensor, uint64_t d) {
  std::vector<uint32_t> *v;
  static_cast<SparseTensorStorageBase *>(tensor)->getIndices(&v, d);
  return {v->data(), v->data(), 0, {v->size()}, {1}};
}

MemRef1DU16 sparseIndices16(void *tensor, uint64_t d) {
  std::vector<uint16_t> *v;
  static_cast<SparseTensorStorageBase *>(tensor)->getIndices(&v, d);
  return {v->data(), v->data(), 0, {v->size()}, {1}};
}

MemRef1DU8 sparseIndices8(void *tensor, uint64_t d) {
  std::vector<uint8_t> *v;
  static_cast<SparseTensorStorageBase *>(tensor)->getIndices(&v, d);
  return {v->data(), v->data(), 0, {v->size()}, {1}};
}

MemRef1DF64 sparseValuesF64(void *tensor) {
  std::vector<double> *v;
  static_cast<SparseTensorStorageBase *>(tensor)->getValues(&v);
  return {v->data(), v->data(), 0, {v->size()}, {1}};
}

MemRef1DF32 sparseValuesF32(void *tensor) {
  std::vector<float> *v;
  static_cast<SparseTensorStorageBase *>(tensor)->getValues(&v);
  return {v->data(), v->data(), 0, {v->size()}, {1}};
}

void delSparseTensor(void *tensor) {
  delete static_cast<SparseTensorStorageBase *>(tensor);
}

} // extern "C"

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
