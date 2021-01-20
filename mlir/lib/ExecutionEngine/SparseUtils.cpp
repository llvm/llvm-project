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
// Internal support for reading sparse tensors in one of the following
// external file formats:
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
  // Add element as indices and value.
  void add(const std::vector<uint64_t> &ind, double val) {
    assert(sizes.size() == ind.size());
    for (int64_t r = 0, rank = sizes.size(); r < rank; r++)
      assert(ind[r] < sizes[r]); // within bounds
    elements.emplace_back(Element(ind, val));
  }
  // Sort elements lexicographically by index.
  void sort() { std::sort(elements.begin(), elements.end(), lexOrder); }
  // Primitive one-time iteration.
  const Element &next() { return elements[pos++]; }

private:
  // Returns true if indices of e1 < indices of e2.
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
// the data layout of a MemRef<?xT>:
//
//   struct MemRef {
//     T *base;
//     T *data;
//     int64_t off;
//     int64_t sizes[1];
//     int64_t strides[1];
//   }
//
//===----------------------------------------------------------------------===//

/// Reads in a sparse tensor with the given filename. The call yields a
/// pointer to an opaque memory-resident sparse tensor object that is only
/// understood by other methods in the sparse runtime support library. An
/// array parameter is used to pass the rank, the number of nonzero elements,
/// and the dimension sizes (one per rank).
extern "C" void *openTensorC(char *filename, uint64_t *idata) {
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
extern "C" void *openTensor(char *filename, uint64_t *ibase, uint64_t *idata,
                            uint64_t ioff, uint64_t isize, uint64_t istride) {
  assert(istride == 1);
  return openTensorC(filename, idata + ioff);
}

/// Yields the next element from the given opaque sparse tensor object.
extern "C" void readTensorItemC(void *tensor, uint64_t *idata, double *ddata) {
  const Element &e = static_cast<SparseTensor *>(tensor)->next();
  for (uint64_t r = 0, rank = e.indices.size(); r < rank; r++)
    idata[r] = e.indices[r];
  ddata[0] = e.value;
}

/// "MLIRized" version.
extern "C" void readTensorItem(void *tensor, uint64_t *ibase, uint64_t *idata,
                               uint64_t ioff, uint64_t isize, uint64_t istride,
                               double *dbase, double *ddata, uint64_t doff,
                               uint64_t dsize, uint64_t dstride) {
  assert(istride == 1 && dstride == 1);
  readTensorItemC(tensor, idata + ioff, ddata + doff);
}

/// Closes the given opaque sparse tensor object, releasing its memory
/// resources. After this call, the opague object cannot be used anymore.
extern "C" void closeTensor(void *tensor) {
  delete static_cast<SparseTensor *>(tensor);
}

/// Helper method to read a sparse tensor filename from the environment,
/// defined with the naming convention ${TENSOR0}, ${TENSOR1}, etc.
extern "C" char *getTensorFilename(uint64_t id) {
  char var[80];
  sprintf(var, "TENSOR%" PRIu64, id);
  char *env = getenv(var);
  return env;
}

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
