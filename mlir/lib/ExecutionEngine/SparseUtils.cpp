//===- SparseUtils.cpp - Sparse Utils for MLIR execution ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a light-weight runtime library that is useful for
// sparse tensor manipulations. The functionality provided in this library
// is meant to simplify benchmarking, testing, and debugging MLIR code that
// operates on sparse tensors. The provided functionality is **not** part
// of core MLIR, however.
//
//===----------------------------------------------------------------------===//

#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

//===----------------------------------------------------------------------===//
//
// Internal support for reading matrices in the Matrix Market Exchange Format.
// See https://math.nist.gov/MatrixMarket for details on this format.
//
//===----------------------------------------------------------------------===//

// Helper to convert string to lower case.
static char *toLower(char *token) {
  for (char *c = token; *c; c++)
    *c = tolower(*c);
  return token;
}

// Read the header of a general sparse matrix of type real.
//
// TODO: support other formats as well?
//
static void readHeader(FILE *file, char *name, uint64_t *m, uint64_t *n,
                       uint64_t *nnz) {
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
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "%" PRIu64, m, n, nnz) != 3) {
    fprintf(stderr, "Cannot find size in %s\n", name);
    exit(1);
  }
}

// Read next data item.
static void readItem(FILE *file, char *name, uint64_t *i, uint64_t *j,
                     double *d) {
  if (fscanf(file, "%" PRIu64 " %" PRIu64 " %lg\n", i, j, d) != 3) {
    fprintf(stderr, "Cannot find next data item in %s\n", name);
    exit(1);
  }
  // Translate 1-based to 0-based.
  *i = *i - 1;
  *j = *j - 1;
}

//===----------------------------------------------------------------------===//
//
// Public API of the sparse runtime library.
//
// Enables MLIR code to read a matrix in Matrix Market Exchange Format
// as follows:
//
//   call @openMatrix("A.mtx", %m, %n, %nnz) : (!llvm.ptr<i8>,
//                                              memref<index>,
//                                              memref<index>,
//                                              memref<index>) -> ()
//   .... prepare reading in m x n matrix A with nnz nonzero elements ....
//   %u = load %nnz[] : memref<index>
//   scf.for %k = %c0 to %u step %c1 {
//     call @readMatrixItem(%i, %j, %d) : (memref<index>,
//                                         memref<index>, memref<f64>) -> ()
//     .... process next nonzero element A[i][j] = d ....
//   }
//   call @closeMatrix() : () -> ()
//
// The implementation is *not* thread-safe. Also, only *one* matrix file can
// be open at the time. A matrix file must be closed before reading in a next.
//
// Note that input parameters mimic the layout of a MemRef<T>:
//   struct MemRef {
//     T *base;
//     T *data;
//     int64_t off;
//   }
//===----------------------------------------------------------------------===//

// Currently open matrix. This is *not* thread-safe or re-entrant.
static FILE *sparseFile = nullptr;
static char *sparseFilename = nullptr;

extern "C" void openMatrix(char *filename, uint64_t *mbase, uint64_t *mdata,
                           int64_t moff, uint64_t *nbase, uint64_t *ndata,
                           int64_t noff, uint64_t *nnzbase, uint64_t *nnzdata,
                           int64_t nnzoff) {
  if (sparseFile != nullptr) {
    fprintf(stderr, "Other file still open %s vs. %s\n", sparseFilename,
            filename);
    exit(1);
  }
  sparseFile = fopen(filename, "r");
  if (!sparseFile) {
    fprintf(stderr, "Cannot find %s\n", filename);
    exit(1);
  }
  sparseFilename = filename;
  readHeader(sparseFile, filename, mdata, ndata, nnzdata);
}

extern "C" void readMatrixItem(uint64_t *ibase, uint64_t *idata, int64_t ioff,
                               uint64_t *jbase, uint64_t *jdata, int64_t joff,
                               double *dbase, double *ddata, int64_t doff) {
  if (sparseFile == nullptr) {
    fprintf(stderr, "Cannot read item from unopened matrix\n");
    exit(1);
  }
  readItem(sparseFile, sparseFilename, idata, jdata, ddata);
}

extern "C" void closeMatrix() {
  if (sparseFile == nullptr) {
    fprintf(stderr, "Cannot close unopened matrix\n");
    exit(1);
  }
  fclose(sparseFile);
  sparseFile = nullptr;
  sparseFilename = nullptr;
}

// Helper method to read sparse matrix filenames from the environment, defined
// with the naming convention ${SPARSE_MATRIX0}, ${SPARSE_MATRIX1}, etc.
extern "C" char *getSparseMatrix(uint64_t id) {
  char var[80];
  sprintf(var, "SPARSE_MATRIX%lu", id);
  char *env = getenv(var);
  return env;
}
