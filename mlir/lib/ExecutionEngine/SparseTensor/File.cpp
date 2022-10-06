//===- File.cpp - Parsing sparse tensors from files -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements parsing and printing of files in one of the
// following external formats:
//
// (1) Matrix Market Exchange (MME): *.mtx
//     https://math.nist.gov/MatrixMarket/formats.html
//
// (2) Formidable Repository of Open Sparse Tensors and Tools (FROSTT): *.tns
//     http://frostt.io/tensors/file-formats.html
//
// This file is part of the lightweight runtime support library for sparse
// tensor manipulations.  The functionality of the support library is meant
// to simplify benchmarking, testing, and debugging MLIR code operating on
// sparse tensors.  However, the provided functionality is **not** part of
// core MLIR itself.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/SparseTensor/File.h"

#include <cctype>
#include <cstring>

using namespace mlir::sparse_tensor;

/// Opens the file for reading.
void SparseTensorFile::openFile() {
  if (file)
    MLIR_SPARSETENSOR_FATAL("Already opened file %s\n", filename);
  file = fopen(filename, "r");
  if (!file)
    MLIR_SPARSETENSOR_FATAL("Cannot find file %s\n", filename);
}

/// Closes the file.
void SparseTensorFile::closeFile() {
  if (file) {
    fclose(file);
    file = nullptr;
  }
}

// TODO(wrengr/bixia): figure out how to reorganize the element-parsing
// loop of `openSparseTensorCOO` into methods of this class, so we can
// avoid leaking access to the `line` pointer (both for general hygiene
// and because we can't mark it const due to the second argument of
// `strtoul`/`strtoud` being `char * *restrict` rather than
// `char const* *restrict`).
//
/// Attempts to read a line from the file.
char *SparseTensorFile::readLine() {
  if (fgets(line, kColWidth, file))
    return line;
  MLIR_SPARSETENSOR_FATAL("Cannot read next line of %s\n", filename);
}

/// Reads and parses the file's header.
void SparseTensorFile::readHeader() {
  assert(file && "Attempt to readHeader() before openFile()");
  if (strstr(filename, ".mtx"))
    readMMEHeader();
  else if (strstr(filename, ".tns"))
    readExtFROSTTHeader();
  else
    MLIR_SPARSETENSOR_FATAL("Unknown format %s\n", filename);
  assert(isValid() && "Failed to read the header");
}

/// Asserts the shape subsumes the actual dimension sizes.  Is only
/// valid after parsing the header.
void SparseTensorFile::assertMatchesShape(uint64_t rank,
                                          const uint64_t *shape) const {
  assert(rank == getRank() && "Rank mismatch");
  for (uint64_t r = 0; r < rank; ++r)
    assert((shape[r] == 0 || shape[r] == idata[2 + r]) &&
           "Dimension size mismatch");
}

bool SparseTensorFile::canReadAs(PrimaryType valTy) const {
  switch (valueKind_) {
  case ValueKind::kInvalid:
    assert(false && "Must readHeader() before calling canReadAs()");
    return false; // In case assertions are disabled.
  case ValueKind::kPattern:
    return true;
  case ValueKind::kInteger:
    // When the file is specified to store integer values, we still
    // allow implicitly converting those to floating primary-types.
    return isRealPrimaryType(valTy);
  case ValueKind::kReal:
    // When the file is specified to store real/floating values, then
    // we disallow implicit conversion to integer primary-types.
    return isFloatingPrimaryType(valTy);
  case ValueKind::kComplex:
    // When the file is specified to store complex values, then we
    // require a complex primary-type.
    return isComplexPrimaryType(valTy);
  case ValueKind::kUndefined:
    // The "extended" FROSTT format doesn't specify a ValueKind.
    // So we allow implicitly converting the stored values to both
    // integer and floating primary-types.
    return isRealPrimaryType(valTy);
  }
  MLIR_SPARSETENSOR_FATAL("Unknown ValueKind: %d\n",
                          static_cast<uint8_t>(valueKind_));
}

/// Helper to convert C-style strings (i.e., '\0' terminated) to lower case.
static inline void toLower(char *token) {
  for (char *c = token; *c; ++c)
    *c = tolower(*c);
}

/// Idiomatic name for checking string equality.
static inline bool streq(const char *lhs, const char *rhs) {
  return strcmp(lhs, rhs) == 0;
}

/// Idiomatic name for checking string inequality.
static inline bool strne(const char *lhs, const char *rhs) {
  return strcmp(lhs, rhs); // aka `!= 0`
}

/// Read the MME header of a general sparse matrix of type real.
void SparseTensorFile::readMMEHeader() {
  char header[64];
  char object[64];
  char format[64];
  char field[64];
  char symmetry[64];
  // Read header line.
  if (fscanf(file, "%63s %63s %63s %63s %63s\n", header, object, format, field,
             symmetry) != 5)
    MLIR_SPARSETENSOR_FATAL("Corrupt header in %s\n", filename);
  // Convert all to lowercase up front (to avoid accidental redundancy).
  toLower(header);
  toLower(object);
  toLower(format);
  toLower(field);
  toLower(symmetry);
  // Process `field`, which specify pattern or the data type of the values.
  if (streq(field, "pattern"))
    valueKind_ = ValueKind::kPattern;
  else if (streq(field, "real"))
    valueKind_ = ValueKind::kReal;
  else if (streq(field, "integer"))
    valueKind_ = ValueKind::kInteger;
  else if (streq(field, "complex"))
    valueKind_ = ValueKind::kComplex;
  else
    MLIR_SPARSETENSOR_FATAL("Unexpected header field value in %s\n", filename);
  // Set properties.
  isSymmetric_ = streq(symmetry, "symmetric");
  // Make sure this is a general sparse matrix.
  if (strne(header, "%%matrixmarket") || strne(object, "matrix") ||
      strne(format, "coordinate") ||
      (strne(symmetry, "general") && !isSymmetric_))
    MLIR_SPARSETENSOR_FATAL("Cannot find a general sparse matrix in %s\n",
                            filename);
  // Skip comments.
  while (true) {
    readLine();
    if (line[0] != '%')
      break;
  }
  // Next line contains M N NNZ.
  idata[0] = 2; // rank
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "%" PRIu64 "\n", idata + 2, idata + 3,
             idata + 1) != 3)
    MLIR_SPARSETENSOR_FATAL("Cannot find size in %s\n", filename);
}

/// Read the "extended" FROSTT header. Although not part of the documented
/// format, we assume that the file starts with optional comments followed
/// by two lines that define the rank, the number of nonzeros, and the
/// dimensions sizes (one per rank) of the sparse tensor.
void SparseTensorFile::readExtFROSTTHeader() {
  // Skip comments.
  while (true) {
    readLine();
    if (line[0] != '#')
      break;
  }
  // Next line contains RANK and NNZ.
  if (sscanf(line, "%" PRIu64 "%" PRIu64 "\n", idata, idata + 1) != 2)
    MLIR_SPARSETENSOR_FATAL("Cannot find metadata in %s\n", filename);
  // Followed by a line with the dimension sizes (one per rank).
  for (uint64_t r = 0; r < idata[0]; ++r)
    if (fscanf(file, "%" PRIu64, idata + 2 + r) != 1)
      MLIR_SPARSETENSOR_FATAL("Cannot find dimension size %s\n", filename);
  readLine(); // end of line
  // The FROSTT format does not define the data type of the nonzero elements.
  valueKind_ = ValueKind::kUndefined;
}
