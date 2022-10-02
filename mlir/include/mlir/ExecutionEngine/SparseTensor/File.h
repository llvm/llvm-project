//===- File.h - Parsing sparse tensors from files ---------------*- C++ -*-===//
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

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_FILE_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_FILE_H

#include "mlir/ExecutionEngine/SparseTensor/Storage.h"

#include <fstream>

namespace mlir {
namespace sparse_tensor {

// TODO: benchmark whether to keep various methods inline vs moving them
// off to the cpp file.

// TODO: consider distinguishing separate classes for before vs
// after reading the header; so as to statically avoid the need
// to `assert(isValid())`.

/// This class abstracts over the information stored in file headers,
/// as well as providing the buffers and methods for parsing those headers.
class SparseTensorFile final {
public:
  enum class ValueKind : uint8_t {
    // The value before calling `readHeader`.
    kInvalid = 0,
    // Values that can be set by `readMMEHeader`.
    kPattern = 1,
    kReal = 2,
    kInteger = 3,
    kComplex = 4,
    // The value set by `readExtFROSTTHeader`.
    kUndefined = 5
  };

  explicit SparseTensorFile(const char *filename) : filename(filename) {
    assert(filename && "Received nullptr for filename");
  }

  // Disallows copying, to avoid duplicating the `file` pointer.
  SparseTensorFile(const SparseTensorFile &) = delete;
  SparseTensorFile &operator=(const SparseTensorFile &) = delete;

  // This dtor tries to avoid leaking the `file`.  (Though it's better
  // to call `closeFile` explicitly when possible, since there are
  // circumstances where dtors are not called reliably.)
  ~SparseTensorFile() { closeFile(); }

  /// Opens the file for reading.
  void openFile();

  /// Closes the file.
  void closeFile();

  /// Attempts to read a line from the file.
  char *readLine();

  /// Reads and parses the file's header.
  void readHeader();

  ValueKind getValueKind() const { return valueKind_; }

  /// Checks if a header has been successfully read.
  bool isValid() const { return valueKind_ != ValueKind::kInvalid; }

  /// Checks if the file's ValueKind can be converted into the given
  /// tensor PrimaryType.  Is only valid after parsing the header.
  bool canReadAs(PrimaryType valTy) const;

  /// Gets the MME "pattern" property setting.  Is only valid after
  /// parsing the header.
  bool isPattern() const {
    assert(isValid() && "Attempt to isPattern() before readHeader()");
    return valueKind_ == ValueKind::kPattern;
  }

  /// Gets the MME "symmetric" property setting.  Is only valid after
  /// parsing the header.
  bool isSymmetric() const {
    assert(isValid() && "Attempt to isSymmetric() before readHeader()");
    return isSymmetric_;
  }

  /// Gets the rank of the tensor.  Is only valid after parsing the header.
  uint64_t getRank() const {
    assert(isValid() && "Attempt to getRank() before readHeader()");
    return idata[0];
  }

  /// Gets the number of non-zeros.  Is only valid after parsing the header.
  uint64_t getNNZ() const {
    assert(isValid() && "Attempt to getNNZ() before readHeader()");
    return idata[1];
  }

  /// Gets the dimension-sizes array.  The pointer itself is always
  /// valid; however, the values stored therein are only valid after
  /// parsing the header.
  const uint64_t *getDimSizes() const { return idata + 2; }

  /// Safely gets the size of the given dimension.  Is only valid
  /// after parsing the header.
  uint64_t getDimSize(uint64_t d) const {
    assert(d < getRank() && "Dimension out of bounds");
    return idata[2 + d];
  }

  /// Asserts the shape subsumes the actual dimension sizes.  Is only
  /// valid after parsing the header.
  void assertMatchesShape(uint64_t rank, const uint64_t *shape) const;

private:
  /// Reads the MME header of a general sparse matrix of type real.
  void readMMEHeader();

  /// Reads the "extended" FROSTT header. Although not part of the
  /// documented format, we assume that the file starts with optional
  /// comments followed by two lines that define the rank, the number of
  /// nonzeros, and the dimensions sizes (one per rank) of the sparse tensor.
  void readExtFROSTTHeader();

  static constexpr int kColWidth = 1025;
  const char *const filename;
  FILE *file = nullptr;
  ValueKind valueKind_ = ValueKind::kInvalid;
  bool isSymmetric_ = false;
  uint64_t idata[512];
  char line[kColWidth];
};

//===----------------------------------------------------------------------===//
namespace detail {

template <typename T>
struct is_complex final : public std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> final : public std::true_type {};

/// Reads an element of a non-complex type for the current indices in
/// coordinate scheme.
template <typename V>
inline std::enable_if_t<!is_complex<V>::value, V>
readCOOValue(char **linePtr, bool is_pattern) {
  // The external formats always store these numerical values with the type
  // double, but we cast these values to the sparse tensor object type.
  // For a pattern tensor, we arbitrarily pick the value 1 for all entries.
  return is_pattern ? 1.0 : strtod(*linePtr, linePtr);
}

/// Reads an element of a complex type for the current indices in
/// coordinate scheme.
template <typename V>
inline std::enable_if_t<is_complex<V>::value, V> readCOOValue(char **linePtr,
                                                              bool is_pattern) {
  // Read two values to make a complex. The external formats always store
  // numerical values with the type double, but we cast these values to the
  // sparse tensor object type. For a pattern tensor, we arbitrarily pick the
  // value 1 for all entries.
  double re = is_pattern ? 1.0 : strtod(*linePtr, linePtr);
  double im = is_pattern ? 1.0 : strtod(*linePtr, linePtr);
  // Avoiding brace-notation since that forbids narrowing to `float`.
  return V(re, im);
}

} // namespace detail

/// Reads a sparse tensor with the given filename into a memory-resident
/// sparse tensor in coordinate scheme.
template <typename V>
inline SparseTensorCOO<V> *
openSparseTensorCOO(const char *filename, uint64_t rank, const uint64_t *shape,
                    const uint64_t *perm, PrimaryType valTp) {
  SparseTensorFile stfile(filename);
  stfile.openFile();
  stfile.readHeader();
  // Check tensor element type against the value type in the input file.
  if (!stfile.canReadAs(valTp))
    MLIR_SPARSETENSOR_FATAL(
        "Tensor element type %d not compatible with values in file %s\n",
        static_cast<int>(valTp), filename);
  stfile.assertMatchesShape(rank, shape);
  // Prepare sparse tensor object with per-dimension sizes
  // and the number of nonzeros as initial capacity.
  uint64_t nnz = stfile.getNNZ();
  auto *coo = SparseTensorCOO<V>::newSparseTensorCOO(rank, stfile.getDimSizes(),
                                                     perm, nnz);
  // Read all nonzero elements.
  std::vector<uint64_t> indices(rank);
  for (uint64_t k = 0; k < nnz; ++k) {
    char *linePtr = stfile.readLine();
    for (uint64_t r = 0; r < rank; ++r) {
      // Parse the 1-based index.
      uint64_t idx = strtoul(linePtr, &linePtr, 10);
      // Add the 0-based index.
      indices[perm[r]] = idx - 1;
    }
    const V value = detail::readCOOValue<V>(&linePtr, stfile.isPattern());
    // TODO: <https://github.com/llvm/llvm-project/issues/54179>
    coo->add(indices, value);
    // We currently chose to deal with symmetric matrices by fully
    // constructing them.  In the future, we may want to make symmetry
    // implicit for storage reasons.
    if (stfile.isSymmetric() && indices[0] != indices[1])
      coo->add({indices[1], indices[0]}, value);
  }
  // Close the file and return tensor.
  stfile.closeFile();
  return coo;
}

/// Writes the sparse tensor to `filename` in extended FROSTT format.
template <typename V>
inline void writeExtFROSTT(const SparseTensorCOO<V> &coo,
                           const char *filename) {
  assert(filename && "Got nullptr for filename");
  auto &dimSizes = coo.getDimSizes();
  auto &elements = coo.getElements();
  const uint64_t rank = coo.getRank();
  const uint64_t nnz = elements.size();
  std::fstream file;
  file.open(filename, std::ios_base::out | std::ios_base::trunc);
  assert(file.is_open());
  file << "; extended FROSTT format\n" << rank << " " << nnz << std::endl;
  for (uint64_t r = 0; r < rank - 1; ++r)
    file << dimSizes[r] << " ";
  file << dimSizes[rank - 1] << std::endl;
  for (uint64_t i = 0; i < nnz; ++i) {
    auto &idx = elements[i].indices;
    for (uint64_t r = 0; r < rank; ++r)
      file << (idx[r] + 1) << " ";
    file << elements[i].value << std::endl;
  }
  file.flush();
  file.close();
  assert(file.good());
}

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_FILE_H
