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

#include "mlir/ExecutionEngine/SparseTensor/PermutationRef.h"
#include "mlir/ExecutionEngine/SparseTensor/Storage.h"

#include <fstream>

namespace mlir {
namespace sparse_tensor {

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

//===----------------------------------------------------------------------===//

// TODO: benchmark whether to keep various methods inline vs moving them
// off to the cpp file.

// TODO: consider distinguishing separate classes for before vs
// after reading the header; so as to statically avoid the need
// to `assert(isValid())`.

/// This class abstracts over the information stored in file headers,
/// as well as providing the buffers and methods for parsing those headers.
class SparseTensorReader final {
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

  explicit SparseTensorReader(const char *filename) : filename(filename) {
    assert(filename && "Received nullptr for filename");
  }

  // Disallows copying, to avoid duplicating the `file` pointer.
  SparseTensorReader(const SparseTensorReader &) = delete;
  SparseTensorReader &operator=(const SparseTensorReader &) = delete;

  // This dtor tries to avoid leaking the `file`.  (Though it's better
  // to call `closeFile` explicitly when possible, since there are
  // circumstances where dtors are not called reliably.)
  ~SparseTensorReader() { closeFile(); }

  /// Opens the file for reading.
  void openFile();

  /// Closes the file.
  void closeFile();

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

  /// Reads a sparse tensor element from the next line in the input file and
  /// returns the value of the element. Stores the coordinates of the element
  /// to the `indices` array.
  template <typename V>
  V readCOOElement(uint64_t rank, uint64_t *indices) {
    char *linePtr = readCOOIndices(rank, indices);
    return detail::readCOOValue<V>(&linePtr, isPattern());
  }

private:
  /// Attempts to read a line from the file.  Is private because there's
  /// no reason for client code to call it.
  void readLine();

  /// Reads the next line of the input file and parses the coordinates
  /// into the `indices` argument.  Returns the position in the `line`
  /// buffer where the element's value should be parsed from.  This method
  /// has been factored out from `readCOOElement` to minimize code bloat
  /// for the generated library.
  char *readCOOIndices(uint64_t rank, uint64_t *indices);

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

/// Reads a sparse tensor with the given filename into a memory-resident
/// sparse tensor.
///
/// Preconditions:
/// * `dimShape` and `dim2lvl` must be valid for `dimRank`.
/// * `lvlTypes` and `lvl2dim` must be valid for `lvlRank`.
/// * `dim2lvl` is the inverse of `lvl2dim`.
///
/// Asserts:
/// * the file's actual value type can be read as `valTp`.
/// * the file's actual dimension-sizes match the expected `dimShape`.
/// * `dim2lvl` is a permutation, and therefore also `dimRank == lvlRank`.
//
// TODO: As currently written, this function uses `dim2lvl` in two
// places: first, to construct the level-sizes from the file's actual
// dimension-sizes; and second, to map the file's dimension-indices into
// level-indices.  The latter can easily generalize to arbitrary mappings,
// however the former cannot.  Thus, once we functionalize the mappings,
// this function will need both the sizes-to-sizes and indices-to-indices
// variants of the `dim2lvl` mapping.  For the `lvl2dim` direction we only
// need the indices-to-indices variant, for handing off to `newFromCOO`.
template <typename P, typename I, typename V>
inline SparseTensorStorage<P, I, V> *
openSparseTensor(uint64_t dimRank, const uint64_t *dimShape, uint64_t lvlRank,
                 const DimLevelType *lvlTypes, const uint64_t *lvl2dim,
                 const uint64_t *dim2lvl, const char *filename,
                 PrimaryType valTp) {
  // Read the file's header and check the file's actual element type and
  // dimension-sizes against the expected element type and dimension-shape.
  SparseTensorReader stfile(filename);
  stfile.openFile();
  stfile.readHeader();
  if (!stfile.canReadAs(valTp))
    MLIR_SPARSETENSOR_FATAL(
        "Tensor element type %d not compatible with values in file %s\n",
        static_cast<int>(valTp), filename);
  stfile.assertMatchesShape(dimRank, dimShape);
  const uint64_t *dimSizes = stfile.getDimSizes();
  // Construct the level-sizes from the file's dimension-sizes
  // TODO: This doesn't generalize to arbitrary mappings. (See above.)
  assert(dimRank == lvlRank && "Rank mismatch");
  detail::PermutationRef d2l(dimRank, dim2lvl);
  std::vector<uint64_t> lvlSizes = d2l.pushforward(dimRank, dimSizes);
  // Prepare a COO object with the number of nonzeros as initial capacity.
  uint64_t nnz = stfile.getNNZ();
  auto *lvlCOO = new SparseTensorCOO<V>(lvlSizes, nnz);
  // Read all nonzero elements.
  std::vector<uint64_t> dimInd(dimRank);
  std::vector<uint64_t> lvlInd(lvlRank);
  for (uint64_t k = 0; k < nnz; ++k) {
    const V value = stfile.readCOOElement<V>(dimRank, dimInd.data());
    d2l.pushforward(dimRank, dimInd.data(), lvlInd.data());
    // TODO: <https://github.com/llvm/llvm-project/issues/54179>
    lvlCOO->add(lvlInd, value);
    // We currently chose to deal with symmetric matrices by fully
    // constructing them.  In the future, we may want to make symmetry
    // implicit for storage reasons.
    if (stfile.isSymmetric() && lvlInd[0] != lvlInd[1])
      lvlCOO->add({lvlInd[1], lvlInd[0]}, value);
  }
  // Close the file, convert the COO to SparseTensorStorage, and return.
  stfile.closeFile();
  auto *tensor = SparseTensorStorage<P, I, V>::newFromCOO(
      dimRank, dimSizes, lvlRank, lvlTypes, lvl2dim, *lvlCOO);
  delete lvlCOO;
  return tensor;
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
