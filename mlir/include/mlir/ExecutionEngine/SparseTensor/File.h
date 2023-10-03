//===- File.h - Reading/writing sparse tensors from/to files ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements reading and writing sparse tensor files in one of the
// following external formats:
//
// (1) Matrix Market Exchange (MME): *.mtx
//     https://math.nist.gov/MatrixMarket/formats.html
//
// (2) Formidable Repository of Open Sparse Tensors and Tools (FROSTT): *.tns
//     http://frostt.io/tensors/file-formats.html
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_FILE_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_FILE_H

#include "mlir/ExecutionEngine/SparseTensor/Storage.h"

#include <fstream>

namespace mlir {
namespace sparse_tensor {

namespace detail {

template <typename T>
struct is_complex final : public std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> final : public std::true_type {};

/// Returns an element-value of non-complex type.  If `IsPattern` is true,
/// then returns an arbitrary value.  If `IsPattern` is false, then
/// reads the value from the current line buffer beginning at `linePtr`.
template <typename V, bool IsPattern>
inline std::enable_if_t<!is_complex<V>::value, V> readValue(char **linePtr) {
  // The external formats always store these numerical values with the type
  // double, but we cast these values to the sparse tensor object type.
  // For a pattern tensor, we arbitrarily pick the value 1 for all entries.
  if constexpr (IsPattern)
    return 1.0;
  return strtod(*linePtr, linePtr);
}

/// Returns an element-value of complex type.  If `IsPattern` is true,
/// then returns an arbitrary value.  If `IsPattern` is false, then reads
/// the value from the current line buffer beginning at `linePtr`.
template <typename V, bool IsPattern>
inline std::enable_if_t<is_complex<V>::value, V> readValue(char **linePtr) {
  // Read two values to make a complex. The external formats always store
  // numerical values with the type double, but we cast these values to the
  // sparse tensor object type. For a pattern tensor, we arbitrarily pick the
  // value 1 for all entries.
  if constexpr (IsPattern)
    return V(1.0, 1.0);
  double re = strtod(*linePtr, linePtr);
  double im = strtod(*linePtr, linePtr);
  // Avoiding brace-notation since that forbids narrowing to `float`.
  return V(re, im);
}

/// Returns an element-value.  If `isPattern` is true, then returns an
/// arbitrary value.  If `isPattern` is false, then reads the value from
/// the current line buffer beginning at `linePtr`.
template <typename V>
inline V readValue(char **linePtr, bool isPattern) {
  return isPattern ? readValue<V, true>(linePtr) : readValue<V, false>(linePtr);
}

} // namespace detail

//===----------------------------------------------------------------------===//

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

  /// Factory method to allocate a new reader, open the file, read the
  /// header, and validate that the actual contents of the file match
  /// the expected `dimShape` and `valTp`.
  static SparseTensorReader *create(const char *filename, uint64_t dimRank,
                                    const uint64_t *dimShape,
                                    PrimaryType valTp) {
    SparseTensorReader *reader = new SparseTensorReader(filename);
    reader->openFile();
    reader->readHeader();
    if (!reader->canReadAs(valTp))
      MLIR_SPARSETENSOR_FATAL(
          "Tensor element type %d not compatible with values in file %s\n",
          static_cast<int>(valTp), filename);
    reader->assertMatchesShape(dimRank, dimShape);
    return reader;
  }

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

  /// Gets the dimension-rank of the tensor.  Is only valid after parsing
  /// the header.
  uint64_t getRank() const {
    assert(isValid() && "Attempt to getRank() before readHeader()");
    return idata[0];
  }

  /// Gets the number of stored elements.  Is only valid after parsing
  /// the header.
  uint64_t getNSE() const {
    assert(isValid() && "Attempt to getNSE() before readHeader()");
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
  /// to the `dimCoords` array.
  template <typename V>
  V readElement(uint64_t dimRank, uint64_t *dimCoords) {
    assert(dimRank == getRank() && "rank mismatch");
    char *linePtr = readCoords(dimCoords);
    return detail::readValue<V>(&linePtr, isPattern());
  }

  /// Allocates a new COO object for `lvlSizes`, initializes it by reading
  /// all the elements from the file and applying `dim2lvl` to their
  /// dim-coordinates, and then closes the file. Templated on V only.
  template <typename V>
  SparseTensorCOO<V> *readCOO(uint64_t lvlRank, const uint64_t *lvlSizes,
                              const uint64_t *dim2lvl);

  /// Allocates a new sparse-tensor storage object with the given encoding,
  /// initializes it by reading all the elements from the file, and then
  /// closes the file. Templated on P, I, and V.
  template <typename P, typename I, typename V>
  SparseTensorStorage<P, I, V> *
  readSparseTensor(uint64_t lvlRank, const uint64_t *lvlSizes,
                   const DimLevelType *lvlTypes, const uint64_t *lvl2dim,
                   const uint64_t *dim2lvl) {
    auto *lvlCOO = readCOO<V>(lvlRank, lvlSizes, dim2lvl);
    auto *tensor = SparseTensorStorage<P, I, V>::newFromCOO(
        getRank(), getDimSizes(), lvlRank, lvlTypes, lvl2dim, *lvlCOO);
    delete lvlCOO;
    return tensor;
  }

  /// Reads the COO tensor from the file, stores the coordinates and values to
  /// the given buffers, returns a boolean value to indicate whether the COO
  /// elements are sorted.
  /// Precondition: the buffers should have enough space to hold the elements.
  template <typename C, typename V>
  bool readToBuffers(uint64_t lvlRank, const uint64_t *dim2lvl,
                     C *lvlCoordinates, V *values);

private:
  /// Attempts to read a line from the file.  Is private because there's
  /// no reason for client code to call it.
  void readLine();

  /// Reads the next line of the input file and parses the coordinates
  /// into the `dimCoords` argument.  Returns the position in the `line`
  /// buffer where the element's value should be parsed from.  This method
  /// has been factored out from `readElement` to minimize code bloat
  /// for the generated library.
  ///
  /// Precondition: `dimCoords` is valid for `getRank()`.
  template <typename C>
  char *readCoords(C *dimCoords) {
    readLine();
    // Local variable for tracking the parser's position in the `line` buffer.
    char *linePtr = line;
    for (uint64_t dimRank = getRank(), d = 0; d < dimRank; ++d) {
      // Parse the 1-based coordinate.
      uint64_t c = strtoul(linePtr, &linePtr, 10);
      // Store the 0-based coordinate.
      dimCoords[d] = static_cast<C>(c - 1);
    }
    return linePtr;
  }

  /// The internal implementation of `readCOO`.  We template over
  /// `IsPattern` in order to perform LICM without needing to duplicate the
  /// source code.
  //
  // TODO: We currently take the `dim2lvl` argument as a `PermutationRef`
  // since that's what `readCOO` creates.  Once we update `readCOO` to
  // functionalize the mapping, then this helper will just take that
  // same function.
  template <typename V, bool IsPattern>
  void readCOOLoop(uint64_t lvlRank, detail::PermutationRef dim2lvl,
                   SparseTensorCOO<V> *lvlCOO);

  /// The internal implementation of `readToBuffers`.  We template over
  /// `IsPattern` in order to perform LICM without needing to duplicate the
  /// source code.
  template <typename C, typename V, bool IsPattern>
  bool readToBuffersLoop(uint64_t lvlRank, detail::PermutationRef dim2lvl,
                         C *lvlCoordinates, V *values);

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

template <typename V>
SparseTensorCOO<V> *SparseTensorReader::readCOO(uint64_t lvlRank,
                                                const uint64_t *lvlSizes,
                                                const uint64_t *dim2lvl) {
  assert(isValid() && "Attempt to readCOO() before readHeader()");
  const uint64_t dimRank = getRank();
  assert(lvlRank == dimRank && "Rank mismatch");
  detail::PermutationRef d2l(dimRank, dim2lvl);
  // Prepare a COO object with the number of stored elems as initial capacity.
  auto *lvlCOO = new SparseTensorCOO<V>(lvlRank, lvlSizes, getNSE());
  // Do some manual LICM, to avoid assertions in the for-loop.
  const bool IsPattern = isPattern();
  if (IsPattern)
    readCOOLoop<V, true>(lvlRank, d2l, lvlCOO);
  else
    readCOOLoop<V, false>(lvlRank, d2l, lvlCOO);
  // Close the file and return the COO.
  closeFile();
  return lvlCOO;
}

template <typename V, bool IsPattern>
void SparseTensorReader::readCOOLoop(uint64_t lvlRank,
                                     detail::PermutationRef dim2lvl,
                                     SparseTensorCOO<V> *lvlCOO) {
  const uint64_t dimRank = getRank();
  std::vector<uint64_t> dimCoords(dimRank);
  std::vector<uint64_t> lvlCoords(lvlRank);
  for (uint64_t nse = getNSE(), k = 0; k < nse; ++k) {
    // We inline `readElement` here in order to avoid redundant
    // assertions, since they're guaranteed by the call to `isValid()`
    // and the construction of `dimCoords` above.
    char *linePtr = readCoords(dimCoords.data());
    const V value = detail::readValue<V, IsPattern>(&linePtr);
    dim2lvl.pushforward(dimRank, dimCoords.data(), lvlCoords.data());
    // TODO: <https://github.com/llvm/llvm-project/issues/54179>
    lvlCOO->add(lvlCoords, value);
  }
}

template <typename C, typename V>
bool SparseTensorReader::readToBuffers(uint64_t lvlRank,
                                       const uint64_t *dim2lvl,
                                       C *lvlCoordinates, V *values) {
  assert(isValid() && "Attempt to readCOO() before readHeader()");
  // Construct a `PermutationRef` for the `pushforward` below.
  // TODO: This specific implementation does not generalize to arbitrary
  // mappings, but once we functionalize the `dim2lvl` argument we can
  // simply use that function instead.
  const uint64_t dimRank = getRank();
  assert(lvlRank == dimRank && "Rank mismatch");
  detail::PermutationRef d2l(dimRank, dim2lvl);
  // Do some manual LICM, to avoid assertions in the for-loop.
  bool isSorted =
      isPattern()
          ? readToBuffersLoop<C, V, true>(lvlRank, d2l, lvlCoordinates, values)
          : readToBuffersLoop<C, V, false>(lvlRank, d2l, lvlCoordinates,
                                           values);

  // Close the file and return isSorted.
  closeFile();
  return isSorted;
}

template <typename C, typename V, bool IsPattern>
bool SparseTensorReader::readToBuffersLoop(uint64_t lvlRank,
                                           detail::PermutationRef dim2lvl,
                                           C *lvlCoordinates, V *values) {
  const uint64_t dimRank = getRank();
  const uint64_t nse = getNSE();
  std::vector<C> dimCoords(dimRank);
  // Read the first element with isSorted=false as a way to avoid accessing its
  // previous element.
  bool isSorted = false;
  char *linePtr;
  // We inline `readElement` here in order to avoid redundant assertions,
  // since they're guaranteed by the call to `isValid()` and the construction
  // of `dimCoords` above.
  const auto readNextElement = [&]() {
    linePtr = readCoords<C>(dimCoords.data());
    dim2lvl.pushforward(dimRank, dimCoords.data(), lvlCoordinates);
    *values = detail::readValue<V, IsPattern>(&linePtr);
    if (isSorted) {
      // Note that isSorted was set to false while reading the first element,
      // to guarantee the safeness of using prevLvlCoords.
      C *prevLvlCoords = lvlCoordinates - lvlRank;
      // TODO: define a new CoordsLT which is like ElementLT but doesn't have
      // the V parameter, and use it here.
      for (uint64_t l = 0; l < lvlRank; ++l) {
        if (prevLvlCoords[l] != lvlCoordinates[l]) {
          if (prevLvlCoords[l] > lvlCoordinates[l])
            isSorted = false;
          break;
        }
      }
    }
    lvlCoordinates += lvlRank;
    ++values;
  };
  readNextElement();
  isSorted = true;
  for (uint64_t n = 1; n < nse; ++n)
    readNextElement();

  return isSorted;
}

/// Writes the sparse tensor to `filename` in extended FROSTT format.
template <typename V>
inline void writeExtFROSTT(const SparseTensorCOO<V> &coo,
                           const char *filename) {
  assert(filename && "Got nullptr for filename");
  const auto &dimSizes = coo.getDimSizes();
  const auto &elements = coo.getElements();
  const uint64_t dimRank = coo.getRank();
  const uint64_t nse = elements.size();
  std::fstream file;
  file.open(filename, std::ios_base::out | std::ios_base::trunc);
  assert(file.is_open());
  file << "; extended FROSTT format\n" << dimRank << " " << nse << std::endl;
  for (uint64_t d = 0; d < dimRank - 1; ++d)
    file << dimSizes[d] << " ";
  file << dimSizes[dimRank - 1] << std::endl;
  for (uint64_t i = 0; i < nse; ++i) {
    const auto &coords = elements[i].coords;
    for (uint64_t d = 0; d < dimRank; ++d)
      file << (coords[d] + 1) << " ";
    file << elements[i].value << std::endl;
  }
  file.flush();
  file.close();
  assert(file.good());
}

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_FILE_H
