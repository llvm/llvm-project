//===- SparseTensorUtils.h - Enums shared with the runtime ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file provides the enums and functions which comprise the
// public API of `ExecutionEngine/SparseUtils.cpp`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSORUTILS_H_
#define MLIR_EXECUTIONENGINE_SPARSETENSORUTILS_H_

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/Float16bits.h"

#include <cinttypes>
#include <complex>
#include <vector>

extern "C" {

//===----------------------------------------------------------------------===//
//
// Typedefs and enums.  These are required to be public so that they
// can be shared with `Transforms/SparseTensorConversion.cpp`, since
// they define the arguments to the public functions declared later on.
//
// This section also defines x-macros <https://en.wikipedia.org/wiki/X_Macro>
// so that we can generate variations of the public functions for each
// supported primary- and/or overhead-type.
//
//===----------------------------------------------------------------------===//

/// This type is used in the public API at all places where MLIR expects
/// values with the built-in type "index". For now, we simply assume that
/// type is 64-bit, but targets with different "index" bit widths should
/// link with an alternatively built runtime support library.
// TODO: support such targets?
using index_type = uint64_t;

/// Encoding of overhead types (both pointer overhead and indices
/// overhead), for "overloading" @newSparseTensor.
enum class OverheadType : uint32_t {
  kIndex = 0,
  kU64 = 1,
  kU32 = 2,
  kU16 = 3,
  kU8 = 4
};

// This x-macro calls its argument on every overhead type which has
// fixed-width.  It excludes `index_type` because that type is often
// handled specially (e.g., by translating it into the architecture-dependent
// equivalent fixed-width overhead type).
#define FOREVERY_FIXED_O(DO)                                                   \
  DO(64, uint64_t)                                                             \
  DO(32, uint32_t)                                                             \
  DO(16, uint16_t)                                                             \
  DO(8, uint8_t)

// This x-macro calls its argument on every overhead type, including
// `index_type`.
#define FOREVERY_O(DO)                                                         \
  FOREVERY_FIXED_O(DO)                                                         \
  DO(0, index_type)

// These are not just shorthands but indicate the particular
// implementation used (e.g., as opposed to C99's `complex double`,
// or MLIR's `ComplexType`).
using complex64 = std::complex<double>;
using complex32 = std::complex<float>;

/// Encoding of the elemental type, for "overloading" @newSparseTensor.
enum class PrimaryType : uint32_t {
  kF64 = 1,
  kF32 = 2,
  kF16 = 3,
  kBF16 = 4,
  kI64 = 5,
  kI32 = 6,
  kI16 = 7,
  kI8 = 8,
  kC64 = 9,
  kC32 = 10
};

// This x-macro includes all `V` types.
#define FOREVERY_V(DO)                                                         \
  DO(F64, double)                                                              \
  DO(F32, float)                                                               \
  DO(F16, f16)                                                                 \
  DO(BF16, bf16)                                                               \
  DO(I64, int64_t)                                                             \
  DO(I32, int32_t)                                                             \
  DO(I16, int16_t)                                                             \
  DO(I8, int8_t)                                                               \
  DO(C64, complex64)                                                           \
  DO(C32, complex32)

/// The actions performed by @newSparseTensor.
enum class Action : uint32_t {
  kEmpty = 0,
  kFromFile = 1,
  kFromCOO = 2,
  kSparseToSparse = 3,
  kEmptyCOO = 4,
  kToCOO = 5,
  kToIterator = 6
};

/// This enum mimics `SparseTensorEncodingAttr::DimLevelType` for
/// breaking dependency cycles.  `SparseTensorEncodingAttr::DimLevelType`
/// is the source of truth and this enum should be kept consistent with it.
enum class DimLevelType : uint8_t {
  kDense = 0,
  kCompressed = 1,
  kSingleton = 2
};

//===----------------------------------------------------------------------===//
//
// Public functions which operate on MLIR buffers (memrefs) to interact
// with sparse tensors (which are only visible as opaque pointers externally).
// Because these functions deal with memrefs, they should only be used
// by MLIR compiler-generated code (or code similarly guaranteed to remain
// in sync with MLIR; e.g., internal development tools like benchmarks).
//
// Where appropriate, we use macros to generate all variations of these
// functions for each supported primary- and overhead-type.
//
//===----------------------------------------------------------------------===//

/// The @newSparseTensor function for constructing a new sparse tensor.
/// This is the "swiss army knife" method for materializing sparse
/// tensors into the computation.  The types of the `ptr` argument and
/// the result depend on the action, as explained in the following table
/// (where "STS" means a sparse-tensor-storage object, and "COO" means
/// a coordinate-scheme object).
///
/// Action:         `ptr`:          Returns:
/// kEmpty          unused          STS, empty
/// kEmptyCOO       unused          COO, empty
/// kFromFile       char* filename  STS, read from the file
/// kFromCOO        COO             STS, copied from the COO source
/// kToCOO          STS             COO, copied from the STS source
/// kSparseToSparse STS             STS, copied from the STS source
/// kToIterator     STS             COO-Iterator, call @getNext to use
MLIR_CRUNNERUTILS_EXPORT void *
_mlir_ciface_newSparseTensor(StridedMemRefType<DimLevelType, 1> *aref, // NOLINT
                             StridedMemRefType<index_type, 1> *sref,
                             StridedMemRefType<index_type, 1> *pref,
                             OverheadType ptrTp, OverheadType indTp,
                             PrimaryType valTp, Action action, void *ptr);

/// Tensor-storage method to obtain direct access to the values array.
#define DECL_SPARSEVALUES(VNAME, V)                                            \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_sparseValues##VNAME(              \
      StridedMemRefType<V, 1> *out, void *tensor);
FOREVERY_V(DECL_SPARSEVALUES)
#undef DECL_SPARSEVALUES

/// Tensor-storage method to obtain direct access to the pointers array
/// for the given dimension.
#define DECL_SPARSEPOINTERS(PNAME, P)                                          \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_sparsePointers##PNAME(            \
      StridedMemRefType<P, 1> *out, void *tensor, index_type d);
FOREVERY_O(DECL_SPARSEPOINTERS)
#undef DECL_SPARSEPOINTERS

/// Tensor-storage method to obtain direct access to the indices array
/// for the given dimension.
#define DECL_SPARSEINDICES(INAME, I)                                           \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_sparseIndices##INAME(             \
      StridedMemRefType<I, 1> *out, void *tensor, index_type d);
FOREVERY_O(DECL_SPARSEINDICES)
#undef DECL_SPARSEINDICES

/// Coordinate-scheme method for adding a new element.
#define DECL_ADDELT(VNAME, V)                                                  \
  MLIR_CRUNNERUTILS_EXPORT void *_mlir_ciface_addElt##VNAME(                   \
      void *coo, StridedMemRefType<V, 0> *vref,                                \
      StridedMemRefType<index_type, 1> *iref,                                  \
      StridedMemRefType<index_type, 1> *pref);
FOREVERY_V(DECL_ADDELT)
#undef DECL_ADDELT

/// Coordinate-scheme method for getting the next element while iterating.
#define DECL_GETNEXT(VNAME, V)                                                 \
  MLIR_CRUNNERUTILS_EXPORT bool _mlir_ciface_getNext##VNAME(                   \
      void *coo, StridedMemRefType<index_type, 1> *iref,                       \
      StridedMemRefType<V, 0> *vref);
FOREVERY_V(DECL_GETNEXT)
#undef DECL_GETNEXT

/// Tensor-storage method to insert elements in lexicographical index order.
#define DECL_LEXINSERT(VNAME, V)                                               \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_lexInsert##VNAME(                 \
      void *tensor, StridedMemRefType<index_type, 1> *cref,                    \
      StridedMemRefType<V, 0> *vref);
FOREVERY_V(DECL_LEXINSERT)
#undef DECL_LEXINSERT

/// Tensor-storage method to insert using expansion.
#define DECL_EXPINSERT(VNAME, V)                                               \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_expInsert##VNAME(                 \
      void *tensor, StridedMemRefType<index_type, 1> *cref,                    \
      StridedMemRefType<V, 1> *vref, StridedMemRefType<bool, 1> *fref,         \
      StridedMemRefType<index_type, 1> *aref, index_type count);
FOREVERY_V(DECL_EXPINSERT)
#undef DECL_EXPINSERT

//===----------------------------------------------------------------------===//
//
// Public functions which accept only C-style data structures to interact
// with sparse tensors (which are only visible as opaque pointers externally).
// These functions can be used both by MLIR compiler-generated code
// as well as by any external runtime that wants to interact with MLIR
// compiler-generated code.
//
//===----------------------------------------------------------------------===//

/// Tensor-storage method to get the size of the given dimension.
MLIR_CRUNNERUTILS_EXPORT index_type sparseDimSize(void *tensor, index_type d);

/// Tensor-storage method to finalize lexicographic insertions.
MLIR_CRUNNERUTILS_EXPORT void endInsert(void *tensor);

/// Coordinate-scheme method to write to file in extended FROSTT format.
#define DECL_OUTSPARSETENSOR(VNAME, V)                                         \
  MLIR_CRUNNERUTILS_EXPORT void outSparseTensor##VNAME(void *coo, void *dest,  \
                                                       bool sort);
FOREVERY_V(DECL_OUTSPARSETENSOR)
#undef DECL_OUTSPARSETENSOR

/// Releases the memory for the tensor-storage object.
void delSparseTensor(void *tensor);

/// Releases the memory for the coordinate-scheme object.
#define DECL_DELCOO(VNAME, V)                                                  \
  MLIR_CRUNNERUTILS_EXPORT void delSparseTensorCOO##VNAME(void *coo);
FOREVERY_V(DECL_DELCOO)
#undef DECL_DELCOO

/// Helper function to read a sparse tensor filename from the environment,
/// defined with the naming convention ${TENSOR0}, ${TENSOR1}, etc.
MLIR_CRUNNERUTILS_EXPORT char *getTensorFilename(index_type id);

/// Helper function to read the header of a file and return the
/// shape/sizes, without parsing the elements of the file.
MLIR_CRUNNERUTILS_EXPORT void readSparseTensorShape(char *filename,
                                                    std::vector<uint64_t> *out);

/// Initializes sparse tensor from a COO-flavored format expressed using
/// C-style data structures.  The expected parameters are:
///
///   rank:    rank of tensor
///   nse:     number of specified elements (usually the nonzeros)
///   shape:   array with dimension size for each rank
///   values:  a "nse" array with values for all specified elements
///   indices: a flat "nse * rank" array with indices for all specified elements
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
#define DECL_CONVERTTOMLIRSPARSETENSOR(VNAME, V)                               \
  MLIR_CRUNNERUTILS_EXPORT void *convertToMLIRSparseTensor##VNAME(             \
      uint64_t rank, uint64_t nse, uint64_t *shape, V *values,                 \
      uint64_t *indices, uint64_t *perm, uint8_t *sparse);
FOREVERY_V(DECL_CONVERTTOMLIRSPARSETENSOR)
#undef DECL_CONVERTTOMLIRSPARSETENSOR

/// Converts a sparse tensor to COO-flavored format expressed using
/// C-style data structures.  The expected output parameters are pointers
/// for these values:
///
///   rank:    rank of tensor
///   nse:     number of specified elements (usually the nonzeros)
///   shape:   array with dimension size for each rank
///   values:  a "nse" array with values for all specified elements
///   indices: a flat "nse * rank" array with indices for all specified elements
///
/// The input is a pointer to `SparseTensorStorage<P, I, V>`, typically
/// returned from `convertToMLIRSparseTensor`.
#define DECL_CONVERTFROMMLIRSPARSETENSOR(VNAME, V)                             \
  MLIR_CRUNNERUTILS_EXPORT void convertFromMLIRSparseTensor##VNAME(            \
      void *tensor, uint64_t *pRank, uint64_t *pNse, uint64_t **pShape,        \
      V **pValues, uint64_t **pIndices);
FOREVERY_V(DECL_CONVERTFROMMLIRSPARSETENSOR)
#undef DECL_CONVERTFROMMLIRSPARSETENSOR

} // extern "C"

#endif // MLIR_EXECUTIONENGINE_SPARSETENSORUTILS_H_
