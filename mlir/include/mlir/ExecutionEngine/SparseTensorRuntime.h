//===- SparseTensorRuntime.h - SparseTensor runtime support lib -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file provides the enums and functions which comprise the
// public API of the `ExecutionEngine/SparseTensorRuntime.cpp` runtime
// support library for the SparseTensor dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSORRUNTIME_H
#define MLIR_EXECUTIONENGINE_SPARSETENSORRUNTIME_H

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/Float16bits.h"

#include <cinttypes>
#include <complex>
#include <vector>

using namespace mlir::sparse_tensor;

extern "C" {

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
/// (where "STS" means a sparse-tensor-storage object, "COO" means
/// a coordinate-scheme object, and "Iterator" means an iterator object).
///
/// Action:         `ptr`:          Returns:
/// kEmpty          unused          STS, empty
/// kEmptyCOO       unused          COO, empty
/// kFromFile       char* filename  STS, read from the file
/// kFromCOO        COO             STS, copied from the COO source
/// kToCOO          STS             COO, copied from the STS source
/// kSparseToSparse STS             STS, copied from the STS source
/// kToIterator     STS             Iterator, call @getNext to use and
///                                 @delSparseTensorIterator to free.
MLIR_CRUNNERUTILS_EXPORT void *_mlir_ciface_newSparseTensor( // NOLINT
    StridedMemRefType<index_type, 1> *dimSizesRef,
    StridedMemRefType<index_type, 1> *lvlSizesRef,
    StridedMemRefType<DimLevelType, 1> *lvlTypesRef,
    StridedMemRefType<index_type, 1> *lvl2dimRef,
    StridedMemRefType<index_type, 1> *dim2lvlRef, OverheadType posTp,
    OverheadType crdTp, PrimaryType valTp, Action action, void *ptr);

// TODO: document what all the arguments are/mean for the functions below,
// especially with regards to "dim"-vs-"lvl" and mappings/permutations.

/// Tensor-storage method to obtain direct access to the values array.
#define DECL_SPARSEVALUES(VNAME, V)                                            \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_sparseValues##VNAME(              \
      StridedMemRefType<V, 1> *out, void *tensor);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_SPARSEVALUES)
#undef DECL_SPARSEVALUES

/// Tensor-storage method to obtain direct access to the positions array
/// for the given level.
#define DECL_SPARSEPOSITIONS(PNAME, P)                                         \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_sparsePositions##PNAME(           \
      StridedMemRefType<P, 1> *out, void *tensor, index_type lvl);
MLIR_SPARSETENSOR_FOREVERY_O(DECL_SPARSEPOSITIONS)
#undef DECL_SPARSEPOSITIONS

/// Tensor-storage method to obtain direct access to the coordinates array
/// for the given level.
#define DECL_SPARSECOORDINATES(CNAME, C)                                       \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_sparseCoordinates##CNAME(         \
      StridedMemRefType<C, 1> *out, void *tensor, index_type lvl);
MLIR_SPARSETENSOR_FOREVERY_O(DECL_SPARSECOORDINATES)
#undef DECL_SPARSECOORDINATES

/// Coordinate-scheme method for adding a new element.
#define DECL_ADDELT(VNAME, V)                                                  \
  MLIR_CRUNNERUTILS_EXPORT void *_mlir_ciface_addElt##VNAME(                   \
      void *lvlCOO, StridedMemRefType<V, 0> *vref,                             \
      StridedMemRefType<index_type, 1> *dimCoordsRef,                          \
      StridedMemRefType<index_type, 1> *dim2lvlRef);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_ADDELT)
#undef DECL_ADDELT

/// Coordinate-scheme method for getting the next element while iterating.
/// The `cref` argument uses the same coordinate-space as the `iter` (which
/// can be either dim- or lvl-coords, depending on context).
#define DECL_GETNEXT(VNAME, V)                                                 \
  MLIR_CRUNNERUTILS_EXPORT bool _mlir_ciface_getNext##VNAME(                   \
      void *iter, StridedMemRefType<index_type, 1> *cref,                      \
      StridedMemRefType<V, 0> *vref);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_GETNEXT)
#undef DECL_GETNEXT

/// Tensor-storage method to insert elements in lexicographical
/// level-coordinate order.
#define DECL_LEXINSERT(VNAME, V)                                               \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_lexInsert##VNAME(                 \
      void *tensor, StridedMemRefType<index_type, 1> *lvlCoordsRef,            \
      StridedMemRefType<V, 0> *vref);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_LEXINSERT)
#undef DECL_LEXINSERT

/// Tensor-storage method to insert using expansion.
#define DECL_EXPINSERT(VNAME, V)                                               \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_expInsert##VNAME(                 \
      void *tensor, StridedMemRefType<index_type, 1> *lvlCoordsRef,            \
      StridedMemRefType<V, 1> *vref, StridedMemRefType<bool, 1> *fref,         \
      StridedMemRefType<index_type, 1> *aref, index_type count);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_EXPINSERT)
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

/// Tensor-storage method to get the size of the given level.
MLIR_CRUNNERUTILS_EXPORT index_type sparseLvlSize(void *tensor, index_type l);

/// Tensor-storage method to get the size of the given dimension.
MLIR_CRUNNERUTILS_EXPORT index_type sparseDimSize(void *tensor, index_type d);

/// Tensor-storage method to finalize lexicographic insertions.
MLIR_CRUNNERUTILS_EXPORT void endInsert(void *tensor);

/// Coordinate-scheme method to write to file in extended FROSTT format.
#define DECL_OUTSPARSETENSOR(VNAME, V)                                         \
  MLIR_CRUNNERUTILS_EXPORT void outSparseTensor##VNAME(void *coo, void *dest,  \
                                                       bool sort);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_OUTSPARSETENSOR)
#undef DECL_OUTSPARSETENSOR

/// Releases the memory for the tensor-storage object.
MLIR_CRUNNERUTILS_EXPORT void delSparseTensor(void *tensor);

/// Releases the memory for the coordinate-scheme object.
#define DECL_DELCOO(VNAME, V)                                                  \
  MLIR_CRUNNERUTILS_EXPORT void delSparseTensorCOO##VNAME(void *coo);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_DELCOO)
#undef DECL_DELCOO

/// Releases the memory for an iterator object.
#define DECL_DELITER(VNAME, V)                                                 \
  MLIR_CRUNNERUTILS_EXPORT void delSparseTensorIterator##VNAME(void *iter);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_DELITER)
#undef DECL_DELITER

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
///   coordinates: a flat "nse * rank" array with coordinates for all
///            specified elements
///   perm:    the permutation of the levels in the storage
///   sparse:  the sparsity for the levels
///
/// For example, the sparse matrix
///     | 1.0 0.0 0.0 |
///     | 0.0 5.0 3.0 |
/// can be passed as
///      rank    = 2
///      nse     = 3
///      shape   = [2, 3]
///      values  = [1.0, 5.0, 3.0]
///      coordinates = [ 0, 0,  1, 1,  1, 2]
#define DECL_CONVERTTOMLIRSPARSETENSOR(VNAME, V)                               \
  MLIR_CRUNNERUTILS_EXPORT void *convertToMLIRSparseTensor##VNAME(             \
      uint64_t rank, uint64_t nse, uint64_t *dimSizes, V *values,              \
      uint64_t *dimCoordinates, uint64_t *dim2lvl, uint8_t *lvlTypes);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_CONVERTTOMLIRSPARSETENSOR)
#undef DECL_CONVERTTOMLIRSPARSETENSOR

/// Converts a sparse tensor to COO-flavored format expressed using
/// C-style data structures.  The expected output parameters are pointers
/// for these values:
///
///   rank:    rank of tensor
///   nse:     number of specified elements (usually the nonzeros)
///   shape:   array with size for each dimension
///   values:  a "nse" array with values for all specified elements
///   coordinates: a flat "nse * rank" array with coordinates for all
///            specified elements
///
/// The input is a pointer to `SparseTensorStorage<P, C, V>`, typically
/// returned from `convertToMLIRSparseTensor`.
#define DECL_CONVERTFROMMLIRSPARSETENSOR(VNAME, V)                             \
  MLIR_CRUNNERUTILS_EXPORT void convertFromMLIRSparseTensor##VNAME(            \
      void *tensor, uint64_t *pRank, uint64_t *pNse, uint64_t **pShape,        \
      V **pValues, uint64_t **pCoordinates);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_CONVERTFROMMLIRSPARSETENSOR)
#undef DECL_CONVERTFROMMLIRSPARSETENSOR

/// Creates a SparseTensorReader for reading a sparse tensor from a file with
/// the given file name. This opens the file and read the header meta data based
/// of the sparse tensor format derived from the suffix of the file name.
//
// FIXME: update `SparseTensorCodegenPass` to use
// `_mlir_ciface_createCheckedSparseTensorReader` instead.
MLIR_CRUNNERUTILS_EXPORT void *createSparseTensorReader(char *filename);

/// Constructs a new SparseTensorReader object, opens the file, reads the
/// header, and validates that the actual contents of the file match
/// the expected `dimShapeRef` and `valTp`.
MLIR_CRUNNERUTILS_EXPORT void *_mlir_ciface_createCheckedSparseTensorReader(
    char *filename, StridedMemRefType<index_type, 1> *dimShapeRef,
    PrimaryType valTp);

/// Constructs a new sparse-tensor storage object with the given encoding,
/// initializes it by reading all the elements from the file, and then
/// closes the file.
MLIR_CRUNNERUTILS_EXPORT void *_mlir_ciface_newSparseTensorFromReader(
    void *p, StridedMemRefType<index_type, 1> *lvlSizesRef,
    StridedMemRefType<DimLevelType, 1> *lvlTypesRef,
    StridedMemRefType<index_type, 1> *lvl2dimRef,
    StridedMemRefType<index_type, 1> *dim2lvlRef, OverheadType posTp,
    OverheadType crdTp, PrimaryType valTp);

/// Returns the rank of the sparse tensor being read.
MLIR_CRUNNERUTILS_EXPORT index_type getSparseTensorReaderRank(void *p);

/// Returns the is_symmetric bit for the sparse tensor being read.
MLIR_CRUNNERUTILS_EXPORT bool getSparseTensorReaderIsSymmetric(void *p);

/// Returns the number of stored elements for the sparse tensor being read.
MLIR_CRUNNERUTILS_EXPORT index_type getSparseTensorReaderNSE(void *p);

/// Returns the size of a dimension for the sparse tensor being read.
MLIR_CRUNNERUTILS_EXPORT index_type getSparseTensorReaderDimSize(void *p,
                                                                 index_type d);

/// SparseTensorReader method to copy the dimension-sizes into the
/// provided memref.
//
// FIXME: update `SparseTensorCodegenPass` to use
// `_mlir_ciface_getSparseTensorReaderDimSizes` instead.
MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_copySparseTensorReaderDimSizes(
    void *p, StridedMemRefType<index_type, 1> *dref);

/// SparseTensorReader method to obtain direct access to the
/// dimension-sizes array.
MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_getSparseTensorReaderDimSizes(
    StridedMemRefType<index_type, 1> *out, void *p);

/// Releases the SparseTensorReader. This also closes the file associated with
/// the reader.
MLIR_CRUNNERUTILS_EXPORT void delSparseTensorReader(void *p);

/// Returns the next element for the sparse tensor being read.
#define DECL_GETNEXT(VNAME, V)                                                 \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_getSparseTensorReaderNext##VNAME( \
      void *p, StridedMemRefType<index_type, 1> *dimCoordsRef,                 \
      StridedMemRefType<V, 0> *vref);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_GETNEXT)
#undef DECL_GETNEXT

/// Reads the sparse tensor, stores the coordinates and values to the given
/// memrefs. Returns a boolean value to indicate whether the COO elements are
/// sorted.
#define DECL_GETNEXT(VNAME, V, CNAME, C)                                       \
  MLIR_CRUNNERUTILS_EXPORT bool                                                \
      _mlir_ciface_getSparseTensorReaderRead##CNAME##VNAME(                    \
          void *p, StridedMemRefType<index_type, 1> *dim2lvlRef,               \
          StridedMemRefType<C, 1> *iref, StridedMemRefType<V, 1> *vref)        \
          MLIR_SPARSETENSOR_FOREVERY_V_O(DECL_GETNEXT)
#undef DECL_GETNEXT

using SparseTensorWriter = std::ostream;

/// Creates a SparseTensorWriter for outputing a sparse tensor to a file with
/// the given file name. When the file name is empty, std::cout is used.
//
// Only the extended FROSTT format is supported currently.
MLIR_CRUNNERUTILS_EXPORT void *createSparseTensorWriter(char *filename);

/// Finalizes the outputing of a sparse tensor to a file and releases the
/// SparseTensorWriter.
MLIR_CRUNNERUTILS_EXPORT void delSparseTensorWriter(void *p);

/// Outputs the sparse tensor dim-rank, nse, and dim-shape.
MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_outSparseTensorWriterMetaData(
    void *p, index_type dimRank, index_type nse,
    StridedMemRefType<index_type, 1> *dimSizesRef);

/// Outputs an element for the sparse tensor.
#define DECL_OUTNEXT(VNAME, V)                                                 \
  MLIR_CRUNNERUTILS_EXPORT void _mlir_ciface_outSparseTensorWriterNext##VNAME( \
      void *p, index_type dimRank,                                             \
      StridedMemRefType<index_type, 1> *dimCoordsRef,                          \
      StridedMemRefType<V, 0> *vref);
MLIR_SPARSETENSOR_FOREVERY_V(DECL_OUTNEXT)
#undef DECL_OUTNEXT

} // extern "C"

#endif // MLIR_EXECUTIONENGINE_SPARSETENSORRUNTIME_H
