//===- StorageBase.cpp - TACO-flavored sparse tensor representation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains method definitions for `SparseTensorStorageBase`.
// In particular we want to ensure that the default implementations of
// the "partial method specialization" trick aren't inline (since there's
// no benefit).
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/SparseTensor/Storage.h"

using namespace mlir::sparse_tensor;

SparseTensorStorageBase::SparseTensorStorageBase( // NOLINT
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const DimLevelType *lvlTypes,
    const uint64_t *dim2lvl, const uint64_t *lvl2dim)
    : dimSizes(dimSizes, dimSizes + dimRank),
      lvlSizes(lvlSizes, lvlSizes + lvlRank),
      lvlTypes(lvlTypes, lvlTypes + lvlRank),
      dim2lvlVec(dim2lvl, dim2lvl + lvlRank),
      lvl2dimVec(lvl2dim, lvl2dim + dimRank),
      map(dimRank, lvlRank, dim2lvlVec.data(), lvl2dimVec.data()) {
  assert(dimSizes && lvlSizes && lvlTypes && dim2lvl && lvl2dim);
  // Validate dim-indexed parameters.
  assert(dimRank > 0 && "Trivial shape is unsupported");
  for (uint64_t d = 0; d < dimRank; ++d)
    assert(dimSizes[d] > 0 && "Dimension size zero has trivial storage");
  // Validate lvl-indexed parameters.
  assert(lvlRank > 0 && "Trivial shape is unsupported");
  for (uint64_t l = 0; l < lvlRank; ++l) {
    assert(lvlSizes[l] > 0 && "Level size zero has trivial storage");
    const auto dlt = lvlTypes[l];
    if (!(isDenseDLT(dlt) || isCompressedDLT(dlt) || isSingletonDLT(dlt))) {
      MLIR_SPARSETENSOR_FATAL("unsupported level type: %d\n",
                              static_cast<uint8_t>(dlt));
    }
  }
}

// Helper macro for wrong "partial method specialization" errors.
#define FATAL_PIV(NAME)                                                        \
  MLIR_SPARSETENSOR_FATAL("<P,I,V> type mismatch for: " #NAME);

#define IMPL_GETPOSITIONS(PNAME, P)                                            \
  void SparseTensorStorageBase::getPositions(std::vector<P> **, uint64_t) {    \
    FATAL_PIV("getPositions" #PNAME);                                          \
  }
MLIR_SPARSETENSOR_FOREVERY_FIXED_O(IMPL_GETPOSITIONS)
#undef IMPL_GETPOSITIONS

#define IMPL_GETCOORDINATES(CNAME, C)                                          \
  void SparseTensorStorageBase::getCoordinates(std::vector<C> **, uint64_t) {  \
    FATAL_PIV("getCoordinates" #CNAME);                                        \
  }
MLIR_SPARSETENSOR_FOREVERY_FIXED_O(IMPL_GETCOORDINATES)
#undef IMPL_GETCOORDINATES

#define IMPL_GETVALUES(VNAME, V)                                               \
  void SparseTensorStorageBase::getValues(std::vector<V> **) {                 \
    FATAL_PIV("getValues" #VNAME);                                             \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_GETVALUES)
#undef IMPL_GETVALUES

#define IMPL_FORWARDINGINSERT(VNAME, V)                                        \
  void SparseTensorStorageBase::forwardingInsert(const uint64_t *, V) {        \
    FATAL_PIV("forwardingInsert" #VNAME);                                      \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_FORWARDINGINSERT)
#undef IMPL_FORWARDINGINSERT

#define IMPL_LEXINSERT(VNAME, V)                                               \
  void SparseTensorStorageBase::lexInsert(const uint64_t *, V) {               \
    FATAL_PIV("lexInsert" #VNAME);                                             \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_LEXINSERT)
#undef IMPL_LEXINSERT

#define IMPL_EXPINSERT(VNAME, V)                                               \
  void SparseTensorStorageBase::expInsert(uint64_t *, V *, bool *, uint64_t *, \
                                          uint64_t, uint64_t) {                \
    FATAL_PIV("expInsert" #VNAME);                                             \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_EXPINSERT)
#undef IMPL_EXPINSERT

#undef FATAL_PIV
