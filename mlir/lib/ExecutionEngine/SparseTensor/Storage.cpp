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
// no benefit).  Though this also helps ensure that we avoid weak-vtables:
// <https://llvm.org/docs/CodingStandards.html#provide-a-virtual-method-anchor-for-classes-in-headers>
//
// This file is part of the lightweight runtime support library for sparse
// tensor manipulations.  The functionality of the support library is meant
// to simplify benchmarking, testing, and debugging MLIR code operating on
// sparse tensors.  However, the provided functionality is **not** part of
// core MLIR itself.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/SparseTensor/Storage.h"

#ifdef MLIR_SPARSETENSOR_DEFINE_FUNCTIONS // We are building this library

using namespace mlir::sparse_tensor;

SparseTensorStorageBase::SparseTensorStorageBase(
    const std::vector<uint64_t> &dimSizes, const uint64_t *perm,
    const DimLevelType *sparsity)
    : dimSizes(dimSizes), rev(getRank()),
      dimTypes(sparsity, sparsity + getRank()) {
  assert(perm && sparsity);
  const uint64_t rank = getRank();
  // Validate parameters.
  assert(rank > 0 && "Trivial shape is unsupported");
  for (uint64_t r = 0; r < rank; r++) {
    assert(dimSizes[r] > 0 && "Dimension size zero has trivial storage");
    assert((isDenseDim(r) || isCompressedDim(r) || isSingletonDim(r)) &&
           "Unsupported DimLevelType");
  }
  // Construct the "reverse" (i.e., inverse) permutation.
  for (uint64_t r = 0; r < rank; r++)
    rev[perm[r]] = r;
}

// Helper macro for generating error messages when some
// `SparseTensorStorage<P,I,V>` is cast to `SparseTensorStorageBase`
// and then the wrong "partial method specialization" is called.
#define FATAL_PIV(NAME)                                                        \
  MLIR_SPARSETENSOR_FATAL("<P,I,V> type mismatch for: " #NAME);

#define IMPL_NEWENUMERATOR(VNAME, V)                                           \
  void SparseTensorStorageBase::newEnumerator(                                 \
      SparseTensorEnumeratorBase<V> **, uint64_t, const uint64_t *) const {    \
    FATAL_PIV("newEnumerator" #VNAME);                                         \
  }
FOREVERY_V(IMPL_NEWENUMERATOR)
#undef IMPL_NEWENUMERATOR

#define IMPL_GETPOINTERS(PNAME, P)                                             \
  void SparseTensorStorageBase::getPointers(std::vector<P> **, uint64_t) {     \
    FATAL_PIV("getPointers" #PNAME);                                           \
  }
FOREVERY_FIXED_O(IMPL_GETPOINTERS)
#undef IMPL_GETPOINTERS

#define IMPL_GETINDICES(INAME, I)                                              \
  void SparseTensorStorageBase::getIndices(std::vector<I> **, uint64_t) {      \
    FATAL_PIV("getIndices" #INAME);                                            \
  }
FOREVERY_FIXED_O(IMPL_GETINDICES)
#undef IMPL_GETINDICES

#define IMPL_GETVALUES(VNAME, V)                                               \
  void SparseTensorStorageBase::getValues(std::vector<V> **) {                 \
    FATAL_PIV("getValues" #VNAME);                                             \
  }
FOREVERY_V(IMPL_GETVALUES)
#undef IMPL_GETVALUES

#define IMPL_LEXINSERT(VNAME, V)                                               \
  void SparseTensorStorageBase::lexInsert(const uint64_t *, V) {               \
    FATAL_PIV("lexInsert" #VNAME);                                             \
  }
FOREVERY_V(IMPL_LEXINSERT)
#undef IMPL_LEXINSERT

#define IMPL_EXPINSERT(VNAME, V)                                               \
  void SparseTensorStorageBase::expInsert(uint64_t *, V *, bool *, uint64_t *, \
                                          uint64_t) {                          \
    FATAL_PIV("expInsert" #VNAME);                                             \
  }
FOREVERY_V(IMPL_EXPINSERT)
#undef IMPL_EXPINSERT

#undef FATAL_PIV

#endif // MLIR_SPARSETENSOR_DEFINE_FUNCTIONS
