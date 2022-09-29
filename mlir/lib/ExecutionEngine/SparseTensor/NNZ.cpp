//===- NNZ.cpp - NNZ-statistics for direct sparse2sparse conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains method definitions for `SparseTensorNNZ`.
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

SparseTensorNNZ::SparseTensorNNZ(const std::vector<uint64_t> &dimSizes,
                                 const std::vector<DimLevelType> &sparsity)
    : dimSizes(dimSizes), dimTypes(sparsity), nnz(getRank()) {
  assert(dimSizes.size() == dimTypes.size() && "Rank mismatch");
  bool uncompressed = true;
  (void)uncompressed;
  uint64_t sz = 1; // the product of all `dimSizes` strictly less than `r`.
  for (uint64_t rank = getRank(), r = 0; r < rank; r++) {
    switch (dimTypes[r]) {
    case DimLevelType::kCompressed:
      assert(uncompressed &&
             "Multiple compressed layers not currently supported");
      uncompressed = false;
      nnz[r].resize(sz, 0); // Both allocate and zero-initialize.
      break;
    case DimLevelType::kDense:
      assert(uncompressed && "Dense after compressed not currently supported");
      break;
    case DimLevelType::kSingleton:
      // Singleton after Compressed causes no problems for allocating
      // `nnz` nor for the yieldPos loop.  This remains true even
      // when adding support for multiple compressed dimensions or
      // for dense-after-compressed.
      break;
    default:
      MLIR_SPARSETENSOR_FATAL("unsupported dimension level type");
    }
    sz = detail::checkedMul(sz, dimSizes[r]);
  }
}

void SparseTensorNNZ::forallIndices(uint64_t stopDim,
                                    SparseTensorNNZ::NNZConsumer yield) const {
  assert(stopDim < getRank() && "Stopping-dimension is out of bounds");
  assert(dimTypes[stopDim] == DimLevelType::kCompressed &&
         "Cannot look up non-compressed dimensions");
  forallIndices(yield, stopDim, 0, 0);
}

void SparseTensorNNZ::add(const std::vector<uint64_t> &ind) {
  uint64_t parentPos = 0;
  for (uint64_t rank = getRank(), r = 0; r < rank; r++) {
    if (dimTypes[r] == DimLevelType::kCompressed)
      nnz[r][parentPos]++;
    parentPos = parentPos * dimSizes[r] + ind[r];
  }
}

void SparseTensorNNZ::forallIndices(SparseTensorNNZ::NNZConsumer yield,
                                    uint64_t stopDim, uint64_t parentPos,
                                    uint64_t d) const {
  assert(d <= stopDim);
  if (d == stopDim) {
    assert(parentPos < nnz[d].size() && "Cursor is out of range");
    yield(nnz[d][parentPos]);
  } else {
    const uint64_t sz = dimSizes[d];
    const uint64_t pstart = parentPos * sz;
    for (uint64_t i = 0; i < sz; i++)
      forallIndices(yield, stopDim, pstart + i, d + 1);
  }
}

#endif // MLIR_SPARSETENSOR_DEFINE_FUNCTIONS
