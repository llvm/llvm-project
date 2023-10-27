//===- MapRef.cpp - A dim2lvl/lvl2dim map reference wrapper ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/SparseTensor/MapRef.h"

mlir::sparse_tensor::MapRef::MapRef(uint64_t d, uint64_t l, const uint64_t *d2l,
                                    const uint64_t *l2d)
    : dimRank(d), lvlRank(l), dim2lvl(d2l), lvl2dim(l2d),
      isPermutation(isPermutationMap()) {
  if (isPermutation) {
    for (uint64_t i = 0; i < dimRank; i++)
      assert(lvl2dim[dim2lvl[i]] == i);
  }
}

bool mlir::sparse_tensor::MapRef::isPermutationMap() const {
  if (dimRank != lvlRank)
    return false;
  std::vector<bool> seen(dimRank, false);
  for (uint64_t i = 0; i < dimRank; i++) {
    const uint64_t j = dim2lvl[i];
    if (j >= dimRank || seen[j])
      return false;
    seen[j] = true;
  }
  return true;
}
