//===- PermutationRef.cpp - Permutation reference wrapper -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/SparseTensor/PermutationRef.h"

#include <cassert>
#include <cinttypes>
#include <vector>

bool mlir::sparse_tensor::detail::isPermutation(uint64_t size,
                                                const uint64_t *perm) {
  assert(perm && "Got nullptr for permutation");
  // TODO: If we ever depend on LLVMSupport, then use `llvm::BitVector` instead.
  std::vector<bool> seen(size, false);
  for (uint64_t i = 0; i < size; ++i) {
    const uint64_t j = perm[i];
    if (j >= size || seen[j])
      return false;
    seen[j] = true;
  }
  for (uint64_t i = 0; i < size; ++i)
    if (!seen[i])
      return false;
  return true;
}

std::vector<uint64_t>
mlir::sparse_tensor::detail::PermutationRef::inverse() const {
  std::vector<uint64_t> out(permSize);
  for (uint64_t i = 0; i < permSize; ++i)
    out[perm[i]] = i;
  return out;
}
