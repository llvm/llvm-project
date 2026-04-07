//===- Utils.h - General AMDGPU Enums utilities -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_AMDGPU_UTILS_AMDGPU_ENUMS_H_
#define MLIR_DIALECT_AMDGPU_UTILS_AMDGPU_ENUMS_H_

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUEnums.h.inc"
#include "llvm/ADT/STLExtras.h"

namespace mlir::amdgpu {

inline int32_t getGlobalPrefetchLLVMEncoding(amdgpu::LoadTemporalHint hint,
                                             amdgpu::Scope scope,
                                             bool isSpeculative) {
  int32_t immArg = static_cast<int32_t>(hint);

  // Note that only RT and HT can operate in both speculative and
  // non-speculative modes. The other variants (NT_RT, RT_NT, NT_HT, etc.)
  // operate only in the speculative mode and, therefore, do not require
  // toggling the least significant bit for mode changes
  // Temporal hint is encoded in lower bits - i.e. [2:0]
  if (llvm::is_contained({LoadTemporalHint::RT, LoadTemporalHint::HT}, hint))
    immArg = isSpeculative ? immArg : immArg | 1;

  // Prefetch scope level is encoded in upper bits - i.e., [4:3]
  return static_cast<int32_t>(scope) << 3 | immArg;
}

} // namespace mlir::amdgpu

#endif
