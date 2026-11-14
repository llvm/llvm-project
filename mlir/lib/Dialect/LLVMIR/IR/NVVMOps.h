//===- NVVMOps.h - NVVM operation implementation helpers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the helpers that the source files of several op
// partitions of the NVVM dialect use. Helpers used by one op partition are
// defined in its source file only.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_LLVMIR_IR_NVVMOPS_H
#define MLIR_LIB_DIALECT_LLVMIR_IR_NVVMOPS_H

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Casting.h"

namespace mlir::NVVM {
/// The ID of no intrinsic, used in the intrinsic tables of the ops.
constexpr unsigned notIntrinsic = llvm::Intrinsic::not_intrinsic;

/// Returns whether `ptr` is a pointer into the address space `targetAS`.
inline bool isPtrInAddrSpace(mlir::Value ptr, NVVMMemorySpace targetAS) {
  auto ptrTy = llvm::cast<LLVM::LLVMPointerType>(ptr.getType());
  return ptrTy.getAddressSpace() == static_cast<unsigned>(targetAS);
}

/// Returns whether `ptr` is a pointer into the shared memory of the CTA.
inline bool isPtrInSharedCTASpace(mlir::Value ptr) {
  return isPtrInAddrSpace(ptr, NVVMMemorySpace::Shared);
}
} // namespace mlir::NVVM

#endif // MLIR_LIB_DIALECT_LLVMIR_IR_NVVMOPS_H
