//===- NVVMAttributes.h - NVVM enums and attributes -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVVM dialect enums and attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMATTRIBUTES_H_
#define MLIR_DIALECT_LLVMIR_NVVMATTRIBUTES_H_

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"
#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "mlir/Dialect/LLVMIR/NVVMOpsEnums.h.inc"

namespace mlir {
namespace NVVM {
/// Utility functions to compare NVVMMemorySpace with unsigned values.
inline bool operator==(unsigned as, NVVMMemorySpace memSpace) {
  return as == static_cast<unsigned>(memSpace);
}
inline bool operator==(NVVMMemorySpace memSpace, unsigned as) {
  return static_cast<unsigned>(memSpace) == as;
}
inline bool operator!=(unsigned as, NVVMMemorySpace memSpace) {
  return as != static_cast<unsigned>(memSpace);
}
inline bool operator!=(NVVMMemorySpace memSpace, unsigned as) {
  return static_cast<unsigned>(memSpace) != as;
}
} // namespace NVVM
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMOpsAttributes.h.inc"

#endif // MLIR_DIALECT_LLVMIR_NVVMATTRIBUTES_H_
