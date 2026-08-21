//===- OpenACCUtilsType.h - OpenACC Type Utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines type utilities for OpenACC.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCUTILSTYPE_H_
#define MLIR_DIALECT_OPENACC_OPENACCUTILSTYPE_H_

#include "llvm/Support/TypeSize.h"
#include <optional>
#include <utility>

namespace mlir {
class DataLayout;
class ModuleOp;
class Type;

namespace acc {

class OpenACCSupport;

using TypeSizeAndAlignment = std::pair<llvm::TypeSize, llvm::TypeSize>;

/// Returns the size and ABI alignment in bytes.
///
/// For aggregate structures and arrays, padding between members or elements is
/// not taken into account. The result is a close estimate suitable for early
/// OpenACC layout decisions, but not a complete ABI guarantee. For final size
/// computations, use LLVM materialized types.
///
/// When \p support is provided, aggregate element types are sized by recursing
/// through \p support so dialect-specific implementations can handle nested
/// types.
///
/// Returns std::nullopt when the size is not statically computable or the type
/// is not supported.
std::optional<TypeSizeAndAlignment>
getTypeSizeAndAlignment(Type ty, ModuleOp module, const DataLayout &dl,
                        OpenACCSupport *support = nullptr);

/// Same as above, obtaining \p dl from \p module via getDataLayout.
std::optional<TypeSizeAndAlignment>
getTypeSizeAndAlignment(Type ty, ModuleOp module,
                        OpenACCSupport *support = nullptr);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCUTILSTYPE_H_
