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

#include "mlir/IR/Value.h"
#include "llvm/Support/TypeSize.h"
#include <optional>
#include <utility>

namespace mlir {
class DataLayout;
class Location;
class ModuleOp;
class OpBuilder;
class Type;
class Value;

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
/// \p ty itself is sized dialect-agnostically; when \p support is provided it
/// sizes aggregate element types, so that nested dialect types are handled.
/// Callers that hold an OpenACCSupport should therefore ask it directly -
/// OpenACCSupport::getTypeSizeAndAlignment covers dialect types and falls back
/// to this utility - and call this utility directly only for a type this
/// utility is expected to know.
///
/// When \p var is provided, MappableType sizes the mapped object rather than
/// the type's storage alone.
///
/// Returns std::nullopt when the size is not statically computable or the type
/// is not supported.
std::optional<TypeSizeAndAlignment>
getTypeSizeAndAlignment(Type ty, ModuleOp module, const DataLayout &dl,
                        OpenACCSupport *support = nullptr, Value var = {});

/// Same as above, obtaining \p dl from \p module via getDataLayout.
std::optional<TypeSizeAndAlignment>
getTypeSizeAndAlignment(Type ty, ModuleOp module,
                        OpenACCSupport *support = nullptr, Value var = {});

/// Cast \p value to \p resultType via PointerLikeType::genCast when needed.
/// Returns \p value unchanged if types already match. Emits an error and
/// returns \p value if no cast can be generated.
Value castPointerLikeTypeIfNeeded(OpBuilder &builder, Location loc, Value value,
                                  Type resultType);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCUTILSTYPE_H_
