//===- OpenACCUtilsCG.h - OpenACC Code Generation Utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utility functions for OpenACC code generation, including
// data layout and type-related utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCUTILSCG_H_
#define MLIR_DIALECT_OPENACC_OPENACCUTILSCG_H_

#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <optional>

namespace mlir {
class Operation;

namespace acc {

/// Get the data layout for an operation.
///
/// Attempts to get the data layout from the operation or its parent module.
/// If `allowDefault` is true (default), a default data layout may be
/// constructed when no explicit data layout spec is found.
///
/// \param op The operation to get the data layout for.
/// \param allowDefault If true, allow returning a default data layout.
/// \return The data layout if available, std::nullopt otherwise.
std::optional<DataLayout> getDataLayout(Operation *op,
                                        bool allowDefault = true);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCUTILSCG_H_
