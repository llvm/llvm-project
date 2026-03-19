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

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <optional>

namespace mlir {
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

/// Build an `acc.compute_region` operation by cloning a source region.
///
/// Creates a new `acc.compute_region` with the given launch arguments and
/// origin string, then clones the operations from `regionToClone` into its
/// body. Multi-block regions are wrapped with `scf.execute_region`.
///
/// The `mapping` is used and updated during cloning, allowing callers to
/// track value correspondences. Optional `output`, `kernelFuncName`,
/// `kernelModuleName`, and `stream` arguments are forwarded to the op.
///
/// When `inputArgsToMap` is non-empty, it is used as the key set for the
/// clone mapping (instead of `inputArgs`). Use this when cloning a region
/// that references one set of values (e.g. the source function's args) while
/// the op's operands are another set (e.g. the current block's args).
/// `inputArgsToMap` must have the same size as `inputArgs` when provided.
ComputeRegionOp buildComputeRegion(Location loc, ValueRange launchArgs,
                                   ValueRange inputArgs, llvm::StringRef origin,
                                   Region &regionToClone,
                                   RewriterBase &rewriter, IRMapping &mapping,
                                   ValueRange output = {},
                                   FlatSymbolRefAttr kernelFuncName = {},
                                   FlatSymbolRefAttr kernelModuleName = {},
                                   Value stream = {},
                                   ValueRange inputArgsToMap = {});

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCUTILSCG_H_
