//===- ACCToLLVMUtils.h - OpenACC to LLVM helpers ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENACCTOLLVM_ACCTOLLVMUTILS_H
#define MLIR_CONVERSION_OPENACCTOLLVM_ACCTOLLVMUTILS_H

#include "mlir/Dialect/OpenACC/OpenACCRuntimeUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir {
namespace acc {

/// Unfuses fused locations, returning the last sub-location.
Location unfuseLoc(Location loc);

/// Returns file:line:column location information when available.
std::optional<FileLineColLoc> getFileLineColLoc(Location loc,
                                                bool errorOnInvalidLocation);

/// Returns the enclosing function symbol name for \p op.
StringRef getParentFunctionName(Operation *op);

/// Returns the enclosing function symbol name for \p value's defining op.
StringRef getParentFunctionName(Value value);

/// Returns the first non-empty enclosing function name from \p values.
StringRef getParentFunctionName(ValueRange values);

/// Creates or reuses a module-internal null-terminated string global.
Value getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name,
                              StringRef value, ModuleOp module);

/// Returns a pointer to a constant global holding an ident_t for OpenACC
/// runtime calls.
Value createIdent(Location loc, StringRef functionName, OpBuilder &builder,
                  ModuleOp module, const ACCRuntimeCallConfig &config);

} // namespace acc
} // namespace mlir

#endif // MLIR_CONVERSION_OPENACCTOLLVM_ACCTOLLVMUTILS_H
