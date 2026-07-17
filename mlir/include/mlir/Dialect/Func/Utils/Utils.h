//===- Utils.h - General Func transformation utilities ----*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// the Func dialect. These are not passes by themselves but are used
// either by passes, optimization sequences, or in turn by other transformation
// utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FUNC_UTILS_H
#define MLIR_DIALECT_FUNC_UTILS_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include <string>

namespace mlir {

class ModuleOp;

namespace func {

class FuncOp;
class CallOp;

/// Creates a new function operation with the same name as the original
/// function operation, but with the arguments mapped according to
/// the `oldArgToNewArg` and `oldResToNewRes`.
/// The `funcOp` operation must have exactly one block.
/// Returns the new function operation or failure if `funcOp` doesn't
/// have exactly one block.
/// Note: the method asserts that the `oldArgToNewArg` and `oldResToNewRes`
/// maps the whole function arguments and results.
mlir::FailureOr<mlir::func::FuncOp> replaceFuncWithNewMapping(
    mlir::RewriterBase &rewriter, mlir::func::FuncOp funcOp,
    ArrayRef<int> oldArgIdxToNewArgIdx, ArrayRef<int> oldResIdxToNewResIdx);
/// Creates a new call operation with the values as the original
/// call operation, but with the arguments mapped according to
/// the `oldArgToNewArg` and `oldResToNewRes`.
/// Note: the method asserts that the `oldArgToNewArg` and `oldResToNewRes`
/// maps the whole call operation arguments and results.
mlir::func::CallOp replaceCallOpWithNewMapping(
    mlir::RewriterBase &rewriter, mlir::func::CallOp callOp,
    ArrayRef<int> oldArgIdxToNewArgIdx, ArrayRef<int> oldResIdxToNewResIdx);

/// This utility function examines all call operations within the given
/// `moduleOp` that target the specified `funcOp`. It identifies duplicate
/// operands in the call operations, creates mappings to deduplicate them, and
/// then applies the transformation to both the function and its call sites. For
/// now, it only supports one call operation for the function operation. The
/// function returns a pair containing the new funcOp and the new callOp. Note:
/// after the transformation, the original funcOp and callOp will be erased.
mlir::FailureOr<std::pair<mlir::func::FuncOp, mlir::func::CallOp>>
deduplicateArgsOfFuncOp(mlir::RewriterBase &rewriter, mlir::func::FuncOp funcOp,
                        mlir::ModuleOp moduleOp);

/// Look up a FuncOp with signature `resultTypes`(`paramTypes`)` and name
/// `name`. Return a failure if the FuncOp is found but with a different
/// signature.
FailureOr<FuncOp> lookupFnDecl(SymbolOpInterface symTable, StringRef name,
                               FunctionType funcT,
                               SymbolTableCollection *symbolTables = nullptr);

/// Create a FuncOp decl and insert it into `symTable` operation. If
/// `symbolTables` is provided, then the decl will be inserted into the
/// SymbolTableCollection.
FuncOp createFnDecl(OpBuilder &b, SymbolOpInterface symTable, StringRef name,
                    FunctionType funcT, bool setPrivate,
                    SymbolTableCollection *symbolTables = nullptr);

/// Helper function to look up or create the symbol for a runtime library
/// function with the given parameter types. Returns an int64_t, unless a
/// different result type is specified.
FailureOr<FuncOp>
lookupOrCreateFnDecl(OpBuilder &b, SymbolOpInterface symTable, StringRef name,
                     TypeRange paramTypes,
                     SymbolTableCollection *symbolTables = nullptr,
                     Type resultType = {});

} // namespace func
} // namespace mlir

#endif // MLIR_DIALECT_FUNC_UTILS_H
