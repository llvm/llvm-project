//===-- mlir-c/AffineExpr.h - C API for MLIR Affine Expressions ---*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_AFFINEEXPR_H
#define MLIR_C_AFFINEEXPR_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Opaque type declarations.
//
// Types are exposed to C bindings as structs containing opaque pointers. They
// are not supposed to be inspected from C. This allows the underlying
// representation to change without affecting the API users. The use of structs
// instead of typedefs enables some type safety as structs are not implicitly
// convertible to each other.
//
// Instances of these types may or may not own the underlying object. The
// ownership semantics is defined by how an instance of the type was obtained.
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirAffineExpr, const void);

#undef DEFINE_C_API_STRUCT

/// Gets the context that owns the affine expression.
MLIR_CAPI_EXPORTED MlirContext
mlirAffineExprGetContext(MlirAffineExpr affineExpr);

/// Returns `true` if the two affine expressions are equal.
MLIR_CAPI_EXPORTED bool mlirAffineExprEqual(MlirAffineExpr lhs,
                                            MlirAffineExpr rhs);

/// Returns `true` if the given affine expression is a null expression. Note
/// constant zero is not a null expression.
inline static bool mlirAffineExprIsNull(MlirAffineExpr affineExpr) {
  return affineExpr.ptr == NULL;
}

/// Prints an affine expression by sending chunks of the string representation
/// and forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
MLIR_CAPI_EXPORTED void mlirAffineExprPrint(MlirAffineExpr affineExpr,
                                            MlirStringCallback callback,
                                            void *userData);

/// Prints the affine expression to the standard error stream.
MLIR_CAPI_EXPORTED void mlirAffineExprDump(MlirAffineExpr affineExpr);

/// Checks whether the given affine expression is made out of only symbols and
/// constants.
MLIR_CAPI_EXPORTED bool
mlirAffineExprIsSymbolicOrConstant(MlirAffineExpr affineExpr);

/// Checks whether the given affine expression is a pure affine expression, i.e.
/// mul, floordiv, ceildic, and mod is only allowed w.r.t constants.
MLIR_CAPI_EXPORTED bool mlirAffineExprIsPureAffine(MlirAffineExpr affineExpr);

/// Returns the greatest known integral divisor of this affine expression. The
/// result is always positive.
MLIR_CAPI_EXPORTED int64_t
mlirAffineExprGetLargestKnownDivisor(MlirAffineExpr affineExpr);

/// Checks whether the given affine expression is a multiple of 'factor'.
MLIR_CAPI_EXPORTED bool mlirAffineExprIsMultipleOf(MlirAffineExpr affineExpr,
                                                   int64_t factor);

/// Checks whether the given affine expression involves AffineDimExpr
/// 'position'.
MLIR_CAPI_EXPORTED bool mlirAffineExprIsFunctionOfDim(MlirAffineExpr affineExpr,
                                                      intptr_t position);

//===----------------------------------------------------------------------===//
// Affine Dimension Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is a dimension expression.
MLIR_CAPI_EXPORTED bool mlirAffineExprIsADim(MlirAffineExpr affineExpr);

/// Creates an affine dimension expression with 'position' in the context.
MLIR_CAPI_EXPORTED MlirAffineExpr mlirAffineDimExprGet(MlirContext ctx,
                                                       intptr_t position);

/// Returns the position of the given affine dimension expression.
MLIR_CAPI_EXPORTED intptr_t
mlirAffineDimExprGetPosition(MlirAffineExpr affineExpr);

//===----------------------------------------------------------------------===//
// Affine Symbol Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is a symbol expression.
MLIR_CAPI_EXPORTED bool mlirAffineExprIsASymbol(MlirAffineExpr affineExpr);

/// Creates an affine symbol expression with 'position' in the context.
MLIR_CAPI_EXPORTED MlirAffineExpr mlirAffineSymbolExprGet(MlirContext ctx,
                                                          intptr_t position);

/// Returns the position of the given affine symbol expression.
MLIR_CAPI_EXPORTED intptr_t
mlirAffineSymbolExprGetPosition(MlirAffineExpr affineExpr);

//===----------------------------------------------------------------------===//
// Affine Constant Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is a constant expression.
MLIR_CAPI_EXPORTED bool mlirAffineExprIsAConstant(MlirAffineExpr affineExpr);

/// Creates an affine constant expression with 'constant' in the context.
MLIR_CAPI_EXPORTED MlirAffineExpr mlirAffineConstantExprGet(MlirContext ctx,
                                                            int64_t constant);

/// Returns the value of the given affine constant expression.
MLIR_CAPI_EXPORTED int64_t
mlirAffineConstantExprGetValue(MlirAffineExpr affineExpr);

//===----------------------------------------------------------------------===//
// Affine Add Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is an add expression.
MLIR_CAPI_EXPORTED bool mlirAffineExprIsAAdd(MlirAffineExpr affineExpr);

/// Creates an affine add expression with 'lhs' and 'rhs'.
MLIR_CAPI_EXPORTED MlirAffineExpr mlirAffineAddExprGet(MlirAffineExpr lhs,
                                                       MlirAffineExpr rhs);

//===----------------------------------------------------------------------===//
// Affine Mul Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is an mul expression.
MLIR_CAPI_EXPORTED bool mlirAffineExprIsAMul(MlirAffineExpr affineExpr);

/// Creates an affine mul expression with 'lhs' and 'rhs'.
MLIR_CAPI_EXPORTED MlirAffineExpr mlirAffineMulExprGet(MlirAffineExpr lhs,
                                                       MlirAffineExpr rhs);

//===----------------------------------------------------------------------===//
// Affine Mod Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is an mod expression.
MLIR_CAPI_EXPORTED bool mlirAffineExprIsAMod(MlirAffineExpr affineExpr);

/// Creates an affine mod expression with 'lhs' and 'rhs'.
MLIR_CAPI_EXPORTED MlirAffineExpr mlirAffineModExprGet(MlirAffineExpr lhs,
                                                       MlirAffineExpr rhs);

//===----------------------------------------------------------------------===//
// Affine FloorDiv Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is an floordiv expression.
MLIR_CAPI_EXPORTED bool mlirAffineExprIsAFloorDiv(MlirAffineExpr affineExpr);

/// Creates an affine floordiv expression with 'lhs' and 'rhs'.
MLIR_CAPI_EXPORTED MlirAffineExpr mlirAffineFloorDivExprGet(MlirAffineExpr lhs,
                                                            MlirAffineExpr rhs);

//===----------------------------------------------------------------------===//
// Affine CeilDiv Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is an ceildiv expression.
MLIR_CAPI_EXPORTED bool mlirAffineExprIsACeilDiv(MlirAffineExpr affineExpr);

/// Creates an affine ceildiv expression with 'lhs' and 'rhs'.
MLIR_CAPI_EXPORTED MlirAffineExpr mlirAffineCeilDivExprGet(MlirAffineExpr lhs,
                                                           MlirAffineExpr rhs);

//===----------------------------------------------------------------------===//
// Affine Binary Operation Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is binary.
MLIR_CAPI_EXPORTED bool mlirAffineExprIsABinary(MlirAffineExpr affineExpr);

/// Returns the left hand side affine expression of the given affine binary
/// operation expression.
MLIR_CAPI_EXPORTED MlirAffineExpr
mlirAffineBinaryOpExprGetLHS(MlirAffineExpr affineExpr);

/// Returns the right hand side affine expression of the given affine binary
/// operation expression.
MLIR_CAPI_EXPORTED MlirAffineExpr
mlirAffineBinaryOpExprGetRHS(MlirAffineExpr affineExpr);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_AFFINEEXPR_H
