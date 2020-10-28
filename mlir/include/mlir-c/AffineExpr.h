/*===-- mlir-c/AffineExpr.h - C API for MLIR Affine Expressions ---*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef MLIR_C_AFFINEEXPR_H
#define MLIR_C_AFFINEEXPR_H

#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

DEFINE_C_API_STRUCT(MlirAffineExpr, const void);

/** Gets the context that owns the affine expression. */
MlirContext mlirAffineExprGetContext(MlirAffineExpr affineExpr);

/** Prints an affine expression by sending chunks of the string representation
 * and forwarding `userData to `callback`. Note that the callback may be called
 * several times with consecutive chunks of the string. */
void mlirAffineExprPrint(MlirAffineExpr affineExpr, MlirStringCallback callback,
                         void *userData);

/** Prints the affine expression to the standard error stream. */
void mlirAffineExprDump(MlirAffineExpr affineExpr);

/** Checks whether the given affine expression is made out of only symbols and
 * constants. */
int mlirAffineExprIsSymbolicOrConstant(MlirAffineExpr affineExpr);

/** Checks whether the given affine expression is a pure affine expression, i.e.
 * mul, floordiv, ceildic, and mod is only allowed w.r.t constants. */
int mlirAffineExprIsPureAffine(MlirAffineExpr affineExpr);

/** Returns the greatest known integral divisor of this affine expression. The
 * result is always positive. */
int64_t mlirAffineExprGetLargestKnownDivisor(MlirAffineExpr affineExpr);

/** Checks whether the given affine expression is a multiple of 'factor'. */
int mlirAffineExprIsMultipleOf(MlirAffineExpr affineExpr, int64_t factor);

/** Checks whether the given affine expression involves AffineDimExpr
 * 'position'. */
int mlirAffineExprIsFunctionOfDim(MlirAffineExpr affineExpr, intptr_t position);

/*============================================================================*/
/* Affine Dimension Expression.                                               */
/*============================================================================*/

/** Creates an affine dimension expression with 'position' in the context. */
MlirAffineExpr mlirAffineDimExprGet(MlirContext ctx, intptr_t position);

/** Returns the position of the given affine dimension expression. */
intptr_t mlirAffineDimExprGetPosition(MlirAffineExpr affineExpr);

/*============================================================================*/
/* Affine Symbol Expression.                                                  */
/*============================================================================*/

/** Creates an affine symbol expression with 'position' in the context. */
MlirAffineExpr mlirAffineSymbolExprGet(MlirContext ctx, intptr_t position);

/** Returns the position of the given affine symbol expression. */
intptr_t mlirAffineSymbolExprGetPosition(MlirAffineExpr affineExpr);

/*============================================================================*/
/* Affine Constant Expression.                                                */
/*============================================================================*/

/** Creates an affine constant expression with 'constant' in the context. */
MlirAffineExpr mlirAffineConstantExprGet(MlirContext ctx, int64_t constant);

/** Returns the value of the given affine constant expression. */
int64_t mlirAffineConstantExprGetValue(MlirAffineExpr affineExpr);

/*============================================================================*/
/* Affine Add Expression.                                                     */
/*============================================================================*/

/** Checks whether the given affine expression is an add expression. */
int mlirAffineExprIsAAdd(MlirAffineExpr affineExpr);

/** Creates an affine add expression with 'lhs' and 'rhs'. */
MlirAffineExpr mlirAffineAddExprGet(MlirAffineExpr lhs, MlirAffineExpr rhs);

/*============================================================================*/
/* Affine Mul Expression.                                                     */
/*============================================================================*/

/** Checks whether the given affine expression is an mul expression. */
int mlirAffineExprIsAMul(MlirAffineExpr affineExpr);

/** Creates an affine mul expression with 'lhs' and 'rhs'. */
MlirAffineExpr mlirAffineMulExprGet(MlirAffineExpr lhs, MlirAffineExpr rhs);

/*============================================================================*/
/* Affine Mod Expression.                                                     */
/*============================================================================*/

/** Checks whether the given affine expression is an mod expression. */
int mlirAffineExprIsAMod(MlirAffineExpr affineExpr);

/** Creates an affine mod expression with 'lhs' and 'rhs'. */
MlirAffineExpr mlirAffineModExprGet(MlirAffineExpr lhs, MlirAffineExpr rhs);

/*============================================================================*/
/* Affine FloorDiv Expression.                                                */
/*============================================================================*/

/** Checks whether the given affine expression is an floordiv expression. */
int mlirAffineExprIsAFloorDiv(MlirAffineExpr affineExpr);

/** Creates an affine floordiv expression with 'lhs' and 'rhs'. */
MlirAffineExpr mlirAffineFloorDivExprGet(MlirAffineExpr lhs,
                                         MlirAffineExpr rhs);

/*============================================================================*/
/* Affine CeilDiv Expression.                                                 */
/*============================================================================*/

/** Checks whether the given affine expression is an ceildiv expression. */
int mlirAffineExprIsACeilDiv(MlirAffineExpr affineExpr);

/** Creates an affine ceildiv expression with 'lhs' and 'rhs'. */
MlirAffineExpr mlirAffineCeilDivExprGet(MlirAffineExpr lhs, MlirAffineExpr rhs);

/*============================================================================*/
/* Affine Binary Operation Expression.                                        */
/*============================================================================*/

/** Returns the left hand side affine expression of the given affine binary
 * operation expression. */
MlirAffineExpr mlirAffineBinaryOpExprGetLHS(MlirAffineExpr affineExpr);

/** Returns the right hand side affine expression of the given affine binary
 * operation expression. */
MlirAffineExpr mlirAffineBinaryOpExprGetRHS(MlirAffineExpr affineExpr);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_AFFINEEXPR_H
