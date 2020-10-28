/*===-- mlir-c/AffineMap.h - C API for MLIR Affine maps -----------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef MLIR_C_AFFINEMAP_H
#define MLIR_C_AFFINEMAP_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

DEFINE_C_API_STRUCT(MlirAffineMap, const void);

/** Gets the context that the given affine map was created with*/
MlirContext mlirAffineMapGetContext(MlirAffineMap affineMap);

/** Checks whether an affine map is null. */
inline int mlirAffineMapIsNull(MlirAffineMap affineMap) {
  return !affineMap.ptr;
}

/** Checks if two affine maps are equal. */
int mlirAffineMapEqual(MlirAffineMap a1, MlirAffineMap a2);

/** Prints an affine map by sending chunks of the string representation and
 * forwarding `userData to `callback`. Note that the callback may be called
 * several times with consecutive chunks of the string. */
void mlirAffineMapPrint(MlirAffineMap affineMap, MlirStringCallback callback,
                        void *userData);

/** Prints the affine map to the standard error stream. */
void mlirAffineMapDump(MlirAffineMap affineMap);

/** Creates a zero result affine map with no dimensions or symbols in the
 * context. The affine map is owned by the context. */
MlirAffineMap mlirAffineMapEmptyGet(MlirContext ctx);

/** Creates a zero result affine map of the given dimensions and symbols in the
 * context. The affine map is owned by the context. */
MlirAffineMap mlirAffineMapGet(MlirContext ctx, intptr_t dimCount,
                               intptr_t symbolCount);

/** Creates a single constant result affine map in the context. The affine map
 * is owned by the context. */
MlirAffineMap mlirAffineMapConstantGet(MlirContext ctx, int64_t val);

/** Creates an affine map with 'numDims' identity in the context. The affine map
 * is owned by the context. */
MlirAffineMap mlirAffineMapMultiDimIdentityGet(MlirContext ctx,
                                               intptr_t numDims);

/** Creates an identity affine map on the most minor dimensions in the context.
 * The affine map is owned by the context. The function asserts that the number
 * of dimensions is greater or equal to the number of results. */
MlirAffineMap mlirAffineMapMinorIdentityGet(MlirContext ctx, intptr_t dims,
                                            intptr_t results);

/** Creates an affine map with a permutation expression and its size in the
 * context. The permutation expression is a non-empty vector of integers.
 * The elements of the permutation vector must be continuous from 0 and cannot
 * be repeated (i.e. `[1,2,0]` is a valid permutation. `[2,0]` or `[1,1,2]` is
 * an invalid invalid permutation.) The affine map is owned by the context. */
MlirAffineMap mlirAffineMapPermutationGet(MlirContext ctx, intptr_t size,
                                          unsigned *permutation);

/** Checks whether the given affine map is an identity affine map. The function
 * asserts that the number of dimensions is greater or equal to the number of
 * results. */
int mlirAffineMapIsIdentity(MlirAffineMap affineMap);

/** Checks whether the given affine map is a minor identity affine map. */
int mlirAffineMapIsMinorIdentity(MlirAffineMap affineMap);

/** Checks whether the given affine map is an empty affine map. */
int mlirAffineMapIsEmpty(MlirAffineMap affineMap);

/** Checks whether the given affine map is a single result constant affine
 * map. */
int mlirAffineMapIsSingleConstant(MlirAffineMap affineMap);

/** Returns the constant result of the given affine map. The function asserts
 * that the map has a single constant result. */
int64_t mlirAffineMapGetSingleConstantResult(MlirAffineMap affineMap);

/** Returns the number of dimensions of the given affine map. */
intptr_t mlirAffineMapGetNumDims(MlirAffineMap affineMap);

/** Returns the number of symbols of the given affine map. */
intptr_t mlirAffineMapGetNumSymbols(MlirAffineMap affineMap);

/** Returns the number of results of the given affine map. */
intptr_t mlirAffineMapGetNumResults(MlirAffineMap affineMap);

/** Returns the number of inputs (dimensions + symbols) of the given affine
 * map. */
intptr_t mlirAffineMapGetNumInputs(MlirAffineMap affineMap);

/** Checks whether the given affine map represents a subset of a symbol-less
 * permutation map. */
int mlirAffineMapIsProjectedPermutation(MlirAffineMap affineMap);

/** Checks whether the given affine map represents a symbol-less permutation
 * map. */
int mlirAffineMapIsPermutation(MlirAffineMap affineMap);

/** Returns the affine map consisting of the `resultPos` subset. */
MlirAffineMap mlirAffineMapGetSubMap(MlirAffineMap affineMap, intptr_t size,
                                     intptr_t *resultPos);

/** Returns the affine map consisting of the most major `numResults` results.
 * Returns the null AffineMap if the `numResults` is equal to zero.
 * Returns the `affineMap` if `numResults` is greater or equals to number of
 * results of the given affine map. */
MlirAffineMap mlirAffineMapGetMajorSubMap(MlirAffineMap affineMap,
                                          intptr_t numResults);

/** Returns the affine map consisting of the most minor `numResults` results.
 * Returns the null AffineMap if the `numResults` is equal to zero.
 * Returns the `affineMap` if `numResults` is greater or equals to number of
 * results of the given affine map. */
MlirAffineMap mlirAffineMapGetMinorSubMap(MlirAffineMap affineMap,
                                          intptr_t numResults);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_AFFINEMAP_H
