/*===-- mlir-c/Support.h - Helpers for C API to Core MLIR ---------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the auxiliary data structures used in C APIs to core  *|
|* MLIR functionality.                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef MLIR_C_SUPPORT_H
#define MLIR_C_SUPPORT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================*/
/* MlirStringRef.                                                             */
/*============================================================================*/

/** A pointer to a sized fragment of a string, not necessarily null-terminated.
 * Does not own the underlying string. This is equivalent to llvm::StringRef.
 */
struct MlirStringRef {
  const char *data; /**< Pointer to the first symbol. */
  size_t length;    /**< Length of the fragment. */
};
typedef struct MlirStringRef MlirStringRef;

/** Constructs a string reference from the pointer and length. The pointer need
 * not reference to a null-terminated string.
 */
inline MlirStringRef mlirStringRefCreate(const char *str, size_t length) {
  MlirStringRef result;
  result.data = str;
  result.length = length;
  return result;
}

/** Constructs a string reference from a null-terminated C string. Prefer
 * mlirStringRefCreate if the length of the string is known.
 */
MlirStringRef mlirStringRefCreateFromCString(const char *str);

/*============================================================================*/
/* MlirLogicalResult.                                                         */
/*============================================================================*/

/** A logical result value, essentially a boolean with named states. LLVM
 * convention for using boolean values to designate success or failure of an
 * operation is a moving target, so MLIR opted for an explicit class.
 * Instances of MlirLogicalResult must only be inspected using the associated
 * functions. */
struct MlirLogicalResult {
  int8_t value;
};
typedef struct MlirLogicalResult MlirLogicalResult;

/** Checks if the given logical result represents a success. */
inline int mlirLogicalResultIsSuccess(MlirLogicalResult res) {
  return res.value != 0;
}

/** Checks if the given logical result represents a failure. */
inline int mlirLogicalResultIsFailure(MlirLogicalResult res) {
  return res.value == 0;
}

/** Creates a logical result representing a success. */
inline static MlirLogicalResult mlirLogicalResultSuccess() {
  MlirLogicalResult res = {1};
  return res;
}

/** Creates a logical result representing a failure. */
inline static MlirLogicalResult mlirLogicalResultFailure() {
  MlirLogicalResult res = {0};
  return res;
}

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_SUPPORT_H
