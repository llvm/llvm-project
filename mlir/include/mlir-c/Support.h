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

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_SUPPORT_H
