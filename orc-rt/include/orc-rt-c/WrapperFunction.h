/*===--------- WrapperFunction.h - Wrapper function utils ---------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Defines orc_rt_WrapperFunctionBuffer and related APIs.                     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_WRAPPERFUNCTION_H
#define ORC_RT_C_WRAPPERFUNCTION_H

#include "orc-rt-c/ExternC.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

ORC_RT_C_EXTERN_C_BEGIN

typedef union {
  char *ValuePtr;
  char Value[sizeof(char *)];
} orc_rt_WrapperFunctionBufferDataUnion;

/**
 * orc_rt_WrapperFunctionBuffer is a kind of C-SmallVector with an
 * out-of-band error state.
 *
 * If Size == 0 and Data.ValuePtr is non-zero then the value is in the
 * 'out-of-band error' state, and Data.ValuePtr points at a malloc-allocated,
 * null-terminated string error message.
 *
 * If Size <= sizeof(orc_rt_WrapperFunctionBufferData) then the value is in
 * the 'small' state and the content is held in the first Size bytes of
 * Data.Value.
 *
 * If Size > sizeof(orc_rt_WrapperFunctionBufferData) then the value is in the
 * 'large' state and the content is held in the first Size bytes of the
 * memory pointed to by Data.ValuePtr. This memory must have been allocated by
 * malloc, and will be freed with free when this value is destroyed.
 */
typedef struct {
  orc_rt_WrapperFunctionBufferDataUnion Data;
  size_t Size;
} orc_rt_WrapperFunctionBuffer;

/**
 * Zero-initialize an orc_rt_WrapperFunctionBuffer.
 */
static inline void
orc_rt_WrapperFunctionBufferInit(orc_rt_WrapperFunctionBuffer *B) {
  B->Size = 0;
  B->Data.ValuePtr = 0;
}

/**
 * Create an orc_rt_WrapperFunctionBuffer with an uninitialized buffer of
 * size Size. The buffer is returned via the DataPtr argument.
 */
static inline orc_rt_WrapperFunctionBuffer
orc_rt_WrapperFunctionBufferAllocate(size_t Size) {
  orc_rt_WrapperFunctionBuffer B;
  B.Size = Size;
  // If Size is 0 ValuePtr must be 0 or it is considered an out-of-band error.
  B.Data.ValuePtr = 0;
  if (Size > sizeof(B.Data.Value))
    B.Data.ValuePtr = (char *)malloc(Size);
  return B;
}

/**
 * Create an orc_rt_WrapperFunctionBuffer from the given data range.
 */
static inline orc_rt_WrapperFunctionBuffer
orc_rt_CreateWrapperFunctionBufferFromRange(const char *Data, size_t Size) {
  orc_rt_WrapperFunctionBuffer B;
  B.Size = Size;
  if (B.Size > sizeof(B.Data.Value)) {
    char *Tmp = (char *)malloc(Size);
    memcpy(Tmp, Data, Size);
    B.Data.ValuePtr = Tmp;
  } else
    memcpy(B.Data.Value, Data, Size);
  return B;
}

/**
 * Create an orc_rt_WrapperFunctionBuffer by copying the given string,
 * including the null-terminator.
 *
 * This function copies the input string. The client is responsible for freeing
 * the ErrMsg arg.
 */
static inline orc_rt_WrapperFunctionBuffer
orc_rt_CreateWrapperFunctionBufferFromString(const char *Source) {
  return orc_rt_CreateWrapperFunctionBufferFromRange(Source,
                                                     strlen(Source) + 1);
}

/**
 * Create an orc_rt_WrapperFunctionBuffer representing an out-of-band
 * error.
 *
 * This function copies the input string. The client is responsible for freeing
 * the ErrMsg arg.
 */
static inline orc_rt_WrapperFunctionBuffer
orc_rt_CreateWrapperFunctionBufferFromOutOfBandError(const char *ErrMsg) {
  orc_rt_WrapperFunctionBuffer B;
  B.Size = 0;
  char *Tmp = (char *)malloc(strlen(ErrMsg) + 1);
  strcpy(Tmp, ErrMsg);
  B.Data.ValuePtr = Tmp;
  return B;
}

/**
 * This should be called to destroy orc_rt_WrapperFunctionBuffer values
 * regardless of their state.
 */
static inline void
orc_rt_WrapperFunctionBufferDispose(orc_rt_WrapperFunctionBuffer *B) {
  if (B->Size > sizeof(B->Data.Value) || (B->Size == 0 && B->Data.ValuePtr))
    free(B->Data.ValuePtr);
}

/**
 * Get a pointer to the data contained in the given
 * orc_rt_WrapperFunctionBuffer.
 */
static inline char *
orc_rt_WrapperFunctionBufferData(orc_rt_WrapperFunctionBuffer *B) {
  assert((B->Size != 0 || B->Data.ValuePtr == NULL) &&
         "Cannot get data for out-of-band error value");
  return B->Size > sizeof(B->Data.Value) ? B->Data.ValuePtr : B->Data.Value;
}

/**
 * Safely get the size of the given orc_rt_WrapperFunctionBuffer.
 *
 * Asserts that we're not trying to access the size of an error value.
 */
static inline size_t
orc_rt_WrapperFunctionBufferSize(const orc_rt_WrapperFunctionBuffer *B) {
  assert((B->Size != 0 || B->Data.ValuePtr == NULL) &&
         "Cannot get size for out-of-band error value");
  return B->Size;
}

/**
 * Returns 1 if this value is equivalent to a value just initialized by
 * orc_rt_WrapperFunctionBufferInit, 0 otherwise.
 */
static inline size_t
orc_rt_WrapperFunctionBufferEmpty(const orc_rt_WrapperFunctionBuffer *B) {
  return B->Size == 0 && B->Data.ValuePtr == 0;
}

/**
 * Returns a pointer to the out-of-band error string for this
 * orc_rt_WrapperFunctionBuffer, or null if there is no error.
 *
 * The orc_rt_WrapperFunctionBuffer retains ownership of the error
 * string, so it should be copied if the caller wishes to preserve it.
 */
static inline const char *orc_rt_WrapperFunctionBufferGetOutOfBandError(
    const orc_rt_WrapperFunctionBuffer *B) {
  return B->Size == 0 ? B->Data.ValuePtr : 0;
}

ORC_RT_C_EXTERN_C_END

#endif /* ORC_RT_WRAPPERFUNCTION_H */
