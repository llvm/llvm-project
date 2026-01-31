/*===----------- Error.h - C API for ORC Runtime Errors -----------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file defines the C interface to LLVM's Error class.                   *|
|*                                                                            *|
|* TODO: Explain ownership model.                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_ERROR_H
#define ORC_RT_C_ERROR_H

#include "orc-rt-c/CoreTypes.h"
#include "orc-rt-c/ExternC.h"
#include "orc-rt-c/Visibility.h"

ORC_RT_C_EXTERN_C_BEGIN

#define orc_rt_ErrorSuccess ((orc_rt_ErrorRef)0)

/**
 * Error type identifier.
 */
typedef const void *orc_rt_Error_TypeId;

/**
 * Returns the type id for the given error instance, which must be a failure
 * value (i.e. non-null).
 */
ORC_RT_C_ABI orc_rt_Error_TypeId orc_rt_Error_getTypeId(orc_rt_ErrorRef Err);

/**
 * Dispose of the given error without handling it. This operation consumes the
 * error, and the given orc_rt_ErrorRef value is not usable once this call
 * returns.
 * Note: This method *only* needs to be called if the error is not being passed
 * to some other consuming operation, e.g. LLVMGetErrorMessage.
 */
ORC_RT_C_ABI void orc_rt_Error_consume(orc_rt_ErrorRef Err);

/**
 * Report a fatal error if Err is a failure value.
 *
 * This function can be used to wrap calls to fallible functions ONLY when it is
 * known that the Error will always be a success value.
 */
ORC_RT_C_ABI void orc_rt_Error_cantFail(orc_rt_ErrorRef Err);

/**
 * Returns the given string's error message. This operation consumes the error,
 * and the given orc_rt_ErrorRef value is not usable once this call returns.
 * The caller is responsible for disposing of the string by calling
 * LLVMDisposeErrorMessage.
 */
ORC_RT_C_ABI char *orc_rt_Error_toString(orc_rt_ErrorRef Err);

/**
 * Dispose of the given error message.
 */
ORC_RT_C_ABI void orc_rt_Error_freeErrorMessage(char *ErrMsg);

/**
 * Returns the type id for llvm StringError.
 */
ORC_RT_C_ABI orc_rt_Error_TypeId orc_rt_StringError_getTypeId(void);

/**
 * Create a StringError.
 */
ORC_RT_C_ABI orc_rt_ErrorRef orc_rt_StringError_create(const char *ErrMsg);

ORC_RT_C_EXTERN_C_END

#endif // ORC_RT_C_ERROR_H
