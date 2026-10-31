/*===----------- Error.h - C API for ORC Runtime Errors -----------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file defines the C interface to the ORC runtime's Error class.        *|
|*                                                                            *|
|* TODO: Explain ownership model.                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_SUPPORT_ERROR_H
#define ORC_RT_C_SUPPORT_ERROR_H

#include "orc-rt-c/support/Compiler.h"
#include "orc-rt-c/support/CoreTypes.h"
#include "orc-rt-c/support/RTTI.h"

ORC_RT_C_EXTERN_C_BEGIN

/**
 * Opaque reference to an error instance. Null serves as the 'success' value.
 */
typedef struct orc_rt_OpaqueError *orc_rt_ErrorRef;

#define orc_rt_ErrorSuccess ((orc_rt_ErrorRef)0)

ORC_RT_RTTI_PARTICIPANT(Error)

typedef struct orc_rt_OpaqueStringError *orc_rt_StringErrorRef;

ORC_RT_RTTI_PARTICIPANT(StringError)

/**
 * Dispose of the given error without handling it. This operation consumes the
 * error, and the given orc_rt_ErrorRef value is not usable once this call
 * returns.
 * Note: This method *only* needs to be called if the error is not being passed
 * to some other consuming operation, e.g. LLVMGetErrorMessage.
 */
ORC_RT_C_EXPORT void orc_rt_Error_consume(orc_rt_ErrorRef Err) ORC_RT_C_NOTHROW;

/**
 * Report a fatal error if Err is a failure value.
 *
 * This function can be used to wrap calls to fallible functions ONLY when it is
 * known that the Error will always be a success value.
 */
ORC_RT_C_EXPORT void
orc_rt_Error_cantFail(orc_rt_ErrorRef Err) ORC_RT_C_NOTHROW;

/**
 * Returns the given string's error message. This operation consumes the error,
 * and the given orc_rt_ErrorRef value is not usable once this call returns.
 * The caller is responsible for disposing of the string by calling
 * LLVMDisposeErrorMessage.
 */
ORC_RT_C_EXPORT char *
orc_rt_Error_toString(orc_rt_ErrorRef Err) ORC_RT_C_NOTHROW;

/**
 * Dispose of the given error message.
 */
ORC_RT_C_EXPORT void
orc_rt_Error_freeErrorMessage(char *ErrMsg) ORC_RT_C_NOTHROW;

/**
 * Create a StringError.
 */
ORC_RT_C_EXPORT orc_rt_ErrorRef orc_rt_StringError_create(const char *ErrMsg)
    ORC_RT_C_NOTHROW;

ORC_RT_C_EXTERN_C_END

#endif /* ORC_RT_C_SUPPORT_ERROR_H */
