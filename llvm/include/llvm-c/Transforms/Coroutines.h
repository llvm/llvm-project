/*===-- Coroutines.h - Coroutines Library C Interface -----------*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMCoroutines.a, which         *|
|* implements various scalar transformations of the LLVM IR.                  *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_TRANSFORMS_COROUTINES_H
#define LLVM_C_TRANSFORMS_COROUTINES_H

#include "llvm-c/ExternC.h"
#include "llvm-c/Types.h"
#include "llvm-c/Transforms/PassManagerBuilder.h"

LLVM_C_EXTERN_C_BEGIN

/**
 * @defgroup LLVMCTransformsCoroutines Coroutine transformations
 * @ingroup LLVMCTransforms
 *
 * @{
 */

/** See llvm::createCoroEarlyLegacyPass function. */
void LLVMAddCoroEarlyPass(LLVMPassManagerRef PM);

/** See llvm::createCoroSplitLegacyPass function. */
void LLVMAddCoroSplitPass(LLVMPassManagerRef PM);

/** See llvm::createCoroElideLegacyPass function. */
void LLVMAddCoroElidePass(LLVMPassManagerRef PM);

/** See llvm::createCoroCleanupLegacyPass function. */
void LLVMAddCoroCleanupPass(LLVMPassManagerRef PM);

/** See llvm::addCoroutinePassesToExtensionPoints. */
void LLVMPassManagerBuilderAddCoroutinePassesToExtensionPoints(LLVMPassManagerBuilderRef PMB);

/**
 * @}
 */

LLVM_C_EXTERN_C_END

#endif
