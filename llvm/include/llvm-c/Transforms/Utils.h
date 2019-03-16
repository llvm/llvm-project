/*===-- Utils.h - Transformation Utils Library C Interface ------*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMTransformUtils.a, which     *|
|* implements various transformation utilities of the LLVM IR.                *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_TRANSFORMS_UTILS_H
#define LLVM_C_TRANSFORMS_UTILS_H

#include "llvm-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup LLVMCTransformsUtils Transformation Utilities
 * @ingroup LLVMCTransforms
 *
 * @{
 */

/** See llvm::createLowerSwitchPass function. */
void LLVMAddLowerSwitchPass(LLVMPassManagerRef PM);

/** See llvm::createPromoteMemoryToRegisterPass function. */
void LLVMAddPromoteMemoryToRegisterPass(LLVMPassManagerRef PM);

/** See llvm::createAddDiscriminatorsPass function. */
void LLVMAddAddDiscriminatorsPass(LLVMPassManagerRef PM);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif /* defined(__cplusplus) */

#endif

