/*===-- IPO.h - Interprocedural Transformations C Interface -----*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMIPO.a, which implements     *|
|* various interprocedural transformations of the LLVM IR.                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_TRANSFORMS_IPO_H
#define LLVM_C_TRANSFORMS_IPO_H

#include "llvm-c/ExternC.h"
#include "llvm-c/Types.h"

LLVM_C_EXTERN_C_BEGIN

/**
 * @defgroup LLVMCTransformsIPO Interprocedural transformations
 * @ingroup LLVMCTransforms
 *
 * @{
 */

/** See llvm::createDeadArgEliminationPass function. */
void LLVMAddDeadArgEliminationPass(LLVMPassManagerRef PM);

/** See llvm::createFunctionAttrsPass function. */
void LLVMAddFunctionAttrsPass(LLVMPassManagerRef PM);

/** See llvm::createAlwaysInlinerPass function. */
void LLVMAddAlwaysInlinerPass(LLVMPassManagerRef PM);

/**
 * @}
 */

LLVM_C_EXTERN_C_END

#endif
