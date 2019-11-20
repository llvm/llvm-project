/*===-- Scalar.h - Scalar Transformation Library C Interface ----*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMInstCombine.a, which        *|
|* combines instructions to form fewer, simple IR instructions.               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_TRANSFORMS_INSTCOMBINE_H
#define LLVM_C_TRANSFORMS_INSTCOMBINE_H

#include "llvm-c/ExternC.h"
#include "llvm-c/Types.h"

LLVM_C_EXTERN_C_BEGIN

/**
 * @defgroup LLVMCTransformsInstCombine Instruction Combining transformations
 * @ingroup LLVMCTransforms
 *
 * @{
 */

/** See llvm::createInstructionCombiningPass function. */
void LLVMAddInstructionCombiningPass(LLVMPassManagerRef PM);

/**
 * @}
 */

LLVM_C_EXTERN_C_END

#endif

