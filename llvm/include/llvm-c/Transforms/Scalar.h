/*===-- Scalar.h - Scalar Transformation Library C Interface ----*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMScalarOpts.a, which         *|
|* implements various scalar transformations of the LLVM IR.                  *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_TRANSFORMS_SCALAR_H
#define LLVM_C_TRANSFORMS_SCALAR_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

/** See llvm::createConstantPropagationPass function. */
void LLVMAddConstantPropagationPass(LLVMPassManagerRef PM);

/** See llvm::createInstructionCombiningPass function. */
void LLVMAddInstructionCombiningPass(LLVMPassManagerRef PM);

/** See llvm::createPromoteMemoryToRegisterPass function. */
void LLVMAddPromoteMemoryToRegisterPass(LLVMPassManagerRef PM);

/** See llvm::demotePromoteMemoryToRegisterPass function. */
void LLVMAddDemoteMemoryToRegisterPass(LLVMPassManagerRef PM);

/** See llvm::createReassociatePass function. */
void LLVMAddReassociatePass(LLVMPassManagerRef PM);

/** See llvm::createGVNPass function. */
void LLVMAddGVNPass(LLVMPassManagerRef PM);

/** See llvm::createCFGSimplificationPass function. */
void LLVMAddCFGSimplificationPass(LLVMPassManagerRef PM);

#ifdef __cplusplus
}
#endif /* defined(__cplusplus) */

#endif
