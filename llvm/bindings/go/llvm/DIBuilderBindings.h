//===- DIBuilderBindings.h - Bindings for DIBuilder -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines C bindings for the DIBuilder class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINDINGS_GO_LLVM_DIBUILDERBINDINGS_H
#define LLVM_BINDINGS_GO_LLVM_DIBUILDERBINDINGS_H

#include "IRBindings.h"
#include "llvm-c/Core.h"
#include "llvm-c/DebugInfo.h"

#ifdef __cplusplus
extern "C" {
#endif

// FIXME: These bindings shouldn't be Go-specific and should eventually move to
// a (somewhat) less stable collection of C APIs for use in creating bindings of
// LLVM in other languages.

typedef struct LLVMOpaqueDIBuilder *LLVMDIBuilderRef;

LLVMMetadataRef LLVMDIBuilderCreateAutoVariable(
    LLVMDIBuilderRef D, LLVMMetadataRef Scope, const char *Name,
    LLVMMetadataRef File, unsigned Line, LLVMMetadataRef Ty, int AlwaysPreserve,
    unsigned Flags, uint32_t AlignInBits);

LLVMMetadataRef LLVMDIBuilderCreateParameterVariable(
    LLVMDIBuilderRef D, LLVMMetadataRef Scope, const char *Name, unsigned ArgNo,
    LLVMMetadataRef File, unsigned Line, LLVMMetadataRef Ty, int AlwaysPreserve,
    unsigned Flags);

LLVMMetadataRef LLVMDIBuilderCreateTypedef(LLVMDIBuilderRef D,
                                           LLVMMetadataRef Ty, const char *Name,
                                           LLVMMetadataRef File, unsigned Line,
                                           LLVMMetadataRef Context);

LLVMMetadataRef LLVMDIBuilderGetOrCreateSubrange(LLVMDIBuilderRef D, int64_t Lo,
                                                 int64_t Count);

LLVMMetadataRef LLVMDIBuilderGetOrCreateArray(LLVMDIBuilderRef D,
                                              LLVMMetadataRef *Data,
                                              size_t Length);

LLVMMetadataRef LLVMDIBuilderGetOrCreateTypeArray(LLVMDIBuilderRef D,
                                                  LLVMMetadataRef *Data,
                                                  size_t Length);

LLVMMetadataRef LLVMDIBuilderCreateExpression(LLVMDIBuilderRef Dref,
                                              int64_t *Addr, size_t Length);

LLVMValueRef LLVMDIBuilderInsertDeclareAtEnd(LLVMDIBuilderRef D,
                                             LLVMValueRef Storage,
                                             LLVMMetadataRef VarInfo,
                                             LLVMMetadataRef Expr,
                                             LLVMBasicBlockRef Block);

LLVMValueRef LLVMDIBuilderInsertValueAtEnd(LLVMDIBuilderRef D, LLVMValueRef Val,
                                           LLVMMetadataRef VarInfo,
                                           LLVMMetadataRef Expr,
                                           LLVMBasicBlockRef Block);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
