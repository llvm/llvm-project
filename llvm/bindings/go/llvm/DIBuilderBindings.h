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

LLVMMetadataRef LLVMDIBuilderCreateTypedef(LLVMDIBuilderRef D,
                                           LLVMMetadataRef Ty, const char *Name,
                                           LLVMMetadataRef File, unsigned Line,
                                           LLVMMetadataRef Context);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
