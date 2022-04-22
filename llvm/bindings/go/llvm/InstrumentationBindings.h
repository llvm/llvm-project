//===- InstrumentationBindings.h - instrumentation bindings -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines C bindings for the Transforms/Instrumentation component.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINDINGS_GO_LLVM_INSTRUMENTATIONBINDINGS_H
#define LLVM_BINDINGS_GO_LLVM_INSTRUMENTATIONBINDINGS_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

// FIXME: These bindings shouldn't be Go-specific and should eventually move to
// a (somewhat) less stable collection of C APIs for use in creating bindings of
// LLVM in other languages.

void LLVMAddAddressSanitizerFunctionPass(LLVMPassManagerRef PM);
void LLVMAddAddressSanitizerModulePass(LLVMPassManagerRef PM);
void LLVMAddThreadSanitizerPass(LLVMPassManagerRef PM);
void LLVMAddDataFlowSanitizerPass(LLVMPassManagerRef PM, int ABIListFilesNum,
                                  const char **ABIListFiles);

#ifdef __cplusplus
}
#endif

#endif
