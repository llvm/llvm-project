//===- SupportBindings.h - Additional bindings for Support ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines additional C bindings for the Support component.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINDINGS_GO_LLVM_SUPPORTBINDINGS_H
#define LLVM_BINDINGS_GO_LLVM_SUPPORTBINDINGS_H

#ifdef __cplusplus
extern "C" {
#endif

// This function duplicates the LLVMLoadLibraryPermanently function in the
// stable C API and adds an extra ErrMsg parameter to retrieve the error
// message.
void LLVMLoadLibraryPermanently2(const char *Filename, char **ErrMsg);

#ifdef __cplusplus
}
#endif

#endif
