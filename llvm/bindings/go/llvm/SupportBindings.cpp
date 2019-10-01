//===- SupportBindings.cpp - Additional bindings for support --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines additional C bindings for the support component.
//
//===----------------------------------------------------------------------===//

#include "SupportBindings.h"
#include "llvm/Support/DynamicLibrary.h"
#include <stdlib.h>
#include <string.h>

void LLVMLoadLibraryPermanently2(const char *Filename, char **ErrMsg) {
  std::string ErrMsgStr;
  if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(Filename, &ErrMsgStr)) {
    *ErrMsg = static_cast<char *>(malloc(ErrMsgStr.size() + 1));
    memcpy(static_cast<void *>(*ErrMsg),
           static_cast<const void *>(ErrMsgStr.c_str()), ErrMsgStr.size() + 1);
  }
}
