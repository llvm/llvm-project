//===-- LanguageRegistration.h - Language registration API declarations ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_LANGUAGE_REGISTRATION_H
#define LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_LANGUAGE_REGISTRATION_H

#include "OffloadAPI.h"
#include "Types.h"

#include <cstdint>
#include <iterator>

/// Hidden, but exported, Registration API
///{
extern "C" {

void __llvmRegisterFunction(const char *Binary, const char *KernelID,
                            char *KernelName, const char *KernelName1, int,
                            uint3 *, uint3 *, dim3 *, dim3 *, int *);

void __llvmRegisterVar(void **, char *, char *, const char *, int, int, int,
                       int);

void __llvmRegisterManagedVar(void **, char *, char *, const char *, size_t,
                              unsigned);

void __llvmRegisterSurface(void **, const struct surfaceReference *,
                           const void **, const char *, int, int);

void __llvmRegisterTexture(void **, const struct textureReference *,
                           const void **, const char *, int, int, int);

struct __tgt_bin_desc;
void __tgt_register_lib(__tgt_bin_desc *Desc);
void __tgt_unregister_lib(__tgt_bin_desc *Desc);
}
///}

#endif // LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_LANGUAGE_REGISTRATION_H
