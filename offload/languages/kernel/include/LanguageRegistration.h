//===---- LanguageRegistration.h - Language (CUDA/HIP) registration api ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "OffloadAPI.h"
#include "Types.h"

#include <cstdint>
#include <iterator>

#define HIP_FATBIN_MAGIC_STR "__CLANG_OFFLOAD_BUNDLE__"
constexpr auto HIP_FATBIN_MAGIC_STR_LEN = sizeof(HIP_FATBIN_MAGIC_STR) - 1;

namespace {
struct FatbinWrapperTy {
  int Magic;
  int Version;
  const char *Data;
  const char *DataEnd;
};
} // namespace

static void readTUFatbin(const char *Binary, const FatbinWrapperTy *FW);

static void readHIPFatbinEntries(const char *Binary, const char *HIPFatbinPtr);

/// Hidden, but exported, Registration API
///{
extern "C" {

void __llvmRegisterFunction(const char *Binary, const char *KernelID,
                            char *KernelName, const char *KernelName1, int,
                            uint3 *, uint3 *, dim3 *, dim3 *, int *);

const char *__llvmRegisterFatBinary(const char *Binary);

void __llvmUnregisterFatBinary(void *Handle);

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
