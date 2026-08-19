//===-- cuda_runtime.h - CUDA runtime API declarations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGES_INCLUDE_CUDA_CUDA_RUNTIME_H
#define LLVM_OFFLOAD_LANGUAGES_INCLUDE_CUDA_CUDA_RUNTIME_H

#define LANGUAGE cuda

#include "../kernel/DefineLanguageNames.inc"

#include "../kernel/LanguageRuntime.h"

#include "../kernel/UndefineLanguageNames.inc"

// we dont rename the math symbols
#include "../kernel/LanguageMath.h"

#undef LANGUAGE

using cudaDeviceProp = cudaDeviceProp_t;

#endif // LLVM_OFFLOAD_LANGUAGES_INCLUDE_CUDA_CUDA_RUNTIME_H
