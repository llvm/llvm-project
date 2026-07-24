/*===---- cuda_runtime.cpp - CUDA runtime api implementations --------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#include "cuda_runtime.h"

#include "OffloadAPI.h"

#define LANGUAGE cuda

#include "../../kernel/src/LanguageRuntime.cpp"

extern "C" {
void __cudaRegisterFatBinaryEnd(void *) {}
}
