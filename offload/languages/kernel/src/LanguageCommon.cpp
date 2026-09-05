//===-- LanguageCommon.cpp - Shared CUDA/HIP runtime entry points ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LanguageLaunch.h"
#include "LanguageRegistration.h"

#define LANGUAGE cuda
#include "LanguageAliases.inc"
#undef LANGUAGE

#define LANGUAGE hip
#include "LanguageAliases.inc"
#undef LANGUAGE
