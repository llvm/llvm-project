//===-- DefineTestLanguageNames.inc - Test language API renaming ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_TEST_LANGUAGE
#error "OFFLOAD_TEST_LANGUAGE must be defined to cuda or hip"
#endif

#define LANGUAGE OFFLOAD_TEST_LANGUAGE
#include "kernel/DefineLanguageNames.inc"

#define OFFLOAD_TEST_STRINGIZE_IMPL(X) #X
#define OFFLOAD_TEST_STRINGIZE(X) OFFLOAD_TEST_STRINGIZE_IMPL(X)
#define OFFLOAD_TEST_LANGUAGE_NAME OFFLOAD_TEST_STRINGIZE(OFFLOAD_TEST_LANGUAGE)
