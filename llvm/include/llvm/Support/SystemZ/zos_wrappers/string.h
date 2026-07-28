//===- string.h - Common z/OS Include File ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares z/OS implementations for common functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ZOSWRAPPERS_STRING_H
#define LLVM_SUPPORT_ZOSWRAPPERS_STRING_H

#include_next <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// z/OS Unix System Services does not have support for:
// - strsignal()
// - strnlen()
// Implementations are provided for z/OS.

char *strsignal(int sig) asm("llvm_zos_strsignal");

size_t strnlen(const char *S, size_t MaxLen) asm("llvm_zos_strnlen");

#ifdef __cplusplus
}
#endif

#endif
