//===-- debugserver_LogCallback.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//------------------------------------------------------------------------------
// this function is defined in debugserver.cpp, but is needed to link the
// debugserver Common library. It is for logging only, so it is left
// unimplemented here.
//------------------------------------------------------------------------------

#include <stdint.h>
#include <stdarg.h>

void FileLogCallback(void *baton, uint32_t flags, const char *format,
                     va_list args) {}
