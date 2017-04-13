//===-- debugserver_LogCallback.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
