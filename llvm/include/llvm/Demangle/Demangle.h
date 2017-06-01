//===--- Demangle.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstddef>

namespace llvm {
/// This is a llvm local version of __cxa_demangle. Other than the name and
/// being in the llvm namespace it is identical.
///
/// The mangled_name is demangled into buf and returned. If the buffer is not
/// large enough, realloc is used to expand it.
///
/// The *status will be set to
///   unknown_error: -4
///   invalid_args:  -3
///   invalid_mangled_name: -2
///   memory_alloc_failure: -1
///   success: 0

char *itaniumDemangle(const char *mangled_name, char *buf, size_t *n,
                      int *status);
}
