//===--- DemangleConfig.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains DEMANGLE_ aliases for LLVM_ definitions. The canonical copy of
// ItaniumDemangle.h cannot depend on LLVM headers because lives in the
// libcxxabi project.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEMANGLE_DEMANGLECONFIG_H
#define LLVM_DEMANGLE_DEMANGLECONFIG_H

#include "llvm/Support/Compiler.h"

#define DEMANGLE_DUMP_METHOD LLVM_DUMP_METHOD
#define DEMANGLE_FALLTHROUGH LLVM_FALLTHROUGH

#if defined(LLVM_BUILTIN_UNREACHABLE)
#define DEMANGLE_UNREACHABLE LLVM_BUILTIN_UNREACHABLE
#else
#define DEMANGLE_UNREACHABLE
#endif

#ifndef DEMANGLE_ASSERT
#include <cassert>
#define DEMANGLE_ASSERT(__expr, __msg) assert((__expr) && (__msg))
#endif

#define DEMANGLE_NAMESPACE_BEGIN namespace llvm { namespace itanium_demangle {
#define DEMANGLE_NAMESPACE_END } }

#endif
