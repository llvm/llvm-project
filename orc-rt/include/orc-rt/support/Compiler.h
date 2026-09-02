//===--------- Compiler.h - Compiler abstraction support --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime support library.
//
// Most functionality in this file was swiped from llvm/Support/Compiler.h.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_COMPILER_H
#define ORC_RT_SUPPORT_COMPILER_H

#include "orc-rt-c/support/Compiler.h"

#include <cassert>

// ORC_RT_EXPORT marks a symbol declared in orc-rt as part of the ORC runtime's
// binary interface: exported from the runtime when it is built as a shared
// library, and imported by consumers of that library.
//
// Symbols belonging to the C API use ORC_RT_C_EXPORT instead. The two are
// equivalent today, but the C++ API is expected to change far more often than
// the C API, so ORC_RT_EXPORT may become separately switchable to allow the C++
// API to be hidden.
#define ORC_RT_EXPORT ORC_RT_C_EXPORT

#if ORC_RT_HAS_BUILTIN(__builtin_expect)
#define ORC_RT_LIKELY(EXPR) __builtin_expect((bool)(EXPR), true)
#define ORC_RT_UNLIKELY(EXPR) __builtin_expect((bool)(EXPR), false)
#else
#define ORC_RT_LIKELY(EXPR) (EXPR)
#define ORC_RT_UNLIKELY(EXPR) (EXPR)
#endif

#if defined(__APPLE__)
#define ORC_RT_WEAK_IMPORT __attribute__((weak_import))
#elif defined(_WIN32)
#define ORC_RT_WEAK_IMPORT
#else
#define ORC_RT_WEAK_IMPORT __attribute__((weak))
#endif

// ORC_RT_BUILTIN_UNREACHABLE: an optimizer hint that the current location is
// not reachable.
#if ORC_RT_HAS_BUILTIN(__builtin_unreachable) || defined(__GNUC__)
#define ORC_RT_BUILTIN_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
#define ORC_RT_BUILTIN_UNREACHABLE __assume(false)
#else
#define ORC_RT_BUILTIN_UNREACHABLE
#endif

// ORC_RT_UNREACHABLE(MSG): marks a point the program must never reach. In
// +Asserts builds it aborts with MSG; otherwise it lowers to
// ORC_RT_BUILTIN_UNREACHABLE.
#ifndef NDEBUG
#define ORC_RT_UNREACHABLE(MSG)                                                \
  do {                                                                         \
    assert(false && (MSG));                                                    \
    ORC_RT_BUILTIN_UNREACHABLE;                                                \
  } while (false)
#else
#define ORC_RT_UNREACHABLE(MSG) ORC_RT_BUILTIN_UNREACHABLE
#endif

#endif // ORC_RT_SUPPORT_COMPILER_H
