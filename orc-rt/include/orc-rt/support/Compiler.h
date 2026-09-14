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
// Compiler-abstraction macros specific to the ORC runtime's C++ API. Macros
// that are usable from both C and C++ live in orc-rt-c/support/Compiler.h,
// which this header includes, so including this header gives access to both
// sets.
//
// ORC_RT_CXX_EXPORT is currently the only C++-specific macro.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_COMPILER_H
#define ORC_RT_SUPPORT_COMPILER_H

#include "orc-rt-c/support/Compiler.h"

// ORC_RT_CXX_EXPORT marks a symbol declared in orc-rt as part of the ORC
// runtime's binary interface: exported from the runtime when it is built as a
// shared library, and imported by consumers of that library.
//
// Symbols belonging to the C API use ORC_RT_C_EXPORT instead. The two are
// equivalent today, but the C++ API is expected to change far more often than
// the C API, so ORC_RT_CXX_EXPORT may become separately switchable to allow the
// C++ API to be hidden.
#define ORC_RT_CXX_EXPORT ORC_RT_C_EXPORT

#endif // ORC_RT_SUPPORT_COMPILER_H
