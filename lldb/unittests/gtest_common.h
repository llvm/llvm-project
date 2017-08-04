//===-- gtest_common.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(LLDB_GTEST_COMMON_H)
#error "gtest_common.h should not be included manually."
#else
#define LLDB_GTEST_COMMON_H
#endif

// This header file is force included by all of LLDB's unittest compilation
// units.  Be very leary about putting anything in this file.

#if defined(_MSC_VER) && (_HAS_EXCEPTIONS == 0)
// Due to a bug in <thread>, when _HAS_EXCEPTIONS == 0 the header will try to
// call
// uncaught_exception() without having a declaration for it.  The fix for this
// is
// to manually #include <eh.h>, which contains this declaration.
#include <eh.h>
#endif
