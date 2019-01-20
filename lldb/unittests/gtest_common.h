//===-- gtest_common.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(LLDB_GTEST_COMMON_H)
#error "gtest_common.h should not be included manually."
#else
#define LLDB_GTEST_COMMON_H
#endif

// This header file is force included by all of LLDB's unittest compilation
// units.  Be very leary about putting anything in this file.
