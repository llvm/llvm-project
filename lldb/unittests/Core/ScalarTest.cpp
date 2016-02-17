//===-- ScalarTest.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER) && (_HAS_EXCEPTIONS == 0)
// Workaround for MSVC standard library bug, which fails to include <thread> when
// exceptions are disabled.
#include <eh.h>
#endif

#include "gtest/gtest.h"

#include "lldb/Core/Scalar.h"

using namespace lldb_private;

TEST(ScalarTest, RightShiftOperator)
{
    int a = 0x00001000;
    int b = 0xFFFFFFFF;
    int c = 4;
    Scalar a_scalar(a);
    Scalar b_scalar(b);
    Scalar c_scalar(c);
    ASSERT_EQ(a >> c, a_scalar >> c_scalar);
    ASSERT_EQ(b >> c, b_scalar >> c_scalar);
}
