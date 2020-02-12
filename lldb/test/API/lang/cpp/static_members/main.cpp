//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

struct A
{
    short m_a;
    static long s_b;
    char m_c;
    static int s_d;

    long access() {
        return m_a + s_b + m_c + s_d; // breakpoint 2
    }
};

long A::s_b = 2;
int A::s_d = 4;

int main()
{
    A my_a;
    my_a.m_a = 1;
    my_a.m_c = 3;

    my_a.access(); // breakpoint 1 
    return 0;
}

