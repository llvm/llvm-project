//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

int g_global_var = 123;
static int g_static_var = 123;

int main (int argc, char const *argv[])
{
    static int static_var = 123;
    g_static_var = 123; // clang bug. Need to touch this variable, otherwise it disappears.
    int i = 0;                                  // breakpoint 1
    for (i=0; i<1; ++i)
    {
        int j = i*2;
        printf("i = %i, j = %i\n", i, j);       // breakpoint 2
        {
            int k = i*j*3;
            printf("i = %i, j = %i\n", i, j);   // breakpoint 3
        }
    }
    return 0;
}
