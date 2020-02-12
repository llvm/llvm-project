//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

// This simple program is to test the lldb Python API related to events.

int a(int);
int b(int);
int c(int);

int a(int val)
{
    if (val <= 1)
        return b(val);
    else if (val >= 3)
        return c(val);

    return val;
}

int b(int val)
{
    return c(val);
}

int c(int val)
{
    return val + 3; // Find the line number of function "c" here.
}

int main (int argc, char const *argv[])
{
    int A1 = a(1);  // a(1) -> b(1) -> c(1)
    printf("a(1) returns %d\n", A1);
    
    int B2 = b(2);  // b(2) -> c(2)
    printf("b(2) returns %d\n", B2);
    
    int A3 = a(3);  // a(3) -> c(3)
    printf("a(3) returns %d\n", A3);
    
    return 0;
}
