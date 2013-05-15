//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <stdlib.h>

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
    int rc = c(val);
    void *ptr = malloc(1024);
    if (!ptr)  // Set breakpoint here to test target stop-hook.
        return -1;
    else
        printf("ptr=%p\n", ptr); // We should stop here after stepping.
    return rc; // End of the line range for which stop-hook is to be run.
}

int c(int val)
{
    return val + 3;
}

int main (int argc, char const *argv[])
{
    int A1 = a(1);
    printf("a(1) returns %d\n", A1);
    
    int C2 = c(2); // Another breakpoint which is outside of the stop-hook range.
    printf("c(2) returns %d\n", C2);
    
    int A3 = a(3);
    printf("a(3) returns %d\n", A3);
    
    return 0;
}
