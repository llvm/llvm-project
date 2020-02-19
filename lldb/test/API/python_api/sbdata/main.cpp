//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdint.h>

struct foo
{
    uint32_t a;
    uint32_t b;
    float c;
    foo() : a(0), b(1), c(3.14) {}
    foo(uint32_t A, uint32_t B, float C) :
        a(A),
        b(B),
        c(C)
    {}
};

int main (int argc, char const *argv[])
{
    foo* foobar = new foo[2];
    
    foobar[0].a = 1;
    foobar[0].b = 9;

    foobar[1].a = 8;
    foobar[1].b = 5;
    
    foobar[1].b = 7; // set breakpoint here
    
    foobar[1].c = 6.28;
    
    foo barfoo[] = {foo(1,2,3), foo(4,5,6)};
    
    delete[] foobar;
    
    return 0;
}
