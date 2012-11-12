//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

struct contained
{
    int first;
    int second;
};

struct container
{
    int scalar;
    struct contained *pointer;
};

int main ()
{
    struct container mine = {1, 0};
    printf ("Mine's scalar is the only thing that is good: %d.\n", mine.scalar); // Set break point at this line.
    return 0;
}

