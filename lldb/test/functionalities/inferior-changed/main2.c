//===-- main2.c -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char* argv[])
{
    int *int_ptr = (int *)malloc(sizeof(int));
    *int_ptr = 7;
    printf("Hello, world!\n");
    printf("Now not crash %d\n", *int_ptr); // Not crash here.
}
