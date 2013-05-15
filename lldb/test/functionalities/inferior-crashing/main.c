//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

const char *hello_world = "Hello, segfault!";

int main(int argc, const char* argv[])
{
    int *null_ptr = 0;
    printf("%s\n", hello_world);
    printf("Now crash %d\n", *null_ptr); // Crash here.
}
