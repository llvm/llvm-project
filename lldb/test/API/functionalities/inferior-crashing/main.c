//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
