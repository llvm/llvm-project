//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdint.h>

extern "C"
{
   int foo();
};

int foo()
{
    puts("foo");
    return 2;
}

int main (int argc, char const *argv[], char const *envp[])
{          
    foo();
    return 0; //% self.expect("expression -- foo()", substrs = ['2'])
}

