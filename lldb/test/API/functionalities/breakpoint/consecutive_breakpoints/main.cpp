//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

int
main(int argc, char const *argv[])
{
    int a = 0;
    int b = 1;
    a = b + 1; // Set breakpoint here
    b = a + 1;
    return 0;
}

