//===-- a.c -----------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
extern int b_function ();

int a_init()
{
    return 234;
}

int a_global = a_init();

extern "C" int
a_function ()
{
    return b_function ();
}
