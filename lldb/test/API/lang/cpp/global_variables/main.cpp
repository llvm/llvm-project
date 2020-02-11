//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

namespace abc {
	int g_file_global_int = 42;
}

int main (int argc, char const *argv[])
{
    return abc::g_file_global_int; // Set break point at this line.
}
