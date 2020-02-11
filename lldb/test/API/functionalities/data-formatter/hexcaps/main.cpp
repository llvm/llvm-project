//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

struct foo
{
    int first;
    int second;
};

int main ()
{
    struct foo mine = {0x001122AA, 0x1122BB44};
    printf("main.first = 0x%8.8x, main.second = 0x%8.8x\n", mine.first, mine.second);
	mine.first = 0xAABBCCDD; // Set break point at this line.
    printf("main.first = 0x%8.8x, main.second = 0x%8.8x\n", mine.first, mine.second);
	mine.second = 0xFF00FF00;
    printf("main.first = 0x%8.8x, main.second = 0x%8.8x\n", mine.first, mine.second);
    return 0;
}

