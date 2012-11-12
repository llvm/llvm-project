//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
	mine.first = 0xAABBCCDD; // Set break point at this line.
	mine.second = 0xFF00FF00;
    return 0;
}

