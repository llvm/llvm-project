//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

struct Point {
    int x;
    int y;
    Point(int X = 3, int Y = 2) : x(X), y(Y) {}
};

Point g_point(3,4);
Point* g_point_pointer = new Point(7,5);

int main (int argc, const char * argv[])
{
    return 0; // Set break point at this line.
}

