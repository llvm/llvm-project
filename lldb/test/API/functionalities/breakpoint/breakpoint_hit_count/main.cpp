//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

int a(int val)
{
   return val; // Breakpoint Location 1
}

float a(float val)
{
    return val; // Breakpoint Location 2
}

int main (int argc, char const *argv[])
{
    int A1 = a(1);
    float A2 = a(2.0f);
    float A3 = a(3.0f);

    return 0;
}
