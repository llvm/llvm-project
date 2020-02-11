//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies the correct handling of program counter jumps.

int otherfn();

template<typename T>
T min(T a, T b)
{
    if (a < b)
    {
        return a; // 1st marker
    } else {
        return b; // 2nd marker
    }
}

int main ()
{
    int i;
    double j;
    int min_i_a = 4, min_i_b = 5;
    double min_j_a = 7.0, min_j_b = 8.0;
    i = min(min_i_a, min_i_b); // 3rd marker
    j = min(min_j_a, min_j_b); // 4th marker

    return 0;
}
