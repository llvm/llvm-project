//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <vector>

#include <stdio.h>
#include <stdint.h>

int main (int argc, char const *argv[], char const *envp[])
{
    std::vector<long> longs;
    std::vector<short> shorts;  
    for (int i=0; i<12; i++)
    {
        longs.push_back(i);
        shorts.push_back(i);
    }
    return 0; // Set breakpoint here to verify that std::vector 'longs' and 'shorts' have unique types.
}
