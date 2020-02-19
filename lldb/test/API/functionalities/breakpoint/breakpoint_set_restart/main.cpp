//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <stdio.h>
#include <thread>


int main(int argc, char const *argv[])
{
    static bool done = false;
    while (!done)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    printf("Set a breakpoint here.\n");
    return 0;
}

