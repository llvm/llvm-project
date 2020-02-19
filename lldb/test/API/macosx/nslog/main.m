//===-- main.m --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <Foundation/Foundation.h>

int main(int argc, char** argv)
{
    printf("About to log\n"); // break here
    NSLog(@"This is a message from NSLog");

    return 0;
}
