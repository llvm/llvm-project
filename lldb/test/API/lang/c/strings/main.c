//===-- main.c ----------------------------------------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

int main()
{
  const char a[] = "abcde";
  const char *z = "vwxyz";
  
  printf("%s %s", a, z); // breakpoint 1
}
