//===-- main.c ----------------------------------------------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

int main()
{
  const char a[] = "abcde";
  const char *z = "vwxyz";
  
  printf("%s %s", a, z); // breakpoint 1
}
