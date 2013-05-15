//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdint.h>
int main ()
{
  int32_t myvar = -1;
  printf ("%d\n", myvar); // Set break point at this line.
  return myvar+1;
}
