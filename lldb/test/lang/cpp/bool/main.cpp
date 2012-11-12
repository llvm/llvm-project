//===-- main.cpp ------------------------------------------------*- C++ -*-===//
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
  bool my_bool = false;

  printf("%s\n", my_bool ? "true" : "false"); // breakpoint 1
}
