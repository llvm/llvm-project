//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

// breakpoint set -n foo
//
//
int foo (int value)
{
  printf ("I got the value: %d.\n", value);
  return 0;
}

int main (int argc, char **argv)
{
  foo (argc);
  printf ("Hello there: %d.\n", argc);
  return 0;
}
