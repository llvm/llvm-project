//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
typedef signed char v4i8 __attribute__ ((vector_size(4)));
v4i8 global_vector = {1, 2, 3, 4};

int
main ()
{
  return 0;
}
