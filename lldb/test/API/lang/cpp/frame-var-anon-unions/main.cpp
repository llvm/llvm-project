//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

int main() {
  union {
    int i;
    char c;
  };
  struct {
    int x;
    char y;
    short z;
  } s{3,'B',14};
  i = 0xFFFFFF00;
  c = 'A';
  return c; // break here
}
