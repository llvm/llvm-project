//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

class foo {
public:
  template <class T> T func(T x) const {
    return x+2; //% self.expect("expr 2+3", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["5"])
  }
};

int i;

int main() {
  return foo().func(i);
}
