//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

void bar(int const *foo) {
  __builtin_trap(); // Set break point at this line.
}

int main() {
  int foo = 5;
  bar(&foo);
}
