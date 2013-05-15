//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

bool bar(int const *foo) {
  return foo != 0; // Set break point at this line.
}

int main() {
  int foo[] = {1,2,3};
  return bar(foo);
}
