//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

class A
{
public:
  static int getStaticValue();
  int getMemberValue();
  int a;
};

int A::getStaticValue()
{
  return 5;
} 

int A::getMemberValue()
{
  return a;
}

int main()
{
  A my_a;

  my_a.a = 3;

  printf("%d\n", A::getStaticValue()); // Break at this line
  printf("%d\n", my_a.getMemberValue());
}
