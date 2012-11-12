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
  void accessMember(int a);
  int accessMemberConst() const;
  static int accessStaticMember();

  void accessMemberInline(int a) __attribute__ ((always_inline))
  {
    m_a = a; // breakpoint 4
  }

  int m_a;
  static int s_a;
};

int A::s_a = 5;

void A::accessMember(int a)
{
  m_a = a; // breakpoint 1
}

int A::accessMemberConst() const
{
  return m_a; // breakpoint 2
}

int A::accessStaticMember()
{
  return s_a; // breakpoint 3
} 

int main()
{
  A my_a;

  my_a.accessMember(3);
  my_a.accessMemberConst();
  A::accessStaticMember();
  my_a.accessMemberInline(5);
}
