//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class Conversion
{
public:
    Conversion (int i) :
      m_i (i)
      {}

    operator bool()
    {
        return m_i != 0;
    }
    
private:
    int m_i;
};

class A
{
public:
    A(int i=0):
        m_a_int(i),
        m_aa_int(i+1)
    {
    }

    //virtual
    ~A()
    {
    }

    int
    GetInteger() const
    {
        return m_a_int;
    }
    void
    SetInteger(int i)
    {
        m_a_int = i;
    }

protected:
    int m_a_int;
    int m_aa_int;
};

class B : public A
{
public:
    B(int ai, int bi) :
        A(ai),
        m_b_int(bi)
    {
    }

    //virtual
    ~B()
    {
    }

    int
    GetIntegerB() const
    {
        return m_b_int;
    }
    void
    SetIntegerB(int i)
    {
        m_b_int = i;
    }

protected:
    int m_b_int;
};

#include <cstdio>
class C : public B
{
public:
    C(int ai, int bi, int ci) :
        B(ai, bi),
        m_c_int(ci)
    {
        printf("Within C::ctor() m_c_int=%d\n", m_c_int); // Set break point at this line.
    }

    //virtual
    ~C()
    {
    }

    int
    GetIntegerC() const
    {
        return m_c_int;
    }
    void
    SetIntegerC(int i)
    {
        m_c_int = i;
    }

protected:
    int m_c_int;
};

int
main (int argc, char const *argv[])
{
    A a(12);
    B b(22,33);
    C c(44,55,66);
    Conversion conv(1);
    if (conv)
        return b.GetIntegerB() - a.GetInteger() + c.GetInteger();
    return 0;
}
