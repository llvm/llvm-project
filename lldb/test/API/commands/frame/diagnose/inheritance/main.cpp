#include <stdio.h>
#include <stdint.h>

class A
{
public:
    A(int a) : 
        m_a(a)
    {
    }
    virtual ~A(){}
    virtual int get2() const { return m_a; }
    virtual int get() const { return m_a; }
protected:
    int m_a;    
};

class B : public A
{
public:
    B(int a, int b) : 
        A(a),
        m_b(b)
    {
    }

    ~B() override
    {
    }

    int get2() const override
    {
        return m_b;
    }
    int get() const override
    {
        return m_b;
    }   
            
protected:
    int m_b;
};

struct C
{
    C(int c) : m_c(c){}
    virtual ~C(){}
    int m_c;
};

class D : public C, public B
{
public:
    D(int a, int b, int c, int d) : 
        C(c),
        B(a, b),
        m_d(d)
    {
    }
protected:
    int m_d;
};
int main (int argc, char const *argv[], char const *envp[])
{
    D *good_d = new D(1, 2, 3, 4);
    D *d = nullptr;
    return d->get();
}

