#include <stdio.h>
#include <stdint.h>

class A
{
public:
    A () : m_pad ('c') {}

    virtual ~A () {}
    
    virtual const char * a()
    {
        return __PRETTY_FUNCTION__;
    }

    virtual const char * b()
    {
        return __PRETTY_FUNCTION__;
    }

    virtual const char * c()
    {
        return __PRETTY_FUNCTION__;
    }
protected:
    char m_pad;
};

class AA
{
public:
    AA () : m_pad('A') {}
    virtual ~AA () {}

    virtual const char * aa()
    {
        return __PRETTY_FUNCTION__;
    }
  
protected:
    char m_pad;
};

class B : virtual public A, public AA
{
public:
    B () : m_pad ('c')  {}

    virtual ~B () {}
    
    virtual const char * a()
    {
        return __PRETTY_FUNCTION__;
    }

    virtual const char * b()
    {
        return __PRETTY_FUNCTION__;
    }
protected:
    char m_pad;
};

class C : public B, virtual public A
{
public:
    C () : m_pad ('c') {}

    virtual ~C () {}
    
    virtual const char * a()
    {
        return __PRETTY_FUNCTION__;
    }
protected:
    char m_pad;
};

int main (int argc, char const *argv[], char const *envp[])
{
    A *a_as_A = new A();
    B *b_as_B = new B();
    A *b_as_A = b_as_B;
    C *c_as_C = new C();
    A *c_as_A = c_as_C;

    char golden[4096];
    char *p = golden;
    char *end = p + sizeof golden;
    p += snprintf(p, end-p, "a_as_A->a() = '%s'\n", a_as_A->a());
    p += snprintf(p, end-p, "a_as_A->b() = '%s'\n", a_as_A->b());
    p += snprintf(p, end-p, "a_as_A->c() = '%s'\n", a_as_A->c());
    p += snprintf(p, end-p, "b_as_A->a() = '%s'\n", b_as_A->a());
    p += snprintf(p, end-p, "b_as_A->b() = '%s'\n", b_as_A->b());
    p += snprintf(p, end-p, "b_as_A->c() = '%s'\n", b_as_A->c());
    p += snprintf(p, end-p, "b_as_B->aa() = '%s'\n", b_as_B->aa());
    p += snprintf(p, end-p, "c_as_A->a() = '%s'\n", c_as_A->a());
    p += snprintf(p, end-p, "c_as_A->b() = '%s'\n", c_as_A->b());
    p += snprintf(p, end-p, "c_as_A->c() = '%s'\n", c_as_A->c());
    p += snprintf(p, end-p, "c_as_C->aa() = '%s'\n", c_as_C->aa());
    puts("");// Set first breakpoint here.
    // then evaluate:
    // expression a_as_A->a()
    // expression a_as_A->b()
    // expression a_as_A->c()
    // expression b_as_A->a()
    // expression b_as_A->b()
    // expression b_as_A->c()
    // expression b_as_B->aa()
    // expression c_as_A->a()
    // expression c_as_A->b()
    // expression c_as_A->c()
    // expression c_as_C->aa()
    
    return 0;
}
