#include <stdio.h>

struct A {
    int aa;
    char ab;
};

struct B {
    int ba;
    int bb;
};

struct C {
    int ca;
    int cb;
};

int Dump (A &a)
{
    return 1;
}

int Dump (B &b)
{
    return 2;
}

int Dump (C &c)
{
    return 3;
}

extern int CallStaticA();
extern int CallStaticB();

int main()
{
    A myA;
    B myB;
    C myC;

    printf("%d\n", CallStaticA() + CallStaticB()); // breakpoint
}
