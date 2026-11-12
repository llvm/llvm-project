
#include "length.h"

class Foo {
public:
    A a;
};

class MyA : public A {
};

int main()
{
    Foo f;
    MyA a;

    return length(a); // break here
}
