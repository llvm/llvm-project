#include "base.h"

class Foo : public FooNS
{
public:
    Foo() {
        a = 12345;
    }

    char baz() override;
    int a;
};

