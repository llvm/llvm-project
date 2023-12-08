#include "base.h"

class Foo : public FooNS
{
public:
    Foo();

    // Deliberately defined by hand.
    Foo &operator=(const Foo &rhs) {
      a = rhs.a;
      return *this;
    }

    char baz() override;
    int a;
};

extern Foo foo1;
extern Foo foo2;
