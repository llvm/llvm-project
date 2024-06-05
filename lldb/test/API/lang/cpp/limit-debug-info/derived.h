#include "base.h"

class Foo : public FooBase {
public:
  Foo();

  // Deliberately defined by hand.
  Foo &operator=(const Foo &rhs) {
    x = rhs.x; // break1
    a = rhs.a;
    return *this;
  }
  int a;
};

namespace ns {
class Foo2 : public Foo2Base {
public:
  Foo2();

  // Deliberately defined by hand.
  Foo2 &operator=(const Foo2 &rhs) {
    x = rhs.x; // break2
    a = rhs.a;
    return *this;
  }

  int a;
};
} // namespace ns

extern Foo foo1;
extern Foo foo2;

extern ns::Foo2 foo2_1;
extern ns::Foo2 foo2_2;
