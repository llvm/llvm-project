#include "derived.h"

Foo::Foo() { a = 12345; }
ns::Foo2::Foo2() { a = 23456; }

Foo foo1;
Foo foo2;

ns::Foo2 foo2_1;
ns::Foo2 foo2_2;
