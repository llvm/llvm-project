#include "derived.h"

Foo foo1;
Foo foo2;

Foo::Foo() { a = 12345; }

char Foo::baz() {
    return (char)(x&0xff);
}

