#include "derived.h"

char Foo::baz() {
    return (char)(x&0xff);
}

