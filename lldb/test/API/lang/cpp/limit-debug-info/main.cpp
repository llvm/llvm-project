#include "derived.h"

int main() {
    foo1 = foo2; // break here

    foo1.bar();
    return foo1.baz();
}
