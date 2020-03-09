#include "derived.h"

int main() {
    Foo f; // break here
    f.bar();
    return f.baz();
}
