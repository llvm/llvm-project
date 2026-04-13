//#include "local.h"

void bar(...) {
    struct S {void method() {}} s;
    s.method();
}

int main() {
    auto foo = [](auto) -> int {
        return 50;
    };
    foo(50);

    auto foo2 = [](auto) -> int {
        return 50;
    };
    foo2(50);
    // TODO: This generates a lambda-sig mangling because the context is deemed to require a mangling numbering context (see getCurrentMangleNumberContext)
    //bar();

    bar(1, 2, 3);
}
