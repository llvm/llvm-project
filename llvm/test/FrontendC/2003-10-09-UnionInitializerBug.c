// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

struct Foo {
    unsigned a;
    unsigned b;
    unsigned c;
};

struct Bar {
    union {
        void **a;
        struct Foo b;
    }u;
};

struct Bar test = {0};

