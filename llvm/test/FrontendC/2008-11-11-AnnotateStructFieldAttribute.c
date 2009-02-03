// RUN: %llvmgcc -c -emit-llvm %s -o - | llvm-dis | grep llvm.ptr.annotation | count 3

#include <stdio.h>

/* Struct with element X being annotated */
struct foo {
    int X  __attribute__((annotate("StructAnnotation")));
    int Y;
    int Z;
};


void test(struct foo *F) {
    F->X = 42;
    F->Z = 1;
    F->Y = F->X;
}

