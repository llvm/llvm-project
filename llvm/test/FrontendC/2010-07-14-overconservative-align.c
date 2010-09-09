// RUN: %llvmgcc %s -emit-llvm -S -o - | FileCheck %s
// PR 5995
struct s {
    int word;
    struct {
        int filler __attribute__ ((aligned (8)));
    };
};

void func (struct s *s)
{
// CHECK: load %struct.s** %s_addr, align {{[48]}}
    s->word = 0;
}
