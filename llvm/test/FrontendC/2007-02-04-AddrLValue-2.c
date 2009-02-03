// RUN: %llvmgcc %s -O3 -S -o - -emit-llvm
// PR1173

struct S { char s; };
struct T { struct S t; };

struct S *const p = &((struct T * const) (0x4000))->t;

void
foo (void)
{
  p->s = 0;
}
