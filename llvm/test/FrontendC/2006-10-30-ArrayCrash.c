// RUN: %llvmgcc -O3 -S -o - %s
// PR954, PR911

extern void foo();

struct S {
  short        f1[3];
  unsigned int f2 : 1;
};

void bar()
{
  struct S *A;

  if (A->f2)
    foo();
}
