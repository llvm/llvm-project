// RUN: %llvmgcc -S -g -O %s -o - | llvm-as | llc
// RUN: %llvmgcc -S -g %s -o - | llvm-as | llc

extern void foo (void);

static
void baz (int i)
{
  foo ();
  typedef char A[i];
  struct { A b; } *x = 0;
}

void
bar (i)
{
  baz (i);
}
