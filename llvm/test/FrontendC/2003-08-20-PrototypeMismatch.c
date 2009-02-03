// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null



static int foo(int);

static int foo(C)
char C;
{
  return C;
}

void test() {
  foo(7);
}
