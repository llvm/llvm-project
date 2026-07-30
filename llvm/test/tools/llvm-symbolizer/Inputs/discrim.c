// Compile with:  clang -gmlt -fdebug-info-for-profiling -O2 discrim.c -o discrim
// to get an input file with DWARF line table discriminators in it.
// Tested in test/tools/llvm-symbolizer/sym-verbose.test

static volatile int do_mul;
static volatile int x, v;

int foo () {
  if (do_mul) x *= v; else x /= v;
  return x;
}

int main() {
  return foo() + foo();
}
