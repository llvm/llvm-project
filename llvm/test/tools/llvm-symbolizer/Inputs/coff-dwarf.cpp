// To generate the corresponding EXE, run:
// clang-cl -MD -O2 -gdwarf -c coff-dwarf.cpp && lld-link -debug coff-dwarf.obj

extern "C" int puts(const char *str);

void __declspec(noinline) foo() {
  puts("foo1");
  puts("foo2");
}

// LLVM should inline this into main.
static void bar() {
  foo();
}

int main() {
  bar();
  return 0;
}
