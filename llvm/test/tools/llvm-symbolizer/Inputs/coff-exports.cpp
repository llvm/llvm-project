// To generate the corresponding EXE, run:
// clang-cl -MD -c coff-exports.cpp && lld-link /MANIFEST:NO coff-exports.obj

#define EXPORT __declspec(dllexport)

extern "C" int puts(const char *str);

EXPORT void __declspec(noinline) foo() {
  puts("foo1");
  puts("foo2");
}

void bar() {
  foo();
}

EXPORT int main() {
  bar();
  return 0;
}
