// clang -c -g -o dwarfdump-test-loc-list-32bit.elf.o -m32 dwarfdump-test-loc-list-32bit.elf.cpp

namespace pr14763 {
struct foo {
  foo(const foo&);
};

foo func(bool b, foo f, foo g) {
  if (b)
    return f;
  return g;
}
}
