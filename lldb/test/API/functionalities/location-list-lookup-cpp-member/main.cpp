#include <cstdio>
#include <cstdlib>

void func(int in);

struct Foo {
  int x;
  [[clang::noinline]] void bar(Foo *f);
};

int main(int argc, char **argv) {
  Foo f{.x = 5};
  std::printf("%p\n", &f.x);
  f.bar(&f);
  return f.x;
}

void Foo::bar(Foo *f) {
  std::printf("%p %p\n", f, this);
  std::abort(); /// 'this' should be still accessible
}

void func(int in) { printf("%d\n", in); }
