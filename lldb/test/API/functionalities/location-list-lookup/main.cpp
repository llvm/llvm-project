#include <cstdio>
#include <cstdlib>

void func(int in);

struct Foo {
  int x;
  [[clang::noinline]] void bar(char **argv);
};

int main(int argc, char **argv) {
  Foo f{.x = 5};
  std::printf("%p\n", &f.x);
  f.bar(argv);
  return f.x;
}

void Foo::bar(char **argv) {
  std::printf("%p %p\n", argv, this);
  std::abort(); /// 'this' should be still accessible
}

void func(int in) { printf("%d\n", in); }
