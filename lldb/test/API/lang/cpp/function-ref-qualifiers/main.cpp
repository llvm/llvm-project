#include <cstdint>
#include <cstdio>

struct Foo {
  uint32_t func() const & { return 0; }
  int64_t func() const && { return 1; }
  uint32_t func() & { return 2; }
  int64_t func() && { return 3; }
};

int main() {
  Foo foo;
  const Foo const_foo;
  auto res = foo.func() + const_foo.func() + Foo{}.func() +
             static_cast<Foo const &&>(Foo{}).func();

  std::puts("Break here");
  return res;
}
