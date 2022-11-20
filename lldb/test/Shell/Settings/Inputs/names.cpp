#include <functional>

namespace detail {
template <typename T> struct Quux {};
} // namespace detail

using FuncPtr = detail::Quux<double> (*(*)(int))(float);

struct Foo {
  template <typename T> void foo(T const &t) const noexcept(true) {}

  template <size_t T> void operator<<(size_t) {}

  template <typename T> FuncPtr returns_func_ptr(detail::Quux<int> &&) const noexcept(false) { return nullptr; }
};

namespace ns {
template <typename T> int foo(T const &t) noexcept(false) { return 0; }

template <typename T> FuncPtr returns_func_ptr(detail::Quux<int> &&) { return nullptr; }
} // namespace ns

int bar() { return 1; }

namespace {
int anon_bar() { return 1; }
auto anon_lambda = [](std::function<int(int (*)(int))>) mutable {};
} // namespace

int main() {
  ns::foo(bar);
  ns::foo(std::function{bar});
  ns::foo(anon_lambda);
  ns::foo(std::function{anon_bar});
  ns::foo(&Foo::foo<std::function<int(int)>>);
  ns::returns_func_ptr<int>(detail::Quux<int>{});
  Foo f;
  f.foo(std::function{bar});
  f.foo(std::function{anon_bar});
  f.operator<< <(2 > 1)>(0);
  f.returns_func_ptr<int>(detail::Quux<int>{});
  return 0;
}
