namespace detail {
template <typename T> struct Quux {};
} // namespace detail

using FuncPtr = detail::Quux<double> (*(*)(int))(float);

struct Foo {
  template <typename T> void foo(T arg) const noexcept(true) {}

  template <int T> void operator<<(int) {}

  template <typename T> FuncPtr returns_func_ptr(detail::Quux<int> &&) const noexcept(false) { return nullptr; }
};

namespace ns {
template <typename T> int foo(char const *str) noexcept(false) { return 0; }
template <typename T> int foo(T t) { return 1; }

template <typename T> FuncPtr returns_func_ptr(detail::Quux<int> &&) { return nullptr; }
} // namespace ns

int bar() { return 1; }

namespace {
int anon_bar() { return 1; }
auto anon_lambda = [] {};
} // namespace

int main() {
  ns::foo<decltype(bar)>(bar);
  ns::foo<decltype(bar)>("bar");
  ns::foo(anon_lambda);
  ns::foo(anon_bar);
  ns::foo<decltype(&Foo::foo<int(int)>)>("method");
  ns::returns_func_ptr<int>(detail::Quux<int>{});
  Foo f;
  f.foo(anon_bar);
  f.operator<< <(2 > 1)>(0);
  f.returns_func_ptr<int>(detail::Quux<int>{});

  return 0;
}
