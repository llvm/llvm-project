#include <cstdio>

namespace A {
template <typename T> bool operator<(const T &, const T &) { return true; }

template <typename T>
[[gnu::abi_tag("Two", "Tags")]] bool operator>(const T &, const T &) {
  return true;
}

template <typename T>
[[gnu::abi_tag("OneTag")]] bool operator==(const T &, const T &) {
  return true;
}

[[gnu::abi_tag("Foo")]] int withAbiTagInNS(const int &, const int &) {
  return 1;
}

template <typename T>
[[gnu::abi_tag("Bar")]] int withAbiTagInNS(const T &, const T &) {
  return 2;
}

struct B {};
} // namespace A

template <typename T>
[[gnu::abi_tag("Baz")]] int withAbiTag(const T &, const T &) {
  return 3;
}

[[gnu::abi_tag("Baz")]] int withAbiTag(const int &, const int &) { return -3; }

struct Simple {
  int mem;
};

struct [[gnu::abi_tag("Qux")]] Tagged {
  int mem;

  int const &Value() const { return mem; }
};

template <typename T> struct [[gnu::abi_tag("Quux", "Quuux")]] TaggedTemplate {
  T mem;

  T const &Value() const { return mem; }
};

// clang-format off
inline namespace [[gnu::abi_tag("Inline", "NS")]] v1 {
template <typename T> int withImplicitTag(T const &t) { return t.mem; }
} // namespace
// clang-format on

int main() {
  A::B b1;
  A::B b2;
  Tagged t{.mem = 4};
  TaggedTemplate<int> tt{.mem = 5};

  int result = (b1 < b2) + (b1 > b2) + (b1 == b2) + withAbiTag(b1, b2) +
               A::withAbiTagInNS(1.0, 2.0) + withAbiTagInNS(b1, b2) +
               A::withAbiTagInNS(1, 2) + withImplicitTag(Tagged{.mem = 6}) +
               withImplicitTag(Simple{.mem = 6}) + t.Value() + tt.Value();

  std::puts("Break here");

  return result;
}
