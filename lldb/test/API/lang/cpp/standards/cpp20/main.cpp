#include <compare>

struct Foo {
  friend auto operator<=>(Foo const &, Foo const &) { return true; }
};

int main() { return Foo{} <=> Foo{}; }
