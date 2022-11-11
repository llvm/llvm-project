namespace a {
struct Foo {};
} // namespace a

namespace b {
struct Foo {};
} // namespace b

int main() {
  a::Foo a;
  b::Foo b;
  return 0; // Set breakpoint here
}
