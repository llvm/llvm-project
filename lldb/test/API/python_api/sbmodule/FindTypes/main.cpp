namespace ns1 {
template <typename T> struct Foo {
  struct LookMeUp {};
};
} // namespace ns1

namespace ns2 {
template <typename T> struct Bar {
  struct LookMeUp {};
};
} // namespace ns2

ns1::Foo<void>::LookMeUp l1;
ns2::Bar<void>::LookMeUp l2;
ns1::Foo<ns2::Bar<void>>::LookMeUp l3;

int main() {}
