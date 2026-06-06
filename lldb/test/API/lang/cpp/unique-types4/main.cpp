namespace ns {

template <typename T> struct Foo {
  static T value;
};

template <class T> T Foo<T>::value = 0;

template <typename T> using Bar = Foo<T>;

using FooInt = Foo<int>;
using FooDouble = Foo<double>;

} // namespace ns

ns::Foo<double> a;
ns::Foo<int> b;
ns::Bar<double> c;
ns::Bar<int> d;
ns::FooInt e;
ns::FooDouble f;

int main() {
  // Set breakpoint here
  return (int)a.value + b.value + (int)c.value + d.value + e.value +
         (int)f.value;
}
