template <typename T> using Foo = T;

template <typename T> using Bar = Foo<T>;

template <typename T> struct Container {};

int main() {
  Foo<int> f1;
  Foo<double> f2;
  Bar<int> b1;
  Bar<double> b2;
  Bar<Foo<int>> bf1;
  Bar<Foo<double>> bf2;
  Container<Bar<Foo<int>>> cbf1;
  return 0;
}
