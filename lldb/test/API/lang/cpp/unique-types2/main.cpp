template <class T> struct Foo {
  T t;
  template <class U> class Nested {
    U u;
  };
};

template <class T, class... Ss> class FooPack {
  T t;
};

int main() {
  Foo<char> t1;
  Foo<int> t2;
  Foo<Foo<int>> t3;

  FooPack<char> p1;
  FooPack<int> p2;
  FooPack<Foo<int>> p3;
  FooPack<char, int> p4;
  FooPack<char, float> p5;
  FooPack<int, int> p6;
  FooPack<int, int, int> p7;

  Foo<int>::Nested<char> n1;
  // Set breakpoint here
}
