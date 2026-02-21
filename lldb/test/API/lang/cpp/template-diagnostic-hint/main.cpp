template <typename T, typename K> static K some_template_func(int x) {
  return (K)x;
}

template <typename T> struct Foo {
  template <typename K> T method(K k) { return (T)k; }
  static T smethod() { return (T)10; }
};

int main() {
  Foo<int> f;
  return some_template_func<int, long>(5) + Foo<int>::smethod() +
         f.method<long>(10);
}
