template <typename T> struct Foo;

typedef Foo<int> BarInt;
typedef Foo<double> BarDouble;

template <typename T> using Bar = Foo<T>;

template <typename T>
struct [[clang::preferred_name(BarInt), clang::preferred_name(BarDouble),
         clang::preferred_name(Bar<short>), clang::preferred_name(Bar<short>),
         clang::preferred_name(Bar<double>),
         clang::preferred_name(Bar<char>)]] Foo{};

int main() {
  BarInt barInt;
  BarDouble barDouble;
  Bar<short> barShort;
  Bar<char> barChar;

  Foo<int> varInt;
  Foo<double> varDouble;
  Foo<short> varShort;
  Foo<char> varChar;
  Foo<Foo<int>> varFooInt;
  return 0;
}
