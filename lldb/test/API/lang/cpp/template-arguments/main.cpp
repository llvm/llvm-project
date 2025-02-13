template<typename T, unsigned value>
struct C {
  T member = value;
};

C<int, 2> temp1;

template <typename T, T value> struct Foo {};
Foo<short, -2> temp2;
Foo<char, 'v'> temp3;
Foo<float, 2.0f> temp4;

int main() {}
