template<typename T, unsigned value>
struct C {
  T member = value;
};

C<int, 2> temp1;

template <typename T, T value> struct Foo {};
Foo<short, -2> temp2;
Foo<char, 'v'> temp3;
Foo<float, 2.0f> temp4;
Foo<double, -250.5> temp5;
Foo<int *, &temp1.member> temp6;

template <typename T, T... values> struct Bar {};
Bar<double, 1.2> temp7;
Bar<float, 1.0f, 2.0f> temp8;

int main() {}
