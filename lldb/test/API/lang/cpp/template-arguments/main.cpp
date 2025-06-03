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
Foo<_Float16, _Float16(1.0)> temp7;
Foo<__bf16, __bf16(1.0)> temp8;

template <typename T, T... values> struct Bar {};
Bar<double, 1.2> temp9;
Bar<float, 1.0f, 2.0f> temp10;

int main() {}
