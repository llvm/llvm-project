#ifndef LIB_H_IN
#define LIB_H_IN

template <typename T> struct FooImpl;

struct Foo {
  FooImpl<char> *impl = nullptr;
};

#endif // LIB_H_IN
