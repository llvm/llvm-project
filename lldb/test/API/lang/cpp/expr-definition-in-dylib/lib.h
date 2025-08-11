#ifndef LIB_H_IN
#define LIB_H_IN

struct Foo {
  int method();
  Foo();
  ~Foo();
};

struct Bar {
  [[gnu::abi_tag("Ctor")]] Bar();
  [[gnu::abi_tag("Dtor")]] ~Bar();
};

#endif // LIB_H_IN
