#ifndef LIB_H_IN
#define LIB_H_IN

struct Foo {
  int method();
  Foo(int val);
  ~Foo();

  int x;
};

struct Base {
  [[gnu::abi_tag("BaseCtor")]] Base();
  [[gnu::abi_tag("BaseDtor")]] ~Base();
};

struct Bar : public Base {
  [[gnu::abi_tag("Ctor")]] Bar();
  [[gnu::abi_tag("Dtor")]] ~Bar();
};

#endif // LIB_H_IN
