struct Foo {
  int bar;
  int baz;
};

struct Foo g_foo = {10, 20};

int return_bar() { return g_foo.bar; }

int return_baz() { return g_foo.baz; }
