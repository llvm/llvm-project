struct Foo {
  int real_child = 47;
};

struct HasFoo {
  Foo f;
};

int main() {
  Foo foo;
  HasFoo has_foo;
  return 0; // break here
}
