struct Bar {
  int c;
  int d;
};

struct Foo {
  int a;
  struct Bar *b;
};

struct Foo *GetAFoo() {
  static struct Foo f = { 0, 0 };
  return &f;
}

int GetSum(struct Foo *f) {
  return f->a + f->b->d;
}

int main() {
  return GetSum(GetAFoo());
}
