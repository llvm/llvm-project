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

int SumTwoIntegers(int x, int y) {
  return x + y;
}

int GetSum(struct Foo *f) {
  return SumTwoIntegers(f->a, f->b->d ? 0 : 1);
}

int main() {
  return GetSum(GetAFoo());
}
