struct Foo {
  int b;
  int c;
};

int main() {
  struct Foo *a = 0;
  return a[10].c;
}
