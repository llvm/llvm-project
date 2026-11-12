struct Foo
{
  double x;
  int y;
  Foo() : x(3.1415), y(1234) {}
};

int main() {
  Foo f;
  return 0; // break here
}
