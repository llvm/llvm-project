static volatile int do_mul;
static volatile int x, v;

int foo () {
  if (do_mul) x *= v; else x /= v;
  return x;
}

int main() {
  return foo() + foo();
}
