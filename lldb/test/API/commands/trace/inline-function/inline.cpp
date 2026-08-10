__attribute__((always_inline)) inline int mult(int x, int y) {
  int f = x * y;
  f++;
  f *= f;
  return f;
}

int foo(int x) {
  int z = mult(x, x - 1);
  z++;
  return z;
}

int main() {
  int x = 12;
  int z = foo(x);
  return z + x;
}
