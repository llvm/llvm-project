inline __attribute__((always_inline)) int sum2(int a, int b) {
  int result = a + b;
  return result;
}

int sum3(int a, int b, int c) {
  int result = a + b + c;
  return result;
}

inline __attribute__((always_inline)) int sum4(int a, int b, int c, int d) {
  int result = sum2(a, b) + sum2(c, d);
  result += 0;
  return result;
}

int main(int argc, char **argv) {
  sum3(3, 4, 5) + sum2(1, 2);
  int sum = sum4(1, 2, 3, 4);
  sum2(5, 6);
  return 0;
}
