int compute(int a, int b) {
  int sum = a + b;
  return sum; // Break here.
}

int main(void) { return compute(3, 4) == 7 ? 0 : 1; }
