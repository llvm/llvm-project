int fun(int a) { return a * a + 1; }

int main() {
  int z = 0;
  for (int i = 0; i < 10000; i++) { // Break for loop
    z += fun(z);
  }

  return 0; // Break 1
}
