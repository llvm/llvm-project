__attribute__((section("__codesection")))
int f(int a) {
  return a + 1; // Set break point at this line.
}

int main() {
  return f(10);
}
