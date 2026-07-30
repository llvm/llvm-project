int a(int);
int b(int);
int c(int);

int a(int val) {
  if (val <= 1)
    val = b(val);
  else if (val >= 3)
    val = c(val);

  return val;
}

int b(int val) { return c(val); }

int c(int val) {
  return val + 3; // break here.
}

int main(int argc, char const *argv[]) {
  int A1 = a(1); // a(1) -> b(1) -> c(1)
  int B2 = b(2); // b(2) -> c(2)
  int A3 = a(3); // a(3) -> c(3)
  return 0;
}
