int a(int);
int b(int);
int c(int);
int complex(int, int, int);

int a(int val) {
  int return_value = val;

  if (val <= 1) {
    return_value = b(val);
  } else if (val >= 3) {
    return_value = c(val);
  }

  return return_value;
}

int b(int val) {
  int rc = c(val);
  return rc;
}

int c(int val) { return val + 3; }

int complex(int first, int second, int third) { return first + second + third; }

int main(int argc, char const *argv[]) {
  int A1 = a(1);

  int B2 = b(2);

  int A3 = a(3);

  int A4 = complex(a(1), b(2), c(3));

  return 0;
}
