int test1() {
  int a = 5;
  int b = 7;
  int *p = &a;
  p = &b; // break here 1
  return *p;
}

char test2() {
  struct S {
    char a;
    char b;
  };
  struct S arr[2] = {{'a', 'b'}, {'c', 'd'}};
  struct S *p = arr;
  ++p; // break here 2
  return p->b;
}

int test3() {
  int a = 5;
  int b = 7;
  return a + b; // break here 3
}

int main() {
  test1();
  test2();
  test3();
  return 0;
}
