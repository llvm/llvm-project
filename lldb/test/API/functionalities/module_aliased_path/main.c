extern int foo(void);

int main(void) {
  int result = foo(); // break main
  return result == 42 ? 0 : 1;
}
