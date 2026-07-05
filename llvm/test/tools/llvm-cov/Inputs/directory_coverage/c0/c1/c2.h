int mul(int a, int b);

int div(int a, int b);

#ifdef DEF
int div(int a, int b) {
  return a / b;
}
#endif
