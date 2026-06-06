static int static_func(int);
static int static_var = 0;

int static_func(int x) {
  static_var++;
  return static_var + x;
}

int func_06(int x) {
  return static_func(x);
}

