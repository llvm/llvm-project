static int static_func(int);
static int static_var = 5;

int static_func(int x) {
  static_var++;
  return static_var + x;
}

int func_07(int x) {
  static_var++;
  return static_func(x);
}

