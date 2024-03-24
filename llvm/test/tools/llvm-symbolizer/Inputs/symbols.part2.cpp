#include "symbols.h"

int static static_var = 4;

static int static_func_01(int x) {
  static_var--;
  return x;
}

int func_02(int x) {
  static_var = x;
  return static_func_01(x);
}

int func_05(int x) {
  int res = static_var;
  return res + func_03(x);
}
