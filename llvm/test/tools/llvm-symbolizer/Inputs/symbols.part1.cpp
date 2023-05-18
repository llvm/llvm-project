#include "symbols.h"

int global_01 = 22;

int static static_var = 0;

static int static_func_01(int x) {
  static_var = x;
  return global_01;
}

int func_01() {
  int res = 1;
  return res + static_func_01(22);
}

int func_04() {
  static_var = 0;
  return 22;
}

int func_04(int x) {
  int res = static_var;
  return res + func_03(x);
}
