static int __c_global = 3;

int c(int arg) {
  int result = arg + __c_global;
  return result;
}

int cc(int arg1) {
  int result3 = arg1 - __c_global;
  return result3;
}
