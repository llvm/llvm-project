int __a_global = 1;

int a(int arg) {
  int result = arg + __a_global;
  return result;
}

int aa(int arg1) {
  int result1 = arg1 - __a_global;
  return result1;
}
