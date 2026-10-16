static int __b_global = 2;
char __extra[4096]; // Make sure sizeof b.o differs from a.o
int b(int arg) {
  int result = arg + __b_global;
  return result;
}

int bb(int arg1) {
  int result2 = arg1 - __b_global;
  return result2;
}
