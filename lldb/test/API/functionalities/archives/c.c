static int __c_global = 3;
char __extra[4096]; // Make sure sizeof b.o differs from a.o and c.o
char __extra2[4096]; // Make sure sizeof b.o differs from a.o and c.o
int c(int arg) {
  int result = arg + __c_global;
  return result;
}

int cc(int arg1) {
  int result3 = arg1 - __c_global;
  return result3;
}
