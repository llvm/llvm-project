int main() {
  int foo[2][3][4];
  int (*bar)[3][4] = foo;
  return 0; // Break at this line
}
