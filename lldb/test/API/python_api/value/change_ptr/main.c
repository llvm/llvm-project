int main() {
  int a = 5;
  int b = 7;
  int *p = &a;
  p = &b; // break here
  return 0;
}
