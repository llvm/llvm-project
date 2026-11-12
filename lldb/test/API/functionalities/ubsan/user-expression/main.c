int foo() {
  int data[4];
  int x = *(int *)(((char *)&data[0]) + 2);
  return 42;
}

int main() {
  return 0; // breakpoint line
}
