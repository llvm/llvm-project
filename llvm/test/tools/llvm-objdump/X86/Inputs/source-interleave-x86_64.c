int a = 1;
int foo() {
  return a;
}

int main() {
  int *b = &a;
  return *b + foo();
}
