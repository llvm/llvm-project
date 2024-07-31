
int multiply(int x, int y) {
  return x * y; // breakpoint 1
}

int main(int argc, char const *argv[]) {
  int result = multiply(argc, 20);
  return result < 0;
}
