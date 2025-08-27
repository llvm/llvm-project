int test() {
  int my_ints[] = {0x42};
  return 0; // Set break point at this line.
}

int main() {
  int dummy[100];
  return test();
}
