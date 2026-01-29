int main(int argc, char const *argv[]) {
  int data[4] = {0};
  int *p = data + 5; // ubsan
  return *p;
}
