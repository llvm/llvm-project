// This function will be embedded within the .symtab section of the
// .gnu_debugdata section.
int multiplyByFour(int num) { return num * 4; }

// This function will be embedded within the .dynsym section of the main binary.
int multiplyByThree(int num) { return num * 3; }

int main(int argc, char *argv[]) {
  int x = multiplyByThree(argc);
  int y = multiplyByFour(x);
  return y;
}
