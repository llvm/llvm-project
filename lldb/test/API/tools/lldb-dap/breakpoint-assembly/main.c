__attribute__((nodebug)) int assembly_func(int n) {
  n += 1;
  n += 2;
  n += 3;

  return n;
}

int main(int argc, char const *argv[]) {
  assembly_func(10);
  assembly_func(20);
  assembly_func(30);
  return 0;
}
