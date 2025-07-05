// main.cpp


__attribute__((noinline))
void test(int a) {
  for (int i = 0; i < 3; ++i) {       // i should be in a register
    int temp = i + 1;                 // temp also likely register
  }
}

int main() {
  test(5);
  return 0;
}

