// Check that we can compile files of different types together.
// RUN: llvmc %s %p/test_data/together.c -o %t
// RUN: ./%t | grep hello

extern "C" void test();

int main() {
  test();
}
