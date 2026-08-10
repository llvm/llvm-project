#include "symlink1/foo.h"
#include "symlink2/bar.h"
#include "symlink2/qux.h"

int main(int argc, char const *argv[]) {
  int a = foo();    // 1
  int b = bar();    // 2
  int c = qux();    // 3
  return a + b - c; // Set break point at this line.
}
