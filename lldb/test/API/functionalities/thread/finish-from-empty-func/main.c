#include <stdio.h>
void done() {}
int main() {
  puts("in main");
  done(); // Set breakpoint here
  done();
  puts("leaving main");
  return 0;
}
