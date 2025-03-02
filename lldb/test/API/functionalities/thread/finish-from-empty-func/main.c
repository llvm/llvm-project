#include <stdio.h>
void done() {}
int main() {
  puts("in main");
  done();
  puts("leaving main");
  return 0;
}
