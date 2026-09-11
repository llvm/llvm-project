#include <stdio.h>
void done() {}
int main() {
  puts("in main");
  done(); // Set breakpoint here
  done(); // Second call to done
  puts("leaving main");
  return 0;
}
