#include <stdio.h>
void doNothing() { printf("doNothing\n"); }

void doSomethiing() {
  doNothing();
  doNothing();
  doNothing();
}

int main() {
  doSomethiing();
  doNothing();
  doSomethiing();
  return 0;
}
