#include <stdio.h>

int g_global[3] = {0, 100000, 100000};

void doSomething() {
  g_global[0] = 1; // Set outer breakpoint here
}

int main() {
  doSomething(); // Set a breakpoint here

  return g_global[0];
}
