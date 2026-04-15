#include <stdio.h>

int g_value = 0;

int increment() { return ++g_value; }

int target_func() {
  printf("target_func: %d\n", g_value);
  return g_value;
}

int main() {
  for (int i = 0; i < 10; i++) {
    target_func();
  }
  increment();
  return 0;
}
