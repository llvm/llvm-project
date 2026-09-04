#include <stdio.h>

volatile int sink;

__attribute__((always_inline))
void work(int n) {
  int s = 0;
  for (int i = 1; i <= n; i++) {
    if (i % 3)
      s += i;
    else
      s -= i;
  }
  sink = s;
}

__attribute__((always_inline))
void wrapper(int n) {
  work(n);
}

int main() {
  wrapper(4000 * 4000);
  printf("result is %d\n", sink);
  return 0;
}
