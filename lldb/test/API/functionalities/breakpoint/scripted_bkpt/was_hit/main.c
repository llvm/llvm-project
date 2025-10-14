#include <stdio.h>

int stop_symbol() {
  static int s_cnt = 0;
  printf("I am in the stop symbol: %d\n", s_cnt++);
  return s_cnt;
}

int main() {
  for (int i = 0; i < 100; i++) {
    stop_symbol();
  }
  return 0;
}
