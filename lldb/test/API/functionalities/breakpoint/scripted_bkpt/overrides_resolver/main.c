#include <stdio.h>

int g_change_me = 0;

int change_him() { return ++g_change_me; }

void stop_here_instead() { printf("Stopped here instead?\n"); }

int stop_symbol() {
  static int s_cnt = 0;
  printf("I am in the stop symbol: %d\n", s_cnt++);
  stop_here_instead();
  return s_cnt;
}

int main() {
  stop_symbol();
  change_him();
  return 0;
}
