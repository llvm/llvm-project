#include <stdio.h>

static int bad_read(int index) {
  int array[] = {0, 1, 2};
  return array[index];
}

static void breakpoint_func(void) {}

int main(int argc, char **argv) {
  breakpoint_func();
  bad_read(10);
  printf("Execution continued\n");
  breakpoint_func();
  bad_read(20);
  breakpoint_func();
  bad_read(30);
  return 0;
}
