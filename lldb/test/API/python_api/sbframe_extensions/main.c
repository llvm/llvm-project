#include <stdio.h>

// Global and static variables for testing
int g_global_var = 42;
static int g_static_var = 100;

// Function declarations
int func1(int arg1, char arg2);
int func2(int arg1, int arg2);

int func1(int arg1, char arg2) {
  static int static_var = 200;
  int local1 = arg1 * 2;
  char local2 = arg2;
  // Set breakpoint here
  return local1 + local2 + static_var;
}

int func2(int arg1, int arg2) {
  int local1 = arg1 + arg2;
  int local2 = arg1 * arg2;
  // Set breakpoint here
  return func1(local1, 'X');
}

int main(int argc, char const *argv[]) {
  int main_local = 10;
  static int main_static = 50;
  // Set breakpoint here
  int result = func2(5, 7);
  printf("Result: %d\n", result);
  return 0;
}
