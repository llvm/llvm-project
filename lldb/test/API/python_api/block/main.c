#include <stdio.h>

#ifdef _WIN32
__declspec(dllimport) int fn(int a, int b);
#else
extern int fn(int a, int b);
#endif

int main(int argc, char const *argv[]) {
  int a = 3;
  int b = 17;
  int sum = fn(a, b); // breakpoint 1
  printf("fn(3, 17) returns %d\n", sum);
  return 0;
}
