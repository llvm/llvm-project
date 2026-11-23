#include <stdio.h>

__attribute__((nodebug)) int no_branch_func(void) {
  int result = 0;

  __asm__ __volatile__("movl $0, %%eax;" // Assembly start
                       "incl %%eax;"
                       "incl %%eax;"
                       "incl %%eax;"
                       "incl %%eax;"
                       "incl %%eax;"
                       "incl %%eax;"
                       "incl %%eax;"
                       "incl %%eax;"
                       "incl %%eax;"
                       "incl %%eax;"
                       "movl %%eax, %0;" // Assembly end
                       : "=r"(result)
                       :
                       : "%eax");

  return result;
}

int main(void) {
  int result = no_branch_func(); // Break here
  printf("Result: %d\n", result);
  return 0;
}
