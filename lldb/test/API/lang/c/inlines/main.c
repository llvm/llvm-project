#include <stdio.h>

inline void test1(int) __attribute__ ((always_inline));
inline void test2(int) __attribute__ ((always_inline));

// Called once from main with b==42 then called from test1 with b==24.
void test2(int b) {
  printf("test2(%d)\n", b); // first breakpoint
  {
    int c = b * 2;
    printf("c=%d\n", c); // second breakpoint
  }
}

void test1(int a) {
    printf("test1(%d)\n",  a);
    test2(a + 1); // third breakpoint
}

int main(int argc) {
  test2(42);
  test1(23);
  return 0;
}
