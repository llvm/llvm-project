#include <stdio.h>
int one();
int two();
int main() {
  puts("this is the standalone binary test program");
  return one() + two();
}
