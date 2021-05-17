#include <stdio.h>

void __attribute__((noinline)) bar(unsigned i) { printf("%d\n", i); }

void __attribute__((noinline)) foo(unsigned j) {
  unsigned i = j;
  bar(i);
  i = 10;
  bar(i); // Set break point at this line.
}

int main(int argc, char** argv)
{
  foo(argc);
}
