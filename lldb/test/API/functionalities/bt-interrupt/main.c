#include <stdio.h>

// This example is meant to recurse infinitely.
// The extra struct is just to make the frame dump
// more complicated.

struct Foo {
  int a;
  int b;
  char *c;
};

int
forgot_termination(int input, struct Foo my_foo) {
  char frame_increasing_buffer[0x1000]; // To blow the stack sooner.
  return forgot_termination(++input, my_foo);
}

int
main()
{
  struct Foo myFoo = {100, 300, "A string you will print a lot"}; // Set a breakpoint here
  return forgot_termination(1, myFoo);
}
