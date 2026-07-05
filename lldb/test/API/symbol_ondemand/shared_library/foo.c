#include <stdio.h>

int global_foo = 321;
void foo(void) { puts("Hello, I am a shared library"); }
