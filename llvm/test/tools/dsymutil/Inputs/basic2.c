/* For compilation instructions see basic1.c. */

static int baz = 42;
static int private_int;
extern volatile int val;
int unused_data = 1;

int bar(int);

void unused1() {
  bar(baz);
}

static int inc() {
  return ++private_int;
}

__attribute__((noinline))
int foo(int arg) {
  return bar(arg+val) + inc() + baz++;
}

/* This file was also used to create basic2-custom-linetable.macho.x86_64.o
   with a custom clang that had different settings for the linetable
   encoding constants: line_base == -1 and line_range == 4.

   clang -c -g basic2.c -o basic2-custom-linetable.macho.x86_64.o 
*/
