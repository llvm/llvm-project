// RUN: %llvmgcc %s -S -o /dev/null

// GCC allows variable sized arrays in structures, crazy!

// This is PR360.

int sub1(int i, char *pi) {
  typedef int foo[i];
  struct bar {foo f1; int f2;} *p = (struct bar *) pi;
  return p->f2;
}
