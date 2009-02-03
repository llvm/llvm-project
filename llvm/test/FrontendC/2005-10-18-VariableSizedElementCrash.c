// RUN: %llvmgcc %s -S -o -

int sub1(int i, char *pi) {
  typedef int foo[i];
  struct bar {foo f1; int f2:3; int f3:4;} *p = (struct bar *) pi;
  xxx(p->f1);  
  return p->f3;
}

