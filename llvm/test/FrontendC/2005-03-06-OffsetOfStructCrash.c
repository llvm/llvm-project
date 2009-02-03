// RUN: %llvmgcc %s -S -o -

struct Y {};
struct XXX {
  struct  Y F;
};

void test1() {
   (int)&((struct XXX*)(((void *)0)))->F;
}

void test2() {
   &((struct XXX*)(((void *)0)))->F;
}
