// RUN: %llvmgcc %s -S -o -


enum {
  tA = 0,
  tB = 1
};

struct MyStruct {
  unsigned long A;
  void * B;
};

void bar(){
struct MyStruct MS = { tB, 0 };
}
